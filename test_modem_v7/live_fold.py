#!/usr/bin/env python3
"""Prototype live folding for tools/v7_live.py (--experimental-fold M).

Both ends load the same frozen fold table, test_modem_v7/fold_table_<M>.json
(built from the repository's reference fixture, so no local frames are
involved). The sender folds each frame's values before the normal V7 encode.
The receiver keeps each decoded packet's equaliser output and unfolds the host
slots before display, with the low-confidence fallback. Nothing in
animation_modem changes: the receiver hook wraps v7.decode_frame and the two
equaliser entry points while the prototype is loaded.

Fail closed:
- A table file must match its pinned SHA-256 (TABLE_SHA256 below). A
  rebuilt table must be pinned deliberately, like the V7 model tables.
- A table only folds or unfolds the exact model it was built for (the
  canonical `box` profile). The sender refuses any other encode filter or
  fixture. The receiver shows packets from any other model unfolded.
- The 16 weakest host slots carry a signature drawn from the table's
  identity. The receiver unfolds only packets that carry its own table's
  signature. Normal packets, and packets folded with a different table, are
  shown as they are.

A normal receiver shows folded packets slightly degraded (loopback, one
frame: -18.7 normal, -22.1 folded but not unfolded, -7.6 folded and
unfolded).

    python test_modem_v7/live_fold.py build [--folds 500 1000] [--frames F...]
    python test_modem_v7/live_fold.py selftest FRAME... [--folds 500] [--quick]
"""
import argparse
import hashlib
import json
import sys
import threading
from pathlib import Path

import numpy as np

from common import REPO, TARGET, v7                                    # noqa: F401
from folding import FoldCodec, table_identity, table_text

TABLE_DIR = Path(__file__).resolve().parent
# The shipped tables. Both ends must hold byte-identical tables, so a table
# whose SHA-256 is not pinned here is refused. `build` prints the new values.
TABLE_SHA256 = {
    500: '1b3ef13f3457a4e4c02a92e149b3bb0eac3c0e233ed1f18eefb84459e24d8ded',
    1000: '9f48877e5f6b9f6d3e52e04f9e3f611a3d3ca3296cc9f05453491bd8c53710a2',
}


def table_path(slots, table_dir=TABLE_DIR):
    return Path(table_dir)/f'fold_table_{int(slots)}.json'


class LiveFold:
    """Sender and receiver glue for one frozen, pinned fold table."""

    def __init__(self, slots, table_dir=TABLE_DIR, pins=TABLE_SHA256):
        slots = int(slots)
        path = table_path(slots, table_dir)
        if slots not in pins:
            raise ValueError(f'no pinned fold table for M={slots} '
                             f'(pinned: {sorted(pins)})')
        if not path.exists():
            raise ValueError(f'fold table {path} is missing')
        blob = path.read_bytes()
        digest = hashlib.sha256(blob).hexdigest()
        if digest != pins[slots]:
            raise ValueError(f'{path.name} SHA-256 {digest[:12]}… does not match the '
                             f'pinned {pins[slots][:12]}…: refusing an unexpected table')
        self.table = json.loads(blob)
        if table_text(self.table).encode() != blob:
            raise ValueError(f'{path.name} is not in canonical form')
        self.slots, self.digest = slots, digest[:12]
        self.identity = table_identity(self.table)
        self._codecs = {}
        self._refused = {}
        self._local = threading.local()
        self._installed = None

    def codec(self, model):
        """The codec for `model`; ValueError if the table is not for it."""
        key = model.encoding_type
        if key not in self._codecs:
            self._codecs[key] = FoldCodec.from_table(model, self.table)
        return self._codecs[key]

    # ---------------------------------------------------------------- sender
    def check(self, model):
        """Fail before sending anything if this table cannot fold `model`."""
        self.codec(model)

    def encode(self, model, values):
        """Values whose V7 encode is a folded packet."""
        return self.codec(model).encode(values)

    # -------------------------------------------------------------- receiver
    def install(self):
        """Keep each decoded packet's equaliser output on its Result."""
        if self._installed:
            return
        local = self._local
        real = (v7._equalize_numba, v7._equalize_numpy, v7.decode_frame)

        def equalize_numba(model, Z, H, noise, counter):
            local.eq = real[0](model, Z, H, noise, counter)
            return local.eq

        def equalize_numpy(model, Z, H, noise, counter, force_float32=False):
            local.eq = real[1](model, Z, H, noise, counter, force_float32=force_float32)
            return local.eq

        def decode_frame(*args, **kwargs):
            local.eq = None
            result = real[2](*args, **kwargs)
            if result is not None and local.eq is not None:
                xhat, conf = local.eq[0], local.eq[1]
                result.diag['fold_eq'] = (np.asarray(xhat, float), np.asarray(conf, float))
            return result

        v7._equalize_numba, v7._equalize_numpy, v7.decode_frame = (
            equalize_numba, equalize_numpy, decode_frame)
        self._installed = real

    def uninstall(self):
        if self._installed:
            v7._equalize_numba, v7._equalize_numpy, v7.decode_frame = self._installed
            self._installed = None

    def values(self, model, result):
        """Display values for a decoded packet: unfolded when the equaliser
        output is available, the packet carries this table's signature and the
        table is for this packet's model; the normal reconstruction otherwise."""
        eq = result.diag.get('fold_eq')
        key = model.encoding_type
        if eq is None or key in self._refused:
            return v7.values_from(model, result.coeffs)
        try:
            codec = self.codec(model)
        except ValueError as exc:                     # another filter or profile
            self._refused[key] = str(exc)
            return v7.values_from(model, result.coeffs)
        return codec.grid.inverse(codec.decode(result.coeffs, eq[0], eq[1], fallback=True))


# --------------------------------------------------------------------- build
def build(args):
    from PIL import Image
    frames = [Image.open(p).convert('RGB') for p in (args.frames or [v7.REFERENCE_FIXTURE])]
    model = v7.load_model(TARGET, 'box')
    sources = [Path(p) for p in (args.frames or [v7.REFERENCE_FIXTURE])]
    fitted_on = ', '.join(f'{p.name} sha256:{hashlib.sha256(p.read_bytes()).hexdigest()[:12]}'
                          for p in sources)
    for slots in args.folds:
        codec = FoldCodec(model, frames, slots, design_db=args.design_db,
                          signature=args.signature, fitted_on=fitted_on)
        text = table_text(codec.table())
        table_path(slots).write_text(text)
        print(f'{table_path(slots).name}: M={slots} D={codec.D:.4f} '
              f"pin: {slots}: '{hashlib.sha256(text.encode()).hexdigest()}'")
    print('update TABLE_SHA256 in test_modem_v7/live_fold.py to pin these tables')


# ------------------------------------------------------------------ selftest
def selftest(args):
    """The live receive path without audio devices: the live sender's value
    path, a live-style packet stream, LiveInput in 1,024-sample blocks and
    decode_pulse_stream with the live receiver's arguments. Every shown
    picture is scored against its own source frame."""
    import dataclasses
    from scipy.signal import resample_poly
    from common import CASES, RATE, impair, jitter_warp, load_frames, reference, ssimulacra2
    from animation_modem.imaging import values_image
    from animation_modem.v7_live_input import LiveInput
    sys.path.insert(0, str(REPO/'tools'))
    import v7_live

    _, test = load_frames(args.frames)
    packets = 12 if args.quick else args.packets
    C = {c.name: c for c in CASES}
    impairments = {
        'clean': lambda w: impair(w, C['clean-96k'], seed=2026),
        'dropouts': lambda w: impair(w, C['dropouts'], seed=2026),
        'wow-flutter': lambda w: impair(w, C['wow-flutter'], seed=2026),
        'fast-flutter': lambda w: impair(w, C['fast-flutter'], seed=2026),
        'jitter 0.1%': lambda w: impair(jitter_warp(w), C['clean-96k'], seed=2026),
        'jitter 0.3%': lambda w: impair(jitter_warp(w, .003), C['clean-96k'], seed=2026),
        'lowpass-10k': lambda w: impair(w, C['lowpass-10k'], seed=2026),
        'warble+lpf12k+dropouts': lambda w: impair(w, dataclasses.replace(
            C['fast-flutter'], lowpass=12000, dropout_ms=12, dropout_every=.7), seed=2026),
    }
    # Any torture-matrix case can be named too (hiss-40, type-ii, ...).
    for name, case in C.items():
        impairments.setdefault(name, lambda w, case=case: impair(w, case, seed=2026))
    cases = ['clean', 'dropouts'] if args.quick else args.cases or [
        'clean', 'dropouts', 'wow-flutter', 'fast-flutter', 'jitter 0.1%', 'jitter 0.3%',
        'lowpass-10k', 'warble+lpf12k+dropouts']
    sender = v7.load_model(TARGET, 'box')
    bootstrap = v7.load_model(TARGET, 'nearest')
    v7.warmup_equalizer(bootstrap)
    refs = [reference(frame) for frame in test]
    base_values = [v7_live._values(sender, frame, 'box', brightness=1.0)[0] for frame in test]
    order = [i % len(test) for i in range(packets)]                    # a moving sequence

    def run(fold, case):
        models = {bootstrap.encoding_type: bootstrap}

        def factory(code):
            if code not in models:
                models[code] = v7.load_model(TARGET, v7.ENCODING_FILTERS[int(code)])
            return models[code]
        values = [fold.encode(sender, v) if fold else v for v in base_values]
        audio = v7.encode_pulse_stream(sender, [values[i] for i in order], 1,
                                       [0]*packets, source_indices=order,
                                       pilot_tones=True, eof_marker=True)
        audio = impairments[case](resample_poly(audio, 2, 1, axis=0).astype(np.float32))
        live = LiveInput(rate=RATE)
        state = v7.PulseState()
        scores, decoded = [], 0
        for start in range(0, len(audio), 1024):
            live.add(np.array(audio[start:start+1024], copy=True))
            chunk = live.take(start/RATE)
            if chunk is None:
                continue
            results, _ = v7.decode_pulse_stream(
                bootstrap, chunk, latest_only=True, input_gain=live.gain, models=models,
                model_factory=factory, state=state, pulse_starts=live.pulse_starts(chunk),
                sample_rate=RATE, pilot_timing='tone-seeded', frame_boundary='eof')
            live.decoded()
            for r in results[-1:]:
                decoded += 1
                index = r.diag.get('source_index')
                shown = r.status in ('received', 'verified') or r.diag.get('displayable')
                # The first packets arrive before tail memory has filled.
                if index is None or not shown or decoded <= 3:
                    continue
                model = models.get(r.diag.get('encoding_type'), bootstrap)
                picture_values = fold.values(model, r) if fold else v7.values_from(model, r.coeffs)
                picture = values_image(picture_values, v7.V7_GRIDS).resize(
                    refs[0].size, resample=3)
                scores.append(ssimulacra2(refs[index], picture))
        return scores

    for case in cases:
        row = {'case': case}
        scores = run(None, case)
        row['no fold'] = [round(float(np.mean(scores)), 1) if scores else None, len(scores)]
        for slots in args.folds:
            fold = LiveFold(slots)
            fold.install()
            try:
                scores = run(fold, case)
            finally:
                fold.uninstall()
            row[f'fold {slots}'] = [round(float(np.mean(scores)), 1) if scores else None,
                                    len(scores)]
        print(json.dumps(row), flush=True)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest='command', required=True)
    b = sub.add_parser('build', help='freeze fold tables from the reference fixture')
    b.add_argument('--folds', type=int, nargs='+', default=[500, 1000])
    b.add_argument('--frames', type=Path, nargs='*',
                   help='fit guest statistics on these instead of the fixture')
    b.add_argument('--design-db', type=float, default=30.0)
    b.add_argument('--signature', type=int, default=16,
                   help='host slots given to the folded-packet signature')
    t = sub.add_parser('selftest', help='live receive path offline, fold on/off')
    t.add_argument('frames', nargs='+', type=Path)
    t.add_argument('--folds', type=int, nargs='+', default=[500])
    t.add_argument('--packets', type=int, default=30)
    t.add_argument('--cases', nargs='+')
    t.add_argument('--quick', action='store_true')
    args = ap.parse_args(argv)
    (build if args.command == 'build' else selftest)(args)


if __name__ == '__main__':
    main()
