"""Score the nested-fold modes against stock through the live receiver.

Everything here uses the code a real link uses: the live sender's values and
wires build the packets, the receiver's profile dispatcher decodes them, and
the picture scored is the one the dispatcher would show.  Impairments are the
repository's torture cases; speed and reverse follow
``modem_tests/test_v7_reverse_torture.py`` (a waveform is generated, resampled
or reversed, and decoded as usual).

    .venv/bin/python tools/v7_nested_eval.py torture --pictures a.png b.png
    .venv/bin/python tools/v7_nested_eval.py playback --pictures a.png

Synthetic channels are not tape validation.
"""
import argparse
import io
import json
import sys
from contextlib import redirect_stdout
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
from scipy.signal import resample_poly

from animation_modem import v7
from tools import v7_live, v7_sk_study as study
from tools.v7_sk_wire import shipped_plane
from tools.v7_torture_matrix import CASES, Case, impair

v7_live._ensure_test_modem_path()
import nested_fold                                                      # noqa: E402
import tone_code                                                        # noqa: E402
from aspect_fold import AspectFoldWire                                  # noqa: E402
from aspect_mono import AspectMonoWire                                  # noqa: E402

TARGET = .1521/np.sqrt(1+10**(v7.CLOCK_REL_DB/10))
MODES = ('aspect-fold-500', 'aspect-mono-500', 'mono nested', 'stereo-slices', 'stereo nested')
SPEEDS = {.5: (2, 1), 1.5: (2, 3), 2.0: (1, 2)}
EXTRA_CASES = (Case('hiss-50', noise_dbfs=-50), Case('hiss-30', noise_dbfs=-30),
               Case('azimuth-50us', azimuth_us=50), Case('right-minus-12db', right_gain_db=-12),
               Case('crosstalk-25pct', crosstalk=.25),
               Case('left-leg-only', one_leg_only=True))


class Rig:
    """Live sender wires and the live receiver's dispatcher, in memory."""

    def __init__(self, layout='auto'):
        self.base = v7.load_model(TARGET, 'box')
        self.dispatcher = v7_live._AdaptiveProfileDecoder(
            v7_live._experimental_fold(500), self.base,
            aspect_tail=v7_live.DEFAULT_ASPECT_TAIL)
        self.senders = {
            'aspect-mono-500': AspectMonoWire(self.base, side='left'),
            'mono nested': nested_fold.enable_mono(AspectMonoWire(self.base, side='left'), send=True),
            'stereo-slices': nested_fold.slice_wire('auto'),
            'stereo nested': nested_fold.slice_wire('auto', send=True),
            'aspect-fold-500': AspectFoldWire('auto', v7_live.DEFAULT_ASPECT_TAIL)}

    # ----------------------------------------------------------------- send
    def _values(self, rgb, luma_mask, chroma_masks):
        """Grid values as the live sender makes them for this wire."""
        return v7_live._values(self.base, rgb, 'box', 1.0, dct_encode=True,
                               chroma_sent_for=chroma_masks)[:2]

    def audio(self, mode, rgb, packets):
        """48 kHz stereo stream of ``packets`` packets of one picture."""
        wire = self.senders[mode]
        if mode in ('aspect-mono-500', 'mono nested'):
            values, code = self._values(rgb, None, lambda c: v7_live._chroma_sent_masks(
                wire._codec(wire._packet_model(self.base, c))))
            return np.concatenate([
                wire.encode_packet(self.base, values, counter, aspect_code=code,
                                   source_index=counter-1)
                for counter in range(1, packets+1)]).astype(np.float64)
        if mode in ('stereo-slices', 'stereo nested'):
            for code in range(8):
                wire.model_for(self.base, wire.layout_for(code))
            values, code = self._values(rgb, None, lambda c: wire.chroma_sent_masks(wire.layout_for(c)))
            return wire.encode(self.base, [values]*packets,
                               aspect_codes=[code]*packets).astype(np.float64)
        values, code = self._values(rgb, None, lambda c: v7_live._chroma_sent_masks(
            wire.codec(wire.model_for(self.base, wire.layout_for(c)))))
        model, coefficients = wire.encode_coefficients(self.base, values, code)
        return np.concatenate([
            tone_code.add_tone_code(
                v7.encode_pulse_frame_coeffs(
                    model, coefficients, counter, aspect_code=code,
                    source_index=counter-1, pilot_tones=False, eof_marker=True,
                    pulse_profile_code=wire.pulse_profile_code),
                counter, tone_code.encode_status(wire.status_mode))
            for counter in range(1, packets+1)]).astype(np.float64)

    # -------------------------------------------------------------- receive
    def _results(self, audio, rate, mode, single, reverse):
        """Decoded results of one input (stereo, or one channel as mono)."""
        dispatcher = self.dispatcher
        audio = np.ascontiguousarray(audio[::-1] if reverse else audio, dtype=np.float32)
        dispatcher.active_mode = dispatcher.dispatch_mode = mode
        if single:
            dispatcher.active_side = 0
        state = v7.PulseState(tail_memory=False)
        if reverse:
            state.set_playback_direction(-1)
            results = []
            for start, scale, _confidence, way in v7.pulse_frame_hits(audio, sample_rate=rate):
                if way < 0:
                    results += v7.decode_reverse_packet(
                        self.base, audio, start, scale, state=state, sample_rate=rate,
                        pilot_timing='tone-seeded')[0]
            return results
        return v7.decode_pulse_stream(self.base, audio, state=state, sample_rate=rate,
                                      pilot_timing='tone-seeded', frame_boundary='eof')[0]

    def shown(self, mode, capture, rate=96000, reverse=False, legs=None):
        """[(source index, grid values, note)] for every picture shown.

        ``capture``: what the receiver hears (stereo, or one channel for a
        mono sum).  ``legs``: for the two-channel wires, which input channels
        to read (default: every channel present, forwards or reversed)."""
        dispatcher = self.dispatcher
        dispatcher._last_slices = False
        dispatcher._last_layouts.clear()
        dispatcher.slice_wire.reset()
        dispatcher.aspect_mono_wire.reset_nested()
        capture = np.asarray(capture)
        if capture.ndim == 1:
            capture = capture[:, None]
        channels = capture.shape[1]
        out = []
        dispatcher.install()
        try:
            with tone_code.coded_pilot_timing(), redirect_stdout(io.StringIO()):
                if mode == 'aspect-fold-500':
                    if channels < 2:
                        capture = np.repeat(capture, 2, axis=1)
                    for result in self._results(capture, rate, dispatcher.aspect_mode, False, reverse):
                        self._keep(out, result, lambda r: dispatcher.values(self.base, r))
                elif mode in ('aspect-mono-500', 'mono nested'):
                    # The mono wire is on the left channel; a mono sum has one.
                    for result in self._results(capture[:, :1], rate,
                                                dispatcher.aspect_mono_mode, True, reverse):
                        self._keep(out, result, lambda r: dispatcher.values(self.base, r))
                else:
                    use = list(range(channels)) if legs is None else list(legs)
                    per_leg = {}
                    for leg in use:
                        for result in self._results(capture[:, leg:leg+1], rate,
                                                    dispatcher.aspect_mono_mode, True, reverse):
                            index = result.diag.get('source_index')
                            if index is None or result.status == 'lost':
                                continue
                            half = dispatcher.slice_half(self.base, result, leg)
                            if half is not None:
                                per_leg.setdefault(int(index), []).append((result, half))
                    for index in sorted(per_leg, reverse=reverse):
                        result = per_leg[index][0][0]
                        halves = [half for _, half in per_leg[index]]
                        values = dispatcher.slice_values(halves, result)
                        out.append((index, np.asarray(values, float),
                                    result.diag.get('slices_shown', '')))
        finally:
            dispatcher.uninstall()
        return out

    @staticmethod
    def _keep(out, result, values):
        index = result.diag.get('source_index')
        if index is None or (result.status == 'lost' and not result.diag.get('displayable')):
            return
        out.append((int(index), np.asarray(values(result), float), result.status))


def wide(audio48):
    """The stock matrix's conversion of the 48 kHz wire to a 96 kHz capture."""
    return resample_poly(audio48, 2, 1, axis=0)


class Scorer:
    def __init__(self, files):
        self.geometry = study.Geometry('aspect-mono-500')
        big = study.Analyzer(self.geometry.big)
        self.rgbs = [study.portrait(f) for f in files]
        self.references = [big(rgb) for rgb in self.rgbs]
        self.tail = study.ensemble_tail(self.geometry, study.Statistics(self.references).variance)

    def error(self, picture, values):
        return study.score(self.geometry, self.references[picture], shipped_plane(values))['error']

    def effective(self, errors):
        return study.ensemble_equivalent(self.tail, float(np.mean(errors))) if errors else None


def run_cases(args, cases, modes, tag):
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rig, scorer = Rig(), Scorer(args.pictures)
    streams = {(mode, i): wide(rig.audio(mode, rgb, args.frames))
               for mode in modes for i, rgb in enumerate(scorer.rgbs)}
    report = {'pictures': [Path(f).name for f in args.pictures], 'frames': args.frames,
              'tables': str(nested_fold.TABLES), 'cases': {}}
    for case in cases:
        row = {}
        for mode in modes:
            errors, shown = [], 0
            for i in range(len(scorer.rgbs)):
                capture = impair(streams[mode, i], case, seed=args.seed+i)
                for _, values, _ in rig.shown(mode, capture):
                    errors.append(scorer.error(i, values))
                    shown += 1
            row[mode] = {'shown': shown, 'sent': args.frames*len(scorer.rgbs),
                         'effective_coefficients': scorer.effective(errors)}
        report['cases'][case.name] = row
        print(case.name, json.dumps({m: (row[m]['shown'], row[m]['effective_coefficients'])
                                     for m in modes}), flush=True)
        (out/f'{tag}.json').write_text(json.dumps(report, indent=1)+'\n')
    lines = ['| Case | '+' | '.join(modes)+' |', '|---|'+'---:|'*len(modes)]
    for name, row in report['cases'].items():
        lines.append(f'| {name} | '+' | '.join(
            'none shown' if row[m]['effective_coefficients'] is None else
            f"{row[m]['effective_coefficients']:,} ({row[m]['shown']}/{row[m]['sent']})"
            for m in modes)+' |')
    (out/f'{tag}.md').write_text('\n'.join(lines)+'\n')


def playback(args):
    """Other playback speeds and reverse, each against normal playback."""
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rig, scorer = Rig(), Scorer(args.pictures)
    modes = args.modes
    report = {}
    for mode in modes:
        rows = {}
        for i, rgb in enumerate(scorer.rgbs):
            normal = wide(rig.audio(mode, rgb, args.frames))
            variants = {'1x forward': (normal, False), '1x reverse': (normal, True)}
            for speed, (up, down) in SPEEDS.items():
                capture = resample_poly(normal, up, down, axis=0)
                variants[f'{speed}x forward'] = (capture, False)
                variants[f'{speed}x reverse'] = (capture, True)
            for name, (capture, reverse) in variants.items():
                shown = rig.shown(mode, capture, reverse=reverse)
                entry = rows.setdefault(name, {'errors': [], 'shown': 0, 'order_ok': True})
                order = [index for index, _, _ in shown]
                entry['order_ok'] &= order == sorted(order, reverse=reverse)
                entry['shown'] += len(shown)
                entry['errors'] += [scorer.error(i, values) for _, values, _ in shown]
        report[mode] = {name: {'shown': e['shown'], 'sent': args.frames*len(scorer.rgbs),
                               'in_order': bool(e['order_ok']),
                               'effective_coefficients': scorer.effective(e['errors'])}
                        for name, e in rows.items()}
        print(mode, json.dumps({n: (r['shown'], r['effective_coefficients'], r['in_order'])
                                for n, r in report[mode].items()}), flush=True)
    (out/'PLAYBACK.json').write_text(json.dumps(report, indent=1)+'\n')
    names = list(next(iter(report.values())))
    lines = ['| Playback | '+' | '.join(modes)+' |', '|---|'+'---:|'*len(modes)]
    for name in names:
        lines.append(f'| {name} | '+' | '.join(
            f"{report[m][name]['effective_coefficients']:,} ({report[m][name]['shown']}/{report[m][name]['sent']})"
            if report[m][name]['effective_coefficients'] is not None else 'none shown'
            for m in modes)+' |')
    (out/'PLAYBACK.md').write_text('\n'.join(lines)+'\n')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('command', choices=('torture', 'playback'))
    parser.add_argument('--pictures', nargs='+', required=True)
    parser.add_argument('--out', default='tmp/sk/nested-eval')
    parser.add_argument('--frames', type=int, default=9)
    parser.add_argument('--modes', nargs='+', default=list(MODES))
    parser.add_argument('--only', nargs='+', default=[])
    parser.add_argument('--seed', type=int, default=2026)
    args = parser.parse_args(argv)
    if args.command == 'playback':
        return playback(args)
    cases = [case for case in CASES+EXTRA_CASES if not args.only or case.name in args.only]
    run_cases(args, cases, args.modes, 'TORTURE')


if __name__ == '__main__':
    main()
