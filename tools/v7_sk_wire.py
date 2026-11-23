"""Real-wire confirmation of the nested fold: actual V7 packets, fresh receiver.

Every scored picture is a real stereo ``aspect-fold-500`` packet: production
header, pilots, metadata, EOF, chroma slots, slot gains and level are
untouched.  Only the values in the 1,920 luma slots differ between schemes.
The receiver is the production pulse/equaliser chain; schemes read their
symbols from its equaliser output (the same point the production fold reads).

    .venv/bin/python tools/v7_sk_wire.py measure --out tmp/sk/wire
    .venv/bin/python tools/v7_sk_study.py --noise tmp/sk/wire/NOISE.npz --out tmp/sk/study
    .venv/bin/python tools/v7_sk_wire.py confirm --study tmp/sk/study/STUDY.json --out tmp/sk/wire

Synthetic hiss is not tape validation.
"""
import argparse
import io
import json
import math
import sys
from contextlib import redirect_stdout
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
from numba import njit

from animation_modem import impairments, v7
from animation_modem.v7_core import adapt_packet_for_output
from tools import v7_live, v7_sk_fold as sk, v7_sk_study as study
from tools.v7_local_detail_metrics import attenuate, audio_levels

CONDITIONS = {'clean': None, 'hiss-60': -60., 'hiss-50': -50., 'hiss-45': -45., 'hiss-40': -40.}
PAD = 4800


@njit(cache=True)
def running_power(square, half):
    """Mean of neighbouring slots' squared error, in importance order."""
    n = len(square)
    out = np.empty(n)
    for i in range(n):
        lo, hi = max(0, i-half), min(n, i+half+1)
        total = 0.0
        for j in range(lo, hi):
            total += square[j]
        out[i] = total/(hi-lo)
    return out


@njit(cache=True)
def mean_square(x):
    total = 0.0
    for value in x.ravel():
        total += value*value
    return total/x.size


class Link:
    """One production link (stereo fold or mono) whose luma slot values can
    be replaced.  Everything else in the packet is the production encoder's."""

    def __init__(self, geometry):
        from folding import equaliser_capture
        self.capture = equaliser_capture
        self.g = geometry
        self.wire = geometry.wire
        self.model = geometry.model
        self.index = geometry.model_index
        self.mu = np.asarray(self.model.mu)[self.index]
        self.sd = np.sqrt(np.asarray(self.model.lam)[self.index])

    def packet(self, rgb, symbols=None, counter=1, source_index=1):
        wire = self.wire
        values = wire.values(rgb)
        if self.g.mono:
            return self._mono_packet(values, symbols, counter, source_index)
        model, coefficients = wire.wire.encode_coefficients(wire.base, values, wire.aspect)
        if symbols is not None:
            coefficients = np.array(coefficients, float)
            coefficients[self.index] = self.mu+self.sd*np.asarray(symbols, float)
        packet = v7_live._encode_pulse_frame_coeffs(
            model, coefficients, counter, aspect_code=wire.aspect,
            source_index=source_index,
            pulse_profile_code=wire.wire.pulse_profile_code)
        packet = v7_live._add_coded_pilots(packet, counter, 500, mode=wire.wire.status_mode)
        return adapt_packet_for_output(packet, 96000)

    def _mono_packet(self, values, symbols, counter, source_index):
        """Production mono packet (one leg, mono pilots, coded status) with
        the luma slot values replaced after the production fold has run."""
        wire, codec = self.wire, self.g.codec
        real = codec.encode_coefficients

        def replaced(values, full=None):
            coefficients = np.array(real(values, full), float)
            if symbols is not None:
                coefficients[self.index] = self.mu+self.sd*np.asarray(symbols, float)
            return coefficients

        codec.encode_coefficients = replaced
        try:
            packet = wire.wire.encode_packet(wire.base, values, counter, wire.aspect, source_index)
        finally:
            del codec.encode_coefficients
        return adapt_packet_for_output(packet, 96000)

    def symbols_of(self, result, captured=None):
        """Luma slot values at the equaliser output for one decoded packet."""
        equalised = result.diag.get('mono_fold_eq') if self.g.mono else captured
        if equalised is None:
            return None
        xhat, confidence = equalised[0], equalised[1]
        return (xhat/np.maximum(confidence, 1e-3))[self.index]/self.sd

    def receive(self, audio, noise_dbfs=None, seed=1):
        """(luma symbols or None, production decoded values or None, status)."""
        silence = np.zeros((PAD, 2), np.float32)
        padded = np.concatenate([silence, np.asarray(audio, np.float32), silence])
        if noise_dbfs is not None:
            padded = impairments.Emulator(impairments.Settings(
                noise_dbfs=noise_dbfs, seed=seed)).process(padded)
        with self.capture() as captured, redirect_stdout(io.StringIO()):
            packets, _, _ = self.wire.decode(np.asarray(padded, np.float64))
        if not packets or (not captured and not self.g.mono):
            return None, None, 'lost'
        result, values = packets[0]
        symbols = self.symbols_of(result, captured[0] if captured else None)
        if symbols is None:
            return None, values, 'lost'
        return symbols, values, result.status


def matched(audio, reference):
    """Turn a candidate down to the production packet's RMS and peak."""
    rms, peak = audio_levels(reference)
    return attenuate(audio, rms, peak)


def pictures(folder):
    train_files, test_files = study.corpus(folder)
    return ([study.portrait(f) for f in train_files], [study.portrait(f) for f in test_files],
            train_files, test_files)


def measure(args):
    """Per-slot noise of the real wire on training pictures only."""
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    geometry = study.Geometry(args.profile)
    link = Link(geometry)
    train, _, train_files, _ = pictures(args.corpus)
    small = study.Analyzer(study.LATTICE)
    planes = [small(rgb) for rgb in train]
    stats = study.Statistics(planes)
    linear = study.Linear(geometry, stats, np.zeros(len(geometry.power)))
    arrays, summary = {}, {}
    for condition, level in CONDITIONS.items():
        errors, gains = [], []
        for i, (rgb, plane) in enumerate(zip(train, planes)):
            symbols = linear.encode(plane)
            audio, gain = matched(link.packet(rgb, symbols, i+1, i+1), link.packet(rgb, None, i+1, i+1))
            for seed in range(1 if level is None else args.seeds):
                received, _, status = link.receive(audio, level, 100*seed+i+1)
                if received is not None:
                    errors.append(received-symbols)
                    gains.append(gain)
        errors = np.stack(errors)
        per_slot = np.sqrt(running_power(np.mean(errors**2, axis=0), 16))
        arrays[condition] = per_slot
        half = len(errors)//2
        first = np.sqrt(running_power(np.mean(errors[:half]**2, axis=0), 16))
        second = np.sqrt(running_power(np.mean(errors[half:]**2, axis=0), 16))
        summary[condition] = {
            'packets': len(errors), 'mean_volume_gain': float(np.mean(gains)),
            'wire_noise_sd': float(math.sqrt(mean_square(errors*np.sqrt(geometry.power)))),
            'slot_snr_db_median': float(np.median(-20*np.log10(per_slot))),
            'slot_snr_db_range': [float(-20*np.log10(per_slot.max())), float(-20*np.log10(per_slot.min()))],
            'excess_kurtosis': float(np.mean((errors/per_slot)**4)/np.mean((errors/per_slot)**2)**2-3),
            'split_half_profile_correlation': float(np.corrcoef(np.log(first), np.log(second))[0, 1])}
        print(condition, json.dumps(summary[condition]), flush=True)
    np.savez(out/'NOISE.npz', **arrays)
    (out/'NOISE.json').write_text(json.dumps({
        'pictures': [f.name for f in train_files], 'meaning':
        'per-slot received-minus-sent standard deviation in unit-power symbol units, '
        'training pictures only, smoothed over 33 neighbouring slots', 'conditions': summary},
        indent=1)+'\n')


def build(geometry, stats, noise, training, spec):
    kind = spec['kind']
    if kind == 'linear':
        return study.Linear(geometry, stats, noise)
    if kind == 'production':
        return study.Production(geometry, stats, noise)
    if kind == 'surface':
        return study.PriorSurface(geometry, stats, noise, *spec['dimensions'])
    if kind in ('adaptive', 'levelled'):
        from tools.v7_sk_adaptive import Adaptive, Leveled
        scheme = Adaptive(geometry, stats, noise, training, name=spec['name'], **spec['parameters'])
        return Leveled(scheme, spec['name']) if kind == 'levelled' else scheme
    return study.Nested(geometry, stats, noise, training, name=spec['name'], **spec['parameters'])


def shipped_plane(values):
    """Luma coefficients of the unmodified production decode."""
    rows, cols = study.LATTICE
    plane = np.asarray(values)[:rows*cols].reshape(rows, cols)
    return sk.separable(sk.dct_matrix(rows), np.ascontiguousarray(plane), sk.dct_matrix(cols))


def confirm(args):
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    geometry = study.Geometry(args.profile)
    if geometry.mono:
        # No stereo study tables apply: the mono run scores the stock mono
        # mapping against the adaptive fold tables chosen for mono.
        _, held = study.corpus(args.corpus)
        frozen = {'held_out': [f.name for f in held], 'training': 'see adaptive file',
                  'criterion': {'pictures_needed': len(held)-1},
                  'conditions': {c: {'schemes': {}} for c in CONDITIONS}}
    else:
        frozen = json.loads(Path(args.study).read_text())
    noise_file = np.load(args.noise)
    link = Link(geometry)
    train, test, train_files, test_files = pictures(args.corpus)
    if args.score:
        # Frozen tables, different scored pictures (for example video frames).
        test_files = sorted(Path(args.score).glob('*.png'))+sorted(Path(args.score).glob('*.jpg'))
        test = [study.portrait(f) for f in test_files]
    elif [f.name for f in test_files] != frozen['held_out']:
        raise SystemExit('held-out pictures differ from the frozen study')
    small, big = study.Analyzer(study.LATTICE), study.Analyzer(geometry.big)
    train_planes = [small(rgb) for rgb in train]
    test_planes, test_big = [small(rgb) for rgb in test], [big(rgb) for rgb in test]
    stats = study.Statistics(train_planes)
    tail = study.ensemble_tail(geometry, study.Statistics(test_big).variance)
    need = len(test)-1 if args.score else frozen['criterion']['pictures_needed']
    report = {'study': str(args.study), 'trained_on': frozen['training'],
              'scored': [f.name for f in test_files], 'conditions': {}}
    planes_out = {}
    for condition in args.conditions:
        level = CONDITIONS[condition]
        noise = noise_file[condition]
        specs = [{'kind': 'linear', 'name': 'linear 1,920'},
                 {'kind': 'production', 'name': 'production fold 500'},
                 {'kind': 'surface', 'name': 'earlier surface 3to2, 3 strips', 'dimensions': (3, 2, 3)}]
        if geometry.mono:
            specs = specs[:2]
        for name, summary in frozen['conditions'][condition]['schemes'].items():
            if 'parameters' in summary:
                specs.append({'kind': 'nested', 'name': name, 'parameters': summary['parameters']})
        if args.adaptive and Path(args.adaptive).exists():
            chosen = json.loads(Path(args.adaptive).read_text())['conditions'][condition]
            specs.append({'kind': 'adaptive', 'name': 'adaptive nested fold',
                          'parameters': chosen['parameters']})
            specs.append({'kind': 'levelled', 'name': 'levelled adaptive fold',
                          'parameters': chosen['parameters']})
        schemes = [build(geometry, stats, noise, train_planes, spec) for spec in specs]
        rows = {scheme.name: [] for scheme in schemes}
        rows['production as shipped'] = []
        levels = {name: [] for name in rows}
        lost = {name: 0 for name in rows}
        for i, (rgb, plane, reference) in enumerate(zip(test, test_planes, test_big)):
            index = 100+i
            baseline = link.packet(rgb, None, 1, index)
            for seed in range(args.seeds):
                _, values, status = link.receive(baseline, level, 7000+31*seed+i)
                if values is None:
                    lost['production as shipped'] += 1
                    continue
                rows['production as shipped'].append((i, study.score(geometry, reference, shipped_plane(values))))
            levels['production as shipped'].append(audio_levels(baseline)+(1.,))
            for scheme in schemes:
                symbols = scheme.encode(plane)
                audio, gain = matched(link.packet(rgb, symbols, 1, index), baseline)
                levels[scheme.name].append(audio_levels(audio)+(gain,))
                for seed in range(args.seeds):
                    received, _, status = link.receive(audio, level, 7000+31*seed+i)
                    if received is None:
                        lost[scheme.name] += 1
                        continue
                    decoded = scheme.decode(received)
                    rows[scheme.name].append((i, study.score(geometry, reference, decoded)))
                    if seed == 0:
                        planes_out[f'{condition}|{scheme.name}|{i}'] = decoded
        results = {}
        for name, scored in rows.items():
            merged = []
            for i in range(len(test)):
                group = [r for j, r in scored if j == i]
                if not group:
                    continue
                merged.append({'error': np.mean([r['error'] for r in group]), 'energy': group[0]['energy'],
                               'psnr': np.mean([r['psnr'] for r in group]),
                               'equivalent': np.mean([r['equivalent'] for r in group]),
                               'passed': np.all([r['passed'] for r in group], 0),
                               'rho': np.mean([r['rho'] for r in group], 0)})
            summary = study.summarise(geometry, merged, need)
            summary['ensemble_equivalent_coefficients'] = study.ensemble_equivalent(tail, summary['mean_error'])
            summary['packets_lost'] = lost[name]
            summary['rms'] = float(np.mean([x[0] for x in levels[name]]))
            summary['peak'] = float(np.max([x[1] for x in levels[name]]))
            summary['mean_volume_gain'] = float(np.mean([x[2] for x in levels[name]]))
            results[name] = summary
        for scheme in schemes:
            results[scheme.name]['luma_coefficients_carried'] = scheme.carried()
        base = results['production fold 500']
        for summary in results.values():
            summary['linear_resolution_vs_production'] = {
                'by_ensemble_equivalent': math.sqrt(summary['ensemble_equivalent_coefficients'] /
                                                    base['ensemble_equivalent_coefficients']),
                'by_resolved_cycles': (summary['resolved_cycles']/base['resolved_cycles']
                                       if base['resolved_cycles'] else None)}
        report['conditions'][condition] = results
        print(condition, json.dumps({k: (round(v['psnr_db'], 2), v['ensemble_equivalent_coefficients'],
                                         v['resolved_cycles'], v['packets_lost'])
                                     for k, v in results.items()}), flush=True)
    (out/f'WIRE{args.tag}.json').write_text(json.dumps(report, indent=1)+'\n')
    np.savez_compressed(out/f'decoded_planes{args.tag}.npz', **planes_out)


def grating(height, width, cycles, axis, phase, contrast):
    """Grey sinusoid, ``cycles`` per picture height along rows (axis 0) or
    the same physical pitch along columns (axis 1)."""
    y, x = np.mgrid[:height, :width]
    position = (y+.5)/height if axis == 0 else (x+.5)/height
    value = .5+.5*contrast*np.sin(2*np.pi*cycles*position+phase)
    return np.ascontiguousarray(np.repeat(np.uint8(np.rint(value*255))[..., None], 3, -1))


def gratings(args):
    """Known-frequency probes through the real wire with frozen tables."""
    out = Path(args.out)
    frozen = json.loads(Path(args.study).read_text())
    noise_file = np.load(args.noise)
    geometry = study.Geometry()
    link = Link(geometry)
    train, _, _, _ = pictures(args.corpus)
    small, big = study.Analyzer(study.LATTICE), study.Analyzer(geometry.big)
    train_planes = [small(rgb) for rgb in train]
    stats = study.Statistics(train_planes)
    frequencies = list(range(8, 48, 2))
    phases = [0, math.pi/4, math.pi/2, 3*math.pi/4]
    report = {'frequencies_cycles_per_height': frequencies, 'phases': len(phases),
              'criterion': 'signed transfer 0.5..1.5 and off-pattern residual <= 0.25 of the pattern; '
                           'a frequency passes with 3 of 4 phases; the limit is the highest frequency '
                           'with every lower tested frequency passing', 'conditions': {}}
    for condition in args.conditions:
        level, noise = CONDITIONS[condition], noise_file[condition]
        specs = [{'kind': 'linear', 'name': 'linear 1,920'},
                 {'kind': 'production', 'name': 'production fold 500'}]
        for name, summary in frozen['conditions'][condition]['schemes'].items():
            if 'parameters' in summary:
                specs.append({'kind': 'nested', 'name': name, 'parameters': summary['parameters']})
        schemes = [build(geometry, stats, noise, train_planes, spec) for spec in specs]
        results = {}
        for scheme in schemes:
            results[scheme.name] = {}
            for contrast in (.05, .3):
                for axis in (0, 1):
                    passes = []
                    for cycles in frequencies:
                        count = 0
                        for k, phase in enumerate(phases):
                            rgb = grating(500, 375, cycles, axis, phase, contrast)
                            truth = big(rgb).ravel()
                            audio, _ = matched(link.packet(rgb, scheme.encode(small(rgb)), 1, 5),
                                               link.packet(rgb, None, 1, 5))
                            received, _, _ = link.receive(audio, level, 90+k)
                            if received is None:
                                continue
                            full = np.zeros(truth.size)
                            full[geometry.embed] = scheme.decode(received).ravel()
                            t, d = truth[1:], full[1:]
                            transfer = float(np.dot(t, d)/np.dot(t, t))
                            residual = float(np.sqrt(np.sum((d-transfer*t)**2)/np.dot(t, t)))
                            count += int(.5 <= transfer <= 1.5 and residual <= .25)
                        passes.append(count)
                    limit = 0
                    for cycles, count in zip(frequencies, passes):
                        if count < 3:
                            break
                        limit = cycles
                    results[scheme.name][f'contrast {contrast}, axis {"vertical" if axis == 0 else "horizontal"}'] = {
                        'passing_phases': passes, 'limit_cycles': limit}
            print(condition, scheme.name, json.dumps({k: v['limit_cycles'] for k, v in results[scheme.name].items()}), flush=True)
        report['conditions'][condition] = results
    (out/'GRATINGS.json').write_text(json.dumps(report, indent=1)+'\n')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('command', choices=('measure', 'confirm', 'gratings'))
    parser.add_argument('--corpus', default='tmp/sk/corpus')
    parser.add_argument('--out', default='tmp/sk/wire')
    parser.add_argument('--study', default='tmp/sk/study/STUDY.json')
    parser.add_argument('--noise', default='tmp/sk/wire/NOISE.npz')
    parser.add_argument('--adaptive', default='tmp/sk/study/ADAPTIVE.json')
    parser.add_argument('--conditions', nargs='+', default=list(CONDITIONS))
    parser.add_argument('--seeds', type=int, default=3)
    parser.add_argument('--profile', default='aspect-fold-500', choices=('aspect-fold-500', 'aspect-mono-500'))
    parser.add_argument('--score', help='folder of pictures to score with the frozen tables')
    parser.add_argument('--tag', default='', help='suffix for output file names')
    args = parser.parse_args(argv)
    {'measure': measure, 'confirm': confirm, 'gratings': gratings}[args.command](args)


if __name__ == '__main__':
    main()
