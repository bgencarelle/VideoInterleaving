#!/usr/bin/env python3
"""Synthetic-ideal modem bench: decode cost + fidelity on a clean wire.

Runs a deterministic encode -> decode round trip entirely in memory (no audio
devices) on the one wire (transport3.WIRE), and reports what matters for the
long-term tape goal:

  - per-packet decode CPU (measure_pulses acquisition, demod, equalise, gather,
    inverse) -- the number any decode optimization has to move,
  - encode CPU per frame (composite + values + OFDM assembly),
  - round-trip fidelity (PSNR of decoded values against what was sent),
  - the geometry each pair costs: band, frame rate, value capacity.

The wire is IDEAL by design: no noise, no wow/flutter, no band-limit by
default. Tape emulation is deliberately not attempted here -- it cannot be made
realistic, and real tape measurement is a hardware job. Two cheap dimensions
probe the recovery margin without pretending to be tape:

  - `--band-limit-hz` brick-walls the emitted stream. When a tape deck cannot
    hold the wire's full band, does the image still DECODE with acceptable
    color, or is softness the only loss?
  - `--noise-dbfs` adds deterministic AWGN relative to a peak-1.0 wire: the
    allowed-noise margin of the wire at full band.

    .venv/bin/python tools/bench_modem.py
    .venv/bin/python tools/bench_modem.py --profile color-dct --frames 32
    .venv/bin/python tools/bench_modem.py \
        --band-limit-hz 15000      # tape deck stops at 15 kHz
    .venv/bin/python tools/bench_modem.py \
        --band-limit-hz 15000 --noise-dbfs -40   # ...and a dirty deck
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from animation_modem import transport3 as V3                        # noqa: E402
from animation_modem.imaging import (plane_shapes, plane_grids,      # noqa: E402
                                     fit_shapes, image_values)
from modem_bake import ModemLibrary                                  # noqa: E402


def coder_for(layout, profile):
    """The (shapes, grids, coder) the runtime builds for this pair."""
    wanted = plane_shapes(profile)
    grids = plane_grids(profile)
    shapes = fit_shapes(wanted, layout.capacity)
    if shapes != wanted:
        grids = shapes          # a shrunk corner is not a corner (see fit_allocation)
    return shapes, grids, V3.SourceCoder(shapes, grids=grids)


def psnr(a, b, peak=2.0):
    """PSNR over two [-1,1] value vectors of the same length."""
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    mse = float(np.mean((a-b)**2))
    if mse <= 0:
        return float('inf')
    return 10*np.log10(peak*peak/mse)


def band_limit(wire, cutoff, rate=V3.REFERENCE_RATE):
    """Brick-wall HF roll-off on an otherwise ideal wire.

    Deterministic: zero every FFT bin above `cutoff` on the whole emitted
    stream, per channel. The preamble peak is unchanged, so nothing downstream
    renormalises -- this is a pure lost-band probe.
    """
    win = np.fft.rfft(wire, axis=0)
    f = np.fft.rfftfreq(len(wire), 1/rate)
    win[f > cutoff] = 0
    return np.fft.irfft(win, n=len(wire), axis=0).astype(np.float32)


def add_noise(wire, dbfs, seed=0):
    """Deterministic additive white Gaussian noise on a peak-1.0 wire.

    `dbfs` is the noise level relative to a peak of 1.0; -40 means noise at
    1% of peak amplitude. Same seed every run so sweeps are comparable.
    """
    if not dbfs:
        return wire
    amp = 10.0 ** (float(dbfs) / 20.0)
    rng = np.random.default_rng(seed)
    return (wire + rng.normal(0, amp, wire.shape)).astype(np.float32)


def round_trip(layout, coder, grids, library, pair, frames, identify=False,
               band_limit_hz=None, header_tolerance=2, noise_dbfs=None):
    """Encode `frames` packets, decode them back, return per-frame records."""
    candidates = None
    if identify:
        candidates = [(layout, coder, None)]

    # --- encode the clean stream ------------------------------------------
    stream = []
    encode_ms = []
    for n in range(frames):
        started = time.perf_counter()
        im = library.composite(n % library.frames, *pair)
        values = image_values(im, grids)
        audio = V3.encode(values, layout, coder, n+1,
                          (n % library.frames)+1, frames, profile=0)
        encode_ms.append((time.perf_counter()-started)*1000)
        stream.append(np.asarray(audio, np.float32))
    wire = np.concatenate(stream)
    if band_limit_hz:
        wire = band_limit(wire, band_limit_hz)
    wire = add_noise(wire, noise_dbfs)
    source = image_values(library.composite(frames-1, *pair), grids)

    # --- decode the clean stream ------------------------------------------
    receiver = V3.Receiver(layout, coder, pulse_only=True,
                           candidates=candidates, fast=True,
                           header_tolerance=header_tolerance)
    records = []
    chunk = 8192
    for start in range(0, len(wire), chunk):
        records.extend(receiver.feed(wire[start:start+chunk]))
    records.extend(receiver.flush())

    out = []
    for result in records:
        if result.values is None:
            out.append({'status': result.status, 'identity': result.identity,
                        'acquisition_path': result.extra.get('acquisition_path'),
                        'decode_ms': result.extra.get('decode_ms'),
                        'receive_cpu_ms': result.extra.get('receive_cpu_ms'),
                        'pilot_error': result.pilot_error,
                        'coverage': result.coverage, 'psnr': None})
            continue
        out.append({'status': result.status, 'identity': result.identity,
                    'acquisition_path': result.extra.get('acquisition_path'),
                    'decode_ms': result.extra.get('decode_ms'),
                    'receive_cpu_ms': result.extra.get('receive_cpu_ms'),
                    'pilot_error': result.pilot_error,
                    'coverage': result.coverage,
                    'psnr': psnr(source, result.values)})
    return out, encode_ms


def bench(profile, library, frames, identify=False,
          band_limit_hz=None, header_tolerance=2, noise_dbfs=None):
    layout = V3.WIRE
    preset = layout.name
    shapes, grids, coder = coder_for(layout, profile)
    records, encode_ms = round_trip(layout, coder, grids, library,
                                    (1, 0), frames, identify, band_limit_hz,
                                    header_tolerance, noise_dbfs)
    verified = [r for r in records if r['identity'] == 'verified_header']
    good = [r for r in records if r['psnr'] is not None]
    decode = [r['decode_ms'] for r in good if r['decode_ms'] is not None]
    acquire = [r['receive_cpu_ms'] for r in good
               if r['receive_cpu_ms'] is not None]
    psnrs = [r['psnr'] for r in good]
    modes = {}
    for r in records:
        modes[r['acquisition_path']] = modes.get(r['acquisition_path'], 0)+1
    return {
        'preset': preset, 'profile': profile,
        'capacity': layout.capacity, 'band_hz': int(layout.band[1]),
        'fps': layout.fps, 'values': coder.count, 'bytes': layout.frame,
        'shape': '{}x{}'.format(shapes[0][1], shapes[0][0]),
        'grid': '{}x{}'.format(grids[0][1], grids[0][0]),
        'frames': len(records), 'verified': len(verified),
        'psnr': float(np.mean(psnrs)) if psnrs else None,
        'decode_ms': float(np.mean(decode)) if decode else None,
        'receive_cpu_ms': float(np.mean(acquire)) if acquire else None,
        'encode_ms': float(np.mean(encode_ms)),
        'paths': modes, 'band_limit_hz': band_limit_hz,
        'header_tolerance': header_tolerance, 'noise_dbfs': noise_dbfs,
    }


HEADER = ('preset        profile    cap   bandH  fps   wire   vals  '
          'psnr    decms  ccpums  encms  verif paths/acquire')


def row(r):
    return (f"{r['preset']:<14s} {r['profile']:<10s} "
            f"{r['capacity']:5d}  {r['band_hz']/1000:5.2f}k "
            f"{r['fps']:5.2f} {r['bytes']:5d} {r['values']:5d} "
            f"{r['psnr']:6.1f} {r['decode_ms']:6.2f}  "
            f"{r['receive_cpu_ms']:6.2f}  {r['encode_ms']:6.2f} "
            f"{r['verified']:4d}/{r['frames']:4d} {r['paths']}")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--modem-dir', type=Path, default='images_modem',
                   help='Baked 80x96 modem library (default images_modem)')
    p.add_argument('--profile', default='auto',
                   help='Profile to test, or auto to follow the bake')
    p.add_argument('--frames', type=int, default=24)
    p.add_argument('--band-limit-hz', type=int, default=None,
                   help='Brick-wall the emitted stream above this frequency '
                        '(a tape deck that cannot hold the band).')
    p.add_argument('--noise-dbfs', type=float, default=None,
                   help='Add deterministic AWGN at this level relative to a '
                        'peak-1.0 wire (dirty medium; e.g. -40).')
    p.add_argument('--header-tolerance', type=int, default=2,
                   help='Max header bits to correct from soft reliability '
                        'when the CRC fails (default 2).')
    p.add_argument('--tolerance-sweep', type=int, default=None, metavar='MAX',
                   help='Report verified-frame counts across header-tolerance '
                        '0..MAX for the given band-limit.')
    p.add_argument('--identify', action='store_true',
                   help='Decode against the header-declared profile coders '
                        'instead of a pinned coder')
    args = p.parse_args(argv)

    library = ModemLibrary(args.modem_dir)
    profile = args.profile if args.profile != 'auto' else library.profile
    note = f'bake {library.root}: {library.frames} frames, {library.profile}'
    if args.band_limit_hz:
        note += f' -- wire band-limited to {args.band_limit_hz} Hz'
    if args.noise_dbfs:
        note += f' -- noise at {args.noise_dbfs} dBFS'
    print(note)
    if args.tolerance_sweep is not None:
        print(f'tolerance sweep ({V3.WIRE.name}, frames {args.frames})')
        print('tol  verified  psnr    decms')
        for tol in range(args.tolerance_sweep + 1):
            r = bench(profile, library, args.frames, args.identify,
                      args.band_limit_hz, tol, args.noise_dbfs)
            print(f"{tol:3d}  {r['verified']:6d}/{r['frames']:4d} "
                  f"{r['psnr']:6.1f} {r['decode_ms']:6.2f}")
        return
    print(HEADER)
    r = bench(profile, library, args.frames, args.identify,
              args.band_limit_hz, args.header_tolerance, args.noise_dbfs)
    print(row(r))


if __name__ == '__main__':
    main()