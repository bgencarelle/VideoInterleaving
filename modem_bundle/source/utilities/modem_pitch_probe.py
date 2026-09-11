"""Build reproducible pitch/speed controls from source images; measure reception.

FFmpeg with its rubberband filter is required only by `make`. Effects are
applied to rendered WAVs, never to the production encoder. This models one
processor, not every pitch shifter or a physical tape deck.
"""
import argparse
from fractions import Fraction
import json
from pathlib import Path
import re
import subprocess
import sys
import time

import numpy as np
from PIL import Image
from scipy.io import wavfile
from scipy.signal import resample_poly

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from animation_modem import transport2 as v
from animation_modem.imaging import image_values, plane_shapes


def setup():
    return v.PRESETS['wide'], v.SourceCoder(plane_shapes())


def write(path, audio):
    wavfile.write(path, v.RATE, np.rint(np.clip(audio, -1, 1)*32767).astype(np.int16))


def make(args):
    files = list(args.folder.glob(args.pattern))
    files.sort(key=lambda p: [int(x) if x.isdigit() else x.lower()
                             for x in re.split(r'(\d+)', p.name)])
    files = files[:args.frames]
    if not files:
        raise ValueError('No source images match --folder and --pattern')
    args.out.mkdir(parents=True, exist_ok=True)
    layout, coder = setup()
    values, packets = [], []
    for n, path in enumerate(files, 1):
        with Image.open(path) as image:
            value = image_values(image, coder.shapes)
        values.append(value)
        packets.append(v.encode(value, layout, coder, n, n, len(files)))
    audio = np.concatenate(packets)
    clean = args.out/'clean.wav'
    write(clean, audio)
    np.save(args.out/'values.npy', np.asarray(values))
    ratio = 2**(args.cents/1200)
    for name, pitch in [('processor_zero', 1.), ('pitch_up', ratio),
                        ('pitch_down', 1/ratio)]:
        # Keep stereo linked and tempo fixed. The zero-cent control measures
        # changes introduced by the processor even with no requested shift.
        subprocess.run([args.ffmpeg, '-v', 'error', '-y', '-i', str(clean),
                        '-af', f'rubberband=tempo=1:pitch={pitch}:channels=together',
                        '-c:a', 'pcm_s16le', str(args.out/f'{name}.wav')],
                       check=True, timeout=12)
    factor = Fraction(1/ratio).limit_denominator(10000)
    write(args.out/'speed_up.wav',
          resample_poly(audio, factor.numerator, factor.denominator, axis=0))
    # Coupled time/frequency modulation: 0.3% peak speed variation at 4 Hz.
    high = resample_poly(audio, 8, 1, axis=0)
    t = np.arange(len(audio))/v.RATE
    positions = (np.arange(len(audio)) + .003*v.RATE/(2*np.pi*4)*np.sin(2*np.pi*4*t))*8
    warped = np.stack([np.interp(positions, np.arange(len(high)), high[:, c])
                       for c in range(2)], axis=1)
    write(args.out/'warble.wav', warped)
    manifest = dict(frames=len(files), source_files=[p.name for p in files],
                    cents=args.cents, speed_ratio=ratio, rate=v.RATE,
                    processor='FFmpeg rubberband, tempo=1, channels=together',
                    warble_peak_speed_fraction=.003, warble_hz=4.)
    (args.out/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    print(json.dumps(manifest))


def check(args):
    rate, audio = wavfile.read(args.wav)
    if rate != v.RATE or audio.ndim != 2 or audio.shape[1] != 2:
        raise ValueError('Expected 48 kHz stereo WAV')
    if audio.dtype != np.int16:
        raise ValueError('Expected 16-bit PCM, as produced by make')
    audio = audio.astype(np.float32)/32768
    expected = np.load(args.wav.parent/'values.npy', allow_pickle=False)
    layout, coder = setup()
    if expected.ndim != 2 or expected.shape[1] != coder.count:
        raise ValueError('Reference values do not match the fixed image format')
    rx = v.Receiver(layout, coder, recovery=False, fast=True)
    results, calls = [], []
    started = time.perf_counter()
    for at in range(0, len(audio), 256):
        before = time.perf_counter()
        results.extend(rx.feed(audio[at:at+256]))
        calls.append((time.perf_counter()-before)*1000)
    before = time.perf_counter()
    results.extend(rx.flush())
    calls.append((time.perf_counter()-before)*1000)
    elapsed = time.perf_counter()-started
    verified = [r for r in results if r.identity == 'verified_header']
    matched = [r for r in verified if 1 <= r.absolute <= len(expected)]
    mse = [np.mean((r.values-expected[r.absolute-1])**2) for r in matched]
    speeds = [r.extra['playback_speed'] for r in results]
    print(json.dumps(dict(file=args.wav.name, source_frames=len(expected),
          packets=len(results), verified=len(verified),
          unique_verified=len({r.absolute for r in matched}),
          duration_seconds=len(audio)/rate, receiver_ms=elapsed*1000,
          max_call_ms=max(calls), p95_call_ms=float(np.percentile(calls,95)),
          verified_value_rmse=float(np.sqrt(np.mean(mse))) if mse else None,
          median_reported_speed=float(np.median(speeds)) if speeds else None)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    build = sub.add_parser('make')
    build.add_argument('--folder', type=Path, required=True)
    build.add_argument('--pattern', default='*.webp')
    build.add_argument('--frames', type=int, default=20)
    build.add_argument('--out', type=Path, default=Path('pitch_probe'))
    build.add_argument('--cents', type=float, default=5.)
    build.add_argument('--ffmpeg', default='ffmpeg')
    read = sub.add_parser('check')
    read.add_argument('--wav', type=Path, required=True)
    args = parser.parse_args()
    if args.command == 'make':
        if not 1 <= args.frames <= 65535 or not np.isfinite(args.cents) or abs(args.cents)>100:
            parser.error('Use 1..65535 frames and a finite shift within +/-100 cents')
        make(args)
    else:
        check(args)


if __name__ == '__main__':
    main()
