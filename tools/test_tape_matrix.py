#!/usr/bin/env python3
"""Run repeatable 96 kHz synthetic tape tests on modem_screen's test source.

This is a regression matrix, not a claim to model a particular deck. Real tape
captures remain authoritative. Outputs (WAVs, decoded PNGs, logs, and CSV) go
under scratch/ by default.
"""
import argparse
import csv
import subprocess
import sys
import wave
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from scipy.signal import butter, resample_poly, sosfilt

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from tools.measure_plane_survival import plane_metrics                  # noqa: E402


RATE = 96000


@dataclass(frozen=True)
class TapeCase:
    name: str
    lowpass: float = 0
    highpass: float = 0
    noise_dbfs: float | None = None
    wow: float = 0                 # peak speed variation, percent
    flutter: float = 0             # peak speed variation, percent
    azimuth_us: float = 0          # right-channel delay
    crosstalk: float = 0
    right_gain_db: float = 0
    dc: float = 0
    hum_dbfs: float | None = None
    bias_hz: float = 0
    bias_dbfs: float | None = None
    saturation: float = 0          # tanh drive; 0 disables
    dropout_ms: float = 0
    dropout_every: float = 0
    nr_pump: float = 0             # approximate unmatched treble companding


CASES = (
    TapeCase('clean-96k'),
    TapeCase('lowpass-18k', lowpass=18000),
    TapeCase('lowpass-15k', lowpass=15000),
    TapeCase('lowpass-12k', lowpass=12000),
    TapeCase('lowpass-10k', lowpass=10000),
    TapeCase('hiss-45', noise_dbfs=-45),
    TapeCase('hiss-40', noise_dbfs=-40),
    TapeCase('hiss-35', noise_dbfs=-35),
    TapeCase('wow-flutter', wow=.45, flutter=.12),
    TapeCase('azimuth-12us', azimuth_us=12),
    TapeCase('crosstalk-10pct', crosstalk=.10),
    TapeCase('right-minus-4db', right_gain_db=-4),
    TapeCase('dc-hum', dc=.025, hum_dbfs=-38),
    TapeCase('bias-leak-30k', bias_hz=30000, bias_dbfs=-30),
    TapeCase('soft-saturation', saturation=2.0),
    TapeCase('dropouts', dropout_ms=12, dropout_every=.7),
    TapeCase('nr-pumping', nr_pump=.55),
    TapeCase('type-i', highpass=70, lowpass=12000, noise_dbfs=-39,
             wow=.35, flutter=.10, azimuth_us=7, crosstalk=.08,
             right_gain_db=-2, saturation=1.35),
    TapeCase('type-ii', highpass=50, lowpass=16000, noise_dbfs=-44,
             wow=.22, flutter=.07, azimuth_us=4, crosstalk=.05,
             right_gain_db=-1, saturation=1.15),
)


def read_wav(path):
    with wave.open(str(path), 'rb') as src:
        rate = src.getframerate()
        if src.getnchannels() != 2 or src.getsampwidth() != 2:
            raise ValueError(f'expected stereo 16-bit WAV: {path}')
        raw = np.frombuffer(src.readframes(src.getnframes()), '<i2')
    return raw.reshape(-1, 2).astype(np.float64)/32768, rate


def write_wav(path, samples, rate=RATE):
    pcm = np.clip(np.rint(samples*32767), -32768, 32767).astype('<i2')
    with wave.open(str(path), 'wb') as out:
        out.setnchannels(2)
        out.setsampwidth(2)
        out.setframerate(rate)
        out.writeframes(pcm.tobytes())


def delay_channel(x, delay):
    at = np.arange(len(x), dtype=float) - delay
    return np.interp(at, np.arange(len(x)), x, left=0, right=0)


def time_warp(x, case):
    if not case.wow and not case.flutter:
        return x
    t = np.arange(len(x))/RATE
    # Integrating speed error gives source-position error. Fixed phases keep the
    # matrix deterministic while avoiding a conveniently zero starting slope.
    speed = ((case.wow/100)*np.sin(2*np.pi*.55*t + .4) +
             (case.flutter/100)*np.sin(2*np.pi*7.3*t + 1.1))
    position = np.arange(len(x), dtype=float) + np.cumsum(speed)
    base = np.arange(len(x), dtype=float)
    return np.column_stack([np.interp(position, base, x[:, channel])
                            for channel in range(2)])


def tape_noise(count, dbfs, rng):
    noise = rng.standard_normal((count, 2))
    # Cassette hiss is weighted toward the upper band, not flat AWGN.
    high = sosfilt(butter(2, 1800, btype='highpass', fs=RATE, output='sos'),
                   noise, axis=0)
    high /= max(np.sqrt(np.mean(high*high)), 1e-12)
    return high*10**(dbfs/20)


def nr_pump(x, amount):
    """Cheap unmatched-NR stress: envelope-controlled upper-band attenuation."""
    low = sosfilt(butter(2, 2500, fs=RATE, output='sos'), x, axis=0)
    high = x-low
    envelope = np.maximum(np.abs(low[:, 0]), np.abs(low[:, 1]))
    alpha = np.exp(-1/(.060*RATE))
    tracked = np.empty_like(envelope)
    state = 0.0
    for i, value in enumerate(envelope):
        state = max(value, alpha*state + (1-alpha)*value)
        tracked[i] = state
    reference = max(np.percentile(tracked, 90), 1e-6)
    openness = np.clip(tracked/reference, 0, 1)
    gain = 1-amount*(1-openness)
    return low + high*gain[:, None]


def impair(source, case, seed=2026):
    rng = np.random.default_rng(seed)
    x = np.asarray(source, float).copy()
    x[:, 1] *= 10**(case.right_gain_db/20)
    if case.crosstalk:
        left, right = x[:, 0].copy(), x[:, 1].copy()
        x[:, 0] = (1-case.crosstalk)*left + case.crosstalk*right
        x[:, 1] = case.crosstalk*left + (1-case.crosstalk)*right
    x = time_warp(x, case)
    if case.azimuth_us:
        x[:, 1] = delay_channel(x[:, 1], case.azimuth_us*RATE/1e6)
    filters = []
    if case.highpass:
        filters.append(butter(4, case.highpass, btype='highpass', fs=RATE,
                              output='sos'))
    if case.lowpass:
        filters.append(butter(4, case.lowpass, fs=RATE, output='sos'))
    if filters:
        x = sosfilt(np.concatenate(filters), x, axis=0)
    if case.nr_pump:
        x = nr_pump(x, case.nr_pump)
    t = np.arange(len(x))/RATE
    if case.dc:
        x += case.dc
    if case.hum_dbfs is not None:
        hum = 10**(case.hum_dbfs/20)*np.sin(2*np.pi*60*t + .2)
        x += hum[:, None]
    if case.bias_hz and case.bias_dbfs is not None:
        bias = 10**(case.bias_dbfs/20)*np.sin(2*np.pi*case.bias_hz*t)
        x += bias[:, None]
    if case.noise_dbfs is not None:
        x += tape_noise(len(x), case.noise_dbfs, rng)
    if case.saturation:
        drive = case.saturation
        x = np.tanh(drive*x)/np.tanh(drive)
    if case.dropout_ms and case.dropout_every:
        duration = max(1, round(case.dropout_ms*RATE/1000))
        period = max(duration+1, round(case.dropout_every*RATE))
        fade = min(duration//3, round(.002*RATE))
        for start in range(period, len(x), period):
            stop = min(len(x), start+duration)
            x[start:stop] = 0
            if fade:
                before = max(0, start-fade)
                x[before:start] *= np.linspace(1, 0, start-before)[:, None]
                after = min(len(x), stop+fade)
                x[stop:after] *= np.linspace(0, 1, after-stop)[:, None]
    return np.clip(x, -1, 1)


def score(reference, decoded):
    rows = []
    for ref in sorted(reference.glob('frame_*.png')):
        got = decoded/ref.name
        if got.exists():
            rows.append(plane_metrics(ref, got))
    if not rows:
        return 0, [float('nan')]*6
    values = np.asarray(rows)
    return len(rows), [*values[:, :, 0].mean(axis=0),
                       *values[:, :, 1].mean(axis=0)]


def montage(out, cases, frame='frame_00010.png'):
    cells = []
    for case in cases:
        path = out/f'{case.name}-frames'/frame
        if path.exists():
            cells.append((case.name, Image.open(path).convert('RGB')))
    if not cells:
        return
    width, height = cells[0][1].size
    columns = 5
    canvas = Image.new('RGB', (columns*width, ((len(cells)+columns-1)//columns)*(height+16)),
                       'white')
    draw = ImageDraw.Draw(canvas)
    for i, (name, image) in enumerate(cells):
        x, y = (i % columns)*width, (i//columns)*(height+16)
        draw.text((x+2, y+2), name, fill='black')
        canvas.paste(image, (x, y+16))
    canvas.save(out/'comparison.png')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=Path('scratch/tape-matrix'))
    parser.add_argument('--frames', type=int, default=30)
    parser.add_argument('--profile', choices=('hd-dwt', 'tape-80x60'),
                        default='hd-dwt')
    parser.add_argument('--only', action='append', default=[],
                        help='run only named case(s); may be repeated')
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    root = Path(__file__).resolve().parent.parent
    python = root/'.venv/bin/python'
    source48 = args.out/'source-48k.wav'
    subprocess.run([python, root/'modem_screen.py', '--source', 'test',
                    '--profile', args.profile, '--wire', 'tape',
                    '--frames', str(args.frames),
                    '--write', source48], check=True)
    audio48, rate = read_wav(source48)
    if rate != 48000:
        raise SystemExit(f'expected 48 kHz source, got {rate}')
    source96 = resample_poly(audio48, 2, 1, axis=0)
    write_wav(args.out/'source-96k.wav', source96)

    selected = set(args.only)
    # The clean 96 kHz decode is the objective image reference even when the
    # caller asks for only one damaged case.
    cases = [case for case in CASES
             if not selected or case.name == 'clean-96k' or case.name in selected]
    unknown = selected-{case.name for case in CASES}
    if unknown:
        raise SystemExit(f'unknown cases: {", ".join(sorted(unknown))}')
    summary = []
    for case in cases:
        print(case.name, flush=True)
        wav = args.out/f'{case.name}.wav'
        frames = args.out/f'{case.name}-frames'
        write_wav(wav, impair(source96, case))
        with (args.out/f'{case.name}.log').open('w') as log:
            subprocess.run([python, root/'tools/decode_wav.py', wav,
                            '--out', frames, '--frames', str(args.frames),
                            '--scale', '1', '--raw'], stdout=log,
                           stderr=subprocess.STDOUT, check=False)
        if case.name == 'clean-96k':
            reference = frames
        if 'reference' not in locals():
            continue
        count, metrics = score(reference, frames)
        summary.append([case.name, count, *metrics])
    with (args.out/'summary.csv').open('w', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(['case', 'decoded_frames', 'Y_psnr', 'Cb_psnr', 'Cr_psnr',
                         'Y_ssim', 'Cb_ssim', 'Cr_ssim'])
        writer.writerows(summary)
    montage(args.out, cases)
    print(f'results: {args.out.resolve()}')


if __name__ == '__main__':
    main()
