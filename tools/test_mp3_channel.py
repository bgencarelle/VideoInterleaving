#!/usr/bin/env python3
"""Measure MP3-320 damage from DC through Nyquist.

Generates known-phase and noise probes, round-trips them through libmp3lame,
then reports transfer magnitude, phase, coherence, and impulse ringing.
"""
import argparse
import subprocess
import wave
from pathlib import Path

import numpy as np
from scipy.signal import correlate, correlation_lags


RATE = 48000
NYQUIST = RATE / 2


def write_wav(path, samples):
    samples = np.clip(samples, -1, 1)
    pcm = np.rint(samples * 32767).astype('<i2')
    with wave.open(str(path), 'wb') as out:
        out.setnchannels(2)
        out.setsampwidth(2)
        out.setframerate(RATE)
        out.writeframes(np.repeat(pcm[:, None], 2, axis=1).tobytes())


def read_wav(path):
    with wave.open(str(path), 'rb') as src:
        if src.getnchannels() != 2 or src.getsampwidth() != 2:
            raise ValueError(f'Expected stereo 16-bit WAV: {path}')
        return (np.frombuffer(src.readframes(src.getnframes()), '<i2')
                .reshape(-1, 2)[:, 0].astype(float) / 32768)


def sweep(seconds):
    t = np.arange(round(RATE * seconds)) / RATE
    f0, f1 = 1.0, NYQUIST - 100.0
    k = np.log(f1 / f0) / seconds
    phase = 2 * np.pi * f0 * (np.exp(k * t) - 1) / k
    return .7 * np.sin(phase)


def noise(seconds, pink=False):
    n = round(RATE * seconds)
    rng = np.random.default_rng(20260919)
    x = rng.normal(size=n)
    if pink:
        spec = np.fft.rfft(x)
        freq = np.fft.rfftfreq(n, 1 / RATE)
        spec[1:] /= np.sqrt(freq[1:])
        x = np.fft.irfft(spec, n=n)
    return .7 * x / max(np.max(np.abs(x)), 1e-12)


def multitone(seconds):
    t = np.arange(round(RATE * seconds)) / RATE
    frequencies = np.unique(np.rint(np.geomspace(1, NYQUIST - 100, 96)))
    out = np.zeros_like(t)
    for i, frequency in enumerate(frequencies):
        out += np.sin(2 * np.pi * frequency * t + i * .713)
    return .7 * out / max(np.max(np.abs(out)), 1e-12)


def impulse(seconds):
    out = np.zeros(round(RATE * seconds))
    out[len(out) // 2] = .8
    return out


def roundtrip(src, out):
    mp3 = out.with_suffix('.mp3')
    wav = out.with_suffix('.decoded.wav')
    subprocess.run(['ffmpeg', '-y', '-hide_banner', '-loglevel', 'error',
                    '-i', str(src), '-ar', str(RATE), '-ac', '2',
                    '-c:a', 'libmp3lame', '-b:a', '320k', '-joint_stereo', '1',
                    str(mp3)], check=True)
    subprocess.run(['ffmpeg', '-y', '-hide_banner', '-loglevel', 'error',
                    '-i', str(mp3), '-ar', str(RATE), '-ac', '2',
                    '-c:a', 'pcm_s16le', str(wav)], check=True)
    return wav


def align(original, damaged):
    n = min(len(original), len(damaged))
    probe = original[:n]
    got = damaged[:n]
    corr = correlate(got, probe, mode='full', method='fft')
    lag = correlation_lags(len(got), len(probe), mode='full')[np.argmax(corr)]
    if lag >= 0:
        return probe[:n-lag], got[lag:n], int(lag)
    return probe[-lag:n], got[:n+lag], int(lag)


def transfer(original, damaged, window=4096, hop=2048):
    original, damaged, lag = align(original, damaged)
    count = min(len(original), len(damaged))
    original, damaged = original[:count], damaged[:count]
    windows = np.hanning(window)
    cross = np.zeros(window // 2 + 1, complex)
    power = np.zeros_like(cross.real)
    output_power = np.zeros_like(power)
    used = 0
    for start in range(0, count - window + 1, hop):
        x = np.fft.rfft(original[start:start+window] * windows)
        y = np.fft.rfft(damaged[start:start+window] * windows)
        cross += y * np.conj(x)
        power += np.abs(x) ** 2
        output_power += np.abs(y) ** 2
        used += 1
    h = cross / np.maximum(power, 1e-12)
    coherence = np.abs(cross) ** 2 / np.maximum(power * output_power, 1e-12)
    frequency = np.fft.rfftfreq(window, 1 / RATE)
    return frequency, h, coherence, lag, used


def impulse_report(original, damaged):
    original, damaged, lag = align(original, damaged)
    peak = int(np.argmax(np.abs(damaged)))
    radius = round(.005 * RATE)
    total = float(np.sum(damaged ** 2))
    local = float(np.sum(damaged[max(0, peak-radius):peak+radius+1] ** 2))
    return lag, peak / RATE, 1 - local / max(total, 1e-12)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, default=Path('scratch/mp3-channel'))
    parser.add_argument('--seconds', type=float, default=8.0)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    probes = {
        'sweep': sweep(args.seconds),
        'white': noise(args.seconds),
        'pink': noise(args.seconds, pink=True),
        'multitone': multitone(args.seconds),
        'impulse': impulse(args.seconds),
    }
    print('probe,frequency_hz,gain_db,phase_deg,coherence')
    for name, signal in probes.items():
        source = args.out / f'{name}.wav'
        write_wav(source, signal)
        decoded = roundtrip(source, args.out / name)
        original, damaged = read_wav(source), read_wav(decoded)
        if name == 'impulse':
            lag, peak, ringing = impulse_report(original, damaged)
            print(f'{name},lag={lag},peak_s={peak:.6f},ringing_outside_5ms={ringing:.6f}')
            continue
        frequency, h, coherence, lag, used = transfer(original, damaged)
        for hz in (0, 1, 10, 30, 100, 375, 750, 1500, 3000, 6000,
                   10000, 12000, 14000, 16000, 18000, 20250, 22000, 23900):
            i = int(np.argmin(abs(frequency - hz)))
            print(f'{name},{frequency[i]:.1f},{20*np.log10(max(abs(h[i]), 1e-12)):.3f},'
                  f'{np.degrees(np.angle(h[i])):.3f},{coherence[i]:.6f}')
        print(f'# {name}: alignment_lag={lag} samples windows={used}', flush=True)


if __name__ == '__main__':
    main()
