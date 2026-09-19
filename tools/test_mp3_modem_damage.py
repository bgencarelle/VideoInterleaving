#!/usr/bin/env python3
"""Measure MP3 damage on actual randomized hd-dwt modem frames."""
import argparse
import subprocess
import sys
import wave
from pathlib import Path

import numpy as np
from scipy.signal import correlate, correlation_lags

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from animation_modem import engines
from animation_modem import transport3 as V3
from animation_modem.core import SYNC_LEN, SYMBOL, profile_code


RATE = 48000


def write_wav(path, audio):
    pcm = np.clip(np.rint(audio * 32767), -32768, 32767).astype('<i2')
    with wave.open(str(path), 'wb') as out:
        out.setnchannels(2); out.setsampwidth(2); out.setframerate(RATE)
        out.writeframes(pcm.tobytes())


def read_wav(path):
    with wave.open(str(path), 'rb') as src:
        raw = np.frombuffer(src.readframes(src.getnframes()), '<i2')
        return raw.reshape(-1, 2).astype(float) / 32768


def align(original, damaged):
    n = min(len(original), len(damaged))
    x, y = original[:n, 0], damaged[:n, 0]
    corr = correlate(y, x, mode='full', method='fft')
    lag = int(correlation_lags(len(y), len(x), mode='full')[np.argmax(corr)])
    if lag >= 0:
        return original[:n-lag], damaged[lag:n], lag
    return original[-lag:n], damaged[:n+lag], lag


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, default=Path('scratch/mp3-modem-damage'))
    parser.add_argument('--frames', type=int, default=60)
    parser.add_argument('--profile', choices=('hd-dwt', 'tape-80x60'),
                        default='hd-dwt')
    parser.add_argument('--wire', choices=('wide', 'tape'), default='wide')
    parser.add_argument('--bitrate', default='320k')
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    if args.wire == 'tape':
        layout = V3.WIRE_TAPE if args.profile == 'hd-dwt' else V3.WIRE_TAPE_25
    else:
        if args.profile != 'hd-dwt':
            parser.error('tape-80x60 requires --wire tape')
        layout = V3.WIRE_HD
    coder, _ = engines.coder_for(args.profile, layout)
    rng = np.random.default_rng(20260919)
    packets = []
    for n in range(args.frames):
        values = rng.uniform(-.9, .9, coder.source_count)
        packets.append(V3.encode(values, layout, coder, n + 1, n + 1,
                                  args.frames, profile=profile_code(args.profile)))
    original = np.concatenate(packets).astype(float)
    source = args.out / 'modem-random.wav'
    write_wav(source, original)
    stem = f"{args.profile}-{args.wire}-{args.bitrate}"
    mp3 = args.out / f'{stem}.mp3'
    decoded = args.out / f'{stem}-decoded.wav'
    subprocess.run(['ffmpeg', '-y', '-hide_banner', '-loglevel', 'error',
                    '-i', str(source), '-ar', str(RATE), '-ac', '2',
                    '-c:a', 'libmp3lame', '-b:a', args.bitrate, '-joint_stereo', '1',
                    str(mp3)], check=True)
    subprocess.run(['ffmpeg', '-y', '-hide_banner', '-loglevel', 'error',
                    '-i', str(mp3), '-ar', str(RATE), '-ac', '2',
                    '-c:a', 'pcm_s16le', str(decoded)], check=True)
    recovered, damaged, lag = align(original, read_wav(decoded))
    usable = min(len(recovered), len(damaged)) // layout.frame
    carrier_sum = np.zeros(layout.top_bin, complex)
    carrier_sq = np.zeros(layout.top_bin)
    carrier_n = np.zeros(layout.top_bin)
    for n in range(usable):
        a = recovered[n*layout.frame:n*layout.frame+layout.packet]
        b = damaged[n*layout.frame:n*layout.frame+layout.packet]
        a = a[SYNC_LEN:].reshape(layout.symbols, SYMBOL, 2)[:, 12:140]
        b = b[SYNC_LEN:].reshape(layout.symbols, SYMBOL, 2)[:, 12:140]
        xa = np.fft.rfft(a, n=128, axis=1)
        xb = np.fft.rfft(b, n=128, axis=1)
        for channel in range(2):
            x, y = xa[:, 1:layout.top_bin+1, channel], xb[:, 1:layout.top_bin+1, channel]
            good = np.abs(x) > .01
            ratio = np.divide(y, x, out=np.zeros_like(y), where=good)
            carrier_sum += np.sum(np.where(good, ratio, 0), axis=0)
            carrier_sq += np.sum(np.where(good, np.abs(ratio-1)**2, 0), axis=0)
            carrier_n += np.sum(good, axis=0)
    mean = carrier_sum / np.maximum(carrier_n, 1)
    evm = np.sqrt(carrier_sq / np.maximum(carrier_n, 1))
    print(f'profile={args.profile} wire={args.wire} bitrate={args.bitrate}')
    print(f'frames={usable} alignment_lag={lag} samples')
    print('bin,frequency_hz,gain_db,phase_deg,evm,samples')
    for i, value in enumerate(mean):
        print(f'{i+1},{(i+1)*RATE/128:.1f},'
              f'{20*np.log10(max(abs(value), 1e-12)):.3f},'
              f'{np.degrees(np.angle(value)):.3f},{evm[i]:.5f},{int(carrier_n[i])}')


if __name__ == '__main__':
    main()
