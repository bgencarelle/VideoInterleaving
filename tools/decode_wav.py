#!/usr/bin/env python
"""Offline hd-dwt WAV -> PNG decoder.

Decodes a recorded v5 (hd-dwt, WIRE_HD) stereo WAV into per-frame PNGs using
the standard v3 Receiver -- the same acquisition path that live-receive uses
(pulse_only=True). The coder is constructed EXACTLY as modem_screen.build does
so the gains tables and geometry match the transmitter (the header's profile
code 2 is cross-checked against 'hd-dwt').

Usage:
  .venv/bin/python tools/decode_wav.py test.wav --out scratch/decode --frames 30
"""
import argparse
import sys
import wave
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from animation_modem import transport3 as V3
from animation_modem.imaging import display_image, stabilize_chroma
from animation_modem.wavelet import hd_dwt_coder
from animation_modem.audio_common import InputLevel

SCALE = 4  # nice view size for the decoded 80x96 grid (320x384 PNG)


def load_wav(path):
    """Stereo int16 WAV -> (n, 2) float32 in [-1, 1], plus sample rate."""
    with wave.open(str(path), 'rb') as w:
        channels, width, rate, nframes = w.getnchannels(), w.getsampwidth(), \
            w.getframerate(), w.getnframes()
        if channels != 2 or width != 2:
            raise SystemExit(f'Expected a stereo 16-bit WAV, got '
                             f'{channels}ch x {width}B')
        raw = w.readframes(nframes)
    audio = np.frombuffer(raw, '<i2').astype(np.float32) / 32768.0
    return audio.reshape(-1, 2), rate


def build_coder():
    """The one v5 coder (wavelet.hd_dwt_coder), same as every sender.

    The returned `grids` -- the baked 80x96 -- are what frames render at.
    """
    coder = hd_dwt_coder()
    return coder, coder.grids


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('wav', help='path to the recorded stereo WAV')
    ap.add_argument('--out', default='scratch/decode',
                    help='directory for the decoded PNGs')
    ap.add_argument('--frames', type=int, default=60,
                    help='max frames to decode (0 = all)')
    ap.add_argument('--start', type=int, default=0,
                    help='frame to start decoding from')
    ap.add_argument('--scale', type=int, default=SCALE,
                     help='upscale factor for the saved PNG')
    ap.add_argument('--max-carrier-hz', type=float, default=None,
                     help='Ignore decoded carriers above this frequency')
    ap.add_argument('--stabilize-chroma', action='store_true',
                     help='Opt into experimental temporal/spatial chroma stabilization')
    ap.add_argument('--raw', action='store_true',
                     help='Deprecated compatibility option; raw decoding is the default')
    ap.add_argument('-v', '--verbose', action='store_true',
                    help='Print per-channel recovery diagnostics, including failed acquisition')
    args = ap.parse_args()

    audio, rate = load_wav(args.wav)
    coder, grids = build_coder()
    layout = V3.WIRE_HD
    coders = {2: coder}
    rx = V3.Receiver(layout, coder, coders=coders,
                     candidates=[(layout, coder, coders)], input_rate=rate,
                     diagnostics=args.verbose,
                     max_carrier_hz=args.max_carrier_hz)
    level = InputLevel()
    if args.verbose:
        import json
        from utilities.modem_v3_check import record, recovery_record

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    frame_samples = layout.frame             # stereo pairs per frame
    start_at = args.start * frame_samples
    total = len(audio)
    limit = min(total, start_at + (args.frames or (1 << 62)) * frame_samples)

    sent = 0
    decoded = 0
    seen = {}
    files = []
    previous_values = None
    reported = start_at
    for i in range(start_at, limit, 256):
        chunk = audio[i:min(i + 256, limit)]
        results = rx.feed(level.process(chunk))
        if args.verbose and i + len(chunk) - reported >= rate:
            print(json.dumps(recovery_record(rx, level,
                             audio_seconds=round((i+len(chunk))/rate, 3))), flush=True)
            reported = i + len(chunk)
        for result in results:
            if args.verbose:
                print(json.dumps(record(result)), flush=True)
            sent += 1
            seen[result.identity] = seen.get(result.identity, 0) + 1
            if result.values is None:
                continue
            decoded += 1
            if args.stabilize_chroma:
                result.values = stabilize_chroma(
                    result.values, previous_values, result.extra.get('shapes', grids),
                    result.pilot_error, result.coverage)
            previous_values = result.values.copy()
            img = display_image(result, grids, scale=args.scale)
            idx = result.absolute if result.absolute is not None else decoded
            name = out / f'frame_{idx:05d}.png'
            img.save(name)
            files.append((name, result.identity, result.status,
                          result.extra.get('profile'),
                          float(np.std(result.values))))
    for result in rx.flush():
        if args.verbose:
            print(json.dumps(record(result)), flush=True)
        sent += 1
        seen[result.identity] = seen.get(result.identity, 0) + 1
        if result.values is not None:
            decoded += 1
            if args.stabilize_chroma:
                result.values = stabilize_chroma(
                    result.values, previous_values, result.extra.get('shapes', grids),
                    result.pilot_error, result.coverage)
            previous_values = result.values.copy()
            img = display_image(result, grids, scale=args.scale)
            idx = result.absolute if result.absolute is not None else decoded
            name = out / f'frame_{idx:05d}.png'
            img.save(name)
            files.append((name, result.identity, result.status,
                          result.extra.get('profile'),
                          float(np.std(result.values))))

    if args.verbose:
        print(json.dumps(recovery_record(rx, level, final=True)), flush=True)
    print(f'input: {rate} Hz, {total // frame_samples} frames, '
          f'peak {float(np.max(np.abs(audio))):.3f}')
    print(f'wire: {layout.name} frame={layout.frame} capacity={layout.capacity}')
    print(f'coder: {type(coder).__name__} shapes={coder.shapes} '
          f'grids={coder.grids} count={coder.count}')
    print(f'packets: {sent} decoded images: {decoded} identities: {seen}')
    for name, ident, status, prof, std in files[:8]:
        print(f'  {name.name}: {ident}/{status} profile={prof} '
              f'values_std={std:.2f}')
    if files:
        print(f'PNGs written to {out}/')


if __name__ == '__main__':
    main()
