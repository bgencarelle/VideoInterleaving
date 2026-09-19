#!/usr/bin/env python3
"""Controlled V6 DCT/wavelet comparison on one fixed source frame.

This is a synthetic diagnostic, not a tape ranking.  Both transforms receive
the same prepared RGB frame and use the same V6 wire.  Acquisition, identity,
picture recovery, source fidelity, additional channel damage, decode cost, and
wire geometry are reported separately.

    .venv/bin/python tools/bench_v6.py
    .venv/bin/python tools/bench_v6.py --source frame.jpg --crop-left-half
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from animation_modem import transport3 as V3                         # noqa: E402
from animation_modem.imaging import image_values, prepare_image       # noqa: E402
from animation_modem.v6 import coder_for                              # noqa: E402


def synthetic_frame():
    image = Image.new('RGB', (720, 960), (26, 38, 64))
    draw = ImageDraw.Draw(image)
    draw.ellipse((150, 130, 570, 710), fill=(202, 146, 111))
    draw.ellipse((245, 320, 285, 365), fill=(25, 32, 38))
    draw.ellipse((435, 320, 475, 365), fill=(25, 32, 38))
    draw.arc((270, 420, 450, 610), 10, 170, fill=(92, 35, 42), width=18)
    draw.rectangle((0, 760, 720, 960), fill=(42, 112, 76))
    return image


def load_source(path, crop_left_half):
    image = Image.open(path).convert('RGB') if path else synthetic_frame()
    original = image.size
    crop = (0, 0, image.width, image.height)
    if crop_left_half:
        crop = (0, 0, image.width//2, image.height)
        image = image.crop(crop)
    return image, original, crop


def impair(wire, case, lowpass_hz):
    out = wire.copy()
    if case == 'mute-left':
        out[:, 0] = 0
    elif case == 'mute-right':
        out[:, 1] = 0
    elif case == 'lowpass':
        spectrum = np.fft.rfft(out, axis=0)
        frequency = np.fft.rfftfreq(len(out), 1/V3.REFERENCE_RATE)
        spectrum[frequency > lowpass_hz] = 0
        out = np.fft.irfft(spectrum, n=len(out), axis=0).astype(np.float32)
    return out


def plane_rmse(want, got):
    cuts = (96*80, 48*40)
    parts = ((0, cuts[0]), (cuts[0], cuts[0]+cuts[1]),
             (cuts[0]+cuts[1], cuts[0]+2*cuts[1]))
    return {name: float(np.sqrt(np.mean((want[a:b]-got[a:b])**2)))
            for name, (a, b) in zip(('y', 'cb', 'cr'), parts)}


def json_safe(value):
    """Replace non-finite diagnostics so each output line is strict JSON."""
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, (float, np.floating)) and not np.isfinite(value):
        return None
    if isinstance(value, np.generic):
        return value.item()
    return value


def decode_stream(wire, coder, code):
    receiver = V3.Receiver(V3.WIRE_V6, coder, pulse_only=True, fast=True,
                           coders={code: coder},
                           candidates=[(V3.WIRE_V6, coder, {code: coder})],
                           diagnostics=True)
    records = []
    started = time.perf_counter()
    for offset in range(0, len(wire), 257):
        records.extend(receiver.feed(wire[offset:offset+257]))
    records.extend(receiver.flush())
    elapsed_ms = (time.perf_counter()-started)*1000
    return records, receiver.diagnostic_state(), elapsed_ms


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path)
    parser.add_argument('--crop-left-half', action='store_true')
    parser.add_argument('--frames', type=int, default=4)
    parser.add_argument('--lowpass-hz', type=float, default=8000)
    args = parser.parse_args(argv)
    if args.frames < 1:
        parser.error('--frames must be positive')

    source, original_size, crop = load_source(args.source, args.crop_left_half)
    prepared = prepare_image(source, 'auto', 'lanczos')
    provenance = {
        'source': str(args.source) if args.source else 'built-in-fixed-frame',
        'source_size': original_size, 'crop': crop,
        'cropped_size': source.size, 'prepared_size': prepared.size,
        'native_grid': (80, 96), 'display_aspect': source.width/source.height,
        'frames': args.frames,
    }
    print(json.dumps({'run': provenance}, sort_keys=True))

    for code, transform in enumerate(('dct', 'wavelet')):
        coder = coder_for(transform)
        values = image_values(prepared, coder.grids)
        packets, encode_ms = [], []
        for frame in range(args.frames):
            started = time.perf_counter()
            packets.append(V3.encode(values, V3.WIRE_V6, coder, frame+1,
                                     frame+1, args.frames, profile=code))
            encode_ms.append((time.perf_counter()-started)*1000)
        wire = np.concatenate(packets)
        spectrum = np.sum(np.abs(np.fft.rfft(wire, axis=0))**2, axis=1)
        frequency = np.fft.rfftfreq(len(wire), 1/V3.REFERENCE_RATE)
        above = float(spectrum[frequency > 14000].sum()/spectrum.sum())
        clean_values = {}
        clean_reference = None

        for case in ('clean', 'mute-left', 'mute-right', 'lowpass'):
            records, diagnostic, elapsed_ms = decode_stream(
                impair(wire, case, args.lowpass_hz), coder, code)
            pictures = [record for record in records if record.values is not None]
            verified = [record for record in records
                        if record.identity == 'verified_header']
            identified = [record for record in pictures
                          if record.identity == 'verified_header']
            paths = {}
            for record in records:
                path = record.extra.get('acquisition_path', 'unknown')
                paths[path] = paths.get(path, 0) + 1
            if case == 'clean' and pictures:
                clean_values = {record.absolute: record.values.copy()
                                for record in pictures}
                clean_reference = pictures[0].values.copy()
            source_errors = ([plane_rmse(values, record.values)
                              for record in pictures])
            # A verified absolute is the preferred alignment.  If damage loses
            # identity, this benchmark still has independently known alignment:
            # every packet carries the one fixed source frame.
            damage_errors = [plane_rmse(
                clean_values.get(record.absolute, clean_reference), record.values)
                for record in pictures if clean_reference is not None]
            mean = lambda rows, key: (float(np.mean([row[key] for row in rows]))
                                      if rows else None)
            decode_ms = [record.extra.get('decode_ms') for record in records
                         if record.extra.get('decode_ms') is not None]
            report = {
                'transform': transform, 'case': case,
                'acquired': len(records), 'expected': args.frames,
                'acquisition_paths': paths,
                'verified_identity': len(verified),
                'recovered_pictures': len(pictures),
                'identified_pictures': len(identified),
                'unidentified_pictures': len(pictures)-len(identified),
                'source_rmse': {p: mean(source_errors, p)
                                for p in ('y', 'cb', 'cr')},
                'damage_vs_clean_rmse': {p: mean(damage_errors, p)
                                         for p in ('y', 'cb', 'cr')},
                'decode_ms': float(np.mean(decode_ms)) if decode_ms else None,
                'receive_wall_ms': elapsed_ms,
                'encode_ms': float(np.mean(encode_ms)),
                'carrier_band_hz': V3.WIRE_V6.band,
                'whole_wave_ceiling_hz': V3.WIRE_V6.emission_ceiling,
                'energy_above_ceiling': above,
                'fps': V3.WIRE_V6.fps,
                'frame_latency_ms': 1000/V3.WIRE_V6.fps,
                'wire_values': coder.count,
                'foundation_copies': len(coder.copy_of),
                'receiver': diagnostic,
            }
            print(json.dumps(json_safe(report), sort_keys=True, allow_nan=False))


if __name__ == '__main__':
    main()
