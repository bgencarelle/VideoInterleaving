#!/usr/bin/env python3
"""Compare mono V7 layouts on deterministic moving scenes and wire cases.

The original review's ``tmp/mono/video.py`` is not present in this checkout,
so this runner recreates the same scene classes (slow/fast pans, six-packet
cuts, moving blob) from the frozen V7 reference image and a generated graphic.
Impairment cases are synthetic regression inputs, not tape predictions.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from scipy.signal import resample_poly

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT/'test_modem_v7'))

from animation_modem import v7                                           # noqa: E402
from animation_modem.imaging import values_image                         # noqa: E402
from common import TARGET, ssimulacra2                                    # noqa: E402
from mono_video import (MonoColourFoldWire, MonoFreshFoldWire)             # noqa: E402
from tools import v7_live                                                 # noqa: E402
from tools.v7_torture_matrix import CASES, RATE, impair                   # noqa: E402
from tools.v7_wire_profile import WireProfile                             # noqa: E402


PACKETS = 24
DISPLAY_SIZE = (400, 480)  # 5:6, the V7 source/display aspect.
SCORE_RANGE = range(7, 23)  # packets 8..23, matching the review's steady window.
SCENE_NAMES = ('slow-pan', 'fast-pan', 'cut-every-6', 'moving-blob',
               'colour-chart')
CASE_NAMES = ('clean-96k', 'type-ii', 'fast-flutter')
CHART_COLOURS = ((220, 40, 40), (40, 200, 40), (40, 60, 220), (230, 210, 40),
                 (40, 200, 210), (210, 40, 200), (240, 140, 30), (120, 70, 40),
                 (250, 180, 170))


def _base_frames():
    with Image.open(v7.REFERENCE_FIXTURE) as image:
        source = image.convert('RGB')
    return source.resize((160, 192), Image.Resampling.LANCZOS)


def _pan_frames(step):
    canvas = _base_frames()
    frames = []
    maximum = canvas.width-80
    for index in range(PACKETS):
        left = min(maximum, int(round(index*step)))
        frames.append(canvas.crop((left, 48, left+80, 144)))
    return frames


def _graphic_frame():
    image = Image.new('RGB', (80, 96))
    pixels = np.empty((96, 80, 3), dtype=np.uint8)
    y, x = np.mgrid[:96, :80]
    pixels[..., 0] = 72 + (x*2+y//3) % 112
    pixels[..., 1] = 88 + (y*2+x//4) % 120
    pixels[..., 2] = 96 + (x+y) % 112
    image = Image.fromarray(pixels, 'RGB')
    draw = ImageDraw.Draw(image)
    draw.rectangle((9, 12, 70, 82), outline=(236, 208, 81), width=3)
    draw.line((10, 80, 70, 14), fill=(48, 67, 204), width=4)
    draw.ellipse((25, 31, 49, 55), fill=(222, 71, 117))
    return image


def _cut_frames():
    first = _base_frames().crop((40, 48, 120, 144))
    second = _graphic_frame()
    return [first.copy() if (index//6) % 2 == 0 else second.copy()
            for index in range(PACKETS)]


def _blob_frames():
    frames = []
    for index in range(PACKETS):
        image = Image.new('RGB', (80, 96))
        pixels = np.empty((96, 80, 3), dtype=np.uint8)
        y, x = np.mgrid[:96, :80]
        pixels[..., 0] = 88 + y//3
        pixels[..., 1] = 94 + x//2
        pixels[..., 2] = 108 + (x+y)//5
        image = Image.fromarray(pixels, 'RGB')
        draw = ImageDraw.Draw(image)
        cx = 12 + (index*3) % 58
        cy = 26 + int(round(18*np.sin(index*2*np.pi/12)))
        draw.ellipse((cx-10, cy-10, cx+10, cy+10),
                     fill=(224, 74, 52), outline=(255, 222, 137), width=2)
        draw.rectangle((8, 72, 71, 86), outline=(47, 75, 119), width=2)
        frames.append(image)
    return frames


def _colour_chart_frames():
    """Saturated patches drifting 0-2 px: the colour case the face scenes miss."""
    frames = []
    for index in range(PACKETS):
        image = Image.new('RGB', (400, 480), (128, 128, 128))
        draw = ImageDraw.Draw(image)
        for i, colour in enumerate(CHART_COLOURS):
            x, y = (i % 5)*80 + index % 3, (i//5)*160 + 40
            draw.rectangle((x+4, y, x+76, y+120), fill=colour)
        frames.append(image.resize((80, 96), Image.Resampling.LANCZOS))
    return frames


def scenes():
    return {
        'slow-pan': _pan_frames(1.0),
        'fast-pan': _pan_frames(3.0),
        'cut-every-6': _cut_frames(),
        'moving-blob': _blob_frames(),
        'colour-chart': _colour_chart_frames(),
    }


def _render(values):
    image = values_image(values, v7.V7_GRIDS)
    return image.resize(DISPLAY_SIZE, Image.Resampling.LANCZOS)


def _reference(image):
    return image.resize(DISPLAY_SIZE, Image.Resampling.LANCZOS)


def _decode_rows(profile, audio, model, source_frames, case_name):
    if profile == 'stereo-fold500-both':
        wire = WireProfile('default')
        decode_started = time.perf_counter()
        results, info = wire.decode(
            model, audio, sample_rate=RATE,
            state=v7.PulseState(tail_memory=True))
        value_fn = lambda result: wire.values(model, result)
    elif profile == 'stereo-fold500-mono-sum':
        wire = WireProfile('default')
        mono = audio.sum(axis=1)/np.sqrt(2.0)
        decode_started = time.perf_counter()
        results, info = wire.decode(
            model, mono, sample_rate=RATE,
            state=v7.PulseState(tail_memory=True))
        value_fn = lambda result: wire.values(model, result)
    elif profile == 'mono-colour-500':
        wire = MonoColourFoldWire(model, side='left')
        mono_model = wire.model_for(model)
        decode_started = time.perf_counter()
        with wire.receiving():
            results, info = v7.decode_pulse_stream(
                mono_model, audio[:, 0], sample_rate=RATE,
                pilot_timing='tone-seeded',
                state=v7.PulseState(tail_memory=False))
        value_fn = lambda result: wire.values(mono_model, result)
    else:
        wire = MonoFreshFoldWire(model, side='left')
        mono_model = wire.model_for(model)
        decode_started = time.perf_counter()
        with wire.receiving():
            results, info = v7.decode_pulse_stream(
                mono_model, audio[:, 0], sample_rate=RATE,
                pilot_timing='tone-seeded',
                state=v7.PulseState(tail_memory=False))
        value_fn = lambda result: wire.values(mono_model, result)

    decode_elapsed_ms = (time.perf_counter()-decode_started)*1000
    by_source = {}
    for ordinal, result in enumerate(results):
        source_index = result.diag.get('source_index')
        if source_index is None:
            source_index = ordinal
        by_source[int(source_index)] = result

    shown = None
    scores = []
    received = displayable = held = missing = unavailable = visible = 0
    for index, source in enumerate(source_frames):
        result = by_source.get(index)
        if result is None:
            missing += 1
            if shown is None:
                unavailable += 1
            else:
                held += 1
        else:
            current_available = (result.status in ('received', 'verified') or
                                 bool(result.diag.get('displayable')))
            received += result.status in ('received', 'verified')
            if current_available:
                displayable += 1
                shown = _render(value_fn(result))
            else:
                if shown is None:
                    unavailable += 1
                else:
                    held += 1
        if shown is not None:
            visible += 1
        if index in SCORE_RANGE and shown is not None:
            scores.append(ssimulacra2(_reference(source), shown))

    return {
        'scene': None,
        'channel': case_name,
        'profile': profile,
        'scores_packets_8_23': [round(value, 3) for value in scores],
        'mean_ssimulacra2': round(float(np.mean(scores)), 3) if scores else None,
        'received': int(received),
        'current_frame_displayable': int(displayable),
        'picture_visible': int(visible),
        'held_last_picture': int(held),
        'no_picture_yet': int(unavailable),
        'missing_results': int(missing),
        'decoded_results': len(results),
        'eof_validated': int(info.get('eof_markers_validated') or 0),
        'decode_elapsed_ms': round(decode_elapsed_ms, 3),
        'decode_ms_per_input_packet': round(
            decode_elapsed_ms/max(len(source_frames), 1), 3),
    }


def run(out, selected_scenes=None, selected_channels=None):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    model = v7.load_model(TARGET, 'box')
    v7.warmup_equalizer(model)
    v7.PULSE.warmup_pulse_kernels()
    from tone_code import warmup_coded_decoder
    warmup_coded_decoder(model)
    scene_frames = scenes()
    selected_scenes = tuple(selected_scenes or scene_frames)
    selected_channels = tuple(selected_channels or CASE_NAMES)
    value_frames = {
        name: [v7_live._values(model, frame, 'box', brightness=1.0)[0]
               for frame in frames]
        for name, frames in scene_frames.items()
    }
    profiles = (
        'stereo-fold500-both',
        'stereo-fold500-mono-sum',
        'mono-fresh-500-fold',
        'mono-colour-500',
    )
    cases = {case.name: case for case in CASES}
    unknown_scenes = set(selected_scenes)-set(scene_frames)
    unknown_channels = set(selected_channels)-set(cases)
    if unknown_scenes or unknown_channels:
        raise ValueError(f'unknown scenes={sorted(unknown_scenes)} '
                         f'channels={sorted(unknown_channels)}')
    rows = []
    started = time.monotonic()

    for scene in selected_scenes:
        frames = scene_frames[scene]
        values = value_frames[scene]
        # Encode each layout once; all capture paths receive the same source
        # indices so scores are always against that packet's own frame.
        encode_times = {}
        before = time.perf_counter()
        stereo_wire = WireProfile('default').encode(
            model, values, source_indices=list(range(PACKETS)))
        encode_times['stereo-fold500-both'] = (
            time.perf_counter()-before)*1000
        encode_times['stereo-fold500-mono-sum'] = encode_times[
            'stereo-fold500-both']
        before = time.perf_counter()
        fresh_wire = MonoFreshFoldWire(model, side='left').encode(
            model, values, source_indices=list(range(PACKETS)))
        encode_times['mono-fresh-500-fold'] = (
            time.perf_counter()-before)*1000
        before = time.perf_counter()
        colour_wire = MonoColourFoldWire(model, side='left').encode(
            model, values, source_indices=list(range(PACKETS)))
        encode_times['mono-colour-500'] = (
            time.perf_counter()-before)*1000
        audio_by_profile = {
            'stereo-fold500-both': stereo_wire,
            'stereo-fold500-mono-sum': stereo_wire,
            'mono-fresh-500-fold': fresh_wire,
            'mono-colour-500': colour_wire,
        }

        for case_name in selected_channels:
            for profile in profiles:
                audio = audio_by_profile[profile]
                captured = resample_poly(audio, 2, 1, axis=0).astype(np.float32)
                captured = impair(captured, cases[case_name], seed=2026)
                row = _decode_rows(profile, captured, model, frames, case_name)
                row['scene'] = scene
                row['encode_elapsed_ms_per_24_packets'] = round(
                    encode_times[profile], 3)
                row['encode_ms_per_packet'] = round(
                    encode_times[profile]/PACKETS, 3)
                rows.append(row)
                print(json.dumps(row), flush=True)

    report = {
        'note': ('Synthetic generated motion scenes and synthetic channel '
                 'regressions; not real tape evidence. The prior tmp/mono/video.py '
                 'was unavailable, so these are reproducible scene-class matches, '
                 'not byte-identical fixtures.'),
        'sample_rate': RATE,
        'packets_per_scene': PACKETS,
        'scored_packets': [8, 23],
        'mono_video_side': 'left',
        'profiles': list(profiles),
        'channels': list(selected_channels),
        'scenes': list(selected_scenes),
        'elapsed_seconds': round(time.monotonic()-started, 2),
        'rows': rows,
    }
    output = out/'results.json'
    output.write_text(json.dumps(report, indent=2)+'\n')
    print(f'Wrote {output}; {len(rows)} profile/scene/channel comparisons')
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=Path('tmp/v7-mono-video'))
    parser.add_argument('--scene', action='append', choices=SCENE_NAMES,
                        help='limit to one or more moving scenes')
    parser.add_argument('--channel', action='append', choices=CASE_NAMES,
                        help='limit to one or more synthetic wire stresses')
    args = parser.parse_args(argv)
    run(args.out, args.scene, args.channel)


if __name__ == '__main__':
    main()
