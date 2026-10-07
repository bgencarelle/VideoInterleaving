#!/usr/bin/env python3
"""Compare native-source DCT preparation through the actual V7 modem.

This synthetic clean/Type-II experiment keeps input images at their original
pixel dimensions, sends stereo packets through the application encoder and
mono packets through their experimental wire profiles, and saves source-grid,
ideal-fold, and decoded pictures.
It is a benchmark utility; it does not change the normal sender default.
"""
import argparse
from dataclasses import asdict
import json
import statistics
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from scipy.fft import idctn

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT/'test_modem_v7'))

from animation_modem import v7
from animation_modem.v7_core import adapt_packet_for_output
from animation_modem.v7_source_dct import (
    FoldBlockDCTProjector, _frequency_weights, source_dct_values,
    source_fold_dct_coefficients)
from common import TARGET, ssimulacra2
from mono_video import MonoColourFoldWire, MonoFreshFoldWire
from tools.measure_plane_survival import plane_metrics_arrays
from tools.v7_torture_matrix import CASES, RATE, impair
from tools.v7_wire_profile import WireProfile


PACKETS = 12
PREP_TIMING_REPEATS = 7
DISPLAY_SIZE = (1080, 900)
VIEWING_DISTANCE_MM = 600.0
DISPLAY_DPI = 96.0
PROFILE_NAMES = ('stereo-fold-500', 'mono-fold-500', 'mono-colour-500')
CHANNEL_NAMES = ('clean-96k', 'type-ii')
DEFAULT_CHANNELS = ('clean-96k',)
DEFAULT_SOURCE_DIR = ROOT/'images_sbs/face/14_D_BG_First_Upscale_960'


_VARIANT_CATALOG = (
    {'name': 'resize-first', 'kind': 'baseline',
     'description': 'current box resize to 80x96, then current V7 image conversion'},
    {'name': 'direct-retention', 'kind': 'dct',
     'description': 'direct source DCT; retain the coder-grid frequency rectangle'},
    {'name': 'direct-unclipped', 'kind': 'dct', 'options': {'clip_values': False},
     'description': 'direct source DCT without [-1,1] coder-grid clipping'},
    {'name': 'area-box-resample', 'kind': 'dct',
     'options': {'aggregation': 'area-box'},
     'description': 'exact pixel-area box resampling followed by the target-grid DCT'},
    {'name': 'fold-native-projection', 'kind': 'fold-native',
     'description': 'project Fold-500 kept and guest DCT support regions; fold exact slots'},
    {'name': 'fold-block-4x', 'kind': 'fold-block',
     'options': {'block_size': 4},
     'description': 'block-average RGB; project with integrated native DCT basis'},
    {'name': 'fold-block-5x', 'kind': 'fold-block',
     'options': {'block_size': 5},
     'description': 'block-average RGB; project with integrated native DCT basis'},
    {'name': 'fold-block-6x', 'kind': 'fold-block',
     'options': {'block_size': 6},
     'description': 'block-average RGB; project with integrated native DCT basis'},
    {'name': 'fold-block-8x', 'kind': 'fold-block',
     'options': {'block_size': 8},
     'description': 'block-average RGB; project with integrated native DCT basis'},
    {'name': 'fold-block-10x', 'kind': 'fold-block',
     'options': {'block_size': 10},
     'description': 'block-average RGB; project with integrated native DCT basis'},
    {'name': 'fold-block-12x', 'kind': 'fold-block',
     'options': {'block_size': 12},
     'description': 'block-average RGB; project with integrated native DCT basis'},
    {'name': 'weighted-tent', 'kind': 'dct',
     'options': {'aggregation': 'weighted-tent'},
     'description': 'separable tent gain on matching whole-frame DCT modes'},
    {'name': 'weighted-tent-unclipped', 'kind': 'dct',
     'options': {'aggregation': 'weighted-tent', 'clip_values': False},
     'description': 'separable tent DCT gain, unclipped diagnostic'},
    {'name': 'weighted-cosine', 'kind': 'dct',
     'options': {'aggregation': 'weighted-cosine'},
     'description': 'separable cosine gain on matching whole-frame DCT modes'},
    {'name': 'weighted-cosine-unclipped', 'kind': 'dct',
     'options': {'aggregation': 'weighted-cosine', 'clip_values': False},
     'description': 'separable cosine DCT gain, unclipped diagnostic'},
    {'name': 'weighted-gaussian', 'kind': 'dct',
     'options': {'aggregation': 'weighted-gaussian'},
     'description': 'separable Gaussian gain on matching whole-frame DCT modes'},
    {'name': 'weighted-gaussian-unclipped', 'kind': 'dct',
     'options': {'aggregation': 'weighted-gaussian', 'clip_values': False},
     'description': 'separable Gaussian DCT gain, unclipped diagnostic'},
    {'name': 'taper-025', 'kind': 'dct',
     'options': {'sharpen': 'taper', 'sharpen_strength': .25},
     'description': 'sent-corner luma taper, strength 0.25'},
    {'name': 'taper-050', 'kind': 'dct',
     'options': {'sharpen': 'taper', 'sharpen_strength': .50},
     'description': 'sent-corner luma taper, strength 0.50'},
    {'name': 'usm-025', 'kind': 'dct',
     'options': {'sharpen': 'usm', 'sharpen_strength': .25},
     'description': 'full-resolution luma USM, strength 0.25'},
    {'name': 'usm-050', 'kind': 'dct',
     'options': {'sharpen': 'usm', 'sharpen_strength': .50},
     'description': 'full-resolution luma USM, strength 0.50'},
    {'name': 'clarity-015', 'kind': 'dct', 'options': {'clarity': .15},
     'description': 'full-resolution luma clarity, strength 0.15'},
    {'name': 'clarity-030', 'kind': 'dct', 'options': {'clarity': .30},
     'description': 'full-resolution luma clarity, strength 0.30'},
    {'name': 'chroma-110', 'kind': 'dct', 'options': {'chroma_gain': 1.10},
     'description': 'chroma gain 1.10 around Pillow neutral'},
    {'name': 'chroma-120', 'kind': 'dct', 'options': {'chroma_gain': 1.20},
     'description': 'chroma gain 1.20 around Pillow neutral'},
    {'name': 'band-mid-luma', 'kind': 'dct',
     'options': {'band_profile': 'mid-luma'},
     'description': 'fixed mid-frequency luma emphasis'},
    {'name': 'band-perceptual-color', 'kind': 'dct',
     'options': {'band_profile': 'perceptual-color'},
     'description': 'viewing-geometry CSF weighting, weaker chroma high bands'},
)
_EXPLICIT_ONLY_VARIANTS = frozenset({
    'area-box-resample', 'fold-native-projection', 'fold-block-4x',
    'fold-block-5x', 'fold-block-6x', 'fold-block-8x', 'fold-block-10x',
    'fold-block-12x',
})
VARIANT_BY_NAME = {variant['name']: variant for variant in _VARIANT_CATALOG}
# Keep specialized high-cost probes available only by name.
VARIANTS = tuple(variant for variant in _VARIANT_CATALOG
                 if variant['name'] not in _EXPLICIT_ONLY_VARIANTS)


def _select_variants(selected):
    if not selected:
        return list(VARIANTS)
    unknown = set(selected)-set(VARIANT_BY_NAME)
    if unknown:
        raise ValueError(f'unknown variant(s): {sorted(unknown)}')
    return [VARIANT_BY_NAME[name] for name in selected]


def _safe_name(text):
    return ''.join(char if char.isalnum() or char in '-_' else '_'
                   for char in str(text))


def _source_from_path(path):
    with Image.open(path) as opened:
        image = opened.convert('RGB')
    # images_sbs stores two 720x960 views side-by-side. Select the left eye
    # without resizing either eye or altering its original aspect ratio.
    if image.width == 1.5*image.height:
        image = image.crop((0, 0, image.width//2, image.height))
    return Path(path).stem, image, str(Path(path).resolve())


def _gradient_scene():
    height, width = 960, 720
    y, x = np.mgrid[:height, :width]
    xf = x/(width-1)
    yf = y/(height-1)
    rgb = np.stack((.10+.78*xf, .12+.72*yf,
                    .16+.62*(.6*xf+.4*yf)), axis=-1)
    # A low-amplitude smooth ramp makes output banding visible against the
    # broad gradient without injecting random texture.
    rgb += (.012*np.sin(2*np.pi*yf*3))[..., None]
    return np.asarray(rgb, dtype=np.float64)


def _source_size(source):
    if isinstance(source, Image.Image):
        return source.size
    return int(source.shape[1]), int(source.shape[0])


def _source_rgb_image(source):
    if isinstance(source, Image.Image):
        return source.convert('RGB')
    rgb = np.asarray(source, dtype=np.float64)
    return Image.fromarray(
        np.uint8(np.rint(np.clip(rgb, 0, 1)*255)), 'RGB')


def _chart_scene():
    image = Image.new('RGB', (720, 960), (24, 28, 36))
    draw = ImageDraw.Draw(image)
    colors = ((232, 48, 42), (32, 210, 76), (40, 72, 235),
              (246, 220, 36), (32, 210, 220), (220, 42, 206),
              (244, 142, 32), (145, 88, 48), (245, 183, 174),
              (216, 216, 216), (72, 72, 72), (245, 245, 245))
    for index, color in enumerate(colors):
        col, row = index % 4, index//4
        x0, y0 = 24+col*168, 56+row*174
        draw.rectangle((x0, y0, x0+143, y0+126), fill=color,
                       outline=(250, 250, 250), width=2)
    draw.text((28, 624), 'V7 DCT  TEXT  1234567890', fill=(250, 250, 250),
              stroke_width=0)
    for x in range(28, 692, 8):
        draw.line((x, 672, x, 740), fill=(220, 220, 220) if (x//8) % 2 else
                  (70, 70, 70), width=2)
    for y in range(768, 916, 12):
        draw.line((28, y, 692, y), fill=(105, 112, 124), width=1)
    return image


def load_sources(source_paths=(), source_dir=DEFAULT_SOURCE_DIR,
                 max_sources=2, include_synthetic=True):
    loaded = []
    if source_paths:
        for path in source_paths:
            loaded.append(_source_from_path(path))
    else:
        paths = sorted(path for path in Path(source_dir).iterdir()
                       if path.suffix.lower() in ('.jpg', '.jpeg', '.png'))
        if not paths:
            raise ValueError(f'no image files found in {source_dir}')
        for path in paths[:max_sources]:
            loaded.append(_source_from_path(path))
    if include_synthetic:
        loaded.extend((('smooth-gradient-720x960', _gradient_scene(), None),
                       ('colour-chart-text-720x960', _chart_scene(), None)))
    for name, image, _origin in loaded:
        width, height = _source_size(image)
        if width < 96 or height < 96:
            raise ValueError(
                f'{name} is too small for the V7 coder grid: {(width, height)}')
    return loaded


def _display(image, source_aspect):
    return _fit_canvas(image, DISPLAY_SIZE, source_aspect)


def _fit_canvas(image, size, source_aspect, background='#777777'):
    canvas_width, canvas_height = size
    canvas = Image.new('RGB', size, background)
    viewport_width = min(canvas_width, int(round(canvas_height*source_aspect)))
    viewport_height = min(canvas_height, int(round(canvas_width/source_aspect)))
    viewport = image.convert('RGB').resize(
        (viewport_width, viewport_height), Image.Resampling.LANCZOS)
    canvas.paste(viewport, ((canvas_width-viewport_width)//2,
                            (canvas_height-viewport_height)//2))
    return canvas


def _viewport_size(source_aspect):
    viewport_width = min(DISPLAY_SIZE[0],
                         int(round(DISPLAY_SIZE[1]*source_aspect)))
    viewport_height = min(DISPLAY_SIZE[1],
                          int(round(DISPLAY_SIZE[0]/source_aspect)))
    return viewport_width, viewport_height


def _crop_viewport(image, source_aspect):
    """Crop the exact receiver viewport from a fixed-size comparison canvas."""
    image = image.convert('RGB')
    if image.size != DISPLAY_SIZE:
        raise ValueError(
            f'comparison canvas must be {DISPLAY_SIZE}, got {image.size}')
    width, height = _viewport_size(source_aspect)
    left = (DISPLAY_SIZE[0]-width)//2
    top = (DISPLAY_SIZE[1]-height)//2
    return image.crop((left, top, left+width, top+height))


def _receiver_image(values, shapes, source_aspect):
    """Render float YCbCr values using the viewer's DCT viewport reconstruction."""
    from tools.v7_gl_viewer import dct_reconstruct_planes, float_planes

    planes = dct_reconstruct_planes(
        float_planes(values, shapes), 'viewport', _viewport_size(source_aspect))
    target_height, target_width = planes[0].shape
    yy = np.minimum(((np.arange(target_height)+.5)*
                     planes[1].shape[0]/target_height).astype(int),
                    planes[1].shape[0]-1)
    xx = np.minimum(((np.arange(target_width)+.5)*
                     planes[1].shape[1]/target_width).astype(int),
                    planes[1].shape[1]-1)
    chroma_b = planes[1][yy[:, None], xx[None, :]]
    yy_r = np.minimum(((np.arange(target_height)+.5)*
                       planes[2].shape[0]/target_height).astype(int),
                      planes[2].shape[0]-1)
    xx_r = np.minimum(((np.arange(target_width)+.5)*
                       planes[2].shape[1]/target_width).astype(int),
                      planes[2].shape[1]-1)
    chroma_r = planes[2][yy_r[:, None], xx_r[None, :]]
    y = (planes[0]+1.0)*.5
    cb = chroma_b*.5-.5/255.0
    cr = chroma_r*.5-.5/255.0
    rgb = np.stack((y+1.402*cr,
                    y-.344136*cb-.714136*cr,
                    y+1.772*cb), axis=-1)
    pixels = np.uint8(np.rint(np.clip(rgb, 0.0, 1.0)*255.0))
    viewport = Image.fromarray(pixels, 'RGB')
    canvas = Image.new('RGB', DISPLAY_SIZE, '#777777')
    canvas.paste(viewport, ((DISPLAY_SIZE[0]-viewport.width)//2,
                            (DISPLAY_SIZE[1]-viewport.height)//2))
    return canvas


def _score(reference, picture):
    planes = plane_metrics_arrays(reference, picture)
    return {
        'ssimulacra2': float(ssimulacra2(reference, picture)),
        'psnr_y_db': float(planes[0][0]),
        'ssim_y': float(planes[0][1]),
        'mae_y': float(planes[0][2]),
        'mae_cb': float(planes[1][2]),
        'mae_cr': float(planes[2][2]),
    }


def _band_profile_archive(grids, aspect):
    archive = {}
    for profile in ('mid-luma', 'perceptual-color'):
        planes = []
        for plane_index, grid in enumerate(grids):
            gains = _frequency_weights(
                grid, plane_index, profile, DISPLAY_SIZE,
                VIEWING_DISTANCE_MM, DISPLAY_DPI,
                viewport_aspect=aspect)
            planes.append({
                'shape': list(grid),
                'gains': gains.tolist(),
                'min_gain': float(gains.min()),
                'max_gain': float(gains.max()),
                'dc_gain': float(gains[0, 0]),
            })
        archive[profile] = planes
    return archive


def _profile_wires(model):
    return {
        'stereo-fold-500': WireProfile('default'),
        'mono-fold-500': MonoFreshFoldWire(model, side='left'),
        'mono-colour-500': MonoColourFoldWire(model, side='left'),
    }


def _profile_table_identity(profile_name, wire, model):
    if profile_name == 'stereo-fold-500':
        codec = wire._fold
    else:
        codec = wire._codec(wire.model_for(model))
    return str(codec.identity)


def _ideal_fold_values(profile_name, wire, model, values):
    if profile_name == 'stereo-fold-500':
        codec = wire._fold.codec(model)
    else:
        mono_model = wire.model_for(model)
        codec = wire._codec(mono_model)
    encoded_coefficients = codec.encode_coefficients(values)
    full = codec.decode(
        encoded_coefficients, encoded_coefficients-model.mu,
        np.ones_like(encoded_coefficients), fallback=False,
        metadata_confirmed=True)
    return codec.grid.inverse(full), codec


def _fold_guest_stats(codec, values):
    full = codec.grid.forward(values)
    return _fold_guest_stats_from_coefficients(codec, full)


def _fold_guest_stats_from_coefficients(codec, full):
    normalized = np.asarray(full[codec.guests], dtype=np.float64)/codec.sd_guest
    excess = np.maximum(np.abs(normalized)-2.5, 0.0)
    return {
        'fold_guest_clip_fraction': float(np.count_nonzero(excess)/len(excess)),
        'fold_guest_max_clip_excess_sigma': float(excess.max(initial=0.0)),
        'fold_guest_rms_sigma': float(np.sqrt(np.mean(normalized*normalized))),
    }


def _grid_values_from_dct(coefficients, grids):
    offsets = np.cumsum([0] + [rows*cols for rows, cols in grids])
    return np.concatenate([
        idctn(coefficients[start:end].reshape(shape), norm='ortho').ravel()
        for start, end, shape in zip(offsets, offsets[1:], grids)])


def _ideal_fold_projected_values(production_fold, wire, model, coefficients):
    folded = production_fold.encode_dct_coefficients(coefficients)
    codec = wire._fold.codec(model)
    decoded = codec.decode(
        folded, folded-model.mu, np.ones_like(folded), fallback=False,
        metadata_confirmed=True)
    return codec.grid.inverse(decoded), codec, folded


def _encode_fold_projected(production_fold, model, coefficients,
                           aspect_code, packet_count=PACKETS):
    from modem_v7_display import encode_folded_coefficients_packet

    loop = v7.LoopInfo(packet_count, v7.LOOP_NO_CLOCK, pingpong=True)
    started = time.perf_counter()
    packets = []
    for index in range(packet_count):
        folded = production_fold.encode_dct_coefficients(coefficients)
        packets.append(encode_folded_coefficients_packet(
            model, folded, index+1, index, aspect_code,
            loop=loop, direction=1))
    return np.concatenate(packets), (time.perf_counter()-started)*1000


def _production_output_rate(audio48, rate=RATE, speed=1.0):
    """Adapt each packet exactly as PacketOutput does before device routing."""
    audio48 = np.asarray(audio48, dtype=np.float32)
    if audio48.ndim != 2 or audio48.shape[1] != 2 or \
            len(audio48) % v7.PULSE_FRAME:
        raise ValueError('production output adaptation needs complete stereo packets')
    packets = audio48.reshape((-1, v7.PULSE_FRAME, 2))
    return np.concatenate([
        adapt_packet_for_output(packet, rate, speed) for packet in packets])


def _encode(wire, profile_name, model, values, aspect_code,
            packet_count=PACKETS):
    codes = [aspect_code]*packet_count
    indices = list(range(packet_count))
    loop = v7.LoopInfo(packet_count, v7.LOOP_NO_CLOCK, pingpong=True)
    directions = [1]*packet_count
    started = time.perf_counter()
    if profile_name == 'stereo-fold-500':
        audio = wire.encode(
            model, [values]*packet_count, aspect_codes=codes,
            source_indices=indices, loop=loop, directions=directions)
    else:
        audio = wire.encode(model, [values]*packet_count, aspect_codes=codes,
                            source_indices=indices)
    elapsed = (time.perf_counter()-started)*1000
    return audio, elapsed


def _encode_sequence(wire, profile_name, model, values, aspect_code):
    codes = [aspect_code]*len(values)
    indices = list(range(len(values)))
    loop = v7.LoopInfo(len(values), v7.LOOP_NO_CLOCK, pingpong=True)
    directions = [1]*len(values)
    started = time.perf_counter()
    if profile_name == 'stereo-fold-500':
        audio = wire.encode(
            model, values, aspect_codes=codes, source_indices=indices,
            loop=loop, directions=directions)
    else:
        audio = wire.encode(model, values, aspect_codes=codes,
                            source_indices=indices)
    return audio, (time.perf_counter()-started)*1000


def _decode(wire, profile_name, model, audio96, source_aspect):
    started = time.perf_counter()
    if profile_name == 'stereo-fold-500':
        results, info = wire.decode(
            model, audio96, sample_rate=RATE,
            state=v7.PulseState(tail_memory=True))
        value_fn = lambda result: wire.values(model, result)
    else:
        mono_model = wire.model_for(model)
        with wire.receiving():
            results, info = v7.decode_pulse_stream(
                mono_model, audio96[:, 0], sample_rate=RATE,
                pilot_timing='tone-seeded',
                state=v7.PulseState(tail_memory=False))
        value_fn = lambda result: wire.values(mono_model, result)
    elapsed = (time.perf_counter()-started)*1000
    usable = [result for result in results
              if result.status != 'lost' and result.diag.get('displayable', False)]
    if not usable:
        return None, {
            'decoded_results': len(results),
            'displayable_results': 0,
            'all_packets_displayable': False,
            'decode_elapsed_ms': elapsed,
            'eof_validated': int(info.get('eof_markers_validated') or 0),
        }
    result = usable[-1]
    values = value_fn(result)
    return _receiver_image(values, model.coder.grids, source_aspect), {
        'decoded_results': len(results),
        'displayable_results': len(usable),
        'all_packets_displayable': len(usable) == PACKETS,
        'decode_elapsed_ms': elapsed,
        'eof_validated': int(info.get('eof_markers_validated') or 0),
        'last_counter': int(result.counter),
    }


def _decode_sequence(wire, profile_name, model, audio96, source_aspect):
    started = time.perf_counter()
    if profile_name == 'stereo-fold-500':
        results, info = wire.decode(
            model, audio96, sample_rate=RATE,
            state=v7.PulseState(tail_memory=True))
        value_fn = lambda result: wire.values(model, result)
    else:
        mono_model = wire.model_for(model)
        with wire.receiving():
            results, info = v7.decode_pulse_stream(
                mono_model, audio96[:, 0], sample_rate=RATE,
                pilot_timing='tone-seeded',
                state=v7.PulseState(tail_memory=False))
        value_fn = lambda result: wire.values(mono_model, result)
    elapsed = (time.perf_counter()-started)*1000
    pictures = []
    for result in results:
        if result.status == 'lost' or not result.diag.get('displayable', False):
            continue
        pictures.append((int(result.counter), _receiver_image(
            value_fn(result), model.coder.grids, source_aspect)))
    return pictures, {
        'decoded_results': len(results),
        'displayable_results': len(pictures),
        'all_packets_displayable': len(pictures) == PACKETS,
        'decode_elapsed_ms': elapsed,
        'eof_validated': int(info.get('eof_markers_validated') or 0),
    }


def _moving_detail_frames():
    frames = []
    for frame_index in range(PACKETS):
        image = Image.new('RGB', (720, 960), (32, 44, 58))
        draw = ImageDraw.Draw(image)
        # A translated high-frequency target crosses a fixed textured field;
        # position changes are packet-synchronous and the surrounding border
        # is an explicit static region for temporal-error measurements.
        for x in range(24, 696, 8):
            shade = 176 if (x//8) % 2 else 74
            draw.line((x, 90, x, 870), fill=(shade, shade, shade), width=2)
        x0 = 72+frame_index*24
        y0 = 318+frame_index*5
        for y in range(y0, y0+128, 8):
            for x in range(x0, x0+144, 8):
                color = ((238, 56, 42) if (x//8+y//8) % 2
                         else (28, 218, 220))
                draw.rectangle((x, y, x+7, y+7), fill=color)
        edge_x = 110+frame_index*20
        draw.rectangle((edge_x, 220, edge_x+15, 760),
                       fill=(245, 245, 245))
        draw.rectangle((edge_x+16, 220, edge_x+30, 760),
                       fill=(18, 18, 18))
        draw.text((34, 906), f'MOTION {frame_index+1:02d}',
                  fill=(242, 242, 242))
        frames.append(image)
    return frames


def _mean_scores(scores):
    if not scores:
        return None
    return {name: float(np.mean([score[name] for score in scores]))
            for name in scores[0]}


def _static_temporal_error(pictures, references, source_aspect):
    ordered = sorted(pictures, key=lambda item: item[0])
    if len(ordered) < 2:
        return None
    left = (DISPLAY_SIZE[0]-_viewport_size(source_aspect)[0])//2
    top = (DISPLAY_SIZE[1]-_viewport_size(source_aspect)[1])//2
    viewport_width, viewport_height = _viewport_size(source_aspect)
    x0 = left+int(.92*viewport_width)
    x1 = left+viewport_width
    region = (slice(top, top+viewport_height), slice(x0, x1))
    output_deltas, reference_deltas = [], []
    for index in range(1, len(ordered)):
        previous_counter, previous_image = ordered[index-1]
        current_counter, current_image = ordered[index]
        if current_counter != previous_counter+1:
            continue
        previous = np.asarray(previous_image.convert('YCbCr'))[..., 0].astype(float)
        current = np.asarray(current_image.convert('YCbCr'))[..., 0].astype(float)
        ref_previous = np.asarray(
            references[(previous_counter-1) % len(references)].convert('YCbCr'))[..., 0].astype(float)
        ref_current = np.asarray(
            references[(current_counter-1) % len(references)].convert('YCbCr'))[..., 0].astype(float)
        output_deltas.append(current[region]-previous[region])
        reference_deltas.append(ref_current[region]-ref_previous[region])
    if not output_deltas:
        return None
    error = np.concatenate([
        (got-want).ravel()
        for got, want in zip(output_deltas, reference_deltas)])
    return {
        'static_region_temporal_delta_mae_y': float(np.mean(np.abs(error))),
        'static_region_temporal_delta_max_y': float(np.max(np.abs(error))),
    }


def _contact_sheet(path, reference, tiles):
    columns, rows = 4, 5
    tile_width, tile_height = 360, 300
    margin, heading = 12, 34
    sheet = Image.new(
        'RGB', (columns*(tile_width+margin)+margin,
                rows*(tile_height+heading+margin)+margin), '#d8d8d8')
    draw = ImageDraw.Draw(sheet)
    entries = [('source', reference)] + tiles
    for index in range(columns*rows):
        col, row = index % columns, index//columns
        x = margin+col*(tile_width+margin)
        y = margin+row*(tile_height+heading+margin)
        if index >= len(entries):
            continue
        label, image = entries[index]
        draw.text((x, y), label[:55], fill='black')
        if image is not None:
            thumb = _fit_canvas(image, (tile_width, tile_height),
                                image.width/image.height, background='#777777')
            sheet.paste(thumb, (x, y+heading))
        else:
            draw.rectangle((x, y+heading, x+tile_width, y+heading+tile_height),
                           fill='#aaaaaa')
            draw.text((x+12, y+heading+18), 'NO DECODE', fill='black')
    sheet.save(path)


def _paired_sheet(path, reference, tiles):
    tile_width, tile_height = 480, 360
    margin, heading = 12, 34
    row_height = tile_height+heading+margin
    sheet = Image.new('RGB', (2*tile_width+3*margin,
                              len(tiles)*row_height+margin), '#d8d8d8')
    draw = ImageDraw.Draw(sheet)
    reference_tile = _fit_canvas(
        reference, (tile_width, tile_height), reference.width/reference.height)
    for index, (label, image) in enumerate(tiles, start=1):
        y = margin+(index-1)*row_height
        draw.text((margin, y), f'{index:02d} source', fill='black')
        draw.text((2*margin+tile_width, y), f'{index:02d} {label}', fill='black')
        sheet.paste(reference_tile, (margin, y+heading))
        if image is None:
            draw.rectangle((2*margin+tile_width, y+heading,
                            2*margin+2*tile_width, y+heading+tile_height),
                           fill='#aaaaaa')
            draw.text((2*margin+tile_width+12, y+heading+18),
                      'NO DECODE', fill='black')
        else:
            sheet.paste(_fit_canvas(
                image, (tile_width, tile_height), image.width/image.height),
                (2*margin+tile_width, y+heading))
    sheet.save(path)


def _warmup(model):
    v7.warmup_equalizer(model)
    v7.PULSE.warmup_pulse_kernels()
    from tone_code import warmup_coded_decoder
    warmup_coded_decoder(model)


def _motion_pair_sheet(path, references, pictures):
    tile_width, tile_height = 480, 360
    margin, heading = 12, 32
    row_height = tile_height+heading+margin
    sheet = Image.new('RGB', (2*tile_width+3*margin,
                              len(references)*row_height+margin), '#d8d8d8')
    draw = ImageDraw.Draw(sheet)
    by_counter = {counter: image for counter, image in pictures}
    for index, reference in enumerate(references):
        counter = index+1
        y = margin+index*row_height
        draw.text((margin, y), f'{counter:02d} source', fill='black')
        draw.text((2*margin+tile_width, y), f'{counter:02d} decoded',
                  fill='black')
        sheet.paste(_fit_canvas(
            reference, (tile_width, tile_height),
            reference.width/reference.height), (margin, y+heading))
        image = by_counter.get(counter)
        if image is None:
            draw.rectangle((2*margin+tile_width, y+heading,
                            2*margin+2*tile_width, y+heading+tile_height),
                           fill='#aaaaaa')
            draw.text((2*margin+tile_width+12, y+heading+18), 'NO DECODE',
                      fill='black')
        else:
            sheet.paste(_fit_canvas(
                image, (tile_width, tile_height),
                image.width/image.height), (2*margin+tile_width, y+heading))
    sheet.save(path)


def run_motion(out, selected_variants=None, selected_profiles=None,
               selected_channels=None):
    """Transmit a 12-frame moving-detail scene and score temporal behavior."""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    variants = _select_variants(selected_variants)
    profiles = tuple(selected_profiles or PROFILE_NAMES)
    channels = tuple(selected_channels or DEFAULT_CHANNELS)
    if any(variant['kind'] in ('fold-native', 'fold-block')
           for variant in variants):
        raise ValueError('Fold coefficient projection is a still-image bench variant')
    if not variants:
        raise ValueError('no motion variants selected')
    if set(profiles)-set(PROFILE_NAMES):
        raise ValueError(f'unknown profile(s): {sorted(set(profiles)-set(PROFILE_NAMES))}')
    if set(channels)-set(CHANNEL_NAMES):
        raise ValueError(f'unknown channel(s): {sorted(set(channels)-set(CHANNEL_NAMES))}')
    model = v7.load_model(TARGET, 'box')
    wires = _profile_wires(model)
    _warmup(model)
    frames = _moving_detail_frames()
    width, height = _source_size(frames[0])
    aspect = width/height
    aspect_code = v7.aspect_wire_code((width, height))
    references = [_display(frame, aspect) for frame in frames]
    (out/'sources').mkdir(exist_ok=True)
    (out/'source-grid').mkdir(exist_ok=True)
    (out/'ideal-fold').mkdir(exist_ok=True)
    (out/'decoded').mkdir(exist_ok=True)
    (out/'paired').mkdir(exist_ok=True)
    for index, frame in enumerate(frames, start=1):
        frame.save(out/'sources'/f'moving-detail-frame-{index:02d}.png')

    prepared = {}
    source_grid_scores = {}
    source_prepare_ms = {}
    for variant in variants:
        name = variant['name']
        values_list = []
        images = []
        scores = []
        prepare_elapsed_ms = 0.0
        for frame, reference in zip(frames, references):
            prepare_started = time.perf_counter()
            if variant['kind'] == 'baseline':
                values, _ = application_values(model, frame, 1.0, 1.0)
            else:
                values, _ = source_dct_values(
                    np.asarray(frame), model.coder.grids, model.coder.shapes,
                    brightness=1.0, gamma=1.0,
                    display_size=DISPLAY_SIZE,
                    viewing_distance_mm=VIEWING_DISTANCE_MM,
                    display_dpi=DISPLAY_DPI,
                    **variant.get('options', {}))
            prepare_elapsed_ms += (
                time.perf_counter()-prepare_started)*1000
            values_list.append(values)
            image = _receiver_image(values, model.coder.grids, aspect)
            images.append(image)
            scores.append(_score(reference, image))
        prepared[name] = values_list
        source_grid_scores[name] = scores
        source_prepare_ms[name] = prepare_elapsed_ms
        for index, image in enumerate(images, start=1):
            image.save(out/'source-grid'/f'{name}-frame-{index:02d}.png')

    type_ii = next(case for case in CASES if case.name == 'type-ii')
    profile_manifest = {}
    rows = []
    for profile_name in profiles:
        wire = wires[profile_name]
        _encode(wire, profile_name, model,
                np.zeros(model.coder.source_count, dtype=np.float64), 0,
                packet_count=1)
        profile_manifest[profile_name] = {
            'wire_table_sha256': _profile_table_identity(
                profile_name, wire, model),
            'video_side': getattr(wire, 'side', 'stereo'),
            'fold_slots': 500,
        }
        for variant in variants:
            name = variant['name']
            values_list = prepared[name]
            ideal_images, ideal_scores = [], []
            codec = None
            for values, reference in zip(values_list, references):
                ideal_values, codec = _ideal_fold_values(
                    profile_name, wire, model, values)
                image = _receiver_image(
                    ideal_values, model.coder.grids, aspect)
                ideal_images.append(image)
                ideal_scores.append(_score(reference, image))
            for index, image in enumerate(ideal_images, start=1):
                image.save(out/'ideal-fold'/
                           f'{name}_{profile_name}_frame-{index:02d}.png')

            audio48, encode_ms = _encode_sequence(
                wire, profile_name, model, values_list, aspect_code)
            audio96 = _production_output_rate(audio48)
            frame_fold_stats = [_fold_guest_stats(codec, values)
                                for values in values_list]
            fold_stats = {
                'fold_guest_clip_fraction': float(np.mean([
                    item['fold_guest_clip_fraction']
                    for item in frame_fold_stats])),
                'fold_guest_max_clip_excess_sigma': float(np.max([
                    item['fold_guest_max_clip_excess_sigma']
                    for item in frame_fold_stats])),
                'fold_guest_rms_sigma': float(np.mean([
                    item['fold_guest_rms_sigma'] for item in frame_fold_stats])),
            }
            for channel_name in channels:
                channel_audio = (audio96 if channel_name == 'clean-96k' else
                                 impair(audio96, type_ii, seed=2026))
                pictures, decode_stats = _decode_sequence(
                    wire, profile_name, model, channel_audio, aspect)
                frame_scores = []
                for counter, image in pictures:
                    frame_index = (counter-1) % PACKETS
                    image.save(out/'decoded'/
                               f'{name}_{profile_name}_{channel_name}_frame-{counter:02d}.png')
                    frame_scores.append({
                        'counter': counter,
                        'source_frame': frame_index+1,
                        'score': _score(references[frame_index], image),
                    })
                temporal = _static_temporal_error(
                    pictures, references, aspect)
                comparison_path = out/'paired'/(
                    f'{name}_{profile_name}_{channel_name}_motion.png')
                _motion_pair_sheet(comparison_path, references, pictures)
                row = {
                    'source': 'moving-detail-720x960',
                    'source_size': [width, height],
                    'variant': name,
                    'description': variant['description'],
                    'profile': profile_name,
                    'wire_table_sha256': profile_manifest[
                        profile_name]['wire_table_sha256'],
                    'channel': channel_name,
                    'packets': PACKETS,
                    'source_grid_mean_score': _mean_scores(
                        source_grid_scores[name]),
                    'ideal_fold_mean_score': _mean_scores(ideal_scores),
                    'received_mean_score': _mean_scores(
                        [item['score'] for item in frame_scores]),
                    'frame_scores': frame_scores,
                    'source_prepare_total_ms': source_prepare_ms[name],
                    'encode_ms_per_packet': encode_ms/PACKETS,
                    'wave_peak': float(np.max(np.abs(audio48))),
                    'wave_rms': float(np.sqrt(np.mean(
                        audio48.astype(np.float64)**2))),
                    **fold_stats,
                    **decode_stats,
                    **(temporal or {}),
                    'comparison_png': str(comparison_path.relative_to(out)),
                    'decoded_frames': [
                        str((out/'decoded'/
                             f'{name}_{profile_name}_{channel_name}_frame-{counter:02d}.png').relative_to(out))
                        for counter, _image in pictures],
                }
                rows.append(row)
                print(json.dumps(row, allow_nan=True), flush=True)

    report = {
        'objective': '12-frame moving-detail transmission with frame-matched reconstruction and static-region temporal error',
        'wire_encode_engine': (
            'application packet path: modem_v7_display.encode_values_packet '
            'with animation_modem.v7_fold.Fold500 for stereo; '
            'MonoFreshFoldWire.encode for experimental mono profiles'),
        'output_rate_conversion': (
            'animation_modem.v7_core.adapt_packet_for_output per packet '
            '(PacketOutput path), speed=1.0, 48 kHz reference to 96 kHz output'),
        'display_size': list(DISPLAY_SIZE),
        'display_reconstruction': 'receiver DCT viewport; nearest sampling',
        'packets_per_trial': PACKETS,
        'model_table_sha256': v7.MODEL_TABLES_SHA256,
        'band_profile_assumptions': {
            'perceptual-color_csf': '2.6*(0.0192+0.114*f)*exp(-(0.114*f)^1.1), normalized to the sampled peak; Y gain 0.78+0.42*CSF, chroma gain 0.70+0.30*CSF, DC reset to 1.0',
            'mid-luma': '1+0.20*exp(-0.5*((radius-0.42)/0.15)^2); Y only, DC reset to 1.0',
            'archived_gain_arrays': _band_profile_archive(
                model.coder.grids, aspect),
        },
        'profiles': list(profiles),
        'profile_tables': profile_manifest,
        'channels': list(channels),
        'type_ii_case': asdict(type_ii),
        'type_ii_random_seed': 2026,
        'variants': variants,
        'invocation': sys.argv,
        'rows': rows,
    }
    (out/'results.json').write_text(json.dumps(report, indent=2, allow_nan=True)+'\n')
    print(f'Wrote {len(rows)} moving-detail results to {out}/results.json')
    return report


def run(out, source_paths=(), source_dir=DEFAULT_SOURCE_DIR, max_sources=2,
        include_synthetic=True, selected_variants=None, selected_profiles=None,
        selected_channels=None):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    sources = load_sources(source_paths, source_dir, max_sources, include_synthetic)
    variants = _select_variants(selected_variants)
    profiles = tuple(selected_profiles or PROFILE_NAMES)
    channels = tuple(selected_channels or DEFAULT_CHANNELS)
    if not variants:
        raise ValueError('no benchmark variants selected')
    if set(profiles)-set(PROFILE_NAMES):
        raise ValueError(f'unknown profile(s): {sorted(set(profiles)-set(PROFILE_NAMES))}')
    if set(channels)-set(CHANNEL_NAMES):
        raise ValueError(f'unknown channel(s): {sorted(set(channels)-set(CHANNEL_NAMES))}')
    fold_projection_kinds = ('fold-native', 'fold-block')
    if any(variant['kind'] in fold_projection_kinds for variant in variants) and \
            profiles != ('stereo-fold-500',):
        raise ValueError(
            'Fold coefficient projection requires --profile stereo-fold-500')
    cases = {case.name: case for case in CASES}
    type_ii = cases['type-ii']
    model = v7.load_model(TARGET, 'box')
    wires = _profile_wires(model)
    _warmup(model)
    production_fold = None
    if any(variant['kind'] in fold_projection_kinds for variant in variants):
        from animation_modem.v7_fold import Fold500
        production_fold = Fold500(model)
    profile_manifest = {}
    zero_values = np.zeros(model.coder.source_count, dtype=np.float64)
    for profile_name in profiles:
        wire = wires[profile_name]
        profile_manifest[profile_name] = {
            'wire_table_sha256': _profile_table_identity(
                profile_name, wire, model),
            'video_side': getattr(wire, 'side', 'stereo'),
            'fold_slots': 500,
            'timing': 'pulse-counted coded-pilot acquisition',
        }
        _encode(wire, profile_name, model, zero_values, 0, packet_count=1)
    out.mkdir(parents=True, exist_ok=True)
    (out/'sources').mkdir(exist_ok=True)
    (out/'source-grid').mkdir(exist_ok=True)
    (out/'ideal-fold').mkdir(exist_ok=True)
    (out/'decoded').mkdir(exist_ok=True)
    (out/'paired').mkdir(exist_ok=True)
    rows = []
    source_manifest = []
    started_all = time.monotonic()

    for source_name, source, source_origin in sources:
        source_name = _safe_name(source_name)
        source_width, source_height = _source_size(source)
        aspect = source_width/source_height
        source_rgb = _source_rgb_image(source)
        source_display = _display(source_rgb, aspect)
        source_viewport = _crop_viewport(source_display, aspect)
        source_path = out/'sources'/f'{source_name}.png'
        source_rgb.save(source_path)
        reference_path = out/'sources'/f'{source_name}_display.png'
        source_display.save(reference_path)
        source_manifest.append({
            'name': source_name, 'size': [source_width, source_height],
            'input_dtype': (str(np.asarray(source).dtype)
                            if not isinstance(source, Image.Image)
                            else 'uint8'),
            'aspect': aspect,
            'viewport_size': list(_viewport_size(aspect)),
            'source_png': str(source_path.relative_to(out)),
            'display_png': str(reference_path.relative_to(out)),
            'input_path': source_origin,
        })

        values_by_variant = {}
        coefficients_by_variant = {}
        source_scores = {}
        source_viewport_scores = {}
        prep_ms = {}
        diagnostics = {}
        for variant in variants:
            name = variant['name']
            before = time.perf_counter()
            if variant['kind'] == 'baseline':
                timings = []
                for _ in range(PREP_TIMING_REPEATS):
                    timing_start = time.perf_counter()
                    values, aspect_code = application_values(
                        model, source_rgb, brightness=1.0, gamma=1.0)
                    timings.append((time.perf_counter()-timing_start)*1000)
                prep_ms[name] = statistics.median(timings)
                diagnostic = None
            elif variant['kind'] in fold_projection_kinds:
                # Set up geometry/basis state before timing. A live stream can
                # reuse this plan for each frame with the same source size.
                project_options = variant.get('options', {})
                if variant['kind'] == 'fold-native':
                    project = lambda: source_fold_dct_coefficients(
                        source_rgb, model.coder.grids,
                        production_fold.source_positions)
                    projector_setup_ms = 0.0
                else:
                    setup_started = time.perf_counter()
                    projector = FoldBlockDCTProjector(
                        np.asarray(source_rgb).shape[:2], model.coder.grids,
                        production_fold.source_positions, **project_options)
                    projector_setup_ms = (
                        time.perf_counter()-setup_started)*1000
                    project = lambda: projector.project(source_rgb)
                timings = []
                for _ in range(PREP_TIMING_REPEATS):
                    timing_start = time.perf_counter()
                    full_coefficients = project()
                    timings.append((time.perf_counter()-timing_start)*1000)
                coefficients_by_variant[name] = full_coefficients
                # The source transform ends at the coefficient vector. Build
                # the comparison preview afterward so an IDCT used only for a
                # PNG does not get charged to the image-to-packet latency.
                prep_ms[name] = statistics.median(timings)
                visible = np.zeros_like(full_coefficients)
                visible[production_fold.kept] = full_coefficients[
                    production_fold.kept]
                values = _grid_values_from_dct(
                    visible, model.coder.grids)
                diagnostic = {
                    'projection': (
                        'native RGB into Fold-500 support rectangles'
                        if variant['kind'] == 'fold-native' else
                        'block-averaged RGB with integrated native DCT basis'),
                    'projected_position_count': int(
                        len(production_fold.source_positions)),
                    'unclipped_source_coefficients': True,
                    **project_options,
                    **({'projector_setup_ms': projector_setup_ms}
                       if variant['kind'] == 'fold-block' else {}),
                }
                aspect_code = v7.aspect_wire_code((source_width, source_height))
            else:
                source_array = (np.asarray(source)
                                if not isinstance(source, Image.Image)
                                else np.asarray(source_rgb))
                options = variant.get('options', {})
                # Warm the source transform, then use the same repeated median
                # preparation timing as resize-first and Fold-native. Keep the
                # last prepared values for the unchanged packet/receiver run.
                values, diagnostic = source_dct_values(
                    source_array, model.coder.grids, model.coder.shapes,
                    brightness=1.0, gamma=1.0,
                    display_size=DISPLAY_SIZE,
                    viewing_distance_mm=VIEWING_DISTANCE_MM,
                    display_dpi=DISPLAY_DPI, **options)
                timings = []
                for _ in range(PREP_TIMING_REPEATS):
                    timing_start = time.perf_counter()
                    values, diagnostic = source_dct_values(
                        source_array, model.coder.grids, model.coder.shapes,
                        brightness=1.0, gamma=1.0,
                        display_size=DISPLAY_SIZE,
                        viewing_distance_mm=VIEWING_DISTANCE_MM,
                        display_dpi=DISPLAY_DPI, **options)
                    timings.append((time.perf_counter()-timing_start)*1000)
                prep_ms[name] = statistics.median(timings)
                aspect_code = v7.aspect_wire_code((source_width, source_height))
            if name not in prep_ms:
                prep_ms[name] = (time.perf_counter()-before)*1000
            values_by_variant[name] = values
            diagnostics[name] = diagnostic
            preview = _receiver_image(values, model.coder.grids, aspect)
            source_scores[name] = _score(source_display, preview)
            source_viewport_scores[name] = _score(
                source_viewport, _crop_viewport(preview, aspect))
            preview.save(out/'source-grid'/f'{source_name}_{name}.png')

        for profile_name in profiles:
            wire = wires[profile_name]
            ideal_scores = {}
            ideal_viewport_scores = {}
            for variant in variants:
                name = variant['name']
                values = values_by_variant[name]
                if variant['kind'] in fold_projection_kinds:
                    (ideal_values, codec, _folded_coefficients) = \
                        _ideal_fold_projected_values(
                            production_fold, wire, model,
                            coefficients_by_variant[name])
                    fold_stats = _fold_guest_stats_from_coefficients(
                        codec, coefficients_by_variant[name])
                else:
                    ideal_values, codec = _ideal_fold_values(
                        profile_name, wire, model, values)
                    fold_stats = _fold_guest_stats(codec, values)
                ideal_image = _receiver_image(
                    ideal_values, model.coder.grids, aspect)
                ideal_image.save(
                    out/'ideal-fold'/f'{source_name}_{name}_{profile_name}.png')
                ideal_scores[name] = _score(source_display, ideal_image)
                ideal_viewport_scores[name] = _score(
                    source_viewport, _crop_viewport(ideal_image, aspect))
                if variant['kind'] in fold_projection_kinds:
                    audio48, encode_ms = _encode_fold_projected(
                        production_fold, model,
                        coefficients_by_variant[name], aspect_code)
                else:
                    audio48, encode_ms = _encode(
                        wire, profile_name, model, values, aspect_code)
                wave_peak = float(np.max(np.abs(audio48)))
                wave_rms = float(np.sqrt(np.mean(audio48.astype(np.float64)**2)))
                captured_clean = _production_output_rate(audio48)

                for channel_name in channels:
                    if channel_name == 'clean-96k':
                        audio96 = captured_clean
                    else:
                        audio96 = impair(captured_clean, type_ii, seed=2026)
                    decoded_image, decode_stats = _decode(
                        wire, profile_name, model, audio96, aspect)
                    candidate_path = out/'decoded'/(
                        f'{source_name}_{name}_{profile_name}_{channel_name}.png')
                    if decoded_image is not None:
                        decoded_image.save(candidate_path)
                    direct_score = source_scores[name]
                    ideal_score = ideal_scores[name]
                    received_score = (_score(source_display, decoded_image)
                                      if decoded_image is not None else None)
                    received_viewport_score = (
                        _score(source_viewport,
                               _crop_viewport(decoded_image, aspect))
                        if decoded_image is not None else None)
                    source_diag = diagnostics[name] or {}
                    row = {
                        'source': source_name,
                        'source_size': [source_width, source_height],
                        'variant': name,
                        'description': variant['description'],
                        'profile': profile_name,
                        'wire_table_sha256': profile_manifest[
                            profile_name]['wire_table_sha256'],
                        'channel': channel_name,
                        'aspect_code': int(aspect_code),
                        'source_prepare_ms': prep_ms[name],
                        'encode_ms_per_packet': encode_ms/PACKETS,
                        'wire_peak': wave_peak,
                        'wire_rms': wave_rms,
                        'source_grid_score': direct_score,
                        'source_grid_viewport_score':
                            source_viewport_scores[name],
                        'ideal_fold_score': ideal_score,
                        'ideal_fold_viewport_score':
                            ideal_viewport_scores[name],
                        'received_score': received_score,
                        'received_viewport_score': received_viewport_score,
                        'source_dct_diagnostics': source_diag,
                        **fold_stats,
                        **decode_stats,
                        'source_grid_png': str((out/'source-grid'/
                            f'{source_name}_{name}.png').relative_to(out)),
                        'ideal_fold_png': str((out/'ideal-fold'/
                            f'{source_name}_{name}_{profile_name}.png').relative_to(out)),
                        'decoded_png': (str(candidate_path.relative_to(out))
                                        if decoded_image is not None else None),
                    }
                    rows.append(row)
                    print(json.dumps(row, allow_nan=True), flush=True)

            for channel_name in channels:
                tiles = []
                paired_tiles = []
                viewport_tiles = []
                for variant in variants:
                    row = next(row for row in reversed(rows)
                               if row['source'] == source_name and
                               row['variant'] == variant['name'] and
                               row['profile'] == profile_name and
                               row['channel'] == channel_name)
                    path = row['decoded_png']
                    image = Image.open(out/path).convert('RGB') if path else None
                    value = ((row['received_score'] or {}).get('ssimulacra2'))
                    label = variant['name'] + (
                        f'  S2 {value:.1f}' if value is not None else '  no decode')
                    tiles.append((label, image))
                    paired_tiles.append((label, image))
                    viewport_value = ((row['received_viewport_score'] or {})
                                      .get('ssimulacra2'))
                    viewport_label = variant['name'] + (
                        f'  viewport S2 {viewport_value:.1f}'
                        if viewport_value is not None else '  no decode')
                    viewport_tiles.append((
                        viewport_label,
                        _crop_viewport(image, aspect) if image is not None
                        else None))
                _contact_sheet(
                    out/'decoded'/f'{source_name}_{profile_name}_{channel_name}_sheet.png',
                    source_display, tiles)
                _paired_sheet(
                    out/'paired'/f'{source_name}_{profile_name}_{channel_name}_comparison.png',
                    source_display, paired_tiles)
                _paired_sheet(
                    out/'paired'/f'{source_name}_{profile_name}_{channel_name}_viewport.png',
                    source_viewport, viewport_tiles)

    report = {
        'objective': ('Compare resize-first, native source-DCT, and Fold-500-slot '
                      'projection through the existing V7 modem and receiver'),
        'wire_encode_engine': (
            'application packet path: modem_v7_display.encode_values_packet '
            'with animation_modem.v7_fold.Fold500 for value candidates; '
            'fold-native-projection uses encode_folded_coefficients_packet '
            'after the same production Fold500; '
            'MonoFreshFoldWire.encode for experimental mono profiles'),
        'output_rate_conversion': (
            'animation_modem.v7_core.adapt_packet_for_output per packet '
            '(PacketOutput path), speed=1.0, 48 kHz reference to 96 kHz output'),
        'synthetic_channel_note': 'Type II is a fixed synthetic regression channel, not tape emulation.',
        'source_processing_note': (
            'Native source-DCT variants analyze the full source eye; '
            'resize-first is the production baseline.'),
        'display_size': list(DISPLAY_SIZE),
        'display_reconstruction': (
            'receiver DCT viewport reconstruction via '
            'tools.v7_gl_viewer.dct_reconstruct_planes'),
        'viewport_score_note': (
            'Each source-grid, ideal-fold, and received score is reported both '
            'on the fixed letterboxed canvas and on the exact centered viewport '
            'crop; viewport scores exclude the shared background bars.'),
        'display_sampling': 'nearest, matching the receiver GUI default',
        'viewing_distance_mm': VIEWING_DISTANCE_MM,
        'display_dpi_assumption': DISPLAY_DPI,
        'band_profile_assumptions': {
            'perceptual-color_csf': '2.6*(0.0192+0.114*f)*exp(-(0.114*f)^1.1), normalized to the sampled peak; Y gain 0.78+0.42*CSF, chroma gain 0.70+0.30*CSF, DC reset to 1.0',
            'mid-luma': '1+0.20*exp(-0.5*((radius-0.42)/0.15)^2); Y only, DC reset to 1.0',
            'archived_gain_arrays_by_viewport_aspect': {
                str(aspect): _band_profile_archive(
                    model.coder.grids, aspect)
                for aspect in sorted({source['aspect']
                                      for source in source_manifest})
            },
        },
        'model': 'canonical box V7 model',
        'model_table_sha256': v7.MODEL_TABLES_SHA256,
        'packets_per_trial': PACKETS,
        'source_prepare_timing_repeats': PREP_TIMING_REPEATS,
        'profiles': list(profiles),
        'profile_tables': profile_manifest,
        'channels': list(channels),
        'type_ii_case': asdict(type_ii),
        'type_ii_random_seed': 2026,
        'coefficient_reduction': {
            'active': 'direct-retention selects same-index source modes; fold-native-projection evaluates the bounding DCT rectangles for Fold500.kept union Fold500.guests; fold-block variants approximate these native coefficients from block means and integrated full-resolution basis vectors',
            'specialized_variants': 'area-box-resample, fold-native-projection, and fold-block variants are explicit-only; Fold coefficient projection requires stereo-fold-500',
            'weighted_mode_coordinates': 'source and target DCT-II indices share the whole-frame basis; gains multiply matching modes without source_count-to-target_count scaling or signed-mode averaging',
            'area-box_boundary': 'each target sample is the exact normalized overlap of its source-pixel footprint',
            'weighted-tent': 'separable gains g(k)=1-k/N on matching retained modes; preserve DC at unit gain',
            'weighted-cosine': 'separable gains g(k)=0.5+0.5*cos(pi*k/N) on matching retained modes; preserve DC at unit gain',
            'weighted-gaussian': 'separable gains g(k)=exp(-0.5*(k/(0.5*N))^2) on matching retained modes; preserve DC at unit gain',
            'direct_retention_amplitude_normalization': 'sqrt(coder-grid area/source-plane area)',
            'area_box_amplitude_normalization': 'unit target-grid DCT normalization; no source-area compensation',
        },
        'invocation': sys.argv,
        'variants': [{key: value for key, value in variant.items()}
                     for variant in variants],
        'sources': source_manifest,
        'elapsed_seconds': time.monotonic()-started_all,
        'rows': rows,
    }
    (out/'results.json').write_text(json.dumps(report, indent=2, allow_nan=True)+'\n')
    print(f'Wrote {len(rows)} profile/channel results to {out}/results.json')
    return report


def application_values(model, image, brightness, gamma):
    # Use the application's image preparation, not the separate tools/v7_live
    # prototype. Candidate branches replace only this value-preparation step;
    # every encoded packet then goes through modem_v7_display.packet's encoder.
    if brightness != 1.0 or gamma != 1.0:
        raise ValueError('benchmark production baseline uses neutral tone controls')
    from modem_v7_display import _source_values
    values = _source_values(model, image, 'box')
    return values, v7.aspect_wire_code(image.size)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path,
                        default=ROOT/'tmp/v7-source-dct-bench')
    parser.add_argument('--source', action='append', type=Path, default=[],
                        help='native RGB file (repeatable; SBS inputs use the left half)')
    parser.add_argument('--source-dir', type=Path, default=DEFAULT_SOURCE_DIR)
    parser.add_argument('--max-sources', type=int, default=2,
                        help='maximum native image files from --source-dir (default: 2)')
    parser.add_argument('--no-synthetic', action='store_true',
                        help='omit the smooth gradient and chart/text scenes')
    parser.add_argument('--motion-only', action='store_true',
                        help='run the packet-synchronous 12-frame moving-detail check')
    parser.add_argument('--variant', action='append', choices=tuple(VARIANT_BY_NAME),
                        help='limit variants; repeat this option')
    parser.add_argument('--profile', action='append', choices=PROFILE_NAMES,
                        help='limit modem profiles; repeat this option')
    parser.add_argument('--channel', action='append', choices=CHANNEL_NAMES,
                        help='limit channels; repeat this option')
    args = parser.parse_args(argv)
    if args.max_sources < 1:
        parser.error('--max-sources must be at least 1')
    if args.motion_only:
        if args.source or args.no_synthetic:
            parser.error('--motion-only generates its own scene; omit --source and --no-synthetic')
        run_motion(args.out, args.variant, args.profile, args.channel)
    else:
        run(args.out, args.source, args.source_dir, args.max_sources,
            not args.no_synthetic, args.variant, args.profile, args.channel)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
