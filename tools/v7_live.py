#!/usr/bin/env python3
"""Bench-only live V7 camera/screen/video sender and receiver.

This deliberately does not enter ``main.py`` or the production modem engine.
It uses the V7 prototype's fixed 48 kHz reference geometry and follows the
selected DAC's native output clock by default, intended for an explicit audio
loopback device, normally BlackHole 2ch.

Examples::

    .venv/bin/python tools/v7_live.py send --source camera --device 'BlackHole 2ch'
    .venv/bin/python tools/v7_live.py send --source screen --device 'BlackHole 2ch'
    .venv/bin/python tools/v7_live.py send --source video \\
        --video-source clip.mp4 --device 'BlackHole 2ch'
    .venv/bin/python tools/v7_live.py send --source video \\
        --video-source 'rtsp://camera.example/live' --device 'BlackHole 2ch'
    .venv/bin/python tools/v7_live.py receive --device 'BlackHole 2ch'
    .venv/bin/python tools/v7_live.py receive --device 'BlackHole 2ch' --force-float32
    .venv/bin/python tools/v7_live.py gui

The prototype currently emits finite batches. Batch boundaries are therefore a
known live-experiment limitation; the clock counter continues across batches so
the receiver can diagnose the behavior rather than silently restarting it.
"""
import argparse
from collections import deque
import json
import math
import os
import queue
import sys
import threading
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageEnhance

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from animation_modem.imaging import values_image                         # noqa: E402
from animation_modem import v7 as P                                       # noqa: E402
from animation_modem.v7_live_input import (DirectionStreak, LiveInput,
                                           select_packet_hit,
                                           windowed_rate)                 # noqa: E402
image_values = P.image_values
prepare_image = P.prepare_image
try:
    from animation_modem.imaging import stabilize_chroma                  # noqa: E402
except ImportError:
    from scipy.ndimage import gaussian_filter                             # noqa: E402

    def stabilize_chroma(values, previous, shapes, pilot_error=None,
                         coverage=None):
        current = np.asarray(values)
        if previous is None or len(shapes) < 3 or current.shape != np.shape(previous):
            return current
        error = 0.0 if pilot_error is None else max(0.0, float(pilot_error))
        seen = 1.0 if coverage is None else np.clip(float(coverage), 0.0, 1.0)
        amount = max((error - .20)/.80, (.85 - seen)/.85, 0.0)
        amount = float(np.clip(amount*.65, 0.0, .65))
        if amount <= 0:
            return current
        result = current.copy()
        old = np.asarray(previous)
        offset = shapes[0][0]*shapes[0][1]
        for rows, cols in shapes[1:]:
            count = rows*cols
            blended = current[offset:offset+count]*(1-amount) + old[offset:offset+count]*amount
            sigma = min(1.6, .35 + 1.8*amount)
            result[offset:offset+count] = gaussian_filter(
                blended.reshape(rows, cols), sigma=sigma, mode='nearest').ravel()
            offset += count
        return result
from tools.v7_display import LatestFrame                                      # noqa: E402


FPS = P.PULSE_FPS
CAMERA_CAPTURE_FPS = 15
# Retain the pre-low-latency capture cushion: at 96 kHz and 1024-frame input
# blocks, 32 slots absorb about 341 ms of decoder scheduling stalls.
INPUT_AUDIO_QUEUE_BLOCKS = 32
# GUI users often start the receiver before the sender has finished warming up.
PROFILE_PROBE_TIMEOUT = 2.0
GUI_PROFILE_PROBE_TIMEOUT = 8.0
# Start after one packet is ready and allow only one pending encoded batch; the
# modem packet itself is the unavoidable serialization buffer for each image.
SENDER_STARTUP_BUFFER_SECONDS = P.PULSE_FRAME / P.RATE
DEVICE_RECOVERY_POLL_SECONDS = 1/P.PULSE_FPS


class SenderDeviceLost(RuntimeError):
    """The selected transmitter output stopped or changed its sample rate."""


def _wait_for_stable_send_device(sd, preferred, identity, stop):
    from tools.v7_device_recovery import (
        DEVICE_RATE_FRAME_INTERVAL, DeviceRateDebouncer,
        query_device_snapshot)

    stable = DeviceRateDebouncer()
    while not stop.is_set():
        try:
            snapshot = query_device_snapshot(
                sd, preferred, identity, 'output')
        except Exception:
            stable.reset()
        else:
            index, _info, observed_identity, rate = snapshot
            if stable.observe(observed_identity, rate):
                return index, rate
        stop.wait(DEVICE_RATE_FRAME_INTERVAL)
    return None


def _sender_queue_batches(startup_seconds, output_rate, packet_samples):
    if startup_seconds <= 0 or output_rate <= 0 or packet_samples <= 0:
        raise ValueError('startup duration and packet timing must be positive')
    return max(1, int(math.ceil(startup_seconds*output_rate/packet_samples)))
ENCODE_TO_FFMPEG_SCALE = {
    'nearest': 'neighbor',
    'box': 'area',
    'lanczos': 'lanczos',
    'bicubic': 'bicubic',
}
DEFAULT_FIXTURE = ROOT / 'modem_tests/fixtures/v7_reference_face.png'
# The display stays independent of the modem decoder and can consume the latest
# frame without building a backlog.
FRAME_BUFFER = LatestFrame()
# Read-only bridge for an external configuration UI. The receiver does not
# depend on a GUI; an attached UI may poll the diagnostic callback.
RECEIVER_GUI_STATUS = {}


class LiveToneControls:
    """Thread-safe brightness/gamma values updated by the sender GUI pipe."""

    def __init__(self, brightness, gamma):
        self._values = {'brightness': float(brightness), 'gamma': float(gamma)}
        self._lock = threading.Lock()

    def update(self, line):
        try:
            update = json.loads(line)
        except (TypeError, ValueError):
            return False
        if not isinstance(update, dict):
            return False
        accepted = {}
        for key in ('brightness', 'gamma'):
            if key not in update:
                continue
            try:
                value = float(update[key])
            except (TypeError, ValueError):
                continue
            if np.isfinite(value) and value > 0:
                accepted[key] = value
        if not accepted:
            return False
        with self._lock:
            self._values.update(accepted)
        return True

    def snapshot(self):
        with self._lock:
            return dict(self._values)


def _read_live_tone_controls(stream, controls, stop):
    """Consume newline-delimited GUI updates until EOF or shutdown."""
    while not stop.is_set():
        line = stream.readline()
        if not line:
            return
        controls.update(line)


def _device_arg(value):
    """Accept sounddevice names or numeric device indexes."""
    try:
        return int(value)
    except (TypeError, ValueError):
        return value


def _audio_volume_arg(value):
    try:
        volume = float(value)
    except (TypeError, ValueError) as exc:
        raise argparse.ArgumentTypeError(
            'audio volume must be a number from 0 to 1') from exc
    if not np.isfinite(volume) or not 0.0 <= volume <= 1.0:
        raise argparse.ArgumentTypeError(
            'audio volume must be between 0 and 1')
    return volume


def _resolve_send_source(args, interactive=None, input_fn=None):
    """Resolve optional interactive source and video-file/stream prompts."""
    if interactive is None:
        interactive = sys.stdin.isatty()
    if input_fn is None:
        input_fn = input

    def ask(prompt):
        try:
            return input_fn(prompt)
        except EOFError as exc:
            raise ValueError(
                'interactive source selection ended before a choice') from exc

    if args.source is None and args.video_source:
        args.source = 'video'
    if args.source is None:
        if not interactive:
            raise ValueError(
                'specify --source, or run interactively to choose one')
        print('Capture source: camera, screen, video, test, or mouse-follow')
        while True:
            selected = ask('Source: ').strip().lower()
            if selected in ('camera', 'screen', 'video', 'test', 'mouse-follow'):
                args.source = selected
                break
            print('Choose camera, screen, video, test, or mouse-follow.')

    if args.source == 'video':
        if not args.video_source:
            if not interactive:
                raise ValueError(
                    'video source missing; pass --video-source PATH_OR_URL')
            args.video_source = ask(
                'Video file path or live stream URL: ').strip()
        args.video_source = args.video_source.strip()
        if (len(args.video_source) >= 2 and
                args.video_source[0] == args.video_source[-1] and
                args.video_source[0] in ('"', "'")):
            args.video_source = args.video_source[1:-1]
        if not args.video_source:
            raise ValueError('video file path or stream URL cannot be empty')
    elif args.video_source:
        raise ValueError('--video-source can only be used with --source video')
    if getattr(args, 'preview', False) and args.source != 'video':
        raise ValueError('--preview can only be used with --source video')
    return args


def _capture(args):
    """Build one of the shared RGB capture sources."""
    from tools.v7_capture import (camera_source, mouse_follow_source,
                                  screen_capture_source, screen_source,
                                  test_source, video_source, Throttled, _region)

    region = _region(args.region)
    capture_filter = _capture_scale_flags(args)
    if args.source == 'test':
        return test_source()
    if args.source == 'video':
        return video_source(args.video_source, width=args.capture_width,
                            scale_flags=capture_filter,
                            live=True if args.video_live else None,
                            preserve_size=bool(getattr(args, 'dct_encode', False)))
    if args.source == 'mouse-follow':
        return mouse_follow_source(initial_width=args.capture_width)
    if args.source == 'camera':
        # Let FFmpeg probe the lowest mode/rate when the user did not override
        # it.  Some AVFoundation devices advertise 15 fps but reject a forced
        # 15-fps open unless their exact mode is selected first.
        if getattr(args, 'dct_encode', False):
            return camera_source(args.camera, args.capture_fps,
                                 width=args.capture_width, spec=args.ffmpeg_input,
                                 scale_flags=capture_filter, preserve_size=True)
        if getattr(args, 'perceptual_resize', 'off') == 'off':
            return camera_source(
                args.camera, args.capture_fps, width=args.capture_width,
                spec=args.ffmpeg_input, scale_flags=capture_filter,
                output_size=P.PREPARED_SIZE)
        return camera_source(args.camera, args.capture_fps,
                             width=args.capture_width, spec=args.ffmpeg_input,
                             scale_flags=capture_filter)
    if args.screen_backend == 'mss':
        return screen_source(region)
    capture_fps = args.capture_fps
    # AVFoundation screen capture commonly exposes only the display's native
    # refresh rate (for example 60 fps), not the modem wire rate.
    if capture_fps is None and sys.platform == 'darwin':
        capture_fps = 60
    return screen_capture_source(capture_fps or FPS, region, args.display,
                                 args.capture_width, args.ffmpeg_input,
                                 capture_filter,
                                 preserve_size=bool(getattr(args, 'dct_encode', False)))


def _capture_scale_flags(args):
    """Resolve FFmpeg's scaler, deriving the direct camera default from profile."""
    selected = getattr(args, 'capture_filter', None)
    if selected:
        return selected
    if (getattr(args, 'source', None) == 'camera' and
            getattr(args, 'perceptual_resize', 'off') == 'off'):
        encode_filter = getattr(args, 'encode_filter', None) or 'box'
        return ENCODE_TO_FFMPEG_SCALE[encode_filter]
    return 'neighbor'


def _model(fixture, encode_filter='nearest'):
    model = P.build_model(fixture, .1521 / np.sqrt(
        1 + 10**(P.CLOCK_REL_DB/10)), encode_filter=encode_filter)
    return model


def _values(model, frame, encode_filter='nearest', brightness=1.05, gamma=1.0,
            perceptual_resize='off', perceptual_detail_strength=0.25,
            dct_encode=False, dct_options=None, return_resized=False):
    _validate_tone_controls(brightness, gamma)
    source_size = getattr(frame, 'source_size', None)
    capture_prepared = bool(getattr(frame, 'prepared', False))
    if source_size is not None:
        frame = frame.rgb
    if dct_encode:
        if perceptual_resize != 'off':
            raise ValueError('--dct-encode cannot be combined with --perceptual-resize')
        if capture_prepared:
            raise ValueError('--dct-encode requires an unprepared source frame')
        from animation_modem.v7_source_dct import (direct_dct_values,
                                                   source_dct_values)
        rgb = np.asarray(frame.convert('RGB') if isinstance(frame, Image.Image)
                         else frame)
        options = dict(dct_options or {})
        if (options.pop('aggregation', 'off') == 'off' and
                options.pop('band_profile', 'off') == 'off'):
            # The specified direct encode: tone + block pre-shrink fused in
            # one pass, then small cached DCT products per plane.
            values = direct_dct_values(
                rgb, model.coder.grids, model.coder.shapes,
                brightness=brightness, gamma=gamma, **options)
        else:
            # Research reducers keep the full-resolution analysis path.
            values, _stats = source_dct_values(
                rgb, model.coder.grids, model.coder.shapes,
                brightness=brightness, gamma=gamma,
                **(dct_options or {}))
        size = source_size or (rgb.shape[1], rgb.shape[0])
        aspect = P.aspect_wire_code(size)
        if return_resized:
            # Direct source-DCT encoding has no resized RGB intermediate.
            # Show the toned source entering its color/DCT conversion rather
            # than reconstructing pixels from encoded coefficients.
            preview_rgb = np.asarray(rgb, dtype=np.float32)
            if preview_rgb.size and preview_rgb.max() > 1.0:
                preview_rgb = preview_rgb/255.0
            preview_rgb = np.clip(preview_rgb*brightness, 0.0, 1.0)
            if gamma != 1.0:
                preview_rgb = preview_rgb**(1.0/gamma)
            preview = Image.fromarray(
                np.uint8(np.rint(preview_rgb*255.0)), 'RGB')
            return values, aspect, preview
        return values, aspect
    rgb_frame = (isinstance(frame, np.ndarray) and frame.dtype == np.uint8 and
                 frame.ndim == 3 and frame.shape[2] == 3)
    if perceptual_resize == 'off':
        # Keep this default path byte-identical to the established sender.
        image = frame if isinstance(frame, Image.Image) else Image.fromarray(frame)
        prepared = (image.convert('RGB') if capture_prepared else
                    prepare_image(image, encode_filter=encode_filter))
        size = source_size or image.size
    else:
        if encode_filter != 'box':
            raise ValueError('perceptual resize requires --encode-filter box')
        from animation_modem.perceptual_resize import resize_image, resize_rgb
        if rgb_frame:
            # Captured frames are already RGB uint8 arrays. Going through
            # Pillow and back would copy them three times and change nothing.
            # Strided views (mss BGRA->RGB) are made contiguous once, so the
            # kernels see only the C layouts warmup_resize compiled.
            prepared = Image.fromarray(
                resize_rgb(np.ascontiguousarray(frame), perceptual_resize,
                           perceptual_detail_strength), mode='RGB')
            size = source_size or (frame.shape[1], frame.shape[0])
        else:
            image = (frame if isinstance(frame, Image.Image)
                     else Image.fromarray(frame))
            prepared = resize_image(image, perceptual_resize,
                                    perceptual_detail_strength)
            size = source_size or image.size
    if brightness != 1.0:
        prepared = ImageEnhance.Brightness(prepared).enhance(brightness)
    if gamma != 1.0:
        values = np.asarray(prepared, np.float32)/255.0
        values = np.clip(values, 0, 1)**(1.0/gamma)
        adjusted = Image.fromarray(np.uint8(np.rint(values*255)), 'RGB')
        adjusted.info.update(prepared.info)
        prepared = adjusted
    values = image_values(prepared, model.coder.grids,
                          encode_filter=encode_filter)
    aspect = P.aspect_wire_code(size)
    if return_resized:
        return values, aspect, prepared
    return values, aspect


def _experimental_fold(slots):
    """PROTOTYPE: the test_modem_v7 luma fold (see docs/transport_v7_spec.md
    section 10). Both ends load the same pinned table; coded pilot status
    identifies the fold size and supplies the timing estimator's chip signs."""
    if not slots:
        return None
    _ensure_test_modem_path()
    from live_fold import LiveFold
    try:
        fold = LiveFold(slots)
    except ValueError as exc:
        raise SystemExit(f'--experimental-fold {slots}: {exc}')
    print(f'[fold + coded pilot] {slots} luma slots, table {fold.digest} '
          f'(the other end must use the same profile)', flush=True)
    return fold


def _ensure_test_modem_path():
    path = str(ROOT/'test_modem_v7')
    if path not in sys.path:
        sys.path.insert(0, path)


def _fold_slots(args):
    """Resolve the shared live profile; the historical profile is explicit opt-out."""
    requested = getattr(args, 'experimental_fold', None)
    mono_off = bool(getattr(args, 'experimental_mono', False))
    mono_fold = bool(getattr(args, 'experimental_mono_fold', False))
    baseline = bool(getattr(args, 'baseline', False))
    if mono_off and mono_fold:
        raise ValueError('experimental mono profiles are mutually exclusive')
    if (mono_off or mono_fold) and (requested is not None or baseline):
        raise ValueError('experimental mono profiles must be selected on their own')
    if mono_off or mono_fold:
        return 0
    if baseline:
        if requested is not None:
            raise ValueError('--baseline cannot be combined with --experimental-fold')
        return 0
    return 500 if requested is None else int(requested)


def _send_profile(args, slots):
    """Apply profile defaults while preserving explicit user overrides."""
    encode_filter = getattr(args, 'encode_filter', None)
    brightness = getattr(args, 'brightness', None)
    perceptual_resize = getattr(args, 'perceptual_resize', 'off')
    dct_encode = bool(getattr(args, 'dct_encode', False))
    if slots:
        encode_filter = encode_filter or 'box'
        brightness = 1.0 if brightness is None else brightness
        if encode_filter != 'box':
            raise ValueError('folded coded-pilot mode requires --encode-filter box')
        if not P._is_reference(getattr(args, 'fixture', None)):
            if perceptual_resize != 'off':
                raise ValueError('--perceptual-resize requires the canonical V7 fixture and matching pinned fold table')
            raise ValueError('folded coded-pilot mode requires the default V7 fixture')
        if not getattr(args, 'pilot_tones', True):
            raise ValueError('coded-pilot folding cannot be combined with --no-pilot-tones')
    else:
        encode_filter = encode_filter or 'nearest'
        brightness = 1.05 if brightness is None else brightness
    _validate_tone_controls(brightness, getattr(args, 'gamma', 1.0))
    dct_options = _dct_encode_options(args)
    if dct_encode:
        if slots != 500 or encode_filter != 'box':
            raise ValueError('--dct-encode requires the canonical box Fold-500 profile')
        if not P._is_reference(getattr(args, 'fixture', None)):
            raise ValueError('--dct-encode requires the canonical V7 fixture')
        from animation_modem.v7_source_dct import (
            AGGREGATIONS, BAND_PROFILES, SHARPEN_MODES)
        if dct_options['sharpen'] not in SHARPEN_MODES:
            raise ValueError('unknown --dct-sharpen mode')
        if dct_options['aggregation'] not in AGGREGATIONS:
            raise ValueError('unknown --dct-aggregation mode')
        if dct_options['band_profile'] not in BAND_PROFILES:
            raise ValueError('unknown --dct-band-profile')
        if not 0 <= dct_options['sharpen_strength'] <= 1:
            raise ValueError('--dct-sharpen-strength must be in [0, 1]')
        if not 0 <= dct_options['clarity'] <= 1:
            raise ValueError('--dct-clarity must be in [0, 1]')
        if not 1 <= dct_options['chroma_gain'] <= 1.3:
            raise ValueError('--dct-chroma-gain must be in [1, 1.3]')
    elif _dct_options_explicit(args):
        raise ValueError('DCT enhancement options require --dct-encode')
    if perceptual_resize != 'off':
        if slots not in (500, 1000):
            raise ValueError('--perceptual-resize requires a pinned --experimental-fold 500 or 1000 profile')
        if encode_filter != 'box':
            raise ValueError('--perceptual-resize requires --encode-filter box')
        strength = float(getattr(args, 'perceptual_detail_strength', 0.25))
        if not np.isfinite(strength) or not 0.0 <= strength <= 1.0:
            raise ValueError('--perceptual-detail-strength must be finite and in [0, 1]')
    return encode_filter, float(brightness)


def _dct_encode_options(args):
    def option(name, default):
        configured = getattr(args, name, None)
        return default if configured is None else configured

    return {
        'sharpen': option('dct_sharpen', 'off'),
        'sharpen_strength': float(option('dct_sharpen_strength', .25)),
        'clarity': float(option('dct_clarity', 0.0)),
        'chroma_gain': float(option('dct_chroma_gain', 1.0)),
        'aggregation': option('dct_aggregation', 'off'),
        'band_profile': option('dct_band_profile', 'off'),
    }


def _dct_options_explicit(args):
    return any(getattr(args, name, None) is not None for name in (
        'dct_sharpen', 'dct_sharpen_strength', 'dct_clarity',
        'dct_chroma_gain', 'dct_aggregation', 'dct_band_profile'))


def _validate_tone_controls(brightness, gamma):
    """Reject tone controls that could turn a frame into invalid/black data."""
    if not np.isfinite(brightness) or brightness <= 0:
        raise ValueError('--brightness must be finite and positive')
    if not np.isfinite(gamma) or gamma <= 0:
        raise ValueError('--gamma must be finite and positive')


def _encode_pulse_frame_coeffs(model, coeffs, counter, aspect_code=0,
                               source_index=None, eof_marker=True,
                               pilot_values=None, pulse_profile_code=1):
    """Build one folded pulse packet without an inverse/forward DCT round trip."""
    if source_index is None:
        source_index = int(counter)-1
    return P.encode_pulse_frame_coeffs(
        model, np.asarray(coeffs), counter, aspect_code=aspect_code,
        source_index=source_index, pilot_tones=False,
        eof_marker=eof_marker, pilot_values=pilot_values,
        pulse_profile_code=pulse_profile_code)


def _add_coded_pilots(audio, start_counter, fold_slots):
    """Overlay each packet's fold-mode status and coded reference tones."""
    _ensure_test_modem_path()
    from tone_code import (FOLD_500, FOLD_1000, add_tone_code,
                           encode_status)

    modes = {500: FOLD_500, 1000: FOLD_1000}
    if fold_slots not in modes:
        raise ValueError(f'coded pilots support fold sizes {sorted(modes)}')
    audio = np.asarray(audio)
    if audio.ndim != 2 or audio.shape[1] != 2 or len(audio) % P.PULSE_FRAME:
        raise ValueError('coded-pilot audio must contain complete stereo V7 packets')
    packets = audio.reshape((-1, P.PULSE_FRAME, 2))
    status = encode_status(modes[fold_slots])
    if len(packets) == 1:
        return add_tone_code(packets[0], int(start_counter), status)
    return np.concatenate([
        add_tone_code(packet, int(start_counter)+index, status)
        for index, packet in enumerate(packets)])


def _coded_mode_matches_fold(result, fold):
    """A valid per-packet status may authorize only its matching pinned table."""
    _ensure_test_modem_path()
    from tone_code import FOLD_MODE_TO_SLOTS
    timing = result.diag.get('pilot_timing') or {}
    mode = timing.get('coded_status_mode')
    return mode is not None and FOLD_MODE_TO_SLOTS.get(mode) == fold.slots


def _mix_mono_video_audio(modem, frames, source, delay, video_side,
                          sample_rate, gain=1.0):
    """Put delayed source audio opposite the mono video carrier."""
    output = np.array(modem, dtype=np.float32, copy=True, order='C')
    if output.ndim != 2 or output.shape[1] != 2:
        raise ValueError('mono video plus source audio requires stereo output')
    frames = max(1, int(frames))
    audio_side = 1-int(video_side)
    base, extra = divmod(len(output), frames)
    offset = 0
    for frame in range(frames):
        count = base + (frame < extra)
        if count <= 0:
            continue
        samples = (source.read(count) if source is not None else
                   np.zeros(count, dtype=np.float32))
        samples = np.clip(np.asarray(samples, dtype=np.float32)*float(gain),
                          -1.0, 1.0)
        output[offset:offset+count, audio_side] = delay.apply(
            samples, count, sample_rate)
        offset += count
    return output


def _apply_profile_option(args):
    """Translate the sender GUI/CLI profile name to the wire flags."""
    profile = getattr(args, 'profile', None)
    if profile == 'mono-fold-500':
        args.experimental_mono_fold = True
    elif profile == 'mono-colour-500':
        args.experimental_mono_fold = True
        args.experimental_mono_colour = True
    elif profile == 'fold-500':
        args.experimental_fold = 500
    elif profile == 'fold-1000':
        args.experimental_fold = 1000


def run_send(args):
    import sounddevice as sd
    from tools.v7_device_recovery import device_identity

    try:
        identity = device_identity(sd, args.device, 'output')
        if not identity.get('name'):
            identity = None
    except Exception:
        identity = None
    args._sender_device_identity = identity
    args._sender_started_at = None
    args._sender_counter = 1
    args._sender_total = 0
    stop = threading.Event()
    try:
        while not stop.is_set():
            try:
                _run_send_session(args)
                return
            except SenderDeviceLost as exc:
                if identity is None:
                    raise RuntimeError(
                        f'Transmitter device was lost and its identity cannot '
                        f'be resolved for recovery: {exc}') from exc
                print(json.dumps({
                    'status': 'sender_device_lost',
                    'device': identity.get('name'),
                    'message': str(exc),
                }), flush=True)
                recovered = _wait_for_stable_send_device(
                    sd, args.device, identity, stop)
                if recovered is None:
                    return
                args.device, recovered_rate = recovered
                print(json.dumps({
                    'status': 'sender_device_reconnected',
                    'device': identity.get('name'),
                    'sample_rate': recovered_rate,
                }), flush=True)
    except KeyboardInterrupt:
        stop.set()
        if not args.no_log:
            print(f'V7 send stopped after interruption', flush=True)
    finally:
        stop.set()
        video_preview = getattr(args, '_video_preview', None)
        if video_preview is not None:
            video_preview.close()
            args._video_preview = None


def _run_send_session(args):
    import sounddevice as sd
    from tools.v7_capture import Throttled
    from tools.v7_device_recovery import (
        DEVICE_RATE_FRAME_INTERVAL, DeviceRateDebouncer,
        query_device_snapshot)

    _apply_profile_option(args)
    slots = _fold_slots(args)
    mono_profile = bool(getattr(args, 'experimental_mono', False))
    mono_fold_profile = bool(getattr(args, 'experimental_mono_fold', False))
    fold = _experimental_fold(slots)
    mono_wire = None
    if mono_profile or mono_fold_profile:
        if not getattr(args, 'pilot_tones', True):
            raise ValueError('experimental mono profiles require coded pilot tones')
        if not getattr(args, 'eof_marker', True):
            raise ValueError('experimental mono profiles require the EOF marker')
    if mono_fold_profile and getattr(args, 'mono_sum', False):
        raise ValueError('the mono-video side selector requires two output '
                         'channels; do not combine it with --mono-sum')
    source_audio_mode = getattr(args, 'source_audio', None)
    if mono_fold_profile:
        source_audio_mode = source_audio_mode or 'source'
        if (source_audio_mode == 'device' and
                getattr(args, 'source_audio_device', None) is None):
            raise ValueError('--source-audio device requires '
                             '--source-audio-device')
    elif source_audio_mode not in (None, 'off'):
        raise ValueError('source audio requires --experimental-mono-fold')
    audio_gain = float(getattr(args, 'source_audio_gain', 1.0))
    extra_audio_delay_ms = float(getattr(args, 'source_audio_delay_ms', 0.0))
    if not np.isfinite(audio_gain) or not 0.0 <= audio_gain <= 4.0:
        raise ValueError('--source-audio-gain must be between 0 and 4')
    if not np.isfinite(extra_audio_delay_ms) or extra_audio_delay_ms < 0:
        raise ValueError('--source-audio-delay-ms must be finite and non-negative')
    profile_slots = 500 if mono_fold_profile else slots
    args.encode_filter, args.brightness = _send_profile(args, profile_slots)
    dct_options = _dct_encode_options(args)
    tone_controls = LiveToneControls(args.brightness, args.gamma)
    control_stop = threading.Event()
    if getattr(args, 'gui_control', False):
        threading.Thread(
            target=_read_live_tone_controls,
            args=(sys.stdin, tone_controls, control_stop),
            name='v7-live-tone-controls', daemon=True).start()
    if getattr(args, 'perceptual_resize', 'off') != 'off':
        # Compile the optional Numba resize before an output stream is open;
        # first-call JIT latency must not stall the live sender.
        from animation_modem.perceptual_resize import warmup_resize
        warmup_resize((2, 2), args.perceptual_resize,
                      args.perceptual_detail_strength)
    model = _model(args.fixture, args.encode_filter)
    if mono_fold_profile:
        _ensure_test_modem_path()
        from mono_video import MonoColourFoldWire, MonoFreshFoldWire
        wire_class = (MonoColourFoldWire
                      if getattr(args, 'experimental_mono_colour', False)
                      else MonoFreshFoldWire)
        mono_wire = wire_class(
            model, side=getattr(args, 'mono_video_side', 'right'))
        from tone_code import warmup_status_templates
        warmup_status_templates(mono_wire.status_mode)
    elif mono_profile:
        _ensure_test_modem_path()
        from mono_wire import MonoWire
        mono_wire = MonoWire(model)
        from tone_code import warmup_status_templates, MONO_OFF
        warmup_status_templates(MONO_OFF)
    if fold is not None:
        # Fail before any audio: the table folds only the model it was built
        # for (the canonical box profile).
        fold.check(model)
        _ensure_test_modem_path()
        from tone_code import warmup_tone_templates
        warmup_tone_templates(fold.slots)
    requested_rate = getattr(args, 'rate', None)
    output_rate = None
    capture_hz = args.capture_fps or (CAMERA_CAPTURE_FPS
                                      if args.source == 'camera' else
                                      (60 if args.screen_backend == 'ffmpeg' and
                                       sys.platform == 'darwin' else FPS))
    wire_fps = FPS*args.speed
    raw_grab = None
    grab = None
    source_audio = None
    audio_delay = None
    video_preview = getattr(args, '_video_preview', None)
    batches = None
    stop = threading.Event()
    device_monitor_stop = threading.Event()
    device_monitor_thread = None
    device_lost = threading.Event()
    device_lost_reason = ['selected output device stopped']
    producer_done = threading.Event()
    prebuffer_ready = threading.Event()
    buffered_audio_seconds = 0.0
    sentinel = object()
    producer_errors = []
    image_preview_port = getattr(args, 'image_preview_port', None)
    if (image_preview_port is not None and
            not 1 <= int(image_preview_port) <= 65535):
        raise ValueError('--image-preview-port must be between 1 and 65535')
    preview_worker = None
    batch_size = max(1, args.batch_frames)
    total = getattr(args, '_sender_total', 0)
    started = getattr(args, '_sender_started_at', None)
    if started is None:
        started = time.monotonic()
        args._sender_started_at = started

    def encode_batch(frames, aspects, counter):
        values = np.asarray(frames)
        if mono_wire is not None:
            audio = mono_wire.encode(
                model, values, start_counter=counter,
                aspect_codes=aspects,
                source_indices=[counter+i-1 for i in range(len(values))],
                eof_marker=True)
        elif fold is not None:
            from tone_code import FOLD_500, FOLD_1000
            pulse_profile_code = {500: FOLD_500, 1000: FOLD_1000}[fold.slots]
            audio = np.concatenate([
                _encode_pulse_frame_coeffs(
                    model, fold.encode_coefficients(model, value),
                    counter+index, aspect_code=aspects[index],
                    source_index=counter+index-1,
                    eof_marker=getattr(args, 'eof_marker', True),
                    pulse_profile_code=pulse_profile_code)
                for index, value in enumerate(values)])
            audio = _add_coded_pilots(audio, counter, fold.slots)
        else:
            audio = P.encode_pulse_stream(
                model, values, start_counter=counter, aspect_codes=aspects,
                pilot_tones=getattr(args, 'pilot_tones', True),
                eof_marker=getattr(args, 'eof_marker', True))
        report_stats = args.log and not args.no_log
        if report_stats:
            encoded_peak = float(np.max(np.abs(audio))) if audio.size else 0.0
            encoded_rms = float(np.sqrt(np.mean(audio*audio))) if audio.size else 0.0
        limiter_gain = 1.0
        limiter_samples = 0
        if args.mono_sum:
            audio = (audio[:, :1].copy() if mono_wire is not None else
                     audio.sum(axis=1, keepdims=True)/np.sqrt(2))
            before_peak = float(np.max(np.abs(audio))) if audio.size else 0.0
            if report_stats:
                before_rms = float(np.sqrt(np.mean(audio*audio))) if audio.size else 0.0
        else:
            if report_stats:
                # The signal is unchanged, so reuse encoded measurements rather
                # than scanning the full packet again.
                before_peak, before_rms = encoded_peak, encoded_rms
        if args.mono_sum:
            peak = before_peak
            if peak > .89:
                limiter_gain = .89/float(peak)
                if report_stats:
                    limiter_samples = int(np.count_nonzero(np.abs(audio) > .89))
                audio *= limiter_gain
                if report_stats:
                    peak_after_limit = float(np.max(np.abs(audio)))
            elif report_stats:
                peak_after_limit = before_peak
        if not args.mono_sum and report_stats:
            peak_after_limit = before_peak
        if report_stats:
            stats = {
                'frames_encoded': len(frames),
                'encoded_peak': encoded_peak,
                'encoded_rms': encoded_rms,
                'peak_before_limit': before_peak,
                'peak_after_limit': peak_after_limit,
                'rms_before_limit': before_rms,
                'limiter_active': limiter_gain < 1.0,
                'limiter_gain': limiter_gain,
                'samples_limited': limiter_samples,
                'fold_slots': (fold.slots if fold is not None else
                               getattr(mono_wire, 'fold_slots', 0)),
                'coded_pilot': fold is not None or mono_wire is not None,
                'wire_profile': (getattr(mono_wire, 'wire_profile',
                                         'mono-fold-off')
                                 if mono_wire is not None else
                                 'folded' if fold is not None else 'baseline'),
                'mono_video_side': getattr(mono_wire, 'side', None),
            }
        else:
            stats = {'frames_encoded': len(frames)}
        output = P.speed_pulse_stream(audio, args.speed, rate=output_rate)
        if report_stats:
            stats.update({
                'emitted_peak': float(np.max(np.abs(output)))
                if output.size else 0.0,
                'emitted_rms': float(np.sqrt(np.mean(output*output)))
                if output.size else 0.0,
            })
        return output, stats

    def produce():
        nonlocal total, buffered_audio_seconds
        frames = []
        aspects = []
        preview_images = []
        counter = getattr(args, '_sender_counter', 1)
        first_audio_video_frame = (
            getattr(grab, 'first_frame', None) if source_audio is not None
            else None)
        next_capture = time.monotonic()
        failure = None
        try:
            while not stop.is_set() and (args.seconds <= 0 or
                                         time.monotonic()-started < args.seconds):
                delay = next_capture-time.monotonic()
                if delay > 0:
                    time.sleep(delay)
                frame = first_audio_video_frame
                first_audio_video_frame = None
                if frame is None:
                    frame = grab()
                if frame is None and getattr(grab, 'ended', False):
                    break
                current_tones = tone_controls.snapshot()
                source_preview = (getattr(frame, 'rgb', frame)
                                  if image_preview_port is not None else None)
                processed = _values(
                    model, frame, args.encode_filter,
                    current_tones['brightness'], current_tones['gamma'],
                    getattr(args, 'perceptual_resize', 'off'),
                    getattr(args, 'perceptual_detail_strength', 0.25),
                    dct_encode=getattr(args, 'dct_encode', False),
                    dct_options=dct_options,
                    return_resized=image_preview_port is not None)
                if image_preview_port is not None:
                    value, aspect, resized_preview = processed
                    preview_images.append((source_preview, resized_preview))
                else:
                    value, aspect = processed
                # Keep captured source values unfolded. encode_batch folds
                # exactly once, directly in coefficient space; a second fold
                # would quantize the hosts again and erase the guest residuals.
                frames.append(value); aspects.append(aspect)
                # A compressed packet still needs a new source frame at the
                # faster wire cadence; otherwise the audio stream has gaps.
                next_capture += 1/wire_fps
                if len(frames) < batch_size:
                    continue
                audio, stats = encode_batch(frames, aspects, counter)
                preview_frames = (
                    tuple((counter+index, aspects[index], frame)
                          for index, frame in enumerate(preview_images))
                    if image_preview_port is not None else ())
                batches.put((counter, audio, stats, preview_frames))
                buffered_audio_seconds += len(audio)/output_rate
                if buffered_audio_seconds >= startup_buffer_seconds:
                    prebuffer_ready.set()
                total += len(frames)
                counter += len(frames)
                args._sender_counter = counter
                args._sender_total = total
                frames = []
                aspects = []
                preview_images = []
        except Exception as exc:
            failure = exc

        if failure is None and frames and not stop.is_set():
            try:
                audio, stats = encode_batch(frames, aspects, counter)
                preview_frames = (
                    tuple((counter+index, aspects[index], frame)
                          for index, frame in enumerate(preview_images))
                    if image_preview_port is not None else ())
                batches.put((counter, audio, stats, preview_frames))
                buffered_audio_seconds += len(audio)/output_rate
                if buffered_audio_seconds >= startup_buffer_seconds:
                    prebuffer_ready.set()
                total += len(frames)
                args._sender_counter = counter+len(frames)
                args._sender_total = total
            except Exception as exc:
                failure = exc

        close = getattr(grab, 'close', None)
        try:
            if close is not None:
                close()
        except Exception as exc:
            if failure is None:
                failure = exc
        if failure is not None:
            producer_errors.append(failure)
        producer_done.set()
        batches.put(sentinel)

    worker = threading.Thread(target=produce, daemon=True)
    worker_started = False
    try:
        stream_args = {
            'channels': 1 if args.mono_sum else 2,
            'dtype': 'float32',
            'device': args.device,
            'blocksize': 0,
        }
        if requested_rate is not None:
            stream_args['samplerate'] = requested_rate
        with sd.OutputStream(**stream_args) as stream:
            # With no --rate, sounddevice opens the DAC at its native clock.
            # The producer must wait until that clock is known so its packet
            # resampling preserves 1x playback speed on any supported device.
            output_rate = float(stream.samplerate)
            if getattr(args, 'preview', False) and video_preview is None:
                from tools.v7_video_preview import launch_video_preview
                video_preview = launch_video_preview(
                    args.video_source,
                    live=True if getattr(args, 'video_live', False) else None)
                args._video_preview = video_preview
                if video_preview.warning:
                    print({'status': 'video_preview_note',
                           'message': video_preview.warning}, flush=True)
            if (not np.isfinite(args.speed) or
                    not P.MIN_PLAYBACK_SPEED <= args.speed <= P.MAX_PLAYBACK_SPEED):
                raise ValueError(
                    f'--speed must be between {P.MIN_PLAYBACK_SPEED:g} and '
                    f'{P.MAX_PLAYBACK_SPEED:g} (higher speeds may lose high-frequency detail)')
            first_packet_samples = len(P.speed_pulse_stream(
                np.zeros(P.PULSE_FRAME, dtype=np.float32), args.speed,
                rate=output_rate))
            # Keep exactly one emitted packet as startup/queue headroom at the
            # actual DAC rate and speed. Packet duration shrinks at fast speed.
            startup_buffer_seconds = first_packet_samples/output_rate
            queue_batches = _sender_queue_batches(
                startup_buffer_seconds, output_rate, first_packet_samples)
            batches = queue.Queue(maxsize=queue_batches)
            if image_preview_port is not None:
                try:
                    from tools.v7_image_preview import ImagePreviewWorker
                    preview_worker = ImagePreviewWorker(image_preview_port)
                    preview_worker.start()
                except Exception as exc:
                    preview_worker = None
                    print(json.dumps({
                        'status': 'image_preview_unavailable',
                        'message': str(exc),
                    }), flush=True)

            if getattr(args, '_sender_device_identity', None) is not None:
                def monitor_output_device():
                    rate_stability = DeviceRateDebouncer()
                    while not device_monitor_stop.wait(
                            DEVICE_RATE_FRAME_INTERVAL):
                        try:
                            if not getattr(stream, 'active', True):
                                device_lost_reason[0] = (
                                    'PortAudio output stream stopped')
                                device_lost.set()
                                return
                            snapshot = query_device_snapshot(
                                sd, args.device, args._sender_device_identity,
                                'output')
                        except Exception as exc:
                            device_lost_reason[0] = str(exc)
                            device_lost.set()
                            return
                        resolved_index, _info, observed_identity, rate = snapshot
                        if (isinstance(args.device, int) and
                                resolved_index != args.device):
                            device_lost_reason[0] = (
                                'PortAudio device index changed')
                            device_lost.set()
                            return
                        if (requested_rate is None and
                                abs(rate-output_rate) > .5):
                            if rate_stability.observe(observed_identity, rate):
                                device_lost_reason[0] = (
                                    f'output sample rate changed from '
                                    f'{output_rate:g}Hz to {rate:g}Hz')
                                device_lost.set()
                                return
                        else:
                            rate_stability.reset()

                device_monitor_thread = threading.Thread(
                    target=monitor_output_device, daemon=True,
                    name='v7-send-output-monitor')
                device_monitor_thread.start()
            # Compile the per-frame image and pulse encoder path before the
            # first real packet. Numba's first-call work must not become a gap
            # in the recorded modem waveform.
            warm_frame = np.zeros(
                (P.PREPARED_SIZE[1], P.PREPARED_SIZE[0], 3), dtype=np.uint8)
            warm_tones = tone_controls.snapshot()
            warm_values, _ = _values(
                model, warm_frame, args.encode_filter,
                warm_tones['brightness'], warm_tones['gamma'],
                getattr(args, 'perceptual_resize', 'off'),
                getattr(args, 'perceptual_detail_strength', 0.25),
                dct_encode=getattr(args, 'dct_encode', False),
                dct_options=dct_options)
            encode_batch([warm_values], [0], 1)
            # Start picture and soundtrack capture only after the output clock
            # is known, and close together so file/stream timelines begin near
            # the same source time.
            if (mono_fold_profile and source_audio_mode == 'source' and
                    args.source == 'video' and hasattr(os, 'mkfifo')):
                from tools.v7_source_audio import SharedVideoAudioSource
                capture = SharedVideoAudioSource(
                    args.video_source, output_rate, width=args.capture_width,
                    scale_flags=_capture_scale_flags(args),
                    live=True if args.video_live else None,
                    target_samples=first_packet_samples,
                    preserve_size=getattr(args, 'dct_encode', False))
                raw_grab = capture.video_grab
                source_audio = capture if capture.has_audio else None
            else:
                raw_grab = _capture(args)
            if (mono_fold_profile and source_audio_mode == 'source' and
                    args.source == 'video' and not hasattr(os, 'mkfifo')):
                from tools.v7_source_audio import FFmpegSourceAudio
                source_audio = FFmpegSourceAudio(
                    args.video_source, output_rate,
                    live=True if args.video_live else None,
                    target_samples=first_packet_samples)
            elif mono_fold_profile and source_audio_mode == 'device':
                from tools.v7_source_audio import DeviceSourceAudio

                def source_audio_status(error):
                    print(json.dumps({
                        'status': ('sender_source_audio_lost' if error else
                                   'sender_source_audio_restored'),
                        'device': str(args.source_audio_device),
                        'message': error,
                    }), flush=True)

                try:
                    source_audio = DeviceSourceAudio(
                        args.source_audio_device, output_rate,
                        input_side=getattr(
                            args, 'source_audio_input_side', 'mix'),
                        target_samples=first_packet_samples,
                        status_callback=source_audio_status)
                except Exception as exc:
                    source_audio = None
                    source_audio_status(str(exc))
            if mono_fold_profile:
                from tools.v7_source_audio import PacketAudioDelay
                audio_delay = PacketAudioDelay(extra_audio_delay_ms)
            # Drain paced FFmpeg capture continuously and expose the newest
            # frame; reading once per encoded frame creates stale-video lag.
            grab = Throttled(raw_grab, capture_hz)
            if hasattr(grab, 'retune'):
                grab.retune(wire_fps)
            if source_audio is not None:
                # Preserve the soundtrack beginning by waiting for up to one
                # packet of samples. No-track sources end immediately; startup
                # underflow remains zero-filled rather than discarding audio.
                source_audio.wait_for_samples(
                    first_packet_samples,
                    first_packet_samples/output_rate+.2)
            worker.start()
            worker_started = True
            # Start playback only after a small encoded cushion exists. This
            # absorbs transient frame/encoder stalls; the bounded queue limits
            # latency and memory if capture runs faster than the output clock.
            while (not prebuffer_ready.is_set() and
                   not producer_done.wait(.02)):
                if stop.is_set():
                    break
            if not args.no_log:
                camera_text = (f'camera={args.ffmpeg_input or args.camera} '
                               if args.source == 'camera' else '')
                if (args.source == 'camera' and
                        getattr(args, 'perceptual_resize', 'off') == 'off'):
                    capture_text = f'80x96/{_capture_scale_flags(args)}'
                elif (args.source == 'screen' and
                      args.screen_backend == 'mss') or args.source in (
                          'mouse-follow', 'test'):
                    capture_text = 'native'
                else:
                    capture_text = (f'{args.capture_width}px/'
                                    f'{_capture_scale_flags(args)}')
                wire_mode = (getattr(mono_wire, 'wire_profile',
                                     'mono-fold-off')
                             if mono_wire is not None else
                             'mono-sum' if args.mono_sum else 'M/S')
                if mono_fold_profile:
                    audio_side = 'right' if mono_wire.side == 'left' else 'left'
                    audio_label = source_audio_mode
                    if source_audio_mode == 'source' and source_audio is None:
                        audio_label += '/no-track'
                    if source_audio_mode == 'device':
                        audio_label += (
                            f'/{getattr(args, "source_audio_input_side", "mix")}')
                    wire_mode += (f' video={mono_wire.side} '
                                  f'audio={audio_label}->{audio_side}')
                print(f'V7 send ready: source={args.source} device={args.device!r} '
                       f'rate={output_rate:g}Hz wire={wire_fps:.3f}fps '
                       f'speed={args.speed:g}x {camera_text}'
                        f'pilot-tones={"on" if getattr(args, "pilot_tones", False) else "off"} '
                         f'capture={capture_text} '
                         f'encode={args.encode_filter} mode={wire_mode}',
                        flush=True)

            def write_output(audio):
                if device_lost.is_set():
                    raise SenderDeviceLost(device_lost_reason[0])
                try:
                    stream.write(np.ascontiguousarray(audio, dtype=np.float32))
                except Exception as exc:
                    raise SenderDeviceLost(
                        f'output write failed: {exc}') from exc

            while True:
                if device_lost.is_set():
                    raise SenderDeviceLost(device_lost_reason[0])
                try:
                    item = batches.get(timeout=DEVICE_RECOVERY_POLL_SECONDS)
                except queue.Empty:
                    continue
                if item is sentinel:
                    break
                counter, audio, stats, preview_frames = item
                if mono_fold_profile:
                    frames = max(1, int(stats['frames_encoded']))
                    base, extra = divmod(len(audio), frames)
                    offset = 0
                    for index in range(frames):
                        count = base + (index < extra)
                        packet = _mix_mono_video_audio(
                            audio[offset:offset+count], 1, source_audio,
                            audio_delay, mono_wire.carrier_index, output_rate,
                            audio_gain)
                        # One packet per blocking write lets paced soundtrack
                        # capture advance while the DAC plays this video frame.
                        write_output(packet)
                        if preview_worker is not None:
                            handoff_ns = time.monotonic_ns()
                            frame_counter, aspect, values = preview_frames[index]
                            source_image, resized_image = values
                            preview_worker.submit(
                                frame_counter, aspect, source_image,
                                resized_image, handoff_ns)
                        offset += count
                else:
                    # sounddevice requires a C-contiguous interleaved buffer;
                    # filtering/resampling can return a strided view here.
                    write_output(audio)
                    if preview_worker is not None:
                        handoff_ns = time.monotonic_ns()
                        for frame_counter, aspect, images in preview_frames:
                            source_image, resized_image = images
                            preview_worker.submit(
                                frame_counter, aspect, source_image,
                                resized_image, handoff_ns)
                if args.log and not args.no_log:
                    print({
                        'sent_through_frame': (
                            counter+stats['frames_encoded']-1),
                        **{key: value for key, value in stats.items()
                           if key != 'frames_encoded'},
                    }, flush=True)
    except KeyboardInterrupt:
        raise
    finally:
        device_monitor_stop.set()
        if device_monitor_thread is not None:
            device_monitor_thread.join(timeout=1)
        control_stop.set()
        stop.set()
        if worker_started:
            deadline = time.monotonic()+2
            while worker.is_alive() and time.monotonic() < deadline:
                try:
                    batches.get_nowait()
                except queue.Empty:
                    pass
                worker.join(timeout=.02)
        else:
            close = getattr(grab, 'close', None)
            if close is None:
                close = getattr(raw_grab, 'close', None)
            if close is not None:
                close()
        if preview_worker is not None:
            preview_worker.close()
        if source_audio is not None:
            source_audio.close()
            if args.log and not args.no_log:
                buffer = getattr(source_audio, 'buffer', None)
                clock = getattr(source_audio, 'clock_match', None)
                print({
                    'source_audio_dropped_samples': getattr(buffer, 'dropped', 0),
                    'source_audio_max_clock_correction_ppm': round(
                        getattr(clock, 'max_correction', 0.0)*1_000_000, 1),
                }, flush=True)
    if producer_errors:
        failure = producer_errors[0]
        raise RuntimeError(f'V7 live producer failed: {failure}') from failure
    if args.log and not args.no_log:
        print(f'V7 send stopped after {total} frames', flush=True)


# Capture at the device's own rate, like the sender's PacketOutput: forcing
# 48 kHz made PortAudio resample behind our back, and on a 96 kHz device a 2x
# wire (carriers up to 25.5 kHz) lost its top carriers to that converter.  The
# decoder uses the actual capture rate to normalize the frame scale measured
# from each preamble. The raw sample scale is capture rate/(48 kHz x speed),
# so a higher-rate capture provides more timing samples without changing the
# accepted playback-speed range. Above 96 kHz capture is capped at 96 kHz.
MAX_CAPTURE_RATE = 96_000


def capture_rate_for(device_info):
    rate = float(device_info.get('default_samplerate') or P.RATE)
    return min(rate, MAX_CAPTURE_RATE)


class _MonoChannelProbe:
    """Pulse-only monitor for the non-selected receiver leg."""

    MIN_CONFIDENCE = .75

    def __init__(self, rate, direction='auto', decode_history=1,
                 decode_batch=1):
        self.input = LiveInput(decode_history, decode_batch, rate=rate,
                               direction=direction)
        self.last_position = None
        self.last_scale = None
        self.last_direction = None
        self.streak = 0
        self.status_mode = None
        self.status_candidate = None
        self.status_streak = 0
        self.last_valid = None
        self.status_last_seen = None
        self.status_scale = None

    def add(self, block):
        self.input.add(block)

    def reset(self):
        self.input.reset()
        self.last_position = None
        self.last_scale = None
        self.last_direction = None
        self.streak = 0
        self.status_mode = None
        self.status_candidate = None
        self.status_streak = 0
        self.last_valid = None
        self.status_last_seen = None
        self.status_scale = None

    def _reset_status(self):
        self.status_mode = None
        self.status_candidate = None
        self.status_streak = 0
        self.last_valid = None
        self.status_last_seen = None
        self.status_scale = None

    def _reset_status_if_stale(self, now):
        if (self.status_candidate is None or
                self.status_last_seen is None or
                self.status_scale is None):
            return
        timeout = (1.5*P.PULSE_FRAME*self.status_scale/self.input.rate)
        if float(now)-self.status_last_seen > timeout:
            self._reset_status()

    def _observe_status(self, mode, scale, now):
        """Require consecutive, distinct pulse words before routing a leg."""
        if mode is None:
            self._reset_status()
            return
        mode, scale, now = int(mode), float(scale), float(now)
        if mode == self.status_candidate:
            self.status_streak += 1
        else:
            self.status_candidate = mode
            self.status_streak = 1
        self.status_last_seen = now
        self.status_scale = scale
        if self.status_streak >= 2:
            self.status_mode = mode
            self.last_valid = now

    def scan(self, now):
        self._reset_status_if_stale(now)
        audio = self.input.take(now)
        if audio is None:
            return self.streak
        audio_start = self.input.total-len(audio)
        profile_by_position = {
            int(round(position)): mode
            for position, mode in self.input.pulse_profile_hits(audio)}
        for position, scale, confidence, direction in self.input.pulse_hits(audio):
            absolute = int(round(audio_start+position))
            if (self.last_position is not None and
                    absolute <= self.last_position):
                continue
            if confidence < self.MIN_CONFIDENCE:
                self.last_position = absolute
                self._reset_status()
                continue
            mode = profile_by_position.get(int(round(position)))
            self._observe_status(mode, scale, now)
            previous_position = self.last_position
            if previous_position is None:
                self.streak = 1
            else:
                expected = P.PULSE_FRAME*.5*(scale+self.last_scale)
                tolerance = max(3*scale, .15*expected)
                if (direction == self.last_direction and
                        abs((absolute-previous_position)-expected) <= tolerance):
                    self.streak = min(self.streak+1, 8)
                else:
                    self.streak = 1
            self.last_position = absolute
            self.last_scale = scale
            self.last_direction = direction
        self.input.decoded()
        return self.streak


class _ProfileStatusProbe:
    """Find complete coded-profile packets independently on one input leg."""

    MIN_CONFIDENCE = .45

    def __init__(self, rate, side_index, direction='auto', decode_history=1,
                 decode_batch=1):
        self.input = LiveInput(decode_history, decode_batch, rate=rate,
                               direction=direction)
        self.side_index = int(side_index)
        self.last_position = None
        self.audio = None

    def add(self, block):
        self.input.add(block)

    def reset(self):
        self.input.reset()
        self.last_position = None
        self.audio = None

    def scan(self, now):
        """Return one status observation per distinct complete packet."""
        self.audio = self.input.take(now)
        if self.audio is None:
            return ()
        audio_start = self.input.total-len(self.audio)
        profile_by_position = {
            int(round(position)): mode
            for position, mode in self.input.pulse_profile_hits(self.audio)}
        events = []
        for position, scale, confidence, direction in self.input.pulse_hits(
                self.audio):
            absolute = int(round(audio_start+position))
            if (self.last_position is not None and
                    absolute <= self.last_position):
                continue
            # A header arrives before the profile chips. Leave it pending until
            # its complete packet region is in the rolling input buffer.
            if position+P.PULSE_FRAME*scale > len(self.audio)+1.0:
                continue
            if confidence < self.MIN_CONFIDENCE:
                self.last_position = absolute
                events.append({'position': absolute, 'scale': scale,
                               'side_index': self.side_index, 'mode': None})
                continue
            mode = profile_by_position.get(int(round(position)))
            self.last_position = absolute
            events.append({'position': absolute, 'scale': scale,
                           'side_index': self.side_index, 'mode': mode})
        self.input.decoded()
        return tuple(events)


class _AdaptiveProfileDecoder:
    """Dispatch coded Fold-500 and mono-video packets after three confirmations."""

    REQUIRED_STREAK = 3

    def __init__(self, fold, base_model, preferred_side='auto'):
        from animation_modem import v7
        _ensure_test_modem_path()
        from live_fold import LiveFold
        from mono_video import (MonoColourFoldWire, MonoFreshFoldWire,
                                mono_channel_profile)
        from tone_code import FOLD_500, MONO_500, MONO_1000

        self.fold = fold
        self.v7 = v7
        self.base_model = base_model
        self.preferred_side = preferred_side
        self.input_channels = 2
        self.mono_wires = {
            MONO_500: MonoFreshFoldWire(base_model, side='both'),
            MONO_1000: MonoColourFoldWire(base_model, side='both'),
        }
        self.mono_channel_profile = mono_channel_profile
        self.mono_status_modes = frozenset(self.mono_wires)
        self.supported_modes = frozenset((FOLD_500, *self.mono_wires))
        self._mode_names = {
            FOLD_500: 'fold-500',
            MONO_500: 'mono-fold-500',
            MONO_1000: 'mono-colour-500',
        }
        self.active_mode = FOLD_500
        self.active_side = None
        self.candidate = None
        self.streak = 0
        self.candidate_last_seen = None
        self.last_packet = None
        self.last_decoded_packet = None
        self.generation = 0
        self.capture_rate = v7.RATE
        self.candidate_scale = None
        self.state = None
        self.stereo_tail_memory = True
        self.dispatch_mode = None
        self.dispatch_side = None
        self._local = threading.local()
        self._installed = None
        # LiveFold is also the equalizer-output capture hook used by mono
        # profiles; the selected table is applied later, after status dispatch.
        if not isinstance(fold, LiveFold):
            raise TypeError('adaptive V7 profile dispatch requires LiveFold')

    @property
    def profile_name(self):
        return self._mode_names[self.active_mode]

    @property
    def candidate_name(self):
        if self.candidate is None:
            return None
        return self._mode_names[self.candidate[0]]

    def bind_state(self, state):
        self.state = state
        self.stereo_tail_memory = bool(state.tail.enabled)
        if self.active_mode in self.mono_status_modes:
            state.tail.enabled = False

    def reset_candidate(self):
        self.candidate = None
        self.streak = 0
        self.candidate_last_seen = None
        self.candidate_scale = None

    def reset_capture_timeline(self):
        """Forget capture-sample positions after the input stream reopens.

        Packet positions count samples from the start of one input stream. A
        reopened stream counts from zero again, so positions kept from the
        previous stream would reject every new packet as already seen.
        """
        self.reset_candidate()
        self.last_packet = None
        self.last_decoded_packet = None

    def reset_candidate_if_stale(self, now, timeout):
        packet_seconds = (
            1.5*P.PULSE_FRAME*self.candidate_scale/self.capture_rate
            if self.candidate_scale is not None else 0.0)
        timeout = max(float(timeout), packet_seconds)
        if (self.candidate is not None and
                self.candidate_last_seen is not None and
                float(now)-self.candidate_last_seen > max(0.0, timeout)):
            self.reset_candidate()

    def observe_packets(self, events, now=None):
        """Group leg observations and update the confirmed packet profile."""
        if not events:
            return ()
        now = time.monotonic() if now is None else float(now)
        grouped = []
        for event in sorted(events, key=lambda item: item['position']):
            group = next((item for item in reversed(grouped)
                          if abs(event['position']-item['position']) <=
                          max(32.0, .25*P.PULSE_FRAME*min(
                              event['scale'], item['scale']))), None)
            if group is None:
                group = {'position': event['position'],
                         'scale': event['scale'], 'events': []}
                grouped.append(group)
            group['events'].append(event)

        decisions = []
        for group in grouped:
            position = int(round(group['position']))
            scale = float(group['scale'])
            if (self.last_packet is not None and
                    position <= self.last_packet+
                    .5*P.PULSE_FRAME*scale):
                continue
            self.last_packet = position
            channel_modes = [None, None]
            observed = {int(event['mode']) for event in group['events']
                        if event['mode'] is not None}
            for event in group['events']:
                if event['mode'] is not None:
                    channel_modes[event['side_index']] = int(event['mode'])
            mode = next(iter(observed)) if len(observed) == 1 else None
            if mode not in self.supported_modes:
                mode = None
            side = None
            if mode in self.mono_status_modes:
                sides = [event['side_index'] for event in group['events']
                         if event['mode'] == mode]
                preferred_index = {'left': 0, 'right': 1}.get(
                    self.preferred_side)
                if (preferred_index is not None and
                        (preferred_index in sides or
                         self.input_channels == 1 and 0 in sides)):
                    side = (0 if self.input_channels == 1 else
                            preferred_index)
                elif preferred_index is None:
                    side = (1 if 1 in sides else sides[0]) if sides else None
                else:
                    mode = None

            decision = self._observe(position, scale, mode, side, now)
            decision['channel_modes'] = tuple(channel_modes)
            decisions.append(decision)
        return tuple(decisions)

    def _observe(self, position, scale, mode, side, now):
        changed = False
        confirmed = False
        if mode is None:
            self.reset_candidate()
        else:
            key = (mode, side if mode in self.mono_status_modes else None)
            if mode == self.active_mode:
                self.reset_candidate()
                if mode in self.mono_status_modes and side is not None:
                    self.active_side = side
                confirmed = True
            else:
                if key == self.candidate:
                    self.streak += 1
                else:
                    self.candidate = key
                    self.streak = 1
                self.candidate_last_seen = now
                self.candidate_scale = scale
                if self.streak >= self.REQUIRED_STREAK:
                    self.active_mode, self.active_side = key
                    self.reset_candidate()
                    self.generation += 1
                    changed = confirmed = True
                    if self.state is not None:
                        self.state.tail.reset()
                        self.state.tail.enabled = (
                            self.stereo_tail_memory and self.active_mode not in
                            self.mono_status_modes)
                        self.state.last_verified = None
        return {
            'position': position, 'scale': scale, 'mode': mode,
            'side_index': side, 'confirmed': confirmed,
            'switched': changed, 'active_mode': self.active_mode,
            'active_side': self.active_side,
        }

    def claim(self, decision):
        """Prevent a rolling input window from decoding the same status twice."""
        if decision is None or not decision.get('confirmed'):
            return False
        position = int(decision['position'])
        span = P.PULSE_FRAME*float(decision['scale'])
        if (self.last_decoded_packet is not None and
                position <= self.last_decoded_packet+.5*span):
            return False
        self.last_decoded_packet = position
        self.dispatch_mode = int(decision['mode'])
        self.dispatch_side = decision['side_index']
        return True

    def install(self):
        if self._installed is not None:
            return
        self.fold.install()
        real_decode = self.v7.decode_frame
        real_numba = self.v7._equalize_numba
        real_numpy = self.v7._equalize_numpy
        local = self._local

        def equalize_numba(model, Z, H, noise, counter):
            result = real_numba(model, Z, H, noise, counter)
            local.equalized = (result[0].copy(), result[1].copy())
            return result

        def equalize_numpy(model, Z, H, noise, counter,
                           force_float32=False):
            result = real_numpy(model, Z, H, noise, counter,
                                force_float32=force_float32)
            local.equalized = (result[0].copy(), result[1].copy())
            return result

        self._real_decode = real_decode
        self._real_equalizers = (real_numba, real_numpy)
        self.v7._equalize_numba, self.v7._equalize_numpy = (
            equalize_numba, equalize_numpy)
        self.v7.decode_frame = self._decode_frame
        self._installed = True

    def uninstall(self):
        if self._installed is None:
            return
        self.v7.decode_frame = self._real_decode
        self.v7._equalize_numba, self.v7._equalize_numpy = (
            self._real_equalizers)
        self._installed = None
        self.fold.uninstall()

    def _decode_frame(self, model, x, tmap, counter, prev_tail, *args,
                      **kwargs):
        from mono_wire import MonoWire

        def held(reason, observed_mode=None):
            if self.state is not None:
                self.state.last_verified = None
            return self.v7.Result(
                counter, 'lost', np.asarray(prev_tail).copy(),
                {'displayable': False, 'held': True,
                 'coded_status_mode': observed_mode,
                 'profile_rejected': reason})

        mode = self.dispatch_mode
        body = kwargs.get('direct_body')
        observed_mode = None
        if body is not None:
            try:
                status = MonoWire._status_for_body(
                    model, body, kwargs.get('force_float32', False))
                if status and status.get('valid') and status.get('status'):
                    observed_mode = int(status['status']['mode'])
            except (ValueError, FloatingPointError, np.linalg.LinAlgError):
                observed_mode = None
        if (mode is None or observed_mode != mode or mode != self.active_mode):
            return held('status_not_confirmed_for_active_profile',
                        observed_mode)

        profile_model = model
        if mode in self.mono_status_modes:
            try:
                profile_model = self.mono_wires[mode].model_for(model)
            except ValueError:
                return held('unsupported_model_for_profile', observed_mode)
            # Mono layouts have no cross-packet tail memory. The outer pulse
            # loop owns the shared store, so give this decoder its own prior.
            prev_tail = profile_model.mu
        self._local.equalized = None
        if mode in self.mono_status_modes:
            with self.mono_channel_profile():
                result = self._real_decode(
                    profile_model, x, tmap, counter, prev_tail, *args,
                    **kwargs)
        else:
            result = self._real_decode(
                profile_model, x, tmap, counter, prev_tail, *args, **kwargs)
        if result is not None:
            result.diag['coded_status_mode'] = mode
            result.diag['profile_mode'] = mode
            result.diag['wire_profile'] = self._mode_names[mode]
            if mode in self.mono_status_modes and self._local.equalized is not None:
                result.diag['mono_fold_eq'] = self._local.equalized
        return result

    def values(self, model, result):
        mode = result.diag.get('profile_mode')
        if mode == 1:
            return self.fold.values(model, result, metadata_confirmed=True)
        if mode in self.mono_status_modes:
            wire = self.mono_wires[mode]
            return wire.values(wire.model_for(model), result)
        return self.v7.values_from(model, result.coeffs)


def _mono_packet_status_mode(result):
    """Coded status mode carried by a decoded mono-profile candidate."""
    if result is None:
        return None
    diag = getattr(result, 'diag', {}) or {}
    timing = diag.get('pilot_timing') or {}
    return timing.get('coded_status_mode', diag.get('coded_status_mode'))


def _coded_status_mode(samples, frame_start, frame_scale, sample_rate,
                       direction=1):
    """Read the in-band profile code from one pulse-anchored packet."""
    _ensure_test_modem_path()
    from tone_code import decode_tone_code

    audio = np.asarray(samples)
    start = float(frame_start)
    scale = float(frame_scale)
    if direction < 0:
        region = P.reverse_packet_region(audio, start, scale)
        if region is None:
            return None
        audio, region_start = region
        start = len(audio)+region_start-(float(frame_start)+
                                         P.PULSE_FRAME*scale)
    status = decode_tone_code(
        audio, frame_start=start, frame_scale=scale,
        sample_rate=sample_rate)
    if not status.get('valid') or status.get('status') is None:
        return None
    return int(status['status']['mode'])


def _mono_fold_input_side(channel_modes, mono_mode):
    """Choose a leg only when its observed coded statuses consistently agree.

    Require two valid packet statuses on a leg. A missing/damaged status is
    ignored, while a conflicting valid status makes that leg ambiguous. Right
    wins when both input legs independently validate the mono-video profile.
    """
    valid = []
    for index, modes in enumerate(channel_modes):
        modes = tuple(modes)
        if len(modes) >= 2 and all(mode == mono_mode for mode in modes):
            valid.append(index)
    if not valid:
        return None
    selected = 1 if 1 in valid else valid[0]
    return 'right' if selected == 1 else 'left'


def _detect_mono_fold_side(args, timeout=2.0, wait_for_signal=False):
    """Return (side, mode) for a valid mono status, or (None, None).

    A short startup pass lets the ordinary receiver choose its model and rank
    map before it opens the live decode path. If neither mono status validates,
    the caller keeps the ordinary stereo Fold-500 profile. The GUI can wait for
    the first non-silent block before starting its profile-detection timeout, so
    it can be opened before its sender.
    """
    import sounddevice as sd
    _ensure_test_modem_path()
    from tone_code import MONO_500, MONO_1000

    device_info = sd.query_devices(args.device, 'input')
    channels = 1 if int(device_info.get('max_input_channels') or 0) < 2 else 2
    rate = capture_rate_for(device_info)
    direction = getattr(args, 'direction', 'auto')
    inputs = [LiveInput(1, 1, rate=rate, direction=direction)
              for _ in range(channels)]
    observed_modes = [[] for _ in range(channels)]
    last_positions = [set() for _ in range(channels)]
    blocks = queue.Queue(maxsize=32)
    stop = getattr(args, 'stop_event', None)

    def confirmed_profile():
        for candidate in (MONO_1000, MONO_500):
            side = _mono_fold_input_side(observed_modes, candidate)
            if side is not None:
                return side, candidate
        return None, None

    def callback(indata, _frames, _timing, status):
        if status and not args.no_log:
            print(f'profile probe input: {status}', file=sys.stderr, flush=True)
        values = np.asarray(indata)
        if not values.size or np.max(np.abs(values)) <= 1e-5:
            return
        try:
            blocks.put_nowait(np.array(values, dtype=np.float32, copy=True))
        except queue.Full:
            try:
                blocks.get_nowait()
            except queue.Empty:
                pass
            try:
                blocks.put_nowait(np.array(values, dtype=np.float32, copy=True))
            except queue.Full:
                pass

    stream = sd.InputStream(
        samplerate=rate, channels=channels, dtype='float32',
        device=args.device, blocksize=1024, callback=callback)
    timeout = max(0.0, float(timeout))
    deadline = (None if wait_for_signal else time.monotonic()+timeout)
    try:
        stream.start()
        while not (stop and stop.is_set()):
            if deadline is not None and time.monotonic() >= deadline:
                break
            try:
                block = blocks.get(timeout=.05)
            except queue.Empty:
                continue
            now = time.monotonic()
            if deadline is None:
                deadline = now+timeout
            for index, live_input in enumerate(inputs):
                live_input.add(block[:, index:index+1].copy())
                audio = live_input.take(now)
                if audio is None:
                    continue
                audio_start = live_input.total-len(audio)
                profile_by_position = {
                    int(round(position)): mode
                    for position, mode in live_input.pulse_profile_hits(audio)}
                for position, scale, _confidence, way in live_input.pulse_hits(audio):
                    absolute = int(round(audio_start+position))
                    if absolute in last_positions[index]:
                        continue
                    # A live header is found before the packet is complete.
                    # Wait for the full region to align the status decision
                    # with the frame that becomes ready for latest-only decode.
                    if position+P.PULSE_FRAME*scale > len(audio)+1.0:
                        continue
                    last_positions[index].add(absolute)
                    mode = profile_by_position.get(int(round(position)))
                    if mode is not None:
                        observed_modes[index].append(mode)
                live_input.decoded()
            if confirmed_profile()[0] is not None:
                break
    finally:
        stream.stop()
        stream.close()
    return confirmed_profile()


def _make_auto_profile_decoder(args, fold):
    """Build the default status-driven Fold-500/mono-video receiver."""
    return _AdaptiveProfileDecoder(
        fold, _model(args.fixture, 'box'),
        preferred_side=getattr(args, 'mono_video_side', 'auto'))


def run_receive(args):
    if getattr(args, 'image_only', False) and getattr(args, 'headless', False):
        raise ValueError('--image-only cannot be combined with --headless')
    runtime_options = getattr(args, 'runtime_options', None)
    if runtime_options is not None:
        identity = runtime_options.snapshot().get('audio_input_identity')
        if identity is not None:
            import sounddevice as sd
            from tools.v7_receiver_audio import resolve_device_index
            args.device = resolve_device_index(
                sd, args.device, identity, 'input')
    profile_is_explicit = bool(
        getattr(args, 'experimental_mono', False) or
        getattr(args, 'experimental_mono_fold', False) or
        getattr(args, 'experimental_mono_colour', False) or
        getattr(args, 'baseline', False) or
        getattr(args, 'experimental_fold', None) is not None)
    if not profile_is_explicit:
        if getattr(args, 'pilot_timing', 'tone-seeded') == 'baseline':
            raise ValueError('automatic profile selection requires tone-assisted timing')
        if getattr(args, 'frame_boundary', 'eof') != 'eof':
            raise ValueError('automatic profile selection requires EOF packet boundaries')
        fold = _experimental_fold(500)
        profile_decoder = _make_auto_profile_decoder(args, fold)
        _ensure_test_modem_path()
        from tone_code import coded_pilot_timing
        profile_decoder.install()
        try:
            with coded_pilot_timing():
                return _run_receive(
                    args, fold, adaptive_profile=profile_decoder)
        finally:
            profile_decoder.uninstall()
    slots = _fold_slots(args)
    mono_profile = bool(getattr(args, 'experimental_mono', False))
    mono_fold_profile = bool(getattr(args, 'experimental_mono_fold', False) or
                             getattr(args, 'experimental_mono_colour', False))
    fold = _experimental_fold(slots)
    if mono_profile or mono_fold_profile:
        if getattr(args, 'pilot_timing', 'tone-seeded') == 'baseline':
            raise ValueError('experimental mono profiles require tone-assisted timing')
        if getattr(args, 'frame_boundary', 'eof') != 'eof':
            raise ValueError('experimental mono profiles require EOF packet boundaries')
        _ensure_test_modem_path()
        from tone_code import coded_pilot_timing
        if mono_fold_profile:
            from mono_video import MonoColourFoldWire, MonoFreshFoldWire
            from tone_code import MONO_1000
            use_colour = (getattr(args, 'experimental_mono_colour', False) or
                          getattr(args, '_detected_mono_mode', None) == MONO_1000)
            wire_class = MonoColourFoldWire if use_colour else MonoFreshFoldWire
            model = _model(args.fixture, 'box')
            requested_side = getattr(args, 'mono_video_side', 'auto')
            detected_side = getattr(args, '_detected_mono_video_side', None)
            selected_side = (requested_side if requested_side in ('left', 'right')
                             else detected_side or 'right')
            mono_wire = wire_class(
                model, side=selected_side)
        else:
            from mono_wire import MonoWire
            model = _model(args.fixture, 'nearest')
            mono_wire = MonoWire(model)
        mono_wire.install()
        try:
            with coded_pilot_timing():
                return _run_receive(args, None, mono_wire)
        finally:
            mono_wire.uninstall()
    if fold is None:
        return _run_receive(args, None)
    if getattr(args, 'pilot_timing', 'tone-seeded') == 'baseline':
        raise ValueError('coded-pilot folding requires tone-assisted timing; '
                         'use --baseline for the original steady-pilot profile')
    # The prototype wraps v7.decode_frame and the equalisers to keep each
    # packet's equaliser output. The coded-pilot context despreads valid chips
    # before V7 estimates tone timing. Both wrappers are always restored.
    fold.install()
    try:
        _ensure_test_modem_path()
        from tone_code import coded_pilot_timing
        with coded_pilot_timing():
            return _run_receive(args, fold)
    finally:
        fold.uninstall()


def _run_receive(args, fold, mono_wire=None, adaptive_profile=None):
    import sounddevice as sd
    from tools.v7_device_recovery import device_identity

    user_stop = getattr(args, 'stop_event', None) or threading.Event()
    runtime_options = getattr(args, 'runtime_options', None)
    identity = (runtime_options.snapshot().get('audio_input_identity')
                if runtime_options is not None else None)
    if identity is None:
        identity = device_identity(sd, args.device)

    while not user_stop.is_set():
        recovery = _run_receive_session(
            args, fold, mono_wire, adaptive_profile,
            user_stop=user_stop, input_identity=identity)
        if recovery is None or user_stop.is_set():
            return
        device, rate, reason = recovery
        args.device = device
        if adaptive_profile is not None:
            adaptive_profile.reset_capture_timeline()
        print(json.dumps({
            'status': 'receiver_input_reconnected',
            'device': identity.get('name'),
            'sample_rate': rate,
            'after': reason,
        }), flush=True)


def _run_receive_session(args, fold, mono_wire=None, adaptive_profile=None,
                          user_stop=None, input_identity=None):
    import sounddevice as sd
    from tools.v7_receiver_audio import (AudioPassthrough,
                                         ReceiverChannelRouter,
                                         ReceiverRuntimeOptions,
                                         passthrough_audio_side)
    from tools.v7_device_recovery import (
        DEVICE_RATE_FRAME_INTERVAL, DeviceRateDebouncer,
        query_device_snapshot)

    RECEIVER_GUI_STATUS.clear()
    # The standalone configuration GUI can cancel setup before the capture
    # stream is opened. The CLI keeps the original self-owned lifecycle.
    stop = threading.Event()
    # Metadata is decoded with the common bootstrap model; the body model is
    # selected from the protected encoding ID carried by each frame.
    base_model = _model(
        args.fixture,
        'box' if (fold is not None or adaptive_profile is not None or
                  getattr(mono_wire, 'requires_box', False))
        else 'nearest')
    model = mono_wire.model_for(base_model) if mono_wire is not None else base_model
    if fold is not None:
        fold.check(model)
    # Compile acquisition, polarity, equalizer and coded-status kernels before
    # opening audio. The Numba disk cache reuses unchanged kernels on compatible
    # later runs; a cold cache is still compiled before capture begins.
    P.PULSE.warmup_pulse_kernels()
    P.warmup_leg_polarity()
    P.warmup_equalizer(model)
    if fold is not None:
        from tone_code import warmup_coded_decoder
        warmup_coded_decoder(model)
    if mono_wire is not None:
        from tone_code import warmup_coded_decoder
        warmup_coded_decoder(model)
        warmup_wire = mono_wire.encode(
            base_model, [np.zeros(base_model.coder.source_count)]*3,
            start_counter=1, eof_marker=True)
        warmup_audio = warmup_wire
        carrier_index = getattr(mono_wire, 'carrier_index', None)
        if carrier_index is not None:
            warmup_audio = warmup_wire[:, carrier_index]
        P.decode_pulse_stream(
            model, warmup_audio, sample_rate=capture_rate_for(
                sd.query_devices(args.device, 'input')),
            pilot_timing=args.pilot_timing, frame_boundary='eof')
    if adaptive_profile is not None:
        _ensure_test_modem_path()
        from tone_code import warmup_status_templates
        from tone_code import FOLD_500, MONO_500, MONO_1000
        for mode in (FOLD_500, MONO_500, MONO_1000):
            warmup_status_templates(mode)
    if stop.is_set():
        return
    models = {model.encoding_type: model}

    def model_factory(encoding_type):
        """Build an alternate source model only when metadata requests it."""
        if encoding_type not in models:
            name = P.ENCODING_FILTERS[int(encoding_type)]
            candidate = _model(args.fixture, name)
            models[encoding_type] = (mono_wire.model_for(candidate)
                                     if mono_wire is not None else candidate)
        return models[encoding_type]
    device_info = sd.query_devices(args.device, 'input')
    input_channels = 1 if device_info['max_input_channels'] < 2 else 2
    if adaptive_profile is not None:
        adaptive_profile.input_channels = input_channels
    video_input_index = getattr(mono_wire, 'carrier_index', None)
    if video_input_index is not None and input_channels < 2:
        video_input_index = 0
    capture_rate = capture_rate_for(device_info)
    if adaptive_profile is not None:
        adaptive_profile.capture_rate = capture_rate
    profile_probes = (
        [_ProfileStatusProbe(
            capture_rate, index, getattr(args, 'direction', 'auto'),
            args.decode_history, args.decode_batch)
         for index in range(input_channels)]
        if adaptive_profile is not None else None)
    blocks = queue.Queue(maxsize=INPUT_AUDIO_QUEUE_BLOCKS)
    input_gap = threading.Event()
    input_ready = threading.Event()
    input_paused = threading.Event()
    input_stream = [None]
    recovery = [None]
    last_input_device_error = [None]
    capture_sample_cursor = [0]
    live_input = LiveInput(
        args.decode_history, args.decode_batch, rate=capture_rate,
        direction=getattr(args, 'direction', 'auto'))
    auto_mono_side = bool(
        mono_wire is not None and
        getattr(mono_wire, 'wire_profile', None) in
        ('mono-fresh-500', 'mono-colour-500') and
        getattr(args, 'mono_video_side', 'auto') == 'auto' and
        input_channels > 1)
    auto_mono_locked = not auto_mono_side
    auto_mono_switches = 0
    opposite_input_index = (
        1-video_input_index if mono_wire is not None and
        video_input_index is not None and input_channels > 1 else None)
    opposite_probe = (_MonoChannelProbe(
        capture_rate, getattr(args, 'direction', 'auto'),
        args.decode_history, args.decode_batch)
        if opposite_input_index is not None else None)
    direction_streak = DirectionStreak()
    # Absolute capture arrival of the last pulse hit handed to a decoder.
    # The sample clock never rewinds, so an input gap needs no reset here.
    decoded_through = None
    # Tail store and learned loop constants (N, p) persist across decodes.
    pulse_state = P.PulseState(
        tail_memory=(not args.no_tail_memory and
                     not getattr(mono_wire, 'disable_tail_memory', False)))
    if adaptive_profile is not None:
        adaptive_profile.bind_state(pulse_state)
    profile_generation_seen = (adaptive_profile.generation
                               if adaptive_profile is not None else 0)
    lag_ticks = deque(maxlen=32)        # recent picture lags, loop ticks
    decode_times = deque(maxlen=64)     # wall time of every decode cycle
    decode_cpu_times = deque(maxlen=512)  # (completion time, thread CPU seconds)
    shown_times = deque(maxlen=64)      # wall time of every published picture
    latest = None
    display_frames = FRAME_BUFFER
    previous_values = None
    diagnostics = {} if args.diagnostics else None
    auto_gain = 1.0
    if auto_mono_side:
        initial_side = 'left' if video_input_index == 0 else 'right'
        input_mode = f'mono-video-auto-{initial_side}'
    elif mono_wire is not None and video_input_index is not None:
        input_mode = f'mono-video-{getattr(mono_wire, "side", "")}'
    else:
        input_mode = 'mono-input' if input_channels == 1 else 'M/S'
    _ensure_test_modem_path()
    from tone_code import FOLD_500, FOLD_1000, FOLD_OFF, MONO_500, MONO_1000
    mono_status = ((MONO_500, MONO_1000) if adaptive_profile is not None else
                   MONO_1000 if getattr(mono_wire, 'wire_profile', None) ==
                   'mono-colour-500' else MONO_500)
    mono_status_name = ('MONO_500 / MONO_1000'
                        if adaptive_profile is not None else
                        'MONO_1000' if mono_status == MONO_1000 else
                        'MONO_500')
    initial_side = (getattr(mono_wire, 'side', None)
                    if mono_wire is not None else None)
    receiver_router = ReceiverChannelRouter(
        mono_status, (FOLD_OFF, FOLD_500, FOLD_1000),
        initial_video_side=(initial_side if initial_side in (
            'left', 'right', 'both') else None))
    runtime_options = getattr(args, 'runtime_options', None)
    if runtime_options is None:
        runtime_options = ReceiverRuntimeOptions(
            audio_output_device=getattr(args, 'audio_output_device', None),
            audio_muted=getattr(args, 'audio_muted', False),
            freewheel_seconds=getattr(args, 'freewheel_seconds', 2.0),
            show_sync_warning=getattr(args, 'show_sync_warning', True),
            audio_volume=getattr(args, 'audio_volume', 1.0),
            audio_diagnostics=getattr(args, 'audio_diagnostics', False))
    audio_bridge_enabled = bool(
        getattr(args, 'runtime_options', None) is not None or
        runtime_options.snapshot()['audio_output_device'] is not None)
    meter = {'peak': np.zeros(input_channels),
             'rms': np.zeros(input_channels), 'blocks': 0,
             'dropped': 0, 'decoded': 0, 'verified': 0, 'lost': 0,
             'status': 'acquiring', 'counter': None, 'decode_ms': None,
              'decode_cpu_ms': None, 'decode_cpu_percent': 0.0,
             'quality': '--',
             'timing_delta': None, 'playback_speed': None, 'source_index': None,
             # Largest source index (N-1) and the picture's lag behind the live
             # loop, both from the loop constants in the rotating CRC field;
             # '--' until learned (about a second) or if the sender has none.
             'max_index': None, 'lag_ms': None, 'loop': None,
             'shown_index': None, 'shown_direction': None,
             'playback_direction': None, 'direction_candidate': None,
             'direction_streak': 0,
             'pulse': None, 'aspect': 0, 'aspect_candidate': 0,
              'aspect_streak': 0, 'input_samples': 0, 'started': time.monotonic(),
              'wire_profile': (adaptive_profile.profile_name
                               if adaptive_profile is not None else None),
              'profile_candidate': None, 'profile_streak': 0,
               'auto_gain': 1.0, 'polarity': 1,
              'input_fps': 0., 'decode_fps': 0., 'shown_fps': 0.,
               'channel_hint': '',
               'route_state': receiver_router.state,
               'detected_mode': receiver_router.state,
               'channel_modes': (None, None),
               'video_side': receiver_router.video_side,
              'audio_side': receiver_router.audio_side,
              'sync_state': 'acquiring',
              'sync_age': None,
              'sync_warning': bool(runtime_options.snapshot().get(
                  'show_sync_warning', True)),
              'audio_muted': bool(runtime_options.snapshot().get(
                  'audio_muted', False)),
              'audio_volume': runtime_options.snapshot().get(
                  'audio_volume', 1.0),
               'audio_output_device': runtime_options.snapshot().get(
                   'audio_output_device'),
                'audio_output_identity': runtime_options.snapshot().get(
                    'audio_output_identity'),
                 'audio_device_error': None,
                 'input_device_error': None,
                'audio_runtime': {
                    'buffered_ms': 0.0,
                    'underflow_events': 0,
                    'dropped_samples': 0,
                    'current_clock_correction_ppm': 0.0,
                    'max_clock_correction_ppm': 0.0,
                    'input_status': {},
                    'output_status': {},
                },
               'mode': input_mode,
             'device': str(args.device), 'capture_rate': capture_rate,
             'input_channels': input_channels}

    def callback(indata, frames, timing, status):
        if input_paused.is_set():
            return
        values = np.asarray(indata, float)
        if status and passthrough is not None:
            passthrough.note_input_status(status)
        capture_start = capture_sample_cursor[0]
        capture_sample_cursor[0] += len(values)
        if passthrough is not None:
            # Queue the selected leg on every device callback, including
            # silence. Audio delivery must not depend on frame decoding.
            passthrough.queue_capture(capture_start, indata)
        meter['peak'] = np.maximum(meter['peak'],
                                   np.max(np.abs(values), axis=0))
        meter['rms'] = np.sqrt(np.mean(values*values, axis=0))
        has_data = bool(np.max(np.abs(values)) > 1e-5)
        if has_data:
            meter['blocks'] += 1
            meter['input_samples'] += len(values)
        if status:
            print(f'input: {status}', file=sys.stderr, flush=True)
        if not has_data:
            return
        block = np.array(indata, copy=True)
        try:
            blocks.put_nowait(block)
        except queue.Full:
            # Dropping an input block is an explicit discontinuity; keeping
            # stale audio would make the V7 clock appear to run backward.
            meter['dropped'] += 1
            input_gap.set()
        input_ready.set()

    def report_input_device_error(message):
        if message == last_input_device_error[0]:
            return
        last_input_device_error[0] = message
        meter['input_device_error'] = message
        meter['status'] = 'input device lost'
        print(json.dumps({
            'status': 'receiver_input_device_lost',
            'device': (input_identity or {}).get('name'),
            'message': message,
        }), flush=True)

    def clear_input_device_error():
        if last_input_device_error[0] is None:
            return
        previous = last_input_device_error[0]
        last_input_device_error[0] = None
        meter['input_device_error'] = None
        meter['status'] = 'reacquiring'
        if not args.no_log:
            print(json.dumps({
                'status': 'receiver_input_device_restored',
                'device': (input_identity or {}).get('name'),
                'previous_error': previous,
            }), flush=True)

    def monitor_input_device():
        stable = DeviceRateDebouncer()
        temporarily_unavailable = False
        while not stop.wait(DEVICE_RATE_FRAME_INTERVAL):
            stream = input_stream[0]
            try:
                stream_active = bool(stream is not None and stream.active)
            except Exception:
                stream_active = False
            try:
                index, info, observed_identity, _reported_rate = (
                    query_device_snapshot(
                        sd, args.device, input_identity, 'input'))
            except Exception as exc:
                input_paused.set()
                temporarily_unavailable = True
                stable.reset()
                report_input_device_error(str(exc))
                continue

            next_rate = capture_rate_for(info)
            index_changed = (isinstance(args.device, int) and
                             index != args.device)
            rate_changed = abs(next_rate-capture_rate) > .5
            needs_reopen = not stream_active or index_changed or rate_changed
            if needs_reopen:
                input_paused.set()
                if not stream_active:
                    reason = 'input stream stopped'
                elif index_changed:
                    reason = 'input device index changed'
                else:
                    reason = (f'input sample rate changed from '
                              f'{capture_rate:g}Hz to {next_rate:g}Hz')
                report_input_device_error(reason)
                if stable.observe(observed_identity, next_rate):
                    recovery[0] = (index, next_rate, reason)
                    input_ready.set()
                    stop.set()
                    return
                continue

            if temporarily_unavailable:
                input_paused.set()
                if stable.observe(observed_identity, next_rate):
                    temporarily_unavailable = False
                    input_paused.clear()
                    stable.reset()
                    clear_input_device_error()
                continue

            stable.reset()
            if input_paused.is_set():
                input_paused.clear()
            clear_input_device_error()

    def live_loop_position():
        """(calculated index now, its direction, picture index minus it).

        The calculated index is where the loop is right now on this machine's
        clock, from the received N and p.  The offset is plain index
        arithmetic against it -- no timing state -- so it is exactly what the
        two numbers on screen differ by, and one index step is 1/30 s.  It
        grows between pictures and drops as each new one arrives.  Its sign
        follows the ping-pong direction: a late picture reads negative while
        the loop counts up and positive while it counts down.  (None,
        None) until the loop constants are known, or if the sender's index
        does not follow the clock.
        """
        loop = meter['loop']
        if loop is None or not loop.clocked:
            return None, 0, None
        ticks = P.loop_ticks(time.time_ns())
        ideal = P.loop_index(ticks, loop)
        way = P.loop_direction(ticks, loop)
        if meter['shown_index'] is None:
            return ideal, way, None
        return ideal, way, meter['shown_index'] - ideal

    def display_diagnostics():
        """Keep the full receiver readout, sampled at the viewer's UI rate."""
        now = time.monotonic()
        meter['decode_fps'] = windowed_rate(decode_times, now)
        meter['shown_fps'] = windowed_rate(shown_times, now)
        decode_cpu_seconds = sum(
            cpu_seconds for completed, cpu_seconds in decode_cpu_times
            if completed >= now-2.0)
        cpu_window = min(2.0, max(0.0, now-meter['started']))
        meter['decode_cpu_percent'] = (
            100.0*decode_cpu_seconds/cpu_window if cpu_window > 0 else 0.0)
        incoming = live_input.incoming_fps(now)
        meter['input_fps'] = incoming
        shown = meter['shown_index']
        ideal, ideal_way, diff = live_loop_position()
        direction = {1: '+', -1: '-'}
        shown_text = ('--' if shown is None else
                      f'{shown}{direction.get(meter["shown_direction"], "")}')
        ideal_text = ('--' if ideal is None else
                      f'{ideal}{direction.get(ideal_way, "")}')
        diff_text = ('--' if diff is None else
                     f'{diff:+d} frames ({diff*1000/P.LOOP_IPS:+.0f} ms)')
        max_index = '--' if meter['max_index'] is None else str(meter['max_index'])
        decode_ms = ('--' if meter['decode_ms'] is None else
                     f'{meter["decode_ms"]:.1f} ms')
        pulse = meter['pulse']
        pulse_text = '--' if pulse is None else f'{pulse:.3f}'
        timing = meter['timing_delta']
        timing_text = '--' if timing is None else f'{timing:+.1f} ppm'
        speed = meter['playback_speed']
        speed_text = '--' if speed is None else f'{speed:.3f}×'
        playback_direction = meter['playback_direction']
        playback_text = {1: 'forward', -1: 'reverse'}.get(
            playback_direction, 'acquiring')
        if meter['direction_candidate'] in (-1, 1):
            playback_text += ('; candidate ' +
                              ('forward' if meter['direction_candidate'] > 0
                               else 'reverse') +
                              f' ×{meter["direction_streak"]}')
        aspect = P.V7_ASPECT_NAMES[int(meter['aspect']) & 7]
        candidate = P.V7_ASPECT_NAMES[int(meter['aspect_candidate']) & 7]
        peak = 20*np.log10(np.maximum(meter['peak'], 1e-9))
        rms = 20*np.log10(np.maximum(meter['rms'], 1e-9))
        if input_channels > 1:
            channel_levels = (f'peak L/R  {peak[0]:6.1f} / {peak[1]:6.1f} dBFS',
                              f'RMS  L/R  {rms[0]:6.1f} / {rms[1]:6.1f} dBFS')
        else:
            channel_levels = (f'peak mono {peak[0]:6.1f} dBFS',
                              f'RMS  mono {rms[0]:6.1f} dBFS')
        return {
            'status': (str(meter['status']),),
            'sync': (
                f'shown      {shown_text} / {max_index}',
                f'calculated {ideal_text}',
                f'offset     {diff_text}'),
            'decode': (
                f'frame {meter["counter"] if meter["counter"] is not None else "--"}   '
                f'good {meter["verified"]}   lost {meter["lost"]}',
                f'incoming {incoming:5.2f} fps   '
                f'decode {meter["decode_fps"]:5.2f}/s   '
                f'shown {meter["shown_fps"]:5.2f} fps',
                f'decode time {decode_ms}',
                f'input blocks {meter["blocks"]}   dropped {meter["dropped"]}'),
            'decode_cpu': (
                ('last -- ms/frame' if meter['decode_cpu_ms'] is None else
                 f'last {meter["decode_cpu_ms"]:.1f} ms/frame'),
                f'2s avg {meter["decode_cpu_percent"]:.0f}% / core'),
            'input': (*channel_levels,
                      f'auto gain {meter["auto_gain"]:5.2f}×',
                      f'right leg {"inverted" if meter["polarity"] < 0 else "normal"}'),
            'device': ((meter['input_device_error'] or
                        f'input {meter["device"]} · '
                        f'{meter["capture_rate"]:g}Hz'),),
            'audio': (
                f'q{meter["audio_runtime"]["buffered_ms"]:.0f}ms · '
                f'under {meter["audio_runtime"]["underflow_events"]}',
                f'in/out xrun '
                f'{sum(meter["audio_runtime"]["input_status"].values())}/'
                f'{sum(meter["audio_runtime"]["output_status"].values())}',
                f'clk {meter["audio_runtime"]["current_clock_correction_ppm"]:+.0f}/'
                f'{meter["audio_runtime"]["max_clock_correction_ppm"]:.0f}ppm · '
                f'drop {meter["audio_runtime"]["dropped_samples"]}',
                f'{meter["audio_output_device"] or "output off"} · '
                f'volume {meter["audio_volume"]:.2f}'),
            'signal': (
                f'aspect {aspect}  ·  candidate {candidate} ×{meter["aspect_streak"]}',
                (f'wire {meter["wire_profile"] or "--"}' +
                 (f' · candidate {meter["profile_candidate"]} '
                  f'×{meter["profile_streak"]}/3'
                  if meter['profile_candidate'] else '')),
                f'foundation {meter["quality"]}',
                f'pulse {pulse_text}   speed {speed_text}',
                f'audio {playback_text}',
                f'timing {timing_text}',
                f'route {meter["route_state"]} · '
                f'video {meter["video_side"] or "--"} · '
                f'audio {meter["audio_side"] or "--"}',
                f'passthrough {"muted" if meter["audio_muted"] else "live"} · '
                f'output {meter["audio_output_device"] or "not selected"} · '
                f'volume {meter["audio_volume"]:.2f}' +
                ('' if not meter['audio_device_error'] else
                 f' · {meter["audio_device_error"]}'),
                f'sync {meter["sync_state"]}' +
                ('' if meter['sync_age'] is None else
                 f' · {meter["sync_age"]:.1f}s since valid packet'),
                meter['channel_hint'] or 'channel route not independently flagged'),
        }

    RECEIVER_GUI_STATUS.update(meter=meter,
                               diagnostics=display_diagnostics)

    def refresh_runtime_state(now=None):
        now = time.monotonic() if now is None else float(now)
        options = runtime_options.snapshot()
        route = receiver_router.snapshot(now)
        meter['route_state'] = route['state']
        meter['detected_mode'] = route['state']
        meter['channel_modes'] = route['channel_modes']
        meter['video_side'] = route['video_side']
        sync_state = receiver_router.sync_state(
            now, options['freewheel_seconds'])
        if (sync_state == 'acquiring' and
                now-meter['started'] > options['freewheel_seconds']):
            sync_state = 'sync-lost'
        meter['sync_state'] = sync_state
        previous_audio_side = (passthrough.route
                               if passthrough is not None else None)
        meter['audio_side'] = passthrough_audio_side(route, input_channels)
        meter['sync_age'] = (None if route['last_packet'] is None else
                             max(0.0, now-route['last_packet']))
        meter['sync_warning'] = bool(
            meter['sync_state'] == 'sync-lost' and
            options['show_sync_warning'])
        meter['audio_muted'] = options['audio_muted']
        meter['audio_volume'] = options['audio_volume']
        identity = options['audio_output_identity']
        meter['audio_output_identity'] = identity
        meter['audio_output_device'] = (
            identity.get('name') if identity is not None else
            options['audio_output_device'])
        if passthrough is not None:
            passthrough.set_route(meter['audio_side'], options['audio_muted'])
            passthrough.set_volume(options['audio_volume'])
            meter['audio_device_error'] = passthrough.error
            if previous_audio_side != meter['audio_side'] and not args.no_log:
                print({'status': 'audio_route_change',
                       'audio_side': meter['audio_side'],
                       'route_state': route['state'],
                       'sync_state': sync_state,
                       'sync_age': meter['sync_age'],
                       'channel_modes': route['channel_modes']}, flush=True)

    # Output overrides for diagnosing stalls: V7_AUDIO_OUTPUT_MODE =
    # 'blocking' (default: PortAudio plays from its own buffer, Python only
    # refills it) or 'callback' (Python output callback); V7_AUDIO_OUTPUT_
    # LATENCY = PortAudio suggested latency in seconds, 'low' or 'high';
    # V7_AUDIO_OUTPUT_BLOCKSIZE = frames per write or callback (callback mode
    # only: 0 lets the host choose). Unset, AudioPassthrough's defaults apply.
    output_latency = os.environ.get('V7_AUDIO_OUTPUT_LATENCY') or None
    if output_latency not in (None, 'low', 'high'):
        output_latency = float(output_latency)
    output_blocksize = os.environ.get('V7_AUDIO_OUTPUT_BLOCKSIZE')
    output_blocksize = (None if output_blocksize in (None, '')
                        else int(output_blocksize))
    passthrough = (AudioPassthrough(
        capture_rate, sounddevice_module=sd,
        status_callback=lambda error: meter.__setitem__(
            'audio_device_error', error),
        output_latency=output_latency,
        output_blocksize=output_blocksize,
        output_mode=os.environ.get('V7_AUDIO_OUTPUT_MODE') or None)
        if audio_bridge_enabled else None)

    def audio_output_monitor():
        current_device = object()
        last_health_check = 0.0
        last_audio_report = 0.0
        last_audio_error_signature = None
        rate_stability = DeviceRateDebouncer()
        last_device_error = None
        last_passthrough_error = None
        while not stop.is_set():
            options = runtime_options.snapshot()
            device = options['audio_output_device']
            identity = options['audio_output_identity']
            selection = (device, identity)
            now = time.monotonic()
            selection_changed = selection != current_device
            reopen = False
            if passthrough is not None and selection_changed:
                passthrough.close()
                current_device = selection
                rate_stability.reset()
                if device is not None or identity is not None:
                    passthrough.open(device, identity)
                    reopen = passthrough.is_open
            elif (passthrough is not None and
                  (device is not None or identity is not None) and
                  now-last_health_check >= DEVICE_RATE_FRAME_INTERVAL):
                last_health_check = now
                if passthrough.is_open:
                    if not passthrough.check_device(identity):
                        passthrough.close()
                        rate_stability.reset()
                    else:
                        try:
                            index, _info, observed_identity, reported_rate = (
                                query_device_snapshot(
                                    sd, device, identity, 'output'))
                        except Exception as exc:
                            if str(exc) != last_device_error:
                                passthrough._report(
                                    f'Audio output unavailable; waiting to '
                                    f'reconnect: {exc}')
                                last_device_error = str(exc)
                            passthrough.close()
                            rate_stability.reset()
                        else:
                            current_rate = passthrough.device_default_rate
                            if (current_rate is not None and
                                    abs(reported_rate-current_rate) > 0.5):
                                if rate_stability.observe(
                                        observed_identity, reported_rate):
                                    passthrough.close()
                                    passthrough.open(index, identity)
                                    rate_stability.reset()
                                    reopen = passthrough.is_open
                            else:
                                rate_stability.reset()
                else:
                    try:
                        index, _info, observed_identity, reported_rate = (
                            query_device_snapshot(
                                sd, device, identity, 'output'))
                    except Exception as exc:
                        if str(exc) != last_device_error:
                            passthrough._report(
                                f'Audio output unavailable; waiting to '
                                f'reconnect: {exc}')
                            last_device_error = str(exc)
                        rate_stability.reset()
                    else:
                        if rate_stability.observe(observed_identity,
                                                  reported_rate):
                            passthrough.open(index, identity)
                            rate_stability.reset()
                            reopen = passthrough.is_open
            if reopen:
                last_device_error = None
                print(json.dumps({
                    'status': 'audio_output_reconnected',
                    'device': (identity or {}).get('name', device),
                    'sample_rate': passthrough.output_rate,
                }), flush=True)
            if passthrough is not None:
                passthrough.set_muted(options['audio_muted'])
                audio_stats = passthrough.stats_snapshot()
                meter['audio_runtime'] = audio_stats
                current_error = passthrough.error
                if current_error != last_passthrough_error:
                    print(json.dumps({
                        'status': ('receiver_passthrough_unavailable'
                                   if current_error else
                                   'receiver_passthrough_restored'),
                        'message': current_error,
                        'device': (identity or {}).get('name', device),
                    }), flush=True)
                    last_passthrough_error = current_error
                signature = (
                    audio_stats['underflow_events'],
                    tuple(sorted(audio_stats['input_status'].items())),
                    tuple(sorted(audio_stats['output_status'].items())),
                    audio_stats['dropped_samples'])
                has_audio_errors = any((
                    signature[0], signature[1], signature[2], signature[3]))
                audio_diagnostics = bool(options.get(
                    'audio_diagnostics', False))
                error_changed = (
                    signature != last_audio_error_signature and
                    has_audio_errors and
                    (args.log or args.diagnostics or audio_diagnostics))
                periodic_report = (
                    (args.diagnostics or audio_diagnostics) and
                    now-last_audio_report >= 1.0)
                if ((not args.no_log or audio_diagnostics) and
                        (error_changed or periodic_report)):
                    report_stream = getattr(
                        args, 'audio_diagnostics_stream', None) or sys.stdout
                    print({'status': 'audio_runtime', **audio_stats},
                          file=report_stream, flush=True)
                    passthrough.reset_report_peaks()
                    last_audio_report = now
                last_audio_error_signature = signature
            refresh_runtime_state(now)
            stop.wait(DEVICE_RATE_FRAME_INTERVAL)

    audio_threads = [threading.Thread(
        target=audio_output_monitor, daemon=True,
        name='v7-receiver-runtime-monitor')]
    if passthrough is not None:
        refresh_runtime_state()

    def update_channel_hint():
        if adaptive_profile is not None:
            if adaptive_profile.candidate is not None:
                hint = (f'profile candidate {adaptive_profile.candidate_name} '
                        f'×{adaptive_profile.streak}/'
                        f'{adaptive_profile.REQUIRED_STREAK}')
            elif adaptive_profile.active_mode in adaptive_profile.mono_status_modes:
                side = ('left' if adaptive_profile.active_side == 0 else
                        'right' if adaptive_profile.active_side == 1 else 'mono')
                hint = f'{adaptive_profile.profile_name} on {side}'
            else:
                hint = adaptive_profile.profile_name
        elif auto_mono_side and not auto_mono_locked:
            selected = 'left' if video_input_index == 0 else 'right'
            hint = (f'auto probing {selected} for {mono_status_name}; right wins '
                    'if both legs validate')
        elif (auto_mono_side and opposite_probe is not None and
              opposite_probe.streak >= 2):
            hint = ('V7 pulses on both legs; possible duplicated mono '
                    f'({mono_status_name} validated on selected leg)')
        elif auto_mono_side:
            selected = 'left' if video_input_index == 0 else 'right'
            hint = f'auto selected {selected} after {mono_status_name} validation'
        elif opposite_probe is None or opposite_probe.streak < 2:
            hint = ''
        elif meter['verified']:
            hint = ('V7 pulses on both legs; possible duplicated mono '
                    '(decoder profile verified on selected leg)')
        else:
            other = 'left' if opposite_input_index == 0 else 'right'
            selected = 'left' if video_input_index == 0 else 'right'
            hint = (f'V7 pulses on {other}; selected {selected} has not '
                    f'validated {mono_status_name} — check --mono-video-side')
        if hint != meter['channel_hint']:
            meter['channel_hint'] = hint
            if hint and not args.no_log:
                print({'status': 'mono_channel_hint', 'hint': hint}, flush=True)

    def update_route_from_packets(active_mode=None, active_valid=False,
                                  now=None, observed_channel_modes=None):
        now = time.monotonic() if now is None else float(now)
        channel_modes = [None, None]
        channel_times = [None, None]
        channel_confirmed = [False, False]
        if observed_channel_modes is not None:
            for index, mode in enumerate(observed_channel_modes):
                if mode is not None:
                    channel_modes[index] = mode
                    channel_times[index] = now
                    channel_confirmed[index] = True
        elif active_valid and active_mode is not None:
            if video_input_index is None:
                # A stereo decoder's valid coded profile describes the joined
                # two-leg wire; it does not identify either leg as passthrough.
                channel_modes = [active_mode, active_mode]
                channel_times = [now, now]
            else:
                channel_modes[video_input_index] = active_mode
                channel_times[video_input_index] = now
        if (opposite_probe is not None and opposite_input_index is not None and
                opposite_probe.status_mode is not None and
                opposite_probe.last_valid is not None):
            channel_modes[opposite_input_index] = opposite_probe.status_mode
            channel_times[opposite_input_index] = opposite_probe.last_valid
            channel_confirmed[opposite_input_index] = True
        route = receiver_router.observe(
            channel_modes[0], channel_modes[1], now=now,
            left_seen_at=channel_times[0], right_seen_at=channel_times[1],
            left_confirmed=channel_confirmed[0],
            right_confirmed=channel_confirmed[1])
        meter['route_state'] = route['state']
        meter['video_side'] = route['video_side']
        meter['audio_side'] = route['audio_side']
        if auto_mono_side and route['state'] in ('mono-left', 'mono-right'):
            desired = {'mono-left': 0, 'mono-right': 1}[route['state']]
            if desired != video_input_index and route['channel_active'][desired]:
                if switch_to_other_mono_leg():
                    update_channel_hint()
        return route

    def switch_to_other_mono_leg():
        """Try the probed leg while keeping its pulse history for decoding."""
        nonlocal live_input, video_input_index, opposite_input_index
        nonlocal decoded_through, direction_streak, auto_mono_switches
        if not auto_mono_side or opposite_probe is None:
            return False
        live_input, opposite_probe.input = opposite_probe.input, live_input
        video_input_index, opposite_input_index = (
            opposite_input_index, video_input_index)
        opposite_probe.reset()
        auto_mono_switches += 1
        decoded_through = None
        pulse_state.tail.reset()
        pulse_state.last_verified = None
        direction_streak = DirectionStreak()
        meter['playback_direction'] = None
        meter['direction_candidate'] = None
        meter['direction_streak'] = 0
        selected = 'left' if video_input_index == 0 else 'right'
        meter['mode'] = f'mono-video-auto-{selected}'
        meter['channel_hint'] = (
            f'probing {selected} for {mono_status_name}; right wins if both validate')
        return True

    def decode_available():
        nonlocal latest, auto_gain, previous_values, direction_streak
        nonlocal decoded_through, auto_mono_locked, profile_generation_seen
        if input_gap.is_set():
            # Never stitch samples across a callback drop.  Keep displaying
            # the last good image while pulse acquisition starts over.
            input_gap.clear()
            live_input.reset()
            if profile_probes is not None:
                for probe in profile_probes:
                    probe.reset()
                adaptive_profile.reset_candidate()
            pulse_state.tail.reset()
            pulse_state.last_verified = None
            direction_streak = DirectionStreak()
            if opposite_probe is not None:
                opposite_probe.reset()
                meter['channel_hint'] = ''
            meter['playback_direction'] = None
            meter['direction_candidate'] = None
            meter['direction_streak'] = 0
            # Discard stale decoder blocks so video reacquires at the live
            # edge. Audio is queued independently in the input callback.
            while True:
                try:
                    blocks.get_nowait()
                except queue.Empty:
                    break
            if not args.no_log:
                print({'status': 'input_gap_reacquire',
                       'dropped': meter['dropped']}, flush=True)
            return
        added = False
        while True:
            try:
                # Blocks are private copies of the callback data; LiveInput
                # applies the leveler's polarity to the stored audio itself.
                block = blocks.get_nowait()
                if (opposite_probe is not None and block.ndim == 2 and
                        block.shape[1] > opposite_input_index):
                    opposite_probe.add(
                        block[:, opposite_input_index:opposite_input_index+1].copy())
                if profile_probes is not None and block.ndim == 2:
                    for index, probe in enumerate(profile_probes):
                        if block.shape[1] > index:
                            probe.add(block[:, index:index+1].copy())
                if (video_input_index is not None and block.ndim == 2 and
                        block.shape[1] > video_input_index):
                    block = block[:, video_input_index:video_input_index+1]
                live_input.add(block)
                added = True
            except queue.Empty:
                break
        if not added:
            return
        # take() judges polarity, scans only new audio for frame headers and
        # trims to the minimal buffer (two frames at the current speed plus a
        # guard; see animation_modem/v7_live_input.py).  It returns audio
        # only when a new header has arrived, i.e. a new frame is complete, so
        # decode cycles follow the wire, not the capture block size, and an
        # idle or signal-free input never reaches the demodulator.
        now = time.monotonic()
        if opposite_probe is not None:
            opposite_probe.scan(now)
            update_channel_hint()
        profile_decisions = ()
        if profile_probes is not None:
            profile_events = []
            for probe in profile_probes:
                profile_events.extend(probe.scan(now))
            adaptive_profile.reset_candidate_if_stale(
                now, runtime_options.snapshot().get('freewheel_seconds', 2.0))
            profile_decisions = adaptive_profile.observe_packets(
                profile_events, now=now)
            for decision in profile_decisions:
                receiver_router.observe_profile_decision(decision, now=now)
                if decision['switched'] and not args.no_log:
                    print({'status': 'wire_profile_switch',
                           'profile': adaptive_profile.profile_name,
                           'video_side': decision['active_side'],
                           'confirming_packets': adaptive_profile.REQUIRED_STREAK},
                          flush=True)
            if adaptive_profile.generation != profile_generation_seen:
                previous_values = None
                profile_generation_seen = adaptive_profile.generation
            meter['wire_profile'] = adaptive_profile.profile_name
            meter['profile_candidate'] = adaptive_profile.candidate_name
            meter['profile_streak'] = adaptive_profile.streak
            update_channel_hint()
        audio = live_input.take(now)
        if profile_probes is None:
            decode_audio = audio
            decode_input = live_input
            decision = None
        else:
            decision = profile_decisions[-1] if profile_decisions else None
            decode_input = live_input
            decode_audio = audio
            if decision is None or not decision['confirmed']:
                decode_audio = None
            elif decision['mode'] in adaptive_profile.mono_status_modes:
                side = decision['side_index']
                if side is None and input_channels == 1:
                    side = 0
                if side is not None and 0 <= side < len(profile_probes):
                    decode_input = profile_probes[side].input
                    decode_audio = profile_probes[side].audio
            if (decode_audio is not None and
                    not adaptive_profile.claim(decision)):
                decode_audio = None
        meter['input_fps'] = decode_input.incoming_fps(now)
        meter['polarity'] = decode_input.polarity
        meter['auto_gain'] = decode_input.gain
        if decode_audio is None:
            if audio is not None:
                live_input.decoded()
            if profile_probes is not None and decision is not None:
                meter['status'] = 'waiting for three valid profile statuses'
            update_route_from_packets(now=now)
            update_channel_hint()
            return
        if profile_probes is not None:
            adaptive_profile.dispatch_mode = decision['mode']
            adaptive_profile.dispatch_side = decision['side_index']
        pulse_hits = decode_input.pulse_hits(decode_audio)
        if not pulse_hits:
            if profile_probes is None:
                decode_input.decoded()
            elif audio is not None:
                live_input.decoded()
            return
        audio_start = decode_input.total-len(decode_audio)
        # Normally the newest hit; at a reverse-to-forward turn-around, the
        # last reversed packet (see select_packet_hit).
        packet_start, packet_scale, _, packet_direction = select_packet_hit(
            pulse_hits, audio_start, decoded_through)
        pulse_starts = decode_input.pulse_starts(decode_audio)
        absolute_arrival = int(round(audio_start+packet_start))
        decoded_through = absolute_arrival
        # LiveInput levels the input before it searches for headers (a quiet
        # capture is otherwise never found); decode with that same gain.
        auto_gain = decode_input.gain
        meter['auto_gain'] = auto_gain
        if not args.refine:
            P.REFINE = False
        decode_times.append(time.monotonic())
        decode_cpu_start = time.thread_time()
        try:
            pulse_state.set_playback_direction(packet_direction)
            options = dict(
                diagnostics=diagnostics, input_gain=auto_gain, models=models,
                model_factory=model_factory,
                force_float32=args.force_float32, state=pulse_state,
                sample_rate=capture_rate, pilot_timing=args.pilot_timing,
                pilot_speed_diagnostics=args.pilot_speed_diagnostics,
                pulse_timing=args.pulse_timing,
                tone_equalization=args.tone_equalization)
            if packet_direction < 0:
                results, info = P.decode_reverse_packet(
                    model, decode_audio, packet_start, packet_scale, **options)
            else:
                results, info = P.decode_pulse_stream(
                    model, decode_audio, latest_only=True,
                    pulse_starts=pulse_starts,
                    frame_boundary=args.frame_boundary, **options)
                for result in results:
                    result.diag['playback_direction'] = 1
                info['playback_direction'] = 1
        except Exception as exc:
            # Drop the damaged window and let the next retained clock history
            # reacquire.  A single bad frame must not stop the live receiver.
            results, info = [], {'words': 0,
                                 'recovery_error': type(exc).__name__}
        decode_cpu_seconds = time.thread_time()-decode_cpu_start
        decode_cpu_times.append((time.monotonic(), decode_cpu_seconds))
        meter['decode_cpu_ms'] = decode_cpu_seconds*1000.0
        if profile_probes is None:
            decode_input.decoded()
        else:
            if audio is not None:
                live_input.decoded()
            if decode_input is not live_input:
                decode_input.decoded()
        if auto_mono_side:
            modes = [_mono_packet_status_mode(result) for result in results]
            expected_mode = getattr(mono_wire, 'status_mode', None)
            if expected_mode in modes:
                auto_mono_locked = True
                update_channel_hint()
        if results:
            result = results[-1]
            independently_validated = bool(
                result.diag.get('metadata_valid') and
                not result.diag.get('metadata_provisional') and
                (packet_direction < 0 or args.frame_boundary != 'eof' or
                 result.diag.get('eof_marker') is not None))
            result_mode = _mono_packet_status_mode(result)
            packet_valid = bool(
                independently_validated and
                result.status in ('received', 'verified'))
            if adaptive_profile is None:
                update_route_from_packets(
                    result_mode, active_valid=packet_valid,
                    now=time.monotonic())
            elif packet_valid:
                # Successful picture metadata is also sync evidence. Probe
                # scheduling must not expire audio while pictures decode.
                receiver_router.last_packet = time.monotonic()
            confirmed_direction, direction_switched = direction_streak.observe(
                absolute_arrival, packet_direction, independently_validated)
            meter['playback_direction'] = confirmed_direction
            meter['direction_candidate'] = direction_streak.candidate
            meter['direction_streak'] = direction_streak.streak
            if direction_switched and not args.no_log:
                print({'status': 'playback_direction_switch',
                       'playback_direction': confirmed_direction,
                       'capture_position': absolute_arrival}, flush=True)
            # Each call is gated by newly arrived audio.  The short rolling
            # history intentionally restarts the prototype's local counter,
            # so comparing result.counter here would suppress valid frames.
            meter['decoded'] += 1
            displayable = bool(result.diag.get('displayable', False))
            meter['status'] = ('degraded' if displayable and result.status == 'lost'
                               else result.status)
            meter['counter'] = meter['decoded']
            if result.diag.get('source_index') is not None:
                meter['source_index'] = int(result.diag['source_index'])
            meter['pulse'] = result.diag.get('pulse_confidence')
            meter['timing_delta'] = result.diag.get('timing_delta_ppm')
            speed = result.diag.get('playback_speed')
            incoming_fps = decode_input.incoming_fps(time.monotonic())
            meter['playback_speed'] = (
                incoming_fps/P.PULSE_FPS if incoming_fps > 0 else speed)
            meter['quality'] = (
                f'head {result.diag.get("head_confidence", 0):.2f}/'
                f'{result.diag.get("head_coverage", 0):.2f}')
            loop = result.diag.get('loop')
            if loop is not None and loop.frames > 0:
                meter['max_index'] = loop.frames - 1
                meter['loop'] = loop
            candidate = result.diag.get('aspect_code', meter['aspect'])
            if (result.status in ('received', 'verified') and
                    (result.diag.get('pulse_confidence') or 0) >= .45):
                if candidate == meter['aspect']:
                    meter['aspect_streak'] = 0
                elif candidate == meter['aspect_candidate']:
                    meter['aspect_streak'] += 1
                else:
                    meter['aspect_candidate'] = candidate
                    meter['aspect_streak'] = 1
                if meter['aspect_streak'] >= 3:
                    meter['aspect'] = meter['aspect_candidate']
                    meter['aspect_streak'] = 0
            meter['decode_ms'] = (info.get('diagnostics') or {}).get(
                'last_elapsed_ms')
            if result.status in ('received', 'verified') or displayable:
                if adaptive_profile is not None:
                    values = adaptive_profile.values(
                        models.get(result.diag.get('encoding_type'), model),
                        result)
                elif fold is not None:
                    values = fold.values(
                        models.get(result.diag.get('encoding_type'), model), result,
                        metadata_confirmed=_coded_mode_matches_fold(result, fold))
                elif mono_wire is not None:
                    values = mono_wire.values(
                        models.get(result.diag.get('encoding_type'), model), result)
                else:
                    values = P.values_from(model, result.coeffs)
                if args.mono_compatible:
                    noise = result.diag.get('noise') or [0.0]
                    values = stabilize_chroma(
                        values, previous_values, model.coder.grids,
                        pilot_error=max(noise),
                        coverage=result.diag.get('head_coverage'))
                latest = values
                if args.mono_compatible:
                    previous_values = latest.copy()
                display_frames.publish(latest, model.coder.grids,
                                       meter['aspect'])
                shown_times.append(time.monotonic())
                meter['shown_index'] = result.diag.get('source_index')
                meter['shown_direction'] = result.diag.get('direction')
                # Lag: where the live loop is now (this machine's clock and
                # the received N, p) against the index just put on screen.
                if (loop is not None and loop.clocked and
                        result.diag.get('source_index') is not None):
                    # The recent median resolves ping-pong turns (see
                    # loop_lag_ticks); the mean of recent readings gives
                    # sub-tick resolution, since pictures fall at varying
                    # points within a 33 ms tick.
                    lag_ticks.append(P.loop_lag_ticks(
                        result.diag['source_index'],
                        P.loop_ticks(time.time_ns()), loop,
                        expected=round(float(np.median(lag_ticks)))
                        if lag_ticks else 0,
                        direction=result.diag.get('direction')))
                    meter['lag_ms'] = (float(np.mean(lag_ticks)) *
                                       1000/P.LOOP_IPS)
            if result.status in ('received', 'verified'):
                meter['verified'] += 1
                update_channel_hint()
            if result.status == 'lost' and not displayable:
                meter['lost'] += 1
            report = {'counter': meter['decoded'], 'wire_counter': result.counter,
                      'source_index': result.diag.get('source_index'),
                      'status': meter['status'], 'displayable': displayable,
                      'pulse_frames': info.get('pulse_frames'),
                      'encoding': result.diag.get('encoding_name'),
                      'tail_slice': result.diag.get('tail_slice'),
                      'max_index': meter['max_index'],
                      'lag_ms': meter['lag_ms'],
                      'ideal_index': live_loop_position()[0],
                      'direction': result.diag.get('direction'),
                      'source_direction': result.diag.get('direction'),
                      'playback_direction': packet_direction,
                      'confirmed_playback_direction': confirmed_direction,
                      'input_gain': round(meter['auto_gain'], 3),
                      'right_polarity': meter['polarity'],
                      'incoming_fps': round(meter['input_fps'], 3),
                      'decode_cycles_per_s': round(windowed_rate(decode_times, time.monotonic()), 3),
                      'head_confidence': result.diag.get('head_confidence'),
                      'head_coverage': result.diag.get('head_coverage'),
                      'timing_delta_ppm': result.diag.get('timing_delta_ppm'),
                      'playback_speed': meter['playback_speed'],
                      'capture_rate_hz': capture_rate,
                      'pilot_tone_speed': result.diag.get('pilot_tone_speed'),
                      'noise': result.diag.get('noise'),
                      'stereo_erased_symbols': result.diag.get(
                          'stereo_erased_symbols'),
                      'metadata_valid': result.diag.get('metadata_valid'),
                      'skipped_frames': len(info.get('skipped_frames', [])),
                      'recovered': info.get('recovered', False)}
            if args.diagnostics:
                report['diagnostics'] = info.get('diagnostics')
            RECEIVER_GUI_STATUS.update(latest_packet=report,
                                       decode_info=info)
            if (args.log or args.diagnostics) and not args.no_log:
                print(report, flush=True)
            if args.save_dir:
                args.save_dir.mkdir(parents=True, exist_ok=True)
                values_image(latest, model.coder.grids).save(
                    args.save_dir/f'v7_{meter["decoded"]:08d}.png')
        else:
            confirmed_direction, _ = direction_streak.observe(
                absolute_arrival, packet_direction, False)
            meter['playback_direction'] = confirmed_direction
            meter['direction_candidate'] = direction_streak.candidate
            meter['direction_streak'] = direction_streak.streak
            if args.diagnostics and not args.no_log:
                print({'status': 'reacquiring', **info}, flush=True)

    try:
        stream = sd.InputStream(samplerate=capture_rate, channels=input_channels,
                                dtype='float32',
                                device=args.device, blocksize=1024,
                                callback=callback)
        input_stream[0] = stream
        if passthrough is not None:
            passthrough.set_input_rate(float(stream.samplerate))
        stream.start()
        if not args.no_log:
            print(f'V7 receive ready: input={args.device!r} '
                  f'rate={float(stream.samplerate):g}Hz (device {float(device_info["default_samplerate"]):g}Hz) '
                  f'channels={input_channels} ui={"headless" if args.headless else "window"} '
                  f'bootstrap=nearest', flush=True)
    except Exception:
        stop.set()
        raise

    def decode_worker():
        while True:
            input_ready.wait()
            input_ready.clear()
            if stop.is_set():
                break
            try:
                decode_available()
            except Exception as exc:
                # A damaged window must not terminate either the decoder or
                # the UI.  The next retained pulse history can reacquire.
                meter['status'] = f'decoder error: {type(exc).__name__}'
                if not args.no_log:
                    print({'status': 'decoder_exception',
                           'error': repr(exc)}, flush=True)

    decoder_thread = threading.Thread(target=decode_worker, daemon=True)
    audio_threads.append(threading.Thread(
        target=monitor_input_device, daemon=True,
        name='v7-receiver-input-monitor'))

    def watch_user_stop():
        while not stop.wait(.05):
            if user_stop is not None and user_stop.is_set():
                input_ready.set()
                stop.set()
                return

    audio_threads.append(threading.Thread(
        target=watch_user_stop, daemon=True,
        name='v7-receiver-stop-monitor'))
    for thread in audio_threads:
        thread.start()
    decoder_thread.start()

    try:
        if args.headless:
            while not stop.wait(.1):
                pass
        else:
            from tools.v7_gl_viewer import run as run_gl_viewer
            run_gl_viewer(
                display_frames.snapshot, lambda: meter,
                P.V7_ASPECT_RATIOS, fullscreen=args.fullscreen,
                show_diagnostics=args.show_diagnostics,
                diagnostics_source=display_diagnostics,
                profile_cpu=args.profile_ui,
                image_only=getattr(args, 'image_only', False),
                stop_event=stop)
    except KeyboardInterrupt:
        pass
    finally:
        stop.set()
        input_ready.set()
        decoder_thread.join(timeout=2)
        for thread in audio_threads:
            thread.join(timeout=2)
        if passthrough is not None:
            passthrough.close()
        stream = input_stream[0]
        if stream is not None:
            try:
                stream.stop()
            except Exception:
                pass
            try:
                stream.close()
            except Exception:
                pass
    return recovery[0]


def parser():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest='mode', required=True)
    send = sub.add_parser('send', help='capture camera/screen and transmit V7')
    send.add_argument('--source', choices=('camera', 'screen', 'video', 'test',
                                           'mouse-follow'),
                      help='capture source; omitted interactively prompts for one')
    send.add_argument('--device', type=_device_arg, required=True,
                      help='explicit sounddevice output, e.g. BlackHole 2ch')
    send.add_argument('--fixture', type=Path, default=DEFAULT_FIXTURE,
                      help=argparse.SUPPRESS)
    send.add_argument('--encode-filter', choices=('nearest', 'box', 'lanczos', 'bicubic'),
                      default=None,
                      help=argparse.SUPPRESS)
    source_path = send.add_mutually_exclusive_group()
    source_path.add_argument('--dct-encode', action='store_true',
                             help='analyze the unprepared RGB source in the V7 DCT domain')
    source_path.add_argument(
        '--perceptual-resize',
        choices=('off', 'linear-box', 'gamma-detail', 'linear-detail'),
        default='off', help=argparse.SUPPRESS)
    send.add_argument('--perceptual-detail-strength', type=float, default=0.25,
                      help=argparse.SUPPRESS)
    send.add_argument('--dct-sharpen', choices=('off', 'taper', 'usm'),
                      default=None, help=argparse.SUPPRESS)
    send.add_argument('--dct-sharpen-strength', type=float, default=None,
                      help=argparse.SUPPRESS)
    send.add_argument('--dct-clarity', type=float, default=None,
                      help=argparse.SUPPRESS)
    send.add_argument('--dct-chroma-gain', type=float, default=None,
                      help=argparse.SUPPRESS)
    send.add_argument('--dct-aggregation', choices=(
        'off', 'area-box', 'weighted-tent', 'weighted-cosine',
        'weighted-gaussian'), default=None, help=argparse.SUPPRESS)
    send.add_argument('--dct-band-profile', choices=('off', 'mid-luma',
                      'perceptual-color'), default=None,
                      help=argparse.SUPPRESS)
    send.add_argument('--brightness', type=float, default=None,
                      help='source brightness multiplier (default: 1.0 for folded profiles)')
    send.add_argument('--gamma', type=float, default=1.0,
                      help='source gamma; >1 lifts midtones (default: 1.0)')
    send.add_argument('--mono-video-side', choices=('left', 'right'),
                      default='right',
                      help=('all-fresh mono video: send on this output leg and '
                            'leave the other leg for audio (default: right)'))
    send.add_argument('--source-audio', choices=('source', 'device', 'off'),
                      default=None,
                      help=('mono-video channel audio: embedded video soundtrack '
                            '(mono default), selected input device, or off'))
    send.add_argument('--source-audio-device', type=_device_arg,
                      help='explicit input device for --source-audio device')
    send.add_argument('--source-audio-input-side',
                      choices=('mix', 'left', 'right'), default='mix',
                      help='downmix both input legs or choose one (default: mix)')
    send.add_argument('--source-audio-gain', type=float, default=1.0,
                      help='independent source-audio gain, 0..4 (default: 1)')
    send.add_argument('--source-audio-delay-ms', type=float, default=0.0,
                      help='additional audio delay beyond one emitted video packet')
    # These wire essentials remain explicit internal defaults, not user-facing
    # toggles. The GUI and CLI always emit the current reference/pilot/EOF wire.
    send.set_defaults(mono_sum=False, pilot_tones=True, eof_marker=True)
    send_profile = send.add_mutually_exclusive_group()
    send_profile.add_argument(
        '--profile', choices=('mono-fold-500', 'fold-500', 'mono-colour-500',
                              'fold-1000'),
        default=None,
        help=('wire profile: mono video with Fold 500 (recommended), '
              'stereo Fold 500 (default), or advanced Fold 1000, '
              'mono video with colour-weighted Fold 500 (experimental)'))
    send_profile.add_argument('--baseline', action='store_true',
                              help=argparse.SUPPRESS)
    send_profile.add_argument('--experimental-mono', action='store_true',
                              help=argparse.SUPPRESS)
    send_profile.add_argument('--experimental-mono-fold', action='store_true',
                              help=argparse.SUPPRESS)
    send_profile.add_argument('--experimental-fold', type=int, default=None,
                               choices=(0, 500, 1000), metavar='M',
                               help=argparse.SUPPRESS)
    send.add_argument('--camera', type=int, default=0)
    send.add_argument('--video-source', '--video', dest='video_source',
                      help='local video file or FFmpeg-supported live stream URL')
    send.add_argument('--video-live', action='store_true',
                      help='treat an HTTP(S) source as live instead of looping it')
    send.add_argument('--preview', action='store_true',
                      help='open video sources in a desktop player while sending')
    send.add_argument('--image-preview-port', type=int, metavar='PORT',
                      help=argparse.SUPPRESS)
    send.add_argument('--display', type=int)
    send.add_argument('--ffmpeg-input')
    send.add_argument('--screen-backend', choices=('mss', 'ffmpeg'), default='mss',
                      help='screen capture backend; mss avoids an FFmpeg child')
    send.add_argument('--region')
    send.add_argument('--capture-width', type=int, default=160,
                      help=('FFmpeg screen/video width and initial mouse-follow '
                            'crop width; normal camera mode uses 80x96; '
                            '--dct-encode preserves captured dimensions'))
    send.add_argument('--capture-filter',
                      choices=('neighbor', 'area', 'bilinear', 'bicubic',
                               'lanczos'),
                      default=None,
                       help=argparse.SUPPRESS)
    send.add_argument('--capture-fps', '--fps', dest='capture_fps', type=float)
    send.add_argument('--batch-frames', type=int, default=1,
                      help=argparse.SUPPRESS)
    send.add_argument('--speed', type=float, default=1.0,
                       help='pitch-shifted playback speed, 0.25..4.0; speeds '
                       'above the DAC Nyquist limit lose high-frequency detail')
    send.add_argument('--rate', type=int,
                      help='request an output sample rate (default: the DAC '
                           'native rate; the V7 wire remains 48 kHz reference geometry)')
    send.add_argument('--seconds', type=float, default=0,
                      help='0 means until Ctrl-C')
    send.add_argument('--no-log', dest='no_log', action='store_true',
                      default=False, help=argparse.SUPPRESS)
    send.add_argument('--gui-control', action='store_true',
                      help=argparse.SUPPRESS)
    send.add_argument('--log', dest='log', action='store_true',
                      default=False,
                      help='enable routine status output')
    recv = sub.add_parser('receive', help='receive V7 audio and display it')
    recv.add_argument('--device', type=_device_arg, required=True,
                      help='explicit sounddevice input, e.g. BlackHole 2ch')
    recv.add_argument('--audio-output-device', type=_device_arg,
                      help='explicit output device for the non-video input leg')
    recv.add_argument('--audio-muted', action='store_true',
                      help='mute receiver audio passthrough')
    recv.add_argument('--audio-volume', type=_audio_volume_arg, default=1.0,
                      help='passthrough volume from 0 to 1 (default: 1.0, VLC unity gain)')
    recv.add_argument('--freewheel-seconds', type=float, default=2.0,
                      help='time without valid video packets before sync-loss status')
    recv.add_argument('--no-sync-warning', dest='show_sync_warning',
                      action='store_false', default=True,
                      help='show the optional on-screen sync-loss warning')
    recv.add_argument('--direction', choices=('auto', 'forward', 'reverse'),
                      default='auto',
                      help='pulse direction detection (default: auto)')
    recv.add_argument('--fixture', type=Path, default=DEFAULT_FIXTURE)
    recv.add_argument('--headless', action='store_true')
    recv.add_argument('--fullscreen', action='store_true',
                      help='start fullscreen; F toggles, Escape exits fullscreen')
    recv.add_argument('--image-only', action='store_true',
                      help='show only the decoded picture fullscreen; Esc exits')
    recv.add_argument('--no-diagnostics', dest='show_diagnostics',
                      action='store_false',
                      help='hide the diagnostic overlay (I toggles it)')
    recv.add_argument('--profile-ui', action='store_true',
                      help='report viewer-thread and total process CPU every 5 s')
    recv.add_argument('--mono-compatible', action='store_true',
                       help='stabilize weak chroma for mono/one-leg playback')
    recv.add_argument('--mono-video-side', choices=('auto', 'left', 'right'),
                      default='auto',
                      help=('all-fresh mono video input leg (default: auto; '
                            'validates the mono status and prefers right if both work)'))
    recv.set_defaults(show_diagnostics=True)
    recv.add_argument('--no-log', dest='no_log', action='store_true',
                      default=False, help=argparse.SUPPRESS)
    recv.add_argument('--log', dest='log', action='store_true',
                      default=False,
                      help='enable routine status output')
    recv.add_argument('--save-dir', type=Path)
    recv.add_argument('--diagnostics', action='store_true',
                      help='print decoder stage timing and counters')
    recv.add_argument('--decode-batch', type=int, default=1,
                      help='new frame headers required before each decode (default: 1)')
    recv.add_argument('--decode-history', type=int, default=1,
                      help='frames kept after a decode; the buffer holds this '
                           'plus one frame and a small guard (default: 1)')
    recv.add_argument('--refine', action='store_true',
                      help='enable slower clock-template refinement')
    recv.add_argument('--no-tail-memory', action='store_true',
                      help='do not reuse tail coefficients from earlier packets')
    recv.add_argument('--force-float32', action='store_true',
                       help='use the experimental float32/complex64 decode path')
    recv.add_argument('--pilot-timing',
                       choices=('baseline', 'tone-seeded', 'tone-joint',
                                'tone-replaced'),
                       default='tone-seeded',
                       help='pilot timing fit (default: tone-seeded)')
    recv.add_argument('--frame-boundary', choices=('baseline', 'eof'),
                      default='eof',
                      help='packet boundary mode (default: EOF marker)')
    recv.add_argument('--pilot-speed-diagnostics', action='store_true',
                      help='compare raw pilot-tone speed with pulse-measured speed')
    recv.add_argument('--pulse-timing', choices=('baseline', 'pulse-warp'),
                      default='baseline',
                      help='experimental within-packet map from neighboring pulse scales')
    recv.add_argument('--tone-equalization',
                      choices=('off', 'm-reference'), default='off',
                      help='use pilot tones as an opt-in M-path gain reference')
    recv_profile = recv.add_mutually_exclusive_group()
    recv_profile.add_argument('--baseline', action='store_true',
                              help='restore the pre-fold receiver profile; use with sender --baseline')
    recv_profile.add_argument('--experimental-mono', action='store_true',
                              help='receive only the opt-in mono fold-off layout; unknown profiles hold the last picture')
    recv_profile.add_argument('--experimental-mono-fold', action='store_true',
                              help='receive the all-fresh mono video layout with a 500-class fold')
    recv_profile.add_argument('--experimental-mono-colour', action='store_true',
                              help='receive the mono colour-weighted 500-class fold (MONO_1000)')
    recv_profile.add_argument('--experimental-fold', type=int, default=None,
                               choices=(0, 500, 1000), metavar='M',
                               help='unfold M luma slots using coded-pilot status (default: 500); '
                                    '0 selects baseline. Use the sender\'s profile.')
    receiver_tuners = {
        'direction', 'fixture', 'profile_ui', 'mono_compatible',
        'mono_video_side', 'decode_batch', 'decode_history', 'refine',
        'no_tail_memory', 'force_float32', 'pilot_timing', 'frame_boundary',
        'pilot_speed_diagnostics', 'pulse_timing', 'tone_equalization',
        'baseline', 'experimental_mono', 'experimental_mono_fold',
        'experimental_mono_colour',
        'experimental_fold',
    }
    for action in recv._actions:
        if action.dest in receiver_tuners:
            action.help = argparse.SUPPRESS
    sub.add_parser('gui', help='open the receiver configuration and information GUI')
    return ap


if __name__ == '__main__':
    ap = parser()
    args = ap.parse_args()
    if args.mode == 'send':
        try:
            _resolve_send_source(args)
        except ValueError as exc:
            ap.error(str(exc))
        if args.preview and args.source != 'video':
            ap.error('--preview requires --source video')
        if args.rate is not None and args.rate <= 0:
            ap.error('--rate must be positive')
        if (not np.isfinite(args.speed) or
                not P.MIN_PLAYBACK_SPEED <= args.speed <= P.MAX_PLAYBACK_SPEED):
            ap.error(f'--speed must be between {P.MIN_PLAYBACK_SPEED:g} and '
                     f'{P.MAX_PLAYBACK_SPEED:g}')
        try:
            run_send(args)
        except Exception as exc:
            ap.exit(1, f'V7 send failed: {exc}\n')
    elif args.mode == 'gui':
        from tools.v7_receiver_gui import main as gui_main
        gui_main(sys.modules[__name__])
    else:
        run_receive(args)
