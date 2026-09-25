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

The prototype currently emits finite batches. Batch boundaries are therefore a
known live-experiment limitation; the clock counter continues across batches so
the receiver can diagnose the behavior rather than silently restarting it.
"""
import argparse
from collections import deque
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
from animation_modem.v7_live_input import LiveInput, windowed_rate         # noqa: E402
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
DEFAULT_FIXTURE = ROOT / 'modem_tests/fixtures/v7_reference_face.png'
# The display stays independent of the modem decoder and can consume the latest
# frame without building a backlog.
FRAME_BUFFER = LatestFrame()


def _device_arg(value):
    """Accept sounddevice names or numeric device indexes."""
    try:
        return int(value)
    except (TypeError, ValueError):
        return value


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
    return args


def _capture(args):
    """Build one of the shared RGB capture sources."""
    from tools.v7_capture import (camera_source, mouse_follow_source,
                                  screen_capture_source, screen_source,
                                  test_source, video_source, Throttled, _region)

    region = _region(args.region)
    if args.source == 'test':
        return test_source()
    if args.source == 'video':
        return video_source(args.video_source, width=args.capture_width,
                            scale_flags=args.capture_filter,
                            live=True if args.video_live else None)
    if args.source == 'mouse-follow':
        return mouse_follow_source(initial_width=args.capture_width)
    if args.source == 'camera':
        # Let FFmpeg probe the lowest mode/rate when the user did not override
        # it.  Some AVFoundation devices advertise 15 fps but reject a forced
        # 15-fps open unless their exact mode is selected first.
        return camera_source(args.camera, args.capture_fps,
                             width=args.capture_width, spec=args.ffmpeg_input,
                             scale_flags=args.capture_filter)
    if args.screen_backend == 'mss':
        return screen_source(region)
    capture_fps = args.capture_fps
    # AVFoundation screen capture commonly exposes only the display's native
    # refresh rate (for example 60 fps), not the modem wire rate.
    if capture_fps is None and sys.platform == 'darwin':
        capture_fps = 60
    return screen_capture_source(capture_fps or FPS, region, args.display,
                                 args.capture_width, args.ffmpeg_input,
                                 args.capture_filter)


def _model(fixture, encode_filter='nearest'):
    model = P.build_model(fixture, .1521 / np.sqrt(
        1 + 10**(P.CLOCK_REL_DB/10)), encode_filter=encode_filter)
    return model


def _values(model, frame, encode_filter='nearest', brightness=1.05, gamma=1.0):
    if gamma <= 0:
        raise ValueError('gamma must be positive')
    image = frame if isinstance(frame, Image.Image) else Image.fromarray(frame)
    prepared = prepare_image(image, encode_filter=encode_filter)
    if brightness != 1.0:
        prepared = ImageEnhance.Brightness(prepared).enhance(brightness)
    if gamma != 1.0:
        values = np.asarray(prepared, np.float32)/255.0
        values = np.clip(values, 0, 1)**(1.0/gamma)
        adjusted = Image.fromarray(np.uint8(np.rint(values*255)), 'RGB')
        adjusted.info.update(prepared.info)
        prepared = adjusted
    return (image_values(prepared, model.coder.grids,
                         encode_filter=encode_filter),
            P.aspect_wire_code(image.size))


def _experimental_fold(slots):
    """PROTOTYPE: the test_modem_v7 luma fold (see docs/transport_v7_spec.md
    section 10). Both ends must load the same table; nothing on the wire says
    a packet is folded."""
    if not slots:
        return None
    sys.path.insert(0, str(ROOT/'test_modem_v7'))
    from live_fold import LiveFold
    try:
        fold = LiveFold(slots)
    except ValueError as exc:
        raise SystemExit(f'--experimental-fold {slots}: {exc}')
    print(f'[experimental fold] {slots} luma slots, table {fold.digest} '
          f'(the other end must show the same)', flush=True)
    return fold


def run_send(args):
    import sounddevice as sd
    from tools.v7_capture import Throttled

    model = _model(args.fixture, args.encode_filter)
    fold = _experimental_fold(getattr(args, 'experimental_fold', 0))
    if fold is not None:
        # Fail before any audio: the table folds only the model it was built
        # for (the canonical box profile).
        fold.check(model)
    requested_rate = getattr(args, 'rate', None)
    output_rate = None
    raw_grab = _capture(args)
    capture_hz = args.capture_fps or (CAMERA_CAPTURE_FPS
                                      if args.source == 'camera' else
                                      (60 if args.screen_backend == 'ffmpeg' and
                                       sys.platform == 'darwin' else FPS))
    wire_fps = FPS*args.speed
    # Drain a paced FFmpeg source continuously and expose only
    # the newest frame to the audio encoder. Reading the pipe once per encoded
    # frame creates seconds of stale-camera latency.
    grab = Throttled(raw_grab, capture_hz)
    batches = queue.Queue(maxsize=2)
    stop = threading.Event()
    sentinel = object()
    producer_errors = []
    batch_size = max(1, args.batch_frames)
    total = 0
    started = time.monotonic()

    def encode_batch(frames, aspects, counter):
        values = np.asarray(frames)
        audio = P.encode_pulse_stream(model, values, start_counter=counter,
                                      aspect_codes=aspects,
                                      pilot_tones=getattr(args, 'pilot_tones', True),
                                      eof_marker=getattr(args, 'eof_marker', True))
        encoded_peak = float(np.max(np.abs(audio))) if audio.size else 0.0
        encoded_rms = float(np.sqrt(np.mean(audio*audio))) if audio.size else 0.0
        limiter_gain = 1.0
        limiter_samples = 0
        if args.mono_sum:
            audio = audio.sum(axis=1, keepdims=True)/np.sqrt(2)
        before_peak = float(np.max(np.abs(audio))) if audio.size else 0.0
        before_rms = float(np.sqrt(np.mean(audio*audio))) if audio.size else 0.0
        if args.mono_sum:
            peak = before_peak
            if peak > .89:
                limiter_gain = .89/float(peak)
                limiter_samples = int(np.count_nonzero(np.abs(audio) > .89))
                audio *= limiter_gain
        stats = {
            'frames_encoded': len(frames),
            'encoded_peak': encoded_peak,
            'encoded_rms': encoded_rms,
            'peak_before_limit': before_peak,
            'peak_after_limit': float(np.max(np.abs(audio))) if audio.size else 0.0,
            'rms_before_limit': before_rms,
            'limiter_active': limiter_gain < 1.0,
            'limiter_gain': limiter_gain,
            'samples_limited': limiter_samples,
        }
        output = P.speed_pulse_stream(audio, args.speed, rate=output_rate)
        stats.update({
            'emitted_peak': float(np.max(np.abs(output))) if output.size else 0.0,
            'emitted_rms': float(np.sqrt(np.mean(output*output)))
            if output.size else 0.0,
        })
        return output, stats

    def produce():
        nonlocal total
        frames = []
        aspects = []
        counter = 1
        next_capture = time.monotonic()
        failure = None
        try:
            while not stop.is_set() and (args.seconds <= 0 or
                                         time.monotonic()-started < args.seconds):
                delay = next_capture-time.monotonic()
                if delay > 0:
                    time.sleep(delay)
                value, aspect = _values(model, grab(), args.encode_filter,
                                        args.brightness, args.gamma)
                if fold is not None:
                    value = fold.encode(model, value)
                frames.append(value); aspects.append(aspect)
                # A compressed packet still needs a new source frame at the
                # faster wire cadence; otherwise the audio stream has gaps.
                next_capture += 1/wire_fps
                if len(frames) < batch_size:
                    continue
                audio, stats = encode_batch(frames, aspects, counter)
                batches.put((counter, audio, stats))
                total += len(frames)
                counter += len(frames)
                frames = []
                aspects = []
        except Exception as exc:
            failure = exc

        if failure is None and frames and not stop.is_set():
            try:
                audio, stats = encode_batch(frames, aspects, counter)
                batches.put((counter, audio, stats))
                total += len(frames)
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
            if hasattr(grab, 'retune'):
                grab.retune(wire_fps)
            if (not np.isfinite(args.speed) or
                    not P.MIN_PLAYBACK_SPEED <= args.speed <= P.MAX_PLAYBACK_SPEED):
                raise ValueError(
                    f'--speed must be between {P.MIN_PLAYBACK_SPEED:g} and '
                    f'{P.MAX_PLAYBACK_SPEED:g} (higher speeds may lose high-frequency detail)')
            worker.start()
            worker_started = True
            if not args.no_log:
                camera_text = (f'camera={args.camera} '
                               if args.source == 'camera' else '')
                print(f'V7 send ready: source={args.source} device={args.device!r} '
                       f'rate={output_rate:g}Hz wire={wire_fps:.3f}fps '
                       f'speed={args.speed:g}x {camera_text}'
                       f'pilot-tones={"on" if getattr(args, "pilot_tones", False) else "off"} '
                       f'capture={args.capture_width}px/{args.capture_filter} '
                      f'encode={args.encode_filter} mode={"mono-sum" if args.mono_sum else "M/S"}',
                      flush=True)
            while True:
                item = batches.get()
                if item is sentinel:
                    break
                counter, audio, stats = item
                # sounddevice requires a C-contiguous interleaved buffer;
                # filtering/resampling can return a strided view here.
                stream.write(np.ascontiguousarray(audio, dtype=np.float32))
                if args.log and not args.no_log:
                    print({
                        'sent_through_frame': (
                            counter+stats['frames_encoded']-1),
                        **{key: value for key, value in stats.items()
                           if key != 'frames_encoded'},
                    }, flush=True)
    except KeyboardInterrupt:
        stop.set()
    finally:
        stop.set()
        if worker_started:
            worker.join(timeout=2)
        else:
            close = getattr(grab, 'close', None)
            if close is not None:
                close()
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


def run_receive(args):
    fold = _experimental_fold(getattr(args, 'experimental_fold', 0))
    if fold is None:
        return _run_receive(args, None)
    # The prototype wraps v7.decode_frame and the equalisers to keep each
    # packet's equaliser output; always restore them, however the run ends.
    fold.install()
    try:
        return _run_receive(args, fold)
    finally:
        fold.uninstall()


def _run_receive(args, fold):
    import sounddevice as sd

    # Metadata is decoded with the common bootstrap model; the body model is
    # selected from the protected encoding ID carried by each frame.
    model = _model(args.fixture, 'nearest')
    # Numba compiles the equalizer on its first call; do that before opening the
    # audio stream so compilation cannot stall live capture and drop a packet.
    P.warmup_equalizer(model)
    models = {model.encoding_type: model}

    def model_factory(encoding_type):
        """Build an alternate source model only when metadata requests it."""
        if encoding_type not in models:
            name = P.ENCODING_FILTERS[int(encoding_type)]
            models[encoding_type] = _model(args.fixture, name)
        return models[encoding_type]
    device_info = sd.query_devices(args.device, 'input')
    input_channels = 1 if device_info['max_input_channels'] < 2 else 2
    capture_rate = capture_rate_for(device_info)
    blocks = queue.Queue(maxsize=32)
    stop = threading.Event()
    input_gap = threading.Event()
    live_input = LiveInput(args.decode_history, args.decode_batch, rate=capture_rate)
    # Tail store and learned loop constants (N, p) persist across decodes.
    pulse_state = P.PulseState(tail_memory=not args.no_tail_memory)
    lag_ticks = deque(maxlen=32)        # recent picture lags, loop ticks
    decode_times = deque(maxlen=64)     # wall time of every decode cycle
    shown_times = deque(maxlen=64)      # wall time of every published picture
    latest = None
    display_frames = FRAME_BUFFER
    previous_values = None
    diagnostics = {} if args.diagnostics else None
    auto_gain = 1.0
    meter = {'peak': np.zeros(input_channels),
             'rms': np.zeros(input_channels), 'blocks': 0,
             'dropped': 0, 'decoded': 0, 'verified': 0, 'lost': 0,
             'status': 'acquiring', 'counter': None, 'decode_ms': None,
             'quality': '--',
             'timing_delta': None, 'playback_speed': None, 'source_index': None,
             # Largest source index (N-1) and the picture's lag behind the live
             # loop, both from the loop constants in the rotating CRC field;
             # '--' until learned (about a second) or if the sender has none.
             'max_index': None, 'lag_ms': None, 'loop': None,
             'shown_index': None, 'shown_direction': None,
             'pulse': None, 'aspect': 0, 'aspect_candidate': 0,
             'aspect_streak': 0, 'input_samples': 0, 'started': time.monotonic(),
             'auto_gain': 1.0, 'polarity': 1,
             'input_fps': 0., 'decode_fps': 0., 'shown_fps': 0.,
             'mode': 'mono-input' if input_channels == 1 else 'M/S',
             'device': str(args.device), 'capture_rate': capture_rate,
             'input_channels': input_channels}

    def callback(indata, frames, timing, status):
        values = np.asarray(indata, float)
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
        try:
            blocks.put_nowait(np.array(indata, copy=True))
        except queue.Full:
            # Dropping an input block is an explicit discontinuity; keeping
            # stale audio would make the V7 clock appear to run backward.
            meter['dropped'] += 1
            input_gap.set()

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
            'input': (*channel_levels,
                      f'auto gain {meter["auto_gain"]:5.2f}×',
                      f'right leg {"inverted" if meter["polarity"] < 0 else "normal"}'),
            'signal': (
                f'aspect {aspect}  ·  candidate {candidate} ×{meter["aspect_streak"]}',
                f'foundation {meter["quality"]}',
                f'pulse {pulse_text}   speed {speed_text}',
                f'timing {timing_text}'),
        }

    def decode_available():
        nonlocal latest, auto_gain, previous_values
        if input_gap.is_set():
            # Never stitch samples across a callback drop.  Keep displaying
            # the last good image while pulse acquisition starts over.
            input_gap.clear()
            live_input.reset()
            if not args.no_log:
                print({'status': 'input_gap_reacquire',
                       'dropped': meter['dropped']}, flush=True)
            return
        added = False
        while True:
            try:
                # Blocks are private copies of the callback data; LiveInput
                # applies the leveler's polarity to the stored audio itself.
                live_input.add(blocks.get_nowait())
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
        audio = live_input.take(now)
        meter['input_fps'] = live_input.incoming_fps(now)
        meter['polarity'] = live_input.polarity
        if audio is None:
            return
        pulse_starts = live_input.pulse_starts(audio)
        # LiveInput levels the input before it searches for headers (a quiet
        # capture is otherwise never found); decode with that same gain.
        auto_gain = live_input.gain
        meter['auto_gain'] = auto_gain
        if not args.refine:
            P.REFINE = False
        decode_times.append(time.monotonic())
        try:
            results, info = P.decode_pulse_stream(
                model, audio, diagnostics=diagnostics, latest_only=True,
                input_gain=auto_gain, models=models,
                model_factory=model_factory,
                force_float32=args.force_float32, state=pulse_state,
                pulse_starts=pulse_starts, sample_rate=capture_rate,
                pilot_timing=args.pilot_timing,
                pilot_speed_diagnostics=args.pilot_speed_diagnostics,
                pulse_timing=args.pulse_timing,
                frame_boundary=args.frame_boundary,
                tone_equalization=args.tone_equalization)
        except Exception as exc:
            # Drop the damaged window and let the next retained clock history
            # reacquire.  A single bad frame must not stop the live receiver.
            results, info = [], {'words': 0,
                                 'recovery_error': type(exc).__name__}
        live_input.decoded()
        if results:
            result = results[-1]
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
            incoming_fps = live_input.incoming_fps(time.monotonic())
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
                if fold is not None:
                    values = fold.values(
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
                      'metadata_valid': result.diag.get('metadata_valid'),
                      'skipped_frames': len(info.get('skipped_frames', [])),
                      'recovered': info.get('recovered', False)}
            if args.diagnostics:
                report['diagnostics'] = info.get('diagnostics')
            if (args.log or args.diagnostics) and not args.no_log:
                print(report, flush=True)
            if args.save_dir:
                args.save_dir.mkdir(parents=True, exist_ok=True)
                values_image(latest, model.coder.grids).save(
                    args.save_dir/f'v7_{meter["decoded"]:08d}.png')
        elif args.diagnostics and not args.no_log:
            print({'status': 'reacquiring', **info}, flush=True)

    try:
        stream = sd.InputStream(samplerate=capture_rate, channels=input_channels,
                                dtype='float32',
                                device=args.device, blocksize=1024,
                                callback=callback)
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
        while not stop.is_set():
            try:
                decode_available()
            except Exception as exc:
                # A damaged window must not terminate either the decoder or
                # the UI.  The next retained pulse history can reacquire.
                meter['status'] = f'decoder error: {type(exc).__name__}'
                if not args.no_log:
                    print({'status': 'decoder_exception',
                           'error': repr(exc)}, flush=True)
            stop.wait(.01)

    decoder_thread = threading.Thread(target=decode_worker, daemon=True)
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
                profile_cpu=args.profile_ui)
    except KeyboardInterrupt:
        pass
    finally:
        stop.set()
        decoder_thread.join(timeout=2)
        stream.stop()
        stream.close()


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
    send.add_argument('--fixture', type=Path, default=DEFAULT_FIXTURE)
    send.add_argument('--encode-filter', choices=('nearest', 'box', 'lanczos', 'bicubic'),
                      default='nearest')
    send.add_argument('--brightness', type=float, default=1.05,
                      help='source brightness multiplier (default: 1.05)')
    send.add_argument('--gamma', type=float, default=1.0,
                      help='source gamma; >1 lifts midtones (default: 1.0)')
    send.add_argument('--mono-sum', action='store_true',
                       help='emit mono-summed M content on one channel')
    send.add_argument('--pilot-tones', action=argparse.BooleanOptionalAction,
                      default=True,
                      help='add V7 bin-1/bin-3 timing references (default on)')
    send.add_argument('--eof-marker', action=argparse.BooleanOptionalAction,
                      default=True,
                      help='add the V7 packet EOF marker (default on)')
    send.add_argument('--experimental-fold', type=int, default=0, metavar='M',
                      help='PROTOTYPE: fold M luma slots 2:1 for extra detail '
                           '(test_modem_v7/fold_table_M.json, M = 500 or 1000). '
                           'Requires --encode-filter box and the default '
                           'fixture. The receiver needs the same flag.')
    send.add_argument('--camera', type=int, default=0)
    send.add_argument('--video-source', '--video', dest='video_source',
                      help='local video file or FFmpeg-supported live stream URL')
    send.add_argument('--video-live', action='store_true',
                      help='treat an HTTP(S) source as live instead of looping it')
    send.add_argument('--display', type=int)
    send.add_argument('--ffmpeg-input')
    send.add_argument('--screen-backend', choices=('mss', 'ffmpeg'), default='mss',
                      help='screen capture backend; mss avoids an FFmpeg child')
    send.add_argument('--region')
    send.add_argument('--capture-width', type=int, default=160)
    send.add_argument('--capture-filter',
                      choices=('neighbor', 'area', 'bilinear', 'bicubic',
                               'lanczos'),
                      default='neighbor',
                      help='FFmpeg capture scaler (default: neighbor)')
    send.add_argument('--capture-fps', '--fps', dest='capture_fps', type=float)
    send.add_argument('--batch-frames', type=int, default=1,
                      help='frames encoded before submission (default: 1)')
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
    send.add_argument('--log', dest='log', action='store_true',
                      default=False,
                      help='enable routine status output')
    recv = sub.add_parser('receive', help='receive V7 audio and display it')
    recv.add_argument('--device', type=_device_arg, required=True,
                      help='explicit sounddevice input, e.g. BlackHole 2ch')
    recv.add_argument('--fixture', type=Path, default=DEFAULT_FIXTURE)
    recv.add_argument('--headless', action='store_true')
    recv.add_argument('--fullscreen', action='store_true',
                      help='start fullscreen; F toggles, Escape exits fullscreen')
    recv.add_argument('--no-diagnostics', dest='show_diagnostics',
                      action='store_false',
                      help='hide the diagnostic overlay (I toggles it)')
    recv.add_argument('--profile-ui', action='store_true',
                      help='report viewer-thread and total process CPU every 5 s')
    recv.add_argument('--mono-compatible', action='store_true',
                      help='stabilize weak chroma for mono/one-leg playback')
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
    recv.add_argument('--experimental-fold', type=int, default=0, metavar='M',
                      help='PROTOTYPE: unfold M luma slots (use the sender\'s '
                           'M; packets without the fold signature are shown '
                           'unchanged)')
    return ap


if __name__ == '__main__':
    ap = parser()
    args = ap.parse_args()
    if args.mode == 'send':
        try:
            _resolve_send_source(args)
        except ValueError as exc:
            ap.error(str(exc))
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
    else:
        run_receive(args)
