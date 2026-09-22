#!/usr/bin/env python3
"""Bench-only live V7 camera/screen sender and receiver.

This deliberately does not enter ``main.py`` or the production modem engine.
It uses the V7 prototype's fixed 48 kHz reference stream and is intended for
an explicit audio loopback device, normally BlackHole 2ch.

Examples::

    .venv/bin/python tools/v7_live.py send --source camera --device 'BlackHole 2ch'
    .venv/bin/python tools/v7_live.py send --source screen --device 'BlackHole 2ch'
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
# Main's display engine can consume this mailbox without importing the
# transport or changing its renderer.  The standalone Tk preview uses the
# same mailbox while this branch remains runnable by itself.
FRAME_BUFFER = LatestFrame()


def _device_arg(value):
    """Accept sounddevice names or numeric device indexes."""
    try:
        return int(value)
    except (TypeError, ValueError):
        return value


def _capture(args):
    """Build one of the shared RGB capture sources."""
    from tools.v7_capture import (camera_source, mouse_follow_source,
                                  screen_capture_source, screen_source,
                                  test_source, Throttled, _region)

    region = _region(args.region)
    if args.source == 'test':
        return test_source()
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


def run_send(args):
    import sounddevice as sd
    from tools.v7_capture import Throttled

    model = _model(args.fixture, args.encode_filter)
    raw_grab = _capture(args)
    capture_hz = args.capture_fps or (CAMERA_CAPTURE_FPS
                                      if args.source == 'camera' else
                                      (60 if args.screen_backend == 'ffmpeg' and
                                       sys.platform == 'darwin' else FPS))
    # Drain a paced FFmpeg source continuously and expose only
    # the newest frame to the audio encoder. Reading the pipe once per encoded
    # frame creates seconds of stale-camera latency.
    grab = Throttled(raw_grab, capture_hz)
    batches = queue.Queue(maxsize=2)
    stop = threading.Event()
    sentinel = object()
    batch_size = max(1, args.batch_frames)
    total = 0
    started = time.monotonic()

    def produce():
        nonlocal total
        frames = []
        aspects = []
        counter = 1
        next_capture = time.monotonic()
        try:
            while not stop.is_set() and (args.seconds <= 0 or
                                         time.monotonic()-started < args.seconds):
                delay = next_capture-time.monotonic()
                if delay > 0:
                    time.sleep(delay)
                value, aspect = _values(model, grab(), args.encode_filter,
                                        args.brightness, args.gamma)
                frames.append(value); aspects.append(aspect)
                next_capture += 1/FPS
                if len(frames) < batch_size:
                    continue
                values = np.asarray(frames)
                audio = P.encode_pulse_stream(model, values,
                                              start_counter=counter,
                                              aspect_codes=aspects)
                if args.mono_sum:
                    audio = audio.sum(axis=1, keepdims=True)/np.sqrt(2)
                    peak = np.max(np.abs(audio))
                    if peak > .89:
                        audio *= .89/peak
                audio = P.speed_pulse_stream(audio, args.speed)
                batches.put((counter, audio))
                total += len(frames)
                counter += len(frames)
                frames = []
                aspects = []
        finally:
            if frames and not stop.is_set():
                values = np.asarray(frames)
                audio = P.encode_pulse_stream(
                    model, values, start_counter=counter,
                    aspect_codes=aspects)
                if args.mono_sum:
                    audio = audio.sum(axis=1, keepdims=True)/np.sqrt(2)
                    peak = np.max(np.abs(audio))
                    if peak > .89:
                        audio *= .89/peak
                audio = P.speed_pulse_stream(audio, args.speed)
                batches.put((counter, audio))
                total += len(frames)
            batches.put(sentinel)
            close = getattr(grab, 'close', None)
            if close is not None:
                close()

    worker = threading.Thread(target=produce, daemon=True)
    worker.start()
    if not args.no_log:
        print(f'V7 send ready: source={args.source} device={args.device!r} '
              f'wire={FPS*args.speed:.3f}fps speed={args.speed:g}x camera={args.camera} '
              f'capture={args.capture_width}px/{args.capture_filter} '
              f'encode={args.encode_filter} mode={"mono-sum" if args.mono_sum else "M/S"}',
              flush=True)
    try:
        with sd.OutputStream(samplerate=P.RATE,
                             channels=1 if args.mono_sum else 2,
                             dtype='float32',
                             device=args.device, blocksize=0) as stream:
            while True:
                item = batches.get()
                if item is sentinel:
                    break
                counter, audio = item
                # sounddevice requires a C-contiguous interleaved buffer;
                # filtering/resampling can return a strided view here.
                stream.write(np.ascontiguousarray(audio, dtype=np.float32))
                if args.log and not args.no_log:
                    print(f'  sent through frame {counter+len(audio)//P.PULSE_FRAME-1}',
                          flush=True)
    except KeyboardInterrupt:
        stop.set()
    finally:
        stop.set()
        worker.join(timeout=2)
    if args.log and not args.no_log:
        print(f'V7 send stopped after {total} frames', flush=True)


# Capture at the device's own rate, like the sender's PacketOutput: forcing
# 48 kHz made PortAudio resample behind our back, and on a 96 kHz device a 2x
# wire (carriers up to 25.5 kHz) lost its top carriers to that converter.  The
# decoder needs no fixed rate -- it measures each frame's length from its
# preamble -- as long as the frame scale stays within PULSE_MIN/MAX_SCALE:
# capture rate / (48 kHz x speed) in [0.25, 2].  Above 96 kHz the capture is
# capped at 96 kHz (a benign conversion that keeps the whole band up to 3x),
# so 1x on a 192 kHz device still lands inside that range.
MAX_CAPTURE_RATE = 96_000


def capture_rate_for(device_info):
    rate = float(device_info.get('default_samplerate') or P.RATE)
    return min(rate, MAX_CAPTURE_RATE)


def run_receive(args):
    import sounddevice as sd

    # Metadata is decoded with the common bootstrap model; the body model is
    # selected from the protected encoding ID carried by each frame.
    model = _model(args.fixture, 'nearest')
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
    rendered = None
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
             'shown_index': None,
             'pulse': None, 'aspect': 0, 'aspect_candidate': 0,
             'aspect_streak': 0, 'input_samples': 0, 'started': time.monotonic(),
             'auto_gain': 1.0, 'polarity': 1,
             'input_fps': 0., 'decode_fps': 0., 'shown_fps': 0.,
             'mode': 'mono-input' if input_channels == 1 else 'M/S'}

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
        """(ideal index now, frames the picture on screen is behind it).

        The ideal index is where the loop is right now on this machine's
        clock, from the received N and p; the difference grows between
        pictures and drops when a new one arrives.  (None, None) until the
        loop constants are known or if the sender's index is not clocked.
        """
        loop = meter['loop']
        if loop is None or not loop.clocked:
            return None, None
        ticks = P.loop_ticks(time.time_ns())
        ideal = P.loop_index(ticks, loop)
        if meter['shown_index'] is None:
            return ideal, None
        recent = list(lag_ticks)
        diff = P.loop_lag_ticks(meter['shown_index'], ticks, loop,
                                expected=round(float(np.median(recent)))
                                if recent else 0)
        return ideal, diff

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
        peak = float(np.percentile(np.abs(audio[-P.PULSE_FRAME:]), 99.5))
        desired_gain = float(np.clip(.55/max(peak, 1e-6), .5, 32.0))
        auto_gain = (min(desired_gain, auto_gain*1.5)
                     if desired_gain > auto_gain else desired_gain)
        meter['auto_gain'] = auto_gain
        if not args.refine:
            P.REFINE = False
        decode_times.append(time.monotonic())
        try:
            results, info = P.decode_pulse_stream(
                model, audio, diagnostics=diagnostics, latest_only=True,
                input_gain=auto_gain, models=models,
                model_factory=model_factory,
                force_float32=args.force_float32, state=pulse_state)
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
            # The decoder's speed is relative to 48 kHz samples; at another
            # capture rate the same frame length means a different real speed.
            speed = result.diag.get('playback_speed')
            meter['playback_speed'] = (None if speed is None
                                       else speed*capture_rate/P.RATE)
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
                        if lag_ticks else 0))
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
                       'input_gain': round(meter['auto_gain'], 3),
                       'right_polarity': meter['polarity'],
                       'incoming_fps': round(meter['input_fps'], 3),
                       'decode_cycles_per_s': round(windowed_rate(decode_times, time.monotonic()), 3),
                       'head_confidence': result.diag.get('head_confidence'),
                       'head_coverage': result.diag.get('head_coverage'),
                       'timing_delta_ppm': result.diag.get('timing_delta_ppm'),
                       'playback_speed': meter['playback_speed'],
                       'capture_rate_hz': capture_rate,
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

    root = None
    label = None
    if not args.headless:
        import tkinter as tk
        from PIL import ImageTk
        root = tk.Tk()
        root.title('V7 modem receiver')
        root.resizable(True, True)
        root.geometry('640x650')
        root.configure(background='black')
        image_frame = tk.Frame(root, width=620, height=480,
                               background='black')
        image_frame.pack_propagate(False)
        info_visible = args.show_diagnostics
        info_frame = tk.Frame(root, background='black')
        if args.fullscreen:
            root.attributes('-fullscreen', True)
        image_frame.pack(padx=10, pady=8, fill='both', expand=True)

        def toggle_fullscreen(_event=None):
            root.attributes('-fullscreen', not root.attributes('-fullscreen'))

        def toggle_information(_event=None):
            nonlocal info_visible
            info_visible = not info_visible
            if info_visible:
                info_frame.pack(fill='x', padx=6, pady=(0, 4))
            else:
                info_frame.pack_forget()

        root.bind('<Escape>', lambda _event: root.attributes(
            '-fullscreen', False))
        root.bind_all('<KeyPress-f>', toggle_fullscreen)
        root.bind_all('<KeyPress-F>', toggle_fullscreen)
        root.bind_all('<KeyPress-i>', toggle_information)
        root.bind_all('<KeyPress-I>', toggle_information)
        root.focus_force()
        label = tk.Label(image_frame, text='Acquiring V7 clock…',
                         background='black')
        label.configure(background='black')
        label.pack(expand=True, fill='both')
        device_label = status_label = levels_label = stats_label = None
        quality_label = sync_label = None
        if info_visible:
            info_frame.pack(fill='x', padx=6, pady=(0, 4))
            device_label = tk.Label(info_frame, text=f'V7 input: {args.device}',
                                    background='black', foreground='white',
                                    anchor='w', justify='left')
            device_label.pack(fill='x')
            status_label = tk.Label(info_frame, text='status: acquiring', anchor='w',
                                    justify='left',
                                    background='black', foreground='white')
            status_label.pack(fill='x')
            levels_label = tk.Label(info_frame, text='levels: --', anchor='w',
                                    font='TkFixedFont', background='black',
                                    foreground='white')
            levels_label.pack(fill='x')
            stats_label = tk.Label(info_frame, text='frames: --', anchor='w',
                                   font='TkFixedFont', background='black',
                                   foreground='white')
            stats_label.pack(fill='x')
            # Its own line: what is on screen, where the loop should be now,
            # and how far apart they are.
            sync_label = tk.Label(info_frame, text='index --', anchor='w',
                                  font='TkFixedFont', background='black',
                                  foreground='white')
            sync_label.pack(fill='x')
            quality_label = tk.Label(info_frame, text='quality: --', anchor='w',
                                     font='TkFixedFont', background='black',
                                     foreground='white')
            quality_label.pack(fill='x')

        def tick():
            nonlocal rendered
            if info_visible:
                peak = 20*np.log10(np.maximum(meter['peak'], 1e-9))
                rms = 20*np.log10(np.maximum(meter['rms'], 1e-9))
                shown = meter['shown_index']
                ideal, diff = live_loop_position()
                diff_text = ('--' if diff is None else
                             f'{diff:+d} frames ({diff*1000/P.LOOP_IPS:+.0f} ms)')
                sync_label.configure(text=(
                    f'index {"--" if shown is None else shown}'
                    f' / {"--" if meter["max_index"] is None else meter["max_index"]}'
                    f'   calculated {"--" if ideal is None else ideal}'
                    f'   diff {diff_text}'))
                now = time.monotonic()
                # Rates are recomputed at display time so they fall to 0 when
                # input or decoding stops instead of freezing.
                meter['decode_fps'] = windowed_rate(decode_times, now)
                meter['shown_fps'] = windowed_rate(shown_times, now)
                incoming = live_input.incoming_fps(now)
                levels_label.configure(text=(
                    f'peak L/R {peak[0]:6.1f}/{peak[1]:6.1f} dBFS | '
                    f'rms {rms[0]:6.1f}/{rms[1]:6.1f} dBFS'))
                stats_label.configure(text=(
                    f'frames {meter["decoded"]}  '
                    f'verified {meter["verified"]} '
                    f'lost {meter["lost"]}  input blocks {meter["blocks"]} '
                    f'dropped {meter["dropped"]}'))
                quality_label.configure(text=f'quality: {meter["quality"]}')
                status_label.configure(wraplength=max(240, root.winfo_width()-12))
                pulse = meter['pulse']
                pulse_text = '--' if pulse is None else f'{pulse:.4g}'
                timing = meter['timing_delta']
                timing_text = '--' if timing is None else f'{timing:.8f}'
                speed = meter['playback_speed']
                speed_text = '--' if speed is None else f'{speed:.8f}x'
                aspect_text = P.V7_ASPECT_NAMES[int(meter['aspect']) & 7]
                candidate_aspect = P.V7_ASPECT_NAMES[
                    int(meter['aspect_candidate']) & 7]
                status_label.configure(text=(
                  f'status {meter["status"]}  mode {meter["mode"]}  '
                  f'frame {meter["counter"]}  '
                    f'aspect {aspect_text} '
                    f'(candidate {candidate_aspect} '
                    f'x{meter["aspect_streak"]})  pulse {pulse_text}\n'
                    f'gain {meter["auto_gain"]:4.1f}x'
                    f'{" R inverted" if meter["polarity"] < 0 else ""}  '
                    f'speed {speed_text}  timing {timing_text} ppm  '
                    f'decode {meter["decode_ms"] if meter["decode_ms"] is not None else "--"} ms | '
                    f'incoming {incoming:5.2f} fps  '
                    f'decode {meter["decode_fps"]:5.2f} cycles/s  '
                    f'shown {meter["shown_fps"]:5.2f} fps'))
            frame = display_frames.snapshot()
            if frame is not None and frame.generation != rendered:
                image = values_image(frame.values, frame.shapes)
                ratio = P.V7_ASPECT_RATIOS[frame.aspect & 7]
                bound_w = max(1, image_frame.winfo_width())
                bound_h = max(1, image_frame.winfo_height())
                height = bound_h
                width = round(height*ratio)
                if width > bound_w:
                    width = bound_w
                    height = round(width/ratio)
                image = image.resize((max(1, width), max(1, height)),
                                     Image.Resampling.NEAREST)
                photo = ImageTk.PhotoImage(image)
                label.configure(image=photo, text='')
                label.image = photo
                rendered = frame.generation
            root.after(10, tick)
        root.after(10, tick)
        try:
            root.mainloop()
        except KeyboardInterrupt:
            pass
    else:
        try:
            while not stop.is_set():
                stop.wait(.1)
        except KeyboardInterrupt:
            pass
    stop.set()
    decoder_thread.join(timeout=2)
    stream.stop(); stream.close()


def parser():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest='mode', required=True)
    send = sub.add_parser('send', help='capture camera/screen and transmit V7')
    send.add_argument('--source', choices=('camera', 'screen', 'test',
                                           'mouse-follow'), required=True)
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
    send.add_argument('--camera', type=int, default=0)
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
                      help='FFmpeg camera scaler (default: neighbor)')
    send.add_argument('--capture-fps', '--fps', dest='capture_fps', type=float)
    send.add_argument('--batch-frames', type=int, default=1,
                      help='frames encoded before submission (default: 1)')
    send.add_argument('--speed', type=float, default=1.0,
                      help='pitch-shifted playback speed; this sender outputs '
                           f'{P.RATE} Hz, so at most {P.max_wire_speed(P.RATE):.2f}x '
                           '(main.py --mode modem plays at the device rate)')
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
                      help='fullscreen embedded display; Escape exits fullscreen')
    recv.add_argument('--no-diagnostics', dest='show_diagnostics',
                      action='store_false',
                      help='hide diagnostic information from the window')
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
    return ap


if __name__ == '__main__':
    ap = parser()
    args = ap.parse_args()
    if args.mode == 'send':
        if not 0 < args.speed <= P.max_wire_speed(P.RATE):
            ap.error(f'--speed must be above 0 and at most '
                     f'{P.max_wire_speed(P.RATE):.2f} for this {P.RATE} Hz sender')
        run_send(args)
    else:
        run_receive(args)
