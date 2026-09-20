#!/usr/bin/env python3
"""Bench-only live V7 camera/screen sender and receiver.

This deliberately does not enter ``main.py`` or the production modem engine.
It uses the V7 prototype's fixed 48 kHz reference stream and is intended for
an explicit audio loopback device, normally BlackHole 2ch.

Examples::

    .venv/bin/python tools/v7_live.py send --source camera --device 'BlackHole 2ch'
    .venv/bin/python tools/v7_live.py send --source screen --device 'BlackHole 2ch'
    .venv/bin/python tools/v7_live.py receive --device 'BlackHole 2ch'

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
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from animation_modem.imaging import image_values, prepare_image, values_image  # noqa: E402
from animation_modem.aspect import ASPECT_RATIOS                             # noqa: E402
from tools import v7_proto as P                                             # noqa: E402


FPS = P.PULSE_FPS
DEFAULT_FIXTURE = ROOT / 'modem_tests/fixtures/v6_face_1110.png'


def _capture(args):
    """Build one of modem_screen's existing RGB capture sources."""
    from modem_screen import (camera_source, screen_capture_source,
                              screen_source, Throttled, _region)

    region = _region(args.region)
    if args.source == 'camera':
        return camera_source(args.camera, args.capture_fps,
                             width=args.capture_width, spec=args.ffmpeg_input)
    if args.screen_backend == 'mss':
        return screen_source(region)
    return screen_capture_source(args.capture_fps or FPS, region, args.display,
                                 args.capture_width, args.ffmpeg_input)


def _model(fixture, encode_filter='nearest'):
    model = P.build_model(fixture, .1521 / np.sqrt(
        1 + 10**(P.CLOCK_REL_DB/10)), encode_filter=encode_filter)
    return model


def _values(model, frame, encode_filter='nearest'):
    image = frame if isinstance(frame, Image.Image) else Image.fromarray(frame)
    prepared = prepare_image(image, preset='auto', encode_filter=encode_filter)
    return (image_values(prepared, model.coder.grids,
                         encode_filter=encode_filter),
            int(prepared.info.get('aspect_code', 0)))


def run_send(args):
    import sounddevice as sd
    from modem_screen import Throttled

    model = _model(args.fixture, args.encode_filter)
    raw_grab = _capture(args)
    capture_hz = args.capture_fps or (30 if args.source == 'camera' else FPS)
    # Match V3--V6: drain a paced FFmpeg source continuously and expose only
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
                value, aspect = _values(model, grab(), args.encode_filter)
                frames.append(value); aspects.append(aspect)
                next_capture += 1/FPS
                if len(frames) < batch_size:
                    continue
                values = np.asarray(frames)
                audio = P.encode_pulse_stream(model, values,
                                              start_counter=counter,
                                              aspect_codes=aspects)
                batches.put((counter, audio))
                total += len(frames)
                counter += len(frames)
                frames = []
                aspects = []
        finally:
            if frames and not stop.is_set():
                values = np.asarray(frames)
                batches.put((counter, P.encode_pulse_stream(
                    model, values, start_counter=counter,
                    aspect_codes=aspects)))
                total += len(frames)
            batches.put(sentinel)
            close = getattr(grab, 'close', None)
            if close is not None:
                close()

    worker = threading.Thread(target=produce, daemon=True)
    worker.start()
    print(f'V7 send: {args.source}, {FPS:.3f} fps, batch={batch_size}, '
          f'device={args.device!r}', flush=True)
    try:
        with sd.OutputStream(samplerate=P.RATE, channels=2, dtype='float32',
                             device=args.device, blocksize=0) as stream:
            while True:
                item = batches.get()
                if item is sentinel:
                    break
                counter, audio = item
                # sounddevice requires a C-contiguous interleaved buffer;
                # filtering/resampling can return a strided view here.
                stream.write(np.ascontiguousarray(audio, dtype=np.float32))
                print(f'  sent through frame {counter+len(audio)//P.PULSE_FRAME-1}',
                      flush=True)
    except KeyboardInterrupt:
        stop.set()
    finally:
        stop.set()
        worker.join(timeout=2)
    print(f'V7 send stopped after {total} frames', flush=True)


def run_receive(args):
    import sounddevice as sd

    model = _model(args.fixture, args.encode_filter)
    blocks = queue.Queue(maxsize=32)
    stop = threading.Event()
    input_gap = threading.Event()
    samples = []
    processed_samples = 0
    latest = None
    diagnostics = {} if args.diagnostics else None
    auto_gain = 1.0
    meter = {'peak': np.zeros(2), 'rms': np.zeros(2), 'blocks': 0,
             'dropped': 0, 'decoded': 0, 'verified': 0, 'lost': 0,
             'status': 'acquiring', 'counter': None, 'decode_ms': None,
             'quality': '--',
             'timing_delta': None,
             'pulse': None, 'aspect': 0, 'aspect_candidate': 0,
             'aspect_streak': 0, 'input_samples': 0, 'started': time.monotonic(),
             'auto_gain': 1.0,
             'decoded_times': deque(maxlen=32), 'input_fps': 0.,
             'decoded_fps': 0.}

    def callback(indata, frames, timing, status):
        values = np.asarray(indata, float)
        meter['peak'] = np.maximum(meter['peak'],
                                   np.max(np.abs(values), axis=0))
        meter['rms'] = np.sqrt(np.mean(values*values, axis=0))
        meter['blocks'] += 1
        meter['input_samples'] += len(values)
        elapsed = max(time.monotonic()-meter['started'], 1e-6)
        meter['input_fps'] = meter['input_samples']/(P.PULSE_FRAME*elapsed)
        if status:
            print(f'input: {status}', file=sys.stderr, flush=True)
        try:
            blocks.put_nowait(np.array(indata, copy=True))
        except queue.Full:
            # Dropping an input block is an explicit discontinuity; keeping
            # stale audio would make the V7 clock appear to run backward.
            meter['dropped'] += 1
            input_gap.set()

    def decode_available():
        nonlocal latest, processed_samples, auto_gain
        if input_gap.is_set():
            # Never stitch samples across a callback drop.  Keep displaying
            # the last good image while pulse acquisition starts over.
            input_gap.clear()
            samples.clear()
            processed_samples = 0
            if args.diagnostics:
                print({'status': 'input_gap_reacquire',
                       'dropped': meter['dropped']}, flush=True)
            return
        while True:
            try:
                samples.append(blocks.get_nowait())
            except queue.Empty:
                break
        if not samples:
            return
        audio = np.concatenate(samples)
        # Keep enough clock history for reacquisition while bounding memory.
        if len(audio) > P.RATE*30:
            del samples[:-int(P.RATE*20)//1024]
            audio = np.concatenate(samples)
        if len(audio) < P.PULSE_FRAME:
            return
        if len(audio)-processed_samples < P.PULSE_FRAME*args.decode_batch:
            return
        # Do not repeatedly run pulse acquisition/demodulation on an idle
        # input device.  This is deliberately far below normal loopback levels
        # so quiet but valid recordings still reach measure_pulses().
        if float(np.max(np.abs(audio))) < 1e-5:
            processed_samples = len(audio)
            keep = P.PULSE_FRAME*args.decode_history
            if len(audio) > keep:
                samples[:] = [audio[-keep:]]
                processed_samples = len(samples[0])
            if args.diagnostics:
                print({'status': 'idle_input', 'input_peak': 0.0}, flush=True)
            return
        peak = float(np.percentile(np.abs(audio[-P.PULSE_FRAME:]), 99.5))
        desired_gain = float(np.clip(.55/max(peak, 1e-6), .5, 32.0))
        auto_gain = (min(desired_gain, auto_gain*1.5)
                     if desired_gain > auto_gain else desired_gain)
        meter['auto_gain'] = auto_gain
        if not args.refine:
            P.REFINE = False
        try:
            results, info = P.decode_pulse_stream(
                model, audio, diagnostics=diagnostics, latest_only=True,
                input_gain=auto_gain)
        except Exception as exc:
            # Drop the damaged window and let the next retained clock history
            # reacquire.  A single bad frame must not stop the live receiver.
            results, info = [], {'words': 0,
                                 'recovery_error': type(exc).__name__}
        processed_samples = len(audio)
        if results:
            result = results[-1]
            # Each call is gated by newly arrived audio.  The short rolling
            # history intentionally restarts the prototype's local counter,
            # so comparing result.counter here would suppress valid frames.
            meter['decoded'] += 1
            now = time.monotonic()
            meter['decoded_times'].append(now)
            times = meter['decoded_times']
            if len(times) >= 2:
                meter['decoded_fps'] = (len(times)-1)/(times[-1]-times[0])
            displayable = bool(result.diag.get('displayable', False))
            meter['status'] = ('degraded' if displayable and result.status == 'lost'
                               else result.status)
            meter['counter'] = meter['decoded']
            meter['pulse'] = result.diag.get('pulse_confidence')
            meter['timing_delta'] = result.diag.get('timing_delta_ppm')
            meter['quality'] = (
                f'head {result.diag.get("head_confidence", 0):.2f}/'
                f'{result.diag.get("head_coverage", 0):.2f}')
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
                latest = P.values_from(model, result.coeffs)
            if result.status in ('received', 'verified'):
                meter['verified'] += 1
            if result.status == 'lost' and not displayable:
                meter['lost'] += 1
            report = {'counter': meter['decoded'], 'wire_counter': result.counter,
                      'status': meter['status'], 'displayable': displayable,
                      'clock_words': info.get('words'),
                      'input_gain': round(meter['auto_gain'], 3),
                      'head_confidence': result.diag.get('head_confidence'),
                      'head_coverage': result.diag.get('head_coverage'),
                      'metadata_valid': result.diag.get('metadata_valid'),
                      'timing_delta_ppm': result.diag.get('timing_delta_ppm'),
                      'noise': result.diag.get('noise'),
                      'crc_ok': info.get('crc_ok'),
                      'skipped_frames': len(info.get('skipped_frames', [])),
                      'recovered': info.get('recovered', False)}
            if args.diagnostics:
                report['diagnostics'] = info.get('diagnostics')
            print(report, flush=True)
            if args.save_dir:
                args.save_dir.mkdir(parents=True, exist_ok=True)
                values_image(latest, model.coder.grids).save(
                    args.save_dir/f'v7_{meter["decoded"]:08d}.png')
        elif args.diagnostics:
            print({'status': 'reacquiring', **info}, flush=True)
        # Keep the receiver's expensive non-streaming prototype bounded.  The
        # next decode reacquires from this short clock history instead of
        # repeatedly decoding an ever-growing capture.
        keep = P.PULSE_FRAME*args.decode_history
        if len(audio) > keep:
            samples[:] = [audio[-keep:]]
            processed_samples = len(samples[0])

    try:
        stream = sd.InputStream(samplerate=P.RATE, channels=2, dtype='float32',
                                device=args.device, blocksize=1024,
                                callback=callback)
        stream.start()
    except Exception:
        stop.set()
        raise

    root = None
    label = None
    if not args.headless:
        import tkinter as tk
        from PIL import ImageTk
        root = tk.Tk()
        root.title('V7 modem receiver')
        root.resizable(False, False)
        root.geometry('640x650')
        root.configure(background='black')
        image_frame = tk.Frame(root, width=620, height=480,
                               background='black')
        image_frame.pack_propagate(False)
        image_frame.pack(padx=10, pady=8)
        label = tk.Label(image_frame, text='Acquiring V7 clock…',
                         background='black')
        label.configure(background='black')
        label.pack(expand=True, fill='both')
        device_label = tk.Label(root, text=f'V7 input: {args.device}',
                                background='black', foreground='white')
        device_label.pack(fill='x', padx=6)
        status_label = tk.Label(root, text='status: acquiring', anchor='w',
                                background='black', foreground='white')
        status_label.pack(fill='x', padx=6)
        levels_label = tk.Label(root, text='levels: --', anchor='w',
                                font='TkFixedFont', background='black',
                                foreground='white')
        levels_label.pack(fill='x', padx=6)
        stats_label = tk.Label(root, text='frames: --', anchor='w',
                               font='TkFixedFont', background='black',
                               foreground='white')
        stats_label.pack(fill='x', padx=6)
        quality_label = tk.Label(root, text='quality: --', anchor='w',
                                 font='TkFixedFont', background='black',
                                 foreground='white')
        quality_label.pack(fill='x', padx=6)

        def tick():
            try:
                decode_available()
            except Exception as exc:
                # Keep the UI alive through an unexpected damaged-frame
                # exception; the next pulse window can still reacquire.
                meter['status'] = f'decoder error: {type(exc).__name__}'
                if args.diagnostics:
                    print({'status': 'decoder_exception',
                           'error': repr(exc)}, flush=True)
            peak = 20*np.log10(np.maximum(meter['peak'], 1e-9))
            rms = 20*np.log10(np.maximum(meter['rms'], 1e-9))
            levels_label.configure(text=(
                f'peak L/R {peak[0]:6.1f}/{peak[1]:6.1f} dBFS | '
                f'rms {rms[0]:6.1f}/{rms[1]:6.1f} dBFS'))
            stats_label.configure(text=(
                f'frames {meter["decoded"]}  verified {meter["verified"]} '
                f'lost {meter["lost"]}  input blocks {meter["blocks"]} '
                f'dropped {meter["dropped"]}'))
            quality_label.configure(text=f'quality: {meter["quality"]}')
            status_label.configure(text=(
                f'status {meter["status"]}  frame {meter["counter"]} '
                f'aspect {meter["aspect"]} '
                f'(candidate {meter["aspect_candidate"]} '
                f'x{meter["aspect_streak"]})  pulse '
                f'{meter["pulse"] if meter["pulse"] is not None else "--"} '
                f'gain {meter["auto_gain"]:4.1f}x  '
                f'timing {meter["timing_delta"] if meter["timing_delta"] is not None else "--"} ppm  '
                f'decode {meter["decode_ms"] if meter["decode_ms"] is not None else "--"} ms | '
                f'incoming {meter["input_fps"]:5.2f} fps | '
                f'decoded {meter["decoded_fps"]:5.2f} fps'))
            if latest is not None:
                image = values_image(latest, model.coder.grids)
                height = 480
                width = max(1, round(height*ASPECT_RATIOS[int(meter['aspect']) & 7]))
                scale = min(620/width, 480/height)
                image = image.resize((max(1, round(width*scale)),
                                      max(1, round(height*scale))),
                                     Image.Resampling.NEAREST)
                photo = ImageTk.PhotoImage(image)
                label.configure(image=photo, text='')
                label.image = photo
            root.after(10, tick)
        root.after(10, tick)
        try:
            root.mainloop()
        except KeyboardInterrupt:
            pass
    else:
        try:
            while not stop.is_set():
                decode_available()
                time.sleep(.01)
        except KeyboardInterrupt:
            pass
    stream.stop(); stream.close()


def parser():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest='mode', required=True)
    send = sub.add_parser('send', help='capture camera/screen and transmit V7')
    send.add_argument('--source', choices=('camera', 'screen'), required=True)
    send.add_argument('--device', required=True,
                      help='explicit sounddevice output, e.g. BlackHole 2ch')
    send.add_argument('--fixture', type=Path, default=DEFAULT_FIXTURE)
    send.add_argument('--encode-filter', choices=('nearest', 'box', 'lanczos', 'bicubic'),
                      default='nearest')
    send.add_argument('--camera', type=int, default=0)
    send.add_argument('--display', type=int)
    send.add_argument('--ffmpeg-input')
    send.add_argument('--screen-backend', choices=('mss', 'ffmpeg'), default='mss',
                      help='screen capture backend; mss avoids an FFmpeg child')
    send.add_argument('--region')
    send.add_argument('--capture-width', type=int, default=320)
    send.add_argument('--capture-fps', '--fps', dest='capture_fps', type=float)
    send.add_argument('--batch-frames', type=int, default=1,
                      help='frames encoded before submission (default: 1)')
    send.add_argument('--seconds', type=float, default=0,
                      help='0 means until Ctrl-C')
    recv = sub.add_parser('receive', help='receive V7 audio and display it')
    recv.add_argument('--device', required=True,
                      help='explicit sounddevice input, e.g. BlackHole 2ch')
    recv.add_argument('--fixture', type=Path, default=DEFAULT_FIXTURE)
    recv.add_argument('--encode-filter', choices=('nearest', 'box', 'lanczos', 'bicubic'),
                      default='nearest')
    recv.add_argument('--headless', action='store_true')
    recv.add_argument('--save-dir', type=Path)
    recv.add_argument('--diagnostics', action='store_true',
                      help='print decoder stage timing and counters')
    recv.add_argument('--decode-batch', type=int, default=1,
                      help='new frames required before each decode (default: 1)')
    recv.add_argument('--decode-history', type=int, default=1,
                      help='frames retained for clock reacquisition (default: 1)')
    recv.add_argument('--refine', action='store_true',
                      help='enable slower clock-template refinement')
    return ap


if __name__ == '__main__':
    args = parser().parse_args()
    if args.mode == 'send':
        run_send(args)
    else:
        run_receive(args)
