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
from tools import v7_proto as P                                             # noqa: E402


FPS = P.RATE / P.FRAME
DEFAULT_FIXTURE = ROOT / 'modem_tests/fixtures/v6_face_1110.png'


def _capture(args):
    """Build one of modem_screen's existing RGB capture sources."""
    from modem_screen import camera_source, screen_capture_source, _region

    region = _region(args.region)
    if args.source == 'camera':
        return camera_source(args.camera, args.capture_fps or 30,
                             width=args.capture_width, spec=args.ffmpeg_input)
    return screen_capture_source(args.capture_fps or FPS, region, args.display,
                                 args.capture_width, args.ffmpeg_input)


def _model(fixture):
    model = P.build_model(fixture, .1521 / np.sqrt(
        1 + 10**(P.CLOCK_REL_DB/10)))
    return model


def _values(model, frame):
    image = frame if isinstance(frame, Image.Image) else Image.fromarray(frame)
    prepared = prepare_image(image, preset='auto', encode_filter='lanczos')
    return image_values(prepared, model.coder.grids)


def run_send(args):
    import sounddevice as sd

    model = _model(args.fixture)
    grab = _capture(args)
    batches = queue.Queue(maxsize=2)
    stop = threading.Event()
    sentinel = object()
    batch_size = max(1, args.batch_frames)
    total = 0
    started = time.monotonic()

    def produce():
        nonlocal total
        frames = []
        counter = 1
        next_capture = time.monotonic()
        try:
            while not stop.is_set() and (args.seconds <= 0 or
                                         time.monotonic()-started < args.seconds):
                delay = next_capture-time.monotonic()
                if delay > 0:
                    time.sleep(delay)
                frames.append(_values(model, grab()))
                next_capture += 1/FPS
                if len(frames) < batch_size:
                    continue
                values = np.asarray(frames)
                audio = P.encode_stream(model, values, len(frames), lead=0,
                                        tail=0, start_counter=counter)
                batches.put((counter, audio))
                total += len(frames)
                counter += len(frames)
                frames = []
        finally:
            if frames and not stop.is_set():
                values = np.asarray(frames)
                batches.put((counter, P.encode_stream(
                    model, values, len(frames), lead=0, tail=0,
                    start_counter=counter)))
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
                stream.write(np.asarray(audio, np.float32))
                print(f'  sent through frame {counter+len(audio)//P.FRAME-1}',
                      flush=True)
    except KeyboardInterrupt:
        stop.set()
    finally:
        stop.set()
        worker.join(timeout=2)
    print(f'V7 send stopped after {total} frames', flush=True)


def run_receive(args):
    import sounddevice as sd

    model = _model(args.fixture)
    blocks = queue.Queue(maxsize=32)
    stop = threading.Event()
    samples = []
    processed_samples = 0
    last_counter = None
    latest = None
    diagnostics = {} if args.diagnostics else None

    def callback(indata, frames, timing, status):
        if status:
            print(f'input: {status}', file=sys.stderr, flush=True)
        try:
            blocks.put_nowait(np.array(indata, copy=True))
        except queue.Full:
            # Dropping an input block is an explicit discontinuity; keeping
            # stale audio would make the V7 clock appear to run backward.
            pass

    def decode_available():
        nonlocal latest, last_counter, processed_samples
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
        if len(audio) < P.FRAME*3:
            return
        if len(audio)-processed_samples < P.FRAME*args.decode_batch:
            return
        if not args.refine:
            P.REFINE = False
        results, info = P.decode_stream(model, audio, diagnostics=diagnostics)
        processed_samples = len(audio)
        if not results:
            return
        result = results[-1]
        if last_counter == result.counter:
            return
        last_counter = result.counter
        latest = P.values_from(model, result.coeffs)
        report = {'counter': result.counter, 'status': result.status,
                  'clock_words': info.get('words'), 'crc_ok': info.get('crc_ok')}
        if args.diagnostics:
            report['diagnostics'] = info.get('diagnostics')
        print(report, flush=True)
        if args.save_dir:
            args.save_dir.mkdir(parents=True, exist_ok=True)
            values_image(latest, model.coder.grids).save(
                args.save_dir/f'v7_{result.counter:08d}.png')
        # Keep the receiver's expensive non-streaming prototype bounded.  The
        # next decode reacquires from this short clock history instead of
        # repeatedly decoding an ever-growing capture.
        keep = P.FRAME*args.decode_history
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
        label = tk.Label(root, text='Acquiring V7 clock…')
        label.pack()

        def tick():
            decode_available()
            if latest is not None:
                image = values_image(latest, model.coder.grids)
                image = image.resize((400, 480), Image.Resampling.NEAREST)
                photo = ImageTk.PhotoImage(image)
                label.configure(image=photo, text='')
                label.image = photo
            root.after(100, tick)
        root.after(100, tick)
        try:
            root.mainloop()
        except KeyboardInterrupt:
            pass
    else:
        try:
            while not stop.is_set():
                decode_available()
                time.sleep(.1)
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
    send.add_argument('--camera', type=int, default=0)
    send.add_argument('--display', type=int)
    send.add_argument('--ffmpeg-input')
    send.add_argument('--region')
    send.add_argument('--capture-width', type=int, default=320)
    send.add_argument('--capture-fps', type=float)
    send.add_argument('--batch-frames', type=int, default=8)
    send.add_argument('--seconds', type=float, default=0,
                      help='0 means until Ctrl-C')
    recv = sub.add_parser('receive', help='receive V7 audio and display it')
    recv.add_argument('--device', required=True,
                      help='explicit sounddevice input, e.g. BlackHole 2ch')
    recv.add_argument('--fixture', type=Path, default=DEFAULT_FIXTURE)
    recv.add_argument('--headless', action='store_true')
    recv.add_argument('--save-dir', type=Path)
    recv.add_argument('--diagnostics', action='store_true',
                      help='print decoder stage timing and counters')
    recv.add_argument('--decode-batch', type=int, default=4,
                      help='new frames required before each decode (default: 4)')
    recv.add_argument('--decode-history', type=int, default=6,
                      help='frames retained for clock reacquisition (default: 6)')
    recv.add_argument('--refine', action='store_true',
                      help='enable slower clock-template refinement')
    return ap


if __name__ == '__main__':
    args = parser().parse_args()
    if args.mode == 'send':
        run_send(args)
    else:
        run_receive(args)
