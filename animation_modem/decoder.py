#!/usr/bin/env python3
"""Receive stereo audio; display each independently reconstructed image.

Runs the v2 transport. The receiver is not configured against a transmitter:
its job is to make do with whatever arrives. The wire format is fixed, and
acquisition, playback rate and channel conditioning are all derived from the
incoming audio by transport2.Receiver. There is deliberately no preset,
profile, allocation, input-filter or speed switch here -- anything that had to
be kept matching at both ends would be a way to get it wrong.

Presentation follows the source. When a packet carries a plausible shared-time
stamp the frame is held until that instant, which is what keeps modem output
agreeing with local, scope and ascii on a chrony-disciplined install. When it
does not -- anything played back off tape, where a recorded timestamp says
nothing about the current wall clock -- the newest decoded frame is shown as
soon as it lands. Nothing is queued in that mode and nothing waits.
"""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys
import threading
import time

import numpy as np

from .audio_common import device, pair, sounddevice, wav_blocks
from .imaging import DEFAULT_PROFILE, fit_shapes, plane_shapes, values_image
from .transport2 import PRESETS, RATE, Receiver, SourceCoder
from .timing import expand_timestamp, TimingStats, PresentationBuffer
from .impairments import Emulator, add_arguments, settings_from_args

# A stamp further from now than this cannot be a live shared-time deadline, so
# it is treated as a recording and presented immediately. Decided from the
# packet, not from a flag.
LIVE_STAMP_WINDOW_NS = 30_000_000_000
FIXED_PRESET = 'wide'
FIXED_PROFILE = DEFAULT_PROFILE


def build():
    """The fixed receive format. Not negotiated and not configurable."""
    layout = PRESETS[FIXED_PRESET]
    return layout, SourceCoder(fit_shapes(plane_shapes(FIXED_PROFILE), layout.capacity))


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--device', type=device, help='Input device ID or name substring')
    p.add_argument('--channels', type=pair, default=(0, 1), help='Input pair, 1-based (default 1,2)')
    p.add_argument('--wav', type=Path, help='Receive a baked/recorded 48 kHz PCM16 WAV')
    p.add_argument('--list-devices', action='store_true')
    p.add_argument('--quiet', action=argparse.BooleanOptionalAction, default=False,
                   help='Suppress per-frame JSON, including the decode/display timing records')
    p.add_argument('-v', '--verbose', dest='quiet', action='store_false',
                   help='Per-frame JSON records, including decode and display timing')
    p.add_argument('--headless', action='store_true', help='JSON reporting without a display')
    p.add_argument('--fast', action='store_true', help='Decode WAV without real-time pacing')
    p.add_argument('--save-frames', type=Path, help='Optional PNG directory; saving adds processing cost')
    add_arguments(p)
    args = p.parse_args(argv)
    if args.list_devices:
        print(sounddevice().query_devices()); return
    if args.fast and not args.wav:
        p.error('--fast requires --wav')
    try:
        settings = settings_from_args(args)
        layout, coder = build()
    except (ValueError, OSError) as exc:
        p.error(str(exc))
    print(json.dumps({'audio_emulation': asdict(settings), 'preset': args.emulate,
                      'layout': layout.describe()}), file=sys.stderr)
    if args.save_frames:
        args.save_frames.mkdir(parents=True, exist_ok=True)
    updates = PresentationBuffer()
    decode_stats = TimingStats()
    display_stats = TimingStats()
    stop = threading.Event()
    errors = []
    log_lock = threading.Lock()

    def log(record):
        if not args.quiet:
            with log_lock:print(json.dumps(record),flush=True)

    def schedule(result, now_ns):
        """Shared-time deadline if this looks live, otherwise present at once."""
        if args.wav or not result.stamp_ms:
            return None
        target = expand_timestamp(result.stamp_ms, now_ns)
        if abs(target - now_ns) > LIVE_STAMP_WINDOW_NS:
            return None            # a recording, not a deadline
        return target

    def receive():
        receiver = Receiver(layout, coder)
        emulator = Emulator(settings)
        def process(audio):
            for result in receiver.feed(emulator.process(audio)):
                emit(result)
        def emit(result):
            now_ns = time.time_ns()
            target = schedule(result, now_ns)
            # Always present, so every record has the same shape whether or not
            # this source is schedulable.
            result.target_time_ns = target
            result.decode_error_ms = None
            timing = None
            if target is not None:
                result.decode_error_ms = (now_ns-target)/1e6
                timing = decode_stats.record(result.decode_error_ms)
            if not args.headless:
                updates.put(result, target if target is not None else now_ns)
            record = {k: v for k, v in vars(result).items() if k != 'values'}
            record.pop('extra', None)
            # Derived properties are invisible to vars().
            record['source_index'] = result.source_index
            record['face_folder'] = result.face_folder
            record['float_folder'] = result.float_folder
            record.update(result.extra)
            if timing:record['decode_error_window']=timing
            log(record)
            if result.values is not None and args.save_frames:
                name = f'{result.absolute:010d}' if result.absolute is not None else 'unknown'
                values_image(result.values, coder.shapes).save(
                    args.save_frames / f'frame_{name}.png')
        try:
            if args.wav:
                start = time.monotonic()
                samples = 0
                for audio in wav_blocks(args.wav, args.channels, 1024):
                    if stop.is_set(): break
                    samples += len(audio)
                    if not args.fast:
                        stop.wait(max(0, start + samples/RATE-time.monotonic()))
                    process(audio)
                for result in receiver.flush():
                    emit(result)
            else:
                sd = sounddevice()
                with sd.InputStream(samplerate=RATE, channels=max(args.channels)+1,
                                    device=args.device, dtype='float32',
                                    blocksize=256, latency='low') as stream:
                    print(json.dumps({'input_latency_ms': stream.latency*1000}), file=sys.stderr)
                    while not stop.is_set():
                        audio, overflow = stream.read(256)
                        if overflow:
                            log({'status': 'input_overflow', 'identity': 'unknown'})
                        process(audio[:, args.channels])
        except Exception as exc:
            errors.append(exc)
            if not args.headless:
                updates.put(str(exc), time.time_ns())
        finally:
            stop.set() if args.headless else None

    if args.headless:
        receive()
    else:
        import tkinter as tk
        from PIL import Image, ImageTk, ImageOps
        size = (400, 480)
        root = tk.Tk()
        root.title('Stereo image receiver — waiting for synchronization')
        initial = ImageTk.PhotoImage(Image.new('RGB', size, 'black'))
        label = tk.Label(root, background='black', image=initial)
        label.image = initial
        label.pack()
        status = tk.Label(root, text='Frame identity unknown — waiting for signal')
        status.pack()
        def close():
            stop.set()
            root.destroy()
        root.protocol('WM_DELETE_WINDOW', close)
        thread = threading.Thread(target=receive, daemon=True)
        thread.start()
        def refresh():
            result = updates.pop_due(time.time_ns())
            if result is not None:
                if isinstance(result, str):
                    status.config(text=result)
                elif result.values is not None:
                    # Only a real picture replaces the last one; a lost packet
                    # leaves the previous frame up rather than flashing black.
                    canvas = Image.new('RGB', size, 'black')
                    scaled = ImageOps.contain(values_image(result.values, coder.shapes),
                                              size, Image.Resampling.NEAREST)
                    canvas.paste(scaled, ((size[0]-scaled.width)//2, (size[1]-scaled.height)//2))
                    photo = ImageTk.PhotoImage(canvas)
                    label.configure(image=photo, width=size[0], height=size[1])
                    label.image = photo
                    name = result.absolute if result.absolute is not None else 'unknown'
                    source = (f'{result.source_index}/{result.count}'
                              if result.source_index is not None else 'unknown')
                    folders = ('?' if result.face_folder is None
                               else f'{result.face_folder}/{result.float_folder}')
                    status.config(text=f'Frame {name} | source {source} | face/float {folders} '
                                       f'| {result.status} | {result.identity} '
                                       f'| tier {result.tier}')
                    root.title(f'Stereo image — {layout.name} — frame {name} — source {source}')
                    if result.target_time_ns is not None:
                        error_ms=(time.time_ns()-result.target_time_ns)/1e6
                        log({'event':'display_submit','frame':result.absolute,
                            'target_time_ns':result.target_time_ns,
                            'display_submit_error_ms':error_ms,
                            'display_error_window':display_stats.record(error_ms),
                            'presentation_frames_skipped':updates.dropped})
                else:
                    status.config(text=f'{result.status} | {result.identity}')
            if not stop.is_set(): root.after(10, refresh)
        refresh()
        try:
            root.mainloop()
        finally:
            stop.set()
            thread.join(timeout=1)
    if errors:
        raise RuntimeError(str(errors[0])) from errors[0]


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        pass
    except (ValueError, RuntimeError, OSError) as exc:
        sys.exit(str(exc))
