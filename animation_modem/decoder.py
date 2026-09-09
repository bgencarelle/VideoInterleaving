#!/usr/bin/env python3
"""Receive stereo audio; display each independently reconstructed image."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys
import threading
import time

from .audio_common import device, pair, sounddevice, wav_blocks
from .transport import Receiver, RATE, FRAME
from .timing import expand_timestamp, TimingStats, PresentationBuffer
from .impairments import Emulator, add_arguments, settings_from_args


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--device', type=device, help='Input device ID or name substring')
    p.add_argument('--channels', type=pair, default=(0, 1), help='Input pair, 1-based (default 1,2)')
    p.add_argument('--wav', type=Path, help='Receive a baked/recorded 48 kHz PCM16 WAV')
    p.add_argument('--list-devices', action='store_true')
    p.add_argument('--quiet', action=argparse.BooleanOptionalAction, default=None,
                   help='Suppress per-frame JSON. Default on with a display, off with --headless.')
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
    if args.quiet is None:
        # Per-frame printing costs terminal I/O on the thread that must keep
        # reading the input stream; a display run does not need it. --headless
        # exists to emit those records, so it stays verbose unless asked.
        args.quiet = not args.headless
    try:
        settings = settings_from_args(args)
    except ValueError as exc:
        p.error(str(exc))
    print(json.dumps({'audio_emulation': asdict(settings), 'preset': args.emulate}), file=sys.stderr)
    if args.save_frames:
        args.save_frames.mkdir(parents=True, exist_ok=True)
    updates = PresentationBuffer()
    decode_stats = TimingStats()
    display_stats = TimingStats()
    stop = threading.Event()
    done = threading.Event()
    errors = []
    log_lock = threading.Lock()

    def log(record):
        if not args.quiet:
            with log_lock:print(json.dumps(record),flush=True)

    def publish(value):
        if args.headless:return
        target=getattr(value,'target_time_ns',None)
        updates.put(value,target if target is not None else time.time_ns())

    def receive():
        rx = Receiver()
        emulator = Emulator(settings)
        def process(audio):
            for result in rx.feed(emulator.process(audio)):
                now_ns=time.time_ns()
                timing=None
                if result.target_time_ms32 is not None and not args.wav:
                    result.target_time_ns=expand_timestamp(result.target_time_ms32,now_ns)
                    result.decode_error_ms=(now_ns-result.target_time_ns)/1e6
                    timing=decode_stats.record(result.decode_error_ms)
                # Publish before terminal/file I/O. Early frames wait for their
                # shared target time; late frames can be shown at the next GUI tick.
                publish(result)
                record = {k: v for k, v in vars(result).items() if k != 'image'}
                if timing:record['decode_error_window']=timing
                if args.wav and result.target_time_ms32 is not None:
                    record['timing_mode']='replay_no_live_clock_comparison'
                log(record)
                if result.image is not None and args.save_frames:
                    name = f'{result.frame:010d}' if result.frame is not None else 'unknown'
                    result.image.save(args.save_frames / f'frame_{name}_{result.sample}.png')
        try:
            if args.wav:
                start = time.monotonic()
                samples = 0
                for audio in wav_blocks(args.wav, args.channels):
                    if stop.is_set(): break
                    samples += len(audio)
                    if not args.fast:
                        stop.wait(max(0, start + samples/RATE-time.monotonic()))
                    process(audio)
                # Finish reporting a final missing interval if present.
                import numpy as np
                process(np.zeros((33, 2), np.float32))
            else:
                sd = sounddevice()
                with sd.InputStream(samplerate=RATE, channels=max(args.channels)+1,
                                    device=args.device, dtype='float32', latency='low') as stream:
                    print(json.dumps({'input_latency_ms': stream.latency*1000}), file=sys.stderr)
                    while not stop.is_set():
                        audio, overflow = stream.read(256)
                        if overflow:
                            log({'status': 'input_overflow', 'identity': 'unknown'})
                            rx = Receiver()  # Unknown lost sample count invalidates prediction.
                        process(audio[:, args.channels])
        except Exception as exc:
            errors.append(exc)
            publish(str(exc))
        finally:
            done.set()

    if args.headless:
        receive()
    else:
        import tkinter as tk
        from PIL import Image, ImageTk, ImageOps
        root = tk.Tk()
        root.title('Stereo image receiver — waiting for synchronization')
        initial = ImageTk.PhotoImage(Image.new('RGB', (400, 480), 'black'))
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
                else:
                    im = result.image or Image.new('RGB', (40, 48), 'black')
                    canvas = Image.new('RGB', (400, 480), 'black')
                    scaled = ImageOps.contain(im, (400, 480), Image.Resampling.NEAREST)
                    canvas.paste(scaled, ((400-scaled.width)//2, (480-scaled.height)//2))
                    photo = ImageTk.PhotoImage(canvas)
                    label.configure(image=photo, width=400, height=480)
                    label.image = photo
                    name = result.frame if result.frame is not None else 'unknown'
                    status.config(text=f'Frame {name} | {result.status} | {result.identity}')
                    file_id = f'{result.index}/{result.count}' if result.index is not None else 'unknown'
                    root.title(f"Stereo image — {result.profile or 'unknown profile'} — frame {name} — file {file_id}")
                    if result.target_time_ns is not None:
                        error_ms=(time.time_ns()-result.target_time_ns)/1e6
                        log({'event':'display_submit','frame':result.frame,
                            'target_time_ns':result.target_time_ns,
                            'display_submit_error_ms':error_ms,
                            'display_error_window':display_stats.record(error_ms),
                            'presentation_frames_skipped':updates.dropped})
            if not stop.is_set(): root.after(2, refresh)
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
