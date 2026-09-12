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
nothing about the current wall clock -- complete pictures appear immediately.
Partials wait until half the previous received packet duration from the current
packet start, then refine over the held picture. Only the newest unscheduled
reconstruction is retained.
"""
import argparse
from contextlib import closing
from dataclasses import asdict
import json
from pathlib import Path
import sys
import threading
import time

import numpy as np

from .audio_common import device, pair, sounddevice, wav_blocks
from .capture import AudioGate, BufferedInput, CaptureHealth, CaptureResampler, open_input
from .imaging import DEFAULT_PROFILE, fit_shapes, plane_shapes, values_image
from .transport2 import PRESETS, RATE, SourceCoder
from .progressive import Receiver
from .reference_receiver import Receiver as ReferenceReceiver
from .waveform_receiver import Receiver as WaveformReceiver
from .timing import expand_timestamp, ProgressSummary, TimingStats
from .presentation import DeadlinePresentationBuffer
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


def live_results(sd, input_device, channels, layout, coder, settings, stop, report,
                 receiver_factory=Receiver):
    """One ordinary input stream; select a pair explicitly, defaulting to 1/2."""
    if stop.is_set():return
    channels = (0, 1) if channels is None else channels
    device_info = sd.query_devices(input_device, 'input')
    channel_count = max(channels)+1
    if channel_count > int(device_info['max_input_channels']):
        raise ValueError('Selected input channels are unavailable on this device')
    emulator = Emulator(settings)
    native = getattr(receiver_factory, 'native_waveform', False)
    receiver = None if native else receiver_factory(layout, coder)
    with open_input(sd, input_device, channel_count,
                    device_info['default_samplerate']) as stream:
        capture_rate = float(stream.samplerate)
        converter = None if native else CaptureResampler(capture_rate, 2)
        read_size = 256 if native else max(1, round(256*capture_rate/RATE))
        if native:
            receiver = receiver_factory(layout, coder)
        capture = BufferedInput(stream, read_size)
        startup = {'input_latency_ms': stream.latency*1000,
                'capture_rate_hz': capture_rate,
                'decode_rate_hz': capture_rate if native else RATE,
                'modem_clock': 'packet_reference' if native else 'resampled_device_rate',
                'input_channels': [c+1 for c in channels],
                'resample_filter_delay_ms': (None if converter is None
                                              else converter.filter_delay_ms),
                'capture_buffer_capacity_ms': capture.capacity_ms}
        gate = AudioGate(capture_rate)
        announced = False
        with capture:
            health = CaptureHealth(time.monotonic())
            while not stop.is_set():
                summary = health.summary(time.monotonic())
                if summary is not None and gate.active:report(summary)
                captured = capture.read()
                if captured is None:
                    stop.wait(.002)
                    continue
                audio, overflow, skipped = captured
                if overflow or skipped:
                    gate.reset()
                    receiver.reset(preserve_timing=True)
                    emulator = Emulator(settings)
                    if converter is not None:
                        converter.reset()
                was_active = gate.active
                input_samples = len(audio)
                audio = gate.process(audio[:, channels])
                if audio is None:
                    if was_active:
                        receiver.reset(preserve_timing=True)
                        emulator = Emulator(settings)
                        if converter is not None:
                            converter.reset()
                    health = CaptureHealth(time.monotonic())
                    continue
                if not announced:
                    report(startup)
                    announced = True
                started = time.perf_counter()
                if converter is not None:
                    audio = converter.process(audio)
                if len(audio):
                    audio = emulator.process(audio)
                    for at in range(0, len(audio), 256):
                        for result in receiver.feed(audio[at:at+256]):
                            if native:
                                # Metadata only: preserve the existing scheduler's
                                # 48 kHz sample units without resampling audio or
                                # using the device rate to steer the packet clock.
                                units = RATE/capture_rate
                                result.rate_error = (1+result.rate_error)*units-1
                                result.extra['playback_speed'] = 1/(1+result.rate_error)
                                for key in ('packet_duration_samples', 'packet_age_samples'):
                                    if key in result.extra:
                                        result.extra[key] *= units
                            result.extra['input_channels'] = [c+1 for c in channels]
                            yield result
                health.record(input_samples, capture_rate, time.perf_counter()-started, overflow, skipped)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--device', type=device, help='Input device ID or name substring')
    p.add_argument('--channels', type=pair, default=(0, 1),
                   help='Ordered input pair, 1-based (default: 1,2)')
    p.add_argument('--receiver', choices=('progressive', 'reference', 'waveform'), default='progressive',
                   help='waveform: native ADC samples, packet-local reference clock')
    p.add_argument('--wav', type=Path, help='Receive a baked/recorded 48 kHz PCM16 WAV')
    p.add_argument('--list-devices', action='store_true')
    p.add_argument('-v', '--verbose', action='store_true',
                   help='Per-frame JSON records, including decode and display timing')
    p.add_argument('--silent', action='store_true',
                   help='No output at all, not even the periodic summary')
    p.add_argument('--summary-seconds', type=float, default=30.0,
                   help='Fallback digest interval when no source index is decoding '
                        '(default 30; a verified link summarises once per lap instead)')
    p.add_argument('--headless', action='store_true',
                   help='JSON reporting without a display; implies --verbose')
    p.add_argument('--fast', action='store_true', help='Decode WAV without real-time pacing')
    p.add_argument('--save-frames', type=Path, help='Optional PNG directory; saving adds processing cost')
    add_arguments(p)
    args = p.parse_args(argv)
    if args.list_devices:
        print(sounddevice().query_devices()); return
    if args.fast and not args.wav:
        p.error('--fast requires --wav')
    # A live receiver runs for hours at 14 fps, so per-frame JSON is off unless
    # asked for. --headless exists to emit those records, so it turns them on.
    verbose = args.verbose or args.headless
    if args.silent:
        verbose = False
    try:
        settings = settings_from_args(args)
        layout, coder = build()
    except (ValueError, OSError) as exc:
        p.error(str(exc))
    if args.wav and not args.silent:
        print(json.dumps({'audio_emulation': asdict(settings), 'preset': args.emulate,
                          'layout': layout.describe()}), file=sys.stderr)
    if args.save_frames:
        args.save_frames.mkdir(parents=True, exist_ok=True)
    updates = DeadlinePresentationBuffer(RATE, layout.packet, coder.count)
    decode_stats = TimingStats()
    display_stats = TimingStats()
    stop = threading.Event()
    errors = []
    log_lock = threading.Lock()

    summary = ProgressSummary(args.summary_seconds)

    def log(record):
        if verbose:
            with log_lock:print(json.dumps(record),flush=True)

    def digest(result):
        """Periodic one-liner so a quiet run is still legible."""
        if verbose or args.silent or not result.extra.get('complete', True):
            return
        if summary.record(result):
            with log_lock:print(summary.line(), file=sys.stderr, flush=True)
            summary.reset()

    def schedule(result, now_ns):
        """Shared-time deadline if this looks live, otherwise present at once."""
        if args.wav or not result.stamp_ms:
            return None
        target = expand_timestamp(result.stamp_ms, now_ns)
        if abs(target - now_ns) > LIVE_STAMP_WINDOW_NS:
            return None            # a recording, not a deadline
        return target

    def receive():
        receiver_factory = {'progressive': Receiver, 'reference': ReferenceReceiver,
                            'waveform': WaveformReceiver}[args.receiver]
        receiver = receiver_factory(layout, coder)
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
                updates.put(result, target if target is not None else now_ns, now_ns=now_ns)
            record = {k: v for k, v in vars(result).items() if k != 'values'}
            record.pop('extra', None)
            # Derived properties are invisible to vars().
            record['source_index'] = result.source_index
            record['face_folder'] = result.face_folder
            record['float_folder'] = result.float_folder
            record.update(result.extra)
            if timing:record['decode_error_window']=timing
            log(record)
            digest(result)
            if result.values is not None and args.save_frames and result.extra.get('complete', True):
                name = f'{result.absolute:010d}' if result.absolute is not None else 'unknown'
                values_image(result.values, coder.shapes).save(
                    args.save_frames / f'frame_{name}.png')
        try:
            if args.wav:
                start = time.monotonic()
                samples = 0
                for audio in wav_blocks(args.wav, args.channels or (0, 1), 1024):
                    if stop.is_set(): break
                    samples += len(audio)
                    if not args.fast:
                        stop.wait(max(0, start + samples/RATE-time.monotonic()))
                    process(audio)
                for result in receiver.flush():
                    emit(result)
            else:
                def report_input(record):
                    if not args.silent:
                        print(json.dumps(record), file=sys.stderr, flush=True)
                with closing(live_results(sounddevice(), args.device, args.channels,
                                          layout, coder, settings, stop, report_input,
                                          receiver_factory=receiver_factory)) as live:
                    for result in live:
                        emit(result)
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
        from .preview_overlay import PreviewOverlay
        overlay = PreviewOverlay(coder.shapes)
        size = (400, 480)
        root = tk.Tk()
        root.title('Stereo image receiver — waiting for synchronization')
        initial = ImageTk.PhotoImage(Image.new('RGB', size, 'black'))
        label = tk.Label(root, background='black', image=initial)
        label.image = initial
        label.pack()
        # The image determines window width; status updates never resize it.
        status = tk.Label(root, text='Frame identity unknown\nWaiting for signal',
                          width=1, height=2, anchor='w', justify='left',
                          wraplength=size[0]-12)
        status.pack(fill='x', padx=6)
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
                    scaled = ImageOps.contain(overlay.render(result),
                                              size, Image.Resampling.NEAREST)
                    canvas.paste(scaled, ((size[0]-scaled.width)//2, (size[1]-scaled.height)//2))
                    # Keep one Tk image and update its pixels in place.
                    label.image.paste(canvas)
                    name = result.absolute if result.absolute is not None else 'unknown'
                    source = (f'{result.source_index}/{result.count}'
                              if result.source_index is not None else 'unknown')
                    folders = ('?' if result.face_folder is None
                               else f'{result.face_folder}/{result.float_folder}')
                    status.config(text=f'Frame {name} | src {source} | folders {folders}\n'
                                       f'{result.status} | {result.identity} | tier {result.tier}')
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
