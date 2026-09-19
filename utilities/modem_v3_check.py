#!/usr/bin/env python3
"""Self-describing v3 image/audio link.

Stereo, deterministic built-in allocation, 16-byte CRC header. Nothing about
the format has to be named at both ends: the receiver takes the sample rate
from its device, the picture geometry from the header, and the wire layout by
decoding against each candidate until one verifies.

No sample rate is requested of any device. Streams open at whatever rate the
device is already set to, and the rate they report is read back and used for
buffer sizing and for reporting speeds, durations and carrier frequencies.
Files carry their own rate in the header. 48 kHz remains the reference the
geometry was designed against, and is what `write` stamps into a new file.

  python utilities/modem_v3_check.py write --modem-dir images_modem --out clean.wav
  python utilities/modem_v3_check.py read --wav clean.wav
  python utilities/modem_v3_check.py live-receive --device "BlackHole 2ch"
  python utilities/modem_v3_check.py live-send --device "BlackHole 2ch" --modem-dir images_modem
"""
import argparse
from contextlib import closing
import json
from pathlib import Path
import sys
import time
import wave

import numpy as np
from PIL import Image, ImageFilter

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from animation_modem.live_picture import LivePicture
from animation_modem import transport3 as V3                   # noqa: E402
from animation_modem import engines as ENG                     # noqa: E402
from animation_modem import impairments as IMP                 # noqa: E402
from animation_modem.audio_common import (pcm, pair, device,   # noqa: E402
                                          wav_blocks, wav_rate, wire_notice,
                                          InputLevel, sounddevice)
from animation_modem.imaging import (DEFAULT_PROFILE, burn_counters,   # noqa: E402
                                     image_values, values_image, prepare_image, display_image,
                                     ENCODE_FILTERS,
                                     wire_profiles)
from animation_modem.aspect import ASPECT_CHOICES, aspect_code
# Decoded-picture window, in screen pixels. 480x576 is a whole-number 12x of a
# 40x48 picture and 6x of color-dct's 80x96, so neither lands on a fractional
# scale and neither needs resampling to fill it.
WINDOW = (480, 576)
# Raw preserves decoded pixels; smooth interpolation is explicitly opt-in.
SCALING = ('raw', 'smooth')

# Back-compat aliases: the v3 engine is the default, and these names are what
# the tests and fit_allocation import.
WIRE = V3.WIRE
REFERENCE_RATE = V3.REFERENCE_RATE
RATE = REFERENCE_RATE


def frames_from(args, profile=DEFAULT_PROFILE):
    """Real composites from a bake, or a synthetic stand-in."""
    if args.modem_dir:
        from modem_bake import ModemLibrary
        library = ModemLibrary(args.modem_dir)
        picks = range(0, min(library.frames, args.frames*args.stride), args.stride)
        return ([library.composite(i, 0, 0) for i in picks], profile)
    out = []
    for t in range(args.frames):
        rng = np.random.default_rng(4242)
        yy, xx = np.mgrid[0:48, 0:40]
        cx, cy = 20 + 4*np.sin(t*.3), 28 + 3*np.cos(t*.2)
        mask = ((((xx-cx)/8.8)**2 + ((yy-cy)/16.3)**2) < 1) | \
               ((((xx-cx)/5.2)**2 + ((yy-cy+18)/6.2)**2) < 1)
        tex = np.asarray(Image.fromarray(np.uint8(np.clip(
            150+55*rng.standard_normal((48, 40)), 0, 255))).filter(
                ImageFilter.GaussianBlur(1.1)), float)
        im = np.full((48, 40, 3), 4.)
        for c, k in enumerate((1., .82, .70)):
            im[:, :, c] = np.where(mask, tex*k, 4)
        out.append(Image.fromarray(np.uint8(np.clip(im, 0, 255))))
    return out, profile


def coder_for(profile, layout=None):
    """Back-compat: build the coder for a profile (DCT or wavelet by name)."""
    return ENG.coder_for(profile, layout)


def coders_for(layout):
    """code -> coder for every profile the header may name, across all engines.

    The receiver follows the header's declaration, so it is told which engine
    (DCT or wavelet) sent the picture rather than guessing out of band.
    """
    return ENG.coders_for(layout)


def candidates_for(engine):
    """Every wire any engine sends, each carrying every coder its header may name.

    v5 rides WIRE_HD (3488-sample frames), not WIRE, so a receiver offered
    only its own --codec's wire heard a v5 sender as nothing at all -- no
    error, zero packets. The Receiver already tries candidates in packet
    order and locks onto whichever verifies (relocking after RELOCK_AFTER
    misses), so offering all wires makes --codec a hint, not a requirement.
    The requested engine's wire goes first so its coder is the fallback.
    """
    out, seen = [], set()
    for eng in [engine, *ENG.ENGINES.values()]:
        if eng.wire.name in seen:
            continue
        seen.add(eng.wire.name)
        coders = coders_for(eng.wire)
        out.append((eng.wire, coders[eng.profile_code(eng.profiles[0])], coders))
    return out


def receive_for(args, engine, layout, coder, input_rate=None, coders=None,
                candidates=None):
    return engine.receiver(layout, coder, recovery=False, fast=True,
                           input_rate=input_rate, coders=coders,
                           candidates=candidates,
                           diagnostics=getattr(args, 'verbose', False))


def _prepared(image, absolute, count, numbered=False):
    return burn_counters(image, absolute, absolute, count) if numbered else image


def psnr(a, b):
    x = np.asarray(a.convert('RGB'), float)
    y = np.asarray(b.convert('RGB'), float)
    err = np.mean((x-y)**2)
    return float('inf') if err <= 0 else 10*np.log10(255.0**2/err)


def picture_size(shapes):
    """WxH of the decoded picture, BEFORE any display scaling.

    The luma plane's shape is (rows, cols) and the picture is cols x rows.
    Worth reporting on its own rather than leaving as `shapes`, because the
    number people actually want -- how big is the image I am looking at -- is
    otherwise buried in a nested list, and the window scales it up by 6x or
    12x before anyone sees it.
    """
    if not shapes:
        return None
    rows, cols = tuple(shapes[0])
    return f'{cols}x{rows}'


def level_text(lv):
    """One-line stereo meter plus verdict, for the live window.

    lv is (peak L/R, rms L/R) in dBFS off the raw input -- the hardware
    truth, before levelling. Thresholds match utilities/audio_levels.py:
    silence under -50 on both legs, one dead leg, hot near full scale,
    and L/R imbalance past 6 dB (the wire sends identical stereo, so a
    real imbalance is routing, not content).
    """
    if lv is None:
        return 'levels: waiting for input…'
    pl, pr, _, _ = lv

    def bar(v, width=14, floor=-60.0):
        fill = int(round((max(v, floor)-floor)/(0-floor)*width))
        return '#'*fill + '.'*(width-fill)

    text = f'L [{bar(pl)}] {pl:5.1f}  R [{bar(pr)}] {pr:5.1f} dBFS pk'
    if pl < -50 and pr < -50:
        return text + '  -- SILENCE in, nothing to show'
    if pl < -50 or pr < -50:
        return text + '  -- ONE LEG DEAD, check routing'
    if max(pl, pr) > -3:
        return text + '  -- HOT, back the source off'
    if abs(pl-pr) > 6:
        return text + '  -- L/R IMBALANCE'
    return text


def record(r):
    speed = r.extra.get('playback_speed', 1/(1+r.rate_error))
    return {'status': r.status, 'identity': r.identity, 'frame': r.absolute,
            'index': r.index, 'source_index': r.source_index, 'count': r.count,
            'face_folder': r.face_folder, 'float_folder': r.float_folder,
            'tier': r.tier, 'coverage': r.coverage, 'pilot_error': r.pilot_error,
            'playback_speed': speed, 'playback_rate_pct': round(100*(speed-1), 3),
            'picture': picture_size(r.extra.get('shapes')),
            **r.extra}


def recovery_record(receiver, level, **extra):
    """Copy/paste-friendly heartbeat, even if acquisition produces no frames.

    Channel indices are zero-based: 0 = left, 1 = right. Pulse measurements
    describe the last acquisition search; last_decode may be older during loss.
    """
    return {'event': 'receiver_diagnostics', **receiver.diagnostic_state(),
            **level.diagnostic_state(), **extra}


def do_write(args):
    engine = ENG.get_engine(args.codec)
    profile = args.profile or engine.profiles[0]
    frames, _ = frames_from(args, profile)
    layout = engine.wire
    coder, _ = engine.coder_for(profile, layout)
    rate = engine.reference_rate
    fps = layout.fps_at(rate)
    print(layout.describe(rate))
    path = Path(args.out)
    with wave.open(str(path), 'wb') as sink:
        sink.setparams((2, 2, rate, 0, 'NONE', 'not compressed'))
        for n, im in enumerate(frames):
            code = aspect_code(im.size, args.aspect)
            im = prepare_image(im, args.aspect, args.encode_filter)
            values = image_values(_prepared(im, n+1, len(frames), args.numbered),
                                  coder.grids)
            audio = engine.encode(values, coder, n+1, (n % len(frames))+1,
                                  len(frames), stamp_ms=n*int(1000/fps),
                                  profile=engine.profile_code(profile), aspect_code=code)
            sink.writeframesraw(pcm(audio*args.gain))
    seconds = len(frames)/fps
    print(f'wrote {len(frames)} frames, {seconds:.1f} s, peak gain {args.gain} -> {path}')


def do_read(args):
    engine = ENG.get_engine(args.codec)
    profile = args.profile or engine.profiles[0]
    layout = engine.wire
    coder, _ = engine.coder_for(profile, layout)
    rate = wav_rate(args.wav)
    receiver = receive_for(args, engine, layout, coder, input_rate=rate,
                           coders=coders_for(layout),
                           candidates=candidates_for(engine))
    if rate != engine.reference_rate:
        print(f'{args.wav}: {rate} Hz, decoding at that rate', file=sys.stderr)
    if args.save_frames:
        Path(args.save_frames).mkdir(parents=True, exist_ok=True)
    # The audio-path emulator sits between the file and the receiver, so a
    # recorded signal can be re-impaired on the way in -- the same vocabulary
    # as do_bench ('clean'/'mild'/'rough' plus per-knob overrides). It has an
    # identity fast-path when nothing is enabled.
    emulator = IMP.Emulator(IMP.settings_from_args(args))
    tiers, rates, seen, pictures = {}, [], 0, []

    def results():
        level = InputLevel()
        samples, reported = 0, 0
        # Match live levelling block size; its slow release is block-counted.
        for block in wav_blocks(args.wav, args.channels, 256):
            samples += len(block)
            decoded = receiver.feed(emulator.process(level.process(block)))
            if args.verbose:
                for result in decoded:
                    result.extra.update(level.diagnostic_state())
                if samples-reported >= rate:
                    print(json.dumps(recovery_record(receiver, level,
                                     audio_seconds=round(samples/rate, 3))),
                          file=sys.stderr, flush=True)
                    reported = samples
            yield from decoded
        yield from receiver.flush()
        if args.verbose:
            print(json.dumps(recovery_record(receiver, level,
                             audio_seconds=round(samples/rate, 3), final=True)),
                  file=sys.stderr, flush=True)

    for r in results():
        seen += 1
        tiers[r.tier] = tiers.get(r.tier, 0) + 1
        line = record(r)
        rates.append(line['playback_speed'])
        if line.get('picture'):
            pictures.append(line['picture'])
        print(json.dumps(line), flush=True)
        if args.save_frames and r.values is not None:
            name = f'{r.absolute:06d}' if r.absolute is not None else f'x{seen:06d}'
            prof = r.extra.get('profile') or r.extra.get('profile_name')
            img = display_image(r, coder.grids)
            img.save(Path(args.save_frames)/f'frame_{name}.png')
    if rates:
        speed = float(np.median(rates))
        print(f'\n{seen} packets. tiers {tiers}. median playback speed '
              f'{speed:.4f}x' + (f'. picture {pictures[-1]}' if pictures else ''),
              file=sys.stderr)


def do_bench(args):
    """Decode the one wire across simulated channels."""
    engine = ENG.get_engine(args.codec)
    profile = args.profile or engine.profiles[0]
    frames, _ = frames_from(args, profile)
    layout = engine.wire
    coder, _ = engine.coder_for(profile, layout)
    channels = [('clean', {}),
                ('cassette-ish', dict(lowpass_hz=10000, noise_dbfs=-45)),
                ('worn deck', dict(lowpass_hz=8000, noise_dbfs=-40, crosstalk=.07))]
    versions = [('wire', engine.encode, lambda: engine.receiver(layout, coder), layout)]
    for name, settings in channels:
        for vname, enc, make, layout in versions:
            quality, hdr, tier, coverage = [], 0, {}, []
            elapsed = samples = 0
            for n, im in enumerate(frames[:args.frames]):
                ready = _prepared(im, n+1, len(frames), False)
                audio = enc(image_values(ready, coder.grids), layout, coder, n+1, 1, len(frames))
                emulator = IMP.Emulator(IMP.Settings(**settings))
                receiver = make()
                signal = np.concatenate([np.zeros((300, 2), np.float32), audio])
                samples += len(signal)
                got = []
                start = time.perf_counter()
                for i in range(0, len(signal), 256):
                    got += receiver.feed(emulator.process(signal[i:i+256]))
                got += receiver.flush()
                elapsed += time.perf_counter()-start
                good = [g for g in got if g.values is not None]
                if good:
                    quality.append(psnr(values_image(good[0].values, coder.shapes).resize(ready.size), ready))
                    tier[good[0].tier] = tier.get(good[0].tier, 0)+1
                    coverage.append(good[0].coverage or 0)
                    hdr += good[0].identity == 'verified_header'


def do_live_send(args):
    sd = sounddevice()
    if args.list_devices:
        print(sd.query_devices()); return
    engine = ENG.get_engine(args.codec)
    profile = args.profile or engine.profiles[0]
    frames, _ = frames_from(args, profile)
    layout = engine.wire
    coder, _ = engine.coder_for(profile, layout)
    state = {'packet': 0, 'position': 0, 'sent': 0}
    channels = args.channels
    packets = []

    def callback(outdata, count, timing, status):
        outdata.fill(0)
        written = 0
        while written < count:
            block = packets[state['packet']]
            take = min(count-written, len(block)-state['position'])
            outdata[written:written+take, channels] = block[state['position']:state['position']+take]
            written += take
            state['position'] += take
            if state['position'] >= len(block):
                state['position'] = 0
                state['packet'] = (state['packet']+1) % len(packets)
                state['sent'] += 1

    stream = sd.OutputStream(channels=max(channels)+1, dtype='float32',
                             device=args.device, blocksize=256, latency='low',
                             callback=callback)
    try:
        rate = float(stream.samplerate)
        emitted = engine.emit_length(layout.frame, rate)
        fps = rate/emitted
        packets.extend(engine.adapt(engine.encode(image_values(
                           _prepared(prepare_image(im, args.aspect, args.encode_filter), n+1,
                                     len(frames), args.numbered), coder.grids),
                       coder, n+1, (n % len(frames))+1, len(frames),
                       stamp_ms=n*int(1000/fps),
                        profile=engine.profile_code(profile),
                        aspect_code=aspect_code(im.size, args.aspect))*args.gain, rate)
                       for n, im in enumerate(frames))
        stream.start()
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            pass
    finally:
        stream.close()


def do_live_receive(args):
    import threading
    import signal
    import queue
    sd = sounddevice()
    if args.list_devices:
        print(sd.query_devices()); return
    engine = ENG.get_engine(args.codec)
    profile = args.profile or engine.profiles[0]
    layout = engine.wire
    coder, _ = engine.coder_for(profile, layout)
    verbose = args.verbose or args.headless
    if args.silent:
        verbose = False
    if args.save_frames:
        Path(args.save_frames).mkdir(parents=True, exist_ok=True)
    stop = threading.Event()
    latest = {'picture': LivePicture(args.on_loss), 'device': 'Opening audio input…',
              'levels': None}
    lock = threading.Lock()
    errors = []
    active = {'stream': None}
    opened = threading.Event()
    source = {'rate': None, 'level': InputLevel()}

    from animation_modem.audio_buffer import AudioBuffer
    minimum_buffer = ((layout.frame+255)//256)*256
    audio_buffer = AudioBuffer(max(minimum_buffer, args.buffer_frames*layout.frame), layout.frame)
    reports = queue.Queue(maxsize=1)

    def offer_report(result, summary):
        try:
            reports.put_nowait((result, summary))
        except queue.Full:
            try:
                reports.get_nowait()
            except queue.Empty:
                pass
            reports.put_nowait((result, summary))

    def report():
        try:
            while not stop.is_set():
                try:
                    result, summary = reports.get(timeout=.1)
                except queue.Empty:
                    continue
                if result is not None:
                    if verbose:
                        print(json.dumps(record(result)), flush=True)
                    if args.save_frames and result.values is not None:
                        name = (f'{result.absolute:06d}' if result.absolute is not None
                                else f'x{time.monotonic_ns()}')
                        prof = result.extra.get('profile') or result.extra.get('profile_name')
                        img = display_image(result)
                        img.save(Path(args.save_frames)/f'frame_{name}.png')
                if summary is not None:
                    print(json.dumps(summary), file=sys.stderr, flush=True)
        except Exception as exc:
            if not stop.is_set():
                errors.append(exc)
                stop.set()
                audio_buffer.close()

    def capture():
        try:
            channels = (0, 1) if args.channels is None else args.channels
            info = sd.query_devices(args.device, 'input')
            count = max(channels)+1
            if count > int(info['max_input_channels']):
                raise ValueError('Selected input channels unavailable')
            with sd.InputStream(channels=count, dtype='float32',
                                device=args.device, blocksize=256) as stream:
                active['stream'] = stream
                rate = float(stream.samplerate)
                source['rate'] = rate
                opened.set()
                device_text = f'Input: {info["name"]} | {rate:g} Hz'
                with lock:
                    latest['device'] = device_text
                level = source['level']
                while not stop.is_set():
                    audio, overflowed = stream.read(256)
                    # Meter the raw input, before levelling: this is the
                    # hardware truth the window reports (a levelled signal
                    # would hide a dead or screaming leg).
                    ino = np.asarray(audio)[:, channels]
                    peak = 20*np.log10(np.maximum(
                        np.max(np.abs(ino), axis=0), 1e-9))
                    rms = 20*np.log10(np.maximum(
                        np.sqrt(np.mean(ino**2, axis=0)), 1e-9))
                    with lock:
                        latest['levels'] = (
                            float(peak[0]), float(peak[1]),
                            float(rms[0]), float(rms[1]))
                    audio_buffer.put(level.process(np.asarray(audio)[:, channels]), overflowed)
        except Exception as exc:
            if not stop.is_set():
                errors.append(exc)
        finally:
            active['stream'] = None
            opened.set()
            audio_buffer.close()

    def receive():
        try:
            while not opened.wait(.1):
                if stop.is_set():
                    return
            rate = source['rate']
            if rate is None:
                return
            receiver = engine.receiver(layout, coder, pulse_only=True, input_rate=rate,
                                       coders=coders_for(layout),
                                       candidates=candidates_for(engine),
                                       diagnostics=verbose)
            # Same audio-path wiring as read/bench: impairments sit between the
            # input stream and the demodulator.
            emulator = IMP.Emulator(IMP.settings_from_args(args))
            seen = 0
            last_summary = time.monotonic()
            peak = 0.0
            while not stop.is_set():
                batch = audio_buffer.take()
                if batch is None:
                    if audio_buffer.closed:
                        break
                    continue
                audio, gap = batch
                peak = max(peak, float(np.max(np.abs(audio))))
                if gap:
                    receiver.reset()
                results = receiver.feed(emulator.process(audio))
                seen += len(results)
                for result in results:
                    result.extra.update(complete=result.identity == 'verified_header' and result.status == 'received')
                    if verbose:
                        result.extra.update(source['level'].diagnostic_state())
                period_samples = max(256, round(receiver.layout.frame * (1+receiver.rate_error)))
                capacity = (int(rate*args.buffer_ms/1000) if args.buffer_ms is not None else args.buffer_frames*period_samples)
                audio_buffer.configure(max(256, capacity), min(period_samples, 1024))
                newest = results[-1] if results else None
                if newest is not None:
                    with lock:
                        for result in results:
                            latest['picture'].push(result, time.monotonic())
                now = time.monotonic()
                summary = None
                interval = min(args.summary_seconds, 1.) if verbose else args.summary_seconds
                if not args.silent and now-last_summary >= interval:
                    summary = {'receiver_packets': seen, 'input_rate_hz': rate,
                                'picture': (picture_size(newest.extra.get('shapes'))
                                            if newest is not None else None)}
                    if verbose:
                        summary.update(recovery_record(receiver, source['level'],
                                       input_peak_levelled=peak,
                                       display_policy=args.on_loss,
                                       latest_picture_status=(newest.status if newest else None)))
                    last_summary, peak = now, 0.0
                if (newest is not None and (verbose or args.save_frames)) or summary is not None:
                    offer_report(newest, summary)
        except Exception as exc:
            if not stop.is_set():
                errors.append(exc)
        finally:
            stop.set()
            audio_buffer.close()

    report_thread = threading.Thread(target=report, daemon=True)
    capture_thread = threading.Thread(target=capture, daemon=True)
    thread = threading.Thread(target=receive, daemon=True)
    previous_sigint = signal.signal(signal.SIGINT, lambda signum, frame: stop.set())
    try:
        report_thread.start()
        capture_thread.start()
        thread.start()
        if args.headless:
            try:
                while not stop.is_set():
                    time.sleep(.5)
            except KeyboardInterrupt:
                stop.set()
        else:
            import tkinter as tk
            from PIL import ImageTk
            # The decoded picture is tiny (80x96 for color-dct, 40x48 for
            # color-wavelet). Blow it up by an integer factor and hand only that to
            # Tk, instead of allocating a full-window RGB canvas and pasting
            # into it every frame -- that canvas churn is what made the window
            # heavy on older machines.
            window = (args.width, args.height)
            root = tk.Tk()
            def callback_error(exc_type, exc, traceback):
                errors.append(exc)
                stop.set()
                root.destroy()
            root.report_callback_exception = callback_error
            root.title('Stereo image receiver')
            root.configure(background='black')
            root.geometry(f'{window[0]}x{window[1]}')
            photo = tk.Label(root, background='black')
            photo.pack(expand=True)
            device_label = tk.Label(root, text=latest['device'],
                                    background='black', foreground='white')
            device_label.pack(fill='x', padx=6)
            status = tk.Label(root, text='Waiting for signal', width=1, height=2,
                              anchor='w', justify='left', wraplength=window[0]-12,
                              background='black', foreground='white')
            status.pack(fill='x', padx=6)
            levels = tk.Label(root, text='levels: --', font='TkFixedFont',
                              width=1, height=2, anchor='w', justify='left',
                              wraplength=window[0]-12,
                              background='black', foreground='white')
            levels.pack(fill='x', padx=6)

            def close():
                stop.set(); root.destroy()
            root.protocol('WM_DELETE_WINDOW', close)
            root.bind('<Escape>', lambda event: close())
            root.bind('<q>', lambda event: close())

            rendered = [None]

            def refresh():
                if stop.is_set():
                    root.destroy()
                    return
                with lock:
                    r = latest['picture'].current(time.monotonic())
                    device_text = latest['device']
                    level_line = level_text(latest['levels'])
                device_label.config(text=device_text)
                levels.config(text=level_line)
                if r is not rendered[0]:
                    if r is None:
                        status.config(text='Missing or damaged frame')
                    else:
                        prof = (r.extra.get('profile')
                                or r.extra.get('profile_name'))
                        scaled = display_image(r, bounds=window,
                                               smooth=args.scaling == 'smooth')
                        shown = f'{scaled.width}x{scaled.height} {args.scaling}'
                        photo.image = ImageTk.PhotoImage(scaled)
                        photo.configure(image=photo.image)
                        found = r.extra.get('preset', '?')
                        native = picture_size(r.extra.get('shapes')) or '?'
                        status.config(
                            text=f'{found} / {prof or "profile unverified"} | '
                                 f'{native} -> {shown}'
                                 f' | frame {r.absolute} | {r.status}')
                    rendered[0] = r
                root.after(10, refresh)
            refresh()
            try:
                root.mainloop()
            finally:
                stop.set()
    finally:
        stop.set()
        audio_buffer.close()
        try:
            for worker in (thread, capture_thread, report_thread):
                if worker.ident is not None:
                    worker.join(timeout=1)
        finally:
            signal.signal(signal.SIGINT, previous_sigint)
    if errors:
        raise SystemExit(str(errors[0]))


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    codec = argparse.ArgumentParser(add_help=False)
    codec.add_argument('--codec', choices=list(ENG.engine_names()), default='v3',
                       help='Codec engine: v3 is DCT, v4 is wavelet. Each revision '
                            'lives in one script so they can be A/B tested without '
                            'hunting down separate tools.')
    sub = p.add_subparsers(dest='command', required=True)
    p.set_defaults(profile=None, gain=1.0)

    def shared(q):
        q.add_argument('--profile', choices=list(wire_profiles()),
                       default=None, help='Plane geometry to SEND. Defaults to '
                       'the engine\'s primary profile. The receiver reads it '
                       'from the header.')
        q.add_argument('--modem-dir', type=Path)
        q.add_argument('--frames', type=int, default=24)
        q.add_argument('--stride', type=int, default=1)

    b = sub.add_parser('bench', parents=[codec]); shared(b)
    w = sub.add_parser('write', parents=[codec]); shared(w)
    w.add_argument('--aspect', choices=ASPECT_CHOICES, default='auto')
    w.add_argument('--encode-filter', choices=tuple(ENCODE_FILTERS), default='lanczos',
                   help='Source-to-80x96 resize filter (default: lanczos)')
    w.add_argument('--out', type=Path, default=Path('v3_test.wav'))
    w.add_argument('-f', '--numbered', action='store_true')
    r = sub.add_parser('read', parents=[codec])
    r.add_argument('--profile', choices=list(wire_profiles()),
                   default=None,
                   help='Fallback only. The profile is read from the header, '
                        'so this matters just for a packet whose header never '
                        'verifies.')
    r.add_argument('--wav', type=Path, required=True)
    r.add_argument('--channels', type=pair, default=(0, 1))
    r.add_argument('--save-frames', type=Path)
    r.add_argument('-v', '--verbose', action='store_true',
                   help='Per-channel recovery diagnostics, including no-decode heartbeats')
    IMP.add_arguments(r)
    ls = sub.add_parser('live-send', parents=[codec]); shared(ls)
    ls.add_argument('--aspect', choices=ASPECT_CHOICES, default='auto')
    ls.add_argument('--encode-filter', choices=tuple(ENCODE_FILTERS), default='lanczos',
                    help='Source-to-80x96 resize filter (default: lanczos)')
    ls.add_argument('--device', type=device)
    ls.add_argument('--channels', type=pair, default=(0, 1))
    ls.add_argument('-f', '--numbered', action='store_true')
    ls.add_argument('--list-devices', action='store_true')
    lr = sub.add_parser('live-receive', parents=[codec])
    lr.add_argument('--device', type=device)
    lr.add_argument('--channels', type=pair, default=(0, 1))
    buffering = lr.add_mutually_exclusive_group()
    buffering.add_argument('--buffer-frames', type=int, choices=(1, 2), default=2)
    buffering.add_argument('--buffer-ms', type=float)
    lr.add_argument('--save-frames', type=Path)
    lr.add_argument('--on-loss', choices=('hold', 'damaged'), default='damaged',
                    help='hold = keep the last good frame; damaged = show the '
                         'partial decode as-is. There is no black option: the '
                         'picture path never emits black.')
    lr.add_argument('--headless', action='store_true')
    lr.add_argument('-v', '--verbose', action='store_true')
    lr.add_argument('--silent', action='store_true')
    lr.add_argument('--summary-seconds', type=float, default=30.0)
    lr.add_argument('--width', type=int, default=WINDOW[0],
                    help='Decoded-picture window width in screen pixels '
                         f'(default {WINDOW[0]}). The picture itself is '
                         'whatever the profile sends -- 40x48, or 80x96 for '
                         'color-dct -- and is scaled up to fit this.')
    lr.add_argument('--height', type=int, default=WINDOW[1],
                    help=f'Window height in screen pixels (default {WINDOW[1]}). '
                         'The default pair is a whole-number multiple of both '
                         'picture sizes, 12x of 40x48 and 6x of 80x96.')
    lr.add_argument('--scaling', choices=sorted(SCALING), default='raw',
                    help='raw (default) preserves decoded pixel values using '
                         'nearest-neighbour; smooth opts into Lanczos interpolation.')
    lr.add_argument('--list-devices', action='store_true')
    IMP.add_arguments(lr)
    args = p.parse_args(argv)
    {'bench': do_bench, 'write': do_write, 'read': do_read,
     'live-send': do_live_send, 'live-receive': do_live_receive}[args.command](args)


if __name__ == '__main__':
    main()
