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
from PIL import Image, ImageOps, ImageFilter

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from animation_modem.live_picture import LivePicture
from animation_modem import transport3 as V3                   # noqa: E402
from animation_modem import impairments as IMP                 # noqa: E402
from animation_modem.audio_common import (pcm, pair, device,   # noqa: E402
                                          wav_blocks, wav_rate, wire_notice,
                                          InputLevel, sounddevice)
from animation_modem.imaging import (DEFAULT_PROFILE, burn_counters,
                                     fit_shapes, image_values, plane_grids,
                                     plane_shapes, values_image, wire_profiles)
# Decoded-picture window, in screen pixels. 480x576 is a whole-number 12x of a
# 40x48 picture and 6x of color-dct's 80x96, so neither lands on a fractional
# scale and neither needs resampling to fill it.
WINDOW = (480, 576)
# raw = NEAREST, smooth = LANCZOS. raw is the default deliberately: this is a
# diagnostic display as much as a picture, and a smoothed one hides exactly
# what you want to see -- single dead pixels, blocking, chroma blotching. Smooth
# is there for looking at the result rather than debugging it.
SCALING = {'raw': Image.Resampling.NEAREST, 'smooth': Image.Resampling.LANCZOS}

PRESETS = V3.ALL_PRESETS
REFERENCE_RATE = V3.REFERENCE_RATE
RATE = REFERENCE_RATE


def frames_from(args, profile='color'):
    """Real composites from a bake, or a synthetic stand-in."""
    if args.modem_dir:
        from modem_bake import ModemLibrary
        library = ModemLibrary(args.modem_dir)
        picks = range(0, min(library.frames, args.frames*args.stride), args.stride)
        return ([library.composite(i, 0, 0) for i in picks],
                getattr(args, 'profile', None) or 'color')
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
    return out, getattr(args, 'profile', None) or profile


def coder_for(profile, allocation, layout, strict=True):
    shapes = plane_shapes(profile)
    grids = plane_grids(profile)
    if sum(int(np.prod(s)) for s in shapes) > layout.capacity:
        # Shrinking a truncating profile drops the finer grid with it: the
        # corner must stay a corner of something, and there is no reason to
        # believe a shrunk-then-truncated geometry is the right trade.
        shapes = fit_shapes(shapes, layout.capacity)
        grids = shapes
    table = np.load(allocation) if allocation else None
    if table is not None:
        want = int(sum(np.prod(s) for s in shapes))
        if table.shape != (want,):
            # A table is fitted for ONE (preset, profile) pair, because its
            # length is that pair's slot count. Where the caller named the pair,
            # a mismatch is a mistake and has to say so. Where we are SCANNING
            # -- candidate presets, or the four profile coders a receiver holds
            # to follow the header -- a mismatch just means this table is not
            # for that one, and falling back to the default table is how the
            # scan keeps working at all.
            if strict:
                raise SystemExit(
                    f'Allocation {allocation} has {table.size} weights but '
                    f'{layout.name}/{profile} needs {want}. A table only fits '
                    f'the preset and profile it was fitted for -- refit with '
                    f'fit_allocation.py --preset {layout.name} '
                    f'--profile {profile}.')
            table = None
    return V3.SourceCoder(shapes, table, grids=grids), grids


def coders_for(layout, allocation=None):
    """code -> coder, so the receiver follows the header's declaration.

    Every profile holds its own code, color-dct included, so there is nothing
    to opt into and nothing for the two ends to disagree about. That is the
    whole reason color-dct was given a code of its own rather than aliased onto
    color's: an alias cannot be read off the wire, and a receiver that guessed
    wrong would show a frequency-distorted picture that looks plausible.
    """
    return {V3.profile_code(name): coder_for(name, allocation, layout,
                                             strict=False)[0]
            for name in V3.PROFILE_CODES}


def live_presets():
    return [n for n, l in PRESETS.items() if l.progressive]


def candidates_for(allocation=None):
    out = []
    for name in live_presets():
        layout = PRESETS[name]
        coders = coders_for(layout, allocation)
        out.append((layout, coders[V3.profile_code('color-lean')], coders))
    return out


def receive_for(args, layout, coder, input_rate=None, coders=None,
                candidates=None):
    return V3.Receiver(layout, coder, recovery=False, fast=True,
                       input_rate=input_rate, coders=coders,
                       candidates=candidates)


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


def record(r):
    speed = r.extra.get('playback_speed', 1/(1+r.rate_error))
    return {'status': r.status, 'identity': r.identity, 'frame': r.absolute,
            'index': r.index, 'source_index': r.source_index, 'count': r.count,
            'face_folder': r.face_folder, 'float_folder': r.float_folder,
            'tier': r.tier, 'coverage': r.coverage, 'pilot_error': r.pilot_error,
            'playback_speed': speed, 'playback_rate_pct': round(100*(speed-1), 3),
            'picture': picture_size(r.extra.get('shapes')),
            **r.extra}


def do_write(args):
    frames, profile = frames_from(args)
    layout = PRESETS[args.preset]
    coder, _ = coder_for(profile, args.allocation, layout)
    fps = layout.fps_at(REFERENCE_RATE)
    print(layout.describe(REFERENCE_RATE))
    path = Path(args.out)
    with wave.open(str(path), 'wb') as sink:
        sink.setparams((2, 2, REFERENCE_RATE, 0, 'NONE', 'not compressed'))
        for n, im in enumerate(frames):
            values = image_values(_prepared(im, n+1, len(frames), args.numbered),
                                  coder.grids)
            audio = V3.encode(values, layout, coder, n+1, (n % len(frames))+1,
                              len(frames), stamp_ms=n*int(1000/fps),
                              profile=V3.profile_code(profile))
            sink.writeframesraw(pcm(audio*args.gain))
    seconds = len(frames)/fps
    print(f'wrote {len(frames)} frames, {seconds:.1f} s, peak gain {args.gain} -> {path}')


def do_read(args):
    layout = PRESETS[args.preset]
    coder, _ = coder_for(args.profile, args.allocation, layout)
    rate = wav_rate(args.wav)
    receiver = receive_for(args, layout, coder, input_rate=rate,
                           coders=coders_for(layout, args.allocation),
                           candidates=candidates_for(args.allocation))
    if rate != REFERENCE_RATE:
        print(f'{args.wav}: {rate} Hz, decoding at that rate', file=sys.stderr)
    if args.save_frames:
        Path(args.save_frames).mkdir(parents=True, exist_ok=True)
    tiers, rates, seen, pictures = {}, [], 0, []

    def results():
        level = InputLevel()
        for block in wav_blocks(args.wav, args.channels, 1024):
            yield from receiver.feed(level.process(block))
        yield from receiver.flush()

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
            img = values_image(r.values, r.extra.get('shapes', coder.grids))
            img.save(Path(args.save_frames)/f'frame_{name}.png')
    if rates:
        speed = float(np.median(rates))
        print(f'\n{seen} packets. tiers {tiers}. median playback speed '
              f'{speed:.4f}x' + (f'. picture {pictures[-1]}' if pictures else ''),
              file=sys.stderr)


def do_bench(args):
    """Compare v2 and v3 across simulated channels, at equal settings."""
    frames, profile = frames_from(args)
    layout = PRESETS[args.preset]
    coder, _ = coder_for(profile, args.allocation, layout)
    channels = [('clean', {}),
                ('cassette-ish', dict(lowpass_hz=10000, noise_dbfs=-45)),
                ('worn deck', dict(lowpass_hz=8000, noise_dbfs=-40, crosstalk=.07))]
    versions = [(name, V3.encode,
                 (lambda l=PRESETS[name]: V3.Receiver(l, coder)), PRESETS[name])
                for name in ('wide-v3', 'wide-v3-fast')]
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
    frames, profile = frames_from(args)
    if not PRESETS[args.preset].progressive:
        raise SystemExit(f'{args.preset} is v2 wire format; live needs a progressive preset')
    layout = PRESETS[args.preset]
    coder, _ = coder_for(profile, None, layout)
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
        emitted = V3.emit_length(layout.frame, rate)
        fps = rate/emitted
        packets.extend(V3.band_limited(V3.encode(image_values(
                           _prepared(im, n+1, len(frames), args.numbered), coder.grids),
                       layout, coder, n+1, (n % len(frames))+1, len(frames),
                       stamp_ms=n*int(1000/fps),
                       profile=V3.profile_code(profile))*args.gain, rate)
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
    layout = PRESETS['lean-v3']
    coder, _ = coder_for('color-lean', None, layout)
    verbose = args.verbose or args.headless
    if args.silent:
        verbose = False
    if args.save_frames:
        Path(args.save_frames).mkdir(parents=True, exist_ok=True)
    stop = threading.Event()
    latest = {'picture': LivePicture(args.on_loss), 'device': 'Opening audio input…'}
    lock = threading.Lock()
    errors = []
    active = {'stream': None}
    opened = threading.Event()
    source = {'rate': None, 'level': InputLevel()}

    from animation_modem.audio_buffer import AudioBuffer
    widest = max(PRESETS[n].frame for n in live_presets())
    minimum_buffer = ((widest+255)//256)*256
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
                        img = values_image(result.values, result.extra['shapes'])
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
            receiver = V3.Receiver(layout, coder, pulse_only=True, input_rate=rate,
                                   coders=coders_for(layout, args.allocation),
                                   candidates=candidates_for(args.allocation))
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
                results = receiver.feed(audio)
                seen += len(results)
                for result in results:
                    result.extra.update(complete=result.identity == 'verified_header' and result.status == 'received')
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
                if not args.silent and now-last_summary >= args.summary_seconds:
                    summary = {'receiver_packets': seen, 'input_rate_hz': rate,
                               'picture': (picture_size(newest.extra.get('shapes'))
                                           if newest is not None else None)}
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
            size = (args.width, args.height)
            resample = SCALING[args.scaling]
            root = tk.Tk()
            def callback_error(exc_type, exc, traceback):
                errors.append(exc)
                stop.set()
                root.destroy()
            root.report_callback_exception = callback_error
            root.title('Stereo image receiver')
            initial = ImageTk.PhotoImage(Image.new('RGB', size, 'black'))
            label = tk.Label(root, background='black', image=initial)
            label.image = initial
            label.pack()
            device_label = tk.Label(root, text=latest['device'])
            device_label.pack(fill='x', padx=6)
            status = tk.Label(root, text='Waiting for signal', width=1, height=2,
                              anchor='w', justify='left', wraplength=size[0]-12)
            status.pack(fill='x', padx=6)

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
                device_label.config(text=device_text)
                if r is not rendered[0]:
                    canvas = Image.new('RGB', size, 'black')
                    if r is not None:
                        prof = (r.extra.get('profile')
                                or r.extra.get('profile_name'))
                        img = values_image(r.values, r.extra['shapes'])
                        scaled = ImageOps.contain(img, size, resample)
                        canvas.paste(scaled, ((size[0]-scaled.width)//2, (size[1]-scaled.height)//2))
                        shown = prof or 'profile unverified'
                        found = r.extra.get('preset', '?')
                        native = picture_size(r.extra.get('shapes')) or '?'
                        status.config(
                            text=f'{found} / {shown} | {native} -> '
                                 f'{scaled.width}x{scaled.height} {args.scaling}'
                                 f' | frame {r.absolute} | {r.status}')
                    else:
                        status.config(text='Missing or damaged frame')
                    label.image.paste(canvas)
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
    sub = p.add_subparsers(dest='command', required=True)
    p.set_defaults(preset='wide-v3', profile=DEFAULT_PROFILE, allocation=None, gain=1.0)

    def shared(q):
        q.add_argument('--profile', choices=list(wire_profiles()),
                       default=DEFAULT_PROFILE, help='Plane geometry to SEND. '
                       'The receiver reads it from the header.')
        q.add_argument('--modem-dir', type=Path)
        q.add_argument('--frames', type=int, default=24)
        q.add_argument('--stride', type=int, default=1)

    def allocation(q):
        q.add_argument('--allocation', type=Path,
                       help='Power allocation table from fit_allocation.py, '
                            'fitted to the pictures actually being sent. '
                            'Measured +3.4 to +3.8 dB on a held-out frame, for '
                            'no extra slots. SHARED STATE: not on the wire and '
                            'nothing detects it, so both ends need the same '
                            'file, and it only fits the preset/profile pair it '
                            'was made for.')
    b = sub.add_parser('bench'); shared(b); allocation(b)
    b.add_argument('--preset', choices=list(PRESETS), default='wide-v3')
    w = sub.add_parser('write'); shared(w)
    w.add_argument('--preset', choices=list(PRESETS), default='lean-v3')
    w.set_defaults(profile='color-lean')
    w.add_argument('--out', type=Path, default=Path('v3_test.wav'))
    w.add_argument('-f', '--numbered', action='store_true')
    allocation(w)
    r = sub.add_parser('read')
    r.add_argument('--preset', choices=list(PRESETS), default='lean-v3',
                   help='Fallback only. The preset is identified from the '
                        'signal by decoding against each candidate and letting '
                        'the header CRC pick; this is what gets tried first.')
    r.add_argument('--profile', choices=list(wire_profiles()),
                   default='color-lean',
                   help='Fallback only. The profile is read from the header, '
                        'so this matters just for a packet whose header never '
                        'verifies.')
    r.add_argument('--wav', type=Path, required=True)
    r.add_argument('--channels', type=pair, default=(0, 1))
    r.add_argument('--save-frames', type=Path)
    allocation(r)
    ls = sub.add_parser('live-send'); shared(ls)
    ls.add_argument('--preset', choices=list(PRESETS), default='lean-v3')
    ls.set_defaults(profile='color-lean')
    ls.add_argument('--device', type=device)
    ls.add_argument('--channels', type=pair, default=(0, 1))
    ls.add_argument('-f', '--numbered', action='store_true')
    allocation(ls)
    ls.add_argument('--list-devices', action='store_true')
    lr = sub.add_parser('live-receive')
    lr.add_argument('--device', type=device)
    lr.add_argument('--channels', type=pair, default=(0, 1))
    buffering = lr.add_mutually_exclusive_group()
    buffering.add_argument('--buffer-frames', type=int, choices=(1, 2), default=2)
    buffering.add_argument('--buffer-ms', type=float)
    lr.add_argument('--save-frames', type=Path)
    lr.add_argument('--on-loss', choices=('hold', 'black', 'damaged'), default='damaged')
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
                    help='How to scale the picture up to the window. raw '
                         '(default) is nearest-neighbour and shows the pixels '
                         'as sent, which is what you want when judging a '
                         'decode; smooth is Lanczos and looks better but hides '
                         'dead pixels, blocking and chroma blotching.')
    allocation(lr)
    lr.add_argument('--list-devices', action='store_true')
    args = p.parse_args(argv)
    {'bench': do_bench, 'write': do_write, 'read': do_read,
     'live-send': do_live_send, 'live-receive': do_live_receive}[args.command](args)


if __name__ == '__main__':
    main()