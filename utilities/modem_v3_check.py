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

How acquisition works:

* The preamble is an LTC-style biphase-mark word, so edge intervals take two
  values and playback speed falls out of `measured / nominal` the way a timer
  capture gives it to an LTC reader -- roughly 26 us per buffer against the
  5.5 ms a scaled-template correlation bank costs. The default path measures
  transition times directly, without a speed sweep or waveform correlation.

* The preamble is on both channels, and `encode` normalises up as well as
  down so the payload uses the available headroom.

Neither preset nor profile has to be named at both ends any more.

The profile rides in two spare header bits, so a receiver holding every coder
reconstructs the geometry that was sent. The preset cannot be signalled that
way -- you need the layout to know where the header is, which is circular --
so it is identified instead: the preamble is layout-independent, so acquisition
works regardless, and from there the receiver decodes against each candidate
until the header CRC verifies, then pins that layout. One attempt costs 1.4 ms
and is paid once per lock, not per frame.

`--preset` and `--profile` on the receiving side are therefore only fallbacks
for a packet whose header never verified.
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
from animation_modem.imaging import (burn_counters, fit_shapes,  # noqa: E402
                                     image_values, plane_shapes, values_image)

PRESETS = V3.ALL_PRESETS
# Used to stamp files this tool authors, and as the geometry the reported
# playback speed is measured against. Live audio never sees it: both live paths
# take the rate from the device once it is open.
REFERENCE_RATE = V3.REFERENCE_RATE
RATE = REFERENCE_RATE          # existing name, same meaning


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
        from PIL import ImageFilter
        tex = np.asarray(Image.fromarray(np.uint8(np.clip(
            150+55*rng.standard_normal((48, 40)), 0, 255))).filter(
                ImageFilter.GaussianBlur(1.1)), float)
        im = np.full((48, 40, 3), 4.)
        for c, k in enumerate((1., .82, .70)):
            im[:, :, c] = np.where(mask, tex*k, 4)
        out.append(Image.fromarray(np.uint8(np.clip(im, 0, 255))))
    return out, getattr(args, 'profile', None) or profile


def coder_for(profile, allocation, layout):
    shapes = plane_shapes(profile)
    if sum(int(np.prod(s)) for s in shapes) > layout.capacity:
        # Both ends run this same deterministic shrink, so a profile too big
        # for the layout still agrees end to end -- the wire code names the
        # profile, and the layout decides what fits.
        shapes = fit_shapes(shapes, layout.capacity)
    table = np.load(allocation) if allocation else None
    return V3.SourceCoder(shapes, table), shapes


def coders_for(layout, allocation=None):
    """Every profile the wire can name, so a receiver can follow the header.

    Built once at startup. The transmitter declares its geometry in two spare
    header bits; holding all four coders is what lets that declaration mean
    something instead of needing a matching argument on both ends.
    """
    return {V3.profile_code(name): coder_for(name, allocation, layout)[0]
            for name in V3.PROFILE_CODES}


def live_presets():
    """Presets a v3 receiver can identify by trying them.

    Only progressive layouts: the others are v2 wire format with a different
    magic, so they are not candidates at all.
    """
    return [n for n, l in PRESETS.items() if l.progressive]


def candidates_for(allocation=None):
    """(layout, default coder, coders) for every preset worth trying.

    Preset cannot be signalled in the header the way profile is -- you need
    the layout to know where the header is -- so it is identified by decoding
    against each candidate and letting the CRC decide. 28 coders cost 27 ms at
    startup and one attempt costs 1.4 ms, paid once per lock rather than per
    frame.
    """
    out = []
    for name in live_presets():
        layout = PRESETS[name]
        coders = coders_for(layout, allocation)
        out.append((layout, coders[V3.profile_code('color-lean')], coders))
    return out


def receive_for(args, layout, coder, input_rate=None, coders=None,
                candidates=None):
    """The v3 receiver. Playback speed comes from biphase pulse timing.

    `input_rate` is whatever the source turned out to be running at -- an open
    device's reported rate, or a WAV header's. It is not requested anywhere and
    decoding does not depend on it; it only lets the results come back in
    seconds and hertz instead of samples and cycles per sample.
    """
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


def record(r):
    # The decoder reports speed against the clock the audio arrived on. Falling
    # back to 1/(1+rate_error) reproduces that for results from a source whose
    # rate was never learned -- the same number, just without the units.
    speed = r.extra.get('playback_speed', 1/(1+r.rate_error))
    return {'status': r.status, 'identity': r.identity, 'frame': r.absolute,
            'index': r.index, 'source_index': r.source_index, 'count': r.count,
            'face_folder': r.face_folder, 'float_folder': r.float_folder,
            'tier': r.tier, 'coverage': r.coverage, 'pilot_error': r.pilot_error,
            'playback_speed': speed, 'playback_rate_pct': round(100*(speed-1), 3),
            **r.extra}


def do_write(args):
    frames, profile = frames_from(args)
    layout = PRESETS[args.preset]
    coder, _ = coder_for(profile, args.allocation, layout)
    # A file we author has no device to ask, so it is stamped at the reference
    # rate. `read` takes the rate from the header, so a file resampled to any
    # other rate afterwards still decodes.
    fps = layout.fps_at(REFERENCE_RATE)
    print(layout.describe(REFERENCE_RATE))
    path = Path(args.out)
    with wave.open(str(path), 'wb') as sink:
        sink.setparams((2, 2, REFERENCE_RATE, 0, 'NONE', 'not compressed'))
        for n, im in enumerate(frames):
            values = image_values(_prepared(im, n+1, len(frames), args.numbered),
                                  coder.shapes)
            audio = V3.encode(values, layout, coder, n+1, (n % len(frames))+1,
                              len(frames), stamp_ms=n*int(1000/fps),
                              profile=V3.profile_code(profile))
            sink.writeframesraw(pcm(audio*args.gain))
    seconds = len(frames)/fps
    print(f'wrote {len(frames)} frames, {seconds:.1f} s, peak gain {args.gain} -> {path}')
    print('v3 preamble: biphase, both channels. v2 receivers will not acquire this.')


def do_read(args):
    layout = PRESETS[args.preset]
    coder, _ = coder_for(args.profile, args.allocation, layout)
    # Read the file's rate; do not require one. A 44.1 kHz capture of a 48 kHz
    # transmission is a real recording, not a malformed one.
    rate = wav_rate(args.wav)
    receiver = receive_for(args, layout, coder, input_rate=rate,
                           coders=coders_for(layout, args.allocation),
                           candidates=candidates_for(args.allocation))
    if rate != REFERENCE_RATE:
        print(f'{args.wav}: {rate} Hz, decoding at that rate', file=sys.stderr)
    if args.save_frames:
        Path(args.save_frames).mkdir(parents=True, exist_ok=True)
    tiers, rates, seen = {}, [], 0

    def results():
        level = InputLevel()
        for block in wav_blocks(args.wav, args.channels, 1024):
            yield from receiver.feed(level.process(block))
        yield from receiver.flush()

    for r in results():
        seen += 1
        tiers[r.tier] = tiers.get(r.tier, 0) + 1
        rates.append(record(r)['playback_speed'])
        print(json.dumps(record(r)), flush=True)
        if args.save_frames and r.values is not None:
            name = f'{r.absolute:06d}' if r.absolute is not None else f'x{seen:06d}'
            values_image(r.values, r.extra.get('shapes', coder.shapes)).save(
                Path(args.save_frames)/f'frame_{name}.png')
    if rates:
        speed = float(np.median(rates))
        print(f'\n{seen} packets. tiers {tiers}. '
              f'median playback speed {speed:.4f}x '
              f'({100*(speed-1):+.2f}% off nominal)', file=sys.stderr)
        print(f'acquisition: edge {receiver.edge_hits}, coast {receiver.locked_packets}, '
              f'correlation {receiver.correlation_hits}', file=sys.stderr)
    else:
        print('\nNothing decoded. Check the channel pair and the level.', file=sys.stderr)


def do_bench(args):
    """Compare v2 and v3 across simulated channels, at equal settings."""
    frames, profile = frames_from(args)
    print('\n'.join('  '+l.describe() for l in PRESETS.values()) + '\n')
    layout = PRESETS[args.preset]
    coder, _ = coder_for(profile, args.allocation, layout)
    channels = [('clean', {}),
                ('cassette-ish', dict(lowpass_hz=10000, noise_dbfs=-45)),
                ('worn deck', dict(lowpass_hz=8000, noise_dbfs=-40, crosstalk=.07))]
    # v2 is gone; compare the v3 layouts against each other instead.
    versions = [(name, V3.encode,
                 (lambda l=PRESETS[name]: V3.Receiver(l, coder)), PRESETS[name])
                for name in ('wide-v3', 'wide-v3-fast')]
    print(f'  {"channel":<16}{"preset":>13}{"PSNR":>8}{"headers":>9}'
          f'{"tier":>10}{"coverage":>10}{"cpu %":>8}')
    for name, settings in channels:
        for vname, enc, make, layout in versions:
            quality, hdr, tier, coverage = [], 0, {}, []
            elapsed = samples = 0
            for n, im in enumerate(frames[:args.frames]):
                ready = _prepared(im, n+1, len(frames), False)
                audio = enc(image_values(ready, coder.shapes), layout, coder,
                            n+1, 1, len(frames))
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
                    quality.append(psnr(values_image(good[0].values, coder.shapes)
                                        .resize(ready.size), ready))
                    tier[good[0].tier] = tier.get(good[0].tier, 0)+1
                    coverage.append(good[0].coverage or 0)
                    hdr += good[0].identity == 'verified_header'
            print(f'  {name if vname=="wide-v3" else "":<16}{vname:>13}'
                  f'{np.mean(quality) if quality else float("nan"):>8.2f}'
                  f'{hdr:>9}{max(tier, key=tier.get) if tier else "-":>10}'
                  f'{np.mean(coverage) if coverage else float("nan"):>10.3f}'
                  f'{100*elapsed/(samples/RATE):>8.1f}')
    print('\n  cpu % is one core at this host clock, decode only.')


def do_live_send(args):
    """Play v3 packets continuously to an output device."""
    sd = sounddevice()
    if args.list_devices:
        print(sd.query_devices()); return
    frames, profile = frames_from(args)
    # Preset stays fixed -- it is wire format the receiver cannot negotiate.
    # Profile is declared in the header, so any of them is live now.
    if not PRESETS[args.preset].progressive:
        raise SystemExit(f'{args.preset} is v2 wire format; live needs a '
                         f'progressive preset')
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
            outdata[written:written+take, channels] = \
                block[state['position']:state['position']+take]
            written += take
            state['position'] += take
            if state['position'] >= len(block):
                state['position'] = 0
                state['packet'] = (state['packet']+1) % len(packets)
                state['sent'] += 1

    # Built but not started, so the device can report its rate before anything
    # is encoded. No samplerate= is passed: the device keeps whatever its owner
    # set it to, and the frame rate and header timestamps follow from that.
    stream = sd.OutputStream(channels=max(channels)+1, dtype='float32',
                             device=args.device, blocksize=256, latency='low',
                             callback=callback)
    try:
        rate = float(stream.samplerate)
        # Band-limited once here, not per callback: these packets repeat.
        emitted = V3.emit_length(layout.frame, rate)
        fps = rate/emitted
        print(layout.describe(rate))
        packets.extend(V3.band_limited(V3.encode(image_values(
                           _prepared(im, n+1, len(frames), args.numbered), coder.shapes),
                       layout, coder, n+1, (n % len(frames))+1, len(frames),
                       stamp_ms=n*int(1000/fps),
                       profile=V3.profile_code(profile))*args.gain, rate)
                       for n, im in enumerate(frames))
        print(f'{len(packets)} packets ready, {fps:.2f} fps at {rate:g} Hz, '
              f'{emitted} samples each. Ctrl-C to stop.')
        notice = wire_notice(layout, rate)
        if notice:
            print(notice, file=sys.stderr)
        stream.start()
        try:
            while True:
                time.sleep(1)
                print(f'  sent {state["sent"]} packets', end='\r', flush=True)
        except KeyboardInterrupt:
            print(f'\nstopped after {state["sent"]} packets')
    finally:
        stream.close()


def do_live_receive(args):
    """Decode from an input device. Complete pictures only.

    v3 emits one result per packet rather than progressive previews, so there
    is no partial-refinement scheduler here: a packet either reconstructs or it
    does not, and the newest complete picture wins.

    No sample rate is requested. The stream is opened at whatever the device is
    already set to, and that rate -- read back from the open stream -- is the
    only one used anywhere. Pulse timing makes the decode itself indifferent to
    it, so the rate serves buffer sizing, the displayed frame duration and the
    reported numbers, and nothing else.
    """
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
    # Filled in by the capture thread once the device says what it is doing.
    opened = threading.Event()
    source = {'rate': None, 'level': InputLevel()}

    from animation_modem.audio_buffer import AudioBuffer
    # Sized in samples until the rate is known, which is the honest unit: a
    # frame is 2768 samples wherever it is played. --buffer-ms cannot be
    # converted yet, so it waits for the first reconfigure below.
    widest = max(PRESETS[n].frame for n in live_presets())
    minimum_buffer = ((widest+255)//256)*256
    audio_buffer = AudioBuffer(max(minimum_buffer, args.buffer_frames*layout.frame),
                               layout.frame)
    reports = queue.Queue(maxsize=1)

    def offer_report(result, summary):
        # Never make decoding wait for terminal output or filesystem writes.
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
                        values_image(result.values, result.extra['shapes']).save(
                            Path(args.save_frames)/f'frame_{name}.png')
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
            # No samplerate=. Asking for 48 kHz on a device set to 44.1 either
            # failed to open or silently inserted PortAudio's resampler; the
            # decoder wants neither, because it can read the cadence itself.
            with sd.InputStream(channels=count, dtype='float32',
                                device=args.device, blocksize=256) as stream:
                active['stream'] = stream
                rate = float(stream.samplerate)
                source['rate'] = rate
                opened.set()
                device_text = (f'Input: {info["name"]} | {rate:g} Hz | channels '
                               f'{channels[0]+1},{channels[1]+1}')
                with lock:
                    latest['device'] = device_text
                print(f'{device_text} | preset and profile read from the '
                      f'signal; Ctrl-C to stop', file=sys.stderr, flush=True)
                # Levelled here, on the way in, so the decoder's absolute
                # acquisition threshold sees a usable amplitude whatever the
                # deck or interface hands over.
                level = source['level']
                while not stop.is_set():
                    audio, overflowed = stream.read(256)
                    audio_buffer.put(level.process(np.asarray(audio)[:, channels]),
                                     overflowed)
        except Exception as exc:
            if not stop.is_set():
                errors.append(exc)
        finally:
            active['stream'] = None
            opened.set()          # Release the decoder; it checks for a rate.
            audio_buffer.close()

    def receive():
        try:
            # Wait for the device to say what it is running at rather than
            # deciding for it. A failed open sets the event with no rate, and
            # this thread retires instead of decoding against a guess.
            while not opened.wait(.1):
                if stop.is_set():
                    return
            rate = source['rate']
            if rate is None:
                return
            receiver = V3.Receiver(layout, coder, pulse_only=True, input_rate=rate,
                                   coders=coders_for(layout),
                                   candidates=candidates_for())
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
                    # frame_seconds now comes from the decoder, which knows both
                    # the measured scale and the clock it was measured on.
                    # shapes comes from the decoder now: it reports the
                    # geometry it actually reconstructed, which is the sender's
                    # when the header verified and ours when it did not.
                    result.extra.update(
                        complete=result.identity == 'verified_header' and result.status == 'received')
                # receiver.layout is whichever preset identified itself.
                period_samples = max(256, round(receiver.layout.frame *
                                                (1+receiver.rate_error)))
                capacity = (int(rate*args.buffer_ms/1000) if args.buffer_ms is not None
                            else args.buffer_frames*period_samples)
                audio_buffer.configure(max(256, capacity), min(period_samples, 1024))
                newest = results[-1] if results else None
                if newest is not None:
                    with lock:
                        for result in results:
                            latest['picture'].push(result, time.monotonic())
                now = time.monotonic()
                summary = None
                if not args.silent and now-last_summary >= args.summary_seconds:
                    summary = {'receiver_packets': seen,
                               'input_gain': [round(g, 4) for g in source['level'].gain],
                               'input_limited_blocks': source['level'].limited,
                               'input_rate_hz': rate,
                               'nominal_fps': round(layout.fps_at(rate), 3),
                               'input_overflows': audio_buffer.input_overflows,
                               'buffer_dropped_samples': audio_buffer.dropped_samples,
                               'buffer_capacity_ms': round(1000*audio_buffer.capacity/rate, 1),
                               'input_peak': round(peak, 6)}
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
            root = tk.Tk()
            def callback_error(exc_type, exc, traceback):
                # A Tk callback exception otherwise leaves the first picture on
                # screen forever because refresh never schedules its next call.
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
            root.bind('<Q>', lambda event: close())

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
                        scaled = ImageOps.contain(values_image(r.values, r.extra['shapes']),
                                                  size, Image.Resampling.NEAREST)
                        canvas.paste(scaled, ((size[0]-scaled.width)//2,
                                              (size[1]-scaled.height)//2))
                        # Profile is whatever this packet declared, so show
                        # the one on screen rather than one assumed at startup.
                        shown = r.extra.get('profile') or 'profile unverified'
                        found = r.extra.get('preset', '?')
                        status.config(text=f'{found} / {shown} | '
                                           f'frame {r.absolute} | {r.status}')
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
            stream = active['stream']
            if capture_thread.is_alive() and stream is not None:
                stream.abort()
                capture_thread.join(timeout=1)
        finally:
            signal.signal(signal.SIGINT, previous_sigint)
    if errors:
        raise SystemExit(str(errors[0]))


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest='command', required=True)
    # v3 defaults to its own preset: 'wide' works but leaves the training
    # improvement on the table. Both decode without being named: the receiver
    # identifies the layout from the signal.
    p.set_defaults(preset='wide-v3', profile='color', allocation=None, gain=1.0)

    def shared(q):
        q.add_argument('--profile', choices=['color', 'color-lean', 'detail', 'mono'],
                       default='color',
                       help="Plane geometry to SEND. 'color-lean' quarters "
                            "chroma: same picture, 2160 coefficients instead "
                            "of 2880. Receivers read it from the header.")
        q.add_argument('--modem-dir', type=Path, help='Bake to read frames from')
        q.add_argument('--frames', type=int, default=24)
        q.add_argument('--stride', type=int, default=1)
    b = sub.add_parser('bench', help='Offline v2/v3 comparison'); shared(b)
    b.add_argument('--allocation', type=Path)
    b.add_argument('--preset', choices=list(PRESETS), default='wide-v3')
    w = sub.add_parser('write'); shared(w)
    w.add_argument('--preset', choices=list(PRESETS), default='lean-v3')
    w.set_defaults(profile='color-lean')
    w.add_argument('--out', type=Path, default=Path('v3_test.wav'))
    w.add_argument('-f', '--numbered', action='store_true')
    r = sub.add_parser('read')
    r.add_argument('--preset', choices=list(PRESETS), default='lean-v3',
                   help='Fallback only. The preset is identified from the '
                        'signal; this is what gets used if no header verifies.')
    r.add_argument('--profile', choices=['color', 'color-lean', 'detail', 'mono'],
                   default='color-lean',
                   help='Fallback only. The profile is read from the header.')
    r.add_argument('--wav', type=Path, required=True)
    r.add_argument('--channels', type=pair, default=(0, 1))
    r.add_argument('--save-frames', type=Path)
    ls = sub.add_parser('live-send'); shared(ls)
    ls.add_argument('--preset', choices=list(PRESETS), default='lean-v3')
    ls.set_defaults(profile='color-lean')   # the receiver follows the header
    ls.add_argument('--device', type=device)
    ls.add_argument('--channels', type=pair, default=(0, 1))
    ls.add_argument('-f', '--numbered', action='store_true')
    ls.add_argument('--list-devices', action='store_true')
    lr = sub.add_parser('live-receive')
    lr.add_argument('--device', type=device)
    lr.add_argument('--channels', type=pair, default=(0, 1),
                    help='Ordered 1-based input pair (default: 1,2)')
    buffering = lr.add_mutually_exclusive_group()
    buffering.add_argument('--buffer-frames', type=int, choices=(1, 2), default=2,
                           help='Keep only the newest one or two detected frame intervals (default: 2)')
    buffering.add_argument('--buffer-ms', type=float,
                           help='Override the audio capacity in milliseconds')
    lr.add_argument('--save-frames', type=Path,
                    help='Save newest pictures; slow disk writes may skip frames')
    lr.add_argument('--on-loss', choices=('hold', 'black', 'damaged'), default='damaged',
                    help='Show damaged images (default), hold the last good '
                         'one, or show black. A picture whose header did not '
                         'verify still decodes; holding hides it.')
    lr.add_argument('--headless', action='store_true',
                    help='JSON only, no window; implies --verbose')
    lr.add_argument('-v', '--verbose', action='store_true',
                    help='JSON for newest decoded pictures; slow output may skip records')
    lr.add_argument('--silent', action='store_true',
                    help='Suppress packet output and summaries; input device is still shown')
    lr.add_argument('--summary-seconds', type=float, default=30.0,
                    help='Fallback digest interval when no source index is decoding')
    lr.add_argument('--width', type=int, default=480)
    lr.add_argument('--height', type=int, default=576)
    lr.add_argument('--list-devices', action='store_true')
    args = p.parse_args(argv)
    if getattr(args, 'buffer_ms', None) is not None:
        if not np.isfinite(args.buffer_ms) or args.buffer_ms <= 0:
            p.error('--buffer-ms must be finite and positive')
    if getattr(args, 'frames', 1) < 1 or getattr(args, 'stride', 1) < 1:
        p.error('--frames and --stride must be positive')
    if not np.isfinite(getattr(args, 'gain', 1)) or getattr(args, 'gain', 1) <= 0:
        p.error('--gain must be finite and positive')
    {'bench': do_bench, 'write': do_write, 'read': do_read,
     'live-send': do_live_send, 'live-receive': do_live_receive}[args.command](args)


if __name__ == '__main__':
    main()
