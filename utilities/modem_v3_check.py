#!/usr/bin/env python3
"""Fixed-format v3 image/audio link.

48 kHz stereo, wide layout, color 40x48 luma plus 20x24 chroma, 2880 values,
deterministic built-in allocation, 16-byte CRC header.

  python utilities/modem_v3_check.py write --modem-dir images_modem --out clean.wav
  python utilities/modem_v3_check.py read --wav clean.wav
  python utilities/modem_v3_check.py live-receive --device "BlackHole 2ch"
  python utilities/modem_v3_check.py live-send --device "BlackHole 2ch" --modem-dir images_modem

How acquisition works:

* The preamble is an LTC-style biphase-mark word, so edge intervals take two
  values and playback speed falls out of `measured / nominal` the way a timer
  capture gives it to an LTC reader -- roughly 26 us per buffer against the
  5.5 ms a scaled-template correlation bank costs. Correlation remains as a
  fallback for signal too weak to count edges on, decimated rather than run on
  every unlocked buffer.

* The preamble is on both channels, and `encode` normalises up as well as
  down so the payload uses the available headroom.

Presets are not interchangeable: orthogonal training and header placement are
part of the wire format, so transmitter and receiver must name the same one. A
mismatch reports "Nothing decoded" rather than producing a garbled picture.
"""
import argparse
from contextlib import closing
import json
from pathlib import Path
import sys
import time
import wave

import numpy as np
from PIL import Image, ImageOps

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from animation_modem import transport3 as V3                   # noqa: E402
from animation_modem import impairments as IMP                 # noqa: E402
from animation_modem.audio_common import (pcm, pair, device,   # noqa: E402
                                          wav_blocks, sounddevice)
from animation_modem.imaging import (burn_counters, fit_shapes,  # noqa: E402
                                     image_values, plane_shapes, values_image)

PRESETS = V3.ALL_PRESETS
RATE = V3.RATE


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
        shapes = fit_shapes(shapes, layout.capacity)
    table = np.load(allocation) if allocation else None
    return V3.SourceCoder(shapes, table), shapes


def receive_for(args, layout, coder):
    """The v3 receiver. Edge-gated, coasting, correlation only as fallback."""
    return V3.Receiver(layout, coder, recovery=False, fast=True)


def _prepared(image, absolute, count, numbered=False):
    return burn_counters(image, absolute, absolute, count) if numbered else image


def psnr(a, b):
    x = np.asarray(a.convert('RGB'), float)
    y = np.asarray(b.convert('RGB'), float)
    err = np.mean((x-y)**2)
    return float('inf') if err <= 0 else 10*np.log10(255.0**2/err)


def record(r):
    speed = 1/(1+r.rate_error)
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
    print(layout.describe())
    path = Path(args.out)
    with wave.open(str(path), 'wb') as sink:
        sink.setparams((2, 2, RATE, 0, 'NONE', 'not compressed'))
        for n, im in enumerate(frames):
            values = image_values(_prepared(im, n+1, len(frames), args.numbered),
                                  coder.shapes)
            audio = V3.encode(values, layout, coder, n+1, (n % len(frames))+1,
                              len(frames), stamp_ms=n*int(1000/layout.fps))
            sink.writeframesraw(pcm(audio*args.gain))
    seconds = len(frames)/layout.fps
    print(f'wrote {len(frames)} frames, {seconds:.1f} s, peak gain {args.gain} -> {path}')
    print('v3 preamble: biphase, both channels. v2 receivers will not acquire this.')


def do_read(args):
    layout = PRESETS[args.preset]
    coder, _ = coder_for(args.profile, args.allocation, layout)
    print(layout.describe(), file=sys.stderr)
    receiver = receive_for(args, layout, coder)
    if args.save_frames:
        Path(args.save_frames).mkdir(parents=True, exist_ok=True)
    tiers, rates, seen = {}, [], 0

    def results():
        for block in wav_blocks(args.wav, args.channels, 1024):
            yield from receiver.feed(block)
        yield from receiver.flush()

    for r in results():
        seen += 1
        tiers[r.tier] = tiers.get(r.tier, 0) + 1
        rates.append(r.rate_error)
        print(json.dumps(record(r)), flush=True)
        if args.save_frames and r.values is not None:
            name = f'{r.absolute:06d}' if r.absolute is not None else f'x{seen:06d}'
            values_image(r.values, coder.shapes).save(
                Path(args.save_frames)/f'frame_{name}.png')
    if rates:
        speed = 1/(1+float(np.median(rates)))
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
    layout = PRESETS[args.preset]
    coder, _ = coder_for(profile, args.allocation, layout)
    print(layout.describe())
    packets = [V3.encode(image_values(
                   _prepared(im, n+1, len(frames), args.numbered), coder.shapes),
               layout, coder, n+1, (n % len(frames))+1, len(frames),
               stamp_ms=n*int(1000/layout.fps))*args.gain
               for n, im in enumerate(frames)]
    print(f'{len(packets)} packets ready, {layout.fps:.2f} fps. Ctrl-C to stop.')
    state = {'packet': 0, 'position': 0, 'sent': 0}
    channels = args.channels

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

    with sd.OutputStream(samplerate=RATE, channels=max(channels)+1, dtype='float32',
                         device=args.device, blocksize=256, latency='low',
                         callback=callback):
        try:
            while True:
                time.sleep(1)
                print(f'  sent {state["sent"]} packets', end='\r', flush=True)
        except KeyboardInterrupt:
            print(f'\nstopped after {state["sent"]} packets')


def do_live_receive(args):
    """Decode from an input device. Complete pictures only.

    v3 emits one result per packet rather than progressive previews, so there
    is no partial-refinement scheduler here: a packet either reconstructs or it
    does not, and the newest complete picture wins.
    """
    import threading
    import signal
    sd = sounddevice()
    if args.list_devices:
        print(sd.query_devices()); return
    layout = PRESETS[args.preset]
    coder, _ = coder_for(args.profile, args.allocation, layout)
    verbose = args.verbose or args.headless
    if args.silent:
        verbose = False
    if args.save_frames:
        Path(args.save_frames).mkdir(parents=True, exist_ok=True)
    stop = threading.Event()
    latest = {'result': None}
    lock = threading.Lock()
    errors = []
    active = {'stream': None}

    def receive():
        try:
            receiver = V3.Receiver(layout, coder)
            channels = (0, 1) if args.channels is None else args.channels
            info = sd.query_devices(args.device, 'input')
            count = max(channels)+1
            if count > int(info['max_input_channels']):
                raise ValueError('Selected input channels unavailable')
            with sd.InputStream(samplerate=RATE, channels=count, dtype='float32',
                                device=args.device, blocksize=256) as stream:
                active['stream'] = stream
                if not args.silent:
                    print(f'Listening on {info["name"]}: {args.preset}, '
                          f'{args.profile}, channels {channels[0]+1},{channels[1]+1}',
                          file=sys.stderr, flush=True)
                seen = 0
                overflows = 0
                last_summary = time.monotonic()
                peak = 0.0
                while not stop.is_set():
                    audio, overflowed = stream.read(256)
                    audio = np.asarray(audio)[:, channels]
                    peak = max(peak, float(np.max(np.abs(audio))))
                    if overflowed:
                        overflows += 1
                        receiver.reset(preserve_timing=True)
                    for r in receiver.feed(audio):
                        seen += 1
                        if verbose:
                            print(json.dumps(record(r)), flush=True)
                        if r.values is not None:
                            with lock:
                                latest['result'] = r
                            if args.save_frames:
                                name = (f'{r.absolute:06d}' if r.absolute is not None
                                        else f'x{seen:06d}')
                                values_image(r.values, coder.shapes).save(
                                    Path(args.save_frames)/f'frame_{name}.png')
                    now = time.monotonic()
                    if not args.silent and now-last_summary >= args.summary_seconds:
                        print(json.dumps({'receiver_packets': seen,
                                          'input_overflows': overflows,
                                          'input_peak': round(peak, 6)}),
                              file=sys.stderr, flush=True)
                        last_summary, peak = now, 0.0
        except Exception as exc:
            if not stop.is_set():
                errors.append(exc)
        finally:
            active['stream'] = None
            # Always release the main loop. Catching only Exception meant a
            # thread that died any other way -- a closed device, SystemExit
            # from a shutting-down host -- left --headless spinning forever
            # with no output and no way out but Ctrl-C.
            stop.set()

    thread = threading.Thread(target=receive, daemon=True)
    previous_sigint = signal.signal(signal.SIGINT, lambda signum, frame: stop.set())
    try:
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
            root.title('v3 stereo image receiver')
            initial = ImageTk.PhotoImage(Image.new('RGB', size, 'black'))
            label = tk.Label(root, background='black', image=initial)
            label.image = initial
            label.pack()
            status = tk.Label(root, text='Waiting for signal', width=1, height=2,
                              anchor='w', justify='left', wraplength=size[0]-12)
            status.pack(fill='x', padx=6)

            def close():
                stop.set(); root.destroy()
            root.protocol('WM_DELETE_WINDOW', close)
            root.bind('<Escape>', lambda event: close())
            root.bind('<q>', lambda event: close())
            root.bind('<Q>', lambda event: close())

            def refresh():
                if stop.is_set():
                    root.destroy()
                    return
                with lock:
                    r = latest['result']
                    latest['result'] = None
                if r is not None and r.values is not None:
                    canvas = Image.new('RGB', size, 'black')
                    scaled = ImageOps.contain(values_image(r.values, coder.shapes),
                                              size, Image.Resampling.NEAREST)
                    canvas.paste(scaled, ((size[0]-scaled.width)//2,
                                          (size[1]-scaled.height)//2))
                    label.image.paste(canvas)
                    folders = ('?' if r.face_folder is None
                               else f'{r.face_folder}/{r.float_folder}')
                    status.config(
                        text=f'Frame {r.absolute} | src {r.source_index}/{r.count} '
                             f'| folders {folders}\n{r.status} | {r.identity} '
                             f'| tier {r.tier} | {r.extra.get("acquisition_path","?")}')
                if not stop.is_set():
                    root.after(10, refresh)
            refresh()
            try:
                root.mainloop()
            finally:
                stop.set()
    finally:
        stop.set()
        try:
            if thread.ident is not None:
                thread.join(timeout=1)
                stream = active['stream']
                if thread.is_alive() and stream is not None:
                    stream.abort()
                    thread.join(timeout=1)
        finally:
            signal.signal(signal.SIGINT, previous_sigint)
    if errors:
        raise SystemExit(str(errors[0]))


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest='command', required=True)
    # v3 defaults to its own preset: 'wide' works but leaves the training
    # improvement on the table, and the two are not interchangeable on the wire.
    p.set_defaults(preset='wide-v3', profile='color', allocation=None, gain=1.0)

    def shared(q):
        q.add_argument('--profile', choices=['color', 'color-lean', 'detail', 'mono'],
                       default='color',
                       help="Plane geometry. 'color-lean' quarters chroma: same "
                            "picture, 2160 coefficients instead of 2880.")
        q.add_argument('--modem-dir', type=Path, help='Bake to read frames from')
        q.add_argument('--frames', type=int, default=24)
        q.add_argument('--stride', type=int, default=1)
    b = sub.add_parser('bench', help='Offline v2/v3 comparison'); shared(b)
    b.add_argument('--allocation', type=Path)
    b.add_argument('--preset', choices=list(PRESETS), default='wide-v3')
    w = sub.add_parser('write'); shared(w)
    w.add_argument('--preset', choices=list(PRESETS), default='wide-v3')
    w.add_argument('--out', type=Path, default=Path('v3_test.wav'))
    w.add_argument('-f', '--numbered', action='store_true')
    r = sub.add_parser('read')
    r.add_argument('--preset', choices=list(PRESETS), default='wide-v3')
    r.add_argument('--profile', choices=['color', 'color-lean', 'detail', 'mono'],
                   default='color')
    r.add_argument('--wav', type=Path, required=True)
    r.add_argument('--channels', type=pair, default=(0, 1))
    r.add_argument('--save-frames', type=Path)
    ls = sub.add_parser('live-send'); shared(ls)
    ls.add_argument('--preset', choices=list(PRESETS), default='wide-v3')
    ls.add_argument('--device', type=device)
    ls.add_argument('--channels', type=pair, default=(0, 1))
    ls.add_argument('-f', '--numbered', action='store_true')
    ls.add_argument('--list-devices', action='store_true')
    lr = sub.add_parser('live-receive')
    lr.add_argument('--preset', choices=list(PRESETS), default='wide-v3')
    lr.add_argument('--profile', choices=['color', 'color-lean', 'detail', 'mono'],
                    default='color', help='Must match the sender profile')
    lr.add_argument('--receiver', choices=('v3',), default='v3',
                    help='Kept for argument parity with v2; v3 has one receiver')
    lr.add_argument('--device', type=device)
    lr.add_argument('--channels', type=pair, default=(0, 1),
                    help='Ordered 1-based input pair (default: 1,2)')
    lr.add_argument('--save-frames', type=Path)
    lr.add_argument('--headless', action='store_true',
                    help='JSON only, no window; implies --verbose')
    lr.add_argument('-v', '--verbose', action='store_true',
                    help='Per-packet JSON. Off by default: a live link is 14 a second.')
    lr.add_argument('--silent', action='store_true',
                    help='No output at all, not even the periodic summary')
    lr.add_argument('--summary-seconds', type=float, default=30.0,
                    help='Fallback digest interval when no source index is decoding')
    lr.add_argument('--width', type=int, default=480)
    lr.add_argument('--height', type=int, default=576)
    lr.add_argument('--list-devices', action='store_true')
    args = p.parse_args(argv)
    if getattr(args, 'frames', 1) < 1 or getattr(args, 'stride', 1) < 1:
        p.error('--frames and --stride must be positive')
    if not np.isfinite(getattr(args, 'gain', 1)) or getattr(args, 'gain', 1) <= 0:
        p.error('--gain must be finite and positive')
    {'bench': do_bench, 'write': do_write, 'read': do_read,
     'live-send': do_live_send, 'live-receive': do_live_receive}[args.command](args)


if __name__ == '__main__':
    main()
