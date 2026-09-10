#!/usr/bin/env python3
"""Fixed-format v2 image/audio link.

Normal transmit: 48 kHz stereo, wide layout, color 40x48 luma plus
20x24 chroma, 2880 values, deterministic built-in allocation. The receiver
uses the same constants and recovers timing/rate from incoming audio.

  python utilities/modem_v2_check.py write --modem-dir images_modem --out clean.wav
  python utilities/modem_v2_check.py read --wav clean.wav
  python utilities/modem_v2_check.py live-receive --device "BlackHole 2ch"
  python utilities/modem_v2_check.py live-send --device "BlackHole 2ch" --modem-dir images_modem

Apply effects to copies of rendered WAVs. The bench subcommand retains
experimental layouts for offline comparisons only. Live sending loops a
selected set of baked composites; this tool does not drive main.py's scheduler.
"""
import argparse
import json
from pathlib import Path
import sys
import wave

import numpy as np
from PIL import Image, ImageOps

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from animation_modem import transport2 as V2                   # noqa: E402
from animation_modem import impairments as IMP                 # noqa: E402
from animation_modem.audio_common import (pcm, pair, device,   # noqa: E402
                                          wav_blocks)
from animation_modem.imaging import (burn_counters, fit_shapes,  # noqa: E402
                                     image_values, plane_shapes, values_image)


def frames_from(args, profile='color'):
    """Real composites from a bake, or a synthetic stand-in."""
    if args.modem_dir:
        from modem_bake import ModemLibrary
        library = ModemLibrary(args.modem_dir)
        picks = range(0, min(library.frames, args.frames*args.stride), args.stride)
        # Bake profile affects source detail, never the fixed wire format.
        return [library.composite(i, 0, 0) for i in picks], 'color'
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
    return out, profile


def _prepared(image, absolute, count, numbered=False):
    """What actually goes on the wire, counters included when asked."""
    return burn_counters(image, absolute, absolute, count) if numbered else image


def psnr(a, b):
    error = ((np.asarray(a, float) - np.asarray(b, float))**2).mean()
    return float('inf') if error == 0 else 10*np.log10(255**2/error)


def coder_for(profile, allocation, layout=None):
    shapes = plane_shapes(profile)
    if layout:
        shapes = fit_shapes(shapes, layout.capacity)
    table = np.load(allocation) if allocation else None
    return V2.SourceCoder(shapes, table), shapes


def receive_for(args,layout,coder):
    return V2.Receiver(layout,coder)


def record(r):
    speed=1/(1+r.rate_error)
    return {'status':r.status,'identity':r.identity,'frame':r.absolute,
            'index':r.index,'count':r.count,'tier':r.tier,'coverage':r.coverage,
            'pilot_error':r.pilot_error,'playback_speed':speed,
            'playback_rate_pct':round(100*(speed-1),3),**r.extra}


def do_write(args):
    frames, profile = frames_from(args)
    layout = V2.PRESETS[args.preset]
    coder, _ = coder_for(profile, args.allocation,layout)
    print(layout.describe())
    path = Path(args.out)
    with wave.open(str(path), 'wb') as sink:
        sink.setparams((2, 2, V2.RATE, 0, 'NONE', 'not compressed'))
        for n, im in enumerate(frames):
            values = image_values(_prepared(im, n+1, len(frames), args.numbered), coder.shapes)
            audio = V2.encode(values, layout, coder, n+1, (n % len(frames))+1,
                              len(frames), stamp_ms=n*int(1000/layout.fps))
            sink.writeframesraw(pcm(audio*args.gain))
    seconds = len(frames)/layout.fps
    print(f'wrote {len(frames)} frames, {seconds:.1f} s, peak gain {args.gain} -> {path}')
    print('Decode this clean WAV first; apply effects to a separate copy afterward.')


def do_read(args):
    layout = V2.PRESETS[args.preset]
    coder, _ = coder_for(args.profile, args.allocation,layout)
    print(layout.describe(), file=sys.stderr)
    receiver = receive_for(args,layout,coder)
    if args.save_frames:
        Path(args.save_frames).mkdir(parents=True, exist_ok=True)
    tiers, rates, seen = {}, [], 0
    def results():
        for block in wav_blocks(args.wav,args.channels,1024):
            yield from receiver.feed(block)
        yield from receiver.flush()
    for r in results():
        seen += 1
        tiers[r.tier] = tiers.get(r.tier, 0) + 1
        rates.append(r.rate_error)
        print(json.dumps(record(r)),flush=True)
        if args.save_frames and r.values is not None:
            name = f'{r.absolute:06d}' if r.absolute is not None else f'x{seen:06d}'
            values_image(r.values, coder.shapes).save(Path(args.save_frames)/f'frame_{name}.png')
    if rates:
        speed = 1/(1+float(np.median(rates)))
        print(f'\n{seen} packets. tiers {tiers}. '
              f'median playback speed {speed:.4f}x '
              f'({100*(speed-1):+.2f}% off nominal)', file=sys.stderr)
    else:
        print('\nNothing decoded. Check the channel pair and the level.', file=sys.stderr)


def do_bench(args):
    """Compare a preset across simulated channels. v2 only -- there is no v1
    encode or decode left in the runtime to compare against."""
    frames, profile = frames_from(args)
    print('\n'.join('  '+l.describe() for l in V2.PRESETS.values()) + '\n')
    layout = V2.PRESETS[args.preset]
    coder, _ = coder_for(profile, args.allocation, layout)
    channels = [('clean', {}),
                ('cassette-ish', dict(lowpass_hz=10000, noise_dbfs=-45)),
                ('worn deck', dict(lowpass_hz=8000, noise_dbfs=-40, crosstalk=.07))]
    print(f'  {"channel":<16}{"PSNR":>8}{"headers":>9}{"tier":>10}{"coverage":>10}')
    for name, settings in channels:
        quality, hdr, tier, coverage = [], 0, {}, []
        for n, im in enumerate(frames[:args.frames]):
            ready = _prepared(im, n+1, len(frames), False)
            audio = V2.encode(image_values(ready, coder.shapes), layout, coder,
                              n+1, 1, len(frames))
            emulator = IMP.Emulator(IMP.Settings(**settings))
            receiver = V2.Receiver(layout, coder)
            got = []
            signal = np.concatenate([np.zeros((300, 2), np.float32), audio])
            for i in range(0, len(signal), 256):
                got += receiver.feed(emulator.process(signal[i:i+256]))
            got += receiver.flush()
            good = [g for g in got if g.values is not None]
            if good:
                quality.append(psnr(values_image(good[0].values, coder.shapes)
                                    .resize(ready.size), ready))
                tier[good[0].tier] = tier.get(good[0].tier, 0)+1
                coverage.append(good[0].coverage or 0)
                hdr += good[0].identity == 'verified_header'
        print(f'  {name:<16}{np.mean(quality) if quality else float("nan"):>8.2f}'
              f'{hdr:>9}{max(tier, key=tier.get) if tier else "-":>10}'
              f'{np.mean(coverage) if coverage else float("nan"):>10.3f}')
    print('\n  Small presets trade resolution for band. An allocation table must be built\n'
          '  for the same preset and used at both ends.')


def do_live_send(args):
    """Play v2 packets continuously to an output device.

    No shared-clock scheduling here on purpose: this is a link test, so it just
    keeps the carrier fed. Frame identity comes from the header, not a deadline.
    """
    sd = _sounddevice()
    if args.list_devices:
        print(sd.query_devices()); return
    frames, profile = frames_from(args)
    layout = V2.PRESETS[args.preset]
    coder, _ = coder_for(profile, args.allocation,layout)
    print(layout.describe())
    packets = [V2.encode(image_values(
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

    with sd.OutputStream(samplerate=V2.RATE, channels=max(channels)+1, dtype='float32',
                         device=args.device, blocksize=256, latency='low',
                         callback=callback):
        try:
            import time
            while True:
                time.sleep(1)
                print(f'  sent {state["sent"]} packets', end='\r', flush=True)
        except KeyboardInterrupt:
            print(f'\nstopped after {state["sent"]} packets')


def do_live_receive(args):
    """Decode from an input device and show the newest complete frame.

    No timestamp scheduling and no presentation queue: whatever decoded most
    recently is what is on screen. That is the right model for an analog
    source, where the header carries identity and a recorded timestamp means
    nothing against the current wall clock.
    """
    import threading
    sd = _sounddevice()
    if args.list_devices:
        print(sd.query_devices()); return
    layout = V2.PRESETS[args.preset]
    coder, _ = coder_for(args.profile, args.allocation,layout)
    print(layout.describe(), file=sys.stderr)
    channels = args.channels
    receiver = receive_for(args,layout,coder)
    if args.save_frames:
        Path(args.save_frames).mkdir(parents=True, exist_ok=True)
    newest = {'result': None, 'seen': 0, 'tiers': {}}
    stop = threading.Event()

    def pump():
        with sd.InputStream(samplerate=V2.RATE, channels=max(channels)+1, dtype='float32',
                            device=args.device, blocksize=256, latency='low') as stream:
            print(json.dumps({'input_latency_ms': stream.latency*1000}), file=sys.stderr)
            while not stop.is_set():
                audio, overflow = stream.read(256)
                if overflow:
                    print(json.dumps({'status': 'input_overflow'}), flush=True)
                    receiver.reset()
                block = audio[:, channels]
                for r in receiver.feed(block):
                    newest['seen'] += 1
                    newest['tiers'][r.tier] = newest['tiers'].get(r.tier, 0)+1
                    if r.values is not None:
                        newest['result'] = r          # latest wins, nothing queued
                    if not args.quiet:
                        print(json.dumps(record(r)),flush=True)
                    if args.save_frames and r.values is not None:
                        name = (f'{r.absolute:06d}' if r.absolute is not None
                                else f'x{newest["seen"]:06d}')
                        values_image(r.values, coder.shapes).save(
                            Path(args.save_frames)/f'frame_{name}.png')

    if args.headless:
        try:
            pump()
        except KeyboardInterrupt:
            pass
        print(f'\n{newest["seen"]} packets, tiers {newest["tiers"]}', file=sys.stderr)
        return

    import tkinter as tk
    from PIL import ImageTk, ImageOps
    size = (args.width, args.height)
    root = tk.Tk()
    root.title(f'modem v2 - {args.preset} - waiting for signal')
    blank = ImageTk.PhotoImage(Image.new('RGB', size, 'black'))
    label = tk.Label(root, background='black', image=blank)
    label.image = blank
    label.pack()
    status = tk.Label(root, text='waiting for signal')
    status.pack()
    thread = threading.Thread(target=pump, daemon=True)

    def close():
        stop.set()
        root.destroy()
    root.protocol('WM_DELETE_WINDOW', close)
    thread.start()
    shown = {'at': None}

    def refresh():
        r = newest['result']
        if r is not None and r is not shown['at']:
            shown['at'] = r
            im = values_image(r.values, coder.shapes)
            canvas = Image.new('RGB', size, 'black')
            scaled = ImageOps.contain(im, size, Image.Resampling.NEAREST)
            canvas.paste(scaled, ((size[0]-scaled.width)//2, (size[1]-scaled.height)//2))
            photo = ImageTk.PhotoImage(canvas)
            label.configure(image=photo)
            label.image = photo
            frame = r.absolute if r.absolute is not None else '?'
            status.config(text=f'frame {frame} | {r.status} | tier {r.tier} | '
                               f'coverage {r.coverage:.2f} | '
                               f'speed {1/(1+r.rate_error):.3f}x')
            root.title(f'modem v2 - {args.preset} - frame {frame} - {r.tier}')
        if not stop.is_set():
            root.after(10, refresh)

    refresh()
    try:
        root.mainloop()
    finally:
        stop.set()
    print(f'{newest["seen"]} packets, tiers {newest["tiers"]}', file=sys.stderr)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest='command', required=True)
    # Normal sending and receiving share one format, without external tables.
    p.set_defaults(preset='wide', profile='color', allocation=None, gain=1.0)
    def shared(q):
        q.add_argument('--modem-dir', type=Path, help='Bake to read frames from')
        q.add_argument('--frames', type=int, default=24)
        q.add_argument('--stride', type=int, default=1)
    b = sub.add_parser('bench', help='Offline experimental comparison'); shared(b)
    b.add_argument('--allocation', type=Path)
    b.add_argument('--preset', choices=list(V2.PRESETS), default='wide')
    w = sub.add_parser('write'); shared(w)
    w.add_argument('--out', type=Path, default=Path('v2_test.wav'))
    w.add_argument('-f', '--numbered', action='store_true')
    r = sub.add_parser('read')
    r.add_argument('--wav', type=Path, required=True)
    r.add_argument('--channels', type=pair,
                   default=(0, 1))
    r.add_argument('--save-frames', type=Path)
    ls = sub.add_parser('live-send'); shared(ls)
    ls.add_argument('--device',type=device); ls.add_argument('--channels',type=pair,default=(0,1))
    ls.add_argument('-f', '--numbered', action='store_true')
    ls.add_argument('--list-devices', action='store_true')
    lr = sub.add_parser('live-receive')
    lr.add_argument('--device',type=device); lr.add_argument('--channels',type=pair,default=(0,1))
    lr.add_argument('--save-frames', type=Path)
    lr.add_argument('--headless', action='store_true', help='JSON only, no window')
    lr.add_argument('--quiet', action='store_true', help='Window only, no per-packet JSON')
    lr.add_argument('--width', type=int, default=480)
    lr.add_argument('--height', type=int, default=576)
    lr.add_argument('--list-devices', action='store_true')
    args = p.parse_args(argv)
    if getattr(args,'frames',1)<1 or getattr(args,'stride',1)<1:
        p.error('--frames and --stride must be positive')
    if not np.isfinite(getattr(args,'gain',1)) or getattr(args,'gain',1)<=0:
        p.error('--gain must be finite and positive')
    {'bench': do_bench, 'write': do_write, 'read': do_read,
     'live-send': do_live_send, 'live-receive': do_live_receive}[args.command](args)


if __name__ == '__main__':
    main()
