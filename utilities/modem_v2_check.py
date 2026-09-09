#!/usr/bin/env python3
"""Exercise v2 transport: bench it, or run an independent receive/tape loop.

Three things you can do with this:

  # 1. see the presets and bench v2 against v1 on synthetic or baked frames
  python utilities/modem_v2_check.py bench --modem-dir images_modem

  # 2. build the power-allocation table from your bake (v2 needs this to be
  #    worth anything -- without it the source coder is guessing)
  python utilities/modem_v2_check.py allocate --modem-dir images_modem \
      --out modem_allocation.npy

  # 3. the actual tape test: write a WAV, record it to tape, play it back,
  #    capture to a WAV, and decode that
  python utilities/modem_v2_check.py write --modem-dir images_modem \
      --preset tape --frames 200 --out v2_tape.wav --allocation modem_allocation.npy
  python utilities/modem_v2_check.py read --preset tape \
      --wav captured_off_tape.wav --allocation modem_allocation.npy --save-frames out/

  # 4. real time, two terminals. Start the receiver first.
  python utilities/modem_v2_check.py live-receive --device "BlackHole 2ch" --preset tape
  python utilities/modem_v2_check.py live-send --modem-dir images_modem \
      --device "BlackHole 2ch" --preset tape --allocation modem_allocation.npy

  # list audio devices
  python utilities/modem_v2_check.py live-send --list-devices

The read step reports per-frame status, quality tier, measured playback rate
and PSNR where it can, so a deck's speed error and roll-off show up as numbers
rather than a verdict.
"""
import argparse
import json
from pathlib import Path
import sys
import wave

import numpy as np
from PIL import Image, ImageOps
from scipy.fft import dctn

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from animation_modem import transport as V1                    # noqa: E402
from animation_modem import transport2 as V2                   # noqa: E402
from animation_modem import impairments as IMP                 # noqa: E402
from animation_modem.audio_common import (InputFilter, pcm, pair, device,   # noqa: E402
                                          wav_blocks)


def frames_from(args, profile='color'):
    """Real composites from a bake, or a synthetic stand-in."""
    if args.modem_dir:
        from modem_bake import ModemLibrary
        library = ModemLibrary(args.modem_dir)
        picks = range(0, min(library.frames, args.frames*args.stride), args.stride)
        return [library.composite(i, 0, 0) for i in picks], library.profile
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


def psnr(a, b):
    error = ((np.asarray(a, float) - np.asarray(b, float))**2).mean()
    return float('inf') if error == 0 else 10*np.log10(255**2/error)


def coder_for(profile, allocation, layout=None):
    shapes = V1.plane_shapes(profile)

    if layout and sum(np.prod(s) for s in shapes) > layout.capacity:
        h, w = shapes[0]
        choices = []

        for width in range(2, w + 1, 2):
            for height in range(2, h + 1, 2):
                trial = [(height, width)]

                if len(shapes) == 3:
                    trial += [(height // 2, width // 2)] * 2

                count = sum(np.prod(s) for s in trial)

                if count <= layout.capacity:
                    score = count / (1 + abs(width / height - w / h))
                    choices.append((score, trial))

        shapes = max(choices, key=lambda item: item[0])[1]

    table = np.load(allocation) if allocation else None
    return V2.SourceCoder(shapes, table), shapes

def image_values(image, coder):
    h, w = coder.shapes[0]
    image = ImageOps.pad(
        image.convert("RGB"),
        (w, h),
        method=Image.Resampling.LANCZOS,
    )

    planes = image.convert("YCbCr").split()

    return np.concatenate([
        np.asarray(
            plane.resize(
                (shape[1], shape[0]),
                Image.Resampling.BOX,
            )
        ).ravel()
        for plane, shape in zip(planes, coder.shapes)
    ]).astype(float) / 127.5 - 1

def values_image(values,coder):
    planes=[];offset=0
    for h,w in coder.shapes:
        pixels=np.uint8(np.clip(np.rint((values[offset:offset+h*w]+1)*127.5),0,255))
        planes.append(Image.fromarray(pixels.reshape(h,w)))
        offset+=h*w
    if len(planes)==1:return planes[0].convert('RGB')
    return Image.merge('YCbCr',(planes[0],*[p.resize(planes[0].size,Image.Resampling.BILINEAR)
                                          for p in planes[1:]])).convert('RGB')


def receive_for(args,layout,coder):
    return V2.Receiver(layout,coder,min_speed=args.min_speed,max_speed=args.max_speed)


def record(r):
    speed=1/(1+r.rate_error)
    return {'status':r.status,'identity':r.identity,'frame':r.absolute,
            'index':r.index,'count':r.count,'tier':r.tier,'coverage':r.coverage,
            'pilot_error':r.pilot_error,'playback_speed':speed,
            'playback_rate_pct':round(100*(speed-1),3),**r.extra}


def do_allocate(args):
    """Per-coefficient energy over the bake. This is the table v2 wants."""
    frames, profile = frames_from(args, )
    coder,shapes = coder_for(profile,None,V2.PRESETS[args.preset])
    total, n = None, 0
    for im in frames:
        values = image_values(im,coder)
        pieces, offset = [], 0
        for shape in shapes:
            size = int(np.prod(shape))
            pieces.append(dctn(values[offset:offset+size].reshape(shape), norm='ortho').ravel())
            offset += size
        c = np.concatenate(pieces)**2
        total = c if total is None else total + c
        n += 1
    sigma = np.sqrt(total/max(n, 1))
    sigma = np.maximum(sigma, sigma.max()*1e-4)      # never allocate exactly zero
    np.save(args.out, sigma)
    print(f'{n} frames, profile {profile}, {len(sigma)} coefficients -> {args.out}')
    print(f'  energy spread: max/min = {sigma.max()/sigma.min():.1f}  '
          f'(a flat spread means allocation buys nothing)')


def do_write(args):
    frames, profile = frames_from(args)
    layout = V2.PRESETS[args.preset]
    coder, _ = coder_for(profile, args.allocation, layout)

    print(layout.describe())

    path = Path(args.out)
    with wave.open(str(path), "wb") as sink:
        sink.setparams((2, 2, V2.RATE, 0, "NONE", "not compressed"))

        for n, im in enumerate(frames):
            prepared = V1.prepare_image(
                im,
                n + 1,
                1,
                len(frames),
                args.numbered,
                profile,
            )

            values = image_values(prepared, coder)

            audio = V2.encode(
                values,
                layout,
                coder,
                n + 1,
                (n % len(frames)) + 1,
                len(frames),
                stamp_ms=n * int(1000 / layout.fps),
            )

            sink.writeframesraw(pcm(audio * args.gain))

def do_read(args):
    layout = V2.PRESETS[args.preset]
    coder, _ = coder_for(args.profile, args.allocation,layout)
    print(layout.describe(), file=sys.stderr)
    receiver = receive_for(args,layout,coder)
    conditioner = InputFilter((600*args.min_speed,22000)) if args.input_filter else None
    if args.save_frames:
        Path(args.save_frames).mkdir(parents=True, exist_ok=True)
    tiers, rates, seen = {}, [], 0
    def results():
        for block in wav_blocks(args.wav,args.channels,1024):
            if conditioner is not None:block=conditioner.process(block)
            yield from receiver.feed(block)
        yield from receiver.flush()
    for r in results():
        seen += 1
        tiers[r.tier] = tiers.get(r.tier, 0) + 1
        rates.append(r.rate_error)
        print(json.dumps(record(r)),flush=True)
        if args.save_frames and r.values is not None:
            name = f'{r.absolute:06d}' if r.absolute is not None else f'x{seen:06d}'
            values_image(r.values,coder).save(Path(args.save_frames)/f'frame_{name}.png')
    if rates:
        speed = 1/(1+float(np.median(rates)))
        print(f'\n{seen} packets. tiers {tiers}. '
              f'median playback speed {speed:.4f}x '
              f'({100*(speed-1):+.2f}% off nominal)', file=sys.stderr)
    else:
        print('\nNothing decoded. Check the channel pair and the level.', file=sys.stderr)


def do_bench(args):
    frames, profile = frames_from(args)
    print('\n'.join('  '+l.describe() for l in V2.PRESETS.values()) + '\n')
    layout = V2.PRESETS[args.preset]
    coder, _ = coder_for(profile, args.allocation,layout)
    channels = [('clean', {}),
                ('cassette-ish', dict(lowpass_hz=10000, noise_dbfs=-45)),
                ('worn deck', dict(lowpass_hz=8000, noise_dbfs=-40, crosstalk=.07))]
    print(f'  {"channel":<16}{"v1 PSNR":>9}{"v2 PSNR":>9}{"v2 hdr":>8}{"v2 tier":>10}')
    for name, settings in channels:
        v1q, v2q, hdr, tier = [], [], 0, {}
        for n, im in enumerate(frames[:args.frames]):
            ready = V1.prepare_image(im, n+1, 1, len(frames), False, profile)
            values = V1.image_values(ready, profile)
            # v1
            audio, ref, _ = V1.encode(im, n+1, 1, len(frames), False, profile)
            emu = IMP.Emulator(IMP.Settings(**settings))
            rx = V1.Receiver(track_rate=False)
            got = []
            sig = np.concatenate([np.zeros((300, 2), np.float32), audio,
                                  np.zeros((V2.RATE//8, 2), np.float32)])
            for i in range(0, len(sig), 256):
                got += rx.feed(emu.process(sig[i:i+256]))
            good = [g for g in got if g.image is not None]
            if good:
                v1q.append(psnr(good[0].image, ref))
            # v2
            audio2 = V2.encode(image_values(ready,coder), layout, coder, n+1, 1, len(frames))
            emu2 = IMP.Emulator(IMP.Settings(**settings))
            rx2 = V2.Receiver(layout, coder)
            got2 = []
            sig2 = np.concatenate([np.zeros((300, 2), np.float32), audio2])
            for i in range(0, len(sig2), 256):
                got2 += rx2.feed(emu2.process(sig2[i:i+256]))
            got2 += rx2.flush()
            good2 = [g for g in got2 if g.values is not None]
            if good2:
                v2q.append(psnr(values_image(good2[0].values,coder).resize(ready.size), ready))
                tier[good2[0].tier] = tier.get(good2[0].tier, 0)+1
                hdr += good2[0].identity == 'verified_header'
        print(f'  {name:<16}{np.mean(v1q) if v1q else float("nan"):>9.2f}'
              f'{np.mean(v2q) if v2q else float("nan"):>9.2f}'
              f'{hdr:>8}{max(tier, key=tier.get) if tier else "-":>10}')
    print('\n  Small presets include a resolution tradeoff. Compare clean and impaired rows;\n'
          '  an allocation table must be made for the same preset and used on both ends.')


def _sounddevice():
    try:
        import sounddevice as sd
        return sd
    except (ImportError, OSError) as exc:
        raise SystemExit('Live audio needs sounddevice and PortAudio: '
                         'pip install -r requirements-modem.txt') from exc


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
                   V1.prepare_image(im, n+1, 1, len(frames), args.numbered, profile), coder),
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
    conditioner = InputFilter((600*args.min_speed,22000)) if args.input_filter else None
    if args.save_frames:
        Path(args.save_frames).mkdir(parents=True, exist_ok=True)
    newest = {'result': None, 'seen': 0, 'tiers': {}}
    stop = threading.Event()

    def pump():
        nonlocal conditioner
        with sd.InputStream(samplerate=V2.RATE, channels=max(channels)+1, dtype='float32',
                            device=args.device, blocksize=256, latency='low') as stream:
            print(json.dumps({'input_latency_ms': stream.latency*1000}), file=sys.stderr)
            while not stop.is_set():
                audio, overflow = stream.read(256)
                if overflow:
                    print(json.dumps({'status': 'input_overflow'}), flush=True)
                    receiver.reset()
                    if conditioner is not None:
                        conditioner=InputFilter((600*args.min_speed,22000))
                block = audio[:, channels]
                if conditioner is not None:
                    block = conditioner.process(block)
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
                        values_image(r.values,coder).save(
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
            im = values_image(r.values,coder)
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
    def shared(q):
        q.add_argument('--modem-dir', type=Path, help='Bake to read frames from')
        q.add_argument('--frames', type=int, default=24)
        q.add_argument('--stride', type=int, default=7)
        q.add_argument('--allocation', type=Path, help='Table from the allocate step')
        q.add_argument('--preset', choices=list(V2.PRESETS), default='wide')
    b = sub.add_parser('bench'); shared(b)
    a = sub.add_parser('allocate'); shared(a)
    a.add_argument('--out', type=Path, default=Path('modem_allocation.npy'))
    w = sub.add_parser('write'); shared(w)
    w.add_argument('--out', type=Path, default=Path('v2_test.wav'))
    w.add_argument('--gain', type=float, default=1.0, help='Output scale before PCM')
    w.add_argument('-f', '--numbered', action='store_true')
    r = sub.add_parser('read')
    r.add_argument('--wav', type=Path, required=True)
    r.add_argument('--channels', type=pair,
                   default=(0, 1))
    r.add_argument('--allocation', type=Path)
    r.add_argument('--preset', choices=list(V2.PRESETS), default='wide')
    r.add_argument('--profile', choices=list(V1.PROFILES), default='color')
    r.add_argument('--save-frames', type=Path)
    r.add_argument('--input-filter', action=argparse.BooleanOptionalAction, default=True)
    ls = sub.add_parser('live-send'); shared(ls)
    ls.add_argument('--device',type=device); ls.add_argument('--channels',type=pair,default=(0,1))
    ls.add_argument('--gain', type=float, default=1.0)
    ls.add_argument('-f', '--numbered', action='store_true')
    ls.add_argument('--list-devices', action='store_true')
    lr = sub.add_parser('live-receive')
    lr.add_argument('--device',type=device); lr.add_argument('--channels',type=pair,default=(0,1))
    lr.add_argument('--allocation', type=Path)
    lr.add_argument('--preset', choices=list(V2.PRESETS), default='wide')
    lr.add_argument('--profile', choices=list(V1.PROFILES), default='color')
    lr.add_argument('--save-frames', type=Path)
    lr.add_argument('--input-filter', action=argparse.BooleanOptionalAction, default=True)
    lr.add_argument('--headless', action='store_true', help='JSON only, no window')
    lr.add_argument('--quiet', action='store_true', help='Window only, no per-packet JSON')
    lr.add_argument('--width', type=int, default=480)
    lr.add_argument('--height', type=int, default=576)
    lr.add_argument('--list-devices', action='store_true')
    for parser in (r,lr):
        parser.add_argument('--min-speed',type=float,default=.5,
                            help='Minimum playback speed searched (.25 to 1, default .5)')
        parser.add_argument('--max-speed',type=float,default=2.,
                            help='Maximum playback speed searched (1 to 2, default 2)')
    args = p.parse_args(argv)
    if getattr(args,'frames',1)<1 or getattr(args,'stride',1)<1:
        p.error('--frames and --stride must be positive')
    if not np.isfinite(getattr(args,'gain',1)) or getattr(args,'gain',1)<=0:
        p.error('--gain must be finite and positive')
    {'bench': do_bench, 'allocate': do_allocate, 'write': do_write, 'read': do_read,
     'live-send': do_live_send, 'live-receive': do_live_receive}[args.command](args)


if __name__ == '__main__':
    main()
