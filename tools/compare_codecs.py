#!/usr/bin/env python3
"""v3 (color-dct) against v5 (hd-dwt) through the same impaired channels.

Every frame is fitted to the 80x96 picture once, then sent through each
codec's real encode -> impairments.Emulator -> Receiver, and scored as PSNR
(dB, RGB) against that fitted frame. Higher is better; the last row is v5
minus v3. No audio devices.

Frames are synthetic by default (a detailed test card, modem_screen's test
blob, three 1/f "natural" frames). Pass --modem-dir to score your real bake
instead -- that is the number that matters, since synthetic frames only
approximate real content.

The channels are cheap stand-ins, not tape: tape emulation cannot be made
realistic (AGENTS.md). They probe the failure modes that matter -- lost
treble, hiss, a dead leg, a bass cut, clipping.

    .venv/bin/python tools/compare_codecs.py
    .venv/bin/python tools/compare_codecs.py --modem-dir images_modem --frames 8
"""
import argparse
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import modem_screen                                                  # noqa: E402
from animation_modem import engines as ENG                           # noqa: E402
from animation_modem import impairments as IMP                       # noqa: E402
from animation_modem import transport3 as V3                         # noqa: E402
from animation_modem.imaging import image_values, values_image       # noqa: E402

CHANNELS = [
    ('clean', {}),
    ('cassette', dict(lowpass_hz=10000, noise_dbfs=-45)),
    ('worn deck', dict(lowpass_hz=8000, noise_dbfs=-40, crosstalk=.07)),
    ('hiss -30', dict(noise_dbfs=-30)),
    ('lowpass 6k', dict(lowpass_hz=6000, noise_dbfs=-40)),
    ('R leg dead', dict(right_gain_db=-80, noise_dbfs=-40)),
    ('bass cut 1k', dict(highpass_hz=1000, noise_dbfs=-45)),
    ('clip .5', dict(clip=.5, noise_dbfs=-45)),
]


def test_card():
    rng = np.random.default_rng(3)
    yy, xx = np.mgrid[0:480, 0:400]
    src = np.zeros((480, 400, 3), np.uint8)
    src[..., 0] = 128 + 100*np.sin(xx/9.)*np.cos(yy/13.)
    src[..., 1] = xx*255//400
    src[..., 2] = yy*255//480
    im = Image.fromarray(src)
    draw = ImageDraw.Draw(im)
    for i in range(6):
        draw.ellipse([30+i*55, 60+i*50, 90+i*55, 140+i*50],
                     fill=tuple(rng.integers(0, 255, 3).tolist()))
    draw.rectangle([150, 200, 260, 380], outline=(255, 255, 255), width=12)
    return np.asarray(im)


def natural(count, seed=99):
    """Frames with a natural-image 1/f spectrum."""
    rng = np.random.default_rng(seed)
    fy, fx = np.meshgrid(np.fft.fftfreq(480), np.fft.fftfreq(400), indexing='ij')
    f = np.hypot(fy, fx)
    f[0, 0] = 1
    out = []
    for _ in range(count):
        ch = [np.real(np.fft.ifft2((rng.standard_normal(f.shape)
                                    + 1j*rng.standard_normal(f.shape))/f))
              for _ in range(3)]
        rgb = np.stack([ch[0] + .5*ch[1], ch[0] + .3*ch[2],
                        ch[0] - .2*ch[1] + .4*ch[2]], -1)
        rgb = (rgb - rgb.mean())/rgb.std()*55 + rng.uniform(70, 180, 3)
        out.append(np.uint8(np.clip(rgb, 0, 255)))
    return out


def frames(args):
    if args.modem_dir:
        from modem_bake import ModemLibrary
        library = ModemLibrary(args.modem_dir)
        picks = np.linspace(0, library.frames-1, args.frames).astype(int)
        return [np.asarray(library.composite(int(i), 0, 0).convert('RGB'))
                for i in picks]
    return [test_card(), modem_screen.test_source()()] + natural(3)


def score(codec, profile, raws, settings, packets=4):
    engine = ENG.get_engine(codec)
    layout = engine.wire
    coder, _ = engine.coder_for(profile, layout)
    code = engine.profile_code(profile)
    prepare = modem_screen.fitter('hd-dwt')        # the same 80x96 frame for both
    results = []
    for raw in raws:
        ref = prepare(raw)
        want = np.asarray(ref.convert('RGB'), float)
        values = image_values(ref, coder.grids)
        audio = np.concatenate(
            [np.zeros((300, 2), np.float32)] +
            [V3.encode(values, layout, coder, n+1, n+1, packets, profile=code)
             for n in range(packets)])
        emulator = IMP.Emulator(IMP.Settings(**settings))
        receiver = V3.Receiver(layout, coder, coders={code: coder})
        got = [r for i in range(0, len(audio), 512)
               for r in receiver.feed(emulator.process(audio[i:i+512]))]
        got += receiver.flush()
        psnrs = [10*np.log10(255**2/np.mean((np.asarray(
                     values_image(r.values, coder.grids).convert('RGB'), float)
                     - want)**2))
                 for r in got if r.values is not None]
        # A packet that never decoded scores 0 dB, not a skipped sample.
        psnrs += [0.0]*(packets - len(psnrs))
        results.append(float(np.median(psnrs)))
    return float(np.mean(results))


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--modem-dir', type=Path, help='Score frames from a real bake')
    p.add_argument('--frames', type=int, default=6,
                   help='Bake frames to sample with --modem-dir')
    args = p.parse_args(argv)
    raws = frames(args)
    width = max(len(c) for c, _ in CHANNELS) + 2
    print(f'mean PSNR dB over {len(raws)} frames'.ljust(20)
          + ''.join(c.rjust(width) for c, _ in CHANNELS))
    rows = {}
    for codec, profile in (('v3', 'color-dct'), ('v5', 'hd-dwt')):
        rows[codec] = [score(codec, profile, raws, s) for _, s in CHANNELS]
        print(f'{codec} {profile}'.ljust(20)
              + ''.join(f'{v:.2f}'.rjust(width) for v in rows[codec]), flush=True)
    print('v5 - v3'.ljust(20) + ''.join(
        f'{b-a:+.2f}'.rjust(width) for a, b in zip(rows['v3'], rows['v5'])))


if __name__ == '__main__':
    main()
