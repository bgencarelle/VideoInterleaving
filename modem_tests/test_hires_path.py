"""Higher resolution at the same frame rate, end to end, both modes.

The goal is more picture without paying for it in fps. That needs three things
to line up, and each of them was broken separately:

  - a preset that HOLDS 2880 slots at close to lean-v3's rate, or fit_shapes
    collapses the sampling grid and the extra detail never reaches the wire;
  - a bake that keeps pixels above the wire shape, or there is nothing above
    the cut to preserve and truncation is pure loss;
  - a prebaked sender that follows the bake's profile instead of pinning
    DEFAULT_PROFILE, which silently box-downsampled an 80x96 bake to 40x48.
"""
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from animation_modem import transport3 as v3
from animation_modem.core import SourceCoder
from animation_modem.imaging import (PROFILE_GRIDS, image_values, plane_grids,
                                     plane_shapes, source_size, values_image)

HIRES = v3.ALL_PRESETS['hires-v3']


def detailed(size=(320, 384)):
    """A source with real detail above 40x48, which is the whole premise."""
    w, h = size
    y, x = np.mgrid[0:h, 0:w]/max(h, w)
    blob = np.exp(-((x-.42)**2+(y-.45)**2)/.045)
    rgb = np.stack([.55+.25*blob, .42+.20*blob, .38+.15*blob], -1)
    for cx, cy in ((.33, .38), (.53, .38)):
        rgb = rgb - .35*np.exp(-((x-cx)**2+(y-cy)**2)/.0008)[..., None]
    return Image.fromarray(np.uint8(np.clip(rgb*255, 0, 255)))


def native(size=(40, 48)):
    """Content that is genuinely AT the wire shape, with per-pixel structure.

    Not a downscale of `detailed()`: a LANCZOS reduction of a smooth source
    stays smooth, so it upsamples back almost exactly and reads as a win for
    truncation. Real bakes have pixel-level detail, and that is the case where
    a finer sampling grid has nothing to find.
    """
    w, h = size
    rng = np.random.default_rng(4242)
    yy, xx = np.mgrid[0:h, 0:w]
    mask = ((((xx-20)/8.8)**2 + ((yy-28)/16.3)**2) < 1) | \
           ((((xx-20)/5.2)**2 + ((yy-10)/6.2)**2) < 1)
    tex = 150+55*rng.standard_normal((h, w))
    im = np.full((h, w, 3), 4.)
    for c, k in enumerate((1., .82, .70)):
        im[:, :, c] = np.where(mask, tex*k, 4)
    return Image.fromarray(np.uint8(np.clip(im, 0, 255)))


def psnr(a, b, size=(320, 384)):
    x = np.asarray(a.resize(size, Image.LANCZOS), float)/255
    y = np.asarray(b.resize(size, Image.LANCZOS), float)/255
    err = np.mean((x-y)**2)
    return float('inf') if err <= 0 else 10*np.log10(1/err)


def through(layout, profile, image, noise_dbfs=None, grids=True):
    from animation_modem import impairments as IMP
    coder = SourceCoder(plane_shapes(profile),
                        grids=plane_grids(profile) if grids else None)
    audio = v3.encode(image_values(image, coder.grids), layout, coder, 1, 1, 1,
                      profile=v3.profile_code(profile))
    signal = np.concatenate([np.zeros((300, 2)), audio])
    if noise_dbfs is not None:
        emulator = IMP.Emulator(IMP.Settings(noise_dbfs=noise_dbfs))
        signal = np.concatenate([emulator.process(signal[i:i+256])
                                 for i in range(0, len(signal), 256)])
    rx = v3.Receiver(layout, coder)
    out, a = [], np.asarray(signal, np.float32)
    for i in range(0, len(a), 256):
        out += rx.feed(a[i:i+256])
    out += rx.flush()
    got = [r for r in out if r.values is not None]
    return (values_image(got[0].values, coder.grids) if got else None), coder


class PresetTests(unittest.TestCase):
    def test_hires_holds_color_dct_without_shrinking(self):
        """The point of the preset. lean-v3 (2200) and lean-v3-dense (2680)
        both fall short, and falling short is silent apart from a warning."""
        self.assertEqual(HIRES.capacity, 2880)
        shapes = plane_shapes('color-dct')
        self.assertEqual(sum(int(np.prod(s)) for s in shapes), HIRES.capacity)
        for smaller in ('lean-v3', 'lean-v3-dense'):
            self.assertLess(v3.ALL_PRESETS[smaller].capacity, 2880)

    def test_it_keeps_most_of_lean_v3s_frame_rate(self):
        lean = v3.ALL_PRESETS['lean-v3']
        self.assertGreater(HIRES.fps, .94*lean.fps)
        self.assertGreater(HIRES.fps, v3.ALL_PRESETS['wide-v3'].fps)


class ResolutionTests(unittest.TestCase):
    def test_the_picture_really_is_80x96(self):
        image, coder = through(HIRES, 'color-dct', detailed())
        self.assertEqual(image.size, (80, 96))
        self.assertEqual(coder.count, 2880)
        self.assertEqual(coder.source_count, 4*2880)

    def test_the_same_wire_at_a_higher_rate(self):
        """hires-v3 and wide-v3 carry nearly the same slots; hires is 15% faster.
        The claim is that the speed does not cost picture: wider bandwidth at
        the same slot count is the trade, and unlike lean-v3 it does not shrink
        the profile."""
        source = detailed()
        wide = v3.ALL_PRESETS['wide-v3']
        self.assertAlmostEqual(wide.capacity/HIRES.capacity, 3000/2880)
        self.assertGreater(HIRES.fps, wide.fps*1.14)
        for noise in (-45, -38):
            with self.subTest(noise=noise):
                at_wide_rate, _ = through(wide, 'color-dct', source, noise)
                at_hires_rate, _ = through(HIRES, 'color-dct', source, noise)
                self.assertIsNotNone(at_hires_rate)
                self.assertGreater(psnr(source, at_hires_rate),
                                   psnr(source, at_wide_rate)-1.5)

    def test_and_loses_on_a_source_with_no_detail_to_keep(self):
        """The honest other half, so this is not sold as free.

        Controlled: SAME preset, SAME 2880 slots, only the sampling grid
        differs. Truncation can only preserve what was sampled, so a source
        already at the wire shape has nothing above the cut and pays the
        truncation for nothing. Comparing against a different budget instead
        would confound this and read as a win.
        """
        flat = native()
        for noise in (None, -45, -38):
            with self.subTest(noise=noise):
                plain, _ = through(HIRES, 'color-dct', flat, noise, grids=False)
                truncated, _ = through(HIRES, 'color-dct', flat, noise)
                self.assertLess(psnr(flat, truncated, (40, 48)),
                                psnr(flat, plain, (40, 48)))

    def test_the_gain_is_the_sampling_grid_not_the_budget(self):
        """Same controlled pair the other way round: with detail present, the
        finer grid wins at an identical slot count."""
        source = detailed()
        plain, a = through(HIRES, 'color-dct', source, -38, grids=False)
        truncated, b = through(HIRES, 'color-dct', source, -38)
        self.assertEqual(a.count, b.count)
        self.assertGreater(psnr(source, truncated), psnr(source, plain))


class BakeTests(unittest.TestCase):
    def test_a_bake_declares_the_size_its_profile_samples_at(self):
        self.assertEqual(source_size('color-dct'), (80, 96))
        self.assertEqual(source_size('lean-dct'), (80, 96))

    def test_the_library_accepts_an_80x96_color_dct_bake(self):
        """It used to check manifest size against the WIRE shape, which
        rejected exactly the bakes color-dct exists for."""
        from modem_bake import ModemLibrary
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            entries = []
            for kind, name in (('main', 'face'), ('float', 'float')):
                folder = root/name/('0' if kind == 'main' else '255_a')
                folder.mkdir(parents=True)
                slab = np.zeros((2, 96, 80, 4), np.uint8)
                slab[..., 3] = 255
                slab[..., 0] = 120
                np.save(folder/'frames.npy', slab)
                entries.append({'layer': kind,
                                'path': str(folder.relative_to(root)),
                                'count': 2, 'names': ['a.png', 'b.png']})
            (root/'modem.json').write_text(json.dumps({
                'format': 'video-interleaving-modem', 'version': 1,
                'profile': 'color-dct', 'size': [80, 96],
                'jpeg_layout': 'sbs', 'folders': entries}))
            library = ModemLibrary(root)
            self.assertEqual(library.profile, 'color-dct')
            self.assertEqual(library.size, (80, 96))
            self.assertEqual(library.composite(0, 0, 0).size, (80, 96))

    def test_a_mismatched_bake_is_still_refused(self):
        """The check is moved, not removed: an 80x96 bake claiming a 40x48
        profile must still fail."""
        from modem_bake import ModemLibrary
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            folder = root/'face'/'0'
            folder.mkdir(parents=True)
            np.save(folder/'frames.npy', np.zeros((1, 96, 80, 4), np.uint8))
            (root/'modem.json').write_text(json.dumps({
                'format': 'video-interleaving-modem', 'version': 1,
                'profile': 'lean-dct', 'size': [40, 48], 'folders': []}))
            with self.assertRaises(ValueError):
                ModemLibrary(root)


class PrebakedSenderTests(unittest.TestCase):
    def test_the_sender_follows_the_bakes_profile(self):
        """It was pinned to DEFAULT_PROFILE, so an 80x96 bake was sampled back
        down to 40x48 before it ever reached the wire."""
        import modem_display
        import inspect
        source = inspect.getsource(modem_display.run_modem)
        self.assertIn('library.profile', source)
        self.assertNotIn('plane_shapes(DEFAULT_PROFILE)', source)

    def test_the_header_declares_what_the_coder_actually_used(self):
        """These disagreeing is the silent-wrong-geometry failure."""
        import modem_display
        import inspect
        source = inspect.getsource(modem_display.packet)
        self.assertIn('profile_code(profile)', source)


if __name__ == '__main__':
    unittest.main()
