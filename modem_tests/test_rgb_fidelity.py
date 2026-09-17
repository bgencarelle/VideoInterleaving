"""RGB-space picture fidelity: what vision sees, not just what bits say.

Bit-level metrics (identity, tier, pilot_error) can read ``best`` while the
rendered picture is saturated static: they watch QPSK decisions, and a
demod that gets every quadrant right can still reconstruct wild amplitudes.
These tests pin the rendered RGB error on a fixed impairment (causal 4th
order highpass at 300 Hz) that does exactly that -- header verifies, tier
says best, picture is ruined by 4-5x chroma blow-up from Wiener de-bias
dividing by misestimated gains on the bottom carriers.

The hp300 case is marked expectedFailure until the reliability fix lands:
it documents the gap (pixel error ~0.21 vs the <0.10 bar) while the clean
control below guards that any fix changes nothing on a good wire.
"""
import unittest

import numpy as np
from PIL import Image
from scipy.signal import butter, sosfilt

from animation_modem import transport3 as v3
from animation_modem.imaging import image_values, values_image
from modem_bake import ModemLibrary
from utilities.modem_v3_check import coder_for

LIB = ModemLibrary('images_modem')
LAYOUT = v3.WIRE
CODER, GRIDS = coder_for('color-dct', LAYOUT)
FRAMES = 4
PX = np.random.default_rng(11).integers(0, 80, 400)
PY = np.random.default_rng(11).integers(0, 96, 400)


def stream():
    return np.concatenate([
        np.asarray(v3.encode(image_values(LIB.composite(n, 1, 0), GRIDS),
                             LAYOUT, CODER, n+1, n+1, FRAMES, profile=0),
                   np.float32)
        for n in range(FRAMES)])


def highpass(wire, hz=300, order=4, rate=48000):
    return sosfilt(butter(order, hz/(rate/2), 'high', output='sos'), wire,
                   axis=0).astype(np.float32)


def hard_clip(wire, level=0.6):
    """Hot tape levels: symmetric flat tops preserve signs (header safe)
    but spray odd-harmonic distortion, heaviest low."""
    return np.clip(wire, -level, level).astype(np.float32)


def decode_all(wire):
    rx = v3.Receiver(LAYOUT, CODER)
    out = []
    for i in range(0, len(wire), 1024):
        out += rx.feed(np.asarray(wire[i:i+1024], np.float32))
    out += rx.flush()
    return [r for r in out if r.values is not None]


def to_rgb(values):
    img = values_image(values, GRIDS)
    if not isinstance(img, Image.Image):
        img = Image.fromarray(np.asarray(img))
    return np.asarray(img.convert('RGB').resize((80, 96)), float)/255


def src_rgb(n):
    return np.asarray(LIB.composite(n, 1, 0).convert('RGB').resize(
        (80, 96), Image.LANCZOS), float)/255


def pixel_error(results):
    errs = []
    for i, r in enumerate(results[:FRAMES]):
        d = to_rgb(r.values)[PY, PX]-src_rgb(i)[PY, PX]
        errs.append(float(np.mean(np.abs(d))))
    return float(np.mean(errs))


def chroma_range(results):
    cbs = [float(r.values[7680:9600].std()) for r in results[:FRAMES]]
    return float(np.mean(cbs))


class RGBFidelityTests(unittest.TestCase):
    def test_highpass_picture_holds_rgb_accuracy(self):
        """A highpassed channel used to render static while bits read best:
        dispersion biased the scale estimate (+0.26%), and demodulating at
        the biased scale mis-stretched the walk, destroying amplitudes. The
        receiver now also tries unity scale past SCALE_SANITY and keeps the
        better decode, so the picture holds (pixel error ~0.04 vs 0.21)."""
        got = decode_all(highpass(stream()))
        self.assertEqual(sum(1 for r in got
                             if r.identity == 'verified_header'), FRAMES)
        self.assertLess(pixel_error(got), 0.10)
        self.assertLess(chroma_range(got), 0.30)

    def test_clean_picture_is_untouched(self):
        """The bar a fix must not move: same content, no impairment."""
        got = decode_all(stream())
        self.assertEqual(sum(1 for r in got
                             if r.identity == 'verified_header'), FRAMES)
        self.assertLess(pixel_error(got), 0.05)

    def test_clipped_picture_holds_rgb_accuracy(self):
        """Symmetric clipping keeps zero crossings, so the header verifies
        untouched -- but the image inherits the harmonic spray. The decoder
        detects flat tops and discounts the contaminated low carriers
        (measured: pixel 0.21 -> 0.08, Cb std back to clean levels)."""
        got = decode_all(hard_clip(stream()))
        self.assertEqual(sum(1 for r in got
                             if r.identity == 'verified_header'), FRAMES)
        self.assertLess(pixel_error(got), 0.10)


if __name__ == '__main__':
    unittest.main()
