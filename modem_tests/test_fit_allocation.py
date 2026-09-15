"""Spending transmit power where this sequence actually puts its energy.

`default_allocation` estimates coefficient magnitude from spatial frequency
alone, with a 1/(1+12f) law that knows nothing about the content. It is a
reasonable prior and it leaves several dB unclaimed, because a real sequence
has structure a frequency prior cannot see -- a face in roughly the same place
every frame, a fixed background, chroma that barely moves.

Fitted on training frames and scored on a HELD-OUT frame, the gain is +3.4 to
+3.8 dB for no extra slots, which is the same order as doubling the slot count.
It converts into whichever axis is wanted: quality, frame rate, or band.

The table is not on the wire and nothing detects it, so both ends need the same
file -- and it fits exactly one (preset, profile) pair, because its length is
that pair's slot count.
"""
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from animation_modem import transport3 as v3, impairments as IMP
from animation_modem.core import SourceCoder, default_allocation
from animation_modem.imaging import (fit_shapes, image_values, plane_shapes,
                                     values_image)
from utilities import fit_allocation as fa

LEAN14 = v3.ALL_PRESETS['lean-14k']


def sequence(count, seed=100, size=(640, 768)):
    """Frames that share structure, which is the thing a fitted table sees."""
    w, h = size
    y, x = np.mgrid[0:h, 0:w]/max(h, w)
    out = []
    for i in range(count):
        t = i*.4
        cx, cy = .42+.05*np.sin(t), .45+.04*np.cos(t)
        blob = np.exp(-((x-cx)**2+(y-cy)**2)/.045)
        rgb = np.stack([.55+.25*blob, .42+.20*blob, .38+.15*blob], -1)
        for ex in (cx-.09, cx+.11):
            rgb = rgb - .35*np.exp(-((x-ex)**2+(y-cy+.07)**2)/.0008)[..., None]
        rng = np.random.default_rng(seed+i)
        spec = np.fft.rfft2(rng.normal(0, 1, (h, w, 3)), axes=(0, 1))
        spec /= (1+50*np.hypot(np.fft.fftfreq(h)[:, None, None],
                               np.fft.rfftfreq(w)[None, :, None]))
        tex = np.fft.irfft2(spec, s=(h, w), axes=(0, 1))
        tex = (tex-tex.min())/(tex.max()-tex.min())
        out.append(Image.fromarray(np.uint8(np.clip(.75*rgb+.25*tex, 0, 1)*255)))
    return out


def psnr(a, b, size=(320, 384)):
    x = np.asarray(a.resize(size, Image.LANCZOS), float)/255
    y = np.asarray(b.resize(size, Image.LANCZOS), float)/255
    err = np.mean((x-y)**2)
    return float('inf') if err <= 0 else 10*np.log10(1/err)


def through(layout, coder, image, noise_dbfs=-38):
    audio = v3.encode(image_values(image, coder.grids), layout, coder, 1, 1, 1,
                      profile=0)
    signal = np.concatenate([np.zeros((300, 2)), audio])
    emulator = IMP.Emulator(IMP.Settings(noise_dbfs=noise_dbfs))
    signal = np.concatenate([emulator.process(signal[i:i+256])
                             for i in range(0, len(signal), 256)])
    rx = v3.Receiver(layout, coder)
    out, a = [], np.asarray(signal, np.float32)
    for i in range(0, len(a), 256):
        out += rx.feed(a[i:i+256])
    out += rx.flush()
    got = [r for r in out if r.values is not None]
    return values_image(got[0].values, coder.grids) if got else None


class FitTests(unittest.TestCase):
    def setUp(self):
        self.shapes, self.grids = fa.geometry('color', LEAN14)
        self.count = int(sum(np.prod(s) for s in self.shapes))

    def test_the_table_is_one_positive_weight_per_slot(self):
        """SourceCoder refuses anything else, and a zero weight would make a
        slot uncarryable rather than merely cheap."""
        sigma = fa.fit(sequence(4), self.shapes, self.grids)
        self.assertEqual(sigma.shape, (self.count,))
        self.assertTrue(np.all(np.isfinite(sigma)))
        self.assertTrue(np.all(sigma > 0))
        SourceCoder(self.shapes, sigma, grids=self.grids)   # must not raise

    def test_it_beats_the_frequency_prior_on_a_held_out_frame(self):
        """The claim, and the honest way to measure it: the frame scored was
        not one of the frames fitted on."""
        train = sequence(8, seed=100)
        held_out = sequence(1, seed=999)[0]
        sigma = fa.fit(train, self.shapes, self.grids)
        default = through(LEAN14, SourceCoder(self.shapes, grids=self.grids),
                          held_out)
        fitted = through(LEAN14, SourceCoder(self.shapes, sigma,
                                             grids=self.grids), held_out)
        self.assertIsNotNone(fitted)
        self.assertGreater(psnr(held_out, fitted), psnr(held_out, default)+1.0)

    def test_it_differs_from_the_default_table(self):
        """If the fit reproduced the prior there would be nothing to gain."""
        sigma = fa.fit(sequence(6), self.shapes, self.grids)
        prior = default_allocation(self.shapes, self.count)
        a = sigma/np.mean(sigma)
        b = prior/np.mean(prior)
        self.assertGreater(float(np.max(np.abs(a-b))), .5)

    def test_fitting_on_nothing_is_refused_rather_than_guessed(self):
        with self.assertRaises(SystemExit):
            fa.main(['--frames', '0', '--out', '/dev/null'])


class PairingTests(unittest.TestCase):
    """A table belongs to one (preset, profile) pair."""

    def table(self, tmp, preset='lean-14k', profile='color'):
        path = Path(tmp)/'a.npy'
        fa.main(['--preset', preset, '--profile', profile, '--frames', '4',
                 '--out', str(path)])
        return path

    def test_a_named_pair_refuses_a_table_of_the_wrong_length(self):
        """Where the caller named the preset, a mismatch is a mistake."""
        from utilities import modem_v3_check as check
        with tempfile.TemporaryDirectory() as tmp:
            path = self.table(tmp)
            with self.assertRaises(SystemExit) as caught:
                check.coder_for('color', path, v3.ALL_PRESETS['hires-v3'])
            self.assertIn('2880', str(caught.exception))

    def test_scanning_falls_back_instead_of_dying(self):
        """Candidate detection builds a coder for EVERY preset, so a table that
        fits one of them must not stop the others being tried -- that would
        make --allocation and preset detection mutually exclusive."""
        from utilities import modem_v3_check as check
        with tempfile.TemporaryDirectory() as tmp:
            path = self.table(tmp)
            coders = check.coders_for(v3.ALL_PRESETS['hires-v3'], path)
            self.assertTrue(coders)
            candidates = check.candidates_for(path)
            self.assertGreater(len(candidates), 4)

    def test_the_fitted_pair_still_gets_the_table_while_scanning(self):
        from utilities import modem_v3_check as check
        with tempfile.TemporaryDirectory() as tmp:
            path = self.table(tmp)
            plain = check.coder_for('color', None, LEAN14)[0]
            fitted = check.coder_for('color', path, LEAN14, strict=False)[0]
            self.assertFalse(np.allclose(plain.gains, fitted.gains))


if __name__ == '__main__':
    unittest.main()
