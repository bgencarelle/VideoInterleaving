"""Visual mono torture runner: real WAV fold-down and comparable PNG output."""
import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.io import wavfile

from tools.v7_mono_torture import RATE, main, mono_wav


class V7MonoTortureTests(unittest.TestCase):
    def test_wav_contains_the_equal_weight_mono_mix(self):
        stereo = np.array([[.4, -.4], [.8, .2], [-.6, -.2]], np.float32)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'mono.wav'
            samples = mono_wav(path, stereo)
            rate, recorded = wavfile.read(path)
            self.assertEqual(rate, RATE)
            self.assertEqual(recorded.ndim, 1)
            np.testing.assert_array_equal(recorded, samples)
            np.testing.assert_allclose(recorded, [0, .5, -.4], atol=1e-7)

    def test_clean_comparison_exports_decoded_images_and_original_scoring(self):
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory)
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(main(['--out', str(out), '--only', 'clean-96k',
                                       '--frames', '3']), 0)
            rows = {r['variant']: r for r in json.loads(
                (out / 'results.json').read_text())}
            self.assertEqual(set(rows), {'stereo-80x96', 'mono-80x96',
                                         'mono-40x48', 'mono-20x24'})
            for variant, row in rows.items():
                self.assertEqual(row['displayable'], 2, variant)
                self.assertEqual(row['metadata_valid'], 2, variant)
                with Image.open(out / f'clean-96k_{variant}.png') as image:
                    self.assertEqual(image.size, (80, 96))
            with Image.open(out / 'clean-96k_comparison.png') as image:
                self.assertEqual(image.size, (1672, 460))
            # Score against the same original, not each smoothed source: a
            # smaller source must not look artificially better by this metric.
            psnr = lambda variant: rows[variant]['image_quality']['Y']['psnr_db']
            self.assertGreater(psnr('stereo-80x96'), psnr('mono-80x96'))
            self.assertGreater(psnr('mono-80x96'), psnr('mono-20x24'))


if __name__ == '__main__':
    unittest.main()
