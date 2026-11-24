"""P: the frame on screen to the computer's default printer."""
import subprocess
import unittest
from unittest import mock

import numpy as np

import system_printer as sp


def run_result(stdout):
    return subprocess.CompletedProcess([], 0, stdout=stdout, stderr='')


class DefaultPrinterTests(unittest.TestCase):
    def setUp(self):
        patches = [mock.patch.object(sp.sys, 'platform', 'linux'),
                   mock.patch.object(sp.shutil, 'which', lambda name: '/usr/bin/' + name)]
        for p in patches:
            p.start()
            self.addCleanup(p.stop)

    def test_cups_default_is_read_from_lpstat(self):
        with mock.patch.object(sp.subprocess, 'run',
                               return_value=run_result('system default destination: Office_Laser\n')):
            self.assertEqual(sp.default_printer(), 'Office_Laser')

    def test_no_default_means_no_printer(self):
        with mock.patch.object(sp.subprocess, 'run',
                               return_value=run_result('no system default destination\n')):
            self.assertIsNone(sp.default_printer())

    def test_no_cups_means_no_printer(self):
        with mock.patch.object(sp.shutil, 'which', lambda name: None):
            self.assertIsNone(sp.default_printer())

    def test_print_goes_to_lp_fitted_to_the_page(self):
        calls = []
        with mock.patch.object(sp.subprocess, 'run', lambda cmd, **kw: calls.append(cmd)):
            self.assertEqual(sp.print_frame(np.zeros((40, 30, 3), np.uint8),
                                            printer='Office_Laser'), 'Office_Laser')
        self.assertEqual(calls[0][:6], ['lp', '-d', 'Office_Laser', '-t',
                                        'VideoInterleaving frame', '-o'])
        self.assertIn('fit-to-page', calls[0])
        self.assertTrue(calls[0][-1].endswith('.png'))

    def test_nothing_to_print_to_prints_nothing(self):
        with mock.patch.object(sp, 'default_printer', lambda: None):
            self.assertFalse(sp.print_frame(np.zeros((4, 4, 3), np.uint8)))


class FrameImageTests(unittest.TestCase):
    def test_the_picture_prints_as_handed_over(self):
        rgb = np.zeros((40, 30, 3), np.uint8)
        rgb[..., 0] = 200
        image = sp.frame_image(rgb)
        self.assertEqual((image.mode, image.size, image.getpixel((0, 0))), ('RGB', (30, 40), (200, 0, 0)))

    def test_a_wide_picture_goes_on_a_landscape_page(self):
        calls = []
        with mock.patch.object(sp.sys, 'platform', 'linux'), \
                mock.patch.object(sp.subprocess, 'run', lambda cmd, **kw: calls.append(cmd)):
            sp.print_frame(np.zeros((30, 50, 3), np.uint8), printer='P')
            sp.print_frame(np.zeros((50, 30, 3), np.uint8), printer='P')
        self.assertIn('orientation-requested=4', calls[0])
        self.assertNotIn('orientation-requested=4', calls[1])


if __name__ == '__main__':
    unittest.main()
