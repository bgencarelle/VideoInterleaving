"""Screen capture routing without opening screen or audio devices."""
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import modem_screen


class ScreenCaptureTests(unittest.TestCase):
    def test_screen_discovers_screen_after_camera_and_prescales(self):
        listing = SimpleNamespace(stderr=(
            '[AVFoundation] [0] FaceTime HD Camera\n'
            '[AVFoundation] [3] Capture screen 0\n'))
        args = modem_screen.parser().parse_args([
            '--source', 'screen', '--capture-width', '480',
            '--region', '10,20,800,600'])
        with patch.object(modem_screen.sys, 'platform', 'darwin'), \
             patch.object(modem_screen.shutil, 'which', return_value='ffmpeg'), \
             patch.object(modem_screen.subprocess, 'run', return_value=listing), \
             patch.object(modem_screen, 'ffmpeg_source') as capture:
            modem_screen.source_for(args, 13.76)
        capture.assert_called_once_with(None, 13.76, [10, 20, 800, 600], 3, 480)

    def test_explicit_display_skips_discovery(self):
        with patch.object(modem_screen.sys, 'platform', 'darwin'), \
             patch.object(modem_screen.subprocess, 'run') as discover, \
             patch.object(modem_screen, 'ffmpeg_source') as capture:
            modem_screen.screen_capture_source(15, display=4)
        discover.assert_not_called()
        capture.assert_called_once_with(None, 15, None, 4, 320)

    def test_missing_screen_does_not_fall_back_to_camera(self):
        with patch.object(modem_screen.sys, 'platform', 'darwin'), \
             patch.object(modem_screen.shutil, 'which', return_value='ffmpeg'), \
             patch.object(modem_screen.subprocess, 'run',
                          return_value=SimpleNamespace(stderr='[0] Camera')), \
             patch.object(modem_screen, 'ffmpeg_source') as capture:
            with self.assertRaisesRegex(SystemExit, 'No FFmpeg screen device'):
                modem_screen.screen_capture_source(15)
        capture.assert_not_called()
