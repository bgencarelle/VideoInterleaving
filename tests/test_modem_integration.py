import contextlib
import io
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import wave

import numpy as np
from PIL import Image

from utilities.convert_to_modem_dct import bake_tree_dct as bake_tree
from animation_modem import v7

REPO = Path(__file__).resolve().parents[1]


def source_tree(root, count=3):
    for name, color in [
        ('face/0_rest', (0, 0, 0, 255)),
        ('face/2_face', (200, 40, 20, 128)),
        ('face/10_face', (20, 200, 40, 255)),
        ('float/255_overlay', (10, 30, 240, 128)),
    ]:
        folder = root / name
        folder.mkdir(parents=True)
        for i in range(count):
            Image.new('RGBA', (40, 48), color).save(folder / f'frame_{i}.png')


class ModemV7IntegrationTests(unittest.TestCase):
    def test_main_wav_path_uses_v7_without_legacy_modem_stack(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / 'images'
            source_tree(source)
            bake = root / 'images_modem'
            with contextlib.redirect_stdout(io.StringIO()):
                bake_tree(source, bake)

            wav = root / 'test.wav'
            script = """
import sys, runpy
class Block:
    def find_spec(self, name, path=None, target=None):
        if name.split('.')[0] in ('glfw', 'OpenGL', 'moderngl', 'turbojpeg',
                                  'sounddevice', 'mido', 'cv2'):
            raise ImportError('blocked unused dependency ' + name)
sys.meta_path.insert(0, Block())
target = sys.argv.pop(1)
sys.path.insert(0, str(__import__('pathlib').Path(target).parent))
runpy.run_path(target, run_name='__main__')
"""
            result = subprocess.run(
                [sys.executable, '-c', script, str(REPO / 'main.py'),
                 '--mode', 'modem', '--modem-dir', str(bake),
                 '--modem-pair', '1,0', '--modem-wav', str(wav),
                 '--modem-frames', '5', '--modem-speed', '1.5', '-f'],
                cwd=root, text=True, capture_output=True, timeout=30)
            self.assertEqual(result.returncode, 0,
                             result.stdout + result.stderr)

            with wave.open(str(wav), 'rb') as source_wav:
                samples = np.frombuffer(
                    source_wav.readframes(source_wav.getnframes()),
                    dtype='<i2').reshape(-1, 2).astype(float) / 32767
            model = v7.build_model(
                REPO / 'modem_tests/fixtures/v7_reference_face.png',
                .1521 / np.sqrt(1 + 10**(v7.CLOCK_REL_DB / 10)),
                encode_filter='nearest')
            decoded, info = v7.decode_pulse_stream(model, samples)
            self.assertGreaterEqual(len(decoded), 4, info)
            self.assertEqual(
                [item.diag['source_index'] for item in decoded[:4]],
                [0, 1, 2, 0])


if __name__ == '__main__':
    unittest.main()
