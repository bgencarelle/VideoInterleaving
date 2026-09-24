import contextlib
import io
import json
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
from modem_bake import ModemLibrary

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
            Image.new('RGBA', (80, 45), color).save(folder / f'frame_{i}.png')


class ModemV7IntegrationTests(unittest.TestCase):
    def test_generated_bake_matches_v7_geometry(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / 'images'
            source_tree(source)
            bake = root / 'images_modem'
            with contextlib.redirect_stdout(io.StringIO()):
                bake_tree(source, bake)

            library = ModemLibrary(bake)
            self.assertEqual(library.profile, 'color-dct')
            self.assertEqual(library.size, (80, 96))
            self.assertEqual(library.frames, 3)
            composite = library.composite(0, 0, 0)
            self.assertEqual(composite.size, (80, 96))
            self.assertEqual(composite.info['source_dimensions'], (80, 45))
            self.assertEqual(
                v7.aspect_wire_code(composite.info['source_dimensions']), 3)
            rotated = library.composite(0, 0, 0, rotation=90)
            self.assertEqual(rotated.info['source_dimensions'], (45, 80))
            self.assertEqual(
                v7.aspect_wire_code(rotated.info['source_dimensions']), 7)

            manifest_path = bake / 'modem.json'
            manifest = json.loads(manifest_path.read_text())
            for entry in manifest['folders']:
                entry.pop('source_sizes')
            manifest_path.write_text(json.dumps(manifest))
            legacy_library = ModemLibrary(bake)
            self.assertEqual(
                legacy_library.composite(0, 0, 0).info['source_dimensions'],
                (80, 96))

            manifest['profile'] = 'lean-dct'
            manifest_path.write_text(json.dumps(manifest))
            with self.assertRaises(ValueError):
                ModemLibrary(bake)

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
                 '--modem-frames', '5', '--modem-speed', '1.5'],
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
            decoded, info = v7.decode_pulse_stream(
                model, samples, pilot_timing='tone-seeded',
                frame_boundary='eof')
            self.assertEqual(len(decoded), 5, info)
            self.assertEqual(info['eof_markers_validated'], 5)
            self.assertTrue(all(
                item.diag['pilot_timing']['mode_applied'] == 'tone-seeded'
                for item in decoded))
            self.assertEqual(
                [item.diag['source_index'] for item in decoded[:4]],
                [0, 1, 2, 1])
            self.assertTrue(all(item.diag['aspect_code'] == 3
                                for item in decoded[:4]))

    def test_main_runtime_images_use_fifo_and_v7(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / 'runtime_images'
            source_tree(source)
            wav = root / 'runtime.wav'
            result = subprocess.run(
                [sys.executable, str(REPO / 'main.py'),
                 '--mode', 'modem', '--modem-source', 'images',
                 '--dir', str(source), '--modem-pair', '1,0',
                 '--modem-wav', str(wav), '--modem-frames', '5',
                 '--modem-log-frames'],
                cwd=REPO, text=True, capture_output=True, timeout=30)
            self.assertEqual(result.returncode, 0,
                             result.stdout + result.stderr)
            reports = [json.loads(line) for line in result.stdout.splitlines()
                       if line.startswith('{')]
            self.assertTrue(reports)
            self.assertTrue(all('fifo_hits' in report for report in reports))
            # A miss only says the asynchronous image FIFO had to load the
            # exact requested index in the foreground; RuntimeImageLibrary
            # never substitutes a nearby frame.  Correctness is checked by
            # decoding the source indices below, not by requiring a cache hit.
            self.assertTrue(all(report['source_index_misses'] >= 0
                                for report in reports))

            with wave.open(str(wav), 'rb') as source_wav:
                samples = np.frombuffer(
                    source_wav.readframes(source_wav.getnframes()),
                    dtype='<i2').reshape(-1, 2).astype(float) / 32767
            model = v7.build_model(
                REPO / 'modem_tests/fixtures/v7_reference_face.png',
                .1521 / np.sqrt(1 + 10**(v7.CLOCK_REL_DB / 10)),
                encode_filter='nearest')
            decoded, info = v7.decode_pulse_stream(
                model, samples, pilot_timing='tone-seeded',
                frame_boundary='eof')
            self.assertEqual(len(decoded), 5, info)
            self.assertEqual(info['eof_markers_validated'], 5)
            self.assertTrue(all(
                item.diag['pilot_timing']['mode_applied'] == 'tone-seeded'
                for item in decoded))
            self.assertEqual(
                [item.diag['source_index'] for item in decoded[:4]],
                [0, 2, 1, 1])
            self.assertTrue(all(item.diag['aspect_code'] == 3
                                for item in decoded[:4]))

    def test_wav_rejects_live_midi_clock_without_prompting(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / 'images'
            source_tree(source)
            bake = root / 'images_modem'
            with contextlib.redirect_stdout(io.StringIO()):
                bake_tree(source, bake)
            result = subprocess.run(
                [sys.executable, str(REPO / 'main.py'),
                 '--mode', 'modem', '--modem-dir', str(bake),
                 '--modem-pair', '1,0', '--modem-wav', str(root / 'midi.wav'),
                 '--modem-clock', '0', '--modem-frames', '5'],
                cwd=root, text=True, capture_output=True, timeout=30)
            self.assertEqual(result.returncode, 2, result.stdout + result.stderr)
            self.assertIn('--modem-wav requires the free clock', result.stderr)


if __name__ == '__main__':
    unittest.main()
