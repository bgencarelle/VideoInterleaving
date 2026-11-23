"""Compressed RGB samples: independent layers, random access and live GL parity."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
import zlib

import numpy as np
from PIL import Image

from ascii_rgb_layers import (RgbLayerSampler, RgbWebLayers, baked_web_source_size, create_bake_context,
                              close_bake_context, upload_sample)
from baked_layers import LayerFrame, package_output
from utilities.bake_ascii_rgb import bake, pack_sample


class RgbBakeTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(dir='tmp', prefix='rgb_web_test_')
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.source = self.root / 'source'
        rng = np.random.default_rng(329)
        for layer, name in (('face', '0_main'), ('face', '2_other'), ('float', '255_front')):
            dest = self.source / layer / name
            dest.mkdir(parents=True)
            for index in range(3):
                image = rng.integers(0, 256, (32, 24, 4), dtype=np.uint8)
                if layer == 'float':
                    image[:] = (80, 220, 70, 130)  # Repeated layer deduplicates.
                Image.fromarray(image).save(dest / f'{index}.png')

    def build(self):
        return bake(self.source, self.root / 'bake', size=(18, 12))

    def open_library(self, **kwargs):
        return RgbWebLayers(package_output(self.root / 'bake', 'asciiweb'), **kwargs)

    def test_pack_is_bit_exact_and_only_uses_half_when_lossless(self):
        eight_bit = np.random.default_rng(3).integers(0, 256, (2, 3, 4)).astype(np.float32) / np.float32(255)
        for scalar, image in ((1, eight_bit), (2, np.full((2, 3, 4), .5, np.float32)),
                              (4, np.full((2, 3, 4), .1234567, np.float32))):
            block, actual, checksum, _ = pack_sample(image)
            self.assertEqual(actual, scalar)
            raw = zlib.decompress(block)
            self.assertEqual(zlib.crc32(raw), checksum)
            packed = np.frombuffer(raw, np.uint8).reshape(scalar, -1).T.copy().tobytes()
            if scalar == 1:
                restored = np.frombuffer(packed, np.uint8).reshape(image.shape).astype(np.float32) / np.float32(255)
                np.testing.assert_allclose(restored, image, rtol=0, atol=1e-7)
                np.testing.assert_array_equal(np.round(restored * 255), np.round(image * 255))
            else:
                restored = np.frombuffer(packed, dtype=f'<f{scalar}').reshape(image.shape)
                np.testing.assert_array_equal(restored.astype(np.float32), image)

    def test_default_bake_uses_the_new_live_master(self):
        info = bake(self.source, self.root / 'bake')
        self.assertEqual(info['size'], [120, 80])
        with self.open_library() as lib:
            self.assertEqual(lib.size, (120, 80))
            self.assertEqual(lib.load_images(0, 0, 0)[0].shape, (80, 120, 4))

    def test_bake_header_preserves_old_dimensions_and_rejects_invalid_grids(self):
        self.assertIsNone(baked_web_source_size(self.root))
        manifest = self.root / 'rgb_layers.json'
        info = dict(format='video-interleaving-rgb-web-layers', version=1, size=[90,60])
        manifest.write_text(json.dumps(info))
        self.assertEqual(baked_web_source_size(self.root), (90, 60))
        for size in ([0,80], [120,101], [True,60], [90], '90x60'):
            manifest.write_text(json.dumps(dict(info, size=size)))
            with self.assertRaises(ValueError):
                baked_web_source_size(self.root)

    def test_random_access_independent_folders_cache_budget_and_close_ownership(self):
        info = self.build()
        self.assertLess(info['unique_blocks'], 9)
        self.assertLess(info['compressed_bytes'], info['sampled_float32_bytes'])
        with self.open_library(cache_bytes=18 * 12 * 16) as lib:
            self.assertIsInstance(lib.tables['main'][0], np.memmap)
            first = lib.load_images(0, 0, 0)
            other = lib.load_images(0, 1, 0)
            self.assertFalse(np.array_equal(first[0], other[0]))
            np.testing.assert_array_equal(first[1], other[1])
            for i in (2, 0, 1, 2):
                main, front, ms, fs = lib.load_images(i, 0, 0)
                self.assertFalse(ms or fs)
                self.assertFalse(main.flags.writeable)
                self.assertLessEqual(lib.cache_bytes, lib.cache_budget)
            retained = main.copy()
        np.testing.assert_array_equal(main, retained)
        with self.assertRaisesRegex(ValueError, 'closed'):
            lib.read_image(LayerFrame('main', 0, 0))
        with self.open_library(cache_bytes=0) as lib:
            lib.load_images(0, 0, 0)
            self.assertEqual(lib.cache_bytes, 0)

    def test_corruption_truncation_and_path_escape_rejected(self):
        self.build()
        root = package_output(self.root / 'bake', 'asciiweb')
        with self.open_library() as lib:
            row = lib.tables['main'][0][0]
            offset = int(row[0])
        data = root / 'samples.bin'
        with data.open('r+b') as stream:
            stream.seek(offset)
            stream.write(b'broken')
        with self.open_library() as lib:
            with self.assertRaises((ValueError, zlib.error)):
                lib.load_images(0, 0, 0)
        with data.open('r+b') as stream:
            stream.truncate(1)
        with self.assertRaisesRegex(ValueError, 'Truncated'):
            self.open_library()
        manifest = root / 'rgb_layers.json'
        info = json.loads(manifest.read_text())
        info['folders'][0]['path'] = '../../escape'
        manifest.write_text(json.dumps(info))
        with self.assertRaisesRegex(ValueError, 'path'):
            self.open_library()

    def test_failed_bake_is_not_published_and_existing_output_is_not_overwritten(self):
        with patch('utilities.bake_ascii_rgb.pack_sample', side_effect=OSError('interrupted')):
            with self.assertRaises(OSError):
                self.build()
        self.assertFalse((self.root / 'bake').exists())
        self.assertFalse(list(self.root.glob('.rgb-web-bake-*')))
        self.build()
        with self.assertRaises(ValueError):
            self.build()

    def test_rebuild_is_deterministic_and_geometry_mismatch_is_explicit(self):
        self.build()
        bake(self.source, self.root / 'again', size=(18, 12))
        first = self.root / 'bake'
        second = self.root / 'again'
        for path in first.rglob('*'):
            if path.is_file():
                self.assertEqual(path.read_bytes(), (second / path.relative_to(first)).read_bytes())
        import settings
        with self.open_library() as lib:
            with patch.object(settings, 'ASCII_WEB_SOURCE_SIZE', (90, 60), create=True):
                with self.assertRaisesRegex(ValueError, 'differs'):
                    lib.validate_settings()

    def test_decompression_is_size_limited_and_checksum_checked(self):
        self.build()
        root = package_output(self.root / 'bake', 'asciiweb')
        data = root / 'samples.bin'
        index = root / 'face/0_main/index.npy'
        table = np.load(index)
        scalar = int(table[0, 2])
        expected = 18 * 12 * 4 * scalar
        for raw, checksum in ((b'\0' * expected, 1),
                              (b'\0' * (expected * 100), 0)):
            block = zlib.compress(raw)
            offset = data.stat().st_size
            with data.open('ab') as stream:
                stream.write(block)
            table[0] = (offset, len(block), scalar, checksum)
            np.save(index, table, allow_pickle=False)
            with self.open_library() as lib:
                with self.assertRaisesRegex(ValueError, 'Corrupt'):
                    lib.load_images(0, 0, 0)

    def test_backend_history_does_not_reabsorb_itself_or_read_oversized_logs(self):
        import display_manager
        log = self.root / 'backend.json'
        session = {'attempts': ['egl']}
        with patch.object(display_manager, '_BACKEND_USAGE_LOG', log), \
                patch.object(display_manager, '_BACKEND_USAGE_DATA', session):
            for _ in range(3):
                display_manager._log_backend_usage()
            self.assertEqual(session, {'attempts': ['egl']})
            self.assertEqual(json.loads(log.read_text()), {'attempts': ['egl'] * 3})
            oversized = b' ' * (1024 * 1024 + 1)
            log.write_bytes(oversized)
            with patch.object(display_manager.json, 'load', side_effect=AssertionError('large read')):
                display_manager._log_backend_usage()
            self.assertEqual(log.read_bytes(), oversized)

    def test_loaded_samples_match_live_composite_and_ansi_exactly(self):
        self.build()
        import renderer
        import settings
        from ascii_converter import to_ascii
        window = create_bake_context()
        target = window.ctx.simple_framebuffer((18, 12), components=3)
        sampler = RgbLayerSampler(window.ctx, (18, 12), (24, 32), .5)
        textures = []
        try:
            with self.open_library() as lib, patch.object(settings, 'ASCII_TRUECOLOR', True):
                for index in range(3):
                    for folder in range(2):
                        name = ('0_main', '2_other')[folder]
                        main = np.asarray(Image.open(self.source / 'face' / name / f'{index}.png'))
                        front = np.asarray(Image.open(self.source / 'float/255_front' / f'{index}.png'))
                        a, b = renderer.create_texture(main), renderer.create_texture(front)
                        textures.extend((a, b))
                        target.use()
                        renderer.set_transform_parameters(1, 0, 0, (24, 32), 0, 0)
                        renderer.set_viewport_size(18, 12, pixel_aspect=.5)
                        renderer.overlay_images_single_pass(a, b, (21, 37, 62))
                        reference = target.read(components=3, alignment=1)
                        x, y, _, _ = lib.load_images(index, folder, 0)
                        a, b = upload_sample(window.ctx, x), upload_sample(window.ctx, y)
                        textures.extend((a, b))
                        renderer.set_transform_parameters(1, 0, 0, (9, 12), 0, 0)
                        renderer.overlay_images_single_pass(a, b, (21, 37, 62))
                        actual = target.read(components=3, alignment=1)
                        self.assertEqual(reference, actual)
                        ref = np.frombuffer(reference, np.uint8).reshape(12, 18, 3)
                        got = np.frombuffer(actual, np.uint8).reshape(12, 18, 3)
                        self.assertEqual(to_ascii(ref, width=18, height=12, sampled=True),
                                         to_ascii(got, width=18, height=12, sampled=True))
                # SBS samples include the existing <.05 alpha cutoff. Also test
                # letterboxed geometry; sampled transparent margins must not tint.
                rng = np.random.default_rng(56)
                for source_size in ((24, 32), (32, 20)):
                    sampler.close()
                    sampler = RgbLayerSampler(window.ctx, (18, 12), source_size, .5)
                    for _ in range(3):
                        main = rng.integers(0, 256, (20, 64, 3), np.uint8)
                        front = rng.integers(0, 256, (20, 64, 3), np.uint8)
                        front[:, 32:, 0] = rng.choice(np.array([0, 12, 13, 128, 255], np.uint8), (20, 32))
                        a, b = renderer.create_texture(main), renderer.create_texture(front)
                        textures.extend((a, b))
                        target.use()
                        renderer.set_transform_parameters(1, 0, 0, source_size, 0, 0)
                        renderer.set_viewport_size(18, 12, pixel_aspect=.5)
                        renderer.overlay_images_single_pass(a, b, (21, 37, 62), True, True)
                        reference = target.read(components=3, alignment=1)
                        x, y = sampler.sample(main, True), sampler.sample(front, True)
                        a, b = upload_sample(window.ctx, x), upload_sample(window.ctx, y)
                        textures.extend((a, b))
                        target.use()
                        renderer.set_transform_parameters(1, 0, 0, (9, 12), 0, 0)
                        renderer.set_viewport_size(18, 12, pixel_aspect=.5)
                        renderer.overlay_images_single_pass(a, b, (21, 37, 62))
                        self.assertEqual(reference, target.read(components=3, alignment=1))
        finally:
            sampler.close()
            for texture in textures:
                texture.release()
            target.release()
            close_bake_context(window)

    def test_cli_routes_package_without_source_scan(self):
        self.build()
        script = '''
import sys
import settings
settings.WEB_PORT=18490
sys.argv=['main.py','--mode','asciiweb','--bake-dir',sys.argv[1],'--port','18480']
import main, settings
from ascii_rgb_layers import RgbWebLayers
from unittest.mock import patch
with RgbWebLayers(settings.LAYER_BAKE_DIR) as lib:
    assert settings.ASCII_WEB_SOURCE_SIZE == (18,12)
    lib.validate_settings()
    assert lib.frames==3
import make_file_lists
make_file_lists.process_files=lambda: (_ for _ in ()).throw(AssertionError('source scan'))
import image_display, ascii_web_server, web_service
image_display.run_display=lambda *a: None
ascii_web_server.start_server=lambda: None
web_service.start_server=lambda **kw: None
main.main()
'''
        result = subprocess.run([sys.executable, '-c', script, str((self.root / 'bake').resolve())],
                                text=True, capture_output=True, timeout=30)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


if __name__ == '__main__':
    unittest.main()
