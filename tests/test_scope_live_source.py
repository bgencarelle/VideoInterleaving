"""Runtime image thumbnails used by the optional live scope source."""
from pathlib import Path
import tempfile
import time
import threading
import unittest

import numpy as np
from PIL import Image

from scope_bake import calibrate, composite_luma
from scope_image_source import RuntimeScopeImageSource


class RuntimeScopeImageSourceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        root = Path(self.temp.name)
        self.main_path = root / "main.png"
        self.float_path = root / "float.png"
        main = np.zeros((8, 16, 4), dtype=np.uint8)
        main[..., :3] = 255
        main[..., 3] = 255
        main[:, :8, 3] = 0
        front = np.zeros((8, 16, 4), dtype=np.uint8)
        Image.fromarray(main, "RGBA").save(self.main_path)
        Image.fromarray(front, "RGBA").save(self.float_path)
        self.source = RuntimeScopeImageSource(
            [[str(self.main_path)]], [[str(self.float_path)]], width=32)
        self.addCleanup(self.source.close)

    def test_live_luminance_uses_lazy_rgba_thumbnails(self):
        main = self.source.main_libs[0]
        front = self.source.float_libs[0]
        self.assertEqual(main.thumbs.shape, (1, 16, 32, 2))
        lum = composite_luma(main, 0, front, 0)
        self.assertEqual(lum.shape, (16, 32))
        self.assertGreater(float(lum.max()), 0.95)
        self.assertEqual(float(lum.min()), 0.0)

    def test_live_libraries_work_with_fixed_grid_calibration(self):
        result = calibrate(self.source.main_libs, self.source.float_libs,
                           1600, rows=12)
        self.assertGreaterEqual(result["grid_rows"], 12)
        self.assertLessEqual(result["grid_rows"], 16)
        self.assertGreaterEqual(result["grid_cols"], 8)
        self.assertLessEqual(result["grid_cols"], 32)

    def test_prefetch_decodes_the_next_index_without_changing_layer_pair(self):
        self.source.prefetch(0, 0, 0, lookahead=0)
        pair = self._wait_for_pair(self.source, 0, 0, 0)
        first, second = pair.main, pair.floating
        self.assertFalse(first.flags.writeable)
        self.assertFalse(second.flags.writeable)
        self.assertLessEqual(pair.requested_at_ns,
                             pair.decode_started_at_ns)
        self.assertLessEqual(pair.decode_started_at_ns, pair.ready_at_ns)

    @staticmethod
    def _wait_for_pair(source, index, main_folder, float_folder):
        deadline = time.monotonic() + 2.0
        while time.monotonic() < deadline:
            pair = source.ready_pair(index, main_folder, float_folder)
            if pair is not None:
                return pair
            time.sleep(0.001)
        raise AssertionError("runtime thumbnail pair did not become ready")

    def test_pair_readiness_is_nonblocking_and_never_exposes_one_layer(self):
        source = RuntimeScopeImageSource(
            [[str(self.main_path)]], [[str(self.float_path)]], width=32,
            workers=1, max_pending_pairs=1)
        self.addCleanup(source.close)
        entered_float_decode = threading.Event()
        release_float_decode = threading.Event()
        decode = source._decode

        def delayed_decode(path):
            if path == str(self.float_path):
                entered_float_decode.set()
                if not release_float_decode.wait(2.0):
                    raise TimeoutError("test did not release float decode")
            return decode(path)

        source._decode = delayed_decode
        source.prefetch(0, 0, 0, lookahead=0)
        self.assertTrue(entered_float_decode.wait(1.0))
        started = time.perf_counter()
        self.assertIsNone(source.ready_pair(0, 0, 0))
        self.assertLess(time.perf_counter() - started, 0.05)
        self.assertEqual(source.snapshot()["cache_pairs"], 0)
        release_float_decode.set()
        pair = self._wait_for_pair(source, 0, 0, 0)
        self.assertIsNotNone(pair.main)
        self.assertIsNotNone(pair.floating)

    def test_pending_pair_work_is_bounded_and_current_pair_supersedes_lookahead(self):
        source = RuntimeScopeImageSource(
            [[str(self.main_path)] for _ in range(5)],
            [[str(self.float_path)] for _ in range(5)], width=32,
            workers=1, max_pending_pairs=2)
        self.addCleanup(source.close)
        entered_decode = threading.Event()
        release_decode = threading.Event()
        decode = source._decode

        def blocked_decode(path):
            entered_decode.set()
            if not release_decode.wait(2.0):
                raise TimeoutError("test did not release decode")
            return decode(path)

        source._decode = blocked_decode
        source.prefetch(0, 0, 0, direction=1, lookahead=2)
        self.assertTrue(entered_decode.wait(1.0))
        self.assertLessEqual(source.snapshot()["pending_pairs"], 2)
        source.prefetch(3, 0, 0, direction=-1, lookahead=2)
        snapshot = source.snapshot()
        self.assertEqual(snapshot["pending_pairs"], 2)
        self.assertGreaterEqual(snapshot["prefetch_evictions"], 1)
        self.assertIn((3, 0, 0), source._pending_pairs)
        self.assertNotIn((1, 0, 0), source._pending_pairs)
        release_decode.set()

    def test_directional_lookahead_reflects_at_pingpong_endpoints(self):
        lookahead = RuntimeScopeImageSource._lookahead
        self.assertEqual(lookahead(4, 1, 5, 2), (3, 2))
        self.assertEqual(lookahead(0, -1, 5, 2), (1, 2))
        self.assertEqual(lookahead(3, -1, 5, 2), (2, 1))
        self.assertEqual(lookahead(4, 1, 5, 2, pingpong=False), (0, 1))

    def test_composite_uses_the_ready_pair_without_calling_blocking_thumb(self):
        pair = self._wait_for_pair(self.source, 0, 0, 0)
        main = self.source.main_libs[0]
        floating = self.source.float_libs[0]

        def fail_if_called(_index):
            raise AssertionError("hot-path thumb() must not block")

        main.thumb = fail_if_called
        floating.thumb = fail_if_called
        lum = composite_luma(
            main, 0, floating, 0,
            thumbnails=(pair.main, pair.floating))
        self.assertEqual(lum.shape, (16, 32))
        self.assertGreater(float(lum.max()), 0.95)

    def test_failed_layer_marks_the_pair_unavailable(self):
        source = RuntimeScopeImageSource(
            [[str(self.main_path)]], [[str(self.float_path)]], width=32,
            workers=1, max_pending_pairs=1)
        self.addCleanup(source.close)
        decode = source._decode

        def fail_float(path):
            if path == str(self.float_path):
                raise OSError("missing float")
            return decode(path)

        source._decode = fail_float
        source.prefetch(0, 0, 0, lookahead=0)
        deadline = time.monotonic() + 2.0
        while time.monotonic() < deadline:
            if source.snapshot()["decode_failures"]:
                break
            time.sleep(0.001)
        self.assertIsNone(source.ready_pair(0, 0, 0))
        self.assertEqual(source.snapshot()["decode_failures"], 1)
        self.assertEqual(source.snapshot()["cache_pairs"], 0)
        source._decode = decode
        source._retry_failed_after_ns = 0
        self.assertIsNone(source.ready_pair(0, 0, 0))
        recovered = self._wait_for_pair(source, 0, 0, 0)
        self.assertIsNotNone(recovered.main)
        self.assertIsNotNone(recovered.floating)

    def test_ready_pair_never_returns_a_different_selected_index(self):
        source = RuntimeScopeImageSource(
            [[str(self.main_path)] for _ in range(2)],
            [[str(self.float_path)] for _ in range(2)], width=32)
        self.addCleanup(source.close)
        first = self._wait_for_pair(source, 0, 0, 0)
        source.prefetch(1, 0, 0, lookahead=0)
        self.assertIsNone(source.ready_pair(1, 0, 0))
        second = self._wait_for_pair(source, 1, 0, 0)
        self.assertNotEqual(first.requested_at_ns, second.requested_at_ns)

    def test_pair_cache_keeps_the_existing_thumbnail_memory_bound(self):
        source = RuntimeScopeImageSource(
            [[str(self.main_path)] for _ in range(5)],
            [[str(self.float_path)] for _ in range(5)], width=32,
            cache_size=4)
        self.addCleanup(source.close)
        for index in range(5):
            self._wait_for_pair(source, index, 0, 0)
        snapshot = source.snapshot()
        self.assertEqual(snapshot["cache_pair_limit"], 2)
        self.assertLessEqual(snapshot["cache_pairs"], 2)

    def test_live_source_splits_side_by_side_jpeg_luminance_and_matte(self):
        path = Path(self.temp.name) / "side_by_side.jpg"
        pixels = np.empty((8, 32, 3), dtype=np.uint8)
        pixels[:, :16] = 255
        pixels[:, 16:] = 128
        Image.fromarray(pixels, "RGB").save(path, quality=100, subsampling=0)
        source = RuntimeScopeImageSource([[str(path)]], [[str(path)]], width=32)
        try:
            thumb = source.main_libs[0].thumb(0)
            self.assertEqual(thumb.shape, (16, 32, 2))
            self.assertGreater(float(thumb[..., 0].mean()), 0.98 * 255)
            self.assertAlmostEqual(float(thumb[..., 1].mean()), 128, delta=3)
        finally:
            source.close()


if __name__ == "__main__":
    unittest.main()
