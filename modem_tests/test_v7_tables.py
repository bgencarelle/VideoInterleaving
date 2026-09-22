"""The canonical V7 model is frozen, hash-checked data (animation_modem/v7_model_tables.npz)."""
import hashlib
import unittest
from unittest import mock

import numpy as np
from PIL import Image

from animation_modem import v7

TARGET = .1521/np.sqrt(1 + 10**(v7.CLOCK_REL_DB/10))


class V7FrozenTablesTests(unittest.TestCase):
    def test_file_matches_wire_hash(self):
        digest = hashlib.sha256(v7.MODEL_TABLES.read_bytes()).hexdigest()
        self.assertEqual(digest, v7.MODEL_TABLES_SHA256)

    def test_every_filter_loads_a_complete_model(self):
        for name in v7.ENCODING_FILTERS:
            with self.subTest(filter=name):
                model = v7.load_model(TARGET, name)
                self.assertEqual(model.phase.shape, (v7.F, 65))
                self.assertTrue(np.allclose(np.abs(model.phase), 1))
                self.assertEqual(len(model.lam), 2880)
                self.assertEqual(sorted(model.order.tolist()), list(range(2880)))
                self.assertEqual(model.encoding_type, v7.ENCODING_FILTER_CODES[name])
                self.assertGreater(model.scale, 0)

    def test_canonical_model_needs_no_image_or_derivation(self):
        with mock.patch.object(v7.Image, 'open', side_effect=AssertionError('image opened')), \
                mock.patch.object(v7, 'derive_tables', side_effect=AssertionError('derived')):
            v7.build_model(v7.REFERENCE_FIXTURE, TARGET, 'nearest')
            v7.build_model(None, TARGET, 'lanczos')

    def test_wrong_hash_is_refused(self):
        v7._frozen_tables.cache_clear()
        try:
            with mock.patch.object(v7, 'MODEL_TABLES_SHA256', '0'*64):
                with self.assertRaises(ValueError):
                    v7.load_model(TARGET, 'nearest')
        finally:
            v7._frozen_tables.cache_clear()

    def test_frozen_round_trip_decodes_cleanly(self):
        model = v7.load_model(TARGET, 'nearest')
        with Image.open(v7.REFERENCE_FIXTURE) as source:
            values = v7.image_values(v7.prepare_image(source, encode_filter='nearest'),
                                     model.coder.grids, encode_filter='nearest')
        audio = v7.encode_pulse_stream(model, [values]*4, 1, [6]*4)
        for force_float32 in (False, True):
            results, _ = v7.decode_pulse_stream(
                model, audio, force_float32=force_float32)
            with self.subTest(force_float32=force_float32):
                self.assertGreaterEqual(len(results), 3)
                for result in results:
                    error = np.sqrt(np.mean(
                        (v7.values_from(model, result.coeffs)-values)**2))
                    self.assertLess(error, .07)

if __name__ == '__main__':
    unittest.main()
