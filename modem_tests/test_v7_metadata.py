import unittest

from animation_modem import v7


class V7MetadataTests(unittest.TestCase):
    def test_source_index_round_trip_uses_one_based_wire_field(self):
        raw = v7.metadata_word(3, 1, 4, source_index=0x1234)
        self.assertEqual(len(raw), 5)
        self.assertEqual(v7.parse_metadata_word(raw),
                         v7.Metadata(3, 1, 4, 0x1234, 0))
        self.assertEqual(len(v7.metadata_symbols(3, 1, 0, 0x1234)), 20)

    def test_screen_bit_rides_in_the_old_encode_filter_field(self):
        for code in range(8):
            for screen in (0, v7.ASPECT_SCREEN):
                for model in (0, 1):
                    meta = v7.parse_metadata_word(v7.metadata_word(
                        code | screen, model, 2, source_index=77))
                    with self.subTest(code=code, screen=screen, model=model):
                        self.assertEqual(meta.aspect_code, code)
                        self.assertEqual(meta.screen_aspect, bool(screen))
                        self.assertEqual(meta.encoding_type, model)
                        self.assertEqual(meta.mask, 0)
        # Picture-aspect packets keep the old byte layout bit for bit.
        self.assertEqual(v7.metadata_word(3, 1, 4, 9)[0],
                         (3 << 5) | (1 << 3) | 4)

    def test_lanczos_and_bicubic_are_not_wire_models(self):
        for encoding in (2, 3):
            with self.assertRaises(ValueError):
                v7.metadata_word(0, encoding)

    def test_zero_wire_index_is_invalid(self):
        raw = bytes([0, 0, 0, 0, 0])
        self.assertIsNone(v7.parse_metadata_word(raw))

    def test_index_range_is_checked(self):
        with self.assertRaises(ValueError):
            v7.metadata_word(0, source_index=-1)
        with self.assertRaises(ValueError):
            v7.metadata_word(0, source_index=0xffff)


if __name__ == '__main__':
    unittest.main()
