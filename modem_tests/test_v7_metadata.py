import unittest

from animation_modem import v7


class V7MetadataTests(unittest.TestCase):
    def test_source_index_round_trip_uses_one_based_wire_field(self):
        raw = v7.metadata_word(3, 2, 4, source_index=0x1234)
        self.assertEqual(len(raw), 5)
        self.assertEqual(v7.parse_metadata_word(raw),
                         v7.Metadata(3, 2, 4, 0x1234, 0))
        self.assertEqual(len(v7.metadata_symbols(3, 2, 0, 0x1234)), 20)

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
