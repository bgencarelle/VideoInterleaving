"""Pinned text tables load from a Windows checkout (CRLF line endings)."""
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT/'test_modem_v7'))

import live_fold  # noqa: E402


class TableLineEndingTests(unittest.TestCase):
    def test_receiver_fold_table_loads_with_crlf_line_endings(self):
        with tempfile.TemporaryDirectory() as tmp:
            table = Path(tmp)/'fold_table_500.json'
            shutil.copy(ROOT/'test_modem_v7'/'fold_table_500.json', table)
            # What git core.autocrlf=true gives a Windows checkout.
            table.write_bytes(table.read_bytes().replace(b'\n', b'\r\n'))
            fold = live_fold.LiveFold(500, table_dir=tmp)
        self.assertEqual(fold.digest, live_fold.TABLE_SHA256[500][:12])

    def test_a_changed_table_is_still_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            table = Path(tmp)/'fold_table_500.json'
            shutil.copy(ROOT/'test_modem_v7'/'fold_table_500.json', table)
            table.write_bytes(table.read_bytes().replace(b'"M"', b'"M" ', 1))
            with self.assertRaisesRegex(ValueError, 'does not match'):
                live_fold.LiveFold(500, table_dir=tmp)

    def test_tables_are_never_converted_by_git(self):
        attributes = (ROOT/'.gitattributes').read_text()
        self.assertIn('*.json -text', attributes)


if __name__ == '__main__':
    unittest.main()
