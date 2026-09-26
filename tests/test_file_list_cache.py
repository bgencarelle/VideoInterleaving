"""The startup list cache restores only when source metadata is unchanged."""
import csv
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

import make_file_lists
import settings


class FileListCacheTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        root = Path(self.temp.name)
        self.main = root / "face"
        self.float = root / "float"
        self.main_folder = self.main / "0_portrait"
        self.float_folder = self.float / "255_overlay"
        self.main_folder.mkdir(parents=True)
        self.float_folder.mkdir(parents=True)
        self.processed = root / "processed"
        self.generated = root / "generated"
        self.settings_patch = patch.multiple(
            settings, MAIN_FOLDER_PATH=str(self.main),
            FLOAT_FOLDER_PATH=str(self.float))
        self.module_patch = patch.multiple(
            make_file_lists, PROCESSED_DIR_NAME=str(self.processed),
            GENERATED_DIR_NAME=str(self.generated))
        self.settings_patch.start()
        self.module_patch.start()
        self.addCleanup(self.module_patch.stop)
        self.addCleanup(self.settings_patch.stop)
        self.addCleanup(self.temp.cleanup)
        self._write_frame(self.main_folder, "2.npy")
        self._write_frame(self.main_folder, "10.npy")
        self._write_frame(self.float_folder, "2.npy")
        self._write_frame(self.float_folder, "10.npy")

    @staticmethod
    def _write_frame(folder, name):
        with open(folder / name, "wb") as f:
            np.save(f, np.zeros((6, 8, 3), dtype=np.uint8))

    def test_unchanged_sources_restore_lists_and_changes_rescan(self):
        make_file_lists.process_files()
        cache = Path(str(self.generated.resolve()) +
                     ".source-manifest.json.gz")
        self.assertTrue(cache.is_file())
        before = {p.name: p.read_bytes() for p in self.generated.iterdir()}

        # main.py removes these output directories at startup; the adjacent
        # source manifest and snapshot allow regeneration without image scans.
        shutil.rmtree(self.processed)
        shutil.rmtree(self.generated)
        with patch.object(make_file_lists, "scan_directory_recursive",
                          side_effect=AssertionError("unexpected rescan")):
            make_file_lists.process_files()
        after = {p.name: p.read_bytes() for p in self.generated.iterdir()}
        self.assertEqual(after, before)

        self._write_frame(self.main_folder, "20.npy")
        self._write_frame(self.float_folder, "20.npy")
        original_scan = make_file_lists.scan_directory_recursive
        with patch.object(make_file_lists, "scan_directory_recursive",
                          wraps=original_scan) as scan:
            make_file_lists.process_files()
        self.assertEqual(scan.call_count, 2)

        main_list = next(self.generated.glob("main_folder_*_list.csv"))
        with main_list.open(newline="") as f:
            rows = list(csv.reader(f))
        self.assertEqual(len(rows), 3)
        self.assertTrue(rows[-1][1].endswith("20.npy"))


if __name__ == "__main__":
    unittest.main()
