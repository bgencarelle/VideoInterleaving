"""--ascii-bio: the biography the palette is built from, made readable."""
import unittest
from unittest import mock

import numpy as np

import ascii_converter as ac
import settings


def bio(mode, **values):
    values.setdefault('SUBJECT_NAME', 'BEN GENCARELLE')
    values.setdefault('BIRTH_TIME', '1978, 11, 17, 7, 11')
    values.setdefault('BIRTH_PLACE', '')
    values.setdefault('ASCII_PALETTE', 'MWB8GRDNHESAVTOLPmevncray97stji1-/., ')
    return mock.patch.multiple(settings, ASCII_BIO=mode, create=True, **values)


def text(indices):
    return [''.join(row) for row in ac.GLYPHS[indices]]


class BioTests(unittest.TestCase):
    def tearDown(self):
        ac.refresh_palette()
        ac._reveal.clear()

    def test_phrase_is_name_then_birth_date(self):
        with bio(None):
            self.assertEqual(ac.bio_phrase(), 'BEN GENCARELLE 11-17-1978')

    def test_birth_place_follows_the_date_when_set(self):
        with bio(None, BIRTH_PLACE='SOMEWHERE NY'):
            self.assertEqual(ac.bio_phrase(), 'BEN GENCARELLE 11-17-1978 SOMEWHERE NY')
        with bio(None, BIRTH_PLACE=''):
            self.assertEqual(ac.bio_phrase(), 'BEN GENCARELLE 11-17-1978')

    def test_no_argument_leaves_everything_as_it_was(self):
        with bio(None):
            ac.refresh_palette()
            self.assertEqual(''.join(ac.CHARS), 'MWB8GRDNHESAVTOLPmevncray97stji1-/., ')
            grid = np.arange(12).reshape(3, 4)
            self.assertIs(ac.apply_bio(grid), grid)

    def test_palette_keeps_only_the_biography_in_density_order(self):
        with bio('palette'):
            ac.refresh_palette()
            self.assertEqual(''.join(ac.CHARS), 'B8GRNEALencra971- ')

    def test_runs_spell_the_phrase_along_the_rows(self):
        with bio('runs'):
            ac.refresh_palette()
            dense = np.zeros((2, 20), np.int64)          # all ink: every cell spells
            self.assertEqual(''.join(text(ac.apply_bio(dense))),
                             ('BEN GENCARELLE 11-17-1978 ' * 2)[:40])

    def test_runs_leave_the_faintest_cells_alone(self):
        with bio('runs'):
            ac.refresh_palette()
            blank = np.full((2, 5), len(ac.CHARS) - 1, np.int64)
            np.testing.assert_array_equal(ac.apply_bio(blank), blank)

    def test_reveal_writes_the_phrase_on_one_line_for_a_while(self):
        with bio('reveal', ASCII_BIO_REVEAL_EVERY=(1, 1), ASCII_BIO_REVEAL_FOR=(3, 3)):
            ac.refresh_palette()
            ac._reveal.clear()
            grid = np.full((10, 40), len(ac.CHARS) - 1, np.int64)
            self.assertIs(ac.apply_bio(grid, now=100.0), grid)     # first one is a while off
            shown = text(ac.apply_bio(grid, now=101.0))
            lines = [line for line in shown if line.strip()]
            self.assertEqual([line.strip() for line in lines], ['BEN GENCARELLE 11-17-1978'])
            self.assertIs(ac.apply_bio(grid, now=104.5), grid)     # and it goes again

    def test_reveal_wraps_on_a_grid_narrower_than_the_phrase(self):
        with bio('reveal', ASCII_BIO_REVEAL_EVERY=(0, 0), ASCII_BIO_REVEAL_FOR=(5, 5)):
            ac.refresh_palette()
            ac._reveal.clear()
            grid = np.full((10, 15), len(ac.CHARS) - 1, np.int64)
            ac.apply_bio(grid, now=0.0)
            joined = ''.join(text(ac.apply_bio(grid, now=0.5)))
            self.assertIn('BEN GENCARELLE 11-17-1978', joined)

    def test_phrase_letters_missing_from_the_palette_are_added_as_glyphs(self):
        with bio(None):
            ac.refresh_palette()
            self.assertIn('C', ''.join(ac.GLYPHS))
            self.assertNotIn('C', ''.join(ac.CHARS))


if __name__ == '__main__':
    unittest.main()
