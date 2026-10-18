"""Validate the shipped font, licence/source delivery and reproducible builder.

Build-time tests: install fonttools[woff]==4.67.0 and skia-pathops; not an app dependency.
"""
from pathlib import Path
import tempfile
import unittest
import zipfile

try:
    from fontTools.ttLib import TTFont
    from fontTools.pens.boundsPen import BoundsPen
    from fontTools.pens.recordingPen import RecordingPen
    from fontTools.pens.transformPen import TransformPen
except ImportError:
    TTFont = None

from utilities.build_schoolbook_mono import ADVANCE, BLACK_EMBOLDEN, FAMILY, build

ROOT = Path('static/fonts/schoolbook-mono')


@unittest.skipIf(TTFont is None, 'Optional font build dependencies not installed')
class SchoolbookMonoTests(unittest.TestCase):
    def test_all_printable_ascii_has_identical_advances_for_both_weights(self):
        for style in ('Regular', 'Bold'):
            with TTFont(ROOT / f'InterleaveSchoolbookMono-{style}.woff2') as font:
                self.assertEqual(set(font.getBestCmap()), set(range(32, 127)))
                self.assertEqual({m[0] for m in font['hmtx'].metrics.values()}, {ADVANCE})
                self.assertEqual(font['name'].getDebugName(1), FAMILY)
                self.assertEqual(font['OS/2'].fsType, 0)
                self.assertEqual(font['post'].isFixedPitch, 1)
                self.assertTrue(font['CFF '].cff.topDictIndex[0].isFixedPitch)
                self.assertTrue(all(tag not in font for tag in ('kern', 'GPOS', 'GSUB')))
                glyphs = font.getGlyphSet()
                for name in font.getGlyphOrder():
                    bounds = BoundsPen(glyphs)
                    glyphs[name].draw(bounds)
                    self.assertEqual(font['CFF '].cff.topDictIndex[0].CharStrings[name].width, ADVANCE)
                    if bounds.bounds:
                        left, _, right, _ = bounds.bounds
                        self.assertGreaterEqual(left, 0)
                        self.assertLessEqual(right, ADVANCE)
                        self.assertLessEqual(abs(left - (ADVANCE-right)), 1)

    def test_black_is_a_heavier_bold_with_only_bounded_overhang(self):
        with TTFont(ROOT / 'InterleaveSchoolbookMono-Bold.woff2') as bold, \
                TTFont(ROOT / 'InterleaveSchoolbookMono-Black.woff2') as black:
            self.assertEqual(set(black.getBestCmap()), set(range(32, 127)))
            self.assertEqual({m[0] for m in black['hmtx'].metrics.values()}, {ADVANCE})
            self.assertEqual(black['OS/2'].usWeightClass, 900)
            self.assertEqual(black['name'].getDebugName(4), FAMILY + ' Black')
            bold_glyphs, black_glyphs = bold.getGlyphSet(), black.getGlyphSet()
            for codepoint, name in black.getBestCmap().items():
                before, after = BoundsPen(bold_glyphs), BoundsPen(black_glyphs)
                bold_glyphs[bold.getBestCmap()[codepoint]].draw(before)
                black_glyphs[name].draw(after)
                if before.bounds:
                    self.assertIsNotNone(after.bounds, chr(codepoint))
                    bl, bb, br, bt = before.bounds
                    al, ab, ar, at = after.bounds
                    self.assertLessEqual(al, bl)
                    self.assertGreaterEqual(ar, br)
                    self.assertLessEqual(ab, bb)
                    self.assertGreaterEqual(at, bt)
                    self.assertLessEqual(ar - al, (br - bl) + 2 * BLACK_EMBOLDEN + 2, chr(codepoint))

    def test_original_outlines_only_translate_never_scale_or_change_baseline(self):
        for source, style in (('Roman', 'Regular'), ('Bold', 'Bold')):
            with TTFont(ROOT / 'source' / f'C059-{source}.otf') as original, \
                    TTFont(ROOT / f'InterleaveSchoolbookMono-{style}.woff2') as mono:
                before, after = original.getGlyphSet(), mono.getGlyphSet()
                for codepoint, name in mono.getBestCmap().items():
                    src = before[original.getBestCmap()[codepoint]]
                    bounds = BoundsPen(before)
                    src.draw(bounds)
                    offset = 0
                    if bounds.bounds:
                        left, _, right, _ = bounds.bounds
                        offset = round((ADVANCE - (right-left))/2 - left)
                    expected, actual = RecordingPen(), RecordingPen()
                    src.draw(TransformPen(expected, (1, 0, 0, 1, offset, 0)))
                    after[name].draw(actual)
                    self.assertEqual(expected.value, actual.value, (style, chr(codepoint)))

    def test_rebuild_and_downloadable_corresponding_source_are_complete(self):
        with tempfile.TemporaryDirectory(dir='tmp', prefix='mono_font_test_') as folder:
            build(ROOT / 'source', folder)
            for name in ('InterleaveSchoolbookMono-Regular.woff2',
                         'InterleaveSchoolbookMono-Bold.woff2',
                         'InterleaveSchoolbookMono-Black.woff2', 'build.json',
                         'schoolbook-mono-source.zip'):
                self.assertEqual((ROOT / name).read_bytes(), (Path(folder) / name).read_bytes(), name)
        with zipfile.ZipFile(ROOT / 'schoolbook-mono-source.zip') as archive:
            self.assertEqual(archive.read('utilities/build_schoolbook_mono.py'),
                             Path('utilities/build_schoolbook_mono.py').read_bytes())
            for name in ('C059-Roman.otf', 'C059-Bold.otf', 'LICENSE', 'COPYING', 'README.md', 'requirements.txt'):
                self.assertEqual(archive.read('source/' + name), (ROOT / 'source' / name).read_bytes())


if __name__ == '__main__':
    unittest.main()
