"""Build the ASCII-only Interleave Schoolbook Mono webfont from free URW C059.

SPDX-License-Identifier: AGPL-3.0-only
Build dependency only: fonttools[woff]==4.67.0. No runtime font compilation.
"""
import argparse
import hashlib
import json
from pathlib import Path
import zipfile


ROOT = Path(__file__).resolve().parents[1]
FAMILY = 'Interleave Schoolbook Mono'
ADVANCE = 981  # C059's W advance in BOTH Roman and Bold; outlines fit unchanged.
BLACK_EMBOLDEN = 25  # Font units added per side for the heavier 900 face.
UPSTREAM = 'https://github.com/ArtifexSoftware/urw-base35-fonts'
COMMIT = '3c0ba3b5687632dfc66526544a4e811fe0ec0cd9'
HASHES = {
    'C059-Roman.otf': 'e00cc7b88f0cf25ae43f0e48de39d0043fe042b55719a3ac32669ab3369542ec',
    'C059-Bold.otf': '75dea10068264324522e7b96612083582eee0158eb728fa9c219279890f33e26',
    'LICENSE': '772b1b47eead8722156abaed483154e1dd1137e7035290550cea6444192a2e3f',
    'COPYING': '57c8ff33c9c0cfc3ef00e650a1cc910d7ee479a8bc509f6c9209a7c2a11399d6',
}


def emboldened(glyph, delta):
    """Union the outline with a round stroke of radius delta (font units)."""
    import pathops
    fill = pathops.Path()
    glyph.draw(fill.getPen())
    stroke = pathops.Path()
    glyph.draw(stroke.getPen())
    stroke.stroke(2 * delta, pathops.LineCap.ROUND_CAP, pathops.LineJoin.ROUND_JOIN, 4)
    # Skia's round joins are conics; CFF cannot store them, so approximate with quads.
    stroke.convertConicsToQuads(0.1)
    return pathops.op(fill, stroke, pathops.PathOp.UNION)


def build_face(source, destination, style, embolden=0):
    from fontTools import subset
    from fontTools.ttLib import TTFont
    from fontTools.pens.boundsPen import BoundsPen
    from fontTools.pens.t2CharStringPen import T2CharStringPen
    from fontTools.pens.transformPen import TransformPen

    font = TTFont(source, recalcTimestamp=False)
    if font['head'].unitsPerEm != 1000:
        raise ValueError('Expected C059 with 1000 units per em')
    # The animation palette is printable ASCII. Subsetting also prevents marks,
    # alternate substitutions and zero-width controls defeating the fixed grid.
    for tag in ('GPOS', 'GSUB', 'kern', 'PCLT'):
        if tag in font:
            del font[tag]
    options = subset.Options()
    options.name_IDs = ['*']
    options.name_legacy = True
    options.name_languages = ['*']
    options.desubroutinize = True
    sub = subset.Subsetter(options=options)
    sub.populate(unicodes=range(32, 127))
    sub.subset(font)
    glyphs = font.getGlyphSet()
    top = font['CFF '].cff.topDictIndex[0]
    records = {}
    for name in font.getGlyphOrder():
        shape = emboldened(glyphs[name], embolden) if embolden else glyphs[name]
        bounds = BoundsPen(glyphs)
        shape.draw(bounds)
        x = 0
        if bounds.bounds is not None:
            left, _, right, _ = bounds.bounds
            # Emboldening may overhang the advance by a few units; the grid clips per row.
            if right - left > ADVANCE and not embolden:
                raise ValueError(f'Glyph {name} would need compression: {right-left}')
            x = round((ADVANCE - (right-left))/2 - left)
        pen = T2CharStringPen(ADVANCE, glyphs)
        shape.draw(TransformPen(pen, (1, 0, 0, 1, x, 0)))
        top.CharStrings[name] = pen.getCharString(private=top.Private,
                                                globalSubrs=font['CFF '].cff.GlobalSubrs,
                                                optimize=False)
        lsb = round(bounds.bounds[0] + x) if bounds.bounds is not None else 0
        font['hmtx'][name] = (ADVANCE, lsb)
        records[name] = dict(translation=x, original_bounds=bounds.bounds)
    # Type 2 charstrings encode a width DELTA relative to nominalWidthX.
    # We emit ADVANCE explicitly, so zero the inherited proportional nominal.
    # hmtx alone is insufficient: browser sanitizers also read CFF widths.
    top.Private.nominalWidthX = 0
    top.Private.defaultWidthX = ADVANCE
    postscript = 'InterleaveSchoolbookMono-' + style
    font['CFF '].cff.fontNames[0] = postscript
    top.FamilyName, top.FullName = FAMILY, FAMILY + ' ' + style
    top.version = '1.000'
    top.isFixedPitch = True
    top.Notice = ('URW C059 derivative, modified 2026-10-18: ASCII subset, centred '
                  'fixed advances; no outline scaling; renamed. AGPL-3.0 with '
                  'upstream PostScript/PDF font exception. See accompanying source.')
    keep = {0, 7, 8, 9, 11, 12}  # Preserve original copyright/attribution.
    font['name'].names = [n for n in font['name'].names if n.nameID in keep]
    names = {1: FAMILY, 2: style, 3: postscript + '-1.000',
             4: FAMILY + ' ' + style, 5: 'Version 1.000', 6: postscript,
             13: 'AGPL-3.0 with upstream PostScript/PDF font exception. '
                 'Modified 2026-10-18. Corresponding source: schoolbook-mono-source.zip.',
             14: 'https://www.gnu.org/licenses/agpl-3.0.html', 16: FAMILY, 17: style}
    for key, value in names.items():
        font['name'].setName(value, key, 3, 1, 0x409)
        font['name'].setName(value, key, 1, 0, 0)
    font['post'].isFixedPitch = 1
    font['OS/2'].xAvgCharWidth = ADVANCE
    font['OS/2'].fsType = 0  # Derivative is freely redistributable under AGPL.
    font['OS/2'].panose.bProportion = 9
    font['OS/2'].usWeightClass = {'Regular': 400, 'Bold': 700, 'Black': 900}[style]
    font['hhea'].advanceWidthMax = ADVANCE
    font['head'].fontRevision = 1.0
    # Stable metadata: generated bytes must not depend on wall-clock build time.
    font['head'].created = font['head'].modified = 3875126400  # 2026-10-18 UTC
    font.flavor = 'woff2'
    font.save(destination)
    font.close()
    return records


def build(source, output):
    source, output = Path(source), Path(output)
    for name, digest in HASHES.items():
        if hashlib.sha256((source / name).read_bytes()).hexdigest() != digest:
            raise ValueError(f'Unexpected upstream source/license: {name}')
    output.mkdir(parents=True, exist_ok=True)
    faces = {}
    # Black is Bold with a round outline offset: a real heavier face, so the
    # browser needs no per-glyph stroke. Weight 900 selects it.
    for original, style, embolden in (('Roman', 'Regular', 0), ('Bold', 'Bold', 0),
                                      ('Bold', 'Black', BLACK_EMBOLDEN)):
        name = 'InterleaveSchoolbookMono-' + style + '.woff2'
        records = build_face(source / f'C059-{original}.otf', output / name, style, embolden)
        faces[style] = dict(file=name, sha256=hashlib.sha256((output / name).read_bytes()).hexdigest(),
                            embolden_units=embolden, glyphs=records)
    manifest = dict(family=FAMILY, version=1, upstream=UPSTREAM, upstream_commit=COMMIT,
                    upstream_sha256=HASHES, units_per_em=1000, advance=ADVANCE,
                    characters='U+0020-U+007E', outline_scaling=False, faces=faces)
    (output / 'build.json').write_text(json.dumps(manifest, indent=2) + '\n')
    # Serve the source alongside the webfonts, not only from a private checkout.
    # The original OTFs are the upstream form consumed by this modification tool.
    with zipfile.ZipFile(output / 'schoolbook-mono-source.zip', 'w', zipfile.ZIP_DEFLATED) as archive:
        files = {f'source/{name}': source / name for name in (*HASHES, 'README.md', 'requirements.txt')}
        files['utilities/build_schoolbook_mono.py'] = Path(__file__)
        files['build.json'] = output / 'build.json'
        for name, path in sorted(files.items()):
            info = zipfile.ZipInfo(name, date_time=(2026, 10, 18, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            archive.writestr(info, path.read_bytes())
    return manifest


def main():
    default = ROOT / 'static/fonts/schoolbook-mono'
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=default / 'source')
    parser.add_argument('--output', type=Path, default=default)
    args = parser.parse_args()
    try:
        result = build(args.source, args.output)
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    print(f'Built {result["family"]}: Regular/Bold/Black, {ADVANCE}/1000 em fixed advance; no glyph scaling')


if __name__ == '__main__':
    main()
