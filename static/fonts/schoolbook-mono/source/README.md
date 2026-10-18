# Interleave Schoolbook Mono — corresponding source

Modified 2026-10-18 from URW C059 Roman/Bold. This is **not** the proprietary
Monotype/Bitstream Century Schoolbook font or an official URW release.

Upstream: https://github.com/ArtifexSoftware/urw-base35-fonts
Pinned commit: `3c0ba3b5687632dfc66526544a4e811fe0ec0cd9`
Files: `fonts/C059-Roman.otf`, `fonts/C059-Bold.otf`, `LICENSE`, `COPYING`.
Original copyright notices remain in each font. Source hashes are recorded in
`build.json` and enforced by the build script.

The font and this modification tool are distributed under AGPL version 3;
the upstream font additionally carries the PostScript/PDF exception in LICENSE.
COPYING contains the full licence. The font licence applies to these fonts;
it is not a declaration of a new licence for the surrounding application.

Changes: new family/PostScript names, printable ASCII subset (U+0020–U+007E),
981-unit advances for every glyph including space, horizontally centred ink,
fixed-pitch metadata, no kerning or substitution tables, WOFF2 packaging.
Original glyph shapes, height, baseline and stroke weights are retained:
**no outline scaling/compression** is needed for this character set. Regular
and Bold have the same advance. Other browser weights select/synthesise these
faces; this is not a variable font with nine distinct weights.

Rebuild from the repository root (fontTools is a build-only dependency):

```bash
.venv/bin/python -m pip install -r static/fonts/schoolbook-mono/source/requirements.txt
.venv/bin/python utilities/build_schoolbook_mono.py
```

Or, after extracting the source ZIP, run:

```bash
python -m pip install -r source/requirements.txt
python utilities/build_schoolbook_mono.py --source source --output rebuilt
```

The distributed `schoolbook-mono-source.zip` includes the original OTFs,
licences, this README, pinned build requirements, modification script and build manifest. Browsers can
download it directly from the same static directory as the fonts.
