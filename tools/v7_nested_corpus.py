"""Write a training folder for tools/v7_nested_build.py.

This is a video project: tables are built for ordinary moving pictures.  The
fixture movies and the line and moire patterns are for testing resolution and
for calibration, not for training, so by default only photographs go in
(``--kinds screens patterns`` adds the others, for experiments).  The bundled
photographs are a stand-in until real footage is supplied.

    .venv/bin/python tools/v7_nested_corpus.py --out tmp/nested-corpus

Three kinds of picture go in, nothing is downloaded:

* photographs bundled with scikit-image, several windows of each at random
  scales and places, half mirrored (astronaut, chelsea and coffee are left
  out so they stay held-out pictures for scoring);
* frames of the fixture movies in modem_tests/fixtures (the pixel-motion
  line and colour screens at four aspects, and the robot count screen);
* the line, aliasing and moire patterns of tools/v7_test_patterns.py.

Every picture is brought to the same contrast first (``--contrast``, the
standard deviation of its luma as a share of full range).  The tables are
built from the mean power of each coefficient over the folder, and a
full-contrast line pattern has tens of times the power of a photograph: left
as they are, the patterns alone decide the tables and photographs suffer.
Level is handled per packet on the wire, so only the shape of each picture's
spectrum should count here.

The default mix (photographs weighted eight to one, patterns at half the
contrast) was chosen by scoring trial tables on held-out photographs,
held-out movie frames and the patterns (docs/V7_NESTED_FOLD_ISSUES.md): it
keeps photographs within a few percent of tables built from photographs
alone and carries about half again as much of the line patterns.
"""
import argparse
import subprocess
import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
PHOTOS = ('rocket', 'hubble_deep_field', 'immunohistochemistry', 'retina',
          'camera', 'moon', 'coins', 'clock')
MOVIES = ('v7_pixel_motion_16x9', 'v7_pixel_motion_3x4', 'v7_pixel_motion_4x3',
          'v7_pixel_motion_5x6', 'v7_robot_count_sync_test')


def same_contrast(picture, contrast):
    """``picture`` (uint8 RGB) scaled about its mean so its luma's standard
    deviation is ``contrast`` of full range (clipped to range)."""
    data = np.asarray(picture, np.float64)
    luma = data @ np.array([.299, .587, .114])
    spread = float(luma.std())
    if spread < 1e-6:
        return np.asarray(picture, np.uint8)
    mean = data.mean(axis=(0, 1), keepdims=True)
    scaled = mean+(data-mean)*(contrast*255.0/spread)
    # Keep the mean inside the range the scaled picture needs.
    scaled += 127.5-scaled.mean()
    return np.uint8(np.clip(scaled, 0, 255)+.5)


def photographs(rng):
    from skimage import data
    pictures = {name: np.asarray(getattr(data, name)()) for name in PHOTOS}
    pictures['motorcycle'] = np.asarray(data.stereo_motorcycle()[0])
    for name, picture in pictures.items():
        if picture.ndim == 2:
            picture = np.repeat(picture[..., None], 3, -1)
        picture = picture[..., :3].astype(np.uint8)
        height, width = picture.shape[:2]
        for index in range(6):
            scale = rng.uniform(.55, 1.0) if index else 1.0
            rows, cols = int(height*scale), int(width*scale)
            top = int(rng.integers(0, height-rows+1))
            left = int(rng.integers(0, width-cols+1))
            window = picture[top:top+rows, left:left+cols]
            yield f'photo_{name}_{index}', window[:, ::-1] if index % 2 else window


def movie_frames(work, every, count):
    for movie in MOVIES:
        made = subprocess.run(
            ['ffmpeg', '-v', 'error', '-y', '-i',
             str(ROOT/'modem_tests/fixtures'/f'{movie}.mp4'),
             '-vf', f"select='not(mod(n\\,{every}))'", '-vsync', '0',
             '-frames:v', str(count), str(work/f'{movie}_%02d.png')])
        if made.returncode:
            raise SystemExit('ffmpeg is needed to read the fixture movies')
        for path in sorted(work.glob(f'{movie}_*.png')):
            yield f'screen_{path.stem}', np.asarray(Image.open(path).convert('RGB'))


def patterns(work, layouts):
    for layout in layouts:
        folder = work/f"patterns-{layout.replace(':', 'x')}"
        subprocess.run([sys.executable, str(ROOT/'tools/v7_test_patterns.py'),
                        '--out', str(folder), '--layout', layout],
                       check=True, stdout=subprocess.DEVNULL)
        for path in sorted(folder.glob('[0-9]*.png')):
            yield (f"pattern_{layout.replace(':', 'x')}_{path.stem}",
                   np.asarray(Image.open(path).convert('RGB')))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--out', default='tmp/nested-corpus')
    parser.add_argument('--seed', type=int, default=2026)
    parser.add_argument('--contrast', type=float, default=.2,
                        help='luma standard deviation every picture is brought to, as a '
                             'share of full range (0: leave pictures as they are)')
    parser.add_argument('--pattern-contrast', type=float, default=.1,
                        help='the same for the line patterns (default: half the others)')
    parser.add_argument('--photo-weight', type=int, default=8,
                        help='copies written of every photograph window: its weight in '
                             'the tables against one screen or pattern')
    parser.add_argument('--kinds', nargs='+', default=['photos'],
                        choices=('photos', 'screens', 'patterns'))
    parser.add_argument('--pattern-layouts', nargs='+', default=['3:4'])
    args = parser.parse_args(argv)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for old in out.glob('*.png'):
        old.unlink()
    work = out/'_work'
    work.mkdir(exist_ok=True)
    sources = []
    if 'photos' in args.kinds:
        sources.append(photographs(np.random.default_rng(args.seed)))
    if 'screens' in args.kinds:
        sources.append(movie_frames(work, 47, 6))
    if 'patterns' in args.kinds:
        sources.append(patterns(work, args.pattern_layouts))
    counts = {}
    for source in sources:
        for name, picture in source:
            kind = name.split('_')[0]
            contrast = args.pattern_contrast if kind == 'pattern' else args.contrast
            if contrast > 0:
                picture = same_contrast(picture, contrast)
            picture = Image.fromarray(np.ascontiguousarray(picture))
            for copy in range(args.photo_weight if kind == 'photo' else 1):
                picture.save(out/f'{name}_{copy}.png')
                counts[kind] = counts.get(kind, 0)+1
    print(f'{sum(counts.values())} pictures in {out}: {counts}')


if __name__ == '__main__':
    main()
