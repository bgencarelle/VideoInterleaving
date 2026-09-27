"""Preview image files in the standalone V7 display viewer without audio."""
import argparse
from dataclasses import dataclass
import math
from pathlib import Path
import sys
import time

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from animation_modem import v7


@dataclass(frozen=True)
class PreviewFrame:
    generation: int
    values: object
    shapes: tuple
    aspect: int


def _load_frames(paths):
    frames = []
    for path in paths:
        with Image.open(path) as source:
            image = source.convert('RGB')
        frames.append(PreviewFrame(
            len(frames),
            v7.image_values(image, v7.V7_GRIDS, encode_filter='box'),
            v7.V7_GRIDS,
            v7.aspect_wire_code(image.size)))
    return frames


def parser():
    ap = argparse.ArgumentParser(
        description='Preview source images with the V7 GL display controls.')
    ap.add_argument('images', nargs='+', type=Path,
                    help='image files (for example images_sbs/*.jpg)')
    ap.add_argument('--fps', type=float, default=12.0,
                    help='sequence playback rate (default: 12)')
    ap.add_argument('--loop', action='store_true',
                    help='loop the sequence after its final image')
    ap.add_argument('--fullscreen', action='store_true')
    ap.add_argument('--display-mode', choices=('nearest', 'bilinear'),
                    help='initial viewer mode (default: saved preference)')
    ap.add_argument('--show-info', action='store_true',
                    help='open the diagnostics panel initially')
    ap.add_argument('--profile-ui', action='store_true')
    return ap


def main(argv=None):
    args = parser().parse_args(argv)
    if not math.isfinite(args.fps) or args.fps <= 0:
        raise SystemExit('--fps must be finite and positive')
    paths = [path for path in args.images if path.is_file()]
    if not paths:
        raise SystemExit('no readable image paths were provided')
    frames = _load_frames(paths)
    if not frames:
        raise SystemExit('no images could be loaded')

    started = time.monotonic()
    active = {'index': 0}

    def frame_source():
        elapsed_index = int((time.monotonic()-started)*args.fps)
        if args.loop:
            index = elapsed_index % len(frames)
        else:
            index = min(elapsed_index, len(frames)-1)
        active['index'] = index
        return frames[index]

    def status_source():
        return {'status': 'preview', 'source_index': active['index']+1}

    def diagnostics_source():
        frame = frames[active['index']]
        return {
            'status': ('PREVIEW',),
            'sync': (f'frame {active["index"]+1} / {len(frames)}',),
            'decode': ('source image; no modem decode',),
            'input': (f'{args.fps:g} fps preview · no audio',),
            'signal': (f'aspect code {frame.aspect}',),
        }

    from tools.v7_gl_viewer import run
    run(frame_source, status_source, v7.V7_ASPECT_RATIOS,
        fullscreen=args.fullscreen, show_diagnostics=args.show_info,
        diagnostics_source=diagnostics_source, profile_cpu=args.profile_ui,
        display_mode=args.display_mode)


if __name__ == '__main__':
    main()
