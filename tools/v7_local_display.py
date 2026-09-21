"""Run the V7 decoder with the main-branch local ModernGL display.

This is an integration runner only.  The imported display modules remain the
main-branch implementations; V7 supplies newest decoded RGB frames through
``tools.v7_live.FRAME_BUFFER``.
"""
import argparse
from pathlib import Path
import sys
import threading

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import display_manager
import renderer
import settings
from tools import v7_live


def parser():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--device', required=True,
                    help='explicit sounddevice input, e.g. BlackHole 2ch')
    ap.add_argument('--fixture', default=str(v7_live.DEFAULT_FIXTURE))
    ap.add_argument('--fullscreen', action='store_true')
    ap.add_argument('--diagnostics', action='store_true')
    ap.add_argument('--mono-compatible', action='store_true')
    return ap


def run(args):
    # The local display path must not enter the main display's server/ascii
    # branches.  display_init still owns all platform/context decisions.
    settings.SERVER_MODE = False
    settings.ASCII_MODE = False
    state = display_manager.DisplayState(image_size=(80, 96))
    state.fullscreen = bool(args.fullscreen)
    window = display_manager.display_init(state)

    receiver_args = v7_live.parser().parse_args([
        'receive', '--device', str(args.device), '--headless', '--no-log',
        '--fixture', str(args.fixture),
        *(['--diagnostics'] if args.diagnostics else []),
        *(['--mono-compatible'] if args.mono_compatible else []),
    ])
    receiver = threading.Thread(target=v7_live.run_receive,
                                args=(receiver_args,), daemon=True)
    receiver.start()

    main_texture = None
    float_texture = None
    rendered = 0
    try:
        while state.run_mode:
            if display_manager.glfw is not None:
                display_manager.glfw.poll_events()
                if (not isinstance(window, display_manager.HeadlessWindow) and
                        display_manager.glfw.window_should_close(window)):
                    break
            frame = v7_live.FRAME_BUFFER.snapshot()
            if frame is not None and frame.generation != rendered:
                rgb = np.asarray(v7_live.values_image(frame.values,
                                                       frame.shapes),
                                 dtype=np.uint8)
                rgba = np.empty((*rgb.shape[:2], 4), dtype=np.uint8)
                rgba[:, :, :3] = rgb
                rgba[:, :, 3] = 255
                transparent = np.zeros_like(rgba)
                if main_texture is None:
                    main_texture = renderer.create_texture(rgba)
                    float_texture = renderer.create_texture(transparent)
                else:
                    main_texture = renderer.update_texture(main_texture, rgba)
                renderer.overlay_images_single_pass(
                    main_texture, float_texture, settings.BACKGROUND_COLOR)
                rendered = frame.generation
            if hasattr(window, 'swap_buffers'):
                window.swap_buffers()
    finally:
        state.run_mode = False
        if hasattr(window, 'close'):
            window.close()


if __name__ == '__main__':
    run(parser().parse_args())
