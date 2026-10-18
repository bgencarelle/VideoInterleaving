"""Direct per-layer sampling onto requested font-aware character grids."""
import numpy as np
import renderer
import settings
from ascii_terminal import SCALES


class AsciiGridCapture:
    def __init__(self, context, sizes=None):
        self.context = context
        self.targets = {size: context.simple_framebuffer(size, components=3)
                         for size in (SCALES if sizes is None else sizes)}

    def capture(self, main, front, background, main_sbs, front_sbs, window, sizes=None):
        frames = {}
        try:
            for size, target in self.targets.items():
                if sizes is not None and size not in sizes:
                    continue
                target.use()
                renderer.set_viewport_size(*target.size,
                                           pixel_aspect=settings.ASCII_FONT_RATIO)
                renderer.overlay_images_single_pass(main, front, background,
                                                   main_sbs, front_sbs)
                data = target.read(components=3, alignment=1)
                width, height = target.size
                frames[size] = np.frombuffer(data, np.uint8).reshape(height, width, 3)
        finally:
            window.use()
            renderer.set_viewport_size(*window.size)
        return frames

    def close(self):
        for target in self.targets.values():
            target.release()
        self.targets.clear()
