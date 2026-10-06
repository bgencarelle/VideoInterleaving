import threading
import unittest
from collections import OrderedDict
from types import SimpleNamespace
from unittest.mock import Mock, patch

from PIL import Image, ImageDraw, ImageFont

from scope_gui import ScopeGUI


class ScopePresentationTests(unittest.TestCase):
    def make_gui(self):
        gui = ScopeGUI.__new__(ScopeGUI)
        gui.closed = gui.close_requested = False
        gui.window = object()
        gui.glfw = SimpleNamespace(
            poll_events=Mock(), window_should_close=Mock(return_value=False),
            get_window_size=Mock(return_value=(32, 24)),
            get_framebuffer_size=Mock(return_value=(32, 24)),
            swap_buffers=Mock())
        gui._preview_lock = threading.Lock()
        gui._preview_rgb = None
        gui._preview_error = ""
        gui._state = gui._metrics = {}
        gui.image_only = gui.fullscreen = False
        gui.message = ""
        gui._dragging = gui._drag_value = None
        gui.preview_exposure = gui.preview_spot_width = 1.0
        gui._last_draw_signature = None
        gui._refresh_revision = 0
        gui._last_draw = 0.0
        gui._actions = []
        gui._draw = Mock()
        return gui

    def test_rate_limit_retains_changes_and_idle_poll_still_pumps_events(self):
        gui = self.make_gui()
        with patch("scope_gui.time.monotonic", return_value=10.0):
            gui.poll({"mode": "raster"})
        with patch("scope_gui.time.monotonic", return_value=10.1):
            gui._preview_rgb = object()
            gui.poll({"mode": "raster"})
        self.assertEqual(gui._draw.call_count, 1)
        with patch("scope_gui.time.monotonic", return_value=10.3):
            gui.poll({"mode": "raster"})
        self.assertEqual(gui._draw.call_count, 2)
        with patch("scope_gui.time.monotonic", return_value=11.0):
            gui.poll({"mode": "raster"})
        self.assertEqual(gui._draw.call_count, 2)
        self.assertEqual(gui.glfw.poll_events.call_count, 4)

    def test_resize_local_changes_and_worker_publication_during_draw(self):
        gui = self.make_gui()
        gui._draw.side_effect = lambda: setattr(gui, "_preview_rgb", object())
        with patch("scope_gui.time.monotonic", return_value=10.0):
            gui.poll({})
        gui._draw.side_effect = None
        with patch("scope_gui.time.monotonic", return_value=10.3):
            gui.poll({})
        self.assertEqual(gui._draw.call_count, 2)
        for index, change in enumerate((
                lambda: setattr(gui, "message", "changed"),
                lambda: setattr(gui, "image_only", True),
                lambda: setattr(gui, "_drag_value", 2.0),
                lambda: gui._on_refresh(gui.window),
                lambda: setattr(gui.glfw.get_window_size, "return_value", (40, 24)))):
            change()
            with patch("scope_gui.time.monotonic", return_value=11.0 + index):
                gui.poll({})
            self.assertEqual(gui._draw.call_count, 3 + index)

    def test_uploads_changed_color_rectangle_and_recreates_on_resize(self):
        gui = self.make_gui()
        gui.context = SimpleNamespace(clear=Mock(), texture=Mock())
        gui.moderngl = SimpleNamespace(LINEAR=1, TRIANGLES=2)
        gui.vao = SimpleNamespace(render=Mock())
        gui.texture = None
        gui._uploaded_image = None
        texture = SimpleNamespace(size=(32, 24), use=Mock(), write=Mock(),
                                  release=Mock())
        gui.context.texture.return_value = texture
        image = Image.new("RGBA", (32, 24), (7, 7, 7, 255))
        gui._present(image)
        gui._present(image)
        texture.write.assert_not_called()
        image.putpixel((5, 8), (255, 0, 0, 255))
        gui._present(image)
        texture.write.assert_called_once_with(
            bytes((255, 0, 0, 255)), viewport=(5, 8, 1, 1), alignment=1)
        gui._present(Image.new("RGBA", (40, 24)))
        texture.release.assert_called_once()
        self.assertEqual(gui.context.texture.call_count, 2)

    def test_cached_static_text_matches_pillow_and_reuses_surfaces(self):
        gui = self.make_gui()
        gui.Image = Image
        gui.ImageDraw = ImageDraw
        gui._text_surface_cache = OrderedDict()
        font = ImageFont.truetype("DejaVuSans.ttf", 17)
        labels = (
            ((10, 8), "SCOPE  ·  LIVE TUNER", (239, 245, 249)),
            ((13.5, 23.7), "startup default", (159, 173, 181)),
            ((28.25, 37.0), "Phosphor preview  ·  exposure", (119, 145, 160)),
        )
        for position, text, color in labels:
            expected = Image.new("RGBA", (320, 64), (12, 19, 26, 255))
            actual = expected.copy()
            ImageDraw.Draw(expected).text(position, text, fill=color, font=font)
            gui._draw_cached_text(actual, position, text, color, font)
            self.assertEqual(actual.tobytes(), expected.tobytes())
            cached_tile = next(reversed(gui._text_surface_cache.values()))
            repeated = Image.new("RGBA", (320, 64), (12, 19, 26, 255))
            gui._draw_cached_text(repeated, position, text, color, font)
            self.assertIs(next(reversed(gui._text_surface_cache.values())),
                          cached_tile)
            self.assertEqual(repeated.tobytes(), expected.tobytes())


if __name__ == "__main__":
    unittest.main()
