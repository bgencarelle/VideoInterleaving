import ast
import contextlib
import io
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

import numpy as np


def _make_stub_cv2():
    def _resize(image, size, interpolation=None):
        new_w, new_h = size
        h, w = image.shape[:2]
        y_idx = np.linspace(0, h - 1, new_h).astype(int)
        x_idx = np.linspace(0, w - 1, new_w).astype(int)
        return image[np.ix_(y_idx, x_idx)]

    def _cvt_color(image, code):
        if code == 0:  # COLOR_RGB2HSV
            hsv = np.zeros_like(image, dtype=float)
            value = image.mean(axis=2)
            hsv[..., 2] = value
            return hsv
        if code == 1:  # COLOR_HSV2RGB
            value = image[..., 2]
            return np.stack([value, value, value], axis=2)
        if code == 2:  # COLOR_RGB2GRAY
            return image.mean(axis=2)
        raise ValueError("Unsupported color conversion code")

    def _lut(src, table):
        return table[src.astype(int)]

    return types.SimpleNamespace(
        COLOR_RGB2HSV=0,
        COLOR_HSV2RGB=1,
        COLOR_RGB2GRAY=2,
        INTER_NEAREST=0,
        resize=_resize,
        cvtColor=_cvt_color,
        LUT=_lut,
    )


try:  # pragma: no cover - exercised implicitly through ascii_converter import
    import cv2  # type: ignore
except ImportError:  # pragma: no cover - fallback for environments without libGL
    cv2 = _make_stub_cv2()
    sys.modules['cv2'] = cv2
else:  # pragma: no cover - real cv2 path
    pass

import ascii_converter
import settings


class ToAsciiPreHsvAdjustmentsTest(unittest.TestCase):
    def setUp(self):
        self.original_settings = {
            'ASCII_WIDTH': settings.ASCII_WIDTH,
            'ASCII_HEIGHT': settings.ASCII_HEIGHT,
            'ASCII_FONT_RATIO': settings.ASCII_FONT_RATIO,
            'ASCII_CONTRAST': settings.ASCII_CONTRAST,
            'ASCII_SATURATION': settings.ASCII_SATURATION,
            'ASCII_BRIGHTNESS': settings.ASCII_BRIGHTNESS,
            'ASCII_COLOR': settings.ASCII_COLOR,
        }

        settings.ASCII_WIDTH = 1
        settings.ASCII_HEIGHT = 1
        settings.ASCII_FONT_RATIO = 1.0
        settings.ASCII_CONTRAST = 1.0
        settings.ASCII_SATURATION = 1.0
        settings.ASCII_BRIGHTNESS = 1.0
        settings.ASCII_COLOR = False

        self.original_chars = ascii_converter.CHARS
        self.original_gamma = ascii_converter.GAMMA_LUT
        self.original_gamma_val = ascii_converter._gamma_val
        self.original_cv2 = ascii_converter.cv2

        ascii_converter.CHARS = np.array(list("abcde"))
        ascii_converter.GAMMA_LUT = np.arange(256, dtype=np.uint8)
        # GAMMA_LUT is a cache now, rebuilt whenever ASCII_GAMMA differs from
        # the value it was built for. Tell it this identity LUT IS the current
        # gamma, or the first frame silently rebuilds over the patch.
        ascii_converter._gamma_val = getattr(settings, 'ASCII_GAMMA', 1.0)
        ascii_converter.cv2 = _make_stub_cv2()

    def tearDown(self):
        for key, value in self.original_settings.items():
            setattr(settings, key, value)

        ascii_converter.CHARS = self.original_chars
        ascii_converter.GAMMA_LUT = self.original_gamma
        ascii_converter._gamma_val = self.original_gamma_val
        ascii_converter.cv2 = self.original_cv2

    def test_pre_hsv_brightness_boosts_ascii_tone(self):
        frame = np.array([[[100, 100, 100]]], dtype=np.uint8)
        baseline = ascii_converter.to_ascii(frame)

        settings.ASCII_BRIGHTNESS = 1.5
        boosted = ascii_converter.to_ascii(frame)

        reset = ascii_converter.RESET_CODE
        base_char = baseline[:-len(reset)]
        boosted_char = boosted[:-len(reset)]

        chars = ascii_converter.CHARS.tolist()
        self.assertLess(chars.index(boosted_char), chars.index(base_char))


class AsciiContrastCommandLineTest(unittest.TestCase):
    """--ascii-contrast, driven through main.configure_runtime() for real.

    The setting had to be edited in constantStorage to change it, which is a
    source edit to grade a picture -- the same complaint that produced
    --rotation and --mirror.
    """

    def setUp(self):
        self.saved = vars(settings).copy()
        self.addCleanup(self.restore)
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        path = Path(__file__).with_name("main.py")
        tree = ast.parse(path.read_text(), filename=str(path))
        stop = next(i for i, node in enumerate(tree.body)
                    if isinstance(node, ast.Assign)
                    and isinstance(node.value, ast.Call)
                    and isinstance(node.value.func, ast.Name)
                    and node.value.func.id == "configure_runtime")
        tree.body = tree.body[:stop]
        self.ns = {"__file__": str(Path(self.temp.name) / "main.py")}
        exec(compile(tree, str(path), "exec"), self.ns)
        self.ns["LOGS_DIR"] = str(Path(self.temp.name) / "logs")

    def restore(self):
        for key in set(vars(settings)) - set(self.saved):
            delattr(settings, key)
        vars(settings).update(self.saved)
        # This class drives ASCII_GAMMA, which leaves ascii_converter's LUT
        # cache built for a value the restored settings no longer hold.
        ascii_converter._gamma_val = getattr(settings, 'ASCII_GAMMA', 1.0)
        ascii_converter.GAMMA_LUT = ascii_converter._build_gamma_lut(
            ascii_converter._gamma_val)

    def configure(self, *flags, mode="ascii"):
        argv = ["main.py", "--mode", mode, "--dir", self.temp.name, *flags]
        out = io.StringIO()
        with patch.object(sys, "argv", argv), contextlib.redirect_stdout(out):
            self.ns["configure_runtime"]()
        return out.getvalue()

    def test_each_value_reaches_the_converter(self):
        self.configure("--ascii-contrast", "1.6",
                       "--ascii-brightness", "0.8",
                       "--ascii-gamma", "1.3")
        self.assertEqual(settings.ASCII_CONTRAST, 1.6)
        self.assertEqual(settings.ASCII_BRIGHTNESS, 0.8)
        self.assertEqual(settings.ASCII_GAMMA, 1.3)

    def test_gamma_survives_the_import_order_that_used_to_kill_it(self):
        """The trap that made this flag worth more than one line.

        GAMMA_LUT was built once, at import, from settings.ASCII_GAMMA. So
        whether a command-line gamma did anything depended entirely on whether
        ascii_converter had already been imported when the flag was applied --
        the same by-value trap display_manager had with INITIAL_ROTATION.
        ascii_converter is imported at the top of this file, long before
        configure_runtime runs here, which is the losing order.
        """
        frame = np.tile(np.linspace(0, 255, 64, dtype=np.uint8),
                        (8, 1))[:, :, None].repeat(3, 2)
        self.configure("--ascii-gamma", "1.0")
        linear = ascii_converter.to_ascii(frame)
        self.configure("--ascii-gamma", "2.5")
        curved = ascii_converter.to_ascii(frame)
        self.assertNotEqual(curved, linear,
                            "ASCII_GAMMA was read into a LUT that never rebuilt")

    def test_the_grading_flags_compose(self):
        frame = np.tile(np.linspace(0, 255, 64, dtype=np.uint8),
                        (8, 1))[:, :, None].repeat(3, 2)
        self.configure()
        plain = ascii_converter.to_ascii(frame)
        self.configure("--ascii-contrast", "1.5", "--ascii-brightness", "1.2",
                       "--ascii-gamma", "1.4")
        graded = ascii_converter.to_ascii(frame)
        self.assertNotEqual(graded, plain)

    def test_omitting_it_leaves_the_shipped_default_alone(self):
        self.configure()
        from constantStorage import ascii_constants
        self.assertEqual(settings.ASCII_CONTRAST, ascii_constants.ASCII_CONTRAST)

    def test_it_works_in_asciiweb_too(self):
        self.configure("--ascii-contrast", "0.7", mode="asciiweb")
        self.assertEqual(settings.ASCII_CONTRAST, 0.7)

    def test_a_mode_that_cannot_use_them_says_so(self):
        # A flag accepted in the wrong mode that quietly does nothing is the
        # exact failure this whole pass has been about.
        noise = self.configure("--ascii-contrast", "1.6",
                               "--ascii-gamma", "1.2", mode="local")
        self.assertIn("--ascii-contrast", noise)
        self.assertIn("--ascii-gamma", noise)
        self.assertIn("ignored", noise)
        # ...and only the ones actually passed are named.
        self.assertNotIn("--ascii-brightness", noise)

    def test_nonsense_values_are_refused(self):
        for flag in ("--ascii-contrast", "--ascii-brightness", "--ascii-gamma"):
            for bad in ("nan", "inf", "-1"):
                with self.subTest(flag=flag, value=bad), \
                        contextlib.redirect_stderr(io.StringIO()):
                    with self.assertRaises(SystemExit) as err:
                        self.configure(flag, bad)
                    self.assertEqual(err.exception.code, 2)

    def test_the_zero_endpoint_differs_by_what_zero_means(self):
        # Contrast 0 flattens to one tone and brightness 0 is black: both are
        # the honest low end of a multiply, so both are allowed.
        self.configure("--ascii-contrast", "0")
        self.assertEqual(settings.ASCII_CONTRAST, 0.0)
        self.configure("--ascii-brightness", "0")
        self.assertEqual(settings.ASCII_BRIGHTNESS, 0.0)
        # Gamma 0 is not an endpoint, it is a discontinuity: (i/255) ** 0 is
        # 1.0 for every non-zero i, so the whole field goes white.
        with contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):
                self.configure("--ascii-gamma", "0")


class ShippedAsciiDefaultsTest(unittest.TestCase):
    """The shipped grading defaults must not change the picture silently.

    Read from constantStorage directly, not from `settings`, because every
    other test in this file mutates settings in setUp and would mask this.
    """

    def test_contrast_ships_neutral(self):
        from constantStorage import ascii_constants
        self.assertEqual(
            ascii_constants.ASCII_CONTRAST, 1.0,
            "ASCII_CONTRAST is live now. Anything but 1.0 as the SHIPPED "
            "value changes the ASCII picture for everyone who never asked "
            "for it -- which is the whole reason it was wired up at 1.0 "
            "instead of the 1.2 it had been sitting at, inert, for years.")

    def test_the_gates_of_the_deleted_stage_are_not_reintroduced(self):
        from constantStorage import ascii_constants
        for name in ("ASCII_ENABLE_CONTRAST", "ASCII_ENABLE_RGB_BRIGHTNESS",
                     "ASCII_RGB_BRIGHTNESS"):
            self.assertFalse(
                hasattr(ascii_constants, name),
                f"{name} gated a grading stage that no longer exists; a "
                f"setting nothing reads is worse than no setting")


class ToAsciiColorParityTest(unittest.TestCase):
    """The single grading stage: brightness applies once, colour agrees."""

    def setUp(self):
        self.original_settings = {
            'ASCII_WIDTH': settings.ASCII_WIDTH,
            'ASCII_HEIGHT': settings.ASCII_HEIGHT,
            'ASCII_FONT_RATIO': settings.ASCII_FONT_RATIO,
            'ASCII_CONTRAST': settings.ASCII_CONTRAST,
            'ASCII_SATURATION': settings.ASCII_SATURATION,
            'ASCII_BRIGHTNESS': settings.ASCII_BRIGHTNESS,
            'ASCII_COLOR': settings.ASCII_COLOR,
            # Not a grading knob: with blur on, the colour is sampled from a
            # blurred copy while the character stays sharp, so char and colour
            # come from deliberately different pixels. These tests are about
            # the grading stage, so the blur is pinned off.
            'ASCII_COLOR_BLUR': getattr(settings, 'ASCII_COLOR_BLUR', 0),
        }

        settings.ASCII_WIDTH = 1
        settings.ASCII_HEIGHT = 1
        settings.ASCII_FONT_RATIO = 1.0
        settings.ASCII_CONTRAST = 1.0
        settings.ASCII_SATURATION = 1.0
        settings.ASCII_BRIGHTNESS = 1.0
        settings.ASCII_COLOR = True
        settings.ASCII_COLOR_BLUR = 0

        self.original_chars = ascii_converter.CHARS
        self.original_gamma = ascii_converter.GAMMA_LUT
        self.original_gamma_val = ascii_converter._gamma_val
        self.original_cv2 = ascii_converter.cv2

        # A 16-step ramp, not three characters: with a coarse palette two
        # very different grey levels land on the same character, and a test
        # that cannot tell 180 from 255 cannot see brightness applied twice.
        ascii_converter.CHARS = np.array(list("abcdefghijklmnop"))
        ascii_converter.GAMMA_LUT = np.arange(256, dtype=np.uint8)
        # GAMMA_LUT is a cache now, rebuilt whenever ASCII_GAMMA differs from
        # the value it was built for. Tell it this identity LUT IS the current
        # gamma, or the first frame silently rebuilds over the patch.
        ascii_converter._gamma_val = getattr(settings, 'ASCII_GAMMA', 1.0)
        ascii_converter.cv2 = _make_stub_cv2()

    def tearDown(self):
        for key, value in self.original_settings.items():
            setattr(settings, key, value)

        ascii_converter.CHARS = self.original_chars
        ascii_converter.GAMMA_LUT = self.original_gamma
        ascii_converter._gamma_val = self.original_gamma_val
        ascii_converter.cv2 = self.original_cv2

    # A reference implementation of the ONE grading stage to_ascii now has,
    # written from the pipeline's description rather than copied from it, so
    # the two can disagree. Grey in, grey out: the stub's RGB2HSV puts the
    # channel mean in V and HSV2RGB writes V back to all three.
    def reference(self, level, brightness, contrast=1.0):
        value = float(level)
        if contrast != 1.0:
            value = min(255.0, max(0.0, (value - 128.0) * contrast + 128.0))
        value = min(255.0, value * brightness)
        gray = int(np.uint8(value))
        chars = ascii_converter.CHARS.tolist()
        index = int((255 - gray) / 255 * (len(chars) - 1))
        char = chars[index]
        if not settings.ASCII_COLOR:
            return char + ascii_converter.RESET_CODE
        step = gray * 5 // 255
        ansi = 16 + 36 * step + 6 * step + step
        return f"\033[38;5;{ansi}m{char}{ascii_converter.RESET_CODE}"

    def test_brightness_is_applied_exactly_once(self):
        """The invariant the removed pre-HSV stage existed to break.

        These two tests used to drive ascii_converter.CONTRAST_LUT,
        _apply_rgb_brightness and _rgb_brightness -- a pre-HSV grading stage
        that multiplied brightness a second time, so they asserted that the
        HSV value multiplier was SUPPRESSED and a brightened frame came out
        identical to the baseline. Commit 9adda652 ("fixing ascii") deleted
        that stage and those three names with it, and the tests were left
        referencing attributes the module no longer has.

        One stage remains, so brightness must now apply once and visibly.
        """
        # The module globals the deleted stage referenced but never defined.
        for name in ("CONTRAST_LUT", "_apply_rgb_brightness", "_rgb_brightness"):
            self.assertFalse(
                hasattr(ascii_converter, name),
                f"{name} is back: contrast lives in the single HSV stage now, "
                f"so a second one has to decide what it does to "
                f"ASCII_CONTRAST and ASCII_BRIGHTNESS both")
        # ...and the settings that gated it. These are on `settings`, not on
        # ascii_converter, so they need their own check -- asking
        # ascii_converter for them would pass no matter what.
        for name in ("ASCII_ENABLE_CONTRAST", "ASCII_ENABLE_RGB_BRIGHTNESS",
                     "ASCII_RGB_BRIGHTNESS"):
            self.assertFalse(
                hasattr(settings, name),
                f"settings.{name} is back, but nothing reads it: that is the "
                f"state this whole exercise started in")

        settings.ASCII_COLOR = False
        frame = np.array([[[100, 100, 100]]], dtype=np.uint8)

        settings.ASCII_BRIGHTNESS = 1.0
        baseline = ascii_converter.to_ascii(frame)
        self.assertEqual(baseline, self.reference(100, 1.0))

        settings.ASCII_BRIGHTNESS = 1.8
        brightened = ascii_converter.to_ascii(frame)
        self.assertEqual(brightened, self.reference(100, 1.8))

        # Once, not zero times (the old behaviour) and not twice.
        self.assertNotEqual(brightened, baseline)

    def test_contrast_at_one_is_exactly_neutral(self):
        """The property that let a dead setting become live without a redraw.

        ASCII_CONTRAST sat at 1.2 in ascii_constants for a long time doing
        nothing, because the stage that read it referenced globals that were
        never defined and raised NameError on every frame until 9adda652
        removed it. Wiring it into the surviving stage would have silently
        changed everyone's picture, so it scales about mid-grey -- where 1.0
        is an exact identity -- and ships at 1.0.
        """
        settings.ASCII_COLOR = False
        frame = np.array([[[100, 100, 100]]], dtype=np.uint8)
        settings.ASCII_BRIGHTNESS = 1.4          # a non-trivial pipeline

        settings.ASCII_CONTRAST = 1.0
        neutral = ascii_converter.to_ascii(frame)

        # Identical to a pipeline that never mentions contrast at all.
        self.assertEqual(neutral, self.reference(100, 1.4))

        # And it is a real control, not a no-op: mid-grey is the pivot, so a
        # value below it must get darker as contrast rises.
        settings.ASCII_CONTRAST = 2.0
        punchy = ascii_converter.to_ascii(frame)
        self.assertEqual(punchy, self.reference(100, 1.4, contrast=2.0))
        self.assertNotEqual(punchy, neutral)

        chars = ascii_converter.CHARS.tolist()
        reset = ascii_converter.RESET_CODE
        self.assertGreater(chars.index(punchy[:-len(reset)]),
                           chars.index(neutral[:-len(reset)]),
                           "100 is below mid-grey, so more contrast is darker")

    def test_contrast_and_brightness_do_not_collapse_into_each_other(self):
        """Two knobs, two distinct operations, applied in a stated order."""
        settings.ASCII_COLOR = False
        frame = np.array([[[160, 160, 160]]], dtype=np.uint8)

        settings.ASCII_CONTRAST, settings.ASCII_BRIGHTNESS = 1.5, 1.0
        contrast_only = ascii_converter.to_ascii(frame)
        settings.ASCII_CONTRAST, settings.ASCII_BRIGHTNESS = 1.0, 1.5
        brightness_only = ascii_converter.to_ascii(frame)

        # Same multiplier, different pivot -- so they cannot be the same op.
        self.assertEqual(contrast_only, self.reference(160, 1.0, contrast=1.5))
        self.assertEqual(brightness_only, self.reference(160, 1.5))
        self.assertNotEqual(contrast_only, brightness_only)

        # Contrast first, then brightness: the documented order.
        settings.ASCII_CONTRAST, settings.ASCII_BRIGHTNESS = 1.5, 1.5
        both = ascii_converter.to_ascii(frame)
        self.assertEqual(both, self.reference(160, 1.5, contrast=1.5))

    def test_the_colour_path_grades_the_same_pixel_it_colours(self):
        """The char and its ANSI colour must come from the same graded pixel."""
        settings.ASCII_COLOR = True
        frame = np.array([[[100, 100, 100]]], dtype=np.uint8)

        for brightness in (1.0, 1.8, 3.0):          # 3.0 clips at 255
            with self.subTest(brightness=brightness):
                settings.ASCII_BRIGHTNESS = brightness
                self.assertEqual(ascii_converter.to_ascii(frame),
                                 self.reference(100, brightness))


if __name__ == "__main__":
    unittest.main()
