"""
display_manager.py – window creation / sizing and GL transform glue
ModernGL + GLFW version (replaces the legacy pygame/OpenGL path)
"""

from __future__ import annotations

import glfw
import moderngl
import numpy as np
from OpenGL.GL import (
    glEnable,
    glGetError,
    GL_FRAMEBUFFER_SRGB,
    GL_INVALID_ENUM,
)  # only for optional sRGB enabling

import renderer
from settings import (
    INITIAL_ROTATION,
    INITIAL_MIRROR,
    CONNECTED_TO_RCA_HDMI,
    RCA_HDMI_RESOLUTION,
    LOW_RES_FULLSCREEN,
    LOW_RES_FULLSCREEN_RESOLUTION,
    ENABLE_SRGB_FRAMEBUFFER,
)

# -----------------------------------------------------------------------------#
# Globals                                                                      #
# -----------------------------------------------------------------------------#
window: "glfw._GLFWwindow" | None = None
ctx: moderngl.Context | None = None


# -----------------------------------------------------------------------------#
# State container (same fields the old code relied on)                          #
# -----------------------------------------------------------------------------#
class DisplayState:
    def __init__(self, image_size: tuple[int, int] = (640, 480)) -> None:
        self.image_size = image_size  # (width, height) of *original* image
        self.rotation = INITIAL_ROTATION  # 0 / 90 / 180 / 270
        self.mirror = INITIAL_MIRROR  # 0 = normal, 1 = horizontal mirror
        self.fullscreen = True  # start in fullscreen unless overridden
        self.run_mode = True  # main loop flag
        self.needs_update = False  # set by callbacks to trigger re-init


# -----------------------------------------------------------------------------#
# Helper – choose a “best” video mode on a monitor                              #
# -----------------------------------------------------------------------------#
def _largest_mode(monitor: "glfw._GLFWmonitor"):
    """Return the largest (area) glfw video mode for the given monitor."""
    modes = glfw.get_video_modes(monitor)
    return max(modes, key=lambda m: m.size.width * m.size.height)


# -----------------------------------------------------------------------------#
# Display initialisation / (re-)configuration                                   #
# -----------------------------------------------------------------------------#
def display_init(state: DisplayState) -> "glfw._GLFWwindow":
    """
    Create (first call) or reconfigure (subsequent calls) the GLFW window,
    ModernGL context, viewport and renderer transform based on *state*.
    Returns the active window handle.
    """

    global window, ctx

    # ------------------------------------------------------------------#
    # Derived image geometry                                             #
    # ------------------------------------------------------------------#
    img_w, img_h = state.image_size
    eff_w, eff_h = (img_h, img_w) if state.rotation % 180 == 90 else (img_w, img_h)

    # ------------------------------------------------------------------#
    # First-time initialisation                                          #
    # ------------------------------------------------------------------#
    if window is None:
        if not glfw.init():
            raise RuntimeError("GLFW initialisation failed")

        # Core-profile 3.3 is sufficient for ModernGL + simple 2D shaders
        glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
        glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
        glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
        glfw.window_hint(glfw.OPENGL_FORWARD_COMPAT, glfw.TRUE)  # macOS
        glfw.window_hint(glfw.DOUBLEBUFFER, glfw.TRUE)
        if ENABLE_SRGB_FRAMEBUFFER:
            glfw.window_hint(glfw.SRGB_CAPABLE, glfw.TRUE)

        # ------ choose initial window size / monitor ------------------#
        if state.fullscreen:
            mon = glfw.get_primary_monitor()
            if CONNECTED_TO_RCA_HDMI:
                fs_w, fs_h = RCA_HDMI_RESOLUTION
            elif LOW_RES_FULLSCREEN:
                fs_w, fs_h = LOW_RES_FULLSCREEN_RESOLUTION
            else:
                best = _largest_mode(mon)
                fs_w, fs_h = best.size.width, best.size.height
            window = glfw.create_window(fs_w, fs_h, "Fullscreen", mon, None)
        else:
            base_w = max(400, eff_w) if eff_w < 400 else 400
            win_w = base_w
            win_h = int(win_w * eff_h / eff_w)
            window = glfw.create_window(win_w, win_h, "Windowed Mode", None, None)

        if not window:
            glfw.terminate()
            raise RuntimeError("Could not create GLFW window")

        glfw.make_context_current(window)
        glfw.swap_interval(1)  # enable VSync for tear-free timing
        ctx = moderngl.create_context()
        renderer.initialize(ctx)

        # Optional: enable automatic sRGB→linear on framebuffer writes
        if ENABLE_SRGB_FRAMEBUFFER:
            glEnable(GL_FRAMEBUFFER_SRGB)
            if glGetError() == GL_INVALID_ENUM:
                print("⚠️  GL_FRAMEBUFFER_SRGB unsupported – continuing without it")

    # ------------------------------------------------------------------#
    # Fullscreen ↔ windowed toggling                                     #
    # ------------------------------------------------------------------#
    current_monitor = glfw.get_window_monitor(window)
    if state.fullscreen and current_monitor is None:
        # → switch to fullscreen
        mon = glfw.get_primary_monitor()
        if CONNECTED_TO_RCA_HDMI:
            fs_w, fs_h = RCA_HDMI_RESOLUTION
        elif LOW_RES_FULLSCREEN:
            fs_w, fs_h = LOW_RES_FULLSCREEN_RESOLUTION
        else:
            best = _largest_mode(mon)
            fs_w, fs_h = best.size.width, best.size.height
        refresh = glfw.get_video_mode(mon).refresh_rate
        glfw.set_window_monitor(window, mon, 0, 0, fs_w, fs_h, refresh)
        glfw.set_window_title(window, "Fullscreen")
    elif not state.fullscreen and current_monitor is not None:
        # → switch to windowed
        base_w = max(400, eff_w) if eff_w < 400 else 400
        win_w = base_w
        win_h = int(win_w * eff_h / eff_w)
        glfw.set_window_monitor(window, None, 100, 100, win_w, win_h, 0)
        glfw.set_window_title(window, "Windowed Mode")

    # Keep cursor *hidden* but *not* captured (user can Alt-Tab out)
    glfw.set_input_mode(window, glfw.CURSOR, glfw.CURSOR_HIDDEN)

    # ------------------------------------------------------------------#
    # Viewport + renderer transform                                      #
    # ------------------------------------------------------------------#
    fb_w, fb_h = glfw.get_framebuffer_size(window)
    ctx.viewport = (0, 0, fb_w, fb_h)

    # Letterbox / pillarbox scaling
    scale_x = fb_w / eff_w
    scale_y = fb_h / eff_h
    scale = min(scale_x, scale_y)

    scaled_w = eff_w * scale
    scaled_h = eff_h * scale
    offset_x = (fb_w - scaled_w) / 2.0
    offset_y = (fb_h - scaled_h) / 2.0

    renderer.set_transform_parameters(
        scale,
        offset_x,
        offset_y,
        state.image_size,
        state.rotation,
        state.mirror,
    )

    # 2D orthographic MVP (pixel-space -> NDC)
    mvp = np.array(
        [
            [2.0 / fb_w, 0, 0, -1],
            [0, 2.0 / fb_h, 0, -1],
            [0, 0, -1, 0],
            [0, 0, 0, 1],
        ],
        dtype="f4",
    )
    renderer.update_mvp(mvp)

    return window
