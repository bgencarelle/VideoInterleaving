"""
display_manager.py – window creation / sizing and GL transform glue
ModernGL + GLFW version (replaces the legacy pygame/OpenGL path)
"""

from __future__ import annotations

import math

import glfw
import moderngl
import numpy as np
from OpenGL.GL import (
    glEnable,
    glGetError,
    GL_FRAMEBUFFER_SRGB,
    GL_INVALID_ENUM,
)

import renderer
from settings import (
    INITIAL_ROTATION,
    INITIAL_MIRROR,
    CONNECTED_TO_RCA_HDMI,
    RCA_HDMI_RESOLUTION,
    LOW_RES_FULLSCREEN,
    LOW_RES_FULLSCREEN_RESOLUTION,
    ENABLE_SRGB_FRAMEBUFFER,
    GAMMA_CORRECTION_ENABLED,
)

# -----------------------------------------------------------------------------
# Globals
# -----------------------------------------------------------------------------
window: "glfw._GLFWwindow" | None = None
ctx: moderngl.Context | None = None

# -----------------------------------------------------------------------------
# State container (holds display state, similar fields as legacy version)
# -----------------------------------------------------------------------------
class DisplayState:
    def __init__(self, image_size: tuple[int, int] = (640, 480)) -> None:
        self.image_size = image_size    # (width, height) of original image
        self.rotation = INITIAL_ROTATION  # e.g., 0, 90, 180, 270 (in degrees)
        self.mirror = INITIAL_MIRROR      # 0 = normal, 1 = horizontal mirror
        self.fullscreen = True            # start in fullscreen unless overridden
        self.run_mode = True             # main loop flag (False to exit)
        self.needs_update = False        # set by callbacks to trigger re-init

# -----------------------------------------------------------------------------
# Helper – choose the largest-area video mode for a monitor
# -----------------------------------------------------------------------------
def _largest_mode(monitor: "glfw._GLFWmonitor"):
    """Return the glfw video mode with the greatest pixel area for the given monitor."""
    modes = glfw.get_video_modes(monitor)
    return max(modes, key=lambda m: m.size.width * m.size.height)

# -----------------------------------------------------------------------------
# Display initialization / reconfiguration
# -----------------------------------------------------------------------------
def display_init(state: DisplayState) -> "glfw._GLFWwindow":
    """
    Create (on first call) or reconfigure (on subsequent calls) the GLFW window and
    ModernGL context based on the current state. Handles fullscreen toggling,
    window resizing for image rotation, cursor visibility, and updates the renderer
    transform parameters. Returns the active window handle.
    """
    global window, ctx

    # Determine effective image dimensions (swap width/height if rotated 90°/270°)
    img_w, img_h = state.image_size
    eff_w, eff_h = (img_h, img_w) if state.rotation % 180 == 90 else (img_w, img_h)

    # -----------------------------------------------------------------------------
    # First-time initialization
    # -----------------------------------------------------------------------------
    if window is None:
        if not glfw.init():
            raise RuntimeError("GLFW initialization failed")

        # Detect Raspberry Pi by checking for the Pi model string
        is_pi = False
        try:
            with open('/proc/device-tree/model', 'r') as model_file:
                if 'Raspberry Pi' in model_file.read():
                    is_pi = True
        except FileNotFoundError:
            pass

        # Set GLFW context hints based on platform
        if is_pi:
            # Use OpenGL ES 3.1 via EGL on Raspberry Pi
            glfw.window_hint(glfw.CLIENT_API, glfw.OPENGL_ES_API)
            glfw.window_hint(glfw.CONTEXT_CREATION_API, glfw.EGL_CONTEXT_API)
            glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
            glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 1)
        else:
            # Use desktop OpenGL 3.3 core profile elsewhere
            glfw.window_hint(glfw.CLIENT_API, glfw.OPENGL_API)
            glfw.window_hint(glfw.CONTEXT_CREATION_API, glfw.NATIVE_CONTEXT_API)
            glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
            glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
            glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
        if GAMMA_CORRECTION_ENABLED or ENABLE_SRGB_FRAMEBUFFER:
            glfw.window_hint(glfw.SRGB_CAPABLE, glfw.TRUE)

        # Choose initial window size / monitor
        if state.fullscreen:
            mon = glfw.get_primary_monitor()
            if CONNECTED_TO_RCA_HDMI:
                fs_w, fs_h = RCA_HDMI_RESOLUTION
            elif LOW_RES_FULLSCREEN:
                fs_w, fs_h = LOW_RES_FULLSCREEN_RESOLUTION
            else:
                best_mode = _largest_mode(mon)
                fs_w, fs_h = best_mode.size.width, best_mode.size.height
            window = glfw.create_window(fs_w, fs_h, "Fullscreen", mon, None)
        else:
            rotation_mod = state.rotation % 360
            if rotation_mod in (90, 270):
                eff_w, eff_h = state.image_size[1], state.image_size[0]
            else:
                eff_w, eff_h = state.image_size[0], state.image_size[1]
            aspect_ratio = eff_w / eff_h
            win_w = 400
            win_h = int(win_w / aspect_ratio)
            window = glfw.create_window(win_w, win_h, "Windowed Mode", None, None)

        if not window:
            glfw.terminate()
            raise RuntimeError("Could not create GLFW window")

        glfw.make_context_current(window)
        glfw.swap_interval(1)  # Enable VSync

        # Create ModernGL context with appropriate version requirement
        if is_pi:
            # Pi reports ES 3.1 → version code 3*100 + 1*10 = 310
            ctx = moderngl.create_context(require=310)
        else:
            # Default → require desktop GL 3.3
            ctx = moderngl.create_context()
        renderer.initialize(ctx)

        # Enable sRGB framebuffer if requested
        if GAMMA_CORRECTION_ENABLED or ENABLE_SRGB_FRAMEBUFFER:
            glEnable(GL_FRAMEBUFFER_SRGB)
            if glGetError() == GL_INVALID_ENUM:
                print("⚠️  GL_FRAMEBUFFER_SRGB unsupported – continuing without it")

    # -----------------------------------------------------------------------------
    # Fullscreen ↔ Windowed toggling
    # -----------------------------------------------------------------------------
    current_monitor = glfw.get_window_monitor(window)
    if state.fullscreen:
        mon = glfw.get_primary_monitor()
        if CONNECTED_TO_RCA_HDMI:
            fs_w, fs_h = RCA_HDMI_RESOLUTION
        elif LOW_RES_FULLSCREEN:
            fs_w, fs_h = LOW_RES_FULLSCREEN_RESOLUTION
        else:
            best_mode = _largest_mode(mon)
            fs_w, fs_h = best_mode.size.width, best_mode.size.height
        refresh = getattr(best_mode, 'refresh_rate', glfw.get_video_mode(mon).refresh_rate)
        glfw.set_window_monitor(window, mon, 0, 0, fs_w, fs_h, refresh)
        glfw.set_window_title(window, "Fullscreen")
    else:
        rotation_mod = state.rotation % 360
        if rotation_mod in (90, 270):
            eff_w, eff_h = state.image_size[1], state.image_size[0]
        else:
            eff_w, eff_h = state.image_size[0], state.image_size[1]
        aspect_ratio = eff_w / eff_h
        win_w = 400
        win_h = int(win_w / aspect_ratio)
        glfw.set_window_monitor(window, None, 100, 100, win_w, win_h, 0)
        glfw.set_window_title(window, "Windowed Mode")

    # Force window resize on every rotation change in windowed mode
    if not state.fullscreen:
        fb_w, fb_h = glfw.get_window_size(window)
        if (fb_w, fb_h) != (win_w, win_h):
            glfw.set_window_size(window, win_w, win_h)

    # Cursor visibility
    if state.fullscreen:
        glfw.set_input_mode(window, glfw.CURSOR, glfw.CURSOR_HIDDEN)
    else:
        glfw.set_input_mode(window, glfw.CURSOR, glfw.CURSOR_NORMAL)

    # Update viewport & renderer transform
    fb_w, fb_h = glfw.get_framebuffer_size(window)
    ctx.viewport = (0, 0, fb_w, fb_h)

    if state.fullscreen:
        scale_x = fb_w / eff_w
        scale_y = fb_h / eff_h
        scale = min(scale_x, scale_y)
        scaled_w = eff_w * scale
        scaled_h = eff_h * scale
        offset_x = (fb_w - scaled_w) / 2.0
        offset_y = (fb_h - scaled_h) / 2.0
    else:
        scale = fb_w / eff_w  # windowed ratio is exact
        offset_x = 0
        offset_y = 0

    renderer.set_transform_parameters(
        scale,
        offset_x,
        offset_y,
        state.image_size,
        state.rotation,
        state.mirror,
    )

    # 2D orthographic projection (MVP)
    mvp = np.array(
        [
            [2.0 / fb_w, 0, 0, -1],
            [0, 2.0 / fb_h, 0, -1],
            [0, 0, -1,   0],
            [0, 0,  0,   1],
        ],
        dtype="f4",
    )
    renderer.update_mvp(mvp)

    return window
