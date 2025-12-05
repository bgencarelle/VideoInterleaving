"""
display_manager.py – Window creation, sizing, and GL context management.
Supports:
1. Local ModernGL + GLFW (Full GPU)
2. Headless ModernGL (Server GPU)
3. Fallback/CPU Mode (No Window)
"""

from __future__ import annotations
import math
import moderngl
import numpy as np
import settings
import renderer

# --- Conditional Imports for Local/GL Support ---
_GLFW_AVAILABLE = False
try:
    import glfw
    from OpenGL.GL import (
        glEnable, glGetError, GL_FRAMEBUFFER_SRGB, GL_INVALID_ENUM,
    )
    _GLFW_AVAILABLE = True
except ImportError:
    glfw = None
    # If we are strictly local (not server) and missing libs, warn the user.
    if not settings.SERVER_MODE:
        print("Warning: GLFW/PyOpenGL not found. System will run in CPU-only mode.")

from settings import (
    INITIAL_ROTATION,
    INITIAL_MIRROR,
    CONNECTED_TO_RCA_HDMI,
    RCA_HDMI_RESOLUTION,
    LOW_RES_FULLSCREEN,
    LOW_RES_FULLSCREEN_RESOLUTION,
    ENABLE_SRGB_FRAMEBUFFER,
    VSYNC,
    GAMMA_CORRECTION_ENABLED,
)

# Global singleton references
window = None
ctx = None


class DisplayState:
    def __init__(self, image_size: tuple[int, int] = (640, 480)) -> None:
        self.image_size = image_size  # (width, height) of original image
        self.rotation = INITIAL_ROTATION  # e.g., 0, 90, 180, 270
        self.mirror = INITIAL_MIRROR  # 0 = normal, 1 = horizontal mirror
        self.fullscreen = True  # start in fullscreen unless overridden
        self.run_mode = True  # main loop flag
        self.needs_update = False  # set by callbacks to trigger re-init


class HeadlessWindow:
    """
    Minimal wrapper for Headless/Server Mode to mimic a GLFW window.
    """
    def __init__(self, ctx: moderngl.Context, fbo: moderngl.Framebuffer, size: tuple[int, int]):
        self.ctx = ctx
        self.fbo = fbo
        self.size = size  # (width, height)

    def use(self) -> None:
        self.fbo.use()


def _largest_mode(monitor):
    """Return the glfw video mode with the greatest pixel area."""
    if not glfw: return None
    modes = glfw.get_video_modes(monitor)
    return max(modes, key=lambda m: m.size.width * m.size.height)


def display_init(state: DisplayState):
    """
    Initialize or Reconfigure the Display Context.

    Returns:
        - Window Object (HeadlessWindow or glfw Window) if GL is active.
        - None if running in CPU/Fallback mode.
    """
    global window, ctx

    # -------------------------------------------------------------------------
    # PATH A: SERVER / HEADLESS MODE
    # -------------------------------------------------------------------------
    if settings.SERVER_MODE:
        use_gl = getattr(settings, "HEADLESS_USE_GL", False)

        if not use_gl:
            print("[DISPLAY] SERVER_MODE: GL disabled by settings. Using CPU-only compositor.")
            return None

        # Check for libraries
        if not _GLFW_AVAILABLE:
            print("[DISPLAY] SERVER_MODE: GL requested but GLFW/OpenGL missing. Fallback to CPU.")
            return None

        print(f"[DISPLAY] Initializing headless ModernGL (port={getattr(settings, 'STREAM_PORT', 8080)})...")

        try:
            backend = getattr(settings, "HEADLESS_BACKEND", None)
            create_kwargs = {"standalone": True}
            if backend:
                create_kwargs["backend"] = backend

            ctx = moderngl.create_context(**create_kwargs)

        except Exception as e:
            print(f"[DISPLAY] Failed to create headless GL context: {e}")
            print("[DISPLAY] Falling back to CPU-only compositing.")
            return None

        width, height = getattr(settings, "HEADLESS_RES", (1280, 720))
        state.image_size = (width, height)

        tex = ctx.texture((width, height), components=4)
        fbo = ctx.framebuffer(color_attachments=[tex])
        fbo.use()
        ctx.viewport = (0, 0, width, height)

        renderer.initialize(ctx)
        renderer.set_viewport_size(width, height)

        print(f"[DISPLAY] Headless GL context ready at {width}x{height}")
        return HeadlessWindow(ctx, fbo, (width, height))

    # -------------------------------------------------------------------------
    # PATH B: LOCAL WINDOWED MODE
    # -------------------------------------------------------------------------

    # If GLFW is missing locally, we MUST fallback to CPU (for ASCII/Terminal modes)
    if not _GLFW_AVAILABLE:
        return None

    # Calculate effective dimensions based on rotation
    img_w, img_h = state.image_size
    eff_w, eff_h = (img_h, img_w) if state.rotation % 180 == 90 else (img_w, img_h)

    # Create Window if it doesn't exist
    if window is None:
        if not glfw.init():
            print("Error: GLFW init failed.")
            return None

        # Raspberry Pi / GLES detection
        is_pi = False
        try:
            with open('/proc/device-tree/model', 'r') as m:
                if 'Raspberry Pi' in m.read():
                    is_pi = True
        except FileNotFoundError:
            pass

        # Context Hints
        if is_pi:
            glfw.window_hint(glfw.CLIENT_API, glfw.OPENGL_ES_API)
            glfw.window_hint(glfw.CONTEXT_CREATION_API, glfw.EGL_CONTEXT_API)
            glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
            glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 1)
        else:
            glfw.window_hint(glfw.CLIENT_API, glfw.OPENGL_API)
            glfw.window_hint(glfw.CONTEXT_CREATION_API, glfw.NATIVE_CONTEXT_API)
            glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
            glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
            glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)

        if GAMMA_CORRECTION_ENABLED or ENABLE_SRGB_FRAMEBUFFER:
            glfw.window_hint(glfw.SRGB_CAPABLE, glfw.TRUE)

        # Create the actual window
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
            # Windowed mode aspect ratio logic
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
            return None

        glfw.make_context_current(window)
        glfw.swap_interval(1 if VSYNC else 0)

        # Create Context
        if is_pi:
            ctx = moderngl.create_context(require=310)
        else:
            ctx = moderngl.create_context()

        renderer.initialize(ctx)

        if GAMMA_CORRECTION_ENABLED or ENABLE_SRGB_FRAMEBUFFER:
            glEnable(GL_FRAMEBUFFER_SRGB)
            if glGetError() == GL_INVALID_ENUM:
                print("⚠️  GL_FRAMEBUFFER_SRGB unsupported – continuing without it")

    # -------------------------------------------------------------------------
    # RECONFIGURATION (Fullscreen Toggling / Sizing)
    # -------------------------------------------------------------------------
    # Only runs if window exists

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

        # Avoid redundant monitor switching if already correct
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

    # Force resize check
    if not state.fullscreen:
        fb_w, fb_h = glfw.get_window_size(window)
        if (fb_w, fb_h) != (win_w, win_h):
            glfw.set_window_size(window, win_w, win_h)

    # Input Modes
    if state.fullscreen:
        glfw.set_input_mode(window, glfw.CURSOR, glfw.CURSOR_HIDDEN)
    else:
        glfw.set_input_mode(window, glfw.CURSOR, glfw.CURSOR_NORMAL)

    # -------------------------------------------------------------------------
    # TRANSFORM CALCULATION (MVP)
    # -------------------------------------------------------------------------
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
        scale = fb_w / eff_w
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

    # Orthographic Projection
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