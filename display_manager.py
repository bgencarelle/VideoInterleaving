"""
display_manager.py – Window creation, sizing, and GL context management.
"""
from __future__ import annotations
import moderngl
import numpy as np
import settings
import renderer
import sys
import os

# --- Conditional Imports ---
_GLFW_AVAILABLE = False
try:
    import glfw
    from OpenGL.GL import (
        glEnable, glGetError, GL_FRAMEBUFFER_SRGB, GL_INVALID_ENUM,
    )
    _GLFW_AVAILABLE = True
except ImportError:
    glfw = None
    if not settings.SERVER_MODE:
        print("Warning: GLFW/PyOpenGL not found. Local window creation will fail.")

from settings import (
    INITIAL_ROTATION, INITIAL_MIRROR, CONNECTED_TO_RCA_HDMI,
    RCA_HDMI_RESOLUTION, LOW_RES_FULLSCREEN, LOW_RES_FULLSCREEN_RESOLUTION,
    ENABLE_SRGB_FRAMEBUFFER, VSYNC, GAMMA_CORRECTION_ENABLED,
)

window = None
ctx = None

class DisplayState:
    def __init__(self, image_size: tuple[int, int] = (640, 480)) -> None:
        self.image_size = image_size
        self.rotation = INITIAL_ROTATION
        self.mirror = INITIAL_MIRROR
        self.fullscreen = True
        self.run_mode = True
        self.needs_update = False

class HeadlessWindow:
    def __init__(self, ctx: moderngl.Context, fbo: moderngl.Framebuffer, size: tuple[int, int]):
        self.ctx = ctx
        self.fbo = fbo
        self.size = size

    def use(self) -> None:
        self.fbo.use()

    def swap_buffers(self):
        pass

def _largest_mode(monitor):
    if not glfw: return None
    modes = glfw.get_video_modes(monitor)
    return max(modes, key=lambda m: m.size.width * m.size.height)

def _log_renderer_info(ctx):
    try:
        info = ctx.info
        renderer_name = info.get('GL_RENDERER', 'Unknown')
        version_code = ctx.version_code
        print(f"[DISPLAY] GL Context: {renderer_name} (Version {version_code})")
    except:
        pass

def _check_is_pi():
    """Detect Raspberry Pi."""
    try:
        if os.path.exists('/proc/device-tree/model'):
            with open('/proc/device-tree/model', 'r') as m:
                if 'Raspberry Pi' in m.read():
                    return True
    except:
        pass
    return False

def display_init(state: DisplayState):
    global window, ctx

    is_pi = _check_is_pi()

    # --- PATH A: SERVER / HEADLESS MODE ---
    is_server = settings.SERVER_MODE
    is_ascii = getattr(settings, 'ASCII_MODE', False)

    if is_server or is_ascii:
        use_gl = getattr(settings, "HEADLESS_USE_GL", False)

        if not use_gl:
            print("❌ CONFIG ERROR: HEADLESS_USE_GL must be True in settings.py for performance.")
            sys.exit(1)

        # --- CONTEXT CREATION STRATEGY ---
        # 1. Try EGL without version (Let driver decide)
        # 2. Try EGL with version 200 (Force Pi Zero compatibility)
        # 3. Try Default/Standalone (For Mac/PC dev)

        attempts = [
            {"standalone": True, "backend": "egl"},                 # Best for Pi 4/5
            {"standalone": True, "backend": "egl", "require": 200}, # Explicit for Pi Zero
            {"standalone": True},                                   # Fallback for Mac/PC
        ]

        context_created = False
        last_error = None

        print("[DISPLAY] Initializing Headless GL...")

        for kwargs in attempts:
            try:
                # Skip EGL on non-Linux unless specified
                if kwargs.get('backend') == 'egl' and sys.platform == 'darwin':
                    continue

                ctx = moderngl.create_context(**kwargs)
                context_created = True
                _log_renderer_info(ctx)
                break
            except Exception as e:
                last_error = e
                # print(f"DEBUG: Attempt failed {kwargs}: {e}")
                continue

        if not context_created:
            print("\n" + "!"*60)
            print("⚠️  HEADLESS GL CONTEXT FAILED")
            print(f"    Error: {last_error}")
            print("    -> Falling back to CPU Compositor.")
            print("!"*60 + "\n")
            return None

        # --- SMART RESOLUTION SELECTION ---
        if is_ascii:
            w = getattr(settings, 'ASCII_WIDTH', 120)
            h = getattr(settings, 'ASCII_HEIGHT', 60)
            width = w if w % 2 == 0 else w + 1
            height = h if h % 2 == 0 else h + 1
            print(f"[DISPLAY] Optimized ASCII Render Target: {width}x{height}")
        else:
            width, height = getattr(settings, "HEADLESS_RES", (640, 480))

        tex = ctx.texture((width, height), components=3)
        fbo = ctx.framebuffer(color_attachments=[tex])
        fbo.use()

        ctx.viewport = (0, 0, width, height)

        renderer.initialize(ctx)

        renderer.set_transform_parameters(
            fs_scale=1.0, fs_offset_x=0.0, fs_offset_y=0.0,
            image_size=state.image_size,
            rotation_angle=0.0, mirror_mode=0
        )
        renderer.set_viewport_size(width, height)

        print(f"[DISPLAY] Headless GL ready: {width}x{height}")
        return HeadlessWindow(ctx, fbo, (width, height))

    # --- PATH B: LOCAL WINDOWED MODE ---
    if not _GLFW_AVAILABLE:
        print("❌ ERROR: GLFW/PyOpenGL not installed. Cannot open local window.")
        sys.exit(1)

    img_w, img_h = state.image_size
    eff_w, eff_h = (img_h, img_w) if state.rotation % 180 == 90 else (img_w, img_h)

    if window is None:
        if not glfw.init():
            print("❌ ERROR: GLFW Init failed.")
            sys.exit(1)

        # --- WINDOW HINTS ---
        if is_pi:
            # Raspberry Pi: Request ES 2.0
            glfw.window_hint(glfw.CLIENT_API, glfw.OPENGL_ES_API)
            glfw.window_hint(glfw.CONTEXT_CREATION_API, glfw.EGL_CONTEXT_API)
            glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 2)
            glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 0)
        else:
            # Desktop (Mac/PC): Request Core 3.3
            glfw.window_hint(glfw.CLIENT_API, glfw.OPENGL_API)
            glfw.window_hint(glfw.CONTEXT_CREATION_API, glfw.NATIVE_CONTEXT_API)
            glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
            glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
            glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)

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
            aspect = eff_w / eff_h
            win_w = 400
            win_h = int(win_w / aspect)
            window = glfw.create_window(win_w, win_h, "Windowed Mode", None, None)

        if not window:
            glfw.terminate()
            print("❌ ERROR: Window creation failed.")
            sys.exit(1)

        glfw.make_context_current(window)
        glfw.swap_interval(1 if VSYNC else 0)

        # --- LOCAL CONTEXT CREATION ---
        # Try relaxed creation first
        try:
            if is_pi:
                # Force ES 2.0 context object for Pi
                ctx = moderngl.create_context(require=200)
            else:
                # Desktop default
                ctx = moderngl.create_context()
        except Exception as e:
            print(f"❌ Local GL Context Failed: {e}")
            glfw.terminate()
            sys.exit(1)

        _log_renderer_info(ctx)
        renderer.initialize(ctx)

        if GAMMA_CORRECTION_ENABLED or ENABLE_SRGB_FRAMEBUFFER:
            glEnable(GL_FRAMEBUFFER_SRGB)

    current_monitor = glfw.get_window_monitor(window)
    if state.fullscreen:
        mon = glfw.get_primary_monitor()
        if CONNECTED_TO_RCA_HDMI:
            fs_w, fs_h = RCA_HDMI_RESOLUTION
        elif LOW_RES_FULLSCREEN:
            fs_w, fs_h = LOW_RES_FULLSCREEN_RESOLUTION
        else:
            best = _largest_mode(mon)
            fs_w, fs_h = best.size.width, best.size.height
        refresh = getattr(best, 'refresh_rate', 60)
        glfw.set_window_monitor(window, mon, 0, 0, fs_w, fs_h, refresh)
    else:
        pass

    fb_w, fb_h = glfw.get_framebuffer_size(window)
    ctx.viewport = (0, 0, fb_w, fb_h)

    if state.fullscreen:
        scale_x = fb_w / eff_w
        scale_y = fb_h / eff_h
        scale = min(scale_x, scale_y)
        offset_x = (fb_w - (eff_w * scale)) / 2.0
        offset_y = (fb_h - (eff_h * scale)) / 2.0
    else:
        scale = fb_w / eff_w
        offset_x = 0
        offset_y = 0

    renderer.set_transform_parameters(
        scale, offset_x, offset_y, state.image_size,
        state.rotation, state.mirror
    )

    mvp = np.array([
        [2.0/fb_w, 0, 0, -1],
        [0, 2.0/fb_h, 0, -1],
        [0, 0, -1, 0],
        [0, 0, 0, 1]
    ], dtype="f4")
    renderer.update_mvp(mvp)

    return window