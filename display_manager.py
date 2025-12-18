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
import re
import shutil
import subprocess

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
        print(f"[DISPLAY] GL Context: {renderer_name}")
        if "llvmpipe" in renderer_name.lower() or "softpipe" in renderer_name.lower():
            print("[DISPLAY] ℹ️  Using Software Rasterizer (Optimized CPU Rendering).")
    except:
        pass

def _check_is_pi():
    """Detect Raspberry Pi to enforce GLES contexts."""
    return _pi_model() is not None


def _pi_model() -> str | None:
    try:
        if os.path.exists("/proc/device-tree/model"):
            with open("/proc/device-tree/model", "r") as m:
                model = m.read().strip().replace("\x00", "")
                if "Raspberry Pi" in model:
                    return model
    except Exception:
        pass
    return None


def _parse_eglinfo_es_version() -> tuple[int, int] | None:
    eglinfo = shutil.which("eglinfo")
    if not eglinfo:
        return None
    try:
        out = subprocess.check_output([eglinfo, "-B"], text=True, stderr=subprocess.STDOUT, timeout=5)
    except Exception:
        return None

    match = re.search(r"OpenGL ES profile version:\s*OpenGL ES\s*([0-9]+)\.([0-9]+)", out)
    if not match:
        return None
    try:
        return int(match.group(1)), int(match.group(2))
    except Exception:
        return None


def _detect_es_version() -> tuple[int, int] | None:
    """
    Detects the available OpenGL ES version by parsing eglinfo.
    Returns (major, minor) or None if undetectable.
    """
    return _parse_eglinfo_es_version()


def _format_gl_version(require_code: int) -> str:
    major = require_code // 100
    minor = (require_code % 100) // 10
    return f"{major}.{minor}"


def _es_require_codes(pi_model: str | None, detected_es: tuple[int, int] | None) -> list[int | None]:
    """
    ModernGL uses integer "require" codes (e.g. 310 for GLES 3.1).
    Prefer the detected GLES version; fall back to conservative Pi defaults.
    """
    override = getattr(settings, "PI_GLES_REQUIRE", None) or os.environ.get("PI_GLES_REQUIRE")
    if override is not None:
        try:
            return [int(override), None]
        except Exception:
            pass

    if detected_es:
        major, minor = detected_es
        require_code = (major * 100) + (minor * 10)
        return [require_code, None]

    if not pi_model:
        return [None]

    # Conservative defaults: prefer ES 3.0 on VC4-era Pis; allow ES 3.1 on Pi 4/5.
    if "Raspberry Pi 4" in pi_model or "Raspberry Pi 5" in pi_model:
        return [310, 300, None]

    return [300, None]

def display_init(state: DisplayState):
    global window, ctx

    pi_model = _pi_model()
    is_pi = pi_model is not None
    detected_es = _detect_es_version()
    require_codes = _es_require_codes(pi_model if is_pi else None, detected_es)

    if is_pi:
        print(f"[DISPLAY] Hardware: {pi_model}")
    if detected_es:
        print(f"[DISPLAY] GLES capability detected: {detected_es[0]}.{detected_es[1]} (via eglinfo)")
    elif is_pi:
        print("[DISPLAY] GLES capability not detected via eglinfo; using Pi defaults.")

    # --- PATH A: SERVER / HEADLESS MODE ---
    is_server = settings.SERVER_MODE
    is_ascii = getattr(settings, 'ASCII_MODE', False)

    if is_server or is_ascii:
        use_gl = getattr(settings, "HEADLESS_USE_GL", False)

        if not use_gl:
            print("❌ CONFIG ERROR: HEADLESS_USE_GL must be True in settings.py for performance.")
            sys.exit(1)

        # --- BACKEND WATERFALL STRATEGY ---
        preferred = getattr(settings, "HEADLESS_BACKEND", None)
        backends_to_try = []

        if preferred: backends_to_try.append(preferred)
        if 'egl' not in backends_to_try: backends_to_try.append('egl')
        if 'osmesa' not in backends_to_try: backends_to_try.append('osmesa')
        if None not in backends_to_try: backends_to_try.append(None)

        context_created = False
        last_error = None

        print("[DISPLAY] Initializing Headless GL...")

        for backend in backends_to_try:
            for require_code in require_codes:
                try:
                    create_kwargs = {"standalone": True}
                    if backend:
                        create_kwargs["backend"] = backend
                    if require_code is not None:
                        create_kwargs["require"] = require_code

                    ctx = moderngl.create_context(**create_kwargs)
                    context_created = True
                    if require_code is not None:
                        print(f"[DISPLAY] Headless GL: Requested GLES {_format_gl_version(require_code)}")
                    _log_renderer_info(ctx)
                    break
                except Exception as e:
                    last_error = e
                    continue
            if context_created:
                break

        if not context_created:
            print("\n" + "!"*60)
            print("❌ HEADLESS GL CONTEXT FAILED")
            print("All requested backends failed to initialize.")
            print(f"Last Error: {last_error}")
            print("Tip: If on Pi, ensure 'libgles2-mesa-dev' is installed.")
            print("!"*60 + "\n")
            sys.exit(1)

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

        # Apply Pi Hints for Window creation
        if is_pi:
            # Pi 2/3 (VC4) generally cannot do GLES 3.1; try a small fallback list (including None).
            window_created = False
            last_error = None
            for require_code in require_codes:
                try:
                    glfw.default_window_hints()
                    glfw.window_hint(glfw.CLIENT_API, glfw.OPENGL_ES_API)
                    glfw.window_hint(glfw.CONTEXT_CREATION_API, glfw.EGL_CONTEXT_API)
                    if require_code is not None:
                        glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, require_code // 100)
                        glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, (require_code % 100) // 10)

                    if state.fullscreen:
                        mon = glfw.get_primary_monitor()
                        if CONNECTED_TO_RCA_HDMI:
                            fs_w, fs_h = RCA_HDMI_RESOLUTION
                        elif LOW_RES_FULLSCREEN:
                            fs_w, fs_h = LOW_RES_FULLSCREEN_RESOLUTION
                        else:
                            best = _largest_mode(mon)
                            fs_w, fs_h = best.size.width, best.size.height
                        glfw.window_hint(glfw.AUTO_ICONIFY, glfw.FALSE)
                        window = glfw.create_window(fs_w, fs_h, "Fullscreen", mon, None)
                    else:
                        aspect = eff_w / eff_h
                        win_w = 400
                        win_h = int(win_w / aspect)
                        window = glfw.create_window(win_w, win_h, "Windowed Mode", None, None)

                    if window:
                        window_created = True
                        if require_code is None:
                            print("[DISPLAY] Local Window: Requested GLES (no version hint)")
                        else:
                            print(f"[DISPLAY] Local Window: Requested GLES {_format_gl_version(require_code)}")
                        break
                except Exception as e:
                    last_error = e
                    window = None
                    continue

            if not window_created:
                print(f"❌ ERROR: Pi window creation failed: {last_error}")
                glfw.terminate()
                sys.exit(1)
        else:
            glfw.window_hint(glfw.CLIENT_API, glfw.OPENGL_API)
            glfw.window_hint(glfw.CONTEXT_CREATION_API, glfw.NATIVE_CONTEXT_API)
            glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
            glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
            glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)

        if not is_pi:
            if state.fullscreen:
                mon = glfw.get_primary_monitor()
                if CONNECTED_TO_RCA_HDMI:
                    fs_w, fs_h = RCA_HDMI_RESOLUTION
                elif LOW_RES_FULLSCREEN:
                    fs_w, fs_h = LOW_RES_FULLSCREEN_RESOLUTION
                else:
                    best = _largest_mode(mon)
                    fs_w, fs_h = best.size.width, best.size.height
                glfw.window_hint(glfw.AUTO_ICONIFY, glfw.FALSE)
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

        # Apply Pi Hints for Context creation
        if is_pi:
            last_error = None
            for require_code in require_codes:
                try:
                    if require_code is None:
                        ctx = moderngl.create_context()
                        print("[DISPLAY] Local GL: Requested default context")
                    else:
                        ctx = moderngl.create_context(require=require_code)
                        print(f"[DISPLAY] Local GL: Requested GLES {_format_gl_version(require_code)}")
                    break
                except Exception as e:
                    last_error = e
                    ctx = None
                    continue
            if ctx is None:
                print(f"❌ ERROR: Failed to create Pi GL context: {last_error}")
                sys.exit(1)
        else:
            ctx = moderngl.create_context()

        _log_renderer_info(ctx)
        renderer.initialize(ctx)

        if GAMMA_CORRECTION_ENABLED or ENABLE_SRGB_FRAMEBUFFER:
            glEnable(GL_FRAMEBUFFER_SRGB)

        if glfw and window is not None:
            glfw.set_input_mode(window, glfw.CURSOR, glfw.CURSOR_HIDDEN)

    current_monitor = glfw.get_window_monitor(window)
    if state.fullscreen:
        mon = glfw.get_primary_monitor()
        if CONNECTED_TO_RCA_HDMI:
            fs_w, fs_h = RCA_HDMI_RESOLUTION
            refresh = 60  # Default refresh rate
        elif LOW_RES_FULLSCREEN:
            fs_w, fs_h = LOW_RES_FULLSCREEN_RESOLUTION
            refresh = 60  # Default refresh rate
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
