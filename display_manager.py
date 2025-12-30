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
import json
from pathlib import Path
from typing import Optional

# --- Conditional Imports ---
_GLFW_AVAILABLE = False
try:
    import glfw
    from OpenGL.GL import (
        glEnable, glGetError, glGetString, glViewport,
        GL_FRAMEBUFFER_SRGB, GL_INVALID_ENUM, GL_VERSION, GL_RENDERER,
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
    AUTO_OPTIMIZE_DISPLAY_RESOLUTION, RESTORE_DISPLAY_ON_EXIT,
)

window = None
ctx = None

# Backend usage tracking
_BACKEND_USAGE_LOG: Optional[Path] = None
_BACKEND_USAGE_DATA = {
    'headless_legacy_used': False,
    'headless_backends_tried': [],
    'headless_backend_success': None,
    'headless_fallback_used': False,
    'local_window_backend': None,
    'local_legacy_used': False,
    'session_type': None,
    'hardware_type': None,  # Deprecated: use 'driver' instead
    'driver': None,
}

def _is_wayland_session() -> bool:
    return bool(os.environ.get("WAYLAND_DISPLAY") or os.environ.get("XDG_SESSION_TYPE") == "wayland")


def _session_label() -> str:
    if _is_wayland_session():
        return "wayland"
    if os.environ.get("DISPLAY"):
        return "x11"
    return "unknown"

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
        self._cleanup = None

    def use(self) -> None:
        self.fbo.use()

    def swap_buffers(self):
        pass

    def close(self) -> None:
        if self._cleanup:
            try:
                self._cleanup()
            finally:
                self._cleanup = None


class LegacyHeadlessFBO:
    def __init__(self, fbo_id: int, tex_id: int, size: tuple[int, int]):
        self.fbo_id = fbo_id
        self.tex_id = tex_id
        self.size = size

    def use(self) -> None:
        from OpenGL import GL as gl  # type: ignore
        gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, self.fbo_id)

    def read(self, components: int = 3) -> bytes:
        if components != 3:
            raise ValueError("LegacyHeadlessFBO only supports components=3")
        from OpenGL import GL as gl  # type: ignore
        w, h = self.size
        gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, self.fbo_id)
        gl.glPixelStorei(gl.GL_PACK_ALIGNMENT, 1)
        return gl.glReadPixels(0, 0, w, h, gl.GL_RGB, gl.GL_UNSIGNED_BYTE)


class LegacyHeadlessWindow:
    def __init__(self, glfw_window, fbo: LegacyHeadlessFBO):
        self._glfw_window = glfw_window
        self.fbo = fbo
        self.size = fbo.size

    def use(self) -> None:
        from OpenGL import GL as gl  # type: ignore
        # CRITICAL: Make GLFW context current for PyOpenGL to work
        if glfw and self._glfw_window:
            glfw.make_context_current(self._glfw_window)
        self.fbo.use()
        w, h = self.size
        gl.glViewport(0, 0, w, h)

    def swap_buffers(self):
        pass

    def close(self) -> None:
        try:
            if glfw and self._glfw_window:
                glfw.destroy_window(self._glfw_window)
        finally:
            if glfw:
                glfw.terminate()
            self._glfw_window = None

def _current_mode(monitor):
    """
    Get the current video mode of the monitor.
    Returns the resolution currently in use by the display.
    """
    if not glfw:
        return None
    try:
        return glfw.get_video_mode(monitor)
    except Exception:
        return None


def _largest_mode(monitor):
    """
    Get the largest available video mode for the monitor.
    Used as fallback only if current mode is unavailable.
    """
    if not glfw:
        return None
    try:
        modes = glfw.get_video_modes(monitor)
        if not modes:
            return None
        return max(modes, key=lambda m: m.size.width * m.size.height)
    except Exception:
        return None


def _preferred_fullscreen_mode(monitor):
    """
    Get the preferred fullscreen video mode.
    Always prefers the CURRENT resolution to avoid display resolution changes.
    Falls back to largest mode only if current mode is unavailable.
    
    On Linux (X11/Wayland), this ensures we use the current display resolution,
    preventing unwanted resolution changes on low-end systems.
    """
    mode = _current_mode(monitor)
    if mode:
        return mode
    # Fallback: only used if current mode unavailable (should be rare)
    return _largest_mode(monitor)


# Global storage for original display resolution (for restoration)
_original_display_resolution = None
_original_display_output = None


def _find_optimal_display_mode(image_size: tuple[int, int], current_mode) -> Optional[object]:
    """
    Find the optimal display mode that matches the image size.
    
    Args:
        image_size: (width, height) of the source image
        current_mode: Current display mode object from GLFW
    
    Returns:
        Optimal mode object or None if no suitable mode found
    """
    if not glfw or current_mode is None:
        return None
    
    img_w, img_h = image_size
    current_w = current_mode.size.width
    current_h = current_mode.size.height
    current_aspect = current_w / current_h if current_h > 0 else 1.0
    
    # Only optimize if image is smaller than current display
    if img_w >= current_w or img_h >= current_h:
        return None
    
    # Get all available modes
    monitor = glfw.get_primary_monitor()
    if monitor is None:
        return None
    
    try:
        all_modes = glfw.get_video_modes(monitor)
    except Exception:
        return None
    
    if not all_modes:
        return None
    
    # Filter modes by aspect ratio (within tolerance)
    aspect_tolerance = 0.01
    matching_modes = []
    
    for mode in all_modes:
        mode_w = mode.size.width
        mode_h = mode.size.height
        mode_aspect = mode_w / mode_h if mode_h > 0 else 1.0
        
        # Check aspect ratio match
        if abs(mode_aspect - current_aspect) > aspect_tolerance:
            continue
        
        # Check that mode fits image and is smaller than current
        if (mode_w >= img_w and mode_h >= img_h and 
            mode_w <= current_w and mode_h <= current_h):
            matching_modes.append(mode)
    
    if not matching_modes:
        return None
    
    # Select mode with smallest area (closest to image size)
    best_mode = min(matching_modes, key=lambda m: m.size.width * m.size.height)
    return best_mode


def _get_xrandr_output() -> Optional[str]:
    """Detect the primary X11 display output name using xrandr."""
    try:
        result = subprocess.run(
            ["xrandr"], 
            capture_output=True, 
            text=True, 
            timeout=5,
            check=False
        )
        if result.returncode != 0:
            return None
        
        # Find first connected output
        for line in result.stdout.split('\n'):
            if ' connected' in line:
                # Extract output name (first word)
                parts = line.split()
                if parts:
                    return parts[0]
    except (subprocess.TimeoutExpired, FileNotFoundError, Exception):
        pass
    return None


def _change_display_resolution(mode) -> bool:
    """
    Change display resolution using xrandr (X11) or DRM/KMS (Wayland/headless).
    
    Args:
        mode: GLFW video mode object with new resolution
    
    Returns:
        True if successful, False otherwise
    """
    global _original_display_resolution, _original_display_output
    
    if mode is None:
        return False
    
    new_w = mode.size.width
    new_h = mode.size.height
    
    # Store original resolution for restoration
    if _original_display_resolution is None:
        current_mode = _current_mode(glfw.get_primary_monitor())
        if current_mode:
            _original_display_resolution = (current_mode.size.width, current_mode.size.height)
            _original_display_output = _get_xrandr_output()
    
    # Try xrandr first (X11)
    output = _get_xrandr_output()
    if output:
        try:
            # Change resolution using xrandr
            cmd = ["xrandr", "--output", output, "--mode", f"{new_w}x{new_h}"]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=5, check=False)
            if result.returncode == 0:
                print(f"[DISPLAY] Changed display resolution to {new_w}x{new_h} using xrandr")
                return True
            else:
                print(f"[DISPLAY] ⚠️  xrandr failed: {result.stderr}")
        except (subprocess.TimeoutExpired, FileNotFoundError, Exception) as e:
            print(f"[DISPLAY] ⚠️  Failed to change resolution via xrandr: {e}")
    
    # TODO: Add DRM/KMS support for Wayland/headless
    # For now, xrandr is the primary method (most common on Pi with desktop)
    
    return False


def _restore_display_resolution() -> None:
    """Restore original display resolution if it was changed."""
    global _original_display_resolution, _original_display_output
    
    if _original_display_resolution is None or _original_display_output is None:
        return
    
    if not getattr(settings, 'RESTORE_DISPLAY_ON_EXIT', True):
        return
    
    orig_w, orig_h = _original_display_resolution
    output = _original_display_output
    
    try:
        cmd = ["xrandr", "--output", output, "--mode", f"{orig_w}x{orig_h}"]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=5, check=False)
        if result.returncode == 0:
            print(f"[DISPLAY] Restored display resolution to {orig_w}x{orig_h}")
        else:
            print(f"[DISPLAY] ⚠️  Failed to restore resolution: {result.stderr}")
    except (subprocess.TimeoutExpired, FileNotFoundError, Exception) as e:
        print(f"[DISPLAY] ⚠️  Error restoring resolution: {e}")

def _log_renderer_info(ctx):
    try:
        info = ctx.info
        renderer_name = info.get('GL_RENDERER', 'Unknown')
        print(f"[DISPLAY] GL Context: {renderer_name}")
        
        # Detect and store driver info
        driver = _detect_gl_driver(ctx)
        if driver:
            _BACKEND_USAGE_DATA['driver'] = driver
            print(f"[DISPLAY] Driver detected: {driver}")
        else:
            _BACKEND_USAGE_DATA['driver'] = 'unknown'
        
        if "llvmpipe" in renderer_name.lower() or "softpipe" in renderer_name.lower():
            print("[DISPLAY] ℹ️  Using Software Rasterizer (Optimized CPU Rendering).")
    except (AttributeError, KeyError, TypeError) as e:
        # GL context info may not be available on all platforms
        pass

# Cached EGL availability detection
_egl_available_cache: Optional[bool] = None


def _detect_egl_available() -> bool:
    """
    Test if EGL backend is available by attempting to create a test context.
    Caches result to avoid repeated tests.
    """
    global _egl_available_cache
    if _egl_available_cache is not None:
        return _egl_available_cache
    
    try:
        # Try to create a minimal ModernGL context with EGL backend
        test_ctx = moderngl.create_context(standalone=True, backend="egl")
        test_ctx.release()
        _egl_available_cache = True
        return True
    except Exception:
        _egl_available_cache = False
        return False


def _detect_gl_driver(gl_context=None) -> str | None:
    """
    Query GL_RENDERER to identify driver type.
    Should be called with an existing GL context (ModernGL or PyOpenGL).
    
    Returns driver name (vc4, v3d, panfrost, llvmpipe, softpipe, iris, anv, radeonsi, etc.)
    or None if detection fails.
    """
    try:
        # If we have a ModernGL context, try to get info
        if gl_context is not None and hasattr(gl_context, 'info'):
            renderer_name = gl_context.info.get('GL_RENDERER', '')
            if renderer_name:
                renderer_lower = renderer_name.lower()
                # Check for known drivers
                if 'vc4' in renderer_lower:
                    return 'vc4'
                elif 'v3d' in renderer_lower:
                    return 'v3d'
                elif 'panfrost' in renderer_lower:
                    return 'panfrost'
                elif 'llvmpipe' in renderer_lower:
                    return 'llvmpipe'
                elif 'softpipe' in renderer_lower:
                    return 'softpipe'
                elif 'iris' in renderer_lower:
                    return 'iris'
                elif 'anv' in renderer_lower:
                    return 'anv'
                elif 'radeonsi' in renderer_lower or 'radeon' in renderer_lower:
                    return 'radeonsi'
                elif 'nouveau' in renderer_lower:
                    return 'nouveau'
                elif 'nvidia' in renderer_lower:
                    return 'nvidia'
        
        # Try PyOpenGL path if available (context must be current)
        if _GLFW_AVAILABLE:
            try:
                from OpenGL import GL as gl
                renderer_bytes = gl.glGetString(gl.GL_RENDERER)
                if renderer_bytes:
                    renderer_str = renderer_bytes.decode("utf-8", errors="replace").lower()
                    if 'vc4' in renderer_str:
                        return 'vc4'
                    elif 'v3d' in renderer_str:
                        return 'v3d'
                    elif 'panfrost' in renderer_str:
                        return 'panfrost'
                    elif 'llvmpipe' in renderer_str:
                        return 'llvmpipe'
                    elif 'softpipe' in renderer_str:
                        return 'softpipe'
                    elif 'iris' in renderer_str:
                        return 'iris'
                    elif 'anv' in renderer_str:
                        return 'anv'
                    elif 'radeonsi' in renderer_str or 'radeon' in renderer_str:
                        return 'radeonsi'
                    elif 'nouveau' in renderer_str:
                        return 'nouveau'
                    elif 'nvidia' in renderer_str:
                        return 'nvidia'
            except Exception:
                pass
    except Exception:
        pass
    
    return None


def _should_use_egl_backend(driver: str | None = None, detected_es: tuple[int, int] | None = None) -> bool:
    """
    Determine if EGL backend should be used based on:
    - EGL availability
    - Driver type (prefer EGL for vc4, v3d, panfrost, and on Wayland)
    - GLES version detection
    """
    # Check EGL availability first
    if not _detect_egl_available():
        return False
    
    # Prefer EGL for GLES-only drivers
    if driver in ('vc4', 'v3d', 'panfrost'):
        return True
    
    # Prefer EGL on Wayland
    if _is_wayland_session():
        return True
    
    # If GLES is detected, EGL is typically needed
    if detected_es is not None:
        return True
    
    return False


def _should_use_gles_api(driver: str | None = None, detected_es: tuple[int, int] | None = None) -> bool:
    """
    Determine if GLES API should be used based on:
    - Detected GLES version
    - Driver type (vc4, v3d, panfrost are GLES-only)
    """
    # GLES-only drivers require GLES API
    if driver in ('vc4', 'v3d', 'panfrost'):
        return True
    
    # If GLES version is detected, use GLES API
    if detected_es is not None:
        return True
    
    return False


def _parse_eglinfo_es_version() -> tuple[int, int] | None:
    eglinfo = shutil.which("eglinfo")
    if not eglinfo:
        return None
    try:
        # eglinfo can exit non-zero if *any* platform probe fails (e.g. X11), even if it prints valid ES info.
        res = subprocess.run([eglinfo, "-B"], text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=10)
        out = res.stdout or ""
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


def _es_require_codes(detected_es: tuple[int, int] | None) -> list[int | None]:
    """
    ModernGL uses integer "require" codes (e.g. 310 for GLES 3.1).
    Try higher versions first, then fall back.
    """
    # Support both new and legacy environment variable names for backward compatibility
    override = (getattr(settings, "GLES_REQUIRE_OVERRIDE", None) or 
                os.environ.get("GLES_REQUIRE_OVERRIDE") or 
                os.environ.get("PI_GLES_REQUIRE"))  # Legacy support
    if override is not None:
        try:
            return [int(override), None]
        except Exception:
            pass

    if detected_es:
        major, minor = detected_es
        detected_code = (major * 100) + (minor * 10)
        # Start with detected version, but include fallbacks in case detection is optimistic
        # or actual capabilities are lower than reported
        if detected_code >= 310:
            attempts = [310, 300, 200, None]
        elif detected_code >= 300:
            attempts = [300, 200, None]
        elif detected_code >= 200:
            attempts = [200, None]
        else:
            attempts = [detected_code, None]
    else:
        # Unknown capability: probe high→low.
        attempts = [310, 300, 200, None]
    # Remove dupes while preserving order
    seen = set()
    unique_attempts = []
    for code in attempts:
        if code in seen:
            continue
        seen.add(code)
        unique_attempts.append(code)
    return unique_attempts


def _try_hidden_glfw_headless(require_codes: list[int | None], use_egl: bool, use_gles: bool, size: tuple[int, int]):
    """
    Fallback: create a tiny invisible GLFW window to get a GL context on Wayland/X.
    Useful when standalone EGL fails (e.g., VC4 + Wayland).
    """
    if not _GLFW_AVAILABLE:
        return None, None

    if not glfw.init():
        return None, None

    window = None
    ctx_local = None
    last_error = None

    backend_kwargs = {"backend": "egl"} if use_egl else {}

    for require_code in require_codes:
        attempt_version = _format_gl_version(require_code) if require_code is not None else "default"
        print(f"[DISPLAY] Hidden GLFW headless attempt: GLES={attempt_version}")
        try:
            glfw.default_window_hints()
            glfw.window_hint(glfw.VISIBLE, glfw.FALSE)
            glfw.window_hint(glfw.AUTO_ICONIFY, glfw.FALSE)
            if use_gles:
                glfw.window_hint(glfw.CLIENT_API, glfw.OPENGL_ES_API)
                glfw.window_hint(glfw.CONTEXT_CREATION_API, glfw.EGL_CONTEXT_API)
                if require_code is not None:
                    glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, require_code // 100)
                    glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, (require_code % 100) // 10)
            else:
                glfw.window_hint(glfw.CLIENT_API, glfw.OPENGL_API)
                if use_egl or _is_wayland_session():
                    glfw.window_hint(glfw.CONTEXT_CREATION_API, glfw.EGL_CONTEXT_API)
                else:
                    glfw.window_hint(glfw.CONTEXT_CREATION_API, glfw.NATIVE_CONTEXT_API)

            w, h = size
            window = glfw.create_window(w, h, "Headless GL", None, None)
            if not window:
                raise RuntimeError("GLFW window creation failed")

            glfw.make_context_current(window)
            glfw.swap_interval(0)

            if require_code is None:
                ctx_local = moderngl.create_context(**backend_kwargs)
            else:
                ctx_local = moderngl.create_context(require=require_code, **backend_kwargs)

            # ModernGL backend requires GL3.x / GLES3+ features; reject legacy contexts here.
            if getattr(ctx_local, "version_code", 0) < 300:
                raise RuntimeError(f"ModernGL unsupported on legacy context (version_code={ctx_local.version_code})")

            print(f"[DISPLAY] Hidden GLFW headless: created GLES={attempt_version}")
            break
        except Exception as e:
            last_error = e
            ctx_local = None
            if window:
                glfw.destroy_window(window)
                window = None
            print(f"[DISPLAY] Hidden GLFW headless failed (GLES={attempt_version}): {e}")
            continue

    if ctx_local is None:
        glfw.terminate()
    return ctx_local, window


def _try_hidden_glfw_headless_legacy(require_codes: list[int | None], use_gles: bool, size: tuple[int, int]):
    """
    Creates a tiny hidden GLFW window and uses the raw OpenGL context (PyOpenGL path).
    Returns a LegacyHeadlessWindow if a legacy (GL2.x/GLES2) context is created.
    """
    if not _GLFW_AVAILABLE:
        return None
    if not glfw.init():
        return None

    window = None
    last_error = None

    for require_code in require_codes:
        attempt_version = _format_gl_version(require_code) if require_code is not None else "default"
        print(f"[DISPLAY] Hidden GLFW legacy attempt: GLES={attempt_version}")
        try:
            glfw.default_window_hints()
            glfw.window_hint(glfw.VISIBLE, glfw.FALSE)
            glfw.window_hint(glfw.AUTO_ICONIFY, glfw.FALSE)
            if use_gles:
                glfw.window_hint(glfw.CLIENT_API, glfw.OPENGL_ES_API)
                glfw.window_hint(glfw.CONTEXT_CREATION_API, glfw.EGL_CONTEXT_API)
                if require_code is not None:
                    glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, require_code // 100)
                    glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, (require_code % 100) // 10)
            else:
                glfw.window_hint(glfw.CLIENT_API, glfw.OPENGL_API)
                if _is_wayland_session():
                    glfw.window_hint(glfw.CONTEXT_CREATION_API, glfw.EGL_CONTEXT_API)
                else:
                    glfw.window_hint(glfw.CONTEXT_CREATION_API, glfw.NATIVE_CONTEXT_API)

            w, h = size
            window = glfw.create_window(w, h, "Headless Legacy GL", None, None)
            if not window:
                raise RuntimeError("GLFW window creation failed")

            glfw.make_context_current(window)
            glfw.swap_interval(0)

            # Determine if this is a legacy context (< GLES 3.0 / GL 3.0).
            version_str = None
            version_code = None
            try:
                from OpenGL import GL as gl  # type: ignore
                vb = gl.glGetString(gl.GL_VERSION)
                version_str = vb.decode("utf-8", errors="replace") if vb else None
                if version_str:
                    import re
                    m = re.search(r"OpenGL ES\s*([0-9]+)\.([0-9]+)", version_str)
                    if not m:
                        m = re.search(r"^\s*([0-9]+)\.([0-9]+)", version_str)
                    if m:
                        major, minor = int(m.group(1)), int(m.group(2))
                        version_code = (major * 100) + (minor * 10)
            except Exception:
                pass

            if version_code is None or version_code >= 300:
                raise RuntimeError(f"Not a legacy context (GL_VERSION={version_str})")

            # Create an FBO to render/capture without swapping buffers.
            from OpenGL import GL as gl  # type: ignore
            tex_id = gl.glGenTextures(1)
            gl.glBindTexture(gl.GL_TEXTURE_2D, tex_id)
            gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MIN_FILTER, gl.GL_LINEAR)
            gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MAG_FILTER, gl.GL_LINEAR)
            gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_WRAP_S, gl.GL_CLAMP_TO_EDGE)
            gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_WRAP_T, gl.GL_CLAMP_TO_EDGE)
            gl.glTexImage2D(gl.GL_TEXTURE_2D, 0, gl.GL_RGB, w, h, 0, gl.GL_RGB, gl.GL_UNSIGNED_BYTE, None)

            fbo_id = gl.glGenFramebuffers(1)
            gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, fbo_id)
            gl.glFramebufferTexture2D(gl.GL_FRAMEBUFFER, gl.GL_COLOR_ATTACHMENT0, gl.GL_TEXTURE_2D, tex_id, 0)
            status = gl.glCheckFramebufferStatus(gl.GL_FRAMEBUFFER)
            if status != gl.GL_FRAMEBUFFER_COMPLETE:
                raise RuntimeError(f"Legacy FBO incomplete (status={hex(int(status))})")

            fbo = LegacyHeadlessFBO(fbo_id, tex_id, (w, h))
            print(f"[DISPLAY] Hidden GLFW legacy: created (GL_VERSION={version_str})")
            return LegacyHeadlessWindow(window, fbo)

        except Exception as e:
            last_error = e
            if window:
                glfw.destroy_window(window)
                window = None
            print(f"[DISPLAY] Hidden GLFW legacy failed (GLES={attempt_version}): {e}")
            continue

    print(f"[DISPLAY] Hidden GLFW legacy: all attempts failed: {last_error}")
    glfw.terminate()
    return None

def _log_backend_usage():
    """Log backend usage to file for analysis."""
    global _BACKEND_USAGE_LOG, _BACKEND_USAGE_DATA
    if _BACKEND_USAGE_LOG is None:
        log_path = Path.home() / ".videointerleaving_backend_usage.json"
        _BACKEND_USAGE_LOG = log_path
    try:
        # Read existing data if any
        if _BACKEND_USAGE_LOG.exists():
            with open(_BACKEND_USAGE_LOG, 'r') as f:
                existing = json.load(f)
                # Merge with current session data
                for key in _BACKEND_USAGE_DATA:
                    if key not in existing:
                        existing[key] = _BACKEND_USAGE_DATA[key]
                    elif isinstance(existing[key], list) and isinstance(_BACKEND_USAGE_DATA[key], list):
                        existing[key].extend(_BACKEND_USAGE_DATA[key])
                _BACKEND_USAGE_DATA = existing
        # Write updated data
        with open(_BACKEND_USAGE_LOG, 'w') as f:
            json.dump(_BACKEND_USAGE_DATA, f, indent=2)
    except Exception as e:
        # Don't fail if logging fails
        pass

def display_init(state: DisplayState):
    global window, ctx

    is_wayland = _is_wayland_session()
    session_label = _session_label()
    print(f"[DISPLAY] Session: {session_label}")
    
    # Track session info
    _BACKEND_USAGE_DATA['session_type'] = session_label
    
    detected_es = _detect_es_version()
    require_codes = _es_require_codes(detected_es)
    
    # Determine backend and API preferences based on driver/GL capabilities
    # Note: Driver detection will happen after context creation, so we use
    # detected GLES version and EGL availability for initial decisions
    use_egl_backend = _should_use_egl_backend(None, detected_es)
    use_gles_api = _should_use_gles_api(None, detected_es)

    if detected_es:
        major, minor = detected_es
        detected_code = (major * 100) + (minor * 10)
        print(f"[DISPLAY] GLES capability detected: {major}.{minor} (via eglinfo) - will use GLES {major}.{minor}")
    else:
        print("[DISPLAY] GLES capability not detected via eglinfo; trying GLES versions high→low.")

    # --- PATH A: SERVER / HEADLESS MODE ---
    is_server = settings.SERVER_MODE
    is_ascii = getattr(settings, 'ASCII_MODE', False)

    if is_server or is_ascii:
        use_gl = getattr(settings, "HEADLESS_USE_GL", False)

        if not use_gl:
            print("❌ CONFIG ERROR: HEADLESS_USE_GL must be True in settings.py for performance.")
            sys.exit(1)

        # Determine render target size up front (used by both ModernGL and legacy headless paths)
        if is_ascii:
            w = getattr(settings, 'ASCII_WIDTH', 120)
            h = getattr(settings, 'ASCII_HEIGHT', 60)
            width = w if w % 2 == 0 else w + 1
            height = h if h % 2 == 0 else h + 1
        else:
            # Always use HEADLESS_RES from settings - do not calculate from image size
            headless_res = getattr(settings, "HEADLESS_RES", (640, 480))
            # Debug: Check if HEADLESS_RES is actually a tuple
            if not isinstance(headless_res, tuple) or len(headless_res) != 2:
                print(f"[DISPLAY] ⚠️ WARNING: HEADLESS_RES is not a valid tuple: {headless_res} (type: {type(headless_res)})")
                headless_res = (640, 480)
            width, height = headless_res
            # Debug: Print the actual value from settings
            actual_settings_value = getattr(settings, "HEADLESS_RES", None)
            print(f"[DISPLAY] Using HEADLESS_RES from settings.py: {width}x{height} (raw value: {actual_settings_value})")

        force_legacy = os.environ.get("FORCE_LEGACY_GL") in {"1", "true", "TRUE", "yes", "YES"}
        prefer_legacy_headless = force_legacy or (detected_es is not None and (detected_es[0], detected_es[1]) < (3, 0))
        if prefer_legacy_headless:
            print("[DISPLAY] Headless renderer: legacy (detected GLES < 3.0 or FORCE_LEGACY_GL=1)")
            _BACKEND_USAGE_DATA['headless_legacy_used'] = True
            legacy_window = _try_hidden_glfw_headless_legacy(require_codes, use_gles_api, (width, height))
            if legacy_window is None:
                print("❌ HEADLESS LEGACY GL CONTEXT FAILED. Exiting.")
                sys.exit(1)
            renderer.initialize_legacy()
            renderer.set_transform_parameters(
                fs_scale=1.0, fs_offset_x=0.0, fs_offset_y=0.0,
                image_size=state.image_size,
                rotation_angle=0.0, mirror_mode=0
            )
            renderer.set_viewport_size(width, height)
            print(f"[DISPLAY] Headless legacy GL ready: {width}x{height}")
            _log_backend_usage()
            return legacy_window

        # --- BACKEND WATERFALL STRATEGY ---
        preferred = getattr(settings, "HEADLESS_BACKEND", None)
        backends_to_try = []

        if preferred: backends_to_try.append(preferred)
        if 'egl' not in backends_to_try: backends_to_try.append('egl')
        if 'osmesa' not in backends_to_try: backends_to_try.append('osmesa')
        if None not in backends_to_try: backends_to_try.append(None)

        context_created = False
        last_error = None
        hidden_window = None

        print("[DISPLAY] Initializing Headless GL...")

        for backend in backends_to_try:
            for require_code in require_codes:
                attempt_backend = backend or "auto"
                attempt_version = _format_gl_version(require_code) if require_code is not None else "default"
                attempt_str = f"{attempt_backend}:{attempt_version}"
                print(f"[DISPLAY] Headless attempt: backend={attempt_backend}, GLES={attempt_version}")
                _BACKEND_USAGE_DATA['headless_backends_tried'].append(attempt_str)
                try:
                    create_kwargs = {"standalone": True}
                    if backend:
                        create_kwargs["backend"] = backend
                    if require_code is not None:
                        create_kwargs["require"] = require_code

                    ctx = moderngl.create_context(**create_kwargs)
                    context_created = True
                    _BACKEND_USAGE_DATA['headless_backend_success'] = attempt_str
                    if require_code is not None:
                        print(f"[DISPLAY] ✅ Headless GL: Successfully created GLES {_format_gl_version(require_code)} context")
                    else:
                        print(f"[DISPLAY] ✅ Headless GL: Successfully created default context")
                    _log_renderer_info(ctx)
                    break  # Success - exit inner loop
                except Exception as e:
                    last_error = e
                    print(f"[DISPLAY] Headless attempt failed ({attempt_backend}, GLES={attempt_version}): {e}")
                    continue  # Try next version
            if context_created:
                break  # Success - exit outer loop

        if not context_created:
            print("\n" + "!"*60)
            print("❌ HEADLESS GL CONTEXT FAILED")
            print("All requested backends failed to initialize.")
            print(f"Last Error: {last_error}")
            print("Tip: If on Pi, ensure 'libgles2-mesa-dev' is installed.")
            print("!"*60 + "\n")
            # Wayland/VC4 often needs a real surface; try a hidden GLFW window as a last resort.
            _BACKEND_USAGE_DATA['headless_fallback_used'] = True
            ctx_hidden, win_hidden = _try_hidden_glfw_headless(require_codes, use_egl_backend, use_gles_api, (width, height))
            if ctx_hidden is not None:
                ctx = ctx_hidden
                hidden_window = win_hidden
                _BACKEND_USAGE_DATA['headless_backend_success'] = "hidden_glfw"
            else:
                legacy_window = _try_hidden_glfw_headless_legacy(require_codes, use_gles_api, (width, height))
                if legacy_window is None:
                    print("❌ HEADLESS GL CONTEXT FAILED (after hidden GLFW fallback). Exiting.")
                    sys.exit(1)
                # Legacy GPU headless: use PyOpenGL backend and render into legacy FBO
                _BACKEND_USAGE_DATA['headless_legacy_used'] = True
                renderer.initialize_legacy()
                renderer.set_transform_parameters(
                    fs_scale=1.0, fs_offset_x=0.0, fs_offset_y=0.0,
                    image_size=state.image_size,
                    rotation_angle=0.0, mirror_mode=0
                )
                renderer.set_viewport_size(width, height)
                print(f"[DISPLAY] Headless legacy GL ready: {width}x{height}")
                _log_backend_usage()
                return legacy_window

        if is_ascii:
            print(f"[DISPLAY] Optimized ASCII Render Target: {width}x{height}")

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
        hw = HeadlessWindow(ctx, fbo, (width, height))
        if hidden_window is not None and glfw:
            def _cleanup_hidden():
                try:
                    glfw.destroy_window(hidden_window)
                finally:
                    glfw.terminate()
            hw._cleanup = _cleanup_hidden
        _log_backend_usage()
        return hw

    # --- PATH B: LOCAL WINDOWED MODE ---
    if not _GLFW_AVAILABLE:
        print("❌ ERROR: GLFW/PyOpenGL not installed. Cannot open local window.")
        sys.exit(1)

    img_w, img_h = state.image_size
    eff_w, eff_h = (img_h, img_w) if state.rotation % 180 == 90 else (img_w, img_h)

    # If window exists and we're just updating (e.g., fullscreen toggle), handle it specially for Wayland
    if window is not None and is_wayland:
        # Wayland: Just update window properties without recreating
        if state.fullscreen:
            mon = glfw.get_primary_monitor()
            if CONNECTED_TO_RCA_HDMI:
                fs_w, fs_h = RCA_HDMI_RESOLUTION
            elif LOW_RES_FULLSCREEN:
                fs_w, fs_h = LOW_RES_FULLSCREEN_RESOLUTION
            else:
                best = _preferred_fullscreen_mode(mon)
                if best is None:
                    # Fallback: use current window size if mode detection fails
                    current_w, current_h = glfw.get_window_size(window)
                    fs_w, fs_h = current_w, current_h
                else:
                    fs_w, fs_h = best.size.width, best.size.height
            current_w, current_h = glfw.get_window_size(window)
            if current_w != fs_w or current_h != fs_h:
                glfw.set_window_size(window, fs_w, fs_h)
                # Note: Wayland doesn't support set_window_pos(), so we skip it
            glfw.set_window_attrib(window, glfw.DECORATED, glfw.FALSE)
        else:
            glfw.set_window_attrib(window, glfw.DECORATED, glfw.TRUE)
            win_w = 400
            win_h = int(win_w / (eff_w / eff_h)) if eff_h > 0 else 300
            current_w, current_h = glfw.get_window_size(window)
            if current_w != win_w or current_h != win_h:
                glfw.set_window_size(window, win_w, win_h)
        # Continue to framebuffer size calculation below
    elif window is None:
        if not glfw.init():
            print("❌ ERROR: GLFW Init failed.")
            sys.exit(1)

        # --- DISPLAY RESOLUTION OPTIMIZATION ---
        # If image is smaller than display, optimize resolution to avoid upscaling
        # This must happen AFTER GLFW is initialized but BEFORE windows are created
        if getattr(settings, 'AUTO_OPTIMIZE_DISPLAY_RESOLUTION', True):
            try:
                monitor = glfw.get_primary_monitor()
                if monitor:
                    current_mode = _current_mode(monitor)
                    if current_mode:
                        optimal_mode = _find_optimal_display_mode((img_w, img_h), current_mode)
                        if optimal_mode:
                            opt_w = optimal_mode.size.width
                            opt_h = optimal_mode.size.height
                            curr_w = current_mode.size.width
                            curr_h = current_mode.size.height
                            if opt_w != curr_w or opt_h != curr_h:
                                print(f"[DISPLAY] Image ({img_w}x{img_h}) smaller than display ({curr_w}x{curr_h})")
                                print(f"[DISPLAY] Optimizing to {opt_w}x{opt_h} to avoid upscaling")
                                if _change_display_resolution(optimal_mode):
                                    # Wait a moment for resolution change to take effect
                                    import time
                                    time.sleep(0.1)  # Reduced from 0.5s for faster startup
                                else:
                                    print(f"[DISPLAY] ⚠️  Resolution change failed, continuing with current resolution")
            except Exception as e:
                print(f"[DISPLAY] ⚠️  Resolution optimization failed: {e}, continuing with current resolution")

        def _create_window_for_hints() -> object | None:
            if state.fullscreen:
                mon = glfw.get_primary_monitor()
                if CONNECTED_TO_RCA_HDMI:
                    fs_w, fs_h = RCA_HDMI_RESOLUTION
                elif LOW_RES_FULLSCREEN:
                    fs_w, fs_h = LOW_RES_FULLSCREEN_RESOLUTION
                else:
                    best = _preferred_fullscreen_mode(mon)
                    if best is None:
                        # Fallback: use default resolution if mode detection fails
                        fs_w, fs_h = 1920, 1080
                    else:
                        fs_w, fs_h = best.size.width, best.size.height
                glfw.window_hint(glfw.AUTO_ICONIFY, glfw.FALSE)
                # Try actual fullscreen first (even on Wayland)
                # Fall back to borderless window if fullscreen fails
                if is_wayland:
                    try:
                        # Attempt actual fullscreen on Wayland
                        win = glfw.create_window(fs_w, fs_h, "Fullscreen", mon, None)
                        if win:
                            return win
                    except Exception as e:
                        print(f"[DISPLAY] Wayland fullscreen failed: {e}, falling back to borderless window")
                    # Fallback: borderless fullscreen-sized window
                    glfw.window_hint(glfw.DECORATED, glfw.FALSE)
                    win = glfw.create_window(fs_w, fs_h, "Fullscreen", None, None)
                    return win
                return glfw.create_window(fs_w, fs_h, "Fullscreen", mon, None)

            aspect = eff_w / eff_h
            win_w = 400
            win_h = int(win_w / aspect)
            return glfw.create_window(win_w, win_h, "Windowed Mode", None, None)

        # Version-first probing (no hardware assumptions):
        # - Prefer desktop OpenGL 3.3 (ModernGL path)
        # - Fall back to OpenGL 2.1 / GLES 2.0 (legacy path)
        window_created = False
        last_error = None

        candidates: list[tuple[str, int | None, int | None]] = [("opengl", 330, 330)]  # prefer GL 3.3
        # Then probe GLES versions high→low (e.g. 3.1, 3.0, 2.0, default)
        for code in require_codes:
            candidates.append(("opengles", code, code))
        # Finally probe legacy desktop GL 2.1 and default
        candidates.extend([("opengl", 210, 210), ("opengl", None, None)])

        # Remove duplicates while preserving order
        seen = set()
        uniq: list[tuple[str, int | None, int | None]] = []
        for cand in candidates:
            if cand in seen:
                continue
            seen.add(cand)
            uniq.append(cand)

        for api, version_code, require_code in uniq:
            attempt_version = _format_gl_version(require_code) if require_code is not None else "default"
            print(f"[DISPLAY] Local Window attempt: api={api}, ver={attempt_version}")
            try:
                glfw.default_window_hints()
                if api == "opengles":
                    glfw.window_hint(glfw.CLIENT_API, glfw.OPENGL_ES_API)
                    glfw.window_hint(glfw.CONTEXT_CREATION_API, glfw.EGL_CONTEXT_API)
                    if version_code is not None:
                        glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, version_code // 100)
                        glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, (version_code % 100) // 10)
                else:
                    glfw.window_hint(glfw.CLIENT_API, glfw.OPENGL_API)
                    # On Wayland, EGL is usually the working path even for desktop GL.
                    if os.environ.get("WAYLAND_DISPLAY") or os.environ.get("XDG_SESSION_TYPE") == "wayland":
                        glfw.window_hint(glfw.CONTEXT_CREATION_API, glfw.EGL_CONTEXT_API)
                    else:
                        glfw.window_hint(glfw.CONTEXT_CREATION_API, glfw.NATIVE_CONTEXT_API)
                    if version_code is not None:
                        glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, version_code // 100)
                        glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, (version_code % 100) // 10)
                        if version_code >= 330:
                            glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)

                window = _create_window_for_hints()
                if window:
                    window_created = True
                    break
            except Exception as e:
                last_error = e
                window = None
                print(f"[DISPLAY] Local Window attempt failed (api={api}, ver={attempt_version}): {e}")
                continue

        if not window_created:
            print(f"❌ ERROR: Window creation failed: {last_error}")
            glfw.terminate()
            sys.exit(1)

        if not window:
            glfw.terminate()
            print("❌ ERROR: Window creation failed.")
            sys.exit(1)

        glfw.make_context_current(window)
        glfw.swap_interval(1 if VSYNC else 0)

        # Decide renderer backend from *actual* current context version.
        force_legacy = os.environ.get("FORCE_LEGACY_GL") in {"1", "true", "TRUE", "yes", "YES"}
        version_str = None
        version_code = None
        try:
            version_bytes = glGetString(GL_VERSION)
            version_str = version_bytes.decode("utf-8", errors="replace") if version_bytes else None
            if version_str:
                import re
                m = re.search(r"OpenGL ES\s*([0-9]+)\.([0-9]+)", version_str)
                if not m:
                    m = re.search(r"^\s*([0-9]+)\.([0-9]+)", version_str)
                if m:
                    major, minor = int(m.group(1)), int(m.group(2))
                    version_code = (major * 100) + (minor * 10)
        except Exception:
            pass

        if version_str:
            try:
                renderer_str = glGetString(GL_RENDERER)
                renderer_str = renderer_str.decode("utf-8", errors="replace") if renderer_str else "Unknown"
                print(f"[DISPLAY] GL_VERSION: {version_str}")
                print(f"[DISPLAY] GL_RENDERER: {renderer_str}")
                
                # Detect and store driver info
                # In local mode, ctx may be None if using legacy renderer, so use PyOpenGL path
                driver = _detect_gl_driver(ctx if ctx is not None else None)
                if driver:
                    _BACKEND_USAGE_DATA['driver'] = driver
                    print(f"[DISPLAY] Driver detected: {driver}")
                else:
                    _BACKEND_USAGE_DATA['driver'] = 'unknown'
            except Exception:
                pass

        if force_legacy or (version_code is not None and version_code < 300):
            renderer.initialize_legacy()
            ctx = None
            _BACKEND_USAGE_DATA['local_legacy_used'] = True
            _BACKEND_USAGE_DATA['local_window_backend'] = 'legacy'
            print("[DISPLAY] Renderer backend: legacy (PyOpenGL)")
        else:
            try:
                # Wrap the *current* context created by GLFW.
                # Use require=300 so GLES 3.0 contexts are accepted as ModernGL-capable.
                ctx = moderngl.create_context(require=300)
                _BACKEND_USAGE_DATA['local_window_backend'] = 'moderngl'
            except Exception as e:
                print(f"❌ ERROR: Failed to wrap current context with ModernGL: {e}")
                print("Tip: Set FORCE_LEGACY_GL=1 to force the PyOpenGL legacy backend.")
                sys.exit(1)

            _log_renderer_info(ctx)
            renderer.initialize(ctx)
            print("[DISPLAY] Renderer backend: moderngl")
        
        _log_backend_usage()

        if GAMMA_CORRECTION_ENABLED or ENABLE_SRGB_FRAMEBUFFER:
            try:
                glEnable(GL_FRAMEBUFFER_SRGB)
                if glGetError() == GL_INVALID_ENUM:
                    print("[DISPLAY] sRGB framebuffer not supported on this context; continuing.")
            except Exception:
                pass

        if glfw and window is not None:
            glfw.set_input_mode(window, glfw.CURSOR, glfw.CURSOR_HIDDEN)

    # Fullscreen toggling (only if window already exists):
    # - On Wayland, avoid set_window_monitor (can trigger compositor/device resets). 
    #   Instead, resize the existing window and toggle decoration.
    # - On other platforms, use set_window_monitor for proper fullscreen.
    if window is not None:
        if is_wayland:
            # Wayland: Resize existing window and toggle decoration
            current_w, current_h = glfw.get_window_size(window)
            if state.fullscreen:
                mon = glfw.get_primary_monitor()
                if CONNECTED_TO_RCA_HDMI:
                    fs_w, fs_h = RCA_HDMI_RESOLUTION
                elif LOW_RES_FULLSCREEN:
                    fs_w, fs_h = LOW_RES_FULLSCREEN_RESOLUTION
                else:
                    best = _preferred_fullscreen_mode(mon)
                    if best is None:
                        # Fallback: use current window size if mode detection fails
                        current_w, current_h = glfw.get_window_size(window)
                        fs_w, fs_h = current_w, current_h
                    else:
                        fs_w, fs_h = best.size.width, best.size.height
                # Only resize if size changed
                if current_w != fs_w or current_h != fs_h:
                    glfw.set_window_size(window, fs_w, fs_h)
                    # Note: Wayland doesn't support set_window_pos(), so we skip it
                # Ensure borderless
                glfw.set_window_attrib(window, glfw.DECORATED, glfw.FALSE)
            else:
                # Windowed mode: restore decoration and resize
                glfw.set_window_attrib(window, glfw.DECORATED, glfw.TRUE)
                # Resize to windowed size based on image aspect
                win_w = 400
                win_h = int(win_w / (eff_w / eff_h)) if eff_h > 0 else 300
                if current_w != win_w or current_h != win_h:
                    glfw.set_window_size(window, win_w, win_h)
        else:
            # Non-Wayland (X11): Use set_window_monitor for proper fullscreen
            # On Linux X11, set_window_monitor with current resolution does NOT change display resolution
            # This is safe for low-end Linux systems - window goes fullscreen at current resolution
            current_monitor = glfw.get_window_monitor(window)
            if state.fullscreen:
                mon = glfw.get_primary_monitor()
                if current_monitor != mon:
                    if CONNECTED_TO_RCA_HDMI:
                        fs_w, fs_h = RCA_HDMI_RESOLUTION
                        refresh = 60  # Default refresh rate
                    elif LOW_RES_FULLSCREEN:
                        fs_w, fs_h = LOW_RES_FULLSCREEN_RESOLUTION
                        refresh = 60  # Default refresh rate
                    else:
                        # Use current resolution to avoid display resolution changes
                        # _preferred_fullscreen_mode() returns current mode, which is safe on Linux
                        best = _preferred_fullscreen_mode(mon)
                        if best is None:
                            # Fallback: use current window size if mode detection fails
                            current_w, current_h = glfw.get_window_size(window)
                            fs_w, fs_h = current_w, current_h
                            refresh = 60  # Default refresh rate
                        else:
                            fs_w, fs_h = best.size.width, best.size.height
                            refresh = getattr(best, 'refresh_rate', 60)
                    glfw.set_window_monitor(window, mon, 0, 0, fs_w, fs_h, refresh)
            else:
                if current_monitor is not None:
                    # Restore to a small window; actual sizing will be re-derived below via framebuffer size.
                    glfw.set_window_monitor(window, None, 100, 100, 400, 300, 0)

        fb_w, fb_h = glfw.get_framebuffer_size(window)
        if renderer.using_legacy_gl():
            glViewport(0, 0, fb_w, fb_h)
        else:
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
