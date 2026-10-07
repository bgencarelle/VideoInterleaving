"""Optional GLFW/ModernGL live tuner for scope mode.

The layout and interaction model follow the V7 receiver GUI, but this module
owns only scope controls and the scope trace preview.  Graphics dependencies
are imported when the GUI is constructed, so ordinary/headless scope startup
does not depend on them.
"""
from dataclasses import dataclass
from collections import OrderedDict
import math
import threading
import time


WINDOW_SIZE = (1280, 800)
MIN_WINDOW_SIZE = (900, 600)
PREVIEW_SIZE = 680
MAX_PREVIEW_RENDER_SIZE = 1600
TEXT_SURFACE_CACHE_LIMIT = 512
# CPU/GL work is optional and must not crowd the independently scheduled DAC
# producer, especially on software-rendered desktops. Input events are still
# polled at the engine tick; only visible UI/preview refresh is capped here.
GUI_REDRAW_HZ = 5.0
PREVIEW_RENDER_HZ = 2.0
SLIDER_ROW_HEIGHT = 54
SLIDER_RANGES = {
    "ips": (1.0, 60.0, 1.0),
    "fps": (5.0, 120.0, 1.0),
    "fields": (1.0, 4.0, 1.0),
    "trim": (0.0, 0.60, 0.005),
    "gamma": (0.4, 10.0, 0.1),
    "density": (0.20, 4.0, 0.05),
    "rows": (0.0, 400.0, 1.0),
    "lowpass": (0.0, 12000.0, 100.0),
    "exposure": (0.2, 2.5, 0.05),
    "spot": (0.5, 2.0, 0.05),
}
SLIDER_LABELS = {
    "ips": "Picture rate",
    "fps": "Trace refresh",
    "fields": "Raster fields",
    "trim": "Trim cutoff",
    "gamma": "Tone gamma",
    "density": "Samples / cell",
    "rows": "Raster rows (0 = auto)",
    "lowpass": "Output low-pass",
    "exposure": "Preview exposure",
    "spot": "Preview spot width",
}
MODES = ("vector", "raster", "stochastic", "stipple", "fusion")


@dataclass(frozen=True)
class SliderSpec:
    name: str
    minimum: float
    maximum: float
    step: float
    default: float


def make_slider_spec(name, default):
    """Return a slider's range and immutable startup/default reference value."""
    minimum, maximum, step = SLIDER_RANGES[name]
    default = minimum if default is None else float(default)
    if name in ("ips", "fps", "rows", "lowpass") and default >= maximum:
        maximum = default + max(step, round(default * 0.25 / step) * step)
    default = min(max(default, minimum), maximum)
    return SliderSpec(name, minimum, maximum, step, default)


def slider_fraction(value, spec):
    """Clamped horizontal fraction for a slider value."""
    span = spec.maximum - spec.minimum
    if span <= 0:
        return 0.0
    return min(1.0, max(0.0, (float(value) - spec.minimum) / span))


def slider_value_at(x, left, right, spec):
    """Quantize a pointer position to the slider's supported value."""
    if right <= left:
        return spec.minimum
    fraction = min(1.0, max(0.0, (float(x) - left) / (right - left)))
    raw = spec.minimum + fraction * (spec.maximum - spec.minimum)
    value = spec.minimum + round((raw - spec.minimum) / spec.step) * spec.step
    value = min(spec.maximum, max(spec.minimum, value))
    return int(round(value)) if spec.step >= 1.0 else round(value, 4)


def slider_default_x(left, right, spec):
    """The tick position used to mark a slider's startup/default value."""
    return round(left + slider_fraction(spec.default, spec) * (right - left))


def fit_square_image(width, height, maximum=None):
    """Largest square preview that fits its available GUI area."""
    width, height = max(0, int(width)), max(0, int(height))
    size = min(width, height)
    if maximum is not None:
        size = min(size, max(1, int(maximum)))
    return max(1, size)


def window_size_for_workarea(width, height):
    """Choose a useful window size that stays inside the monitor work area."""
    width, height = max(1, int(width)), max(1, int(height))
    available_width = max(1, width - 32)
    available_height = max(1, height - 80)
    minimum_width = min(MIN_WINDOW_SIZE[0], available_width)
    minimum_height = min(MIN_WINDOW_SIZE[1], available_height)
    return (
        min(available_width, max(minimum_width, round(width * 0.94))),
        min(available_height, max(minimum_height, round(height * 0.90))),
    )


def responsive_layout(width, height, scale=1.0):
    """Return panel rectangles and a square preview fitted to this window."""
    width, height = max(1, int(width)), max(1, int(height))
    unit = lambda value: max(1, round(value * scale))
    margin = max(unit(12), min(unit(32), round(width * 0.016)))
    gap = max(unit(12), min(unit(24), round(width * 0.016)))
    available_width = max(1, width - margin * 2 - gap)
    left_column_right = margin + round(available_width * 0.58)
    top = unit(76)
    bottom = height - unit(56)
    preview_rect = (margin, top, left_column_right, bottom)
    panel = (left_column_right + gap, top, width - margin, bottom)
    preview_top = top + unit(16)
    preview_caption_y = bottom - unit(25)
    preview_area_height = max(1, preview_caption_y - preview_top - unit(8))
    preview_size = fit_square_image(
        preview_rect[2] - preview_rect[0] - unit(38),
        preview_area_height)
    preview_x = preview_rect[0] + (
        preview_rect[2] - preview_rect[0] - preview_size) // 2
    preview_y = preview_top + (preview_area_height - preview_size) // 2
    return {
        "margin": margin,
        "unit": unit,
        "preview_rect": preview_rect,
        "panel": panel,
        "preview_size": preview_size,
        "preview_x": preview_x,
        "preview_y": preview_y,
        "preview_caption_y": preview_caption_y,
    }


def _to_slider_value(name, state, exposure):
    if name in ("exposure", "spot"):
        return exposure if name == "exposure" else 1.0
    value = state.get(name)
    if name == "rows" and value is None:
        return 0
    if name == "lowpass" and value is None:
        return 0
    return 0 if value is None else value


def _format_value(name, value):
    value = float(value)
    if name == "ips":
        return f"{int(round(value))} IPS"
    if name == "fps":
        return f"{int(round(value))} Hz"
    if name == "fields":
        return f"{int(round(value))}x"
    if name == "trim":
        return f"{value:.3f}"
    if name == "gamma":
        return f"{value:.2f}"
    if name == "density":
        return f"{value:.2f}"
    if name == "rows":
        return "Auto" if value <= 0 else f"{int(round(value))} rows"
    if name == "lowpass":
        return "Off" if value <= 0 else f"{value / 1000:g} kHz"
    if name == "exposure":
        return f"{value:.2f}x"
    if name == "spot":
        return f"{value:.2f}x"
    return f"{value:g}"


class ScopeGUI:
    """Native scope preview and slider-based live controls.

    ``poll`` is called by scope_display's producer loop, so UI-originated
    commands are applied on the same thread that owns the renderer and audio
    stream.  Only the expensive phosphor preview is computed by a worker.
    """

    def __init__(self, initial_state, preview_exposure=1.0,
                 start_image_only=False, start_fullscreen=False):
        self._state = dict(initial_state)
        self.preview_exposure = float(preview_exposure)
        self.preview_spot_width = 1.0
        self.specs = {
            "ips": make_slider_spec("ips", initial_state.get("ips", 30)),
            "fps": make_slider_spec("fps", initial_state.get("fps", 30)),
            "fields": make_slider_spec("fields", initial_state.get("fields", 1)),
            "trim": make_slider_spec("trim", initial_state.get("trim", 0.02)),
            "gamma": make_slider_spec("gamma", initial_state.get("gamma", 2.2)),
            "density": make_slider_spec("density", initial_state.get("density", 1.0)),
            "rows": make_slider_spec("rows", initial_state.get("rows", 0)),
            "lowpass": make_slider_spec("lowpass", initial_state.get("lowpass", 0)),
            "exposure": make_slider_spec("exposure", preview_exposure),
            "spot": make_slider_spec("spot", 1.0),
        }
        self.closed = False
        self.close_requested = False
        self.message = "Amber ticks mark startup defaults"
        self._actions = []
        self._hits = {}
        self._dragging = None
        self._drag_value = None
        self._last_draw = 0.0
        self._preview_lock = threading.Lock()
        self._preview_rgb = None
        self._preview_error = ""
        self._preview_stop = threading.Event()
        self._preview_thread = None
        self._preview_seq = -1
        self._rendered_seq = -1
        self._preview_target_size = PREVIEW_SIZE
        self._rendered_size = None
        self._preview_points = None
        self._rendered_exposure = None
        self._rendered_spot_width = None
        self._font_cache = {}
        self._text_surface_cache = OrderedDict()
        self._canvas = None
        self._control_tab = "picture"
        self._last_draw_signature = None
        self._refresh_revision = 0
        self._uploaded_image = None
        self.texture = None
        self.program = None
        self.vao = None
        self.fullscreen = False
        self.image_only = bool(start_image_only or start_fullscreen)
        self._windowed_geometry = None
        try:
            self._init_window()
            if start_fullscreen:
                self.set_fullscreen(True)
            if self.image_only:
                self.set_image_only(True)
        except Exception:
            self._release_graphics()
            raise
        self._start_preview_worker()

    def _init_window(self):
        try:
            import glfw
            import moderngl
            from PIL import Image, ImageDraw, ImageFont
        except Exception as exc:
            raise RuntimeError(
                "scope GUI needs glfw, moderngl, and Pillow: " + str(exc)) from exc

        self.glfw = glfw
        self.moderngl = moderngl
        self.Image = Image
        self.ImageDraw = ImageDraw
        self.ImageFont = ImageFont
        if not glfw.init():
            raise RuntimeError("GLFW initialization failed for scope GUI")

        monitor = glfw.get_primary_monitor()
        work_area = None
        if monitor is not None:
            try:
                work_area = tuple(int(value) for value in
                                  glfw.get_monitor_workarea(monitor))
            except Exception:
                try:
                    video_mode = glfw.get_video_mode(monitor)
                    monitor_x, monitor_y = glfw.get_monitor_pos(monitor)
                    work_area = (int(monitor_x), int(monitor_y),
                                 int(video_mode.size.width),
                                 int(video_mode.size.height))
                except Exception:
                    work_area = None
        if work_area is not None and len(work_area) == 4:
            area_x, area_y, area_width, area_height = work_area
            window_size = window_size_for_workarea(area_width, area_height)
            min_width = min(MIN_WINDOW_SIZE[0], max(1, area_width - 32))
            min_height = min(MIN_WINDOW_SIZE[1], max(1, area_height - 80))
        else:
            work_area = None
            window_size = WINDOW_SIZE
            min_width, min_height = MIN_WINDOW_SIZE

        failures = []
        for use_gles in (False, True):
            glfw.default_window_hints()
            if use_gles:
                glfw.window_hint(glfw.CLIENT_API, glfw.OPENGL_ES_API)
                glfw.window_hint(glfw.CONTEXT_CREATION_API, glfw.EGL_CONTEXT_API)
                glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
                glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 0)
                version = 300
                shader_version = "#version 300 es\nprecision highp float;\n"
            else:
                glfw.window_hint(glfw.CLIENT_API, glfw.OPENGL_API)
                glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
                glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
                glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
                glfw.window_hint(glfw.OPENGL_FORWARD_COMPAT, glfw.TRUE)
                version = 330
                shader_version = "#version 330\n"
            glfw.window_hint(glfw.RESIZABLE, glfw.TRUE)
            window = glfw.create_window(*window_size, "Scope · Live Tuner",
                                        None, None)
            if not window:
                failures.append("GLES" if use_gles else "OpenGL")
                continue
            context = None
            try:
                glfw.make_context_current(window)
                context = moderngl.create_context(require=version)
                if getattr(context, "version_code", 0) < 300:
                    raise RuntimeError("OpenGL ES 3.0 / OpenGL 3.3 is required")
                self.window = window
                self.context = context
                self._shader_version = shader_version
                break
            except Exception as exc:
                failures.append(str(exc))
                if context is not None:
                    try:
                        context.release()
                    except Exception:
                        pass
                glfw.make_context_current(None)
                glfw.destroy_window(window)
        else:
            glfw.terminate()
            raise RuntimeError("Could not create scope GUI graphics context: "
                               + "; ".join(failures))

        self.program = self.context.program(
            vertex_shader=self._shader_version + """
                out vec2 uv;
                void main() {
                    vec2 p = vec2((gl_VertexID << 1) & 2, gl_VertexID & 2);
                    // The oversized triangle interpolates p over [0, 1] on
                    // screen; use the full texture range to avoid cropping it.
                    uv = vec2(p.x, 1.0 - p.y);
                    gl_Position = vec4(p * 2.0 - 1.0, 0.0, 1.0);
                }
            """,
            fragment_shader=self._shader_version + """
                uniform sampler2D screen_image;
                in vec2 uv;
                out vec4 color;
                void main() { color = texture(screen_image, uv); }
            """)
        self.program["screen_image"].value = 0
        self.vao = self.context.vertex_array(self.program, [])
        self.font = self._font(17)
        self.small = self._font(13)
        self.tiny = self._font(11, mono=True)
        self.work_area = work_area
        self.primary_monitor = monitor
        glfw.set_window_close_callback(self.window,
                                       lambda _w: setattr(self, "close_requested", True))
        glfw.set_window_size_limits(self.window, min_width, min_height,
                                    glfw.DONT_CARE, glfw.DONT_CARE)
        if work_area is not None:
            try:
                glfw.set_window_pos(
                    self.window,
                    area_x + max(0, (area_width - window_size[0]) // 2),
                    area_y + max(0, (area_height - window_size[1]) // 2))
            except Exception:
                pass
        glfw.set_mouse_button_callback(self.window, self._on_mouse_button)
        glfw.set_cursor_pos_callback(self.window, self._on_cursor)
        glfw.set_key_callback(self.window, self._on_key)
        glfw.set_window_refresh_callback(self.window, self._on_refresh)

    def _font(self, size, mono=False):
        key = int(size), bool(mono)
        cached = self._font_cache.get(key)
        if cached is not None:
            return cached
        names = (("DejaVuSansMono.ttf", "DejaVuSans.ttf") if mono else
                 ("DejaVuSans.ttf", "DejaVuSansMono.ttf"))
        for name in names:
            try:
                font = self.ImageFont.truetype(name, int(size))
                self._font_cache[key] = font
                return font
            except OSError:
                pass
        try:
            font = self.ImageFont.load_default(size=int(size))
        except TypeError:
            font = self.ImageFont.load_default()
        self._font_cache[key] = font
        return font

    def _draw_cached_text(self, image, xy, text, fill, font):
        """Composite a cached, pixel-identical rasterized text surface."""
        text = str(text)
        try:
            color = tuple(int(value) for value in fill)
            if len(color) == 3:
                rgba = color + (255,)
            elif len(color) == 4:
                rgba = color
            else:
                raise ValueError("unsupported text color")
            left, top, right, bottom = map(int, font.getbbox(text))
            x, y = float(xy[0]), float(xy[1])
        except (TypeError, ValueError, AttributeError):
            self.ImageDraw.Draw(image).text(xy, text, fill=fill, font=font)
            return
        if right <= left or bottom <= top:
            return

        # Pillow rasterizes fractional text origins. Keep the fractional phase
        # in the cache key and the integer origin outside the tile so cached
        # surfaces exactly match ImageDraw at both integer and centered labels.
        origin_x = math.floor(x + left) - 1
        origin_y = math.floor(y + top) - 1
        if origin_x < 0 or origin_y < 0:
            self.ImageDraw.Draw(image).text(xy, text, fill=fill, font=font)
            return
        key = (id(font), text, color,
               x - math.floor(x), y - math.floor(y))
        cache = self._text_surface_cache
        tile = cache.get(key)
        if tile is None:
            tile = self.Image.new(
                "RGBA", (right - left + 3, bottom - top + 3), (0, 0, 0, 0))
            self.ImageDraw.Draw(tile).text(
                (x - origin_x, y - origin_y), text, fill=rgba, font=font)
            cache[key] = tile
            if len(cache) > TEXT_SURFACE_CACHE_LIMIT:
                cache.popitem(last=False)
        else:
            cache.move_to_end(key)
        image.alpha_composite(tile, (origin_x, origin_y))

    def _start_preview_worker(self):
        self._preview_thread = threading.Thread(
            target=self._preview_loop, name="ScopePreview", daemon=True)
        self._preview_thread.start()

    def _preview_loop(self):
        try:
            from scope_bake import PreviewWorkspace, _warm_preview_kernels, preview_frame
            from scope_out import Scope
            import numpy as np
            _warm_preview_kernels()
        except Exception as exc:
            with self._preview_lock:
                self._preview_error = f"Preview unavailable: {exc}"
            return
        while not self._preview_stop.is_set():
            Scope.want_tap(1.0)
            seq, points = Scope.read_tap()
            if points is not None:
                if seq != self._preview_seq:
                    self._preview_seq = seq
                    self._preview_points = points
                exposure = self.preview_exposure
                changed_exposure = exposure != self._rendered_exposure
                spot_width = self.preview_spot_width
                changed_spot = spot_width != self._rendered_spot_width
                target_size = min(
                    MAX_PREVIEW_RENDER_SIZE,
                    max(1, int(self._preview_target_size)))
                changed_size = target_size != self._rendered_size
                render_pending = (seq != self._rendered_seq
                                 or changed_exposure or changed_spot
                                 or changed_size)
                render_due = (time.monotonic() - getattr(
                    self, "_last_preview_render", 0.0)
                    >= 1.0 / PREVIEW_RENDER_HZ)
                if self._preview_points is not None and render_pending and render_due:
                    try:
                        workspace = getattr(self, "_preview_workspace", None)
                        if workspace is None or workspace.size != target_size:
                            workspace = PreviewWorkspace(target_size)
                            self._preview_workspace = workspace
                        kwargs = {"size": target_size, "exposure": exposure}
                        kwargs["workspace"] = workspace
                        if spot_width != 1.0:
                            points_array = np.asarray(self._preview_points)
                            from scope_numeric import preview_rows
                            rows = preview_rows(np.asarray(points_array,dtype=np.float32))
                            kwargs["spot"] = max(
                                0.6, 0.40 * target_size / rows * spot_width)
                        frame = preview_frame(self._preview_points, **kwargs)
                        with self._preview_lock:
                            # Detach from the workspace buffer: the worker reuses
                            # it on the next render while the UI may still upload
                            # this snapshot on the presentation thread.
                            self._preview_rgb = frame.copy()
                            self._preview_error = ""
                        self._rendered_seq = self._preview_seq
                        self._rendered_exposure = exposure
                        self._rendered_spot_width = spot_width
                        self._rendered_size = target_size
                        self._last_preview_render = time.monotonic()
                    except Exception as exc:
                        with self._preview_lock:
                            self._preview_error = f"Preview error: {exc}"
            self._preview_stop.wait(0.025)

    def set_preview_exposure(self, value):
        self.preview_exposure = float(value)

    def set_preview_spot_width(self, value):
        """Adjust CPU phosphor blur only; this never changes DAC samples."""
        self.preview_spot_width = float(value)

    def poll(self, state, metrics=None):
        """Pump input, draw the newest preview and return UI actions."""
        if self.closed:
            return []
        self._state = dict(state)
        self._metrics = dict(metrics or {})
        self.glfw.poll_events()
        if self.glfw.window_should_close(self.window):
            self.close_requested = True
        if self.close_requested:
            return []
        now = time.monotonic()
        signature = self._draw_signature()
        if (signature != self._last_draw_signature
                and now - self._last_draw >= 1.0 / GUI_REDRAW_HZ):
            self._draw()
            # Record the pre-draw snapshot. A worker publication during drawing
            # will differ on the next poll and cannot be cleared accidentally.
            self._last_draw_signature = signature
            self._last_draw_preview_rgb = self._signature_preview_rgb
            self._last_draw = now
        actions, self._actions = self._actions, []
        return actions

    def _draw_signature(self):
        with self._preview_lock:
            # Pin the snapshot as well as its identity so Python cannot recycle
            # its id between publications and make a new frame look unchanged.
            self._signature_preview_rgb = self._preview_rgb
            preview = (id(self._preview_rgb), self._preview_error)
        return (repr(self._state), repr(self._metrics), preview,
                self.glfw.get_window_size(self.window),
                self.glfw.get_framebuffer_size(self.window),
                self.image_only, self.fullscreen, self.message,
                self._dragging, self._drag_value,
                self.preview_exposure, self.preview_spot_width,
                getattr(self, "_control_tab", "picture"),
                getattr(self, "_refresh_revision", 0))

    def _on_refresh(self, _window):
        self._refresh_revision += 1

    def set_fullscreen(self, enabled):
        """Switch between the fitted window and the primary display mode."""
        enabled = bool(enabled)
        if enabled == self.fullscreen:
            return True
        monitor = self.primary_monitor or self.glfw.get_primary_monitor()
        if monitor is None:
            self.message = "Fullscreen display is unavailable"
            return False
        if enabled:
            x, y = self.glfw.get_window_pos(self.window)
            width, height = self.glfw.get_window_size(self.window)
            self._windowed_geometry = (int(x), int(y),
                                       int(width), int(height))
            mode = self.glfw.get_video_mode(monitor)
            self.glfw.set_window_monitor(
                self.window, monitor, 0, 0,
                int(mode.size.width), int(mode.size.height),
                int(mode.refresh_rate))
            self.fullscreen = True
            self._update_cursor_visibility()
            return True

        if self._windowed_geometry is None:
            area_size = (self.work_area[2:] if self.work_area is not None
                         else WINDOW_SIZE)
            width, height = window_size_for_workarea(*area_size)
            x = y = 0
        else:
            x, y, width, height = self._windowed_geometry
        self.glfw.set_window_monitor(
            self.window, None, x, y, width, height, self.glfw.DONT_CARE)
        self.fullscreen = False
        self._update_cursor_visibility()
        return True

    def _update_cursor_visibility(self):
        cursor_mode = (self.glfw.CURSOR_HIDDEN
                       if self.image_only and self.fullscreen
                       else self.glfw.CURSOR_NORMAL)
        self.glfw.set_input_mode(self.window, self.glfw.CURSOR, cursor_mode)

    def set_image_only(self, enabled):
        """Show just the scope preview; click it to return to the controls."""
        self.image_only = bool(enabled)
        if self.image_only:
            self._dragging = None
            self._drag_value = None
        self._last_draw = 0.0
        try:
            self._update_cursor_visibility()
        except Exception:
            pass

    def _slider_disabled(self, name, state):
        if name in state.get("disabled_sliders", ()):
            return True
        if name in ("ips", "fps", "fields") and state.get("mode_locked"):
            return True
        if name == "ips" and state.get("clock_locked"):
            return True
        if name in ("fields", "rows") and state.get("mode") != "raster":
            return True
        return False

    def _slider_value(self, name, state):
        if name in ("exposure", "spot"):
            value = self.preview_exposure
            if name == "spot":
                value = self.preview_spot_width
        else:
            value = _to_slider_value(name, state, self.preview_exposure)
        if self._dragging == name and self._drag_value is not None:
            return self._drag_value
        return value

    def _draw(self):
        width, height = self.glfw.get_window_size(self.window)
        if width <= 0 or height <= 0:
            return
        ui_scale = min(1.4, max(0.75, height / WINDOW_SIZE[1]))
        unit = lambda value: max(1, round(value * ui_scale))
        self._ui_unit = unit
        self.font = self._font(round(17 * ui_scale))
        self.small = self._font(max(13, round(13 * ui_scale)))
        self.tiny = self._font(max(11, round(11 * ui_scale)), mono=True)
        layout = responsive_layout(width, height, ui_scale)
        image = self._canvas_for_size(width, height)
        draw = self.ImageDraw.Draw(image)
        state = self._state
        metrics = self._metrics
        self._hits = {}

        if self.image_only:
            self._draw_image_only(image, draw, width, height)
            self._present(image)
            return

        header_height = unit(56)
        draw.rectangle((0, 0, width, header_height), fill=(10, 18, 25, 255))
        title_x, title_y = unit(22), unit(15)
        title = "Scope · Preview"
        self._draw_cached_text(image, (title_x, title_y), title,
                               (239, 245, 249), self.font)
        fullscreen_label = "Restore [F11]" if self.fullscreen else "Fullscreen [F11]"
        button_width = max(unit(112),
                           round(self.small.getlength(fullscreen_label))
                           + unit(20))
        button_height = unit(34)
        margin = layout["margin"]
        button_rect = (width - margin - button_width,
                       max(1, (header_height - button_height) // 2),
                       width - margin,
                       max(1, (header_height + button_height) // 2))
        draw.rounded_rectangle(
            button_rect, radius=unit(4), fill=(20, 36, 47),
            outline=(53, 78, 94), width=max(1, unit(1)))
        self._draw_cached_text(
            image, (button_rect[0] + (button_width -
                                     self.small.getlength(fullscreen_label)) / 2,
                    button_rect[1] + unit(8)),
            fullscreen_label, (216, 229, 237), self.small)
        self._hits["fullscreen:toggle"] = button_rect

        image_only_label = "Image only [F10]"
        image_only_width = max(
            unit(112), round(self.small.getlength(image_only_label))
            + unit(20))
        image_only_rect = (
            button_rect[0] - unit(8) - image_only_width,
            button_rect[1], button_rect[0] - unit(8), button_rect[3])
        draw.rounded_rectangle(
            image_only_rect, radius=unit(4), fill=(20, 36, 47),
            outline=(53, 78, 94), width=max(1, unit(1)))
        self._draw_cached_text(
            image, (image_only_rect[0] + (image_only_width -
                                         self.small.getlength(image_only_label)) / 2,
                    image_only_rect[1] + unit(8)),
            image_only_label, (216, 229, 237), self.small)
        self._hits["image_only:toggle"] = image_only_rect

        header = (f"{metrics.get('device', 'connecting')}   ·   "
                  f"{metrics.get('sample_rate', 0):,} Hz   ·   "
                  f"{str(state.get('mode', 'raster')).upper()}")
        title_end = title_x + self.font.getlength(title)
        header_left = title_end + unit(18)
        header_right = image_only_rect[0] - unit(12)
        header_width = max(0, header_right - header_left)
        if header_width:
            self._draw_cached_text(
                image, (header_left, unit(18)),
                self._fit_text(header, self.small, header_width),
                (155, 187, 204), self.small)

        preview_rect = layout["preview_rect"]
        draw.rounded_rectangle(preview_rect, radius=6, fill=(6, 10, 13, 255),
                               outline=(49, 69, 83, 255), width=unit(1))
        preview_size = layout["preview_size"]
        preview_x = layout["preview_x"]
        preview_y = layout["preview_y"]
        self._preview_target_size = min(preview_size, MAX_PREVIEW_RENDER_SIZE)
        with self._preview_lock:
            rgb = self._preview_rgb
            preview_error = self._preview_error
        if rgb is not None:
            pic = self.Image.fromarray(rgb, mode="RGB")
            if pic.size != (preview_size, preview_size):
                pic = pic.resize((preview_size, preview_size),
                                 self.Image.Resampling.BILINEAR)
            image.alpha_composite(pic.convert("RGBA"), (preview_x, preview_y))
        else:
            msg = preview_error or "Waiting for the first scope trace…"
            self._draw_cached_text(
                image, (preview_rect[0] + unit(22),
                        preview_rect[1] + unit(24)),
                self._fit_text(msg, self.small,
                               preview_rect[2] - preview_rect[0] - unit(44)),
                (151, 174, 192), self.small)
        self._draw_cached_text(
            image, (preview_rect[0] + unit(16), layout["preview_caption_y"]),
            self._fit_text(
                "Phosphor preview  ·  exposure changes this preview only",
                self.tiny, preview_rect[2] - preview_rect[0] - unit(32)),
            (119, 145, 160), self.tiny)

        panel = layout["panel"]
        draw.rounded_rectangle(panel, radius=6, fill=(15, 24, 32, 255),
                               outline=(49, 69, 83, 255), width=unit(1))
        self._draw_cached_text(
            image, (panel[0] + unit(18), panel[1] + unit(13)), "Tune scope",
            (231, 240, 246), self.font)
        draw.line((panel[0] + unit(18), panel[1] + unit(43),
                   panel[2] - unit(18), panel[1] + unit(43)),
                  fill=(42, 58, 70), width=unit(1))
        draw.line((panel[2] - unit(182), panel[1] + unit(25),
                   panel[2] - unit(182), panel[1] + unit(34)),
                  fill=(243, 178, 85), width=unit(2))
        self._draw_cached_text(
            image, (panel[2] - unit(174), panel[1] + unit(22)),
            "startup default", (159, 173, 181), self.tiny)

        tab = getattr(self, "_control_tab", "picture")
        tabs = (("picture", "Drawing"), ("output", "Output"), ("preview", "Preview"))
        tab_width = (panel[2] - panel[0] - unit(40)) // 3
        for i, (name, label) in enumerate(tabs):
            left = panel[0] + unit(18) + i * (tab_width + unit(2))
            rect = (left, panel[1] + unit(48), left + tab_width,
                    panel[1] + unit(82))
            draw.rounded_rectangle(rect, radius=4,
                                   fill=(38, 86, 108) if tab == name else (20, 32, 43))
            self._draw_cached_text(image, (left + unit(10), rect[1] + unit(8)),
                                   label, (224, 236, 244), self.small)
            self._hits[f"tab:{name}"] = rect
        sliders = ({"picture": ("ips", "fps", "fields", "trim", "gamma", "density", "rows"),
                    "output": ("lowpass",), "preview": ("exposure", "spot")})[tab]
        controls_top = panel[1] + unit(94)
        available_controls_height = panel[3] - controls_top
        row_fit = ((available_controls_height - unit(166))
                    // len(sliders))
        preferred_row = max(unit(42), round(SLIDER_ROW_HEIGHT * ui_scale))
        slider_row_height = max(
            unit(30), min(preferred_row, row_fit))
        track_left = panel[0] + unit(20)
        track_right = panel[2] - unit(124)
        for index, name in enumerate(sliders):
            top = controls_top + index * slider_row_height
            spec = self.specs[name]
            value = self._slider_value(name, state)
            disabled = self._slider_disabled(name, state)
            label_color = (119, 137, 149) if disabled else (208, 220, 228)
            readout = _format_value(name, value)
            if disabled:
                if state.get("mode_locked") and name in ("ips", "fps", "fields"):
                    readout = "Fixed"
                elif name in ("fields", "rows") and state.get("mode") != "raster":
                    readout = "Raster only"
                else:
                    readout = "Fixed by source"
            self._draw_cached_text(
                image, (track_left, top + unit(3)), SLIDER_LABELS[name],
                label_color, self.small)
            self._draw_cached_text(
                image, (panel[2] - unit(112), top + unit(3)),
                self._fit_text(readout, self.tiny, unit(100)),
                (115, 132, 145) if disabled else (235, 242, 247), self.tiny)
            y = top + max(unit(24), slider_row_height - unit(8))
            thumb = min(unit(6), max(1, slider_row_height // 8))
            marker_half = min(unit(9), max(1, slider_row_height // 4))
            track_color = (45, 61, 73) if disabled else (61, 78, 90)
            draw.rounded_rectangle(
                (track_left, y - unit(2), track_right, y + unit(2)),
                radius=unit(2), fill=track_color)
            default_x = slider_default_x(track_left, track_right, spec)
            draw.line((default_x, y - marker_half, default_x, y + marker_half),
                      fill=(243, 178, 85) if not disabled else (112, 101, 81),
                      width=unit(2))
            current_x = round(track_left + slider_fraction(value, spec)
                               * (track_right - track_left))
            if current_x > track_left:
                draw.rounded_rectangle((track_left, y - unit(2), current_x,
                                        y + unit(2)), radius=unit(2),
                                       fill=(63, 137, 174) if not disabled
                                       else (54, 66, 73))
            if disabled:
                thumb = max(1, thumb * 2 // 3)
            draw.ellipse((current_x - thumb, y - thumb,
                          current_x + thumb, y + thumb),
                         fill=(205, 232, 245) if not disabled else (105, 119, 128),
                         outline=(66, 133, 164) if not disabled else (82, 94, 101))
            self._hits[f"slider:{name}"] = (track_left - unit(8), top,
                                             track_right + unit(8),
                                             top + slider_row_height - unit(2))

        mode_y = controls_top + len(sliders) * slider_row_height + unit(4)
        mode_gap = unit(5)
        available = state.get("available_modes", MODES)
        modes = tuple(mode for mode in MODES if mode in available)
        if state.get("mode_locked"):
            modes = (state.get("mode", "raster"),)
        mode_width = (panel[2] - panel[0] - unit(36)
                      - mode_gap * (len(modes) - 1)) // max(1, len(modes))
        mode_height = unit(31)
        for i, mode in enumerate(modes):
            left = panel[0] + unit(18) + i * (mode_width + mode_gap)
            rect = (left, mode_y, left + mode_width, mode_y + mode_height)
            selected = state.get("mode") == mode
            locked = bool(state.get("mode_locked"))
            unsupported = mode not in state.get("available_modes", MODES)
            disabled = locked or unsupported
            fill = ((42, 83, 105) if selected else (19, 33, 43)) if not disabled else (
                (35, 48, 57) if selected else (17, 25, 31))
            outline = ((84, 153, 181) if selected else (46, 66, 79)) if not disabled else (
                (63, 78, 88) if selected else (38, 48, 56))
            draw.rounded_rectangle(rect, radius=unit(4), fill=fill,
                                   outline=outline, width=unit(1))
            label = "Raster · live source (fixed)" if locked and mode == "raster" else mode.title()
            text_width = self.small.getlength(label)
            self._draw_cached_text(
                image, (left + (mode_width - text_width) / 2,
                        mode_y + unit(8)), label,
                (231, 241, 247) if not disabled else (113, 127, 136),
                self.small)
            if not disabled:
                self._hits[f"mode:{mode}"] = rect

        button_y = mode_y + unit(40)
        button_gap = unit(7)
        button_width = (panel[2] - panel[0] - unit(36)
                        - 3 * button_gap) // 4
        control_button_height = unit(31)
        buttons = (("key", "i", "Invert"),
                   ("key", "r", "Rotate"),
                   ("key", "m", "Mirror"),
                   ("audio", "toggle",
                     "XY: on" if not state.get("audio_muted", True)
                     else "XY: off"))
        for i, (kind, key, label) in enumerate(buttons):
            left = panel[0] + unit(18) + i * (button_width + button_gap)
            rect = (left, button_y, left + button_width,
                    button_y + control_button_height)
            audio_button = kind == "audio"
            active = audio_button and not state.get("audio_muted", True)
            draw.rounded_rectangle(
                rect, radius=unit(4),
                fill=(42, 83, 105) if active else (20, 36, 47),
                outline=(84, 153, 181) if active else (53, 78, 94), width=1)
            label_width = self.small.getlength(label)
            self._draw_cached_text(
                image, (left + (button_width - label_width) / 2,
                        button_y + unit(8)), label,
                (231, 241, 247) if audio_button else (216, 229, 237),
                self.small)
            if audio_button:
                self._hits["audio:toggle"] = rect
            else:
                self._hits[f"key:{key}"] = rect

        stats_y = button_y + unit(40)
        if tab == "preview":
            for i, (name, label) in enumerate((("crisp", "Less glow"), ("reset", "Reset preview"))):
                left = panel[0] + unit(18) + i * ((panel[2] - panel[0] - unit(40)) // 2)
                rect = (left, stats_y, left + (panel[2] - panel[0] - unit(44)) // 2,
                        stats_y + unit(32))
                draw.rounded_rectangle(rect, radius=4, fill=(20, 48, 62))
                self._draw_cached_text(image, (left + unit(10), stats_y + unit(8)),
                                       label, (222, 238, 244), self.small)
                self._hits[f"preview:{name}"] = rect
            stats_y += unit(45)
            self._draw_cached_text(image, (panel[0] + unit(18), stats_y),
                                   "Appearance only; output samples are unchanged.",
                                   (145, 182, 196), self.tiny)
            stats_y += unit(28)
        trace_hz = float(metrics.get("trace_hz", 0) or 0)
        picture_hz = float(metrics.get("picture_hz", 0) or 0)
        samples = int(metrics.get("samples", 0) or 0)
        grid = metrics.get("grid", "--") or "--"
        buffer_text = (f"{float(metrics.get('buffered_ms', 0.0)):.1f}/"
                       f"{float(metrics.get('buffer_capacity_ms', 0.0)):.1f}ms")
        buffer_kind = metrics.get("buffer_kind", "queue")
        dac_latency = float(metrics.get("dac_latency_ms", 0.0) or 0.0)
        schedule_offset = metrics.get("adoption_dac_schedule_offset_ms")
        try:
            schedule_text = f"{float(schedule_offset):+.1f}ms"
        except (TypeError, ValueError):
            schedule_text = "--"
        errors = (f"xruns {metrics.get('dropouts', 0)}/"
                  f"{metrics.get('underruns', 0)}")
        self._draw_cached_text(
            image, (panel[0] + unit(18), stats_y),
            self._fit_text(
                    f"{trace_hz:.1f} traces/s  ·  {picture_hz:.1f} pictures/s",
                self.small, panel[2] - panel[0] - unit(36)),
            (158, 190, 205), self.small)
        self._draw_cached_text(
            image, (panel[0] + unit(18), stats_y + unit(21)),
            self._fit_text(
                    (f"{buffer_kind} {buffer_text} · latency {dac_latency:.1f}ms"
                     if tab == "output" else f"{samples:,} samples · grid {grid}"),
                self.tiny, panel[2] - panel[0] - unit(36)),
            (218, 153, 122) if (metrics.get("dropouts", 0)
                                or metrics.get("underruns", 0))
            else (133, 158, 173), self.tiny)
        if tab == "output":
            self._draw_cached_text(
                image, (panel[0] + unit(18), stats_y + unit(42)),
                self._fit_text(f"Scheduled DAC {schedule_text} · {errors}",
                               self.tiny, panel[2] - panel[0] - unit(36)),
                (218, 153, 122) if (metrics.get("dropouts", 0)
                                       or metrics.get("underruns", 0))
                else (133, 158, 173), self.tiny)
        note = self.message
        if state.get("mode_locked") and tab == "picture":
            note = "Fixed rates/fields: change in the launcher."
        if note:
            self._draw_cached_text(
                image, (panel[0] + unit(18), panel[3] - unit(23)),
                self._fit_text(note, self.tiny,
                               panel[2] - panel[0] - unit(36)),
                (243, 178, 85), self.tiny)
        footer_height = max(36, unit(36))
        draw.rectangle((0, height - footer_height, width, height),
                       fill=(9, 15, 20, 255))
        self._draw_cached_text(
            image, (unit(20), height - unit(27)),
            "XY starts muted. XY: on enables output; mute parks the beam (no Z blanking).",
            (151, 169, 179), self.tiny)
        footer_note = ("Preview appearance does not change output brightness. "
                       "F10 image only · F11 fullscreen · Esc restore · Q close.")
        self._draw_cached_text(
            image, (unit(20), height - unit(13)),
            self._fit_text(footer_note, self.tiny, width - unit(40)),
            (151, 169, 179), self.tiny)

        self._present(image)

    def _canvas_for_size(self, width, height):
        """Reuse the native UI backing image until the window is resized."""
        image = self._canvas
        if image is None or image.size != (width, height):
            image = self.Image.new("RGBA", (width, height))
            self._canvas = image
        image.paste((12, 19, 26, 255), (0, 0, width, height))
        return image

    def _draw_image_only(self, image, draw, width, height):
        """Letterbox the square phosphor preview across the whole window."""
        size = fit_square_image(width, height, MAX_PREVIEW_RENDER_SIZE)
        x = (width - size) // 2
        y = (height - size) // 2
        self._preview_target_size = size
        with self._preview_lock:
            rgb = self._preview_rgb
            error = self._preview_error
        if rgb is not None:
            pic = self.Image.fromarray(rgb, mode="RGB")
            if pic.size != (size, size):
                pic = pic.resize((size, size), self.Image.Resampling.BILINEAR)
            image.alpha_composite(pic.convert("RGBA"), (x, y))
        else:
            message = error or "Waiting for the first scope trace…"
            text_width = self.small.getlength(message)
            self._draw_cached_text(
                image, ((width - text_width) / 2, height // 2), message,
                (151, 174, 192), self.small)

    def _present(self, image):
        framebuffer = self.glfw.get_framebuffer_size(self.window)
        if framebuffer[0] <= 0 or framebuffer[1] <= 0:
            return
        self.context.viewport = (0, 0, framebuffer[0], framebuffer[1])
        self.context.clear(0.04, 0.06, 0.08, 1.0)
        if self.texture is None or self.texture.size != image.size:
            if self.texture is not None:
                self.texture.release()
            self.texture = self.context.texture(
                image.size, 4, image.tobytes(), alignment=1)
            self.texture.filter = (self.moderngl.LINEAR, self.moderngl.LINEAR)
            self._uploaded_image = image.copy()
        else:
            from PIL import ImageChops
            previous = self._uploaded_image
            # RGBA alpha is usually constant, so include differences in all
            # color channels when determining the changed rectangle.
            bounds = (ImageChops.difference(image, previous)
                      .convert("RGB").getbbox()) if previous is not None else (
                          0, 0, image.width, image.height)
            if bounds is not None:
                left, top, right, bottom = bounds
                self.texture.write(image.crop(bounds).tobytes(),
                                   viewport=(left, top, right - left,
                                             bottom - top), alignment=1)
                self._uploaded_image = image.copy()
        self.texture.use(location=0)
        self.vao.render(mode=self.moderngl.TRIANGLES, vertices=3)
        self.glfw.swap_buffers(self.window)

    @staticmethod
    def _fit_text(text, font, width):
        text = str(text)
        while text and text != "…" and font.getlength(text) > width:
            text = text[:-2] + "…"
        return text

    def _value_for_pointer(self, name, x):
        rect = self._hits.get(f"slider:{name}")
        if rect is None:
            return None
        left, _top, right, _bottom = rect
        pad = getattr(self, "_ui_unit", lambda value: value)(8)
        return slider_value_at(x, left + pad, right - pad, self.specs[name])

    def _on_mouse_button(self, _window, button, action, _mods):
        if button != self.glfw.MOUSE_BUTTON_LEFT:
            return
        x, y = self.glfw.get_cursor_pos(self.window)
        if action == self.glfw.PRESS:
            if self.image_only:
                self.set_image_only(False)
                return
            if self._state.get("mode_locked"):
                locked = {"ips", "fps", "fields"}
            else:
                locked = set()
            for name in SLIDER_RANGES:
                rect = self._hits.get(f"slider:{name}")
                if (rect is not None and rect[0] <= x <= rect[2]
                        and rect[1] <= y <= rect[3]
                        and name not in locked
                        and not self._slider_disabled(name, self._state)):
                    self._dragging = name
                    self._drag_value = self._value_for_pointer(name, x)
                    if name in ("exposure", "spot"):
                        if name == "exposure":
                            self.set_preview_exposure(self._drag_value)
                        else:
                            self.set_preview_spot_width(self._drag_value)
                    return
            for hit, rect in self._hits.items():
                if rect[0] <= x <= rect[2] and rect[1] <= y <= rect[3]:
                    if hit.startswith("mode:"):
                        self._actions.append(("mode", hit.split(":", 1)[1]))
                    elif hit.startswith("tab:"):
                        self._control_tab = hit.split(":", 1)[1]
                        self._last_draw = 0.0
                    elif hit == "preview:crisp":
                        self.set_preview_exposure(0.7)
                        self.set_preview_spot_width(0.5)
                    elif hit == "preview:reset":
                        self.set_preview_exposure(self.specs["exposure"].default)
                        self.set_preview_spot_width(self.specs["spot"].default)
                    elif hit == "fullscreen:toggle":
                        self._actions.append(("fullscreen", not self.fullscreen))
                    elif hit == "image_only:toggle":
                        self._actions.append(("image_only", True))
                    elif hit == "audio:toggle":
                        self._actions.append((
                            "audio", bool(self._state.get("audio_muted", True))))
                    elif hit.startswith("key:"):
                        self._actions.append(("key", hit.split(":", 1)[1]))
                    return
        elif action == self.glfw.RELEASE and self._dragging is not None:
            name = self._dragging
            value = self._value_for_pointer(name, x)
            if value is None:
                value = self._drag_value
            self._dragging = None
            self._drag_value = None
            if name in ("exposure", "spot"):
                if name == "exposure":
                    self.set_preview_exposure(value)
                else:
                    self.set_preview_spot_width(value)
            else:
                self._actions.append(("slider", name, value))

    def _on_cursor(self, _window, x, _y):
        if self._dragging is None:
            return
        value = self._value_for_pointer(self._dragging, x)
        if value is not None:
            self._drag_value = value
            if self._dragging in ("exposure", "spot"):
                if self._dragging == "exposure":
                    self.set_preview_exposure(value)
                else:
                    self.set_preview_spot_width(value)

    def _on_key(self, _window, key, _scancode, action, _mods):
        if action != self.glfw.PRESS:
            return
        if key == self.glfw.KEY_ESCAPE:
            if self.image_only:
                self.set_image_only(False)
            elif self.fullscreen:
                self.set_fullscreen(False)
            else:
                self.close_requested = True
        elif key == self.glfw.KEY_Q:
            self.close_requested = True
        elif key == self.glfw.KEY_F11:
            self._actions.append(("fullscreen", not self.fullscreen))
        elif key == getattr(self.glfw, "KEY_F10", None):
            self._actions.append(("image_only", not self.image_only))
        elif key == getattr(self.glfw, "KEY_TAB", None):
            tabs = ("picture", "output", "preview")
            self._control_tab = tabs[(tabs.index(getattr(self, "_control_tab", "picture")) + 1) % 3]
            self._last_draw = 0.0
        elif key == self.glfw.KEY_V:
            self._actions.append(("key", "v"))
        elif key == self.glfw.KEY_I:
            self._actions.append(("key", "i"))
        elif key == self.glfw.KEY_R:
            self._actions.append(("key", "r"))
        elif key == self.glfw.KEY_M:
            self._actions.append(("key", "m"))

    def close(self):
        if self.closed:
            return
        self.closed = True
        self._preview_stop.set()
        if self._preview_thread is not None:
            self._preview_thread.join(timeout=1.0)
        self._release_graphics()

    def _release_graphics(self):
        try:
            if getattr(self, "texture", None) is not None:
                self.texture.release()
            if getattr(self, "vao", None) is not None:
                self.vao.release()
            if getattr(self, "program", None) is not None:
                self.program.release()
            if getattr(self, "context", None) is not None:
                self.context.release()
        except Exception:
            pass
        try:
            glfw = getattr(self, "glfw", None)
            window = getattr(self, "window", None)
            if glfw is not None:
                if window is not None:
                    glfw.destroy_window(window)
                glfw.terminate()
        except Exception:
            pass
