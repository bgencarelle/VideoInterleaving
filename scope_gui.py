"""Optional GLFW/ModernGL live tuner for scope mode.

The layout and interaction model follow the V7 receiver GUI, but this module
owns only scope controls and the scope trace preview.  Graphics dependencies
are imported when the GUI is constructed, so ordinary/headless scope startup
does not depend on them.
"""
from dataclasses import dataclass
import threading
import time


WINDOW_SIZE = (1360, 860)
PREVIEW_SIZE = 680
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


def _to_slider_value(name, state, exposure):
    if name == "exposure":
        return exposure
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
    return f"{value:g}"


class ScopeGUI:
    """Native scope preview and slider-based live controls.

    ``poll`` is called by scope_display's producer loop, so UI-originated
    commands are applied on the same thread that owns the renderer and audio
    stream.  Only the expensive phosphor preview is computed by a worker.
    """

    def __init__(self, initial_state, preview_exposure=1.0):
        self._state = dict(initial_state)
        self.preview_exposure = float(preview_exposure)
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
        self._preview_points = None
        self._rendered_exposure = None
        self.texture = None
        self.program = None
        self.vao = None
        try:
            self._init_window()
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
            window = glfw.create_window(*WINDOW_SIZE, "Scope · Live Tuner", None, None)
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
                    uv = vec2(p.x * 0.5, 1.0 - p.y * 0.5);
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
        glfw.set_window_close_callback(self.window,
                                       lambda _w: setattr(self, "close_requested", True))
        glfw.set_mouse_button_callback(self.window, self._on_mouse_button)
        glfw.set_cursor_pos_callback(self.window, self._on_cursor)
        glfw.set_key_callback(self.window, self._on_key)

    def _font(self, size, mono=False):
        names = (("DejaVuSansMono.ttf", "DejaVuSans.ttf") if mono else
                 ("DejaVuSans.ttf", "DejaVuSansMono.ttf"))
        for name in names:
            try:
                return self.ImageFont.truetype(name, size)
            except OSError:
                pass
        try:
            return self.ImageFont.load_default(size=size)
        except TypeError:
            return self.ImageFont.load_default()

    def _start_preview_worker(self):
        self._preview_thread = threading.Thread(
            target=self._preview_loop, name="ScopePreview", daemon=True)
        self._preview_thread.start()

    def _preview_loop(self):
        try:
            from scope_bake import preview_frame
            from scope_out import Scope
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
                render_pending = (seq != self._rendered_seq
                                 or changed_exposure)
                render_due = (time.monotonic() - getattr(
                    self, "_last_preview_render", 0.0) >= 1.0 / 15.0)
                if self._preview_points is not None and render_pending and render_due:
                    try:
                        frame = preview_frame(self._preview_points,
                                              size=PREVIEW_SIZE,
                                              exposure=exposure)
                        with self._preview_lock:
                            self._preview_rgb = frame
                            self._preview_error = ""
                        self._rendered_seq = self._preview_seq
                        self._rendered_exposure = exposure
                        self._last_preview_render = time.monotonic()
                    except Exception as exc:
                        with self._preview_lock:
                            self._preview_error = f"Preview error: {exc}"
            self._preview_stop.wait(0.025)

    def set_preview_exposure(self, value):
        self.preview_exposure = float(value)

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
        if now - self._last_draw >= 1.0 / 30.0:
            self._draw()
            self._last_draw = now
        actions, self._actions = self._actions, []
        return actions

    def _slider_disabled(self, name, state):
        if name in ("ips", "fps", "fields") and state.get("mode_locked"):
            return True
        if name == "ips" and state.get("clock_locked"):
            return True
        if name in ("fields", "rows") and state.get("mode") != "raster":
            return True
        return False

    def _slider_value(self, name, state):
        if name == "exposure":
            value = self.preview_exposure
        else:
            value = _to_slider_value(name, state, self.preview_exposure)
        if self._dragging == name and self._drag_value is not None:
            return self._drag_value
        return value

    def _draw(self):
        width, height = self.glfw.get_window_size(self.window)
        if width <= 0 or height <= 0:
            return
        image = self.Image.new("RGBA", (width, height), (12, 19, 26, 255))
        draw = self.ImageDraw.Draw(image)
        state = self._state
        metrics = self._metrics
        self._hits = {}

        draw.rectangle((0, 0, width, 56), fill=(10, 18, 25, 255))
        draw.text((22, 15), "SCOPE  ·  LIVE TUNER", fill=(239, 245, 249),
                  font=self.font)
        header = (f"{metrics.get('device', 'connecting')}   ·   "
                  f"{metrics.get('sample_rate', 0):,} Hz   ·   "
                  f"{str(state.get('mode', 'raster')).upper()}")
        draw.text((width - 520, 18), header, fill=(155, 187, 204),
                  font=self.small)

        preview_rect = (20, 76, 720, min(height - 56, 816))
        draw.rounded_rectangle(preview_rect, radius=6, fill=(6, 10, 13, 255),
                               outline=(49, 69, 83, 255), width=1)
        preview_size = min(PREVIEW_SIZE, preview_rect[3] - preview_rect[1] - 38,
                           preview_rect[2] - preview_rect[0] - 38)
        preview_x = preview_rect[0] + (preview_rect[2] - preview_rect[0]
                                       - preview_size) // 2
        preview_y = preview_rect[1] + 16
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
            draw.text((preview_rect[0] + 22, preview_rect[1] + 24), msg,
                      fill=(151, 174, 192), font=self.small)
        draw.text((preview_rect[0] + 16, preview_rect[3] - 25),
                  "Phosphor preview  ·  exposure changes this preview only",
                  fill=(119, 145, 160), font=self.tiny)

        panel = (740, 76, width - 20, height - 56)
        draw.rounded_rectangle(panel, radius=6, fill=(15, 24, 32, 255),
                               outline=(49, 69, 83, 255), width=1)
        draw.text((panel[0] + 18, panel[1] + 13), "Live controls",
                  fill=(231, 240, 246), font=self.font)
        draw.line((panel[0] + 18, panel[1] + 43, panel[2] - 18,
                   panel[1] + 43), fill=(42, 58, 70), width=1)
        draw.line((panel[2] - 182, panel[1] + 25, panel[2] - 182,
                   panel[1] + 34), fill=(243, 178, 85), width=2)
        draw.text((panel[2] - 174, panel[1] + 22), "startup default",
                  fill=(159, 173, 181), font=self.tiny)

        controls_top = panel[1] + 52
        track_left = panel[0] + 20
        track_right = panel[2] - 124
        for index, name in enumerate(SLIDER_RANGES):
            top = controls_top + index * SLIDER_ROW_HEIGHT
            spec = self.specs[name]
            value = self._slider_value(name, state)
            disabled = self._slider_disabled(name, state)
            label_color = (119, 137, 149) if disabled else (208, 220, 228)
            readout = _format_value(name, value)
            if disabled and name in ("fields", "rows"):
                readout = "Raster only"
            draw.text((track_left, top + 3), SLIDER_LABELS[name],
                      fill=label_color, font=self.small)
            draw.text((panel[2] - 112, top + 3), readout,
                      fill=(115, 132, 145) if disabled else (235, 242, 247),
                      font=self.tiny)
            y = top + 37
            track_color = (45, 61, 73) if disabled else (61, 78, 90)
            draw.rounded_rectangle((track_left, y - 2, track_right, y + 2),
                                   radius=2, fill=track_color)
            default_x = slider_default_x(track_left, track_right, spec)
            draw.line((default_x, y - 9, default_x, y + 9),
                      fill=(243, 178, 85) if not disabled else (112, 101, 81),
                      width=2)
            current_x = round(track_left + slider_fraction(value, spec)
                               * (track_right - track_left))
            if current_x > track_left:
                draw.rounded_rectangle((track_left, y - 2, current_x, y + 2),
                                       radius=2,
                                       fill=(63, 137, 174) if not disabled
                                       else (54, 66, 73))
            thumb = 6 if not disabled else 4
            draw.ellipse((current_x - thumb, y - thumb,
                          current_x + thumb, y + thumb),
                         fill=(205, 232, 245) if not disabled else (105, 119, 128),
                         outline=(66, 133, 164) if not disabled else (82, 94, 101))
            self._hits[f"slider:{name}"] = (track_left - 8, top,
                                             track_right + 8, top + 51)

        mode_y = controls_top + len(SLIDER_RANGES) * SLIDER_ROW_HEIGHT + 4
        mode_gap = 5
        mode_width = (panel[2] - panel[0] - 36 - mode_gap * 4) // 5
        for i, mode in enumerate(MODES):
            left = panel[0] + 18 + i * (mode_width + mode_gap)
            rect = (left, mode_y, left + mode_width, mode_y + 31)
            selected = state.get("mode") == mode
            locked = bool(state.get("mode_locked"))
            unsupported = mode not in state.get("available_modes", MODES)
            disabled = locked or unsupported
            fill = ((42, 83, 105) if selected else (19, 33, 43)) if not disabled else (
                (35, 48, 57) if selected else (17, 25, 31))
            outline = ((84, 153, 181) if selected else (46, 66, 79)) if not disabled else (
                (63, 78, 88) if selected else (38, 48, 56))
            draw.rounded_rectangle(rect, radius=4, fill=fill, outline=outline,
                                   width=1)
            text_width = self.small.getlength(mode.title())
            draw.text((left + (mode_width - text_width) / 2, mode_y + 8),
                      mode.title(),
                      fill=(231, 241, 247) if not disabled else (113, 127, 136),
                      font=self.small)
            if not disabled:
                self._hits[f"mode:{mode}"] = rect

        button_y = mode_y + 40
        for i, (key, label) in enumerate((("i", "Invert"),
                                          ("r", "Rotate"),
                                          ("m", "Mirror"))):
            left = panel[0] + 18 + i * 126
            rect = (left, button_y, left + 116, button_y + 31)
            draw.rounded_rectangle(rect, radius=4, fill=(20, 36, 47),
                                   outline=(53, 78, 94), width=1)
            draw.text((left + 12, button_y + 8), label,
                      fill=(216, 229, 237), font=self.small)
            self._hits[f"key:{key}"] = rect

        stats_y = min(height - 108, button_y + 47)
        sample_rate = int(metrics.get("sample_rate", 0) or 0)
        trace_hz = float(metrics.get("trace_hz", 0) or 0)
        picture_hz = float(metrics.get("picture_hz", 0) or 0)
        samples = int(metrics.get("samples", 0) or 0)
        grid = metrics.get("grid", "--") or "--"
        errors = (f"DAC dropouts {metrics.get('dropouts', 0)}  ·  "
                  f"generator underruns {metrics.get('underruns', 0)}")
        draw.text((panel[0] + 18, stats_y),
                  f"{trace_hz:.1f} traces/s  ·  {picture_hz:.1f} pictures/s  ·  "
                  f"{samples:,} samples/trace",
                  fill=(158, 190, 205), font=self.small)
        draw.text((panel[0] + 18, stats_y + 23),
                  f"Grid {grid}  ·  {metrics.get('fields', 1)} fields  ·  "
                  f"{sample_rate:,} Hz",
                  fill=(133, 158, 173), font=self.small)
        draw.text((panel[0] + 18, stats_y + 46), errors,
                  fill=(218, 153, 122) if (metrics.get("dropouts", 0)
                                            or metrics.get("underruns", 0))
                  else (119, 147, 161), font=self.tiny)
        note = self.message
        if metrics.get("message"):
            note = str(metrics["message"])
        if note:
            draw.text((panel[0] + 18, panel[3] - 35),
                      self._fit_text(note, self.tiny, panel[2] - panel[0] - 36),
                      fill=(243, 178, 85), font=self.tiny)
        draw.rectangle((0, height - 36, width, height), fill=(9, 15, 20, 255))
        draw.text((20, height - 25),
                  "XY has no independent intensity channel: adjust dwell/gamma "
                  "or the scope's intensity knob.  Esc / Q closes the window.",
                  fill=(151, 169, 179), font=self.tiny)

        framebuffer = self.glfw.get_framebuffer_size(self.window)
        if framebuffer[0] <= 0 or framebuffer[1] <= 0:
            return
        self.context.viewport = (0, 0, framebuffer[0], framebuffer[1])
        self.context.clear(0.04, 0.06, 0.08, 1.0)
        raw = image.tobytes()
        if self.texture is None or self.texture.size != image.size:
            if self.texture is not None:
                self.texture.release()
            self.texture = self.context.texture(image.size, 4, raw, alignment=1)
            self.texture.filter = (self.moderngl.LINEAR, self.moderngl.LINEAR)
        else:
            self.texture.write(raw, alignment=1)
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
        return slider_value_at(x, left + 8, right - 8, self.specs[name])

    def _on_mouse_button(self, _window, button, action, _mods):
        if button != self.glfw.MOUSE_BUTTON_LEFT:
            return
        x, y = self.glfw.get_cursor_pos(self.window)
        if action == self.glfw.PRESS:
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
                    if name == "exposure":
                        self.set_preview_exposure(self._drag_value)
                    return
            for hit, rect in self._hits.items():
                if rect[0] <= x <= rect[2] and rect[1] <= y <= rect[3]:
                    if hit.startswith("mode:"):
                        self._actions.append(("mode", hit.split(":", 1)[1]))
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
            if name == "exposure":
                self.set_preview_exposure(value)
            else:
                self._actions.append(("slider", name, value))

    def _on_cursor(self, _window, x, _y):
        if self._dragging is None:
            return
        value = self._value_for_pointer(self._dragging, x)
        if value is not None:
            self._drag_value = value
            if self._dragging == "exposure":
                self.set_preview_exposure(value)

    def _on_key(self, _window, key, _scancode, action, _mods):
        if action != self.glfw.PRESS:
            return
        if key in (self.glfw.KEY_ESCAPE, self.glfw.KEY_Q):
            self.close_requested = True
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
