"""
renderer.py – Unified Rendering Engine (GPU & CPU paths)
"""

import math
import os
import ctypes
import numpy as np
import moderngl
try:
    import cv2
except ImportError:
    cv2 = None
from settings import ENABLE_SRGB_FRAMEBUFFER, GAMMA_CORRECTION_ENABLED, BACKGROUND_COLOR

# Renderer backend selection:
# - "moderngl": requires GL3.3 or GLES3+
# - "legacy": PyOpenGL path for GL2.x / GLES2 contexts
_backend: str = "moderngl"
_legacy_gl = None  # Lazy import: OpenGL.GL module

# ModernGL context and objects
ctx = None
prog = None
vao = None
vbo = None

# Transformation parameters
_fs_scale: float = 1.0
_fs_offset_x: float = 0.0
_fs_offset_y: float = 0.0
_image_size: tuple[int, int] = (0, 0)
_rotation_angle: float = 0.0
_mirror_mode: int = 0

# Viewport / FBO size
_viewport_size: tuple[int, int] = (0, 0)
_quad_data: np.ndarray | None = None
_quad_dirty: bool = True

# Cached background color
_bg_linear_color: tuple[float, float, float] | None = None
_bg_linear_src: tuple[int, int, int] | None = None

# --- CPU OPTIMIZATION GLOBALS ---
_cpu_buffer: np.ndarray | None = None
_cached_canvas: np.ndarray | None = None

# --- LEGACY (PyOpenGL) PROGRAM OBJECTS ---
_legacy_program: int | None = None
_legacy_vbo: int | None = None
_legacy_mvp_loc: int | None = None
_legacy_bg_loc: int | None = None
_legacy_main_sbs_loc: int | None = None
_legacy_float_sbs_loc: int | None = None
_legacy_tex_main_loc: int | None = None
_legacy_tex_float_loc: int | None = None
_legacy_pos_loc: int | None = None
_legacy_uv_loc: int | None = None
_legacy_texture_dims: dict[int, tuple[int, int, int]] = {}


def using_legacy_gl() -> bool:
    return _backend == "legacy"


def _lazy_import_gl():
    global _legacy_gl
    if _legacy_gl is None:
        from OpenGL import GL as _gl  # type: ignore
        _legacy_gl = _gl
    return _legacy_gl


def _compile_shader_legacy(source: str, shader_type: int) -> int:
    gl = _lazy_import_gl()
    shader = gl.glCreateShader(shader_type)
    gl.glShaderSource(shader, source)
    gl.glCompileShader(shader)
    ok = gl.glGetShaderiv(shader, gl.GL_COMPILE_STATUS)
    if not ok:
        raise RuntimeError(gl.glGetShaderInfoLog(shader).decode("utf-8", errors="replace"))
    return shader


def _link_program_legacy(vertex_src: str, fragment_src: str) -> int:
    gl = _lazy_import_gl()
    vs = _compile_shader_legacy(vertex_src, gl.GL_VERTEX_SHADER)
    fs = _compile_shader_legacy(fragment_src, gl.GL_FRAGMENT_SHADER)
    program = gl.glCreateProgram()
    gl.glAttachShader(program, vs)
    gl.glAttachShader(program, fs)
    gl.glLinkProgram(program)
    ok = gl.glGetProgramiv(program, gl.GL_LINK_STATUS)
    if not ok:
        raise RuntimeError(gl.glGetProgramInfoLog(program).decode("utf-8", errors="replace"))
    gl.glDeleteShader(vs)
    gl.glDeleteShader(fs)
    return program


def initialize_legacy() -> None:
    """
    Initializes a minimal shader/VBO pipeline using PyOpenGL.
    This is used for GL2.x / GLES2 contexts where ModernGL cannot function.
    """
    global _backend
    global _legacy_program, _legacy_vbo
    global _legacy_mvp_loc, _legacy_bg_loc, _legacy_main_sbs_loc, _legacy_float_sbs_loc
    global _legacy_tex_main_loc, _legacy_tex_float_loc, _legacy_pos_loc, _legacy_uv_loc

    _backend = "legacy"
    gl = _lazy_import_gl()

    # Try GLES2 shader first; fall back to GLSL 1.20 (desktop GL 2.1).
    vertex_es2 = """
        #version 100
        attribute vec2 position;
        attribute vec2 texcoord;
        uniform mat4 u_MVP;
        varying vec2 v_texcoord;
        void main() {
            gl_Position = u_MVP * vec4(position, 0.0, 1.0);
            v_texcoord = texcoord;
        }
    """

    fragment_es2 = """
        #version 100
        precision mediump float;
        uniform sampler2D texture_main;
        uniform sampler2D texture_float;
        uniform vec3 u_bgColor;

        uniform bool u_main_is_sbs;
        uniform bool u_float_is_sbs;

        varying vec2 v_texcoord;

        vec4 sampleLayer(sampler2D tex, bool is_sbs, vec2 uv) {
            if (is_sbs) {
                vec2 uv_color = vec2(uv.x * 0.5, uv.y);
                vec2 uv_mask  = vec2((uv.x * 0.5) + 0.5, uv.y);
                vec3 color = texture2D(tex, uv_color).rgb;
                float alpha = texture2D(tex, uv_mask).r;
                if (alpha < 0.05) alpha = 0.0;
                return vec4(color, alpha);
            } else {
                return texture2D(tex, uv);
            }
        }

        void main() {
            vec4 mainC = sampleLayer(texture_main, u_main_is_sbs, v_texcoord);
            vec4 floatC = sampleLayer(texture_float, u_float_is_sbs, v_texcoord);

            vec3 color = mix(u_bgColor, mainC.rgb, mainC.a);
            color = mix(color, floatC.rgb, floatC.a);
            gl_FragColor = vec4(color, 1.0);
        }
    """

    vertex_120 = """
        #version 120
        attribute vec2 position;
        attribute vec2 texcoord;
        uniform mat4 u_MVP;
        varying vec2 v_texcoord;
        void main() {
            gl_Position = u_MVP * vec4(position, 0.0, 1.0);
            v_texcoord = texcoord;
        }
    """

    fragment_120 = """
        #version 120
        uniform sampler2D texture_main;
        uniform sampler2D texture_float;
        uniform vec3 u_bgColor;

        uniform bool u_main_is_sbs;
        uniform bool u_float_is_sbs;

        varying vec2 v_texcoord;

        vec4 sampleLayer(sampler2D tex, bool is_sbs, vec2 uv) {
            if (is_sbs) {
                vec2 uv_color = vec2(uv.x * 0.5, uv.y);
                vec2 uv_mask  = vec2((uv.x * 0.5) + 0.5, uv.y);
                vec3 color = texture2D(tex, uv_color).rgb;
                float alpha = texture2D(tex, uv_mask).r;
                if (alpha < 0.05) alpha = 0.0;
                return vec4(color, alpha);
            } else {
                return texture2D(tex, uv);
            }
        }

        void main() {
            vec4 mainC = sampleLayer(texture_main, u_main_is_sbs, v_texcoord);
            vec4 floatC = sampleLayer(texture_float, u_float_is_sbs, v_texcoord);

            vec3 color = mix(u_bgColor, mainC.rgb, mainC.a);
            color = mix(color, floatC.rgb, floatC.a);
            gl_FragColor = vec4(color, 1.0);
        }
    """

    try:
        _legacy_program = _link_program_legacy(vertex_es2, fragment_es2)
    except Exception:
        _legacy_program = _link_program_legacy(vertex_120, fragment_120)

    _legacy_mvp_loc = gl.glGetUniformLocation(_legacy_program, "u_MVP")
    _legacy_bg_loc = gl.glGetUniformLocation(_legacy_program, "u_bgColor")
    _legacy_main_sbs_loc = gl.glGetUniformLocation(_legacy_program, "u_main_is_sbs")
    _legacy_float_sbs_loc = gl.glGetUniformLocation(_legacy_program, "u_float_is_sbs")
    _legacy_tex_main_loc = gl.glGetUniformLocation(_legacy_program, "texture_main")
    _legacy_tex_float_loc = gl.glGetUniformLocation(_legacy_program, "texture_float")
    _legacy_pos_loc = gl.glGetAttribLocation(_legacy_program, "position")
    _legacy_uv_loc = gl.glGetAttribLocation(_legacy_program, "texcoord")

    _legacy_vbo = gl.glGenBuffers(1)

    gl.glUseProgram(_legacy_program)
    if _legacy_tex_main_loc is not None:
        gl.glUniform1i(_legacy_tex_main_loc, 0)
    if _legacy_tex_float_loc is not None:
        gl.glUniform1i(_legacy_tex_float_loc, 1)
    if _legacy_bg_loc is not None:
        gl.glUniform3f(_legacy_bg_loc, 0.0, 0.0, 0.0)
    if _legacy_main_sbs_loc is not None:
        gl.glUniform1i(_legacy_main_sbs_loc, 0)
    if _legacy_float_sbs_loc is not None:
        gl.glUniform1i(_legacy_float_sbs_loc, 0)

    # Default MVP: identity
    if _legacy_mvp_loc is not None:
        mvp = np.eye(4, dtype="f4")
        gl.glUniformMatrix4fv(_legacy_mvp_loc, 1, gl.GL_TRUE, mvp)
    gl.glUseProgram(0)


def _legacy_gl_formats(image: np.ndarray) -> tuple[int, int, bytes]:
    gl = _lazy_import_gl()
    if image.ndim == 2:
        fmt = gl.GL_LUMINANCE
        internal = gl.GL_LUMINANCE
    else:
        channels = image.shape[2]
        if channels == 3:
            fmt = gl.GL_RGB
            internal = gl.GL_RGB
        elif channels == 4:
            fmt = gl.GL_RGBA
            internal = gl.GL_RGBA
        else:
            raise ValueError(f"Unsupported channel count: {channels}")
    data = np.ascontiguousarray(image).tobytes()
    return internal, fmt, data

def _update_bg_linear(background_color: tuple[int, int, int]) -> tuple[float, float, float]:
    global _bg_linear_color, _bg_linear_src
    if _bg_linear_src == background_color and _bg_linear_color is not None:
        return _bg_linear_color

    def srgb_to_linear_component(c: int) -> float:
        return pow(c / 255.0, 2.2)

    if GAMMA_CORRECTION_ENABLED or ENABLE_SRGB_FRAMEBUFFER:
        _bg_linear_color = tuple(srgb_to_linear_component(c) for c in background_color)
    else:
        _bg_linear_color = (
            background_color[0] / 255.0,
            background_color[1] / 255.0,
            background_color[2] / 255.0,
        )
    _bg_linear_src = background_color
    return _bg_linear_color


def set_viewport_size(width: int, height: int) -> None:
    global _viewport_size, _quad_dirty
    _viewport_size = (width, height)
    _quad_dirty = True
    if _backend == "legacy":
        gl = _lazy_import_gl()
        gl.glViewport(0, 0, width, height)
    elif ctx is not None:
        ctx.viewport = (0, 0, width, height)


def initialize(gl_context: moderngl.Context) -> None:
    global _backend
    global ctx, prog, vao, vbo

    force_legacy = os.environ.get("FORCE_LEGACY_GL") in {"1", "true", "TRUE", "yes", "YES"}
    if force_legacy or (getattr(gl_context, "version_code", 0) < 300):
        # ModernGL does not reliably support GL2.x / GLES2 contexts (VAOs, GLSL 3xx, etc).
        ctx = None
        prog = None
        vao = None
        vbo = None
        initialize_legacy()
        return

    _backend = "moderngl"
    ctx = gl_context

    version_code = ctx.version_code
    is_gles = version_code < 330
    is_gles2 = is_gles and version_code < 300

    if is_gles2:
        vertex_src = """
            #version 100
            attribute vec2 position;
            attribute vec2 texcoord;
            uniform mat4 u_MVP;
            varying vec2 v_texcoord;
            void main() {
                gl_Position = u_MVP * vec4(position, 0.0, 1.0);
                v_texcoord = texcoord;
            }
        """

        fragment_src = """
            #version 100
            precision mediump float;
            uniform sampler2D texture_main;
            uniform sampler2D texture_float;
            uniform vec3 u_bgColor;

            uniform bool u_main_is_sbs;
            uniform bool u_float_is_sbs;

            varying vec2 v_texcoord;

            vec4 sampleLayer(sampler2D tex, bool is_sbs, vec2 uv) {
                if (is_sbs) {
                    vec2 uv_color = vec2(uv.x * 0.5, uv.y);
                    vec2 uv_mask  = vec2((uv.x * 0.5) + 0.5, uv.y);
                    vec3 color = texture2D(tex, uv_color).rgb;
                    float alpha = texture2D(tex, uv_mask).r;
                    if (alpha < 0.05) alpha = 0.0;
                    return vec4(color, alpha);
                } else {
                    return texture2D(tex, uv);
                }
            }

            void main() {
                vec4 mainC = sampleLayer(texture_main, u_main_is_sbs, v_texcoord);
                vec4 floatC = sampleLayer(texture_float, u_float_is_sbs, v_texcoord);

                vec3 color = mix(u_bgColor, mainC.rgb, mainC.a);
                color = mix(color, floatC.rgb, floatC.a);
                gl_FragColor = vec4(color, 1.0);
            }
        """
    else:
        header = "#version 300 es\nprecision mediump float;" if is_gles else "#version 330 core"

        vertex_src = f"""
            {header}
            in vec2 position;
            in vec2 texcoord;
            uniform mat4 u_MVP;
            out vec2 v_texcoord;
            void main() {{
                gl_Position = u_MVP * vec4(position, 0.0, 1.0);
                v_texcoord = texcoord;
            }}
        """

        fragment_src = f"""
            {header}
            uniform sampler2D texture_main;
            uniform sampler2D texture_float;
            uniform vec3 u_bgColor;

            uniform bool u_main_is_sbs;
            uniform bool u_float_is_sbs;

            in vec2 v_texcoord;
            out vec4 fragColor;

            vec4 sampleLayer(sampler2D tex, bool is_sbs, vec2 uv) {{
                if (is_sbs) {{
                    vec2 uv_color = vec2(uv.x * 0.5, uv.y);
                    vec2 uv_mask  = vec2((uv.x * 0.5) + 0.5, uv.y);
                    vec3 color = texture(tex, uv_color).rgb;
                    float alpha = texture(tex, uv_mask).r;
                    if (alpha < 0.05) alpha = 0.0;
                    return vec4(color, alpha);
                }} else {{
                    return texture(tex, uv);
                }}
            }}

            void main() {{
                vec4 mainC = sampleLayer(texture_main, u_main_is_sbs, v_texcoord);
                vec4 floatC = sampleLayer(texture_float, u_float_is_sbs, v_texcoord);

                vec3 color = mix(u_bgColor, mainC.rgb, mainC.a);
                color = mix(color, floatC.rgb, floatC.a);
                fragColor = vec4(color, 1.0);
            }}
        """

    prog = ctx.program(vertex_shader=vertex_src, fragment_shader=fragment_src)
    vbo = ctx.buffer(reserve=64)
    vao = ctx.vertex_array(prog, [(vbo, "2f 2f", "position", "texcoord")])

    mvp = np.eye(4, dtype="f4")
    prog["u_MVP"].write(mvp.T.tobytes())
    ctx.disable(moderngl.BLEND)

    prog["texture_main"].value = 0
    prog["texture_float"].value = 1
    prog["u_bgColor"].value = (0.0, 0.0, 0.0)
    prog["u_main_is_sbs"].value = False
    prog["u_float_is_sbs"].value = False


def set_transform_parameters(fs_scale, fs_offset_x, fs_offset_y, image_size, rotation_angle, mirror_mode):
    global _fs_scale, _fs_offset_x, _fs_offset_y, _image_size, _rotation_angle, _mirror_mode, _quad_dirty
    _fs_scale = fs_scale
    _fs_offset_x = fs_offset_x
    _fs_offset_y = fs_offset_y
    _image_size = image_size
    _rotation_angle = rotation_angle
    _mirror_mode = mirror_mode
    _quad_dirty = True


def update_mvp(mvp_matrix):
    if _backend == "legacy":
        if _legacy_program is None:
            initialize_legacy()
        gl = _lazy_import_gl()
        gl.glUseProgram(_legacy_program)
        if _legacy_mvp_loc is not None:
            gl.glUniformMatrix4fv(_legacy_mvp_loc, 1, gl.GL_TRUE, mvp_matrix.astype("f4"))
        gl.glUseProgram(0)
        return

    if prog is not None:
        prog["u_MVP"].write(mvp_matrix.T.tobytes())


def create_texture(image: np.ndarray) -> moderngl.Texture:
    if _backend == "legacy":
        if _legacy_program is None:
            initialize_legacy()
        gl = _lazy_import_gl()
        tex_id = gl.glGenTextures(1)
        gl.glBindTexture(gl.GL_TEXTURE_2D, tex_id)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MIN_FILTER, gl.GL_LINEAR)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MAG_FILTER, gl.GL_LINEAR)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_WRAP_S, gl.GL_CLAMP_TO_EDGE)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_WRAP_T, gl.GL_CLAMP_TO_EDGE)

        h, w = image.shape[:2]
        internal, fmt, data = _legacy_gl_formats(image)
        gl.glTexImage2D(gl.GL_TEXTURE_2D, 0, internal, w, h, 0, fmt, gl.GL_UNSIGNED_BYTE, data)

        components = 1 if image.ndim == 2 else image.shape[2]
        _legacy_texture_dims[tex_id] = (w, h, components)
        return tex_id  # type: ignore[return-value]

    h, w = image.shape[:2]
    components = 3 if (image.ndim == 3 and image.shape[2] == 3) else 4
    if image.ndim == 2:
        components = 1
    tex = ctx.texture((w, h), components, data=image.tobytes())
    tex.filter = (moderngl.LINEAR, moderngl.LINEAR)
    return tex


def update_texture(texture: moderngl.Texture, new_image: np.ndarray) -> moderngl.Texture:
    if _backend == "legacy":
        gl = _lazy_import_gl()
        tex_id = int(texture)  # type: ignore[arg-type]
        h, w = new_image.shape[:2]
        components = 1 if new_image.ndim == 2 else new_image.shape[2]
        expected = _legacy_texture_dims.get(tex_id)

        gl.glBindTexture(gl.GL_TEXTURE_2D, tex_id)
        internal, fmt, data = _legacy_gl_formats(new_image)

        if expected != (w, h, components):
            gl.glTexImage2D(gl.GL_TEXTURE_2D, 0, internal, w, h, 0, fmt, gl.GL_UNSIGNED_BYTE, data)
            _legacy_texture_dims[tex_id] = (w, h, components)
            return tex_id  # type: ignore[return-value]

        gl.glTexSubImage2D(gl.GL_TEXTURE_2D, 0, 0, 0, w, h, fmt, gl.GL_UNSIGNED_BYTE, data)
        return tex_id  # type: ignore[return-value]

    new_h, new_w = new_image.shape[:2]
    new_c = 3 if (new_image.ndim == 3 and new_image.shape[2] == 3) else 4
    if new_image.ndim == 2:
        new_c = 1

    if (new_w, new_h) != texture.size or new_c != texture.components:
        texture.release()
        return create_texture(new_image)
    texture.write(new_image.tobytes())
    return texture


def compute_transformed_quad():
    global _quad_data
    view_w, view_h = _viewport_size

    # Headless / Web: Letterbox inside viewport
    if view_w > 0 and view_h > 0:
        img_w, img_h = _image_size

        target_aspect = view_w / view_h
        image_aspect = (img_w / img_h) if img_h > 0 else 1.0

        if image_aspect > target_aspect:
            x_scale = 1.0
            y_scale = target_aspect / image_aspect
        else:
            x_scale = image_aspect / target_aspect
            y_scale = 1.0

        # Flip Y for Headless
        positions = [
            (-x_scale, y_scale), (x_scale, y_scale),
            (x_scale, -y_scale), (-x_scale, -y_scale)
        ]
        tex_coords = [(0.0, 1.0), (1.0, 1.0), (1.0, 0.0), (0.0, 0.0)]
        if _mirror_mode:
            tex_coords = [(1.0, 1.0), (0.0, 1.0), (0.0, 0.0), (1.0, 0.0)]

        data = []
        for (px, py), (tu, tv) in zip(positions, tex_coords):
            data += [px, py, tu, tv]
        _quad_data = np.array(data, dtype=np.float32)
        return _quad_data

    # Local: Standard fit
    w, h = _image_size
    if w == 0 or h == 0:
        return np.array([-1, 1, 0, 1, 1, 1, 1, 1, 1, -1, 1, 0, -1, -1, 0, 0], dtype=np.float32)

    w_sc, h_sc = w * _fs_scale, h * _fs_scale
    if _rotation_angle % 180 == 90:
        ew, eh = h_sc, w_sc
    else:
        ew, eh = w_sc, h_sc

    cx, cy = _fs_offset_x + ew / 2, _fs_offset_y + eh / 2
    pts = [(-w_sc / 2, -h_sc / 2), (w_sc / 2, -h_sc / 2), (w_sc / 2, h_sc / 2), (-w_sc / 2, h_sc / 2)]

    rad = math.radians(_rotation_angle)
    c, s = math.cos(rad), math.sin(rad)
    rot_pts = [(x * c - y * s + cx, x * s + y * c + cy) for x, y in pts]
    uvs = [(1, 1), (0, 1), (0, 0), (1, 0)] if _mirror_mode else [(0, 1), (1, 1), (1, 0), (0, 0)]

    data = []
    for (px, py), (u, v) in zip(rot_pts, uvs):
        data.extend([px, py, u, v])
    _quad_data = np.array(data, dtype=np.float32)
    return _quad_data


def overlay_images_single_pass(main_texture, float_texture, background_color=(0, 0, 0),
                               main_is_sbs=False, float_is_sbs=False):
    global _quad_dirty
    if _backend == "legacy":
        if _legacy_program is None or _legacy_vbo is None:
            initialize_legacy()
        gl = _lazy_import_gl()

        bg_linear = _update_bg_linear(background_color)
        gl.glClearColor(bg_linear[0], bg_linear[1], bg_linear[2], 1.0)
        gl.glClear(gl.GL_COLOR_BUFFER_BIT)

        gl.glUseProgram(_legacy_program)
        if _legacy_bg_loc is not None:
            gl.glUniform3f(_legacy_bg_loc, bg_linear[0], bg_linear[1], bg_linear[2])
        if _legacy_main_sbs_loc is not None:
            gl.glUniform1i(_legacy_main_sbs_loc, 1 if main_is_sbs else 0)
        if _legacy_float_sbs_loc is not None:
            gl.glUniform1i(_legacy_float_sbs_loc, 1 if float_is_sbs else 0)

        if _quad_dirty or _quad_data is None:
            quad = compute_transformed_quad().astype(np.float32, copy=False)
            gl.glBindBuffer(gl.GL_ARRAY_BUFFER, _legacy_vbo)
            gl.glBufferData(gl.GL_ARRAY_BUFFER, quad.nbytes, quad, gl.GL_DYNAMIC_DRAW)
            _quad_dirty = False
        else:
            gl.glBindBuffer(gl.GL_ARRAY_BUFFER, _legacy_vbo)

        if _legacy_pos_loc is not None and _legacy_pos_loc >= 0:
            gl.glEnableVertexAttribArray(_legacy_pos_loc)
            gl.glVertexAttribPointer(_legacy_pos_loc, 2, gl.GL_FLOAT, gl.GL_FALSE, 16, ctypes.c_void_p(0))
        if _legacy_uv_loc is not None and _legacy_uv_loc >= 0:
            gl.glEnableVertexAttribArray(_legacy_uv_loc)
            gl.glVertexAttribPointer(_legacy_uv_loc, 2, gl.GL_FLOAT, gl.GL_FALSE, 16, ctypes.c_void_p(8))

        gl.glActiveTexture(gl.GL_TEXTURE0)
        gl.glBindTexture(gl.GL_TEXTURE_2D, int(main_texture))
        gl.glActiveTexture(gl.GL_TEXTURE1)
        gl.glBindTexture(gl.GL_TEXTURE_2D, int(float_texture))

        gl.glDrawArrays(gl.GL_TRIANGLE_FAN, 0, 4)

        if _legacy_pos_loc is not None and _legacy_pos_loc >= 0:
            gl.glDisableVertexAttribArray(_legacy_pos_loc)
        if _legacy_uv_loc is not None and _legacy_uv_loc >= 0:
            gl.glDisableVertexAttribArray(_legacy_uv_loc)

        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, 0)
        gl.glUseProgram(0)
        return

    if ctx is None:
        return
    bg_linear = _update_bg_linear(background_color)
    if prog:
        prog["u_bgColor"].value = bg_linear
        prog["u_main_is_sbs"].value = main_is_sbs
        prog["u_float_is_sbs"].value = float_is_sbs
    ctx.clear(*bg_linear)
    if _quad_dirty or _quad_data is None:
        vbo.write(compute_transformed_quad().tobytes())
        _quad_dirty = False
    main_texture.use(location=0)
    float_texture.use(location=1)
    vao.render(mode=moderngl.TRIANGLE_FAN)


# --- CPU COMPOSITOR (Fixed Aspect Ratio & Alpha) ---
def composite_cpu(main_img, float_img, main_is_sbs=False, float_is_sbs=False, target_size=None):
    global _cpu_buffer, _cached_canvas

    if main_img is None and float_img is None: return None

    # Helper: Get view of content (No Copy)
    def get_views(img, is_sbs):
        if img is None: return None, None
        h, w = img.shape[:2]
        if is_sbs:
            mid = w // 2
            return img[:, :mid], img[:, mid:, 0]
        else:
            if img.ndim == 2:
                # Grayscale to RGB (Expensive copy, unavoidable)
                return (cv2.cvtColor(img, cv2.COLOR_GRAY2RGB) if cv2 else np.stack((img,) * 3, axis=-1)), None
            elif img.shape[2] == 4:
                return img[..., :3], img[..., 3]
            else:
                return img, None

    m_rgb, m_a = get_views(main_img, main_is_sbs)
    f_rgb, f_a = get_views(float_img, float_is_sbs)

    # 1. Determine Source Dimensions
    if m_rgb is not None:
        th, tw = m_rgb.shape[:2]
    elif f_rgb is not None:
        th, tw = f_rgb.shape[:2]
    else:
        return None

    # 2. Manage Source Buffer (Lazy Instantiation)
    if _cpu_buffer is None or _cpu_buffer.shape[:2] != (th, tw):
        _cpu_buffer = np.empty((th, tw, 3), dtype=np.uint8)

    # Fast Background Fill
    bg_r, bg_g, bg_b = BACKGROUND_COLOR
    _cpu_buffer[:] = (bg_r, bg_g, bg_b)

    # 3. Apply Main Layer
    if m_rgb is not None:
        if m_a is None:
            np.copyto(_cpu_buffer, m_rgb)
        else:
            # OPTIMIZED BLEND: Integer Math to avoid float casting
            mask = m_a > 0
            if np.any(mask):
                alpha = m_a[mask][:, None].astype(np.uint16)
                src = m_rgb[mask].astype(np.uint16)
                dst = _cpu_buffer[mask].astype(np.uint16)

                blended = (src * alpha + dst * (255 - alpha)) // 255
                _cpu_buffer[mask] = blended.astype(np.uint8)

    # 4. Apply Float Layer
    if f_rgb is not None:
        fh, fw = f_rgb.shape[:2]
        h, w = min(th, fh), min(tw, fw)

        # Define views for cropping
        target_view = _cpu_buffer[:h, :w]
        source_view = f_rgb[:h, :w]

        if f_a is None:
            target_view[:] = source_view
        else:
            alpha_view = f_a[:h, :w]
            mask = alpha_view > 0
            if np.any(mask):
                alpha = alpha_view[mask][:, None].astype(np.uint16)
                src = source_view[mask].astype(np.uint16)
                dst = target_view[mask].astype(np.uint16)

                blended = (src * alpha + dst * (255 - alpha)) // 255
                target_view[mask] = blended.astype(np.uint8)

    # 5. Final Resize (Letterboxed) with CACHING
    if target_size is not None and cv2 is not None:
        target_w, target_h = target_size

        if (tw, th) == (target_w, target_h):
            return _cpu_buffer

        # Calculate Scale
        scale = min(target_w / tw, target_h / th)
        new_w = int(tw * scale)
        new_h = int(th * scale)

        # Fast Resize
        resized = cv2.resize(_cpu_buffer, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        # Cache Canvas
        if _cached_canvas is None or _cached_canvas.shape[:2] != (target_h, target_w):
            _cached_canvas = np.full((target_h, target_w, 3), (bg_r, bg_g, bg_b), dtype=np.uint8)
        else:
            _cached_canvas[:] = (bg_r, bg_g, bg_b)

        # Center Paste
        y_off = (target_h - new_h) // 2
        x_off = (target_w - new_w) // 2
        _cached_canvas[y_off:y_off + new_h, x_off:x_off + new_w] = resized

        return _cached_canvas

    return _cpu_buffer
