"""
renderer.py – Unified Rendering Engine (GPU & CPU paths)
"""

import math
import numpy as np
import moderngl
from settings import ENABLE_SRGB_FRAMEBUFFER, GAMMA_CORRECTION_ENABLED, BACKGROUND_COLOR

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
    if ctx is not None:
        ctx.viewport = (0, 0, width, height)


def initialize(gl_context: moderngl.Context) -> None:
    global ctx, prog, vao, vbo
    ctx = gl_context

    # Handle GLES (Raspberry Pi / Mobile) vs Desktop GL
    is_gles = ctx.version_code < 330
    header = "#version 310 es\nprecision mediump float;" if is_gles else "#version 330 core"

    # Vertex Shader
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

    # Fragment Shader (Universal SBS/Standard)
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
    if prog is not None:
        prog["u_MVP"].write(mvp_matrix.T.tobytes())


def create_texture(image: np.ndarray) -> moderngl.Texture:
    h, w = image.shape[:2]
    components = 3 if (image.ndim == 3 and image.shape[2] == 3) else 4
    if image.ndim == 2: components = 1
    tex = ctx.texture((w, h), components, data=image.tobytes())
    tex.filter = (moderngl.LINEAR, moderngl.LINEAR)
    return tex


def update_texture(texture: moderngl.Texture, new_image: np.ndarray) -> moderngl.Texture:
    new_h, new_w = new_image.shape[:2]
    new_c = 3 if (new_image.ndim == 3 and new_image.shape[2] == 3) else 4
    if new_image.ndim == 2: new_c = 1

    if (new_w, new_h) != texture.size or new_c != texture.components:
        texture.release()
        return create_texture(new_image)
    else:
        texture.write(new_image.tobytes())
        return texture


def compute_transformed_quad():
    global _quad_data
    view_w, view_h = _viewport_size

    # Logic for Headless/Web aspect ratio correctness
    if view_w > 0 and view_h > 0:
        img_w, img_h = _image_size
        r_tex = (img_w / img_h) if img_h > 0 else 1.0
        r_view = (view_w / view_h) if view_h > 0 else 1.0

        if r_tex > r_view:
            x_scale, y_scale = 1.0, r_view / r_tex
        else:
            x_scale, y_scale = r_tex / r_view, 1.0

        positions = [(-x_scale, y_scale), (x_scale, y_scale), (x_scale, -y_scale), (-x_scale, -y_scale)]
        tex_coords = [(1.0, 1.0), (0.0, 1.0), (0.0, 0.0), (1.0, 0.0)] if _mirror_mode else [(0.0, 1.0), (1.0, 1.0),
                                                                                            (1.0, 0.0), (0.0, 0.0)]
        data = []
        for (px, py), (tu, tv) in zip(positions, tex_coords):
            data += [px, py, tu, tv]
        _quad_data = np.array(data, dtype=np.float32)
        return _quad_data

    # Standard Local Window logic
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
    if ctx is None: return
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


# --- OPTIMIZED CPU COMPOSITOR (No CV2, No Rotation) ---
def composite_cpu(main_img, float_img, main_is_sbs=False, float_is_sbs=False):
    if main_img is None and float_img is None: return None

    # Helper: Get view of content without copying data
    def get_views(img, is_sbs):
        if img is None: return None, None
        h, w = img.shape[:2]
        if is_sbs:
            mid = w // 2
            # Slice: RGB is left, Alpha is right (Red channel)
            return img[:, :mid], img[:, mid:, 0]
        else:
            # Standard RGBA (Strict Loader guarantees 3 or 4 channels)
            if img.shape[2] == 4:
                return img[..., :3], img[..., 3]
            else:
                return img, None  # Opaque RGB

    # 1. Prepare Base Canvas
    m_rgb, m_a = get_views(main_img, main_is_sbs)

    if m_rgb is not None:
        # Copy to avoid corrupting cache
        canvas = m_rgb.copy()
    else:
        # Fallback background
        f_rgb, _ = get_views(float_img, float_is_sbs)
        h, w = f_rgb.shape[:2]
        bg_r, bg_g, bg_b = BACKGROUND_COLOR
        canvas = np.full((h, w, 3), (bg_r, bg_g, bg_b), dtype=np.uint8)

    # 2. Blend Float Layer
    f_rgb, f_a = get_views(float_img, float_is_sbs)

    if f_rgb is not None:
        ch, cw = canvas.shape[:2]
        fh, fw = f_rgb.shape[:2]
        h, w = min(ch, fh), min(cw, fw)

        target_slice = canvas[:h, :w]
        source_slice = f_rgb[:h, :w]

        if f_a is None:
            target_slice[:] = source_slice
        else:
            # Vectorized Alpha Blend
            alpha = f_a[:h, :w].astype(np.uint16)[:, :, None]
            src = source_slice.astype(np.uint16)
            dst = target_slice.astype(np.uint16)
            blended = (src * alpha + dst * (255 - alpha)) // 255
            target_slice[:] = blended.astype(np.uint8)

    return canvas