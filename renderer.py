"""
renderer.py – ModernGL rendering for main and float images (textured quad)
Single-pass compositing, tuned for llvmpipe / CPU‐rendered GL.
"""

import math
import numpy as np
import moderngl
import cv2
from settings import ENABLE_SRGB_FRAMEBUFFER, GAMMA_CORRECTION_ENABLED, BACKGROUND_COLOR


# ModernGL context and objects (initialized in initialize())
ctx = None            # type: ignore[assignment]
prog = None           # type: ignore[assignment]
vao = None            # type: ignore[assignment]
vbo = None            # type: ignore[assignment]

# Transformation parameters (set via display_manager or elsewhere)
_fs_scale: float = 1.0
_fs_offset_x: float = 0.0
_fs_offset_y: float = 0.0
_image_size: tuple[int, int] = (0, 0)   # (width, height)
_rotation_angle: float = 0.0
_mirror_mode: int = 0

# Viewport / FBO size (for headless aspect-correct quad)
_viewport_size: tuple[int, int] = (0, 0)

# Cached quad data & dirty flag (avoid recomputing every frame)
_quad_data: np.ndarray | None = None
_quad_dirty: bool = True

# Cached background clear color in linear space
_bg_linear_color: tuple[float, float, float] | None = None
_bg_linear_src: tuple[int, int, int] | None = None


def _update_bg_linear(background_color: tuple[int, int, int]) -> tuple[float, float, float]:
    """
    Cache the gamma-corrected background color so we don't redo pow()
    in Python for every frame.
    """
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
    """Record the viewport (FBO or window) size and set GL viewport."""
    global _viewport_size, _quad_dirty
    _viewport_size = (width, height)
    _quad_dirty = True  # geometry depends on viewport
    if ctx is not None:
        ctx.viewport = (0, 0, width, height)


def initialize(gl_context: moderngl.Context) -> None:
    """Initialize ModernGL shaders, buffer, MVP, and compositing state."""
    global ctx, prog, vao, vbo
    ctx = gl_context

    is_gles = ctx.version_code < 320  # GL ES typically reports < 320

    if is_gles:
        vertex_src = """
            #version 310 es
            precision mediump float;
            in vec2 position;
            in vec2 texcoord;
            uniform mat4 u_MVP;
            out vec2 v_texcoord;
            void main() {
                gl_Position = u_MVP * vec4(position, 0.0, 1.0);
                v_texcoord = texcoord;
            }
        """
        if GAMMA_CORRECTION_ENABLED or ENABLE_SRGB_FRAMEBUFFER:
            fragment_src = """
                #version 310 es
                precision mediump float;
                uniform sampler2D texture_main;
                uniform sampler2D texture_float;
                uniform vec3 u_bgColor;
                in vec2 v_texcoord;
                out vec4 fragColor;
                void main() {
                    vec4 mainColor  = texture(texture_main,  v_texcoord);
                    vec4 floatColor = texture(texture_float, v_texcoord);

                    // background -> main(alpha) -> float(alpha)
                    vec3 color = mix(u_bgColor, mainColor.rgb,  mainColor.a);
                    color = mix(color, floatColor.rgb, floatColor.a);

                    // apply gamma to final color like old path
                    color = pow(color, vec3(2.2));
                    fragColor = vec4(color, 1.0);
                }
            """
        else:
            fragment_src = """
                #version 310 es
                precision mediump float;
                uniform sampler2D texture_main;
                uniform sampler2D texture_float;
                uniform vec3 u_bgColor;
                in vec2 v_texcoord;
                out vec4 fragColor;
                void main() {
                    vec4 mainColor  = texture(texture_main,  v_texcoord);
                    vec4 floatColor = texture(texture_float, v_texcoord);

                    vec3 color = mix(u_bgColor, mainColor.rgb,  mainColor.a);
                    color = mix(color, floatColor.rgb, floatColor.a);

                    fragColor = vec4(color, 1.0);
                }
            """
    else:
        vertex_src = """
            #version 330 core
            in vec2 position;
            in vec2 texcoord;
            uniform mat4 u_MVP;
            out vec2 v_texcoord;
            void main() {
                gl_Position = u_MVP * vec4(position, 0.0, 1.0);
                v_texcoord = texcoord;
            }
        """
        if GAMMA_CORRECTION_ENABLED or ENABLE_SRGB_FRAMEBUFFER:
            fragment_src = """
                #version 330 core
                uniform sampler2D texture_main;
                uniform sampler2D texture_float;
                uniform vec3 u_bgColor;
                in vec2 v_texcoord;
                out vec4 fragColor;
                void main() {
                    vec4 mainColor  = texture(texture_main,  v_texcoord);
                    vec4 floatColor = texture(texture_float, v_texcoord);

                    vec3 color = mix(u_bgColor, mainColor.rgb,  mainColor.a);
                    color = mix(color, floatColor.rgb, floatColor.a);

                    color = pow(color, vec3(2.2));
                    fragColor = vec4(color, 1.0);
                }
            """
        else:
            fragment_src = """
                #version 330 core
                uniform sampler2D texture_main;
                uniform sampler2D texture_float;
                uniform vec3 u_bgColor;
                in vec2 v_texcoord;
                out vec4 fragColor;
                void main() {
                    vec4 mainColor  = texture(texture_main,  v_texcoord);
                    vec4 floatColor = texture(texture_float, v_texcoord);

                    vec3 color = mix(u_bgColor, mainColor.rgb,  mainColor.a);
                    color = mix(color, floatColor.rgb, floatColor.a);

                    fragColor = vec4(color, 1.0);
                }
            """

    prog = ctx.program(vertex_shader=vertex_src, fragment_shader=fragment_src)

    # Small VBO; we overwrite it with quad vertices each time the quad changes.
    vbo = ctx.buffer(reserve=64)
    vao = ctx.vertex_array(prog, [(vbo, "2f 2f", "position", "texcoord")])

    # Default MVP = identity so clip-space positions “just work”
    mvp = np.eye(4, dtype="f4")
    prog["u_MVP"].write(mvp.T.tobytes())

    # Single-pass shader does its own layering; GL blending not needed
    ctx.disable(moderngl.BLEND)

    # Bind samplers to texture units 0 and 1
    prog["texture_main"].value = 0
    prog["texture_float"].value = 1

    # Initialize bg color uniform to black (will be updated per-frame)
    if "u_bgColor" in prog:
        prog["u_bgColor"].value = (0.0, 0.0, 0.0)


def set_transform_parameters(fs_scale: float,
                             fs_offset_x: float,
                             fs_offset_y: float,
                             image_size: tuple[int, int],
                             rotation_angle: float,
                             mirror_mode: int) -> None:
    """
    Update transformation parameters used for rendering (scale, offsets, etc.).
    Called from display_manager / image_display when layout changes.
    """
    global _fs_scale, _fs_offset_x, _fs_offset_y, _image_size, _rotation_angle, _mirror_mode, _quad_dirty
    _fs_scale = fs_scale
    _fs_offset_x = fs_offset_x
    _fs_offset_y = fs_offset_y
    _image_size = image_size
    _rotation_angle = rotation_angle
    _mirror_mode = mirror_mode
    _quad_dirty = True  # geometry depends on these


def update_mvp(mvp_matrix: np.ndarray) -> None:
    """Update the MVP matrix uniform for the shader (called when viewport changes)."""
    if prog is not None:
        prog["u_MVP"].write(mvp_matrix.T.tobytes())  # column-major for OpenGL


def create_texture(image: np.ndarray) -> moderngl.Texture:
    """Create a new GPU texture from a NumPy RGBA image array."""
    h, w = image.shape[0], image.shape[1]
    tex = ctx.texture((w, h), 4, data=image.tobytes())
    tex.filter = (moderngl.LINEAR, moderngl.LINEAR)
    tex.repeat_x = False
    tex.repeat_y = False
    return tex


def update_texture(texture: moderngl.Texture, new_image: np.ndarray) -> moderngl.Texture:
    """Update an existing texture with a new image (recreate if dimensions differ)."""
    new_h, new_w = new_image.shape[0], new_image.shape[1]
    if (new_w, new_h) != texture.size:
        texture.release()
        tex = ctx.texture((new_w, new_h), 4, data=new_image.tobytes())
        tex.filter = (moderngl.LINEAR, moderngl.LINEAR)
        tex.repeat_x = False
        tex.repeat_y = False
        return tex
    else:
        texture.write(new_image.tobytes())
        return texture


def compute_transformed_quad() -> np.ndarray:
    """
    Compute the positions and texture coordinates of the quad.

    Headless mode (viewport known):
        - Build aspect-correct quad in clip space [-1,1]x[-1,1].

    Legacy mode (viewport == (0,0)):
        - Use original pixel-space pipeline with rotation/scale/offset.
    """
    global _quad_data

    view_w, view_h = _viewport_size

    # --- Headless aspect-correct path (viewport known) ---
    if view_w > 0 and view_h > 0:
        img_w, img_h = _image_size

        if img_w <= 0 or img_h <= 0:
            r_tex = 1.0
        else:
            r_tex = img_w / img_h

        r_view = view_w / view_h

        if r_tex > 0 and r_view > 0:
            if r_tex > r_view:
                x_scale = 1.0
                y_scale = r_view / r_tex
            else:
                y_scale = 1.0
                x_scale = r_tex / r_view
        else:
            x_scale = 1.0
            y_scale = 1.0

        positions = [
            (-x_scale, y_scale),
            ( x_scale, y_scale),
            ( x_scale,  -y_scale),
            (-x_scale,  -y_scale),
        ]

        if _mirror_mode:
            tex_coords = [(1.0, 1.0), (0.0, 1.0), (0.0, 0.0), (1.0, 0.0)]
        else:
            tex_coords = [(0.0, 1.0), (1.0, 1.0), (1.0, 0.0), (0.0, 0.0)]

        data = []
        for (px, py), (tu, tv) in zip(positions, tex_coords):
            data += [px, py, tu, tv]
        _quad_data = np.array(data, dtype=np.float32)
        return _quad_data

    # --- Legacy pixel-space path (local window, using MVP) ---
    w, h = _image_size
    if w == 0 or h == 0:
        if _mirror_mode:
            tex_coords = [(1.0, 1.0), (0.0, 1.0), (0.0, 0.0), (1.0, 0.0)]
        else:
            tex_coords = [(0.0, 1.0), (1.0, 1.0), (1.0, 0.0), (0.0, 0.0)]
        positions = [
            (-1.0, 1.0),
            ( 1.0, 1.0),
            ( 1.0,  -1.0),
            (-1.0,  -1.0),
        ]
        data = []
        for (px, py), (tu, tv) in zip(positions, tex_coords):
            data += [px, py, tu, tv]
        _quad_data = np.array(data, dtype=np.float32)
        return _quad_data

    w_scaled = w * _fs_scale
    h_scaled = h * _fs_scale
    if _rotation_angle % 180 == 90:
        effective_w = h_scaled
        effective_h = w_scaled
    else:
        effective_w = w_scaled
        effective_h = h_scaled

    cx = _fs_offset_x + effective_w / 2.0
    cy = _fs_offset_y + effective_h / 2.0
    corners = [(-w_scaled / 2, -h_scaled / 2),
               ( w_scaled / 2, -h_scaled / 2),
               ( w_scaled / 2,  h_scaled / 2),
               (-w_scaled / 2,  h_scaled / 2)]

    rad = math.radians(_rotation_angle)
    cosA, sinA = math.cos(rad), math.sin(rad)
    rotated = [(x * cosA - y * sinA, x * sinA + y * cosA) for (x, y) in corners]
    final_positions = [(cx + x, cy + y) for (x, y) in rotated]

    if _mirror_mode:
        tex_coords = [(1.0, 1.0), (0.0, 1.0), (0.0, 0.0), (1.0, 0.0)]
    else:
        tex_coords = [(0.0, 1.0), (1.0, 1.0), (1.0, 0.0), (0.0, 0.0)]

    data = []
    for (px, py), (tu, tv) in zip(final_positions, tex_coords):
        data += [px, py, tu, tv]
    _quad_data = np.array(data, dtype=np.float32)
    return _quad_data


def overlay_images_single_pass(main_texture: moderngl.Texture,
                               float_texture: moderngl.Texture,
                               background_color: tuple[int, int, int] = (0, 0, 0)) -> None:
    """
    Clear with background_color, then draw a single quad sampling both
    main and float textures in one pass. Much cheaper on llvmpipe than
    two full-screen passes with blending.
    """
    global _quad_dirty

    # If initialize() hasn't run, do nothing (defensive).
    if ctx is None or prog is None or vao is None or vbo is None:
        return

    # Compute / cache background in linear or normalized space
    bg_linear = _update_bg_linear(background_color)

    # Pass background to shader; clear as well (mostly cosmetic, quad covers screen).
    if "u_bgColor" in prog:
        prog["u_bgColor"].value = bg_linear
    ctx.clear(*bg_linear)

    # Update quad data only when something changed
    if _quad_dirty or _quad_data is None:
        quad_data = compute_transformed_quad()
        vbo.write(quad_data.tobytes())
        _quad_dirty = False

    # Bind textures
    main_texture.use(location=0)
    float_texture.use(location=1)

    # Draw single full-screen quad
    vao.render(mode=moderngl.TRIANGLE_FAN)


def composite_cpu(main_img, float_img):
    """
    CPU compositing that respects BOTH main and float alpha channels,
    compositing them over BACKGROUND_COLOR.
    Moved from image_display.py to centralize rendering logic.
    """
    # Nothing at all?
    if main_img is None and float_img is None:
        return None

    # --- Helper: normalize to (RGB uint16, alpha uint16 or None) ---
    def to_rgb_alpha(img):
        if img is None:
            return None, None

        if img.ndim == 2:
            # grayscale -> RGB
            rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB).astype(np.uint16)
            alpha = None
        elif img.ndim == 3:
            if img.shape[2] == 4:
                # RGBA: keep RGB and per-pixel alpha
                rgb = img[..., :3].astype(np.uint16)
                alpha = img[..., 3:4].astype(np.uint16)
            elif img.shape[2] == 3:
                # RGB, treat as fully opaque
                rgb = img.astype(np.uint16)
                alpha = None
            else:
                return None, None  # Invalid shape
        else:
            return None, None

        return rgb, alpha

    # Decide the reference size
    ref = main_img if main_img is not None else float_img
    h, w = ref.shape[:2]

    # Background color in RGB space
    bg_r, bg_g, bg_b = BACKGROUND_COLOR

    # Start with background as uint16 RGB
    base = np.empty((h, w, 3), dtype=np.uint16)
    base[..., 0] = bg_r
    base[..., 1] = bg_g
    base[..., 2] = bg_b

    # Normalize both layers
    main_rgb, main_a = to_rgb_alpha(main_img)
    float_rgb, float_a = to_rgb_alpha(float_img)

    # Size mismatch check
    if main_rgb is not None and float_rgb is not None:
        if main_rgb.shape[:2] != float_rgb.shape[:2]:
            float_rgb = None
            float_a = None

    # Helper: composite "src over dst"
    def over(dst_rgb16, src_rgb16, alpha):
        if src_rgb16 is None:
            return dst_rgb16

        if alpha is None:
            # Fully opaque source: overwrite
            # Optimization: No math needed, just copy
            return src_rgb16

        alpha_arr = alpha
        inv_alpha = 255 - alpha_arr

        # Math: (src * A + dst * (1-A)) / 255
        # We use uint16 to prevent overflow before division
        out = (src_rgb16 * alpha_arr + dst_rgb16 * inv_alpha + 127) // 255
        return out

    # 1) main over background
    if main_rgb is not None:
        base = over(base, main_rgb, main_a)

    # 2) float over that
    if float_rgb is not None:
        base = over(base, float_rgb, float_a)

    # Convert back to uint8
    out_rgb = np.clip(base, 0, 255).astype(np.uint8)

    # NOTE: If your image loader returns RGB, TurboJPEG wants RGB (pixel_format=0).
    # If your image loader returns BGR (OpenCV standard), TurboJPEG wants BGR (pixel_format=1).
    # This function returns the same color order as input.
    return out_rgb