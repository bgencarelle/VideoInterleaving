"""
renderer.py – ModernGL rendering for main and float images (textured quad)
"""
import moderngl
import numpy as np
import math
from settings import ENABLE_SRGB_FRAMEBUFFER, GAMMA_CORRECTION_ENABLED

# ModernGL context and objects
ctx: moderngl.Context | None = None
prog: moderngl.Program | None = None
vao: moderngl.VertexArray | None = None
vbo: moderngl.Buffer | None = None

# Transformation parameters (set via display_manager or elsewhere)
_fs_scale: float = 1.0
_fs_offset_x: float = 0.0
_fs_offset_y: float = 0.0
_image_size: tuple[int, int] = (0, 0)   # original image (width, height)
_rotation_angle: float = 0.0
_mirror_mode: int = 0
# Viewport / FBO size (for headless aspect-correct quad)
_viewport_size: tuple[int, int] = (0, 0)

def set_viewport_size(width: int, height: int) -> None:
    """Record the viewport (FBO or window) size and set GL viewport."""
    global _viewport_size
    _viewport_size = (width, height)
    if ctx is not None:
        ctx.viewport = (0, 0, width, height)


def initialize(gl_context: moderngl.Context) -> None:
    """Initialize ModernGL shaders, buffer, MVP, and blending state."""
    global ctx, prog, vao, vbo
    ctx = gl_context

    is_gles = ctx.version_code < 320  # GL ES typically reports < 320

    if is_gles:
        # GLSL ES 3.10 compatible shaders
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
                uniform sampler2D texture0;
                in vec2 v_texcoord;
                out vec4 fragColor;
                void main() {
                    vec4 texColor = texture(texture0, v_texcoord);
                    fragColor = vec4(pow(texColor.rgb, vec3(2.2)), texColor.a);
                }
            """
        else:
            fragment_src = """
                #version 310 es
                precision mediump float;
                uniform sampler2D texture0;
                in vec2 v_texcoord;
                out vec4 fragColor;
                void main() {
                    fragColor = texture(texture0, v_texcoord);
                }
            """
    else:
        # Desktop GLSL 3.30 shaders
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
                uniform sampler2D texture0;
                in vec2 v_texcoord;
                out vec4 fragColor;
                void main() {
                    vec4 texColor = texture(texture0, v_texcoord);
                    fragColor = vec4(pow(texColor.rgb, vec3(2.2)), texColor.a);
                }
            """
        else:
            fragment_src = """
                #version 330 core
                uniform sampler2D texture0;
                in vec2 v_texcoord;
                out vec4 fragColor;
                void main() {
                    fragColor = texture(texture0, v_texcoord);
                }
            """

    prog = ctx.program(vertex_shader=vertex_src, fragment_shader=fragment_src)

    vbo = ctx.buffer(reserve=64)
    vao = ctx.vertex_array(prog, [(vbo, '2f 2f', 'position', 'texcoord')])

    # Default MVP = identity so clip-space positions “just work”
    mvp = np.eye(4, dtype="f4")
    prog["u_MVP"].write(mvp.T.tobytes())

    ctx.enable(moderngl.BLEND)
    ctx.blend = (
        moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA,
        moderngl.ONE, moderngl.ONE_MINUS_SRC_ALPHA
    )


def set_transform_parameters(fs_scale: float,
                             fs_offset_x: float,
                             fs_offset_y: float,
                             image_size: tuple[int, int],
                             rotation_angle: float,
                             mirror_mode: int) -> None:
    """Update transformation parameters used for rendering (scale, offsets, etc.)."""
    global _fs_scale, _fs_offset_x, _fs_offset_y, _image_size, _rotation_angle, _mirror_mode
    _fs_scale = fs_scale
    _fs_offset_x = fs_offset_x
    _fs_offset_y = fs_offset_y
    _image_size = image_size
    _rotation_angle = rotation_angle
    _mirror_mode = mirror_mode


def update_mvp(mvp_matrix: np.ndarray) -> None:
    """Update the MVP matrix uniform for the shader (called when viewport changes)."""
    if prog is not None:
        prog["u_MVP"].write(mvp_matrix.T.tobytes())  # column-major for OpenGL


def create_texture(image: np.ndarray) -> moderngl.Texture:
    """Create a new GPU texture from a NumPy RGBA image array."""
    h, w = image.shape[0], image.shape[1]
    texture = ctx.texture((w, h), 4, data=image.tobytes())
    texture.filter = (moderngl.LINEAR, moderngl.LINEAR)
    texture.repeat_x = False
    texture.repeat_y = False
    return texture


def update_texture(texture: moderngl.Texture, new_image: np.ndarray) -> moderngl.Texture:
    """Update an existing texture with a new image (recreate if dimensions differ)."""
    new_h, new_w = new_image.shape[0], new_image.shape[1]
    if (new_w, new_h) != texture.size:
        # Size changed: release old texture and create a new one
        texture.release()
        texture = ctx.texture((new_w, new_h), 4, data=new_image.tobytes())
        texture.filter = (moderngl.LINEAR, moderngl.LINEAR)
        texture.repeat_x = False
        texture.repeat_y = False
    else:
        # Same size: just update the pixel data
        texture.write(new_image.tobytes())
    return texture


def compute_transformed_quad() -> np.ndarray:
    """
    Compute the positions and texture coordinates of the quad.

    Two modes:

    - Headless / FBO mode (when _viewport_size != (0,0)):
        Build a clip-space quad in [-1,1]x[-1,1], scaled to preserve
        the image aspect ratio inside the viewport. Background color
        fills the letterbox/pillarbox region.

    - Local / legacy mode (when _viewport_size == (0,0)):
        Use the original pixel-space transform pipeline driven by
        _fs_scale / offsets / rotation and expect an MVP matrix to be
        provided via update_mvp().
    """

    view_w, view_h = _viewport_size

    # --- Headless aspect-correct path (viewport known) ----------------------
    if view_w > 0 and view_h > 0:
        img_w, img_h = _image_size

        # If we somehow don't know the image size, just assume square.
        if img_w <= 0 or img_h <= 0:
            r_tex = 1.0
        else:
            r_tex = img_w / img_h

        r_view = view_w / view_h

        # Fit image inside viewport, preserving aspect:
        # - If texture is wider: full width, letterbox top/bottom.
        # - If texture is taller: full height, pillarbox left/right.
        if r_tex > 0 and r_view > 0:
            if r_tex > r_view:
                # Wider than viewport
                x_scale = 1.0
                y_scale = r_view / r_tex
            else:
                # Taller than viewport
                y_scale = 1.0
                x_scale = r_tex / r_view
        else:
            # Degenerate: just show full screen
            x_scale = 1.0
            y_scale = 1.0

        positions = [
            (-x_scale, -y_scale),
            ( x_scale, -y_scale),
            ( x_scale,  y_scale),
            (-x_scale,  y_scale),
        ]

        if _mirror_mode:
            tex_coords = [(1.0, 1.0), (0.0, 1.0), (0.0, 0.0), (1.0, 0.0)]
        else:
            tex_coords = [(0.0, 1.0), (1.0, 1.0), (1.0, 0.0), (0.0, 0.0)]

        data = []
        for (px, py), (tu, tv) in zip(positions, tex_coords):
            data += [px, py, tu, tv]
        return np.array(data, dtype=np.float32)

    # --- Legacy pixel-space path (local window, using MVP) ------------------
    w, h = _image_size
    # If we still don't know the image size here, just use full-screen quad.
    if w == 0 or h == 0:
        if _mirror_mode:
            tex_coords = [(1.0, 1.0), (0.0, 1.0), (0.0, 0.0), (1.0, 0.0)]
        else:
            tex_coords = [(0.0, 1.0), (1.0, 1.0), (1.0, 0.0), (0.0, 0.0)]
        positions = [
            (-1.0, -1.0),
            ( 1.0, -1.0),
            ( 1.0,  1.0),
            (-1.0,  1.0),
        ]
        data = []
        for (px, py), (tu, tv) in zip(positions, tex_coords):
            data += [px, py, tu, tv]
        return np.array(data, dtype=np.float32)

    # Original pixel-space logic
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
    return np.array(data, dtype=np.float32)


def overlay_images_two_pass_like_old(main_texture: moderngl.Texture,
                                     float_texture: moderngl.Texture,
                                     background_color: tuple[float, float, float] = (0, 0, 0)) -> None:
    """
    Clear the screen with background_color, then draw the main texture and float texture.
    Uses two-pass rendering (draw main, then overlay float) with blending.
    """

    def srgb_to_linear_component(c):
        return pow(c / 255.0, 2.2)

    if GAMMA_CORRECTION_ENABLED or ENABLE_SRGB_FRAMEBUFFER:
        # Convert background_color from sRGB to linear before clearing
        bg_linear = tuple(srgb_to_linear_component(c) for c in background_color)
        ctx.clear(*bg_linear)
    else:
        # No correction needed, treat values as sRGB directly
        ctx.clear(background_color[0] / 255.0,
                  background_color[1] / 255.0,
                  background_color[2] / 255.0)

    # Update quad vertex buffer with current geometry
    quad_data = compute_transformed_quad()
    vbo.write(quad_data.tobytes())

    # Draw main image quad
    main_texture.use(location=0)
    vao.render(mode=moderngl.TRIANGLE_FAN)

    # Draw float image quad (blended on top)
    float_texture.use(location=0)
    vao.render(mode=moderngl.TRIANGLE_FAN)
