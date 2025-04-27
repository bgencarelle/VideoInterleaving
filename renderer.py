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

# Transformation parameters (set via display_manager)
_fs_scale: float = 1.0
_fs_offset_x: float = 0.0
_fs_offset_y: float = 0.0
_image_size: tuple[int, int] = (0, 0)   # original image (width, height)
_rotation_angle: float = 0.0
_mirror_mode: int = 0

def initialize(gl_context: moderngl.Context) -> None:
    """Initialize ModernGL shaders, buffer, and blending state."""
    global ctx, prog, vao, vbo
    ctx = gl_context

    # Vertex and fragment shader sources (GLSL 3.3 core)
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
        # Fragment shader with gamma correction (linearize the texture color)
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
        # Fragment shader without gamma correction (use texture color as-is)
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

    # Create a dynamic vertex buffer (4 vertices × 4 floats = 16 floats total)
    vbo = ctx.buffer(reserve=64)  # 16 floats * 4 bytes = 64 bytes
    # Create a VAO mapping buffer content to shader inputs (vec2 position, vec2 texcoord)
    vao = ctx.vertex_array(prog, [(vbo, '2f 2f', 'position', 'texcoord')])
    # Enable blending for transparency and set blend function (standard alpha blending)
    ctx.enable(moderngl.BLEND)
    ctx.blend_func = (
        moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA,
        moderngl.ONE, moderngl.ONE_MINUS_SRC_ALPHA
    )

def set_transform_parameters(fs_scale: float, fs_offset_x: float, fs_offset_y: float,
                             image_size: tuple[int, int],
                             rotation_angle: float, mirror_mode: int) -> None:
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
    if prog:
        prog["u_MVP"].write(mvp_matrix.T.tobytes())  # transpose to column-major for OpenGL

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
    """Compute the positions and texture coordinates of the quad after scaling/rotation."""
    w, h = _image_size
    # Scale original dimensions
    w_scaled = w * _fs_scale
    h_scaled = h * _fs_scale
    # Swap width/height if rotated 90° or 270°
    if _rotation_angle % 180 == 90:
        effective_w = h_scaled
        effective_h = w_scaled
    else:
        effective_w = w_scaled
        effective_h = h_scaled
    # Compute centered quad coordinates
    cx = _fs_offset_x + effective_w / 2.0
    cy = _fs_offset_y + effective_h / 2.0
    corners = [(-w_scaled/2, -h_scaled/2),
               ( w_scaled/2, -h_scaled/2),
               ( w_scaled/2,  h_scaled/2),
               (-w_scaled/2,  h_scaled/2)]
    # Rotate each corner around (0,0) by _rotation_angle
    rad = math.radians(_rotation_angle)
    cosA, sinA = math.cos(rad), math.sin(rad)
    rotated = [(x*cosA - y*sinA, x*sinA + y*cosA) for (x, y) in corners]
    # Translate corners to the center (apply offsets)
    final_positions = [(cx + x, cy + y) for (x, y) in rotated]
    # Texture coordinates for each vertex (mirror horizontally if _mirror_mode=1)
    if _mirror_mode:
        tex_coords = [(1.0, 1.0), (0.0, 1.0), (0.0, 0.0), (1.0, 0.0)]
    else:
        tex_coords = [(0.0, 1.0), (1.0, 1.0), (1.0, 0.0), (0.0, 0.0)]
    # Interleave position and texcoord data
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
    # Clear screen to the background color
    ctx.clear(background_color[0] / 255.0, background_color[1] / 255.0, background_color[2] / 255.0)
    # Update quad vertex buffer with current geometry
    vbo.write(compute_transformed_quad().tobytes())
    # Draw main image quad
    main_texture.use(location=0)
    vao.render(mode=moderngl.TRIANGLE_FAN)
    # Draw float image quad (blended on top)
    float_texture.use(location=0)
    vao.render(mode=moderngl.TRIANGLE_FAN)
