import moderngl
import numpy as np
import math
from settings import ENABLE_SRGB_FRAMEBUFFER  # [Added for sRGB flag access]

# ModernGL objects and shader program
ctx = None
prog = None
vao = None
vbo = None

# Transformation parameters (set via display_manager)
_fs_scale = 1.0
_fs_offset_x = 0.0
_fs_offset_y = 0.0
_image_size = (0, 0)    # (width, height) of original image
_rotation_angle = 0
_mirror_mode = 0


def initialize(gl_context):
    global ctx, prog, vao, vbo
    ctx = gl_context
    # Vertex and fragment shader sources
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
    if ENABLE_SRGB_FRAMEBUFFER:
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

    # Create a dynamic vertex buffer for 4 vertices (x, y, u, v per vertex)
    vbo = ctx.buffer(reserve=64)  # 4 vertices * 4 floats * 4 bytes = 64 bytes
    # Create a VAO with the buffer, describing 2f position + 2f texcoord layout
    vao = ctx.vertex_array(prog, [(vbo, '2f 2f', 'position', 'texcoord')])
    # Enable blending for transparency and set blend function (pre-multiplied alpha for alpha channel)
    ctx.enable(moderngl.BLEND)
    ctx.blend_func = (
        moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA,
        moderngl.ONE, moderngl.ONE_MINUS_SRC_ALPHA
    )

def set_transform_parameters(fs_scale, fs_offset_x, fs_offset_y, image_size, rotation_angle, mirror_mode):
    """Update transformation parameters for computing the quad geometry."""
    global _fs_scale, _fs_offset_x, _fs_offset_y, _image_size, _rotation_angle, _mirror_mode
    _fs_scale = fs_scale
    _fs_offset_x = fs_offset_x
    _fs_offset_y = fs_offset_y
    _image_size = image_size
    _rotation_angle = rotation_angle
    _mirror_mode = mirror_mode

def update_mvp(mvp_matrix):
    """Update the MVP matrix uniform in the shader."""
    if prog:
        # Write matrix to uniform (transpose to column-major for OpenGL)
        prog['u_MVP'].write(mvp_matrix.T.tobytes())

def create_texture(image):
    """Create a new GPU texture from a NumPy RGBA image array."""
    h, w = image.shape[0], image.shape[1]
    texture = ctx.texture((w, h), 4, data=image.tobytes())
    texture.filter = (moderngl.LINEAR, moderngl.LINEAR)  # linear filtering
    texture.repeat_x = False
    texture.repeat_y = False
    return texture

def update_texture(texture, new_image):
    """Update an existing texture with a new image (resize if dimensions differ)."""
    new_h, new_w = new_image.shape[0], new_image.shape[1]
    if (new_w, new_h) != texture.size:
        # If size changed, create a new texture
        texture.release()
        texture = ctx.texture((new_w, new_h), 4, data=new_image.tobytes())
        texture.filter = (moderngl.LINEAR, moderngl.LINEAR)
        texture.repeat_x = False
        texture.repeat_y = False
    else:
        # Update existing texture content
        texture.write(new_image.tobytes())
    return texture

def compute_transformed_quad():
    """Compute the positions and texture coords of the quad vertices after scaling/rotation."""
    w, h = _image_size
    # Apply scaling to original image dimensions
    w_scaled = w * _fs_scale
    h_scaled = h * _fs_scale
    # Determine effective drawn width/height after rotation
    if _rotation_angle % 180 == 90:
        effective_w = h_scaled
        effective_h = w_scaled
    else:
        effective_w = w_scaled
        effective_h = h_scaled
    # Center of the image in screen space
    cx = _fs_offset_x + effective_w / 2.0
    cy = _fs_offset_y + effective_h / 2.0
    # Quad corners centered at (0,0) before rotation
    corners = [(-w_scaled/2, -h_scaled/2),
               ( w_scaled/2, -h_scaled/2),
               ( w_scaled/2,  h_scaled/2),
               (-w_scaled/2,  h_scaled/2)]
    # Rotate corners around origin by _rotation_angle (degrees)
    rad = math.radians(_rotation_angle)
    cosA, sinA = math.cos(rad), math.sin(rad)
    rotated = [(x*cosA - y*sinA, x*sinA + y*cosA) for (x, y) in corners]
    # Translate rotated corners to screen center (offset)
    final_positions = [(cx + x, cy + y) for (x, y) in rotated]
    # Define texture coordinates for each corner (flip horizontally if mirror_mode)
    if _mirror_mode:
        tex_coords = [(1.0, 1.0), (0.0, 1.0), (0.0, 0.0), (1.0, 0.0)]
    else:
        tex_coords = [(0.0, 1.0), (1.0, 1.0), (1.0, 0.0), (0.0, 0.0)]
    # Interleave position and texcoord data for 4 vertices
    data = []
    for (px, py), (tu, tv) in zip(final_positions, tex_coords):
        data += [px, py, tu, tv]
    return np.array(data, dtype=np.float32)

def overlay_images_two_pass_like_old(main_texture, float_texture, background_color=(0, 0, 0)):
    """
    Render the main and float textures with blending (two-pass draw).
    Clears the screen with the given background color, then draws the main image and overlays the float image.
    """
    ctx.clear(background_color[0] / 255.0, background_color[1] / 255.0, background_color[2] / 255.0)
    # Update VBO with current quad vertex positions and texcoords
    vbo.write(compute_transformed_quad().tobytes())
    # Draw main image quad
    main_texture.use(location=0)
    vao.render(mode=moderngl.TRIANGLE_FAN)
    # Draw float image quad (blended on top)
    float_texture.use(location=0)
    vao.render(mode=moderngl.TRIANGLE_FAN)
