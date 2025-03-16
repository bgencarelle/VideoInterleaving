# renderer.py

from OpenGL.GL import *
import numpy as np
from OpenGL.arrays import vbo
import math

# Global variables for shader programs and their cached locations.
blend_shader_program = None
blend_shader_position_loc = None
blend_shader_texcoord_loc = None
blend_shader_tex0_loc = None
blend_shader_tex1_loc = None
blend_shader_mvp_loc = None

simple_shader_program = None
simple_shader_position_loc = None
simple_shader_texcoord_loc = None
simple_shader_texture0_loc = None
simple_shader_mvp_loc = None

# Global texture dimensions dictionary.
texture_dimensions = {}

# Global quad VBO and preallocated vertex array.
quad_vbo = None
quad_vertices = None

# Internal globals for transformation parameters.
_fs_scale = None
_fs_offset_x = None
_fs_offset_y = None
_image_size = None
_rotation_angle = 0
_mirror_mode = None

def set_transform_parameters(fs_scale, fs_offset_x, fs_offset_y, image_size, rotation_angle, mirror_mode):
    """
    Sets the transformation parameters used for computing the quad vertices.
    This function should be called by the main display initialization (in image_display.py)
    after computing these values.
    """
    global _fs_scale, _fs_offset_x, _fs_offset_y, _image_size, _rotation_angle, _mirror_mode
    _fs_scale = fs_scale
    _fs_offset_x = fs_offset_x
    _fs_offset_y = fs_offset_y
    _image_size = image_size
    _rotation_angle = rotation_angle
    _mirror_mode = mirror_mode

def compile_shader(source, shader_type):
    shader = glCreateShader(shader_type)
    glShaderSource(shader, source)
    glCompileShader(shader)
    result = glGetShaderiv(shader, GL_COMPILE_STATUS)
    if not result:
        error = glGetShaderInfoLog(shader)
        raise RuntimeError(f"Shader compilation failed: {error}")
    return shader

def init_blend_shader():
    global blend_shader_program, blend_shader_position_loc, blend_shader_texcoord_loc
    global blend_shader_tex0_loc, blend_shader_tex1_loc, blend_shader_mvp_loc
    vertex_src = """
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
    fragment_src = """
    #version 120
    uniform sampler2D texture0;
    uniform sampler2D texture1;
    varying vec2 v_texcoord;
    void main() {
        vec4 color0 = texture2D(texture0, v_texcoord);
        vec4 color1 = texture2D(texture1, v_texcoord);
        gl_FragColor = mix(color0, color1, color1.a);
    }
    """
    vertex_shader = compile_shader(vertex_src, GL_VERTEX_SHADER)
    fragment_shader = compile_shader(fragment_src, GL_FRAGMENT_SHADER)
    program = glCreateProgram()
    glAttachShader(program, vertex_shader)
    glAttachShader(program, fragment_shader)
    glLinkProgram(program)
    result = glGetProgramiv(program, GL_LINK_STATUS)
    if not result:
        error = glGetProgramInfoLog(program)
        raise RuntimeError(f"Program link failed: {error}")
    blend_shader_program = program
    # Cache uniform and attribute locations.
    blend_shader_mvp_loc = glGetUniformLocation(blend_shader_program, "u_MVP")
    blend_shader_tex0_loc = glGetUniformLocation(blend_shader_program, "texture0")
    blend_shader_tex1_loc = glGetUniformLocation(blend_shader_program, "texture1")
    blend_shader_position_loc = glGetAttribLocation(blend_shader_program, "position")
    blend_shader_texcoord_loc = glGetAttribLocation(blend_shader_program, "texcoord")

def init_simple_shader():
    global simple_shader_program, simple_shader_position_loc, simple_shader_texcoord_loc
    global simple_shader_texture0_loc, simple_shader_mvp_loc
    vertex_src = """
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
    fragment_src = """
    #version 120
    uniform sampler2D texture0;
    varying vec2 v_texcoord;
    void main() {
        gl_FragColor = texture2D(texture0, v_texcoord);
    }
    """
    vertex_shader = compile_shader(vertex_src, GL_VERTEX_SHADER)
    fragment_shader = compile_shader(fragment_src, GL_FRAGMENT_SHADER)
    program = glCreateProgram()
    glAttachShader(program, vertex_shader)
    glAttachShader(program, fragment_shader)
    glLinkProgram(program)
    result = glGetProgramiv(program, GL_LINK_STATUS)
    if not result:
        error = glGetProgramInfoLog(program)
        raise RuntimeError(f"Simple shader program link failed: {error}")
    simple_shader_program = program
    # Cache uniform and attribute locations.
    simple_shader_mvp_loc = glGetUniformLocation(simple_shader_program, "u_MVP")
    simple_shader_texture0_loc = glGetUniformLocation(simple_shader_program, "texture0")
    simple_shader_position_loc = glGetAttribLocation(simple_shader_program, "position")
    simple_shader_texcoord_loc = glGetAttribLocation(simple_shader_program, "texcoord")

def create_texture(image):
    """
    Creates and initializes an OpenGL texture from the given image.
    Uses GL_LINEAR filtering for smoother scaling and sets wrap modes.
    """
    texture_id = glGenTextures(1)
    glBindTexture(GL_TEXTURE_2D, texture_id)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)
    w, h = image.shape[1], image.shape[0]
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, w, h, 0, GL_RGBA, GL_UNSIGNED_BYTE, image)
    texture_dimensions[texture_id] = (w, h)
    return texture_id

def update_texture(texture_id, new_image):
    """
    Updates an existing texture with new image data using a Pixel Buffer Object (PBO)
    to offload data transfer asynchronously. Reallocates the texture if the dimensions have changed.
    """
    glBindTexture(GL_TEXTURE_2D, texture_id)
    w, h = new_image.shape[1], new_image.shape[0]
    expected = texture_dimensions.get(texture_id, (None, None))
    if expected != (w, h):
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, w, h, 0, GL_RGBA, GL_UNSIGNED_BYTE, None)
        texture_dimensions[texture_id] = (w, h)
    pbo = glGenBuffers(1)
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo)
    size = new_image.nbytes
    glBufferData(GL_PIXEL_UNPACK_BUFFER, size, new_image, GL_STREAM_DRAW)
    glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, w, h, GL_RGBA, GL_UNSIGNED_BYTE, None)
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0)
    glDeleteBuffers(1, [pbo])

def compute_transformed_quad():
    """
    Computes the quad vertices (with texture coordinates) after applying rotation and mirroring.
    Uses the transformation parameters set via set_transform_parameters().
    """
    global _fs_scale, _fs_offset_x, _fs_offset_y, _image_size, _rotation_angle, _mirror_mode
    w, h = _image_size
    w_scaled = w * _fs_scale
    h_scaled = h * _fs_scale
    if _rotation_angle % 180 == 90:
        effective_w = h * _fs_scale
        effective_h = w * _fs_scale
    else:
        effective_w = w * _fs_scale
        effective_h = h * _fs_scale
    target_cx = _fs_offset_x + effective_w / 2
    target_cy = _fs_offset_y + effective_h / 2
    pts = [(-w_scaled/2, -h_scaled/2),
           (w_scaled/2, -h_scaled/2),
           (w_scaled/2, h_scaled/2),
           (-w_scaled/2, h_scaled/2)]
    rad = math.radians(_rotation_angle)
    cosA = math.cos(rad)
    sinA = math.sin(rad)
    rotated_pts = []
    for (x, y) in pts:
        rx = x * cosA - y * sinA
        ry = x * sinA + y * cosA
        rotated_pts.append((rx, ry))
    final_pts = [(target_cx + x, target_cy + y) for (x, y) in rotated_pts]
    if _mirror_mode:
        tex_coords = [(1,1), (0,1), (0,0), (1,0)]
    else:
        tex_coords = [(0,1), (1,1), (1,0), (0,0)]
    vertices = np.zeros(16, dtype=np.float32)
    for i in range(4):
        vertices[i*4 + 0] = final_pts[i][0]
        vertices[i*4 + 1] = final_pts[i][1]
        vertices[i*4 + 2] = tex_coords[i][0]
        vertices[i*4 + 3] = tex_coords[i][1]
    return vertices

def display_image(texture_id):
    """
    Draws a textured quad using the simple shader program.
    """
    global quad_vbo
    vertices = compute_transformed_quad()
    quad_vbo.set_array(vertices)
    quad_vbo.bind()
    glUseProgram(simple_shader_program)
    glActiveTexture(GL_TEXTURE0)
    glBindTexture(GL_TEXTURE_2D, texture_id)
    glUniform1i(simple_shader_texture0_loc, 0)
    glEnableVertexAttribArray(simple_shader_position_loc)
    glVertexAttribPointer(simple_shader_position_loc, 2, GL_FLOAT, GL_FALSE, 16, quad_vbo)
    glEnableVertexAttribArray(simple_shader_texcoord_loc)
    glVertexAttribPointer(simple_shader_texcoord_loc, 2, GL_FLOAT, GL_FALSE, 16, quad_vbo + 8)
    glDrawArrays(GL_TRIANGLE_FAN, 0, 4)
    glDisableVertexAttribArray(simple_shader_position_loc)
    glDisableVertexAttribArray(simple_shader_texcoord_loc)
    glUseProgram(0)
    quad_vbo.unbind()

def overlay_images_fast(texture_id_main, texture_id_float, background_color=(0, 0, 0)):
    """
    Clears the screen with the background color and draws two textures using the blend shader.
    """
    global quad_vbo, _image_size
    width, height = _image_size
    glClearColor(background_color[0] / 255.0,
                 background_color[1] / 255.0,
                 background_color[2] / 255.0,
                 1.0)
    glClear(GL_COLOR_BUFFER_BIT)
    glUseProgram(blend_shader_program)
    glActiveTexture(GL_TEXTURE0)
    glBindTexture(GL_TEXTURE_2D, texture_id_main)
    glActiveTexture(GL_TEXTURE1)
    glBindTexture(GL_TEXTURE_2D, texture_id_float)
    glUniform1i(blend_shader_tex0_loc, 0)
    glUniform1i(blend_shader_tex1_loc, 1)
    vertices = compute_transformed_quad()
    quad_vbo.set_array(vertices)
    quad_vbo.bind()
    glEnableVertexAttribArray(blend_shader_position_loc)
    glVertexAttribPointer(blend_shader_position_loc, 2, GL_FLOAT, GL_FALSE, 16, quad_vbo)
    glEnableVertexAttribArray(blend_shader_texcoord_loc)
    glVertexAttribPointer(blend_shader_texcoord_loc, 2, GL_FLOAT, GL_FALSE, 16, quad_vbo + 8)
    glDrawArrays(GL_TRIANGLE_FAN, 0, 4)
    glDisableVertexAttribArray(blend_shader_position_loc)
    glDisableVertexAttribArray(blend_shader_texcoord_loc)
    quad_vbo.unbind()
    glUseProgram(0)

def setup_opengl(mvp):
    """
    Sets up the OpenGL state by initializing shaders (if not already done),
    setting the MVP matrix, enabling required features, and creating the quad VBO.
    This function should be called after the window/display has been set up.
    """
    global quad_vbo, quad_vertices
    if blend_shader_program is None:
        init_blend_shader()
    if simple_shader_program is None:
        init_simple_shader()
    glUseProgram(blend_shader_program)
    glUniformMatrix4fv(blend_shader_mvp_loc, 1, GL_TRUE, mvp)
    glUseProgram(simple_shader_program)
    glUniformMatrix4fv(simple_shader_mvp_loc, 1, GL_TRUE, mvp)
    glUseProgram(0)
    glEnable(GL_TEXTURE_2D)
    glEnable(GL_BLEND)
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
    if quad_vbo is None:
        quad_vertices = np.zeros(16, dtype=np.float32)
        quad_vbo = vbo.VBO(quad_vertices, usage=GL_DYNAMIC_DRAW)