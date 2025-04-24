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
    simple_shader_mvp_loc = glGetUniformLocation(simple_shader_program, "u_MVP")
    simple_shader_texture0_loc = glGetUniformLocation(simple_shader_program, "texture0")
    simple_shader_position_loc = glGetAttribLocation(simple_shader_program, "position")
    simple_shader_texcoord_loc = glGetAttribLocation(simple_shader_program, "texcoord")


def create_texture(image):
    texture_id = glGenTextures(1)
    glBindTexture(GL_TEXTURE_2D, texture_id)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)
    w, h = image.shape[1], image.shape[0]
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, w, h, 0, GL_RGBA, GL_UNSIGNED_BYTE, image)
    texture_dimensions[texture_id] = (w, h)
    return texture_id


def update_texture(texture_id, new_image):
    glBindTexture(GL_TEXTURE_2D, texture_id)
    w, h = new_image.shape[1], new_image.shape[0]
    expected = texture_dimensions.get(texture_id, (None, None))
    if expected != (w, h):
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, w, h, 0, GL_RGBA, GL_UNSIGNED_BYTE, None)
        texture_dimensions[texture_id] = (w, h)
    # Roll back from PBO update to direct upload for compatibility and reduced stalls
    glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, w, h, GL_RGBA, GL_UNSIGNED_BYTE, new_image)


def compute_transformed_quad():
    w, h = _image_size
    w_scaled = w * _fs_scale
    h_scaled = h * _fs_scale
    if _rotation_angle % 180 == 90:
        effective_w, effective_h = h_scaled, w_scaled
    else:
        effective_w, effective_h = w_scaled, h_scaled
    cx = _fs_offset_x + effective_w / 2
    cy = _fs_offset_y + effective_h / 2
    pts = [(-w_scaled/2, -h_scaled/2), (w_scaled/2, -h_scaled/2),
           (w_scaled/2, h_scaled/2), (-w_scaled/2, h_scaled/2)]
    rad = math.radians(_rotation_angle)
    cosA, sinA = math.cos(rad), math.sin(rad)
    rotated = [(x*cosA - y*sinA, x*sinA + y*cosA) for x,y in pts]
    final = [(cx + x, cy + y) for x,y in rotated]
    tex = [(1,1),(0,1),(0,0),(1,0)] if _mirror_mode else [(0,1),(1,1),(1,0),(0,0)]
    global quad_vertices
    for i,(vx,vy) in enumerate(final):
        quad_vertices[i*4 + 0] = vx
        quad_vertices[i*4 + 1] = vy
        quad_vertices[i*4 + 2] = tex[i][0]
        quad_vertices[i*4 + 3] = tex[i][1]
    return quad_vertices


def display_image(texture_id):
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
    glVertexAttribPointer(simple_shader_texcoord_loc, 2, GL_FLOAT, GL_FALSE, 16, quad_vbo+8)
    glDrawArrays(GL_TRIANGLE_FAN, 0, 4)
    glDisableVertexAttribArray(simple_shader_position_loc)
    glDisableVertexAttribArray(simple_shader_texcoord_loc)
    glUseProgram(0)
    quad_vbo.unbind()


def overlay_images_two_pass_like_old(texture_id_main, texture_id_float, background_color=(0,0,0)):
    glClearColor(background_color[0]/255, background_color[1]/255, background_color[2]/255, 1)
    glClear(GL_COLOR_BUFFER_BIT)
    # blend state set once at init
    display_image(texture_id_main)
    display_image(texture_id_float)


def overlay_images_blend(texture_id_main, texture_id_float, background_color=(0,0,0)):
    # 1) CLEAR
    glClearColor(
        background_color[0]/255.0,
        background_color[1]/255.0,
        background_color[2]/255.0,
        1.0
    )
    glClear(GL_COLOR_BUFFER_BIT)

    # 2) UPDATE QUAD VBO (was missing)
    verts = compute_transformed_quad()        # recalc position & texcoords
    quad_vbo.set_array(verts)                 # upload into VBO
    quad_vbo.bind()

    # 3) BIND BLEND SHADER & MVP (if you didn't already in setup_opengl)
    glUseProgram(blend_shader_program)
    # If you only uploaded MVP once in display_init, you can skip this; otherwise:
    # glUniformMatrix4fv(blend_shader_mvp_loc, 1, GL_TRUE, current_mvp)

    # 4) BIND BOTH TEXTURE UNITS
    glActiveTexture(GL_TEXTURE0)
    glBindTexture(GL_TEXTURE_2D, texture_id_main)
    glUniform1i(blend_shader_tex0_loc, 0)

    glActiveTexture(GL_TEXTURE1)
    glBindTexture(GL_TEXTURE_2D, texture_id_float)
    glUniform1i(blend_shader_tex1_loc, 1)

    # 5) SETUP ATTRIBS & DRAW
    glEnableVertexAttribArray(blend_shader_position_loc)
    glVertexAttribPointer(
        blend_shader_position_loc, 2, GL_FLOAT, GL_FALSE, 16, quad_vbo
    )
    glEnableVertexAttribArray(blend_shader_texcoord_loc)
    # note: here we offset by 8 bytes within the same VBO
    glVertexAttribPointer(
        blend_shader_texcoord_loc, 2, GL_FLOAT, GL_FALSE, 16, quad_vbo + 8
    )

    glDrawArrays(GL_TRIANGLE_FAN, 0, 4)

    # 6) CLEAN UP
    glDisableVertexAttribArray(blend_shader_position_loc)
    glDisableVertexAttribArray(blend_shader_texcoord_loc)
    glUseProgram(0)
    quad_vbo.unbind()


def setup_opengl(mvp):
    global quad_vbo, quad_vertices
    if blend_shader_program is None: init_blend_shader()
    if simple_shader_program is None: init_simple_shader()
    glUseProgram(blend_shader_program)
    glUniformMatrix4fv(blend_shader_mvp_loc,1,GL_TRUE,mvp)
    glUseProgram(simple_shader_program)
    glUniformMatrix4fv(simple_shader_mvp_loc,1,GL_TRUE,mvp)
    glUseProgram(0)
    glEnable(GL_TEXTURE_2D)
    glEnable(GL_BLEND)
    glBlendFuncSeparate(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA, GL_ONE, GL_ONE_MINUS_SRC_ALPHA)
    if quad_vbo is None:
        quad_vertices = np.zeros(16, dtype=np.float32)
        quad_vbo = vbo.VBO(quad_vertices, usage=GL_DYNAMIC_DRAW)
