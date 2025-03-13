import os

os.environ['PYOPENGL_ERROR_CHECKING'] = '0'
import time
from concurrent.futures import ThreadPoolExecutor
from collections import deque
import threading
import pygame.time
from OpenGL.GL import *
from OpenGL.GLU import *
import OpenGL.GL as GL
GL.glGetError = lambda: 0


import calculators
import midi_control
import platform
import pygame

import cv2
import random
import index_client

# Import functions from the new modules
from settings import (FULLSCREEN_MODE, MIDI_MODE, MTC_CLOCK, MIDI_CLOCK, MIXED_CLOCK, PINGPONG, BUFFER_SIZE,
                      CLIENT_MODE, FREE_CLOCK, IPS, FPS, CLOCK_MODE, TEST_MODE, VALID_MODES)
from index_calculator import set_launch_time, update_index
from folder_selector import update_folder_selection
# Import shared globals
from globals import control_data_dictionary, folder_dictionary

# New imports for VBO rendering.
import numpy as np
from OpenGL.arrays import vbo

# Global variables and preallocated vertex array.
run_mode = True
pause_mode = False
png_paths_len = 0
main_folder_path = 0
float_folder_path = 0
float_folder_count = 0
main_folder_count = 0
image_size = (800, 600)
aspect_ratio = 1.333
text_display = False

texture_dimensions = {}

# Global shader program variables and cached uniform/attribute locations.
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

# Preallocated vertex array (quad with 4 vertices, each having (x, y, u, v)).
quad_vertices = None
# Global VBO for the quad; reusing this reduces per-frame allocations.
quad_vbo = None

def toggle_fullscreen(current_fullscreen_status):
    new_fullscreen = not current_fullscreen_status
    display_init(new_fullscreen)
    return new_fullscreen

def event_check(fullscreen):
    global image_size, run_mode
    width, height = image_size
    aspect_ratio_local = width / height
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            run_mode = False
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_q:
                run_mode = False
                pygame.quit()
            if event.key == pygame.K_f:
                fullscreen = toggle_fullscreen(fullscreen)
        elif event.type == pygame.VIDEORESIZE:
            new_width, new_height = event.size
            if new_width / new_height > aspect_ratio_local:
                new_width = int(new_height * aspect_ratio_local)
                image_size = (new_width, new_height)
            else:
                new_height = int(new_width / aspect_ratio_local)
                image_size = (new_width, new_height)
            display_init(fullscreen)
    return fullscreen

def get_aspect_ratio(image_path):
    image = pygame.image.load(image_path)
    w, h = image.get_size()
    a_ratio = h / w
    print(f'This is {w} wide and {h} tall, with an aspect ratio of {a_ratio}')
    return a_ratio, w, h

def read_image(image_path):
    if image_path.lower().endswith(('.webp', '.png')):
        image_np = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
        if image_np is None:
            raise ValueError(f"Failed to load image: {image_path}")
        image_np = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGBA)
        return image_np
    else:
        raise ValueError("Unsupported image format.")

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
    # Cache uniform and attribute locations
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
    # Cache uniform and attribute locations
    simple_shader_mvp_loc = glGetUniformLocation(simple_shader_program, "u_MVP")
    simple_shader_texture0_loc = glGetUniformLocation(simple_shader_program, "texture0")
    simple_shader_position_loc = glGetAttribLocation(simple_shader_program, "position")
    simple_shader_texcoord_loc = glGetAttribLocation(simple_shader_program, "texcoord")

def display_image(texture_id, width, height):
    """
    Draws a textured quad using the simple shader program.
    """
    scaled_width = width * fs_scale
    scaled_height = height * fs_scale
    # Update preallocated quad vertices (each vertex: x, y, u, v)
    quad_vertices[0] = fs_offset_x
    quad_vertices[1] = fs_offset_y
    quad_vertices[2] = 0.0
    quad_vertices[3] = 1.0
    quad_vertices[4] = fs_offset_x + scaled_width
    quad_vertices[5] = fs_offset_y
    quad_vertices[6] = 1.0
    quad_vertices[7] = 1.0
    quad_vertices[8] = fs_offset_x + scaled_width
    quad_vertices[9] = fs_offset_y + scaled_height
    quad_vertices[10] = 1.0
    quad_vertices[11] = 0.0
    quad_vertices[12] = fs_offset_x
    quad_vertices[13] = fs_offset_y + scaled_height
    quad_vertices[14] = 0.0
    quad_vertices[15] = 0.0
    # Update VBO with new vertex data
    quad_vbo.set_array(quad_vertices)
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
    global image_size
    width, height = image_size
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
    scaled_width = width * fs_scale
    scaled_height = height * fs_scale
    # Update the preallocated vertex array in place
    quad_vertices[0] = fs_offset_x
    quad_vertices[1] = fs_offset_y
    quad_vertices[2] = 0.0
    quad_vertices[3] = 1.0
    quad_vertices[4] = fs_offset_x + scaled_width
    quad_vertices[5] = fs_offset_y
    quad_vertices[6] = 1.0
    quad_vertices[7] = 1.0
    quad_vertices[8] = fs_offset_x + scaled_width
    quad_vertices[9] = fs_offset_y + scaled_height
    quad_vertices[10] = 1.0
    quad_vertices[11] = 0.0
    quad_vertices[12] = fs_offset_x
    quad_vertices[13] = fs_offset_y + scaled_height
    quad_vertices[14] = 0.0
    quad_vertices[15] = 0.0
    quad_vbo.set_array(quad_vertices)
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

def load_images(index, main_folder, float_folder):
    main_image = read_image(main_folder_path[index][main_folder])
    float_image = read_image(float_folder_path[index][float_folder])
    return main_image, float_image

class ImageLoaderBuffer:
    def __init__(self, buffer_size):
        self.buffer_size = buffer_size
        self.buffer = deque(maxlen=buffer_size)
    def add_image_future(self, index, future):
        clamped_index = max(0, min(index, png_paths_len - 1))
        self.buffer.append((clamped_index, future))
    def get_future_for_index(self, index):
        for item in list(self.buffer):
            buf_index, future = item
            if buf_index == index:
                self.buffer.remove(item)
                return future
        return None

def run_display_setup():
    if MIDI_MODE:
        midi_control.midi_control_stuff_main()
    elif CLOCK_MODE == CLIENT_MODE:
        threading.Thread(target=index_client.start_client, daemon=True).start()
    pygame.init()
    pygame.mouse.set_visible(False)
    display_init(FULLSCREEN_MODE)
    run_display()
    return

def run_display():
    global run_mode
    vid_clock = pygame.time.Clock()
    index, direction = update_index(png_paths_len, PINGPONG)
    last_index = index
    control_data_dictionary['Index_and_Direction'] = (index, direction)
    update_folder_selection(index, direction, float_folder_count, main_folder_count)
    fullscreen = FULLSCREEN_MODE
    image_buffer = ImageLoaderBuffer(BUFFER_SIZE)
    with ThreadPoolExecutor(max_workers=4) as executor:
        main_folder, float_folder = folder_dictionary['Main_and_Float_Folders']
        for i in range(BUFFER_SIZE):
            buf_idx = (index + i) % png_paths_len
            future = executor.submit(load_images, buf_idx, main_folder, float_folder)
            image_buffer.add_image_future(buf_idx, future)
        main_image, float_image = load_images(index, main_folder, float_folder)
        texture_id1 = create_texture(main_image)
        texture_id2 = create_texture(float_image)
        while run_mode:
            try:
                fullscreen = event_check(fullscreen)
                new_index, new_direction = update_index(png_paths_len, PINGPONG)
                if new_index != last_index:
                    index, direction = new_index, new_direction
                    last_index = index
                    control_data_dictionary['Index_and_Direction'] = (index, direction)
                    update_folder_selection(index, direction, float_folder_count, main_folder_count)
                    main_folder, float_folder = folder_dictionary['Main_and_Float_Folders']
                    future = image_buffer.get_future_for_index(index)
                    if future is not None:
                        main_image, float_image = future.result()
                    else:
                        main_image, float_image = load_images(index, main_folder, float_folder)
                    update_texture(texture_id1, main_image)
                    update_texture(texture_id2, float_image)
                    next_index = (index + direction) % png_paths_len
                    if image_buffer.get_future_for_index(next_index) is None:
                        new_future = executor.submit(load_images, next_index, main_folder, float_folder)
                        image_buffer.add_image_future(next_index, new_future)
                overlay_images_fast(texture_id1, texture_id2)
                pygame.display.flip()
                vid_clock.tick(FPS)
            except Exception as e:
                print(f"An error occurred: {e}")

def display_init(fullscreen=True):
    global fs_scale, fs_offset_x, fs_offset_y, fs_fullscreen_width, fs_fullscreen_height, image_size, quad_vbo, quad_vertices
    w, h = image_size
    if fullscreen:
        modes = pygame.display.list_modes()
        if not modes:
            raise RuntimeError("No display modes available!")
        fs_fullscreen_width, fs_fullscreen_height = modes[0]
        scale_x = fs_fullscreen_width / w
        scale_y = fs_fullscreen_height / h
        fs_scale = min(scale_x, scale_y)
        scaled_width = w * fs_scale
        scaled_height = h * fs_scale
        fs_offset_x = int((fs_fullscreen_width - scaled_width) / 2)
        fs_offset_y = int((fs_fullscreen_height - scaled_height) / 2)
        flags = pygame.OPENGL | pygame.DOUBLEBUF | pygame.FULLSCREEN
        pygame.display.set_caption('Fullscreen Mode')
        pygame.display.set_mode((fs_fullscreen_width, fs_fullscreen_height), flags, vsync=1)
        glViewport(0, 0, fs_fullscreen_width, fs_fullscreen_height)
        viewport_width = fs_fullscreen_width
        viewport_height = fs_fullscreen_height
    else:
        win_width = 400
        win_height = int(400 * h / w)
        scale_x = win_width / w
        scale_y = win_height / h
        win_scale = min(scale_x, scale_y)
        win_offset_x = int((win_width - (w * win_scale)) / 2)
        win_offset_y = int((win_height - (h * win_scale)) / 2)
        flags = pygame.OPENGL | pygame.DOUBLEBUF | pygame.RESIZABLE
        pygame.display.set_caption('Windowed Mode')
        pygame.display.set_mode((win_width, win_height), flags, vsync=1)
        glViewport(0, 0, win_width, win_height)
        viewport_width = win_width
        viewport_height = win_height
        fs_scale = win_scale
        fs_offset_x = win_offset_x
        fs_offset_y = win_offset_y
    # Compute an orthographic MVP matrix mapping screen coordinates to normalized device coordinates.
    mvp = np.array([
        [2.0/viewport_width, 0, 0, -1],
        [0, 2.0/viewport_height, 0, -1],
        [0, 0, -1, 0],
        [0, 0, 0, 1]
    ], dtype=np.float32)
    # Initialize shaders if not already done.
    if blend_shader_program is None:
        init_blend_shader()
    if simple_shader_program is None:
        init_simple_shader()
    # Set the MVP matrix for both shader programs.
    glUseProgram(blend_shader_program)
    glUniformMatrix4fv(blend_shader_mvp_loc, 1, GL_TRUE, mvp)
    glUseProgram(simple_shader_program)
    glUniformMatrix4fv(simple_shader_mvp_loc, 1, GL_TRUE, mvp)
    glUseProgram(0)
    glEnable(GL_TEXTURE_2D)
    glEnable(GL_BLEND)
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
    # Create the global quad VBO and preallocate the vertex array if not already created.
    global quad_vbo, quad_vertices
    if quad_vbo is None:
        quad_vertices = np.zeros(16, dtype=np.float32)
        quad_vbo = vbo.VBO(quad_vertices, usage=GL_DYNAMIC_DRAW)

def display_and_run(clock_source=FREE_CLOCK):
    global png_paths_len, main_folder_path, main_folder_count, \
           float_folder_path, float_folder_count, image_size, aspect_ratio
    random.seed()
    csv_source, main_folder_path, float_folder_path = calculators.init_all(clock_source)
    print(platform.system(), "midi clock mode is:", CLOCK_MODE)
    main_folder_count = len(main_folder_path[0])
    float_folder_count = len(float_folder_path[0])
    png_paths_len = len(main_folder_path) - 1
    aspect_ratio, width, height = get_aspect_ratio(main_folder_path[0][0])
    image_size = (width, height)
    print("Image size:", image_size)
    run_display_setup()

if __name__ == "__main__":
    display_and_run()
