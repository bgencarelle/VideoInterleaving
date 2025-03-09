import time
from concurrent.futures import ThreadPoolExecutor
from collections import deque
import threading
import pygame.time
from OpenGL.GL import *
from OpenGL.GLU import *

import calculators
import midi_control
import platform
import pygame

import cv2
import random
import index_client

# Import functions from the new modules
from settings import (FULLSCREEN_MODE, MIDI_MODE, MTC_CLOCK, MIDI_CLOCK, MIXED_CLOCK, PINGPONG, BUFFER_SIZE,
                      CLIENT_MODE, FREE_CLOCK, IPS, FPS, CLOCK_MODE, VALID_MODES)
from index_calculator import set_launch_time, update_index
from folder_selector import update_folder_selection
# Import shared globals
from globals import control_data_dictionary, folder_dictionary

# New imports for VBO rendering.
import numpy as np
from OpenGL.arrays import vbo

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

# Global shader program variable.
blend_shader_program = None

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
    Uses GL_NEAREST filtering, and now explicitly sets wrap modes.
    """
    texture_id = glGenTextures(1)
    glBindTexture(GL_TEXTURE_2D, texture_id)
    # Set filtering parameters.
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
    # Set wrap modes (optional, but recommended for predictable edge behavior).
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)
    w, h = image.shape[1], image.shape[0]
    # Create the texture; using GL_RGBA for both internal format and data format.
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, w, h, 0, GL_RGBA, GL_UNSIGNED_BYTE, image)
    texture_dimensions[texture_id] = (w, h)
    return texture_id

def update_texture(texture_id, new_image):
    """
    Updates an existing texture with new image data.
    Reallocates the texture if the dimensions have changed.
    """
    glBindTexture(GL_TEXTURE_2D, texture_id)
    w, h = new_image.shape[1], new_image.shape[0]
    expected = texture_dimensions.get(texture_id, (None, None))
    if expected != (w, h):
        # Reallocate texture storage.
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, w, h, 0, GL_RGBA, GL_UNSIGNED_BYTE, new_image)
        texture_dimensions[texture_id] = (w, h)
    else:
        # Update texture content.
        glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, w, h, GL_RGBA, GL_UNSIGNED_BYTE, new_image)

def display_image(texture_id, width, height, rgba=(1, 1, 1, 1)):
    """
    Draws a textured quad using a VBO to reduce CPU overhead.
    This function uses the global fs_offset_x, fs_offset_y, and fs_scale values
    set during display initialization to position and scale the quad.
    """
    glBindTexture(GL_TEXTURE_2D, texture_id)
    if rgba:
        glColor4f(*rgba)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
    # Calculate scaled dimensions.
    scaled_width = width * fs_scale
    scaled_height = height * fs_scale
    # Create an array of vertices (x, y, s, t) for the quad.
    vertices = np.array([
        fs_offset_x,                fs_offset_y,                 0.0, 1.0,  # Bottom-left.
        fs_offset_x + scaled_width, fs_offset_y,                 1.0, 1.0,  # Bottom-right.
        fs_offset_x + scaled_width, fs_offset_y + scaled_height,   1.0, 0.0,  # Top-right.
        fs_offset_x,                fs_offset_y + scaled_height,   0.0, 0.0   # Top-left.
    ], dtype=np.float32)
    # Create and bind a VBO with the vertex data.
    quad_vbo = vbo.VBO(vertices)
    quad_vbo.bind()
    # Enable vertex arrays and define pointers.
    glEnableClientState(GL_VERTEX_ARRAY)
    glVertexPointer(2, GL_FLOAT, 16, quad_vbo)  # 2 floats per vertex; stride = 4 * 4 bytes.
    glEnableClientState(GL_TEXTURE_COORD_ARRAY)
    glTexCoordPointer(2, GL_FLOAT, 16, quad_vbo + 8)  # Texture coords start after 2 floats (8 bytes).
    # Draw the quad.
    glDrawArrays(GL_QUADS, 0, 4)
    # Disable client states and unbind the VBO.
    glDisableClientState(GL_VERTEX_ARRAY)
    glDisableClientState(GL_TEXTURE_COORD_ARRAY)
    quad_vbo.unbind()
    if rgba:
        glColor4f(1, 1, 1, 1)
        glDisable(GL_BLEND)

# New helper functions for shader creation.
def compile_shader(source, shader_type):
    shader = glCreateShader(shader_type)
    glShaderSource(shader, source)
    glCompileShader(shader)
    # Check compile status.
    result = glGetShaderiv(shader, GL_COMPILE_STATUS)
    if not result:
        error = glGetShaderInfoLog(shader)
        raise RuntimeError(f"Shader compilation failed: {error}")
    return shader

def init_blend_shader():
    global blend_shader_program
    vertex_src = """
    #version 120
    attribute vec2 position;
    attribute vec2 texcoord;
    varying vec2 v_texcoord;
    void main() {
        gl_Position = gl_ModelViewProjectionMatrix * vec4(position, 0.0, 1.0);
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
    # Check link status.
    result = glGetProgramiv(program, GL_LINK_STATUS)
    if not result:
        error = glGetProgramInfoLog(program)
        raise RuntimeError(f"Program link failed: {error}")
    blend_shader_program = program

def overlay_images_fast(texture_id_main, texture_id_float, background_color=(0, 0, 0)):
    """
    Clears the screen with the given background color and draws two textures
    using a shader that blends them. The scaling and positioning remain unchanged.
    """
    global image_size, blend_shader_program
    width, height = image_size
    # Clear the background.
    glClearColor(background_color[0] / 255.0,
                 background_color[1] / 255.0,
                 background_color[2] / 255.0,
                 1.0)
    glClear(GL_COLOR_BUFFER_BIT)
    # Use the blend shader program.
    glUseProgram(blend_shader_program)
    # Bind the two textures to texture units 0 and 1.
    glActiveTexture(GL_TEXTURE0)
    glBindTexture(GL_TEXTURE_2D, texture_id_main)
    glActiveTexture(GL_TEXTURE1)
    glBindTexture(GL_TEXTURE_2D, texture_id_float)
    # Set shader sampler uniforms.
    loc0 = glGetUniformLocation(blend_shader_program, "texture0")
    loc1 = glGetUniformLocation(blend_shader_program, "texture1")
    glUniform1i(loc0, 0)
    glUniform1i(loc1, 1)
    # Calculate scaled dimensions.
    scaled_width = width * fs_scale
    scaled_height = height * fs_scale
    # Create an array of vertices (x, y, s, t) for the quad.
    vertices = np.array([
        fs_offset_x,                fs_offset_y,                 0.0, 1.0,  # Bottom-left.
        fs_offset_x + scaled_width, fs_offset_y,                 1.0, 1.0,  # Bottom-right.
        fs_offset_x + scaled_width, fs_offset_y + scaled_height,   1.0, 0.0,  # Top-right.
        fs_offset_x,                fs_offset_y + scaled_height,   0.0, 0.0   # Top-left.
    ], dtype=np.float32)
    quad_vbo = vbo.VBO(vertices)
    quad_vbo.bind()
    # Get attribute locations from the shader.
    pos_loc = glGetAttribLocation(blend_shader_program, "position")
    tex_loc = glGetAttribLocation(blend_shader_program, "texcoord")
    glEnableVertexAttribArray(pos_loc)
    glVertexAttribPointer(pos_loc, 2, GL_FLOAT, GL_FALSE, 16, quad_vbo)
    glEnableVertexAttribArray(tex_loc)
    glVertexAttribPointer(tex_loc, 2, GL_FLOAT, GL_FALSE, 16, quad_vbo + 8)
    # Draw the quad.
    glDrawArrays(GL_QUADS, 0, 4)
    glDisableVertexAttribArray(pos_loc)
    glDisableVertexAttribArray(tex_loc)
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
    # Create a clock for FPS regulation.
    vid_clock = pygame.time.Clock()

    # Initial index and folder update.
    index, direction = update_index(png_paths_len, PINGPONG)
    # Save the last displayed index.
    last_index = index
    control_data_dictionary['Index_and_Direction'] = (index, direction)
    update_folder_selection(index, direction, float_folder_count, main_folder_count)

    fullscreen = FULLSCREEN_MODE
    # Use our fixed-size buffer (based on IPS, not FPS)
    image_buffer = ImageLoaderBuffer(BUFFER_SIZE)

    # Preload BUFFER_SIZE images starting from the initial index.
    with ThreadPoolExecutor(max_workers=4) as executor:
        main_folder, float_folder = folder_dictionary['Main_and_Float_Folders']
        for i in range(BUFFER_SIZE):
            buf_idx = (index + i) % png_paths_len
            future = executor.submit(load_images, buf_idx, main_folder, float_folder)
            image_buffer.add_image_future(buf_idx, future)

        # Synchronously load the current image to create the initial textures.
        main_image, float_image = load_images(index, main_folder, float_folder)
        texture_id1 = create_texture(main_image)
        texture_id2 = create_texture(float_image)

        # Main display loop.
        while run_mode:
            try:
                # Process events (fullscreen toggling, resize, quit, etc.)
                fullscreen = event_check(fullscreen)

                # Calculate new index/direction.
                new_index, new_direction = update_index(png_paths_len, PINGPONG)

                # Only update textures if the index has changed.
                if new_index != last_index:
                    index, direction = new_index, new_direction
                    last_index = index
                    control_data_dictionary['Index_and_Direction'] = (index, direction)
                    update_folder_selection(index, direction, float_folder_count, main_folder_count)
                    main_folder, float_folder = folder_dictionary['Main_and_Float_Folders']

                    # Check if the image for the new index is already loaded.
                    future = image_buffer.get_future_for_index(index)
                    if future is not None:
                        main_image, float_image = future.result()
                    else:
                        # Fallback: load synchronously if missing.
                        main_image, float_image = load_images(index, main_folder, float_folder)

                    update_texture(texture_id1, main_image)
                    update_texture(texture_id2, float_image)

                    # Schedule loading for the next index if not already queued.
                    next_index = (index + direction) % png_paths_len
                    if image_buffer.get_future_for_index(next_index) is None:
                        new_future = executor.submit(load_images, next_index, main_folder, float_folder)
                        image_buffer.add_image_future(next_index, new_future)

                # Draw current textures.
                overlay_images_fast(texture_id1, texture_id2)
                pygame.display.flip()
                vid_clock.tick(FPS)
            except Exception as e:
                print(f"An error occurred: {e}")

def display_init(fullscreen=True):
    global fs_scale, fs_offset_x, fs_offset_y, fs_fullscreen_width, fs_fullscreen_height, image_size
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
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        gluOrtho2D(0, fs_fullscreen_width, 0, fs_fullscreen_height)
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
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        gluOrtho2D(0, win_width, 0, win_height)
        fs_scale = win_scale
        fs_offset_x = win_offset_x
        fs_offset_y = win_offset_y
    glMatrixMode(GL_MODELVIEW)
    glLoadIdentity()
    glEnable(GL_TEXTURE_2D)
    glEnable(GL_BLEND)
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
    # Initialize the blend shader.
    init_blend_shader()

def display_and_run(clock_source=FREE_CLOCK):
    global png_paths_len, main_folder_path, main_folder_count, \
           float_folder_path, float_folder_count, image_size, aspect_ratio
    random.seed(time.time())
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
