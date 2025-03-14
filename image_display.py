import os
os.environ['PYOPENGL_ERROR_CHECKING'] = '0'
import time
from concurrent.futures import ThreadPoolExecutor
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

import random
import index_client

# Import functions from the new modules and settings.
from settings import (FULLSCREEN_MODE, MIDI_MODE, PINGPONG, BUFFER_SIZE, MAIN_FOLDER_PATH, FLOAT_FOLDER_PATH,
                      CLIENT_MODE, FPS, CLOCK_MODE, INITIAL_ROTATION, INITIAL_MIRROR)
from index_calculator import update_index
from folder_selector import update_folder_selection
# Import shared globals.
from globals import control_data_dictionary, folder_dictionary

# New imports for VBO rendering.
import numpy as np

# Import the renderer module for OpenGL functionality.
import renderer

# Import the image loader functions.
from image_loader import load_images, ImageLoaderBuffer, set_folder_paths, set_png_paths_len

# Import event handling functions.
from event_handler import event_check

# Global variables and preallocated vertex array.
run_mode = True
pause_mode = False
png_paths_len = 100
float_folder_count = 100
main_folder_count = 100
image_size = (800, 600)
aspect_ratio = 1.333
text_display = False
main_folder_path = MAIN_FOLDER_PATH
float_folder_path = FLOAT_FOLDER_PATH

# Globals for window/display and transformation.
fs_scale = None
fs_offset_x = None
fs_offset_y = None
fs_fullscreen_width = None
fs_fullscreen_height = None

# Globals for rotation and mirroring.
rotation_angle = INITIAL_ROTATION  # in degrees, e.g. 270 initially
mirror_mode = INITIAL_MIRROR       # 0 for normal, 1 for mirrored

def get_aspect_ratio(image_path):
    image = pygame.image.load(image_path)
    w, h = image.get_size()
    a_ratio = h / w
    print(f'This is {w} wide and {h} tall, with an aspect ratio of {a_ratio}')
    return a_ratio, w, h

def display_init(fullscreen=True):
    """
    Sets up the pygame display and window modes, computes transformation parameters,
    computes the MVP matrix, and then delegates the OpenGL state initialization to renderer.
    """
    global fs_scale, fs_offset_x, fs_offset_y, fs_fullscreen_width, fs_fullscreen_height, image_size, rotation_angle, mirror_mode
    w, h = image_size
    if rotation_angle % 180 == 90:
        effective_w = h
        effective_h = w
    else:
        effective_w = w
        effective_h = h
    if fullscreen:
        modes = pygame.display.list_modes()
        if not modes:
            raise RuntimeError("No display modes available!")
        fs_fullscreen_width, fs_fullscreen_height = modes[0]
        scale_x = fs_fullscreen_width / effective_w
        scale_y = fs_fullscreen_height / effective_h
        fs_scale = min(scale_x, scale_y)
        scaled_width = effective_w * fs_scale
        scaled_height = effective_h * fs_scale
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
        win_height = int(400 * effective_h / effective_w)
        scale_x = win_width / effective_w
        scale_y = win_height / effective_h
        win_scale = min(scale_x, scale_y)
        scaled_width = effective_w * win_scale
        scaled_height = effective_h * win_scale
        fs_offset_x = int((win_width - scaled_width) / 2)
        fs_offset_y = int((win_height - scaled_height) / 2)
        flags = pygame.OPENGL | pygame.DOUBLEBUF | pygame.RESIZABLE
        pygame.display.set_caption('Windowed Mode')
        pygame.display.set_mode((win_width, win_height), flags, vsync=1)
        glViewport(0, 0, win_width, win_height)
        viewport_width = win_width
        viewport_height = win_height
        fs_scale = win_scale
    mvp = np.array([
        [2.0/viewport_width, 0, 0, -1],
        [0, 2.0/viewport_height, 0, -1],
        [0, 0, -1, 0],
        [0, 0, 0, 1]
    ], dtype=np.float32)
    # Pass transformation parameters to renderer.
    renderer.set_transform_parameters(fs_scale, fs_offset_x, fs_offset_y, image_size, rotation_angle, mirror_mode)
    renderer.setup_opengl(mvp)

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
    global run_mode, png_paths_len, float_folder_count, main_folder_count
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
        texture_id1 = renderer.create_texture(main_image)
        texture_id2 = renderer.create_texture(float_image)
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
                    renderer.update_texture(texture_id1, main_image)
                    renderer.update_texture(texture_id2, float_image)
                    next_index = (index + direction) % png_paths_len
                    if image_buffer.get_future_for_index(next_index) is None:
                        new_future = executor.submit(load_images, next_index, main_folder, float_folder)
                        image_buffer.add_image_future(next_index, new_future)
                renderer.overlay_images_fast(texture_id1, texture_id2)
                pygame.display.flip()
                vid_clock.tick(FPS)
            except Exception as e:
                print(f"An error occurred: {e}")

def display_and_run(clock_source=CLOCK_MODE):
    global png_paths_len, main_folder_path, main_folder_count, \
           float_folder_path, float_folder_count, image_size, aspect_ratio
    random.seed(time.time())
    csv_source, main_folder_path, float_folder_path = calculators.init_all(clock_source)
    print(platform.system(), "midi clock mode is:", clock_source)
    main_folder_count = len(main_folder_path[0])
    float_folder_count = len(float_folder_path[0])
    png_paths_len = len(main_folder_path) - 1
    aspect_ratio, width, height = get_aspect_ratio(main_folder_path[0][0])
    image_size = (width, height)
    print("Image size:", image_size)
    # Set folder paths and png_paths_len in the image loader.
    set_folder_paths(main_folder_path, float_folder_path)
    set_png_paths_len(png_paths_len)
    run_display_setup()

if __name__ == "__main__":
    display_and_run()
