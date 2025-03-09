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
from settings import (FULLSCREEN_MODE, MTC_CLOCK, MIDI_CLOCK, MIXED_CLOCK,
                      CLIENT_MODE, FREE_CLOCK, IPS, CLOCK_MODE, VALID_MODES)
from index_calculator import set_launch_time, update_index
from folder_selector import update_folder_selection

# Import shared globals
from globals import control_data_dictionary, folder_dictionary

clock_mode = CLOCK_MODE
midi_mode = False

FPS = 60
run_mode = True

BUFFER_SIZE = FPS // 4  # e.g., 15 if FPS==60
PINGPONG = True

pause_mode = False
png_paths_len = 0
main_folder_path = 0
float_folder_path = 0
float_folder_count = 0
main_folder_count = 0
image_size = (800, 600)
aspect_ratio = 1.333
text_display = False

launch_time = None

def set_launch_time_wrapper(from_birth=False):
    set_launch_time(from_birth)

set_launch_time_wrapper(from_birth=True)


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

# Global dictionary to track texture dimensions by texture_id
texture_dimensions = {}

def create_texture(image):
    texture_id = glGenTextures(1)
    glBindTexture(GL_TEXTURE_2D, texture_id)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST)
    w, h = image.shape[1], image.shape[0]
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, w, h, 0, GL_RGBA, GL_UNSIGNED_BYTE, image)
    texture_dimensions[texture_id] = (w, h)
    return texture_id

def update_texture(texture_id, new_image):
    glBindTexture(GL_TEXTURE_2D, texture_id)
    w, h = new_image.shape[1], new_image.shape[0]
    expected = texture_dimensions.get(texture_id, (None, None))
    if expected != (w, h):
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, w, h, 0, GL_RGBA, GL_UNSIGNED_BYTE, new_image)
        texture_dimensions[texture_id] = (w, h)
    else:
        glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, w, h, GL_RGBA, GL_UNSIGNED_BYTE, new_image)

def display_image(texture_id, width, height, rgba=(1, 1, 1, 1)):
    glBindTexture(GL_TEXTURE_2D, texture_id)
    if rgba:
        glColor4f(*rgba)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
    glBegin(GL_QUADS)
    glTexCoord2f(0, 1)
    glVertex2f(fs_offset_x, fs_offset_y)
    glTexCoord2f(1, 1)
    glVertex2f(fs_offset_x + width * fs_scale, fs_offset_y)
    glTexCoord2f(1, 0)
    glVertex2f(fs_offset_x + width * fs_scale, fs_offset_y + height * fs_scale)
    glTexCoord2f(0, 0)
    glVertex2f(fs_offset_x, fs_offset_y + height * fs_scale)
    glEnd()
    if rgba:
        glColor4f(1, 1, 1, 1)
        glDisable(GL_BLEND)

def set_rgba_relative():
    main_alpha = 1
    float_alpha = 1
    main_rgba = (1, 1, 1, main_alpha)
    float_rgba = (1, 1, 1, float_alpha)
    return main_rgba, float_rgba

def overlay_images_fast(texture_id_main, texture_id_float, index=0, background_color=(0, 0, 0)):
    global image_size
    width, height = image_size
    main_rgba, float_rgba = set_rgba_relative()
    glClearColor(background_color[0] / 255, background_color[1] / 255, background_color[2] / 255, 1.0)
    glClear(GL_COLOR_BUFFER_BIT)
    display_image(texture_id_main, width, height, rgba=float_rgba)
    display_image(texture_id_float, width, height, rgba=main_rgba)

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
    if midi_mode:
        midi_control.midi_control_stuff_main()
    elif CLOCK_MODE == CLIENT_MODE:
        threading.Thread(target=index_client.start_client, daemon=True).start()
    pygame.init()
    pygame.mouse.set_visible(False)
    display_init(True)
    run_display()
    return

def run_display():
    global run_mode
    vid_clock = pygame.time.Clock()
    # Initial index and folder update
    index, direction = update_index(png_paths_len, PINGPONG)
    control_data_dictionary['Index_and_Direction'] = (index, direction)
    update_folder_selection(index, direction, float_folder_count, main_folder_count)
    fullscreen = True
    image_buffer = ImageLoaderBuffer(BUFFER_SIZE)
    with ThreadPoolExecutor(max_workers=8) as executor:
        main_folder, float_folder = folder_dictionary['Main_and_Float_Folders']
        initial_index = index
        for i in range(BUFFER_SIZE):
            buf_idx = (initial_index + i) % png_paths_len
            future = executor.submit(load_images, buf_idx, main_folder, float_folder)
            image_buffer.add_image_future(buf_idx, future)
        main_image, float_image = load_images(index, main_folder, float_folder)
        texture_id1 = create_texture(main_image)
        texture_id2 = create_texture(float_image)
        while run_mode:
            try:
                fullscreen = event_check(fullscreen)
                index, direction = update_index(png_paths_len, PINGPONG)
                control_data_dictionary['Index_and_Direction'] = (index, direction)
                update_folder_selection(index, direction, float_folder_count, main_folder_count)
                main_folder, float_folder = folder_dictionary['Main_and_Float_Folders']
                future = image_buffer.get_future_for_index(index)
                if future is not None:
                    main_image, float_image = future.result()
                    update_texture(texture_id1, main_image)
                    update_texture(texture_id2, float_image)
                    next_index = (index + direction) % png_paths_len
                    new_future = executor.submit(load_images, next_index, main_folder, float_folder)
                    image_buffer.add_image_future(next_index, new_future)
                else:
                    next_index = (index + direction) % png_paths_len
                    new_future = executor.submit(load_images, next_index, main_folder, float_folder)
                    image_buffer.add_image_future(next_index, new_future)
                overlay_images_fast(texture_id1, texture_id2, index)
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
