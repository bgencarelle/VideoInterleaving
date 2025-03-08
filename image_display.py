import math
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from platform import system
from queue import SimpleQueue
import threading

import datetime
import pygame.time
from OpenGL.GL import *
from OpenGL.GLU import *

import calculators
import midi_control
import platform
import pygame
from pygame.locals import *

import cv2
import webp
import random
import index_client

FULLSCREEN_MODE = True
MTC_CLOCK = 0
MIDI_CLOCK = 1
MIXED_CLOCK = 2
CLIENT_MODE = 3

FREE_CLOCK = 255

clock_mode = FREE_CLOCK
midi_mode = False

FPS = 60
IPS = 30  # images per second (used for index updates)
run_mode = True

BUFFER_SIZE = FPS // 4
PINGPONG = True

vid_clock = None
pause_mode = False
png_paths_len = 0
main_folder_path = 0
float_folder_path = 0
float_folder_count = 0

main_folder_count = 0
image_size = (800, 600)
aspect_ratio = 1.333
text_display = False

# Global launch_time variable
launch_time = None

def set_launch_time(from_birth=False):
    global launch_time
    if from_birth:
        # Create a datetime object for November 17, 1978, at 7:11 AM EST (UTC-5)
        fixed_datetime = datetime.datetime(1978, 11, 17, 7, 11, tzinfo=datetime.timezone(datetime.timedelta(hours=-5)))
        launch_time = fixed_datetime.timestamp()
    else:
        launch_time = time.time()

set_launch_time(from_birth=True)

# Custom event for timer updates
UPDATE_INDEX_EVENT = pygame.USEREVENT + 1

control_data_dictionary = {
    'Note_On': (0, 127, 0),
    'Note_Off': (None, None, None),
    'Modulation': (0, 0),
    'Index_and_Direction': (0, 1),
    'BPM': 120,
}

folder_dictionary = {
    'Main_and_Float_Folders': (0, 8),
}

valid_modes = {"MTC_CLOCK": MTC_CLOCK, "MIDI_CLOCK": MIDI_CLOCK, "MIXED_CLOCK": MIXED_CLOCK,
               "CLIENT_MODE": CLIENT_MODE, "FREE_CLOCK": FREE_CLOCK}

def set_clock_mode(mode=None):
    global clock_mode, midi_mode
    if mode and mode in valid_modes.values():
        clock_mode = mode
    else:
        while True:
            print("Please choose a clock mode from the following options:")
            for i, (mode_name, mode_value) in enumerate(valid_modes.items(), 1):
                print(f"{i}. {mode_name}")
            user_choice = input("Enter the number corresponding to your choice: ")
            if user_choice.isdigit() and 1 <= int(user_choice) <= len(valid_modes):
                clock_mode = list(valid_modes.values())[int(user_choice) - 1]
                print(f"Clock mode has been set to {list(valid_modes.keys())[int(user_choice) - 1]}")
                break
            else:
                print(f"Invalid input: '{user_choice}'. Please try again.")
    midi_mode = True if (clock_mode < CLIENT_MODE) else False
    print("YER", list(valid_modes.keys())[list(valid_modes.values()).index(clock_mode)])

def toggle_fullscreen(current_fullscreen_status):
    new_fullscreen = not current_fullscreen_status
    display_init(new_fullscreen)
    return new_fullscreen

def event_check(fullscreen):
    global image_size, run_mode, pause_mode
    width, height = image_size
    aspect_ratio = width / height
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            run_mode = False
        elif event.type == UPDATE_INDEX_EVENT:
            # Update the index based on elapsed time using the timer event.
            index, direction = calculate_free_clock_index(png_paths_len, PINGPONG)
            control_data_dictionary['Index_and_Direction'] = index, direction
        elif event.type == KEYDOWN:
            if event.key == K_q:
                run_mode = False
                pygame.quit()
            if event.key == K_f:
                fullscreen = toggle_fullscreen(fullscreen)
        elif event.type == VIDEORESIZE:
            new_width, new_height = event.size
            if new_width / new_height > aspect_ratio:
                new_width = int(new_height * aspect_ratio)
                image_size = new_width, new_height
            else:
                new_height = int(new_width / aspect_ratio)
                image_size = new_width, new_height
            display_init(fullscreen)
    return fullscreen

def get_aspect_ratio(image_path):
    image = pygame.image.load(image_path)
    w, h = image.get_size()
    a_ratio = h / w
    print(f'this is {w} wide and {h} tall, with an aspect ratio of {aspect_ratio}')
    return a_ratio, w, h,

def read_image(image_path):
    image_np = None
    if image_path.endswith('.webp'):
        image_np = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
        image_np = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGBA)
    if image_path.endswith('.png'):
        image_np = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
        image_np = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGBA)
    return image_np

def create_texture(image):
    texture_id = glGenTextures(1)
    glBindTexture(GL_TEXTURE_2D, texture_id)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, image.shape[1], image.shape[0], 0, GL_RGBA, GL_UNSIGNED_BYTE, image)
    return texture_id

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

def load_texture(texture_id, image):
    glBindTexture(GL_TEXTURE_2D, texture_id)
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, image.shape[1], image.shape[0], 0, GL_RGBA, GL_UNSIGNED_BYTE, image)

def load_images(index, main_folder, float_folder):
    main_image = read_image(main_folder_path[index][main_folder])
    float_image = read_image(float_folder_path[index][float_folder])
    return main_image, float_image

# Globals for running FPS calculation
last_call_time = None
last_index = None
fps_total = 0.0
fps_count = 0

def calculate_free_clock_index(total_images, pingpong=True):
    global launch_time, last_call_time, last_index, fps_total, fps_count
    elapsed = time.time() - launch_time
    if pingpong:
        period = 2 * (total_images - 1)
        raw_index = int(elapsed * IPS) % period
        if raw_index >= total_images:
            index = period - raw_index
            direction = -1
        else:
            index = raw_index
            direction = 1
    else:
        index = int(elapsed * IPS) % total_images
        direction = 1

    current_time = time.time()
    if last_call_time is not None:
        dt = current_time - last_call_time
        if dt > 0:
            delta_index = abs(index - last_index)
            instantaneous_fps = delta_index / dt
            fps_total += instantaneous_fps
            fps_count += 1
            running_fps = fps_total / fps_count
        else:
            instantaneous_fps = 0
            running_fps = 0
    else:
        instantaneous_fps = 0
        running_fps = 0

    last_call_time = current_time
    last_index = index
    return index, direction

def update_index_and_folders(index, direction):
    global control_data_dictionary
    if midi_mode:
        midi_control.process_midi(clock_mode)
        control_data_dictionary = midi_control.midi_data_dictionary
        index, direction = control_data_dictionary['Index_and_Direction']
    elif clock_mode == CLIENT_MODE:
        control_data_dictionary = index_client.midi_data_dictionary
        index, direction = control_data_dictionary['Index_and_Direction']
    elif clock_mode == FREE_CLOCK:
        # For FREE_CLOCK, assume the timer event already updates the index.
        index, direction = control_data_dictionary['Index_and_Direction']
    update_control_data(index, direction)
    return index, direction

def update_control_data(index, direction):
    rand_mult = random.randint(1, 9)
    rand_start = 8 * (IPS - (rand_mult * rand_mult // 2))
    main_folder, float_folder = folder_dictionary['Main_and_Float_Folders']
    if clock_mode == FREE_CLOCK:
        if index <= rand_start * direction or (index > 100 * rand_start and index < 140 * rand_start):
            float_folder = 0
            main_folder = 0
        elif index % (IPS * rand_mult) == 0:
            float_folder = random.randint(0, float_folder_count - 1)
            rand_mult = random.randint(1, 12)
        elif index % (2 * IPS * rand_mult - 1) == 0:
            main_folder = random.randint(0, main_folder_count - 1)
    else:
        note, channel, _ = control_data_dictionary['Note_On']
        modulation, channel = control_data_dictionary['Modulation']
        mod_value = int(modulation / 127 * float_folder_count)
        float_folder = (mod_value) % float_folder_count
        main_folder = (note % 12) % main_folder_count
    folder_dictionary['Main_and_Float_Folders'] = main_folder, float_folder

def print_index_diff_wrapper():
    storage = {'old_index': 0, 'prev_time': 0}
    def print_index_diff(index):
        current_time = time.time()
        if (current_time - storage['prev_time']) >= 1 and not storage['old_index'] == index:
            print("index: ", index, "   index diff:  ", index - storage['old_index'])
            storage['old_index'] = index
            storage['prev_time'] = current_time
            fippy = vid_clock.get_fps()
            print(f'fps: {fippy}')
            print(f'midi_control.bpm: {midi_control.bpm}')
    return print_index_diff

print_index_diff_function = print_index_diff_wrapper()

def run_display_setup():
    global vid_clock
    if midi_mode:
        midi_control.midi_control_stuff_main()
    elif clock_mode == CLIENT_MODE:
        threading.Thread(target=index_client.start_client, daemon=True).start()
    pygame.init()
    pygame.mouse.set_visible(False)
    # Set up the timer to fire at (1000/IPS) ms intervals.
    pygame.time.set_timer(UPDATE_INDEX_EVENT, int(1000 / IPS))
    display_init(True)
    vid_clock = pygame.time.Clock()
    run_display()
    return

def run_display():
    global run_mode
    index, direction = control_data_dictionary['Index_and_Direction']
    buffer_index, buffer_direction = update_index_and_folders(0, 1)
    fullscreen = True
    index_changed = False

    def queue_image(buffer_idx, main_folder_q, float_folder_q, q_image_queue):
        buffer_idx = max(0, min(buffer_idx, png_paths_len - 1))
        image_future = executor.submit(load_images, buffer_idx, main_folder_q, float_folder_q)
        q_image_queue.put(image_future)

    with ThreadPoolExecutor(max_workers=2) as executor:
        update_index_and_folders(index, direction)
        index, direction = control_data_dictionary['Index_and_Direction']
        main_folder, float_folder = folder_dictionary['Main_and_Float_Folders']
        image_queue = SimpleQueue()

        main_image, float_image = load_images(index, main_folder, float_folder)
        texture_id1 = create_texture(main_image)
        texture_id2 = create_texture(float_image)

        for _ in range(BUFFER_SIZE):
            buffer_index += direction
            if buffer_index >= png_paths_len or buffer_index < 0:
                buffer_index += direction * -1
            queue_image(buffer_index, main_folder, float_folder, image_queue)

        last_skipped_index = -1

        while run_mode:
            try:
                prev_index = index
                fullscreen = event_check(fullscreen)
                update_index_and_folders(index, direction)
                index, direction = control_data_dictionary['Index_and_Direction']
                main_folder, float_folder = folder_dictionary['Main_and_Float_Folders']

                if index_changed != index:
                    buffer_index = index
                    if abs(prev_index - index) > 1 and last_skipped_index != index:
                        last_skipped_index = index

                    buffer_synced = False
                    discarded_images = []
                    while not buffer_synced and not image_queue.empty():
                        main_image, float_image = image_queue.get().result()
                        if buffer_index == index:
                            buffer_synced = True
                            buffer_direction = direction
                            load_texture(texture_id1, main_image)
                            load_texture(texture_id2, float_image)
                            if buffer_index > png_paths_len or buffer_index < 0:
                                print("AH SHIT")
                                buffer_index += buffer_direction * -1
                            queue_image(buffer_index, main_folder, float_folder, image_queue)
                        else:
                            discarded_images.append((main_image, float_image))
                    for discarded_image in discarded_images:
                        image_queue.put(discarded_image)

                overlay_images_fast(texture_id1, texture_id2, index)
                pygame.display.flip()
                vid_clock.tick(FPS)
            except Exception as e:
                print(f"An error occurred: {e}")


def display_init(fullscreen=True):
    global fs_scale, fs_offset_x, fs_offset_y, fs_fullscreen_width, fs_fullscreen_height
    w, h = image_size  # native image dimensions

    if fullscreen:
        # Fullscreen: use the native display resolution
        fullscreen_size = pygame.display.list_modes()[0]
        fs_fullscreen_width, fs_fullscreen_height = fullscreen_size
        fs_scale = min(fs_fullscreen_width / w, fs_fullscreen_height / h)
        fs_offset_x = int((fs_fullscreen_width - w * fs_scale) / 2)
        fs_offset_y = int((fs_fullscreen_height - h * fs_scale) / 2)

        flags = OPENGL | DOUBLEBUF | FULLSCREEN
        pygame.display.set_caption('Fullscreen Mode')
        pygame.display.set_mode((fs_fullscreen_width, fs_fullscreen_height), flags, vsync=1)
        glViewport(0, 0, fs_fullscreen_width, fs_fullscreen_height)

        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        gluOrtho2D(0, fs_fullscreen_width, 0, fs_fullscreen_height)
    else:
        # Windowed: force window width to 400, with height computed from the image's aspect ratio
        win_width = 400
        win_height = int(400 * h / w)
        win_scale = min(win_width / w, win_height / h)
        win_offset_x = int((win_width - w * win_scale) / 2)
        win_offset_y = int((win_height - h * win_scale) / 2)

        flags = OPENGL | DOUBLEBUF | RESIZABLE
        pygame.display.set_caption('Windowed Mode')
        pygame.display.set_mode((win_width, win_height), flags, vsync=1)
        glViewport(0, 0, win_width, win_height)

        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        gluOrtho2D(0, win_width, 0, win_height)

        # Use the windowed scale and offsets for drawing
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
    set_clock_mode(clock_source)
    csv_source, main_folder_path, float_folder_path = calculators.init_all(clock_source)
    print(platform.system(), "midi clock mode is:", clock_mode)
    main_folder_count = len(main_folder_path[0])
    float_folder_count = len(float_folder_path[0])
    png_paths_len = len(main_folder_path) - 1
    aspect_ratio, width, height = get_aspect_ratio(main_folder_path[0][0])
    image_size = (width, height)
    print(image_size)
    run_display_setup()

if __name__ == "__main__":
    display_and_run()
