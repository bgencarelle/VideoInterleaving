import math
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from platform import system
from queue import SimpleQueue
import threading

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

FULLSCREEN_MODE = False
MTC_CLOCK = 0
MIDI_CLOCK = 1
MIXED_CLOCK = 2
CLIENT_MODE = 3
FREE_CLOCK = 255

clock_mode = FREE_CLOCK
midi_mode = False

FPS = 30
run_mode = True

BUFFER_SIZE = FPS // 3
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


control_data_dictionary = {
    'Note_On': (0, 127, 0),
    'Note_Off': (None, None, None),
    'Modulation': (0, 0),
    'Index_and_Direction': (0, 1),
    'BPM': 120,
    # 'Stop': False,
    # 'Start': False,
    # 'Pause': False,
    # 'Reset': False
}

folder_dictionary = {
    'Main_and_Float_Folders': (0, 8),
}

valid_modes = {"MTC_CLOCK": MTC_CLOCK, "MIDI_CLOCK": MIDI_CLOCK, "MIXED_CLOCK": MIXED_CLOCK,
               "CLIENT_MODE": CLIENT_MODE, "FREE_CLOCK": FREE_CLOCK}


def set_clock_mode(mode=None):
    global clock_mode, midi_mode
    # If a mode argument is provided and valid, silently set the clock_mode
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
    global image_size
    if current_fullscreen_status:
        # Switch back to windowed mode
        pygame.display.set_mode(image_size, pygame.RESIZABLE | OPENGL | DOUBLEBUF)
        pygame.display.set_caption('Windowed Mode')
    else:
        # Switch to fullscreen mode
        pygame.display.set_mode((0, 0), FULLSCREEN | OPENGL | DOUBLEBUF)
        pygame.display.set_caption('Fullscreen Mode')
    return not current_fullscreen_status


def event_check(fullscreen):
    global image_size, run_mode, pause_mode
    width, height = image_size
    aspect_ratio = width / height
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            run_mode = False
        if event.type == KEYDOWN:
            if event.key == K_q:  # Press 'p' key to toggle pause
                # print("bailing out")
                run_mode = False
                pygame.quit()
            if event.key == K_f:  # Press 'f' key to toggle fullscreen
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
    # Load image with Pygame
    image = pygame.image.load(image_path)

    # Get image dimensions
    w, h = image.get_size()

    # Calculate aspect clock_frame_ratio
    a_ratio = h / w
    print(f'this is {w} wide and {h} tall, with an aspect ratio of {aspect_ratio}')
    return a_ratio, w, h,

def read_image(image_path):
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
    glVertex2f(0, 0)
    glTexCoord2f(1, 1)
    glVertex2f(width, 0)
    glTexCoord2f(1, 0)
    glVertex2f(width, height)
    glTexCoord2f(0, 0)
    glVertex2f(0, height)
    glEnd()

    if rgba:
        glColor4f(1, 1, 1, 1)
        glDisable(GL_BLEND)


def set_rgba_relative():
    main_alpha = 1
    float_alpha = 1 #
    main_rgba = (1, 1, 1, main_alpha)
    float_rgba = (1, 1, 1, float_alpha)
    return main_rgba, float_rgba


def overlay_images_fast(texture_id_main, texture_id_float, index=0, background_color=(0, 0, 0)):
    global image_size
    width, height = image_size
    main_rgba, float_rgba = set_rgba_relative()
    glClearColor(background_color[0] / 255, background_color[1] / 255, background_color[2] / 255, 1.0)
    glClear(GL_COLOR_BUFFER_BIT)

    display_image(texture_id_main, width, height, rgba=main_rgba)
    display_image(texture_id_float, width, height, rgba=float_rgba)


def load_texture(texture_id, image):
    glBindTexture(GL_TEXTURE_2D, texture_id)
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, image.shape[1], image.shape[0], 0, GL_RGBA, GL_UNSIGNED_BYTE, image)


def load_images(index, main_folder, float_folder):
    main_image = read_image(main_folder_path[index][main_folder])
    float_image = read_image(float_folder_path[index][float_folder])
    return main_image, float_image


def update_index_and_folders(index, direction):
    global control_data_dictionary
    if midi_mode:
        midi_control.process_midi(clock_mode)
        control_data_dictionary = midi_control.midi_data_dictionary
        index, direction = control_data_dictionary['Index_and_Direction']
        # print(index*direction)
    elif clock_mode == CLIENT_MODE:
        control_data_dictionary = index_client.midi_data_dictionary
        index, direction = control_data_dictionary['Index_and_Direction']
    elif clock_mode == FREE_CLOCK:
        if PINGPONG:
            # Reverse index_direction at boundaries
            if (index + direction) < 0 or index + direction >= png_paths_len:
                direction *= -1
            index += direction
        else:
            index = (index + 1) % png_paths_len
        control_data_dictionary['Index_and_Direction'] = index, direction
    update_control_data(index, direction)
    return index, direction


def update_control_data(index, direction):
    rand_mult = random.randint(1, 9)
    rand_start = 8 * (FPS - (rand_mult * rand_mult // 2))

    main_folder, float_folder = folder_dictionary['Main_and_Float_Folders']
    if clock_mode == FREE_CLOCK:

        # print('index position and stuff ', index, ' : ', rand_start)
        if index <= rand_start * direction or (index > 100 * rand_start and index < 140 * rand_start):
            float_folder = 0
            main_folder = 0
            print('in stable mode')
        elif index % (FPS * rand_mult) == 0:
            float_folder = random.randint(0, float_folder_count - 1)
            print('background layer:  ', float_folder)
            rand_mult = random.randint(1, 12)
        elif index % (2 * FPS * rand_mult - 1) == 0:
            main_folder = random.randint(0, main_folder_count - 1)
            print('foreground: ', main_folder)

    else:
        # print(control_data_dictionary['Note_On'])
        note, channel, _ = control_data_dictionary['Note_On']
        modulation, channel = control_data_dictionary['Modulation']
        mod_value = int(modulation / 127 * float_folder_count)
        # print(mod_value)
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
    display_init(False)
    vid_clock = pygame.time.Clock()
    run_display()
    return


def run_display():
    global run_mode
    index, direction = control_data_dictionary['Index_and_Direction']
    buffer_index, buffer_direction = update_index_and_folders(0, 1)
    fullscreen = False

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

                # print_index_diff_function(index)
                if index_changed != index:
                    buffer_index = index
                    if abs(prev_index - index) > 1 and last_skipped_index != index:
                        last_skipped_index = index

                    buffer_synced = False
                    discarded_images = []
                    while not buffer_synced and not image_queue.empty():
                        main_image, float_image = image_queue.get().result()
                        if buffer_index == index:
                            # print("MATCH!")
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

                    # Put discarded images back into the queue
                    for discarded_image in discarded_images:
                        image_queue.put(discarded_image)

                overlay_images_fast(texture_id1, texture_id2, index)

                pygame.display.flip()
                vid_clock.tick(FPS)
            except Exception as e:
                print(f"An error occurred: {e}")
                # Optionally, you can add a time.sleep() here to throttle the loop in case of an error
                # time.sleep(0.1)



def display_init(fullscreen=False):
    w, h = image_size
    fullscreen_size = pygame.display.list_modes()[0]
    fullscreen_width, fullscreen_height = fullscreen_size
    # Calculate scaling factor and position for fullscreen mode
    scale = min(fullscreen_width / w, fullscreen_height / h)
    offset_x = int((fullscreen_width - w * scale) / 2)
    offset_y = int((fullscreen_height - h * scale) / 2)
    flags = OPENGL
    flags |= DOUBLEBUF
    if fullscreen:
        flags = OPENGL
        flags |= DOUBLEBUF
        flags |= FULLSCREEN
        pygame.display.set_caption('Fullscreen Mode')
        pygame.display.set_mode((fullscreen_width, fullscreen_height), flags)
        glViewport(offset_x, offset_y, int(w * scale), int(h * scale))
    else:
        flags |= RESIZABLE
        pygame.display.set_mode(image_size, flags)
        pygame.display.set_caption('Windowed Mode')
        glViewport(0, 0, w, h)  # Update to use original width and height

    glEnable(GL_TEXTURE_2D)
    glEnable(GL_BLEND)
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()
    if fullscreen:
        gluOrtho2D(0, int(w * scale), 0, int(h * scale))  # Use scaled dimensions in fullscreen mode
    else:
        gluOrtho2D(0, w, 0, h)  # Use original dimensions in windowed mode
    glMatrixMode(GL_MODELVIEW)
    glLoadIdentity()


def display_and_run(clock_source=None):
    global png_paths_len, main_folder_path, main_folder_count, \
        float_folder_path, float_folder_count, image_size, aspect_ratio
    random.seed(time.time())
    set_clock_mode(clock_source)
    csv_source, main_folder_path, float_folder_path = calculators.init_all()
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
