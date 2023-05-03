import csv
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from queue import SimpleQueue

import cv2
import pygame.time
from OpenGL.GL import *
from OpenGL.GLU import *

from midi_control import MidiProcessor
from midi_control import select_midi_input

if platform.system() == 'Darwin':
    from Quartz import CGWindowListCopyWindowInfo, kCGWindowListOptionOnScreenOnly, kCGNullWindowID
elif platform.system() == 'Linux':
    from Xlib import X, display

import platform
import pygame
from pygame.locals import *


def is_window_maximized():
    if platform.system() == 'Darwin':
        from AppKit import NSScreen
        import os
        pid = os.getpid()
        screen_frame = NSScreen.mainScreen().frame()
        screen_width, screen_height = screen_frame.size.width, screen_frame.size.height
        window_list = CGWindowListCopyWindowInfo(kCGWindowListOptionOnScreenOnly, kCGNullWindowID)
        for window in window_list:
            if window.get('kCGWindowOwnerPID') == pid:
                window_width = window.get('kCGWindowBounds')['Width']
                window_height = window.get('kCGWindowBounds')['Height']
                return window_width >= screen_width or window_height >= screen_height
        return False
    elif platform.system() == 'Linux':
        d = display.Display()
        root = d.screen().root
        pygame_window_id = pygame.display.get_wm_info()['window']
        window = d.create_resource_object('window', pygame_window_id)
        wm_state = d.intern_atom('_NET_WM_STATE')
        max_horz = d.intern_atom('_NET_WM_STATE_MAXIMIZED_HORZ')
        max_vert = d.intern_atom('_NET_WM_STATE_MAXIMIZED_VERT')
        fullscreen = d.intern_atom('_NET_WM_STATE_FULLSCREEN')
        wm_state_data = window.get_full_property(wm_state, X.AnyPropertyType)
        return (max_horz in wm_state_data.value and max_vert in wm_state_data.value) or (
                fullscreen in wm_state_data.value)
    else:
        raise NotImplementedError(f"Maximized window detection is not implemented for {platform.system()}")


def select_csv_file():
    csv_dir = 'generatedPngLists'
    csv_files = [f for f in os.listdir(csv_dir) if f.endswith('.csv')]

    if len(csv_files) == 0:
        print(f"No .csv files found in directory {csv_dir}")
        return None

    if len(csv_files) == 1:
        print("Only one .csv file found, defaulting to:")
        selected_file = os.path.join(csv_dir, csv_files[0])
        print(selected_file)
        return selected_file

    print("Please select a .csv file to use:")
    for i, f in enumerate(csv_files):
        print(f"{i + 1}: {f}")

    while True:
        try:
            selection = int(input("> "))
            if selection not in range(1, len(csv_files) + 1):
                raise ValueError
            break
        except ValueError:
            print("Invalid selection. Please enter a number corresponding to a file.")

    selected_file = os.path.join(csv_dir, csv_files[selection - 1])
    print(f"Selected file: {selected_file}")
    return selected_file


def get_aspect_ratio(image_path):
    image = cv2.imread(image_path)
    h, w, _ = image.shape
    aspect_ratio = h / w
    return aspect_ratio, w


def get_image_names_from_csv(file_path):
    with open(file_path, 'r') as csvfile:
        csv_reader = csv.reader(csvfile)
        png_paths = [row[1:] for row in csv_reader]

    return png_paths


def event_check(image_size, fullscreen, index, text_mode=False):
    width, height = image_size
    aspect_ratio = width / height
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()
        if event.type == KEYDOWN:
            if event.key == K_f:  # Press 'f' key to toggle fullscreen
                fullscreen = toggle_fullscreen(image_size, fullscreen)
            if event.key == K_i:  # Press 'i' key to jump around
                index += 500
                print("jump around")
            if event.key == K_t:  # press 't' toggle text
                text_mode = not text_mode
        elif event.type == VIDEORESIZE:
            new_width, new_height = event.size
            window_check = fullscreen
            if new_width / new_height > aspect_ratio:
                new_width = int(new_height * aspect_ratio)
            else:
                new_height = int(new_width / aspect_ratio)
            display_init((new_width, new_height), window_check)

    return fullscreen, index, text_mode


def toggle_fullscreen(image_size, current_fullscreen_status):
    if current_fullscreen_status and is_window_maximized():
        new_fullscreen_status = not current_fullscreen_status
        display_init(image_size, new_fullscreen_status)
    else:
        new_fullscreen_status = current_fullscreen_status
    return new_fullscreen_status


def read_image(image_path):
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


def display_image(texture_id, width, height, rgba=None):
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


def overlay_images_fast(texture_id_main, texture_id_float, image_size, main_rgba=None, float_rgba=None):
    width, height = image_size
    glClear(GL_COLOR_BUFFER_BIT)

    display_image(texture_id_float, width, height, rgba=float_rgba)
    display_image(texture_id_main, width, height, rgba=main_rgba)


def load_texture(texture_id, image):
    glBindTexture(GL_TEXTURE_2D, texture_id)
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, image.shape[1], image.shape[0], 0, GL_RGBA, GL_UNSIGNED_BYTE, image)


def load_images(index, png_paths, main_folder, float_folder):
    main_image = read_image(png_paths[index][main_folder])
    float_image = read_image(png_paths[index][float_folder])
    return main_image, float_image


def run_display(index, png_paths, main_folder, float_folder, image_size):
    font_size = 24
    font = pygame.font.Font(None, font_size)
    text_display = False
    fps = 30
    clock = pygame.time.Clock()
    fullscreen = True
    print_update = False

    current_time = time.time()
    prev_time = current_time
    old_index = index

    # Initialize PINGPONG related variables
    PINGPONG = True
    direction = 1

    # Create initial textures
    main_image, float_image = load_images(index, png_paths, main_folder, float_folder)
    texture_id1 = create_texture(main_image)
    texture_id2 = create_texture(float_image)

    with ThreadPoolExecutor(max_workers=4) as executor:
        image_queue = SimpleQueue()
        buffer_size = 5

        # Preload images
        for _ in range(buffer_size):
            index += direction
            index = max(0, min(index, len(png_paths) - 1))
            image_future = executor.submit(load_images, index, png_paths, main_folder, float_folder)
            image_queue.put(image_future)

        while True:

            fullscreen, index, text_display = event_check(image_size, fullscreen, index, text_display)
            if not image_queue.empty():
                main_image, float_image = image_queue.get().result()
                load_texture(texture_id1, main_image)
                load_texture(texture_id2, float_image)

                if PINGPONG:
                    # Reverse direction at boundaries
                    if (index + direction) < 0 or index >= len(png_paths) - 1:
                        direction = -direction

                index = (index + direction) % (len(png_paths) - 1)

                current_time = time.time()
                if (current_time - prev_time >= 1) and print_update:
                    print("index: ", index, "   index diff:  ", index - old_index)
                    old_index = index
                    prev_time = current_time
                    float_folder = (float_folder + 1) % (len(png_paths[0]) - 1)
                    part = clock.get_fps()
                    print(part)

                # Start a new future for the next images
                next_index = index + buffer_size * direction
                next_index = max(0, min(next_index, len(png_paths) - 1))
                image_future = executor.submit(load_images, next_index, png_paths, main_folder, float_folder)
                image_queue.put(image_future)

            overlay_images_fast(texture_id1, texture_id2, image_size)

            pygame.display.flip()
            clock.tick(fps)


def setup_blending():
    glEnable(GL_BLEND)
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)


def display_init(image_size, fullscreen=False, setup=False):
    pygame.init()
    width, height = image_size
    fullscreen_size = pygame.display.list_modes()[0]
    fullscreen_width, fullscreen_height = fullscreen_size
    # Calculate scaling factor and position for fullscreen mode
    scale = min(fullscreen_width / width, fullscreen_height / height)
    offset_x = int((fullscreen_width - width * scale) / 2)
    offset_y = int((fullscreen_height - height * scale) / 2)
    fullscreen == fullscreen
    flags = OPENGL
    flags |= DOUBLEBUF
    if fullscreen:
        flags |= FULLSCREEN
        pygame.display.set_caption('Fullscreen Mode')
        pygame.display.set_mode((fullscreen_width, fullscreen_height), flags)

        glViewport(offset_x, offset_y, int(width * scale), int(height * scale))
    else:
        flags |= RESIZABLE
        pygame.display.set_mode(image_size, flags)
        pygame.display.set_caption('Windowed Mode')
        glViewport(0, 0, width, height)
        # Explicitly set the window size back to image_size when returning to windowed mode
        pygame.display.set_mode(image_size, flags)

    if setup:
        glEnable(GL_TEXTURE_2D)
        setup_blending()
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        gluOrtho2D(0, width, 0, height)
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()


def display_and_run():
    midi_port = select_midi_input()
    midi_stuff = MidiProcessor()
    # midi_stuff.process_midi(midi_port)
    print(platform.system())
    csv_source = select_csv_file()
    png_paths = get_image_names_from_csv(csv_source)
    print(len(png_paths[0]))
    aspect_ratio, width = get_aspect_ratio(png_paths[0][0])
    print(width)
    width = 640
    height = int(width * aspect_ratio)
    image_size = (width, height)
    start_index = 0
    main_folder = 6
    float_folder = 6
    display_init(image_size, True, True)
    run_display(start_index, png_paths, main_folder, float_folder, image_size)


if __name__ == "__main__":
    display_and_run()
