import sys
import time
from concurrent.futures import ThreadPoolExecutor
from queue import SimpleQueue

import calculators
from calculators import select_csv_file, get_image_names_from_csv
import midi_control
import cv2
import pygame.time
from OpenGL.GL import *
from OpenGL.GLU import *
from platform import system
import threading

if system() == 'Darwin':
    from Quartz import CGWindowListCopyWindowInfo, kCGWindowListOptionOnScreenOnly, kCGNullWindowID
elif system() == 'Linux':
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


def event_check(image_size, fullscreen, index, direction=1, text_mode=False, force_quit=False):
    width, height = image_size
    aspect_ratio = width / height
    old_direction = 1
    paused = False
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            sys.exit()
            sys.exit()
            pygame.quit()
        if event.type == KEYDOWN:
            if event.key == K_q:  # Press 'p' key to toggle pause
                force_quit=True
                print("bailing out")
                pygame.quit()
                sys.exit()
            if event.key == K_f:  # Press 'f' key to toggle fullscreen
                pass
               # fullscreen = toggle_fullscreen(image_size, fullscreen)
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

    return fullscreen, index, text_mode, force_quit


def get_aspect_ratio(image_path):
    # Load image with Pygame
    image = pygame.image.load(image_path)

    # Get image dimensions
    w, h = image.get_size()

    # Calculate aspect ratio
    aspect_ratio = h / w

    return aspect_ratio, w


def toggle_fullscreen(image_size, current_fullscreen_status):
    if not current_fullscreen_status and is_window_maximized():
        new_fullscreen_status = current_fullscreen_status
    else:
        new_fullscreen_status = not current_fullscreen_status
        display_init(image_size, new_fullscreen_status)
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


def run_display(index, direction, png_paths, main_folder, float_folder, image_size):
    font_size = 24
    font = pygame.font.Font(None, font_size)
    text_display = False
    fps = 30
    clock = pygame.time.Clock()
    fullscreen = False
    print_update = True
    force_quit = False
    midi_mode = True
    current_time = time.time()
    prev_time = current_time
    old_index = index


    # Initialize PINGPONG related variables
    pingpong= True
    direction = 1

    # Create initial textures
    main_image, float_image = load_images(index, png_paths, main_folder, float_folder)
    texture_id1 = create_texture(main_image)
    texture_id2 = create_texture(float_image)
    def queue_image(index, direction, png_paths, main_folder, float_folder, image_queue):
        index = max(0, min(index, len(png_paths) - 1))
        image_future = executor.submit(load_images, index, png_paths, main_folder, float_folder)
        image_queue.put(image_future)

    with ThreadPoolExecutor(max_workers=4) as executor:
        image_queue = SimpleQueue()
        buffer_size = 5

        # Preload images
        for _ in range(buffer_size):
            index += direction
            index = max(0, min(index, len(png_paths) - 1))
            queue_image(index, direction, png_paths, main_folder, float_folder, image_queue)

        while not force_quit:
            fullscreen, index, text_display, force_quit = event_check(image_size, fullscreen, index, text_display, force_quit)
            if midi_mode:
                index = midi_control.index
                direction = midi_control.direction
            current_time = time.time()
            if (current_time - prev_time >= 4) and print_update:
                print("index: ", index, "   index diff:  ", index - old_index)
                old_index = index
                prev_time = current_time
                float_folder = (float_folder + 1) % (len(png_paths[0]) )
                part = clock.get_fps()
                print(part)
            if not image_queue.empty():
                main_image, float_image = image_queue.get().result()
                load_texture(texture_id1, main_image)
                load_texture(texture_id2, float_image)

                if not midi_mode:
                    if pingpong:
                        # Reverse direction at boundaries
                        if (index + direction) < 0 or index >= len(png_paths) - 1:
                            direction = -direction
                        index = (index + direction) % (len(png_paths) - 1)

                # Start a new future for the next images
                next_index = index + buffer_size * direction
                queue_image(next_index, direction, png_paths, main_folder, float_folder, image_queue)

            overlay_images_fast(texture_id1, texture_id2, image_size)

            pygame.display.flip()
            clock.tick(fps)


def setup_blending():
    glEnable(GL_BLEND)
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)


def display_init(image_size, fullscreen=False, setup=False):
    if setup:
        pygame.init()
    width, height = image_size
    fullscreen_size = pygame.display.list_modes()[0]
    fullscreen_width, fullscreen_height = fullscreen_size
    # Calculate scaling factor and position for fullscreen mode
    scale = min(fullscreen_width / width, fullscreen_height / height)
    offset_x = int((fullscreen_width - width * scale) / 2)
    offset_y = int((fullscreen_height - height * scale) / 2)
    flags = OPENGL
    flags |= DOUBLEBUF
    if fullscreen:
        flags |= FULLSCREEN
        pygame.display.set_caption('Fullscreen Mode')
        pygame.display.set_mode((fullscreen_width, fullscreen_height), flags)
        glViewport(offset_x, offset_y, int(width * scale), int(height * scale))
    else:
        flags |= RESIZABLE

        if pygame.display.mode_ok(image_size, flags):
            pygame.display.set_mode(image_size, flags)
            pygame.display.set_caption('Windowed Mode')
        #print(pygame.display.list_modes())
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


import concurrent.futures


def display_and_run():
    midi_control.midi_control_stuff_main()
    print(platform.system())
    csv_source = select_csv_file()
    png_paths = get_image_names_from_csv(csv_source)
    print(len(png_paths[0]))
    aspect_ratio, width = get_aspect_ratio(png_paths[0][0])
    print(width)
    height = int(width * aspect_ratio)
    image_size = (width, height)
    main_folder = 6
    float_folder = 0
    fullscreen = False
    setup_mode = True
    start_frames = midi_control.total_frames
    print(start_frames)
    index, direction = calculators.calculate_index(start_frames)
    display_init(image_size, fullscreen, setup_mode)

    # Create a stop event for the MIDI processing thread
    stop_midi_event = threading.Event()

    with concurrent.futures.ThreadPoolExecutor() as executor:
        # Start the MIDI processing task, passing the stop event
        midi_future = executor.submit(midi_control.process_midi)

        # Your existing code to run the display
        run_display(index, direction, png_paths, main_folder, float_folder, image_size)

        # Set the stop event for the MIDI processing thread
        stop_midi_event.set()

        # Wait for the MIDI processing task to finish
        midi_future.result()


if __name__ == "__main__":
    display_and_run()
