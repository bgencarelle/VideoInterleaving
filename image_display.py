import sys
import time
from concurrent.futures import ThreadPoolExecutor
import concurrent
from queue import SimpleQueue

from calculators import select_csv_file, get_image_names_from_csv
import midi_control
import cv2
import pygame.time
from OpenGL.GL import *
from OpenGL.GLU import *
from platform import system


if system() == 'Darwin':
    from Quartz import CGWindowListCopyWindowInfo, kCGWindowListOptionOnScreenOnly, kCGNullWindowID
elif system() == 'Linux':
    from Xlib import X, display

import platform
import pygame
from pygame.locals import *


FULLSCREEN_MODE = False
MTC_CLOCK = 0
MIDI_CLOCK = 1
FREE_CLOCK = 2
CLOCK_MODE = MTC_CLOCK
FPS = 30
run = True

BUFFER_SIZE = 10
PINGPONG = True

vid_clock = None
png_paths_len = 0
png_paths = 0
folder_count = 0
image_size = (800, 600)
text_display = False


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


def event_check(fullscreen):
    width, height = image_size
    aspect_ratio = width / height
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()
        if event.type == KEYDOWN:
            if event.key == K_q:  # Press 'p' key to toggle pause
                # print("bailing out")
                pygame.quit()
                sys.exit()
            if event.key == K_f:  # Press 'f' key to toggle fullscreen
                fullscreen = toggle_fullscreen(image_size, fullscreen)
            if event.key == K_i:  # Press 'i' key to jump around
                # index += 500
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

    return fullscreen


def get_aspect_ratio(image_path):
    # Load image with Pygame
    image = pygame.image.load(image_path)

    # Get image dimensions
    w, h = image.get_size()

    # Calculate aspect ratio
    aspect_ratio = h / w
    print(f'this is {w} wide and {h} tall, with an aspect ratio of {aspect_ratio}')
    return aspect_ratio, w, h,


def toggle_fullscreen(current_fullscreen_status):
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


def overlay_images_fast(texture_id_main, texture_id_float, main_rgba=None, float_rgba=None):
    width, height = image_size
    glClear(GL_COLOR_BUFFER_BIT)

    display_image(texture_id_float, width, height, rgba=float_rgba)
    display_image(texture_id_main, width, height, rgba=main_rgba)


def load_texture(texture_id, image):
    glBindTexture(GL_TEXTURE_2D, texture_id)
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, image.shape[1], image.shape[0], 0, GL_RGBA, GL_UNSIGNED_BYTE, image)


def load_images(index, main_folder, float_folder):
    main_image = read_image(png_paths[index][main_folder])
    float_image = read_image(png_paths[index][float_folder])
    return main_image, float_image


def set_index(index, direction):
    if CLOCK_MODE == MTC_CLOCK:
        index = midi_control.index
        print(midi_control.index)
        direction = midi_control.direction
    elif CLOCK_MODE == MIDI_CLOCK:
        index = midi_control.clock_index
        direction = midi_control.clock_index_direction
    else:
        if PINGPONG:
            # Reverse direction at boundaries
            if (index + direction) < 0 or index >= png_paths_len:
                direction = -direction
        index= max(0, min(index, png_paths_len))
        print(index)
    return index, direction


def print_index_diff_wrapper():
    storage = {'old_index': 0, 'prev_time': 0}

    def print_index_diff(index):
        current_time = time.time()
        if (current_time - storage['prev_time']) >= 1:
            print("index: ", index, "   index diff:  ", index - storage['old_index'])
            storage['old_index'] = index
            storage['prev_time'] = current_time
            fippy = vid_clock.get_fps()
            print(f'fps: {fippy}')


    return print_index_diff


print_index_diff_function = print_index_diff_wrapper()


def run_display_check():
    if CLOCK_MODE == MIDI_CLOCK or CLOCK_MODE == MTC_CLOCK:
        midi_control.midi_control_stuff_main()
        with concurrent.futures.ThreadPoolExecutor() as executor:
            # Start the MIDI processing task, passing the stop event
            midi_future = executor.submit(midi_control.process_midi)
            # Your existing code to run the display
            run_display()
            # Wait for the MIDI processing task to finish
            midi_future.result()
    else:
        run_display()

    return


def run_display():
    global run
    main_folder = 0
    float_folder = 3
    index, direction = set_index(0, 1)
    buffer_index = index
    buffer_direction = direction
    fullscreen = False

    index_changed = False
    buffer_synced = True
    min_buffered_images = int(BUFFER_SIZE - 1)

    main_image, float_image = load_images(index, main_folder, float_folder)
    texture_id1 = create_texture(main_image)
    texture_id2 = create_texture(float_image)

    def queue_image(buffer_idx, main_folder_q, float_folder_q, image_queue):
        buffer_idx = max(0, min(buffer_idx, png_paths_len - 1))
        image_future = executor.submit(load_images, buffer_idx, main_folder_q, float_folder_q)
        image_queue.put(image_future)

    with ThreadPoolExecutor(max_workers=2) as executor:
        image_queue = SimpleQueue()

        for _ in range(BUFFER_SIZE):
            buffer_index += buffer_direction
            queue_image(buffer_index, main_folder, float_folder, image_queue)

        while run:
            prev_index = index
            index, direction = set_index(index, direction)
            fullscreen = event_check(fullscreen)

            if prev_index != index:
                index_changed = True
                buffer_index = index
                buffer_direction = direction
                buffer_synced = False
            print_index_diff_function(index)
            if index % FPS == 0:
                float_folder = (float_folder + 1) % folder_count
            if index % (FPS * 5) == 0:
                main_folder = (main_folder + 1) % folder_count

            if index_changed and not image_queue.empty() and image_queue.qsize() >= min_buffered_images:
                main_image, float_image = image_queue.get().result()
                load_texture(texture_id1, main_image)
                load_texture(texture_id2, float_image)

                if buffer_index == index:
                    buffer_synced = True

                next_buffer_index = buffer_index + buffer_direction
                if next_buffer_index >= png_paths_len or next_buffer_index < 0:
                    buffer_direction *= -1
                    next_buffer_index = buffer_index + buffer_direction
                queue_image(next_buffer_index, main_folder, float_folder, image_queue)

                index_changed = False  # Reset index_changed to False after queueing the next image

            if buffer_synced:
                overlay_images_fast(texture_id1, texture_id2)

            pygame.display.flip()
            vid_clock.tick(FPS)

def setup_blending():
    glEnable(GL_BLEND)
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)


def display_init(fullscreen=False, setup=False):
    global vid_clock
    if setup:
        pygame.init()
        vid_clock = pygame.time.Clock()
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
            # print(pygame.display.list_modes())
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
    global png_paths_len, png_paths, folder_count, image_size
    print(platform.system())
    csv_source = select_csv_file()
    png_paths = get_image_names_from_csv(csv_source)
    folder_count = len(png_paths[0])
    png_paths_len = len(png_paths) - 1
    aspect_ratio, width, height = get_aspect_ratio(png_paths[0][0])

    height = int(width * aspect_ratio)
    image_size = (width, height)
    display_init(False, True)
    run_display_check()


if __name__ == "__main__":
    display_and_run()
