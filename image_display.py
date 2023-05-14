import math
import sys
import time
from concurrent.futures import ThreadPoolExecutor
import concurrent
from queue import SimpleQueue

import calculators
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
CLOCK_MODE = 0

MIDI_MODE = True if CLOCK_MODE < FREE_CLOCK else False

FPS = 30
run_mode = True

BUFFER_SIZE = 15
PINGPONG = True

vid_clock = None
pause_mode = False
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
    global image_size, run_mode, pause_mode
    width, height = image_size
    aspect_ratio = width / height
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            run_mode = False
            sys.exit()
        if event.type == KEYDOWN:
            if event.key == K_q:  # Press 'p' key to toggle pause
                # print("bailing out")
                run_mode = False
                sys.exit()
            if event.key == K_f:  # Press 'f' key to toggle fullscreen
                fullscreen = toggle_fullscreen(fullscreen)
            if event.key == K_p:  # Press 'p' key to toggle fullscreen
                pause_mode = not pause_mode
                print(pause_mode)
            if event.key == K_i:  # Press 'i' key to jump around
                # index += 500
                print("jump around")
            if event.key == K_t:  # Press ' t ' key to jump around
                text_mode = not text_mode
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

    # Calculate aspect ratio
    aspect_ratio = h / w
    print(f'this is {w} wide and {h} tall, with an aspect ratio of {aspect_ratio}')
    return aspect_ratio, w, h,


def toggle_fullscreen(current_fullscreen_status):
    if is_window_maximized() and not current_fullscreen_status:
        print("dooo")
        return current_fullscreen_status
    else:
        new_fullscreen_status = not current_fullscreen_status
        display_init(new_fullscreen_status)
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


def set_rgba_relative(index=0):
    scale = 30.00
    index_scale = (index/png_paths_len)
    pi_scale = math.pi/2.000

    hapi_scale = math.pi * index_scale
    main_alpha = 1
    if index > 200:
        float_alpha = (math.sin(hapi_scale))
    else:
        float_alpha = 0
    main_rgba = (1, 1, 1, main_alpha)
    float_rgba = (1, 1, 1, float_alpha)
    return main_rgba, float_rgba


def overlay_images_fast(texture_id_main, texture_id_float, index=0, background_color=(32, 30, 32)):
    width, height = image_size
    main_rgba, float_rgba = set_rgba_relative(index)
    glClearColor(background_color[0]/255, background_color[1]/255, background_color[2]/255, 1.0)
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


def get_index(index, direction):
    if MIDI_MODE:
        midi_control.process_midi(MIDI_CLOCK)

    if CLOCK_MODE == MTC_CLOCK:
        index = midi_control.index
        direction = midi_control.direction
    elif CLOCK_MODE == MIDI_CLOCK:
        index = midi_control.clock_index
        direction = midi_control.clock_index_direction
    else:
        if PINGPONG:
            # Reverse direction at boundaries
            if (index + direction) < 0 or index + direction >= png_paths_len:
                direction *= -1
            index += direction
        else:
            index = (index + 1) % png_paths_len
    if pause_mode:
        direction_out = 0
    else:
        direction_out = direction
    return index, direction_out


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
    pygame.init()
    display_init(False)
    vid_clock = pygame.time.Clock()
    if MIDI_MODE:
        midi_control.midi_control_stuff_main()
    run_display()
    return


def run_display():
    global run_mode
    main_folder = 9
    float_folder = 12
    buffer_index, buffer_direction = get_index(0, 1)
    fullscreen = False

    index_changed = False

    def queue_image(buffer_idx, main_folder_q, float_folder_q, image_queue):
        buffer_idx = max(0, min(buffer_idx, png_paths_len - 1))
        image_future = executor.submit(load_images, buffer_idx, main_folder_q, float_folder_q)
        image_queue.put(image_future)

    with ThreadPoolExecutor(max_workers=2) as executor:
        index, direction = get_index(0, 1)
        image_queue = SimpleQueue()

        main_image, float_image = load_images(index, main_folder, float_folder)
        texture_id1 = create_texture(main_image)
        texture_id2 = create_texture(float_image)

        for _ in range(BUFFER_SIZE):
            # index, direction = get_index(index, direction)
            buffer_index += direction
            if buffer_index >= png_paths_len or buffer_index < 0:
                buffer_index += direction * -1
            queue_image(buffer_index, main_folder, float_folder, image_queue)

        last_skipped_index = -1

        while run_mode:
            try:
                float_folder, main_folder = time_stamp_control(float_folder, index, main_folder)
                midi_control.process_midi(CLOCK_MODE)
                prev_index = index
                fullscreen = event_check(fullscreen)
                index, direction = get_index(index, direction)
                print_index_diff_function(index)
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


def time_stamp_control(float_folder, index, main_folder):
    time_stamp = midi_control.total_frames / midi_control.frame_rate
    if time_stamp > 5 and (index % (FPS * 2)) == 0:
        main_folder = (1 + main_folder) % 9  # chromatic
    if time_stamp > 30 and (index % FPS * 3) == 0:
        float_folder = (float_folder + 1) % folder_count
        if float_folder >= 13 or float_folder < 10:
            float_folder = 10
    if time_stamp < 30:
        main_folder = 9
        float_folder = 12
    return float_folder, main_folder


def setup_blending():
    glEnable(GL_BLEND)
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)


def display_init(fullscreen=False):
    aspect_ratio, w, h = get_aspect_ratio(png_paths[0][0])
    fullscreen_size = pygame.display.list_modes()[0]
    fullscreen_width, fullscreen_height = fullscreen_size
    # Calculate scaling factor and position for fullscreen mode
    scale = min(fullscreen_width / w, fullscreen_height / h)
    offset_x = int((fullscreen_width - w * scale) / 2)
    offset_y = int((fullscreen_height - h * scale) / 2)
    flags = OPENGL
    flags |= DOUBLEBUF
    if fullscreen:
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
    setup_blending()
    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()
    if fullscreen:
        gluOrtho2D(0, int(w * scale), 0, int(h * scale))  # Use scaled dimensions in fullscreen mode
    else:
        gluOrtho2D(0, w, 0, h)  # Use original dimensions in windowed mode
    glMatrixMode(GL_MODELVIEW)
    glLoadIdentity()


def display_and_run():
    global png_paths_len, png_paths, folder_count, image_size
    csv_source, png_paths = calculators.init_all()
    print(platform.system(), "midi clock mode is:", CLOCK_MODE)
    # csv_source = select_csv_file()
    # png_paths = get_image_names_from_csv(csv_source)
    folder_count = len(png_paths[2220])
    print(f'folder_count: {folder_count}')
    png_paths_len = len(png_paths)-1
    aspect_ratio, width, height = get_aspect_ratio(png_paths[0][0])
    image_size = (width, height)
    print(image_size)
    run_display_setup()


if __name__ == "__main__":
    display_and_run()
