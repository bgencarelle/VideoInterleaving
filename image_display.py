import time
from concurrent.futures import ThreadPoolExecutor
from collections import deque  # NEW: using deque for the ring-buffer
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
import random
import index_client

# Mode Constants
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

BUFFER_SIZE = FPS // 4  # e.g., 15 if FPS==60
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
    """Create a new texture and allocate memory using glTexImage2D."""
    texture_id = glGenTextures(1)
    glBindTexture(GL_TEXTURE_2D, texture_id)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
    w, h = image.shape[1], image.shape[0]
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, w, h, 0, GL_RGBA, GL_UNSIGNED_BYTE, image)
    texture_dimensions[texture_id] = (w, h)
    return texture_id


def update_texture(texture_id, new_image):
    """Update an existing texture.
       If the incoming image dimensions differ from the texture, reallocate the texture.
       Alternatively, you could resize the new_image to match the texture dimensions."""
    glBindTexture(GL_TEXTURE_2D, texture_id)
    w, h = new_image.shape[1], new_image.shape[0]
    expected = texture_dimensions.get(texture_id, (None, None))
    if expected != (w, h):
        # Option 1: Reallocate texture to new dimensions:
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, w, h, 0, GL_RGBA, GL_UNSIGNED_BYTE, new_image)
        texture_dimensions[texture_id] = (w, h)
        # Option 2 (alternative): Resize new_image to expected size:
        # new_image = cv2.resize(new_image, (expected[0], expected[1]))
        # glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, expected[0], expected[1],
        #                 GL_RGBA, GL_UNSIGNED_BYTE, new_image)
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


# Use the existing load_images() function to load the two images for a given index.
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
        index, direction = control_data_dictionary['Index_and_Direction']
    update_control_data(index, direction)
    return index, direction


def update_control_data(index, direction):
    rand_mult = random.randint(1, 9)
    rand_start = 8 * (IPS - (rand_mult * rand_mult // 2))
    main_folder, float_folder = folder_dictionary['Main_and_Float_Folders']
    if clock_mode == FREE_CLOCK:
        if index <= rand_start * direction or (100 * rand_start < index < 140 * rand_start):
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
        float_folder = mod_value % float_folder_count
        main_folder = (note % 12) % main_folder_count
    folder_dictionary['Main_and_Float_Folders'] = main_folder, float_folder


def print_index_diff_wrapper():
    storage = {'old_index': 0, 'prev_time': 0}

    def print_index_diff(index):
        current_time = time.time()
        if (current_time - storage['prev_time']) >= 1 and storage['old_index'] != index:
            # Reduced logging here
            print("index:", index, "   diff:", index - storage['old_index'])
            storage['old_index'] = index
            storage['prev_time'] = current_time
            fippy = vid_clock.get_fps()
            print(f'fps: {fippy}')
            print(f'midi_control.bpm: {midi_control.bpm}')

    return print_index_diff


print_index_diff_function = print_index_diff_wrapper()


# NEW: Ring-buffer class for image futures
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
    global vid_clock
    if midi_mode:
        midi_control.midi_control_stuff_main()
    elif clock_mode == CLIENT_MODE:
        threading.Thread(target=index_client.start_client, daemon=True).start()
    pygame.init()
    pygame.mouse.set_visible(False)
    pygame.time.set_timer(UPDATE_INDEX_EVENT, int(1000 / IPS))
    display_init(True)
    vid_clock = pygame.time.Clock()
    run_display()
    return


def run_display():
    global run_mode
    index, direction = control_data_dictionary['Index_and_Direction']
    # Initialize buffer index using update_index_and_folders
    buffer_index, buffer_direction = update_index_and_folders(0, 1)
    fullscreen = True

    # NEW: create a ring-buffer for image futures
    image_buffer = ImageLoaderBuffer(BUFFER_SIZE)

    with ThreadPoolExecutor(max_workers=8) as executor:
        # Preload the ring-buffer with initial image futures
        main_folder, float_folder = folder_dictionary['Main_and_Float_Folders']
        initial_index = index
        for i in range(BUFFER_SIZE):
            buf_idx = (initial_index + i) % png_paths_len
            future = executor.submit(load_images, buf_idx, main_folder, float_folder)
            image_buffer.add_image_future(buf_idx, future)
            # Reduced logging: (print statements can be commented out)
            # print(f"[Preload] Queued future for index {buf_idx}")

        # Synchronously load the current images for initial textures
        main_image, float_image = load_images(index, main_folder, float_folder)
        texture_id1 = create_texture(main_image)
        texture_id2 = create_texture(float_image)

        while run_mode:
            try:
                prev_index = index
                fullscreen = event_check(fullscreen)
                update_index_and_folders(index, direction)
                index, direction = control_data_dictionary['Index_and_Direction']
                main_folder, float_folder = folder_dictionary['Main_and_Float_Folders']

                future = image_buffer.get_future_for_index(index)
                if future is not None:
                    # When a future is ready, update textures using glTexSubImage2D (or reallocate if needed)
                    main_image, float_image = future.result()
                    update_texture(texture_id1, main_image)
                    update_texture(texture_id2, float_image)
                    # Queue next image for this slot
                    next_index = (index + direction) % png_paths_len
                    new_future = executor.submit(load_images, next_index, main_folder, float_folder)
                    image_buffer.add_image_future(next_index, new_future)
                else:
                    # If no future is found, queue one immediately (fallback)
                    next_index = (index + direction) % png_paths_len
                    new_future = executor.submit(load_images, next_index, main_folder, float_folder)
                    image_buffer.add_image_future(next_index, new_future)
                    # Optionally log a brief message or skip
                overlay_images_fast(texture_id1, texture_id2, index)
                pygame.display.flip()
                vid_clock.tick(FPS)
            except Exception as e:
                print(f"An error occurred: {e}")


def display_init(fullscreen=True):
    global fs_scale, fs_offset_x, fs_offset_y, fs_fullscreen_width, fs_fullscreen_height
    w, h = image_size  # native image dimensions

    if fullscreen:
        modes = pygame.display.list_modes()
        if not modes:
            raise RuntimeError("No display modes available!")
        fs_fullscreen_width, fs_fullscreen_height = modes[0]

        # Compute scale: choose the scale that makes the image as large as possible
        # while keeping its aspect ratio intact.
        scale_x = fs_fullscreen_width / w
        scale_y = fs_fullscreen_height / h
        fs_scale = min(scale_x, scale_y)

        # Compute offsets to center the image:
        scaled_width = w * fs_scale
        scaled_height = h * fs_scale
        fs_offset_x = int((fs_fullscreen_width - scaled_width) / 2)
        fs_offset_y = int((fs_fullscreen_height - scaled_height) / 2)

        flags = OPENGL | DOUBLEBUF | FULLSCREEN
        pygame.display.set_caption('Fullscreen Mode')
        pygame.display.set_mode((fs_fullscreen_width, fs_fullscreen_height), flags, vsync=1)
        glViewport(0, 0, fs_fullscreen_width, fs_fullscreen_height)

        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        # Set up orthographic projection with origin at bottom left.
        gluOrtho2D(0, fs_fullscreen_width, 0, fs_fullscreen_height)
    else:
        # For windowed mode, similar logic applies but with fixed window dimensions.
        win_width = 400
        win_height = int(400 * h / w)
        scale_x = win_width / w
        scale_y = win_height / h
        win_scale = min(scale_x, scale_y)
        win_offset_x = int((win_width - (w * win_scale)) / 2)
        win_offset_y = int((win_height - (h * win_scale)) / 2)

        flags = OPENGL | DOUBLEBUF | RESIZABLE
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
