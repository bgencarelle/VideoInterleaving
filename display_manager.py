import pygame
import numpy as np
import renderer
from OpenGL.GL import glViewport  # Added import for glViewport
from settings import INITIAL_ROTATION, INITIAL_MIRROR, CONNECTED_TO_RCA_HDMI

class DisplayState:
    def __init__(self, image_size=(640,480)):
        self.image_size = image_size
        self.rotation = INITIAL_ROTATION  # e.g., 270
        self.mirror = INITIAL_MIRROR        # 0 for normal, 1 for mirrored
        self.run_mode = True
        self.needs_update = False

def get_aspect_ratio(image_path):
    image = pygame.image.load(image_path)
    w, h = image.get_size()
    a_ratio = h / w
    print(f'This is {w} wide and {h} tall, with an aspect ratio of {a_ratio}')
    return a_ratio, w, h

def display_init(fullscreen, state):
    """
    Sets up the display and OpenGL parameters based on the current state.
    """
    global fs_scale, fs_offset_x, fs_offset_y, fs_fullscreen_width, fs_fullscreen_height
    w, h = state.image_size
    # Swap dimensions if rotation is 90 or 270.
    if state.rotation % 180 == 90:
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
        # Inside the fullscreen block
        if CONNECTED_TO_RCA_HDMI:
            # Optional override for squashed HDMI displays
            # e.g., many RCA HDMI inputs report 1920x1080 but squish to 4:3
            fs_fullscreen_height = int(fs_fullscreen_width * 3 / 4)
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
        [2.0 / viewport_width, 0, 0, -1],
        [0, 2.0 / viewport_height, 0, -1],
        [0, 0, -1, 0],
        [0, 0, 0, 1]
    ], dtype=np.float32)
    renderer.set_transform_parameters(fs_scale, fs_offset_x, fs_offset_y, state.image_size, state.rotation, state.mirror)
    renderer.setup_opengl(mvp)
