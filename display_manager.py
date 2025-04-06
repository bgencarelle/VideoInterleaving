#display_manager.py
import pygame
import numpy as np
import renderer
from OpenGL.GL import glViewport
from settings import INITIAL_ROTATION, INITIAL_MIRROR, CONNECTED_TO_RCA_HDMI


class DisplayState:
    def __init__(self, image_size=(640, 480)):
        self.image_size = image_size
        self.rotation = INITIAL_ROTATION  # e.g., 270
        self.mirror = INITIAL_MIRROR  # 0 for normal, 1 for mirrored
        self.run_mode = True
        self.needs_update = False


def get_aspect_ratio(image_path):
    image = pygame.image.load(image_path)
    w, h = image.get_size()
    a_ratio = h / w
    print(f"This is {w} wide and {h} tall, with an aspect ratio of {a_ratio}")
    return a_ratio, w, h


def display_init(fullscreen, state):
    """
    Sets up the display and OpenGL parameters based on the current state.
    If fullscreen and CONNECTED_TO_RCA_HDMI == True, forces 640×480
    and does centered letterboxing to preserve aspect ratio.
    """
    global fs_scale, fs_offset_x, fs_offset_y, fs_fullscreen_width, fs_fullscreen_height

    w, h = state.image_size

    # If rotation is 90 or 270, swap the effective width & height
    if state.rotation % 180 == 90:
        effective_w = h
        effective_h = w
    else:
        effective_w = w
        effective_h = h

    if fullscreen:
        # Check if we're forcing 640×480 for RCA HDMI
        if CONNECTED_TO_RCA_HDMI:
            fs_fullscreen_width = 640
            fs_fullscreen_height = 480
            # Compute scale to fill 640x480 while preserving aspect ratio
            scale_x = fs_fullscreen_width / effective_w
            scale_y = fs_fullscreen_height / effective_h
            fs_scale = min(scale_x, scale_y)

            scaled_width = effective_w * fs_scale
            scaled_height = effective_h * fs_scale

            # Center it (letterbox/pillarbox)
            fs_offset_x = int((fs_fullscreen_width - scaled_width) / 2)
            fs_offset_y = int((fs_fullscreen_height - scaled_height) / 2)

            flags = pygame.OPENGL | pygame.DOUBLEBUF | pygame.FULLSCREEN
            pygame.display.set_caption("Fullscreen Mode (RCA HDMI forced 640x480)")
            pygame.display.set_mode((fs_fullscreen_width, fs_fullscreen_height), flags, vsync=1)

            # The viewport is the entire 640×480 “screen”
            glViewport(0, 0, fs_fullscreen_width, fs_fullscreen_height)
            viewport_width = fs_fullscreen_width
            viewport_height = fs_fullscreen_height

        else:
            # Standard "pick the largest available mode"
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
            pygame.display.set_caption("Fullscreen Mode")
            pygame.display.set_mode((fs_fullscreen_width, fs_fullscreen_height), flags, vsync=1)

            glViewport(0, 0, fs_fullscreen_width, fs_fullscreen_height)
            viewport_width = fs_fullscreen_width
            viewport_height = fs_fullscreen_height

    else:
        # Windowed mode
        win_width = 400
        win_height = int(400 * effective_h / effective_w)
        scale_x = win_width / effective_w
        scale_y = win_height / effective_h
        fs_scale = min(scale_x, scale_y)

        scaled_width = effective_w * fs_scale
        scaled_height = effective_h * fs_scale

        fs_offset_x = int((win_width - scaled_width) / 2)
        fs_offset_y = int((win_height - scaled_height) / 2)

        flags = pygame.OPENGL | pygame.DOUBLEBUF | pygame.RESIZABLE
        pygame.display.set_caption("Windowed Mode")
        pygame.display.set_mode((win_width, win_height), flags, vsync=1)

        glViewport(0, 0, win_width, win_height)
        viewport_width = win_width
        viewport_height = win_height

    # Build the default MVP matrix, which the renderer uses
    mvp = np.array([
        [2.0 / viewport_width, 0, 0, -1],
        [0, 2.0 / viewport_height, 0, -1],
        [0, 0, -1, 0],
        [0, 0, 0, 1]
    ], dtype=np.float32)

    # Update the renderer
    renderer.set_transform_parameters(
        fs_scale,
        fs_offset_x,
        fs_offset_y,
        state.image_size,
        state.rotation,
        state.mirror
    )
    renderer.setup_opengl(mvp)
