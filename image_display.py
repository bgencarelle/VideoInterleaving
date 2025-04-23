import datetime
import os
import time

os.environ['PYOPENGL_ERROR_CHECKING'] = '0'

from concurrent.futures import ThreadPoolExecutor
from collections import deque

import pygame
import platform
import calculators
from settings import (
    FULLSCREEN_MODE,
    PINGPONG,
    FPS,
    CLOCK_MODE,
    FIFO_LENGTH,
    FRAME_COUNTER_DISPLAY,
    SHOW_DELTA,
    TEST_MODE,
    HTTP_MONITOR,
)
from index_calculator import update_index
from folder_selector import update_folder_selection, folder_dictionary
import renderer
from event_handler import event_check
from display_manager import DisplayState, get_aspect_ratio, display_init

# Decide whether to start the web monitor
monitor = None
if TEST_MODE and HTTP_MONITOR:
    from lightweight_monitor import start_monitor
    monitor = start_monitor()

class RollingIndexCompensator:
    """
    Maintains a rolling average of the last N (index - displayed_index) differences
    and applies a partial offset to slowly push the difference toward zero.
    """
    def __init__(self, maxlen=10, correction_factor=0.5):
        self.diffs = deque(maxlen=maxlen)
        self.correction_factor = correction_factor

    def update(self, current_index, displayed_index):
        diff = current_index - displayed_index
        self.diffs.append(diff)

    def get_compensated_index(self, current_index):
        if not self.diffs:
            return current_index
        avg_diff = sum(self.diffs) / len(self.diffs)
        partial_offset = round(avg_diff * self.correction_factor)
        return current_index - partial_offset


def run_display(clock_source=CLOCK_MODE):
    # Create a DisplayState instance and persist fullscreen in the state.
    state = DisplayState()
    state.fullscreen = FULLSCREEN_MODE

    csv_source, main_folder_path, float_folder_path = calculators.init_all(clock_source)
    print(platform.system(), "midi clock mode is:", clock_source)
    main_folder_count = len(main_folder_path[0])
    float_folder_count = len(float_folder_path[0])
    png_paths_len = len(main_folder_path)
    aspect_ratio, width, height = get_aspect_ratio(main_folder_path[0][0])
    state.image_size = (width, height)
    print("current time is ", datetime.datetime.now())
    print("Image size:", state.image_size)

    from image_loader import ImageLoader, FIFOImageBuffer

    class FIFOImageBufferPatched(FIFOImageBuffer):
        def current_depth(self):
            with self.lock:
                return len(self.queue)

    image_loader = ImageLoader()
    image_loader.set_paths(main_folder_path, float_folder_path)
    image_loader.set_png_paths_len(png_paths_len)

    pygame.init()
    # Initialize the display using the persistent fullscreen flag in state.
    display_init(state)

    pygame.event.set_grab(False)
    pygame.mouse.set_visible(False)
    vid_clock = pygame.time.Clock()

    index, _ = update_index(png_paths_len, PINGPONG)
    last_index = index
    update_folder_selection(index, float_folder_count, main_folder_count)

    fifo_buffer = FIFOImageBufferPatched(max_size=FIFO_LENGTH)
    main_folder, float_folder = folder_dictionary['Main_and_Float_Folders']
    main_image, float_image = image_loader.load_images(index, main_folder, float_folder)
    fifo_buffer.update(index, (main_image, float_image))

    texture_id1 = renderer.create_texture(main_image)
    texture_id2 = renderer.create_texture(float_image)

    compensator = RollingIndexCompensator(maxlen=10, correction_factor=0.5)

    pygame.mouse.set_visible(False)

    def async_load_callback(fut, scheduled_index):
        try:
            fut_result = fut.result()
            fifo_buffer.update(scheduled_index, fut_result)
        except Exception as e:
            print("Error in async image load:", e)
            if monitor:  # Only record if the monitor is active
                monitor.record_load_error(scheduled_index, e)

    with ThreadPoolExecutor(max_workers=4) as executor:
        # Preload the initial next image
        if index == 0:
            next_index = index + 1
        elif index == png_paths_len - 1:
            next_index = index - 1
        else:
            next_index = index + 1 if index > last_index else index - 1

        future = executor.submit(image_loader.load_images, next_index, main_folder, float_folder)
        future.add_done_callback(lambda fut, s_idx=next_index: async_load_callback(fut, s_idx))

        frame_counter = 0
        start_time = pygame.time.get_ticks()

        def print_fps(arg_frame_counter, arg_start_time, target_fps):
            elapsed_time = pygame.time.get_ticks() - arg_start_time
            if elapsed_time >= 10000:
                print_actual_fps = arg_frame_counter / (elapsed_time / 1000.0)
                print("index:", index)
                print(f"[Display Rate] {print_actual_fps:.2f} frames per second")
                if print_actual_fps < target_fps - 2:
                    print(f"[Warning] Potential frame drop! Target: {target_fps}, Actual: {actual_fps:.2f}")
                return 0, pygame.time.get_ticks()
            return arg_frame_counter, arg_start_time

        while state.run_mode:
            successful_display = False
            events = pygame.event.get()
            # Update the persistent fullscreen state using the new event_check signature.
            state.fullscreen = event_check(events, state)

            if state.needs_update:
                display_init(state)
                pygame.mouse.set_visible(False)
                state.needs_update = False

            previous_index = index
            index, _ = update_index(png_paths_len, PINGPONG)
            if index != previous_index:
                last_index = previous_index
                update_folder_selection(index, float_folder_count, main_folder_count)
                main_folder, float_folder = folder_dictionary['Main_and_Float_Folders']

                compensated_index = compensator.get_compensated_index(index)
                result = fifo_buffer.get(compensated_index)
                if result is not None:
                    displayed_index, main_image, float_image = result
                    compensator.update(index, displayed_index)
                    if SHOW_DELTA:
                        difference = index - displayed_index
                        partial = compensated_index - index
                        print(f"[DEBUG] idx={index}, disp={displayed_index}, Δ={difference}, offset={partial}")
                    renderer.update_texture(texture_id1, main_image)
                    renderer.update_texture(texture_id2, float_image)
                    successful_display = True  # Mark as successful display
                else:
                    print(f"[MISS] FIFO miss for index {index} (Compensated={compensated_index}) at {displayed_index} at",datetime.datetime.now())

                # Schedule the next preload
                if index == 0:
                    next_index = index + 1
                elif index == png_paths_len - 1:
                    next_index = index - 1
                else:
                    next_index = index + 1 if index > previous_index else index - 1

                future = executor.submit(image_loader.load_images, next_index, main_folder, float_folder)
                future.add_done_callback(lambda fut, s_idx=next_index: async_load_callback(fut, s_idx))

            renderer.overlay_images_two_pass_like_old(texture_id1, texture_id2, background_color=(9.0, 10.0, 10.0))
            pygame.display.flip()

            vid_clock.tick(FPS)
            actual_fps = vid_clock.get_fps()
            frame_counter += 1

            if FRAME_COUNTER_DISPLAY:
                frame_counter, start_time = print_fps(frame_counter, start_time, FPS)

            # If monitor is active (TEST_MODE + HTTP_MONITOR), push stats to the web server.
            if monitor:
                monitor.update({
                    "index": index,
                    "displayed": displayed_index if 'displayed_index' in locals() else index,
                    "delta": index - displayed_index,
                    "fps": actual_fps,
                    "fifo_depth": fifo_buffer.current_depth(),
                    "successful_frame": successful_display,
                    "main_folder": main_folder,
                    "float_folder": float_folder,
                    "rand_mult": folder_dictionary['rand_mult'],
                    # NEW – give the monitor the totals once per tick
                    "main_folder_count": main_folder_count,
                    "float_folder_count": float_folder_count,
                })

    pygame.quit()


if __name__ == "__main__":
    run_display(CLOCK_MODE)
