import os

os.environ['PYOPENGL_ERROR_CHECKING'] = '0'

from concurrent.futures import ThreadPoolExecutor
from collections import deque

import pygame
import platform
import calculators
from settings import (FULLSCREEN_MODE, PINGPONG, FPS, CLOCK_MODE, FIFO_LENGTH, FRAME_COUNTER_DISPLAY, SHOW_DELTA)
from index_calculator import update_index
from folder_selector import update_folder_selection, folder_dictionary
import renderer
from event_handler import event_check
from display_manager import DisplayState, get_aspect_ratio, display_init


class RollingIndexCompensator:
    """
    Maintains a rolling average of the last N (index - displayed_index) differences
    and applies a partial offset to slowly push the difference toward zero.
    """

    def __init__(self, maxlen=10, correction_factor=0.5):
        """
        :param maxlen: How many recent diffs to track
        :param correction_factor: Fraction of the average diff to apply each frame
        """
        self.diffs = deque(maxlen=maxlen)
        self.correction_factor = correction_factor

    def update(self, current_index, displayed_index):
        """
        Called after we successfully fetch and display an image.
        """
        diff = current_index - displayed_index
        self.diffs.append(diff)

    def get_compensated_index(self, current_index):
        """
        Returns an integer index after partial compensation
        based on the rolling average difference.
        """
        if not self.diffs:
            return current_index  # No data -> no offset

        avg_diff = sum(self.diffs) / len(self.diffs)
        # Apply only a fraction of the difference
        partial_offset = round(avg_diff * self.correction_factor)

        # Subtract partial_offset from current_index
        compensated_index = current_index - partial_offset
        return compensated_index


def run_display(clock_source=CLOCK_MODE):
    state = DisplayState()

    csv_source, main_folder_path, float_folder_path = calculators.init_all(clock_source)
    print(platform.system(), "midi clock mode is:", clock_source)
    main_folder_count = len(main_folder_path[0])
    float_folder_count = len(float_folder_path[0])
    png_paths_len = len(main_folder_path)

    aspect_ratio, width, height = get_aspect_ratio(main_folder_path[0][0])
    state.image_size = (width, height)
    print("Image size:", state.image_size)

    from image_loader import ImageLoader, FIFOImageBuffer
    image_loader = ImageLoader()
    image_loader.set_paths(main_folder_path, float_folder_path)
    image_loader.set_png_paths_len(png_paths_len)

    pygame.init()
    pygame.mouse.set_visible(False)
    fullscreen = FULLSCREEN_MODE
    display_init(fullscreen, state)
    vid_clock = pygame.time.Clock()

    index, _ = update_index(png_paths_len, PINGPONG)
    last_index = index
    update_folder_selection(index, None, float_folder_count, main_folder_count)

    fifo_buffer = FIFOImageBuffer(max_size=FIFO_LENGTH)
    main_folder, float_folder = folder_dictionary['Main_and_Float_Folders']
    main_image, float_image = image_loader.load_images(index, main_folder, float_folder)
    fifo_buffer.update(index, (main_image, float_image))

    texture_id1 = renderer.create_texture(main_image)
    texture_id2 = renderer.create_texture(float_image)

    # Rolling compensation manager
    compensator = RollingIndexCompensator(
        maxlen=10,         # track last 10 differences
        correction_factor=0.5  # apply 50% of average difference each time
    )

    def async_load_callback(fut, scheduled_index):
        try:
            result = fut.result()
            fifo_buffer.update(scheduled_index, result)
        except Exception as e:
            print("Error in async image load:", e)

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

        def print_fps(frame_counter, start_time, target_fps):
            elapsed_time = pygame.time.get_ticks() - start_time
            if elapsed_time >= 10000:
                actual_fps = frame_counter / (elapsed_time / 1000.0)
                print("index:", index)
                print(f"[Display Rate] {actual_fps:.2f} frames per second")
                if actual_fps < target_fps - 2:
                    print(f"[Warning] Potential frame drop! Target: {target_fps}, Actual: {actual_fps:.2f}")
                return 0, pygame.time.get_ticks()
            return frame_counter, start_time

        while state.run_mode:
            events = pygame.event.get()
            fullscreen = event_check(events, fullscreen, state)

            if state.needs_update:
                display_init(fullscreen, state)
                state.needs_update = False

            previous_index = index
            index, _ = update_index(png_paths_len, PINGPONG)
            if index != previous_index:
                last_index = previous_index
                update_folder_selection(index, None, float_folder_count, main_folder_count)
                main_folder, float_folder = folder_dictionary['Main_and_Float_Folders']

                # 1) Compute a compensated index (based on rolling average)
                compensated_index = compensator.get_compensated_index(index)

                # 2) Fetch from FIFO using the compensated index
                result = fifo_buffer.get(compensated_index)
                if result is not None:
                    displayed_index, main_image, float_image = result
                    # 3) Update the rolling stats
                    compensator.update(index, displayed_index)

                    if SHOW_DELTA:
                        difference = index - displayed_index
                        partial = compensated_index - index
                        print(f"[DEBUG] idx={index}, disp={displayed_index}, Δ={difference}, offset={partial}")

                    renderer.update_texture(texture_id1, main_image)
                    renderer.update_texture(texture_id2, float_image)
                else:
                    print(f"[MISS] FIFO miss for index {index} (Compensated={compensated_index})")

                # 4) Preload next
                if index == 0:
                    next_index = index + 1
                elif index == png_paths_len - 1:
                    next_index = index - 1
                else:
                    next_index = index + 1 if index > previous_index else index - 1

                future = executor.submit(image_loader.load_images, next_index, main_folder, float_folder)
                future.add_done_callback(lambda fut, s_idx=next_index: async_load_callback(fut, s_idx))

            # Render
            renderer.overlay_images_two_pass_like_old(texture_id1, texture_id2, background_color=(9.0, 10.0, 10.0))
            pygame.display.flip()
            vid_clock.tick(FPS)

            frame_counter += 1
            if FRAME_COUNTER_DISPLAY:
                frame_counter, start_time = print_fps(frame_counter, start_time, FPS)

    pygame.quit()


if __name__ == "__main__":
    run_display(CLOCK_MODE)
