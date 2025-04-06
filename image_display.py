import os

os.environ['PYOPENGL_ERROR_CHECKING'] = '0'

from concurrent.futures import ThreadPoolExecutor

import pygame
import platform
import calculators
from settings import (FULLSCREEN_MODE, PINGPONG, FPS, CLOCK_MODE)
from index_calculator import update_index
from folder_selector import update_folder_selection, folder_dictionary
import renderer
from event_handler import event_check
from display_manager import DisplayState, get_aspect_ratio, display_init

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

    # Create and configure an ImageLoader instance.
    from image_loader import ImageLoader, FIFOImageBuffer
    image_loader = ImageLoader()
    image_loader.set_paths(main_folder_path, float_folder_path)
    image_loader.set_png_paths_len(png_paths_len)

    pygame.init()
    pygame.mouse.set_visible(False)
    fullscreen = FULLSCREEN_MODE
    display_init(fullscreen, state)
    vid_clock = pygame.time.Clock()

    # In pingpong mode, update_index now returns (index, None).
    index, _ = update_index(png_paths_len, PINGPONG)
    last_index = index  # store previous index for direction inference
    update_folder_selection(index, None, float_folder_count, main_folder_count)

    # Initialize a FIFO buffer (replaces the old triple buffer).
    fifo_buffer = FIFOImageBuffer(max_size=5)

    # Load the initial image pair synchronously and add to the FIFO.
    main_folder, float_folder = folder_dictionary['Main_and_Float_Folders']
    main_image, float_image = image_loader.load_images(index, main_folder, float_folder)
    fifo_buffer.update(index, (main_image, float_image))

    # Create textures using the initial images.
    texture_id1 = renderer.create_texture(main_image)
    texture_id2 = renderer.create_texture(float_image)

    # Helper callback to update the FIFO when asynchronous loading completes.
    def async_load_callback(fut, scheduled_index):
        try:
            result = fut.result()  # (main_image, float_image)
            fifo_buffer.update(scheduled_index, result)
        except Exception as e:
            print("Error in async image load:", e)

    # Preload the next image pair asynchronously.
    with ThreadPoolExecutor(max_workers=4) as executor:
        # Compute next_index based on the current index.
        if index == 0:
            next_index = index + 1
        elif index == png_paths_len - 1:
            next_index = index - 1
        else:
            # When in the middle, assume ascending if current index > last_index; descending otherwise.
            next_index = index + 1 if index > last_index else index - 1

        future = executor.submit(image_loader.load_images, next_index, main_folder, float_folder)
        future.add_done_callback(lambda fut, s_idx=next_index: async_load_callback(fut, s_idx))

        # Frame rate measurement initialization.
        frame_counter = 0
        start_time = pygame.time.get_ticks()  # In milliseconds.

        # FPS printer function.
        def print_fps(frame_counter, start_time, target_fps):
            elapsed_time = pygame.time.get_ticks() - start_time
            if elapsed_time >= 10000:  # Every 30 seconds
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

            # Capture the previous index before updating.
            previous_index = index
            new_index, _ = update_index(png_paths_len, PINGPONG)
            if new_index != previous_index:
                index = new_index
                last_index = previous_index  # Save the old index for direction inference

                update_folder_selection(index, None, float_folder_count, main_folder_count)
                main_folder, float_folder = folder_dictionary['Main_and_Float_Folders']

                # Retrieve the latest image pair from the FIFO.
                result = fifo_buffer.get(index)
                if result is not None:
                    main_image, float_image = result
                    renderer.update_texture(texture_id1, main_image)
                    renderer.update_texture(texture_id2, float_image)
                else:
                    print(f"FIFO miss for index {index}")

                # Compute the next index based on current and previous index.
                if index == 0:
                    next_index = index + 1
                elif index == png_paths_len - 1:
                    next_index = index - 1
                else:
                    next_index = index + 1 if index > previous_index else index - 1

                future = executor.submit(image_loader.load_images, next_index, main_folder, float_folder)
                future.add_done_callback(lambda fut, s_idx=next_index: async_load_callback(fut, s_idx))

            # Render the images.
            renderer.overlay_images_two_pass_like_old(texture_id1, texture_id2, background_color=(9.0, 10.0, 10.0))
            pygame.display.flip()
            vid_clock.tick(FPS)

            # Update frame counter and print FPS.
            frame_counter += 1
            frame_counter, start_time = print_fps(frame_counter, start_time, FPS)

    pygame.quit()


if __name__ == "__main__":
    run_display(CLOCK_MODE)
