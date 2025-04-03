import os
os.environ['PYOPENGL_ERROR_CHECKING'] = '0'

from concurrent.futures import ThreadPoolExecutor
import pygame
import platform

import calculators
from settings import (FULLSCREEN_MODE, PINGPONG, BUFFER_SIZE, FPS, CLOCK_MODE)
from index_calculator import update_index
from folder_selector import update_folder_selection, folder_dictionary
import renderer
from event_handler import event_check
from display_manager import DisplayState, get_aspect_ratio, display_init

def run_display(clock_source=CLOCK_MODE):
    state = DisplayState()

    # Typical setup
    csv_source, main_folder_path, float_folder_path = calculators.init_all(clock_source)
    print(platform.system(), "midi clock mode is:", clock_source)
    main_folder_count = len(main_folder_path[0])
    float_folder_count = len(float_folder_path[0])
    print(float(main_folder_count), float(float_folder_count))
    png_paths_len = len(main_folder_path) - 1

    aspect_ratio, width, height = get_aspect_ratio(main_folder_path[0][0])
    state.image_size = (width, height)
    print("Image size:", state.image_size)

    # A minimal ephemeral cache so we don't recalc the same index in the same cycle
    # or while prefetching. We'll store only 1 or 2 indexes at a time.
    pending_folders = {}

    def get_or_compute_folder_choice(idx, dir_):
        """
        - If (idx, dir_) is not in pending_folders, call update_folder_selection once.
        - Cache the result so repeated calls for the same (idx, dir_) won't re-randomize.
        - We'll remove old entries after we move past them, so we can get a new random
          assignment if we come back to 'idx' much later (no deterministic lock forever).
        """
        key = (idx, dir_)
        if key not in pending_folders:
            main_f, float_f = update_folder_selection(idx, dir_, float_folder_count, main_folder_count)
            pending_folders[key] = (main_f, float_f)
        return pending_folders[key]

    from image_loader import ImageLoader, ImageLoaderBuffer
    image_loader = ImageLoader()
    image_loader.set_paths(main_folder_path, float_folder_path)
    image_loader.set_png_paths_len(png_paths_len)

    pygame.init()
    invisible_cursor = pygame.cursors.Cursor((8, 8), (0, 0), (0,)*8, (0,)*8)
    pygame.mouse.set_cursor(invisible_cursor)
    fullscreen = FULLSCREEN_MODE
    display_init(fullscreen, state)
    vid_clock = pygame.time.Clock()

    # Initial index/direction
    index, direction = update_index(png_paths_len, PINGPONG)
    last_index = index

    # Decide folder choice for this index
    main_folder, float_folder = get_or_compute_folder_choice(index, direction)
    folder_dictionary['Main_and_Float_Folders'] = (main_folder, float_folder)

    # Preload images
    image_buffer = ImageLoaderBuffer(BUFFER_SIZE)
    with ThreadPoolExecutor(max_workers=4) as executor:
        for i in range(BUFFER_SIZE):
            buf_idx = (index + i) % png_paths_len
            future = executor.submit(image_loader.load_images, buf_idx, main_folder, float_folder)
            image_buffer.add_image_future(buf_idx, future, png_paths_len)

        # Create initial textures
        main_image, float_image = image_loader.load_images(index, main_folder, float_folder)
        texture_id1 = renderer.create_texture(main_image)
        texture_id2 = renderer.create_texture(float_image)

        while state.run_mode:
            events = pygame.event.get()
            fullscreen = event_check(events, fullscreen, state)

            if state.needs_update:
                display_init(fullscreen, state)
                state.needs_update = False

            # Update the index
            new_index, new_direction = update_index(png_paths_len, PINGPONG)
            if new_index != last_index:
                # We'll remove the old index from pending_folders so if we ever
                # come back to it in the future, we can get a fresh random assignment.
                old_key = (last_index, direction)
                if old_key in pending_folders:
                    del pending_folders[old_key]

                index, direction = new_index, new_direction
                last_index = index

                # Decide folder for the new index
                main_folder, float_folder = get_or_compute_folder_choice(index, direction)
                folder_dictionary['Main_and_Float_Folders'] = (main_folder, float_folder)

                # Attempt to load images from the buffer
                future = image_buffer.get_future_for_index(index)
                if future is not None:
                    main_image, float_image = future.result()
                else:
                    main_image, float_image = image_loader.load_images(index, main_folder, float_folder)

                # Update textures
                renderer.update_texture(texture_id1, main_image)
                renderer.update_texture(texture_id2, float_image)

                # Look ahead: pick folder choice for the next index too,
                # so the prefetch won't mismatch
                next_idx = (index + direction) % png_paths_len
                next_main, next_float = get_or_compute_folder_choice(next_idx, direction)

                # Preload next if not buffered
                if image_buffer.get_future_for_index(next_idx) is None:
                    new_future = executor.submit(image_loader.load_images, next_idx, next_main, next_float)
                    image_buffer.add_image_future(next_idx, new_future, png_paths_len)

            # Render
            renderer.overlay_images_two_pass_like_old(texture_id1, texture_id2, background_color=(9.0, 10.0, 10.0))
            pygame.display.flip()
            vid_clock.tick(FPS)

    pygame.quit()


if __name__ == "__main__":
    run_display(CLOCK_MODE)
