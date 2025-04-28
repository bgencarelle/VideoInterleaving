import os, platform, datetime, time
import cv2
import glfw
from concurrent.futures import ThreadPoolExecutor
from collections import deque

# Ensure PyOpenGL checks are off (not needed with ModernGL, but leaving for safety)
os.environ['PYOPENGL_ERROR_CHECKING'] = '0'

from settings import (FULLSCREEN_MODE, PINGPONG, FPS, FRAME_COUNTER_DISPLAY, SHOW_DELTA, TEST_MODE, HTTP_MONITOR,
                      CLOCK_MODE, FIFO_LENGTH, BACKGROUND_COLOR)
from index_calculator import update_index
from folder_selector import update_folder_selection, folder_dictionary
from display_manager import DisplayState, display_init
from event_handler import register_callbacks
import renderer

# Optionally start the lightweight monitor (for HTTP monitoring if enabled)
monitor = None
if TEST_MODE and HTTP_MONITOR:
    from lightweight_monitor import start_monitor
    monitor = start_monitor()

class RollingIndexCompensator:
    """
    Maintains a rolling average of the last N (index - displayed_index) differences,
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
    state = DisplayState()
    state.fullscreen = FULLSCREEN_MODE
    last_actual_fps = FPS

    # --- Initialize folders & image size ---
    import calculators
    _, main_folder_path, float_folder_path = calculators.init_all(clock_source)
    main_folder_count = len(main_folder_path[0])
    float_folder_count = len(float_folder_path[0])
    png_paths_len = len(main_folder_path)

    # determine initial image size
    first_image_path = main_folder_path[0][0]
    img = cv2.imread(first_image_path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise RuntimeError(f"Unable to load image: {first_image_path}")
    height, width = img.shape[:2]
    state.image_size = (width, height)

    print(platform.system(), "clock mode is:", clock_source)
    print("Image size:", state.image_size)

    # --- Image loader and buffer ---
    from image_loader import ImageLoader, FIFOImageBuffer
    class FIFOImageBufferPatched(FIFOImageBuffer):
        def current_depth(self):
            with self.lock:
                return len(self.queue)

    image_loader = ImageLoader()
    image_loader.set_paths(main_folder_path, float_folder_path)
    image_loader.set_png_paths_len(png_paths_len)

    # --- GLFW + ModernGL setup ---
    window = display_init(state)
    register_callbacks(window, state)

    # initial index & folder selection
    index, _ = update_index(png_paths_len, PINGPONG)
    last_index = index
    update_folder_selection(index, float_folder_count, main_folder_count)

    fifo_buffer = FIFOImageBufferPatched(max_size=FIFO_LENGTH)
    main_folder, float_folder = folder_dictionary['Main_and_Float_Folders']
    main_image, float_image = image_loader.load_images(index, main_folder, float_folder)
    fifo_buffer.update(index, (main_image, float_image))

    # create textures
    main_texture = renderer.create_texture(main_image)
    float_texture = renderer.create_texture(float_image)

    compensator = RollingIndexCompensator(maxlen=10, correction_factor=0.5)

    # preload callback
    def async_load_callback(fut, scheduled_index):
        try:
            result = fut.result()
            fifo_buffer.update(scheduled_index, result)
        except Exception as e:
            print("Error in async image load:", e)
            if monitor:
                monitor.record_load_error(scheduled_index, e)

    # start preloading next frame
    with ThreadPoolExecutor(max_workers=2) as executor:
        # schedule first preload
        if index == 0:
            next_index = 1
        elif index == png_paths_len - 1:
            next_index = index - 1
        else:
            next_index = index + 1 if index > last_index else index - 1
        future = executor.submit(image_loader.load_images, next_index, main_folder, float_folder)
        future.add_done_callback(lambda fut, s_idx=next_index: async_load_callback(fut, s_idx))

        # rolling-window timing
        frame_times = deque(maxlen=60)
        frame_start = time.perf_counter()

        while state.run_mode:
            successful_display = False
            glfw.poll_events()
            if not state.run_mode:
                break

            # handle display reinit
            if state.needs_update:
                display_init(state)
                state.needs_update = False

            # update index & images
            prev = index
            index, _ = update_index(png_paths_len, PINGPONG)

            if index != prev:
                last_index = prev
                update_folder_selection(index, float_folder_count, main_folder_count)
                main_folder, float_folder = folder_dictionary['Main_and_Float_Folders']
                compensated = compensator.get_compensated_index(index)
                result = fifo_buffer.get(compensated)

                if result is not None:
                    displayed_index, main_img, float_img = result
                    compensator.update(index, displayed_index)
                    if SHOW_DELTA:
                        d = index - displayed_index
                        off = compensated - index
                        print(f"[DEBUG] idx={index}, disp={displayed_index}, Δ={d}, offset={off}")
                    main_texture = renderer.update_texture(main_texture, main_img)
                    float_texture = renderer.update_texture(float_texture, float_img)
                    successful_display = True
                else:
                    print(f"[MISS] FIFO miss for index {index} at {datetime.datetime.now()}")

                # schedule next preload
                if index == 0:
                    next_index = 1
                elif index == png_paths_len - 1:
                    next_index = index - 1
                else:
                    next_index = index + 1 if index > prev else index - 1
                future = executor.submit(image_loader.load_images, next_index, main_folder, float_folder)
                future.add_done_callback(lambda fut, s_idx=next_index: async_load_callback(fut, s_idx))

            # draw
            renderer.overlay_images_two_pass_like_old(main_texture, float_texture, background_color=BACKGROUND_COLOR)
            glfw.swap_buffers(window)

            # timing
            now = time.perf_counter()
            dt = now - frame_start
            frame_times.append(dt)
            frame_start = now

            # throttle to FPS
            if FPS:
                target = 1.0 / FPS
                to_sleep = target - dt
                if to_sleep > 0:
                    time.sleep(to_sleep)

            # rolling FPS
            if len(frame_times) > 1:
                avg = sum(frame_times) / len(frame_times)
                last_actual_fps = 1.0 / avg if avg > 0 else 0.0
                if not HTTP_MONITOR:
                    print(f"[Display Rate] {last_actual_fps:.2f} FPS")
                    if last_actual_fps < FPS - 2:
                        print(f"[Warning] Frame drop! Target {FPS}, Got {last_actual_fps:.2f}")

            # update HTTP monitor
            if monitor:
                monitor.update({
                    "index": index,
                    "displayed": displayed_index if 'displayed_index' in locals() else index,
                    "delta": index - (displayed_index if 'displayed_index' in locals() else index),
                    "fps": last_actual_fps,
                    "fifo_depth": fifo_buffer.current_depth(),
                    "successful_frame": successful_display,
                    "main_folder": main_folder,
                    "float_folder": float_folder,
                    "rand_mult": folder_dictionary.get('rand_mult'),
                    "main_folder_count": main_folder_count,
                    "float_folder_count": float_folder_count
                })

            # exit on window close
            if glfw.window_should_close(window):
                state.run_mode = False

        glfw.terminate()
