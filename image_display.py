import os, platform, datetime, time
import cv2
import glfw
from concurrent.futures import ThreadPoolExecutor
from collections import deque

# Ensure PyOpenGL checks are off (not needed with ModernGL, but leaving for safety)
os.environ['PYOPENGL_ERROR_CHECKING'] = '0'

from settings import (FULLSCREEN_MODE, PINGPONG, FPS, FRAME_COUNTER_DISPLAY, SHOW_DELTA, TEST_MODE, HTTP_MONITOR,
                      CLOCK_MODE, BACKGROUND_COLOR)
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
    state.fullscreen = FULLSCREEN_MODE  # initial fullscreen setting from config
    last_actual_fps = FPS

    # Initialize image paths and get initial image dimensions
    import calculators  # (assuming calculators.init_all is available)
    _, main_folder_path, float_folder_path = calculators.init_all(clock_source)
    main_folder_count = len(main_folder_path[0])
    float_folder_count = len(float_folder_path[0])
    png_paths_len = len(main_folder_path)
    # Use OpenCV to get aspect ratio and dimensions of the first image
    first_image_path = main_folder_path[0][0]
    img = cv2.imread(first_image_path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise RuntimeError(f"Unable to load image for aspect ratio: {first_image_path}")
    height, width = img.shape[0], img.shape[1]
    state.image_size = (width, height)
    print(platform.system(), "clock mode is:", clock_source)
    print("Image size:", state.image_size)

    # Set up image loader and FIFO buffer
    from image_loader import ImageLoader, FIFOImageBuffer
    class FIFOImageBufferPatched(FIFOImageBuffer):
        def current_depth(self):
            with self.lock:
                return len(self.queue)
    image_loader = ImageLoader()
    image_loader.set_paths(main_folder_path, float_folder_path)
    image_loader.set_png_paths_len(png_paths_len)

    # Initialize GLFW window and OpenGL via display_manager
    window = display_init(state)
    # Register input and window event callbacks
    register_callbacks(window, state)

    index, _ = update_index(png_paths_len, PINGPONG)
    last_index = index
    update_folder_selection(index, float_folder_count, main_folder_count)

    fifo_buffer = FIFOImageBufferPatched(max_size=30)  # FIFO_LENGTH from settings
    main_folder, float_folder = folder_dictionary['Main_and_Float_Folders']
    main_image, float_image = image_loader.load_images(index, main_folder, float_folder)
    fifo_buffer.update(index, (main_image, float_image))

    # Create initial textures for main and float images
    main_texture = renderer.create_texture(main_image)
    float_texture = renderer.create_texture(float_image)

    compensator = RollingIndexCompensator(maxlen=10, correction_factor=0.5)

    # Prepare asynchronous loader for preloading images
    def async_load_callback(fut, scheduled_index):
        try:
            result = fut.result()
            fifo_buffer.update(scheduled_index, result)
        except Exception as e:
            print("Error in async image load:", e)
            if monitor:
                monitor.record_load_error(scheduled_index, e)

    # Start background preload of the next image
    with ThreadPoolExecutor(max_workers=4) as executor:
        if index == 0:
            next_index = 1
        elif index == png_paths_len - 1:
            next_index = index - 1
        else:
            next_index = index + 1 if index > last_index else index - 1
        future = executor.submit(image_loader.load_images, next_index, main_folder, float_folder)
        future.add_done_callback(lambda fut, s_idx=next_index: async_load_callback(fut, s_idx))

        frame_counter = 0
        start_time_sec = time.perf_counter()
        # Main display loop
        while state.run_mode:
            successful_display = False
            glfw.poll_events()  # process input events (updates state via callbacks)
            if not state.run_mode:
                break  # exit loop if signaled to quit

            # Reinitialize display if needed (e.g., toggled fullscreen or rotation changed)
            if state.needs_update:
                display_init(state)
                state.needs_update = False

            previous_index = index
            index, _ = update_index(png_paths_len, PINGPONG)
            if index != previous_index:
                last_index = previous_index
                update_folder_selection(index, float_folder_count, main_folder_count)
                main_folder, float_folder = folder_dictionary['Main_and_Float_Folders']
                # Use RollingIndexCompensator to get adjusted index from FIFO
                compensated_index = compensator.get_compensated_index(index)
                result = fifo_buffer.get(compensated_index)
                if result is not None:
                    displayed_index, main_image, float_image = result
                    compensator.update(index, displayed_index)
                    if SHOW_DELTA:
                        diff = index - displayed_index
                        offset = compensated_index - index
                        print(f"[DEBUG] idx={index}, disp={displayed_index}, Δ={diff}, offset={offset}")
                    # Update textures with the new images (resize if necessary)
                    main_texture = renderer.update_texture(main_texture, main_image)
                    float_texture = renderer.update_texture(float_texture, float_image)
                    successful_display = True
                else:
                    print(f"[MISS] FIFO miss for index {index} (Compensated={compensated_index}) at {datetime.datetime.now()}")
                # Preload the next image in background
                if index == 0:
                    next_index = 1
                elif index == png_paths_len - 1:
                    next_index = index - 1
                else:
                    next_index = index + 1 if index > previous_index else index - 1
                future = executor.submit(image_loader.load_images, next_index, main_folder, float_folder)
                future.add_done_callback(lambda fut, s_idx=next_index: async_load_callback(fut, s_idx))

            # Render the current main and float images
            renderer.overlay_images_two_pass_like_old(main_texture, float_texture, background_color=BACKGROUND_COLOR)
            glfw.swap_buffers(window)  # display the rendered frame

            # Throttle frame rate to target FPS
            frame_time = time.perf_counter() - start_time_sec
            # (Compute time since last frame separately to handle sleep accurately)
            # Sleep if the frame was produced faster than the target frame duration
            time_since_last = frame_time - (frame_counter / FPS if FPS else frame_time)
            if FPS and time_since_last < 1.0 / FPS:
                time.sleep(1.0 / FPS - time_since_last if 1.0 / FPS > time_since_last else 0)
            frame_counter += 1

            # Always calculate FPS for web updates every second (or 10s)
            elapsed = time.perf_counter() - start_time_sec
            if elapsed >= 1.0:  # You lowered from 10.0 to 1.0 for quicker updates
                actual_fps = frame_counter / elapsed
                last_actual_fps = actual_fps  # This feeds the web monitor

                if not HTTP_MONITOR:
                    print("index:", index)
                    print(f"[Display Rate] {actual_fps:.2f} frames per second")
                    if actual_fps < FPS - 2:
                        print(f"[Warning] Potential frame drop! Target: {FPS}, Actual: {actual_fps:.2f}")

                # Reset counter and timer
                frame_counter = 0
                start_time_sec = time.perf_counter()
            # Update web monitor if active
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
                    "rand_mult": folder_dictionary.get('rand_mult', None),
                    "main_folder_count": main_folder_count,
                    "float_folder_count": float_folder_count
                })

            # Break loop if window is closed by the user
            if glfw.window_should_close(window):
                state.run_mode = False

    # Cleanup on exit
    glfw.terminate()
