# image_display.py
import os
import platform
import datetime
import time
from concurrent.futures import ThreadPoolExecutor
from collections import deque

import cv2
import numpy as np

import settings
from shared_state import exchange

# --- ADAPTIVE OPTIMIZATION ---
cpu_count = os.cpu_count() or 1
if settings.SERVER_MODE or cpu_count <= 2:
    cv2.setNumThreads(0)
else:
    cv2.setNumThreads(1)
# -----------------------------

os.environ['PYOPENGL_ERROR_CHECKING'] = '0'

if not settings.SERVER_MODE:
    import glfw

from settings import (
    FULLSCREEN_MODE, PINGPONG, FPS, FRAME_COUNTER_DISPLAY, SHOW_DELTA,
    TEST_MODE, HTTP_MONITOR, CLOCK_MODE, FIFO_LENGTH, BACKGROUND_COLOR,
)
from index_calculator import update_index
from folder_selector import update_folder_selection, folder_dictionary
from display_manager import DisplayState, display_init
from event_handler import register_callbacks
import renderer  # We use this for both GPU and CPU paths now
from turbojpeg import TurboJPEG

jpeg = TurboJPEG()

monitor = None
if TEST_MODE and HTTP_MONITOR:
    from lightweight_monitor import start_monitor

    monitor = start_monitor()


class RollingIndexCompensator:
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


# --- REMOVED: cpu_composite_frame (Moved to renderer.py) ---

def run_display(clock_source=CLOCK_MODE):
    state = DisplayState()
    state.fullscreen = FULLSCREEN_MODE
    last_actual_fps = FPS

    # --- Server capture timing ---
    capture_rate = getattr(settings, "SERVER_CAPTURE_RATE", FPS or 10)
    capture_interval = 1.0 / capture_rate
    last_server_capture = time.time()
    last_captured_index = None

    # --- Initialize folders & image size ---
    import calculators
    _, main_folder_path, float_folder_path = calculators.init_all(clock_source)
    main_folder_count = len(main_folder_path[0])
    float_folder_count = len(float_folder_path[0])
    png_paths_len = len(main_folder_path)

    first_image_path = main_folder_path[0][0]
    img0 = cv2.imread(first_image_path, cv2.IMREAD_UNCHANGED)
    if img0 is None:
        raise RuntimeError(f"Unable to load image: {first_image_path}")
    height, width = img0.shape[:2]
    state.image_size = (width, height)

    print(platform.system(), "clock mode is:", clock_source)
    print("Image size:", state.image_size)

    # --- Image loader and FIFO ---
    from image_loader import ImageLoader, FIFOImageBuffer

    class FIFOImageBufferPatched(FIFOImageBuffer):
        def current_depth(self):
            with self.lock:
                return len(self.queue)

    image_loader = ImageLoader()
    image_loader.set_paths(main_folder_path, float_folder_path)
    image_loader.set_png_paths_len(png_paths_len)

    # --- Display / GL init ---
    window = display_init(state)

    if settings.SERVER_MODE:
        has_gl = window is not None
        if has_gl:
            print("[DISPLAY] SERVER_MODE: using headless ModernGL.")
            # Set params for GL renderer
            renderer.set_transform_parameters(
                fs_scale=1.0, fs_offset_x=0.0, fs_offset_y=0.0,
                image_size=(width, height), rotation_angle=0.0, mirror_mode=0
            )
        else:
            print("[DISPLAY] SERVER_MODE: using CPU-only compositing.")
    else:
        if window is None:
            raise RuntimeError("Local mode requires a GL window.")
        has_gl = True
        register_callbacks(window, state)

    # --- Initial index & images ---
    index, _ = update_index(png_paths_len, PINGPONG)
    last_index = index
    update_folder_selection(index, float_folder_count, main_folder_count)

    fifo_buffer = FIFOImageBufferPatched(max_size=FIFO_LENGTH)
    main_folder, float_folder = folder_dictionary["Main_and_Float_Folders"]
    main_image, float_image = image_loader.load_images(index, main_folder, float_folder)
    fifo_buffer.update(index, (main_image, float_image))

    current_main_img = main_image
    current_float_img = float_image

    if has_gl:
        main_texture = renderer.create_texture(main_image)
        float_texture = renderer.create_texture(float_image)

    compensator = RollingIndexCompensator(maxlen=10, correction_factor=0.5)

    def async_load_callback(fut, scheduled_index):
        try:
            result = fut.result()
            fifo_buffer.update(scheduled_index, result)
        except Exception as e:
            print("Error in async image load:", e)
            if monitor:
                monitor.record_load_error(scheduled_index, e)

    # --- Adaptive Workers ---
    if settings.SERVER_MODE or cpu_count <= 2:
        max_workers = 1
    else:
        max_workers = min(4, cpu_count)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Schedule first preload
        if index == 0:
            next_index = 1
        elif index == png_paths_len - 1:
            next_index = index - 1
        else:
            next_index = index + 1 if index > last_index else index - 1

        future = executor.submit(image_loader.load_images, next_index, main_folder, float_folder)
        future.add_done_callback(lambda fut, s_idx=next_index: async_load_callback(fut, s_idx))

        frame_times = deque(maxlen=60)
        frame_start = time.perf_counter()

        while (state.run_mode and not settings.SERVER_MODE) or settings.SERVER_MODE:
            successful_display = False

            # 1. Event polling (local)
            if not settings.SERVER_MODE and has_gl:
                glfw.poll_events()
                if not state.run_mode:
                    break

            # 2. Reinit
            if state.needs_update and not settings.SERVER_MODE and has_gl:
                display_init(state)
                state.needs_update = False

            # 3. Index & Images
            prev = index
            index, _ = update_index(png_paths_len, PINGPONG)

            if index != prev:
                last_index = prev
                update_folder_selection(index, float_folder_count, main_folder_count)
                main_folder, float_folder = folder_dictionary["Main_and_Float_Folders"]
                compensated = compensator.get_compensated_index(index)
                result = fifo_buffer.get(compensated)

                if result is not None:
                    displayed_index, main_img, float_img = result
                    compensator.update(index, displayed_index)
                    current_main_img = main_img
                    current_float_img = float_img

                    if has_gl:
                        main_texture = renderer.update_texture(main_texture, main_img)
                        float_texture = renderer.update_texture(float_texture, float_img)

                    successful_display = True
                else:
                    if not settings.SERVER_MODE:
                        # Only print miss if local, monitoring handles server misses
                        print(f"[MISS] FIFO miss for index {index}")

                # Schedule next
                if index == 0:
                    next_index = 1
                elif index == png_paths_len - 1:
                    next_index = index - 1
                else:
                    next_index = index + 1 if index > prev else index - 1

                future = executor.submit(image_loader.load_images, next_index, main_folder, float_folder)
                future.add_done_callback(lambda fut, s_idx=next_index: async_load_callback(fut, s_idx))

            # 4. Drawing (GL Path)
            if has_gl:
                if settings.SERVER_MODE:
                    window.use()
                renderer.overlay_images_single_pass(main_texture, float_texture, BACKGROUND_COLOR)

            # 5. Capture (Server Path)
            if settings.SERVER_MODE:
                now = time.time()
                if (index != last_captured_index) and (now - last_server_capture > capture_interval):
                    last_server_capture = now
                    last_captured_index = index
                    try:
                        if has_gl:
                            # GPU Path: Read pixels
                            window.use()
                            raw = window.fbo.read(components=3)
                            w, h = window.size
                            frame = np.frombuffer(raw, dtype=np.uint8).reshape((h, w, 3))
                            # Note: Renderer now handles coordinate flips if setup correctly,
                            # or use transform params.
                        else:
                            # CPU Path: Use the new renderer function
                            frame = renderer.composite_cpu(current_main_img, current_float_img)

                        if frame is not None:
                            # 0 = TJPF_RGB (for GL readback), 1 = TJPF_BGR (for OpenCV images)
                            # You may need to tune pixel_format depending on source.
                            # GL usually gives RGB. CPU path usually gives RGB (via our new function).
                            encoded = jpeg.encode(frame, quality=getattr(settings, "JPEG_QUALITY", 80), pixel_format=0)
                            exchange.set_frame(encoded)

                    except Exception as e:
                        print(f"[CAPTURE ERROR] {e}")

            # 6. Swap (Local)
            if not settings.SERVER_MODE and has_gl:
                glfw.swap_buffers(window)

            # 7. FPS / Monitoring
            now = time.perf_counter()
            dt = now - frame_start
            frame_times.append(dt)
            frame_start = now

            if FPS:
                to_sleep = (1.0 / FPS) - dt
                if to_sleep > 0:
                    time.sleep(to_sleep)

            if len(frame_times) > 1:
                avg = sum(frame_times) / len(frame_times)
                last_actual_fps = 1.0 / avg if avg > 0 else 0.0

            if monitor:
                monitor.update({
                    "index": index,
                    "displayed": displayed_index if "displayed_index" in locals() else index,
                    "delta": index - (displayed_index if "displayed_index" in locals() else index),
                    "fps": last_actual_fps,
                    "fifo_depth": fifo_buffer.current_depth(),
                    "successful_frame": successful_display,
                    "main_folder": main_folder,
                    "float_folder": float_folder,
                    "rand_mult": folder_dictionary.get("rand_mult"),
                    "main_folder_count": main_folder_count,
                    "float_folder_count": float_folder_count,
                })

            if not settings.SERVER_MODE and has_gl and glfw.window_should_close(window):
                state.run_mode = False

        if not settings.SERVER_MODE and has_gl:
            glfw.terminate()