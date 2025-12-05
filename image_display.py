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

os.environ['PYOPENGL_ERROR_CHECKING'] = '0'

if not settings.SERVER_MODE:
    try:
        import glfw
    except ImportError:
        pass

from settings import (
    FULLSCREEN_MODE, PINGPONG, FPS, FRAME_COUNTER_DISPLAY, SHOW_DELTA,
    TEST_MODE, HTTP_MONITOR, CLOCK_MODE, FIFO_LENGTH, BACKGROUND_COLOR,
)
from index_calculator import update_index
from folder_selector import update_folder_selection, folder_dictionary
from display_manager import DisplayState, display_init
from event_handler import register_callbacks
import renderer
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

    def update(self, current, displayed):
        self.diffs.append(current - displayed)

    def get_compensated_index(self, current):
        if not self.diffs: return current
        return current - round((sum(self.diffs) / len(self.diffs)) * self.correction_factor)


from image_loader import ImageLoader, FIFOImageBuffer


class FIFOImageBufferPatched(FIFOImageBuffer):
    def current_depth(self):
        with self.lock: return len(self.queue)


def run_display(clock_source=CLOCK_MODE):
    state = DisplayState()
    state.fullscreen = FULLSCREEN_MODE
    last_actual_fps = FPS

    capture_rate = getattr(settings, "SERVER_CAPTURE_RATE", FPS or 10)
    capture_interval = 1.0 / capture_rate
    last_server_capture = time.time()
    last_captured_index = None

    import calculators
    _, main_folder_path, float_folder_path = calculators.init_all(clock_source)
    main_folder_count = len(main_folder_path[0])
    float_folder_count = len(float_folder_path[0])
    png_paths_len = len(main_folder_path)

    # --- Phase 2: Dimension Logic ---
    temp_loader = ImageLoader()
    temp_loader.set_paths(main_folder_path, float_folder_path)

    # Load first image to determine window size
    img0, is_sbs0 = temp_loader.read_image(main_folder_path[0][0])

    if img0 is None: raise RuntimeError("Failed to load initial image")

    h, w = img0.shape[:2]
    # If initial image is SBS, the visual width is half the texture width
    if is_sbs0:
        state.image_size = (w // 2, h)
    else:
        state.image_size = (w, h)

    # Init Window/Context
    window = display_init(state)

    if settings.SERVER_MODE:
        has_gl = window is not None
        if has_gl:
            print("[DISPLAY] Headless ModernGL Active")
            renderer.set_transform_parameters(
                fs_scale=1.0, fs_offset_x=0.0, fs_offset_y=0.0,
                image_size=state.image_size, rotation_angle=0.0, mirror_mode=0
            )
        else:
            print("[DISPLAY] CPU-only compositing Active")
    else:
        if window is None: raise RuntimeError("Local GL Window Failed")
        has_gl = True
        register_callbacks(window, state)

    index, _ = update_index(png_paths_len, PINGPONG)
    last_index = index
    update_folder_selection(index, float_folder_count, main_folder_count)

    loader = ImageLoader()
    loader.set_paths(main_folder_path, float_folder_path)
    loader.set_png_paths_len(png_paths_len)

    fifo = FIFOImageBufferPatched(max_size=FIFO_LENGTH)

    # Load initial frame (returns 4 items)
    res0 = loader.load_images(index, *folder_dictionary["Main_and_Float_Folders"])
    fifo.update(index, res0)

    # Unpack initial frame
    cur_main, cur_float, cur_m_sbs, cur_f_sbs = res0

    if has_gl:
        main_texture = renderer.create_texture(cur_main)
        float_texture = renderer.create_texture(cur_float)

    compensator = RollingIndexCompensator()

    def async_cb(fut, idx):
        try:
            # Result is now (m_img, f_img, m_sbs, f_sbs)
            fifo.update(idx, fut.result())
        except Exception as e:
            print("Async Load Error:", e)
            if monitor: monitor.record_load_error(idx, e)

    # Dynamic Workers
    cpu_count = os.cpu_count() or 1
    max_workers = 2 if cpu_count <= 2 else min(6, cpu_count)

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        next_idx = 1 if index == 0 else (index - 1 if index == png_paths_len - 1 else index + 1)
        future = pool.submit(loader.load_images, next_idx, *folder_dictionary["Main_and_Float_Folders"])
        future.add_done_callback(lambda f, i=next_idx: async_cb(f, i))

        frame_times = deque(maxlen=60)
        frame_start = time.perf_counter()

        while (state.run_mode and not settings.SERVER_MODE) or settings.SERVER_MODE:
            successful_display = False

            if not settings.SERVER_MODE and has_gl:
                glfw.poll_events()
                if not state.run_mode: break

            if state.needs_update and not settings.SERVER_MODE:
                display_init(state)
                state.needs_update = False

            prev = index
            index, _ = update_index(png_paths_len, PINGPONG)

            if index != prev:
                last_index = prev
                update_folder_selection(index, float_folder_count, main_folder_count)

                comp_idx = compensator.get_compensated_index(index)
                res = fifo.get(comp_idx)

                if res:
                    # Unpack 5 items: index + 4 data items
                    d_idx, m_img, f_img, m_sbs, f_sbs = res
                    compensator.update(index, d_idx)

                    if has_gl:
                        main_texture = renderer.update_texture(main_texture, m_img)
                        float_texture = renderer.update_texture(float_texture, f_img)

                    cur_main, cur_float = m_img, f_img
                    cur_m_sbs, cur_f_sbs = m_sbs, f_sbs
                    successful_display = True
                else:
                    if not settings.SERVER_MODE: print(f"[MISS] {index}")

                next_idx = 1 if index == 0 else (
                    index - 1 if index == png_paths_len - 1 else (index + 1 if index > prev else index - 1))
                future = pool.submit(loader.load_images, next_idx, *folder_dictionary["Main_and_Float_Folders"])
                future.add_done_callback(lambda f, i=next_idx: async_cb(f, i))

            if has_gl:
                if settings.SERVER_MODE: window.use()
                # Pass flags to Shader
                renderer.overlay_images_single_pass(
                    main_texture, float_texture, BACKGROUND_COLOR,
                    main_is_sbs=cur_m_sbs, float_is_sbs=cur_f_sbs
                )

            if settings.SERVER_MODE:
                now = time.time()
                if (index != last_captured_index) and (now - last_server_capture > capture_interval):
                    last_server_capture = now
                    last_captured_index = index
                    try:
                        if has_gl:
                            window.use()
                            raw = window.fbo.read(components=3)
                            w, h = window.size
                            frame = np.frombuffer(raw, dtype=np.uint8).reshape((h, w, 3))
                        else:
                            # Pass flags to CPU fallback
                            frame = renderer.composite_cpu(
                                cur_main, cur_float,
                                main_is_sbs=cur_m_sbs, float_is_sbs=cur_f_sbs
                            )

                        if frame is not None:
                            encoded = jpeg.encode(frame, quality=getattr(settings, "JPEG_QUALITY", 80), pixel_format=0)
                            exchange.set_frame(encoded)
                    except Exception as e:
                        print(f"[CAPTURE ERROR] {e}")

            if not settings.SERVER_MODE and has_gl:
                glfw.swap_buffers(window)

            now = time.perf_counter()
            dt = now - frame_start
            frame_times.append(dt)
            frame_start = now
            if FPS:
                s = (1.0 / FPS) - dt
                if s > 0: time.sleep(s)

            if len(frame_times) > 1:
                last_actual_fps = 1.0 / (sum(frame_times) / len(frame_times))
                if not HTTP_MONITOR and FRAME_COUNTER_DISPLAY: print(f"{last_actual_fps:.1f} FPS")

            if monitor:
                monitor.update({
                    "index": index,
                    "displayed": d_idx if 'd_idx' in locals() else index,
                    "delta": index - (d_idx if 'd_idx' in locals() else index),
                    "fps": last_actual_fps, "fifo_depth": fifo.current_depth(),
                    "successful_frame": successful_display,
                    "main_folder": folder_dictionary["Main_and_Float_Folders"][0],
                    "float_folder": folder_dictionary["Main_and_Float_Folders"][1],
                    "rand_mult": folder_dictionary.get("rand_mult"),
                    "main_folder_count": main_folder_count, "float_folder_count": float_folder_count
                })

            if not settings.SERVER_MODE and has_gl and glfw.window_should_close(window):
                state.run_mode = False

        if not settings.SERVER_MODE and has_gl: glfw.terminate()