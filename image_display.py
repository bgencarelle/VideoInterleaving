import os
import time
from concurrent.futures import ThreadPoolExecutor
from collections import deque
import threading

import numpy as np

import settings
from shared_state import exchange

# Clean up console noise from ModernGL on some systems
os.environ['PYOPENGL_ERROR_CHECKING'] = '0'

# Conditional import logic handled by display_manager,
# but we need glfw here for event polling if it exists.
try:
    import glfw
except ImportError:
    glfw = None

from settings import (
    FULLSCREEN_MODE, PINGPONG, FPS, FRAME_COUNTER_DISPLAY, SHOW_DELTA,
    TEST_MODE, HTTP_MONITOR, CLOCK_MODE, FIFO_LENGTH, BACKGROUND_COLOR,
)
from index_calculator import update_index
from folder_selector import update_folder_selection, folder_dictionary
from display_manager import DisplayState, display_init
from event_handler import register_callbacks
import renderer

# TurboJPEG for Encoding Server Streams
from turbojpeg import TurboJPEG

jpeg = TurboJPEG()

monitor = None
if TEST_MODE and HTTP_MONITOR:
    from lightweight_monitor import start_monitor

    monitor = start_monitor()


# -----------------------------------------------------------------------------
# Helpers: FIFO Buffer & Compensator
# -----------------------------------------------------------------------------

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


# -----------------------------------------------------------------------------
# MAIN LOOP
# -----------------------------------------------------------------------------

def run_display(clock_source=CLOCK_MODE):
    # 1. State Setup
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

    # 2. Initial Dimension Logic (Crucial for correct aspect ratio)
    temp_loader = ImageLoader()
    temp_loader.set_paths(main_folder_path, float_folder_path)

    # Load first image to determine window size
    # We must handle the tuple return (img, is_sbs)
    img0, is_sbs0 = temp_loader.read_image(main_folder_path[0][0])

    if img0 is None:
        raise RuntimeError("Failed to load initial image to determine dimensions.")

    h, w = img0.shape[:2]

    # "The Extension Rule": If SBS, visual width is half the texture width.
    if is_sbs0:
        state.image_size = (w // 2, h)
    else:
        state.image_size = (w, h)

    # 3. Initialize Window / Context (Universal: GPU or CPU)
    window = display_init(state)

    # Determine capabilities based on what display_init returned
    if window is None:
        # Fallback / Server CPU Mode / ASCII Mode
        has_gl = False
        mode_name = "SERVER" if settings.SERVER_MODE else "LOCAL-CPU"
        print(f"[DISPLAY] {mode_name}: Using CPU-only compositing (TurboJPEG Optimized)")
    else:
        # ModernGL Mode (Headless or Local)
        has_gl = True
        # Only register keyboard/mouse callbacks if it's a real local window
        # (HeadlessWindow is a custom class, not a glfw object)
        if not settings.SERVER_MODE and glfw:
            register_callbacks(window, state)

    # 4. Loader & Buffer Init
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

    # Unpack initial frame for immediate display
    cur_main, cur_float, cur_m_sbs, cur_f_sbs = res0

    # Initialize Textures (if GL)
    main_texture = None
    float_texture = None
    if has_gl:
        main_texture = renderer.create_texture(cur_main)
        float_texture = renderer.create_texture(cur_float)

    compensator = RollingIndexCompensator()

    # Async Callback
    def async_cb(fut, idx):
        try:
            # Result is (m_img, f_img, m_sbs, f_sbs)
            fifo.update(idx, fut.result())
        except Exception as e:
            print("Async Load Error:", e)
            if monitor: monitor.record_load_error(idx, e)

    # 5. Thread Pool Setup
    cpu_count = os.cpu_count() or 1
    max_workers = 2 if cpu_count <= 2 else min(6, cpu_count)

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        # Pre-load next frame
        next_idx = 1 if index == 0 else (index - 1 if index == png_paths_len - 1 else index + 1)
        future = pool.submit(loader.load_images, next_idx, *folder_dictionary["Main_and_Float_Folders"])
        future.add_done_callback(lambda f, i=next_idx: async_cb(f, i))

        frame_times = deque(maxlen=60)
        frame_start = time.perf_counter()

        # ---------------------------------------------------------------------
        # MAIN EXECUTION LOOP
        # ---------------------------------------------------------------------
        while (state.run_mode and not settings.SERVER_MODE) or settings.SERVER_MODE:
            successful_display = False

            # A. Poll Events (Local GL only)
            if not settings.SERVER_MODE and has_gl and glfw:
                glfw.poll_events()
                if not state.run_mode: break

            # B. Handle Window Re-init (Local GL only)
            if state.needs_update and not settings.SERVER_MODE and has_gl:
                display_init(state)
                state.needs_update = False

            # C. Logic Update
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

                    # Update GPU Textures
                    if has_gl:
                        main_texture = renderer.update_texture(main_texture, m_img)
                        float_texture = renderer.update_texture(float_texture, f_img)

                    # Update CPU references
                    cur_main, cur_float = m_img, f_img
                    cur_m_sbs, cur_f_sbs = m_sbs, f_sbs
                    successful_display = True
                else:
                    if not settings.SERVER_MODE: print(f"[MISS] {index}")

                # Queue next frame
                next_idx = 1 if index == 0 else (
                    index - 1 if index == png_paths_len - 1 else (index + 1 if index > prev else index - 1))
                future = pool.submit(loader.load_images, next_idx, *folder_dictionary["Main_and_Float_Folders"])
                future.add_done_callback(lambda f, i=next_idx: async_cb(f, i))

            # D. Render Phase
            if has_gl:
                # If Server GL Mode, bind the FBO
                if settings.SERVER_MODE:
                    window.use()

                # Draw
                renderer.overlay_images_single_pass(
                    main_texture, float_texture, BACKGROUND_COLOR,
                    main_is_sbs=cur_m_sbs, float_is_sbs=cur_f_sbs
                )

            # E. Output / Capture Phase (Server or CPU-Local)
            # This block runs if we are streaming OR if we have no GPU (ASCII/CPU mode)
            should_capture = False

            if settings.SERVER_MODE:
                # Throttled capture for web stream
                now = time.time()
                if (index != last_captured_index) and (now - last_server_capture > capture_interval):
                    should_capture = True
                    last_server_capture = now
                    last_captured_index = index
            elif not has_gl:
                # Local CPU mode (e.g. ASCII): Process every frame essentially
                should_capture = True

            if should_capture:
                try:
                    frame = None
                    if has_gl:
                        # Readback from GPU
                        raw = window.fbo.read(components=3)
                        w_fbo, h_fbo = window.size
                        frame = np.frombuffer(raw, dtype=np.uint8).reshape((h_fbo, w_fbo, 3))
                    else:
                        # Pure CPU Composite
                        frame = renderer.composite_cpu(
                            cur_main, cur_float,
                            main_is_sbs=cur_m_sbs, float_is_sbs=cur_f_sbs
                        )

                    # Output Handling
                    if frame is not None:
                        if settings.SERVER_MODE:
                            # Encode to JPEG for stream
                            encoded = jpeg.encode(frame, quality=getattr(settings, "JPEG_QUALITY", 80), pixel_format=0)
                            exchange.set_frame(encoded)
                        else:
                            # LOCAL CPU PLACEHOLDER
                            # This is where you would call: ascii_printer.render(frame)
                            pass

                except Exception as e:
                    print(f"[CAPTURE ERROR] {e}")

            # F. Swap Buffers (Local GL Only)
            if not settings.SERVER_MODE and has_gl and glfw:
                glfw.swap_buffers(window)
                if glfw.window_should_close(window):
                    state.run_mode = False

            # G. Timing
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

            # H. Monitor Update
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

            if not settings.SERVER_MODE and has_gl and glfw and glfw.window_should_close(window):
                state.run_mode = False

        if not settings.SERVER_MODE and has_gl and glfw:
            glfw.terminate()