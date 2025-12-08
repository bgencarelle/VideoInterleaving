import os
import time
from concurrent.futures import ThreadPoolExecutor
from collections import deque
import threading

import numpy as np
import webp  # Strict Requirement: pip install webp
from turbojpeg import TurboJPEG  # Strict Requirement: pip install PyTurboJPEG

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
    WEBP_STREAMING, WEBP_QUALITY, JPEG_QUALITY
)

# Check for optional lossless setting, default to False if missing
WEBP_LOSSLESS = getattr(settings, 'WEBP_LOSSLESS', False)

from index_calculator import update_index
from folder_selector import update_folder_selection, folder_dictionary
from display_manager import DisplayState, display_init
from event_handler import register_callbacks
import renderer
import ascii_converter

# Initialize Encoders
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

    # Identify Modes
    is_ascii = getattr(settings, 'ASCII_MODE', False)
    # Force Web Mode OFF if Ascii Mode is ON to match main.py exclusivity
    is_web = getattr(settings, 'SERVER_MODE', False) and not is_ascii
    # Headless if either server mode is active
    is_headless = is_web or is_ascii

    # --- NEW: LOGGING ---
    if is_web:
        mode_str = "WEBP" if WEBP_STREAMING else "MJPEG"
        qual_str = "LOSSLESS" if (
                    WEBP_STREAMING and WEBP_LOSSLESS) else f"Q={WEBP_QUALITY if WEBP_STREAMING else JPEG_QUALITY}"
        print(f"[DISPLAY] Stream Encoder: {mode_str} ({qual_str})")
    # --------------------

    import calculators
    _, main_folder_path, float_folder_path = calculators.init_all(clock_source)
    main_folder_count = len(main_folder_path[0])
    float_folder_count = len(float_folder_path[0])
    png_paths_len = len(main_folder_path)

    # 2. Initial Dimension Logic (Crucial for correct aspect ratio)
    temp_loader = ImageLoader()
    temp_loader.set_paths(main_folder_path, float_folder_path)

    # Load first image to determine window size
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
        mode_name = "ASCII" if is_ascii else ("WEB-SERVER" if is_web else "LOCAL-CPU")
        print(f"[DISPLAY] {mode_name}: Using CPU-only compositing")
    else:
        # ModernGL Mode (Headless or Local)
        has_gl = True
        if not is_headless and glfw:
            register_callbacks(window, state)

    # 4. Loader & Buffer Init
    index, _ = update_index(png_paths_len, PINGPONG)
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
        while (state.run_mode and not is_headless) or is_headless:
            successful_display = False

            # A. Poll Events (Local GL only)
            if not is_headless and has_gl and glfw:
                glfw.poll_events()
                if not state.run_mode: break

            # B. Handle Window Re-init (Local GL only)
            if state.needs_update and not is_headless and has_gl:
                display_init(state)
                state.needs_update = False

            # C. Logic Update
            prev = index
            index, _ = update_index(png_paths_len, PINGPONG)

            if index != prev:
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

                # Queue next frame
                next_idx = 1 if index == 0 else (
                    index - 1 if index == png_paths_len - 1 else (index + 1 if index > prev else index - 1))
                future = pool.submit(loader.load_images, next_idx, *folder_dictionary["Main_and_Float_Folders"])
                future.add_done_callback(lambda f, i=next_idx: async_cb(f, i))

            # D. Render Phase
            if has_gl:
                if is_headless: window.use()
                renderer.overlay_images_single_pass(
                    main_texture, float_texture, BACKGROUND_COLOR,
                    main_is_sbs=cur_m_sbs, float_is_sbs=cur_f_sbs
                )

            # E. Output / Capture Phase
            should_capture = False
            if is_headless or not has_gl:
                now = time.time()
                if (index != last_captured_index) and (now - last_server_capture > capture_interval):
                    should_capture = True
                    last_server_capture = now
                    last_captured_index = index

            if should_capture:
                try:
                    frame = None
                    # 1. Get Raw Data
                    if has_gl:
                        raw = window.fbo.read(components=3)
                        w_fbo, h_fbo = window.size
                        frame = np.frombuffer(raw, dtype=np.uint8).reshape((h_fbo, w_fbo, 3))
                    else:
                        tgt_size = getattr(settings, "HEADLESS_RES", (480, 640)) if is_web else None
                        frame = renderer.composite_cpu(
                            cur_main, cur_float,
                            main_is_sbs=cur_m_sbs, float_is_sbs=cur_f_sbs,
                            target_size=tgt_size
                        )

                    # 2. Encode Data
                    if frame is not None:
                        if is_web:
                            if WEBP_STREAMING:
                                # --- WEBP ENCODING ---
                                # Create WebPPicture from numpy
                                # Assuming renderer gives RGB. We pass it directly.
                                pic = webp.WebPPicture.from_numpy(frame)

                                # --- NEW: Lossless Logic ---
                                if WEBP_LOSSLESS:
                                    # Lossless mode ignores 'quality'
                                    config = webp.WebPConfig.new(preset=webp.WebPPreset.PHOTO, quality=100)
                                    config.lossless = 1
                                else:
                                    config = webp.WebPConfig.new(preset=webp.WebPPreset.PHOTO, quality=WEBP_QUALITY)
                                    config.lossless = 0

                                # Encode to memory buffer
                                buf = pic.encode(config).buffer()
                                exchange.set_frame(b'w' + bytes(buf))
                            else:
                                # --- JPEG ENCODING ---
                                enc = jpeg.encode(frame, quality=JPEG_QUALITY, pixel_format=0)
                                exchange.set_frame(b'j' + enc)

                        elif is_ascii:
                            text_frame = ascii_converter.to_ascii(frame)
                            exchange.set_frame(text_frame)

                except Exception as e:
                    print(f"[CAPTURE ERROR] {e}")

            # F. Swap Buffers (Local GL Only)
            if not is_headless and has_gl and glfw:
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
                    "index": index, "fps": last_actual_fps, "fifo_depth": fifo.current_depth(),
                    "successful_frame": successful_display,
                    "main_folder": folder_dictionary["Main_and_Float_Folders"][0],
                    "float_folder": folder_dictionary["Main_and_Float_Folders"][1]
                })

            if not is_headless and has_gl and glfw and glfw.window_should_close(window):
                state.run_mode = False

        if not is_headless and has_gl and glfw:
            glfw.terminate()