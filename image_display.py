import os
import time
from concurrent.futures import ThreadPoolExecutor
from collections import deque
import threading

import numpy as np
#import webp  # Strict Requirement: pip install webp
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

# --- SAFE SETTINGS IMPORTS ---
FULLSCREEN_MODE = getattr(settings, 'FULLSCREEN_MODE', True)
PINGPONG = getattr(settings, 'PINGPONG', False)
FPS = getattr(settings, 'FPS', 30)
FRAME_COUNTER_DISPLAY = getattr(settings, 'FRAME_COUNTER_DISPLAY', True)
CLOCK_MODE = getattr(settings, 'CLOCK_MODE', 0)
FIFO_LENGTH = getattr(settings, 'FIFO_LENGTH', 30)
BACKGROUND_COLOR = getattr(settings, 'BACKGROUND_COLOR', (0, 0, 0))

# Streaming Settings
JPEG_QUALITY = getattr(settings, 'JPEG_QUALITY', 55)
SERVER_CAPTURE_RATE = getattr(settings, "SERVER_CAPTURE_RATE", FPS or 10)
HEADLESS_RES = getattr(settings, "HEADLESS_RES", (480, 640))

# --- ASCII PRE-BAKE CONSTANTS ---
# Pre-calculate LUTs once for the merger/renderer
_ansi_colors = [f"\033[38;5;{i}m" for i in range(256)]
_ansi_colors[16] = "\033[38;5;235m"  # Black crush fix
ANSI_LUT = np.array(_ansi_colors)
RESET_CODE = "\033[0m"
# --------------------------------

from index_calculator import update_index
from folder_selector import update_folder_selection, folder_dictionary
from display_manager import DisplayState, display_init
from event_handler import register_callbacks
import renderer
import ascii_converter

# Initialize Encoders
jpeg = TurboJPEG()

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
# WORKER FUNCTION (Runs in Background Threads)
# -----------------------------------------------------------------------------

def load_and_render_frame(loader, index, main_folder, float_folder):
    """
    Loads images AND performs the heavy ASCII merging/string-generation
    in the background thread. Returns the final string if ASCII.
    """
    # 1. Load Data
    m_img, f_img, m_sbs, f_sbs = loader.load_images(index, main_folder, float_folder)

    # 2. Check for ASCII Data
    if isinstance(m_img, dict):
        try:
            # --- MERGE LAYERS ---
            if isinstance(f_img, dict):
                f_chars = f_img["chars"]
                # Determine mask based on byte vs str type
                if f_chars.dtype.kind == 'S':
                    mask = (f_chars != b' ')
                else:
                    mask = (f_chars != ' ')

                final_chars = m_img["chars"].copy()
                final_colors = m_img["colors"].copy()

                # Numpy masking is fast
                final_chars[mask] = f_chars[mask]
                final_colors[mask] = f_img["colors"][mask]
            else:
                final_chars = m_img["chars"]
                final_colors = m_img["colors"]

            # --- DOWNSCALE (STRIDE) ---
            baked_h, baked_w = final_chars.shape
            target_w = getattr(settings, 'ASCII_WIDTH', baked_w)
            target_h = getattr(settings, 'ASCII_HEIGHT', baked_h)

            step_x = max(1, baked_w // target_w)
            step_y = max(1, baked_h // target_h)

            if step_x > 1 or step_y > 1:
                final_chars = final_chars[::step_y, ::step_x]
                final_colors = final_colors[::step_y, ::step_x]

            # --- RENDER TO STRING ---
            # Cast bytes to string if needed
            if final_chars.dtype.kind == 'S':
                final_chars = final_chars.astype(str)

            # Look up ANSI codes and combine
            # This is the most expensive CPU part, now done in a thread!
            color_strings = ANSI_LUT[final_colors]
            image_grid = np.char.add(color_strings, final_chars)

            rows = ["".join(row) for row in image_grid]
            ascii_string = "\r\n".join(rows) + RESET_CODE

            # Return the String as the "Image"
            return ascii_string, None, False, False

        except Exception as e:
            # If render fails, return None or log
            print(f"ASCII Render Error: {e}")
            return None, None, False, False

    # 3. Standard Image Return
    return m_img, f_img, m_sbs, f_sbs


# -----------------------------------------------------------------------------
# MAIN LOOP
# -----------------------------------------------------------------------------

def run_display(clock_source=CLOCK_MODE):
    # 1. State Setup
    state = DisplayState()
    state.fullscreen = FULLSCREEN_MODE
    last_actual_fps = FPS

    capture_rate = SERVER_CAPTURE_RATE
    capture_interval = 1.0 / capture_rate
    last_server_capture = time.time()
    last_captured_index = None

    # Monitor Counters
    fifo_miss_count = 0
    last_fifo_miss = -1
    last_displayed_index = 0

    # Identify Modes
    is_ascii = getattr(settings, 'ASCII_MODE', False)
    is_web = getattr(settings, 'SERVER_MODE', False) and not is_ascii
    is_headless = is_web or is_ascii

    # --- LOGGING ---
    if is_web:
        mode_str = "MJPEG"
        qual_str = JPEG_QUALITY
        print(f"[DISPLAY] Stream Encoder: {mode_str} ({qual_str})")
    # --------------------

    import calculators
    _, main_folder_path, float_folder_path = calculators.init_all(clock_source)
    main_folder_count = len(main_folder_path[0])
    float_folder_count = len(float_folder_path[0])
    png_paths_len = len(main_folder_path)

    # 2. Initial Dimension Logic
    temp_loader = ImageLoader()
    temp_loader.set_paths(main_folder_path, float_folder_path)

    img0, is_sbs0 = temp_loader.read_image(main_folder_path[0][0])
    if img0 is None:
        raise RuntimeError("Failed to load initial image.")

    # ASCII Dict detection for initialization
    if isinstance(img0, dict):
        h, w = img0['chars'].shape
        state.image_size = (w, h)
        has_gl = False
        is_headless = True
    else:
        h, w = img0.shape[:2]
        if is_sbs0:
            state.image_size = (w // 2, h)
        else:
            state.image_size = (w, h)

    # 3. Initialize Window
    if isinstance(img0, dict):
        window = None
        has_gl = False
        print(f"[DISPLAY] Pre-baked ASCII Mode Detected (GL Disabled)")
    else:
        window = display_init(state)

        if window is None:
            has_gl = False
            mode_name = "ASCII" if is_ascii else ("WEB-SERVER" if is_web else "LOCAL-CPU")
            print(f"[DISPLAY] {mode_name}: Using CPU-only compositing")
        else:
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

    # Pre-load using the NEW helper directly for the first frame
    res0 = load_and_render_frame(loader, index, *folder_dictionary["Main_and_Float_Folders"])
    fifo.update(index, res0)

    cur_main, cur_float, cur_m_sbs, cur_f_sbs = None, None, False, False

    main_texture = None
    float_texture = None
    # Only create textures if NOT ascii string and GL active
    if has_gl and not isinstance(res0[0], str):
        main_texture = renderer.create_texture(res0[0])
        float_texture = renderer.create_texture(res0[1])

    compensator = RollingIndexCompensator()

    def async_cb(fut, idx):
        try:
            fifo.update(idx, fut.result())
        except Exception as e:
            if monitor: monitor.record_load_error(idx, e)

    # 5. Thread Pool
    cpu_count = os.cpu_count() or 1
    # Increase workers slightly since we are doing more CPU work in them now
    max_workers = min(8, cpu_count + 2)

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        next_idx = 1 if index == 0 else (index - 1 if index == png_paths_len - 1 else index + 1)
        # Use the NEW worker function
        future = pool.submit(load_and_render_frame, loader, next_idx, *folder_dictionary["Main_and_Float_Folders"])
        future.add_done_callback(lambda f, i=next_idx: async_cb(f, i))

        frame_times = deque(maxlen=60)
        frame_start = time.perf_counter()

        while (state.run_mode and not is_headless) or is_headless:
            successful_display = False

            if not is_headless and has_gl and glfw:
                glfw.poll_events()
                if not state.run_mode: break

            if state.needs_update and not is_headless and has_gl:
                display_init(state)
                state.needs_update = False

            prev = index
            index, _ = update_index(png_paths_len, PINGPONG)

            if index != prev:
                update_folder_selection(index, float_folder_count, main_folder_count)

                comp_idx = compensator.get_compensated_index(index)
                res = fifo.get(comp_idx)

                if res:
                    d_idx, m_img, f_img, m_sbs, f_sbs = res
                    compensator.update(index, d_idx)

                    # Update loop pointers
                    cur_main = m_img
                    cur_float = f_img
                    cur_m_sbs = m_sbs
                    cur_f_sbs = f_sbs

                    # --- [OPTIMIZED] PRE-BAKED ASCII PATH ---
                    # Check if the worker thread already returned a String
                    if isinstance(m_img, str):
                        # It's already rendered! Just send it.
                        exchange.set_frame(m_img)
                        successful_display = True
                        last_displayed_index = d_idx

                        # Trigger Next Load
                        next_idx = 1 if index == 0 else (
                            index - 1 if index == png_paths_len - 1 else (index + 1 if index > prev else index - 1))
                        future = pool.submit(load_and_render_frame, loader, next_idx,
                                             *folder_dictionary["Main_and_Float_Folders"])
                        future.add_done_callback(lambda f, i=next_idx: async_cb(f, i))

                        # Timing
                        now = time.perf_counter()
                        dt = now - frame_start
                        frame_times.append(dt)
                        frame_start = now
                        if FPS:
                            s = (1.0 / FPS) - dt
                            if s > 0: time.sleep(s)

                        if len(frame_times) > 1:
                            last_actual_fps = 1.0 / (sum(frame_times) / len(frame_times))

                        if monitor:
                            monitor.update({
                                "index": index,
                                "displayed": last_displayed_index,
                                "fps": last_actual_fps,
                                "fifo_depth": fifo.current_depth(),
                                "successful_frame": True,
                                "main_folder": folder_dictionary["Main_and_Float_Folders"][0],
                                "float_folder": folder_dictionary["Main_and_Float_Folders"][1],
                                "rand_mult": folder_dictionary.get("rand_mult"),
                                "main_folder_count": main_folder_count,
                                "float_folder_count": float_folder_count,
                                "fifo_miss_count": fifo_miss_count,
                                "last_fifo_miss": last_fifo_miss
                            })

                        continue
                    # ----------------------------------------

                    # GL Texture Update (Only for non-string)
                    if has_gl:
                        main_texture = renderer.update_texture(main_texture, m_img)
                        float_texture = renderer.update_texture(float_texture, f_img)

                    successful_display = True
                    last_displayed_index = d_idx
                else:
                    if not is_headless: print(f"[MISS] {index}")
                    fifo_miss_count += 1
                    last_fifo_miss = index

                next_idx = 1 if index == 0 else (
                    index - 1 if index == png_paths_len - 1 else (index + 1 if index > prev else index - 1))
                future = pool.submit(load_and_render_frame, loader, next_idx,
                                     *folder_dictionary["Main_and_Float_Folders"])
                future.add_done_callback(lambda f, i=next_idx: async_cb(f, i))

            # Render (GL)
            if has_gl:
                if is_headless: window.use()
                renderer.overlay_images_single_pass(
                    main_texture, float_texture, BACKGROUND_COLOR,
                    main_is_sbs=cur_m_sbs, float_is_sbs=cur_f_sbs
                )

            # Capture (Only for Images/Headless Web)
            should_capture = False
            if is_headless or not has_gl:
                now = time.time()
                if (index != last_captured_index) and (now - last_server_capture > capture_interval):
                    should_capture = True
                    last_server_capture = now
                    last_captured_index = index

            if should_capture:
                # If m_img is a string, we already handled it.
                if isinstance(cur_main, str):
                    pass
                elif isinstance(cur_main, dict):
                    pass  # Should be caught by str check, but just in case
                elif cur_main is not None:
                    try:
                        frame = None
                        if has_gl:
                            raw = window.fbo.read(components=3)
                            w_fbo, h_fbo = window.size
                            frame = np.frombuffer(raw, dtype=np.uint8).reshape((h_fbo, w_fbo, 3))
                        else:
                            tgt_size = HEADLESS_RES if is_web else None
                            frame = renderer.composite_cpu(
                                cur_main, cur_float,
                                main_is_sbs=cur_m_sbs, float_is_sbs=cur_f_sbs,
                                target_size=tgt_size
                            )

                        if frame is not None:
                            if is_web:
                                    enc = jpeg.encode(frame, quality=JPEG_QUALITY, pixel_format=0)
                                    exchange.set_frame(b'j' + enc)

                            elif is_ascii:
                                # Fallback (Live Conversion)
                                text_frame = ascii_converter.to_ascii(frame)
                                exchange.set_frame(text_frame)

                    except Exception as e:
                        print(f"[CAPTURE ERROR] {e}")

            if not is_headless and has_gl and glfw:
                glfw.swap_buffers(window)
                if glfw.window_should_close(window):
                    state.run_mode = False

            now = time.perf_counter()
            dt = now - frame_start
            frame_times.append(dt)
            frame_start = now
            if FPS:
                s = (1.0 / FPS) - dt
                if s > 0: time.sleep(s)

            if len(frame_times) > 1:
                last_actual_fps = 1.0 / (sum(frame_times) / len(frame_times))
                #print(f"{last_actual_fps:.1f} FPS")

            if monitor:
                monitor.update({
                    "index": index,
                    "displayed": last_displayed_index,
                    "fps": last_actual_fps,
                    "fifo_depth": fifo.current_depth(),
                    "successful_frame": successful_display,
                    "main_folder": folder_dictionary["Main_and_Float_Folders"][0],
                    "float_folder": folder_dictionary["Main_and_Float_Folders"][1],
                    "rand_mult": folder_dictionary.get("rand_mult"),
                    "main_folder_count": main_folder_count,
                    "float_folder_count": float_folder_count,
                    "fifo_miss_count": fifo_miss_count,
                    "last_fifo_miss": last_fifo_miss
                })

            if not is_headless and has_gl and glfw and glfw.window_should_close(window):
                state.run_mode = False

        if not is_headless and has_gl and glfw:
            glfw.terminate()