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

# --- SAFE SETTINGS IMPORTS ---
FULLSCREEN_MODE = getattr(settings, 'FULLSCREEN_MODE', True)
PINGPONG = getattr(settings, 'PINGPONG', False)
FPS = getattr(settings, 'FPS', 30)
FRAME_COUNTER_DISPLAY = getattr(settings, 'FRAME_COUNTER_DISPLAY', True)
SHOW_DELTA = getattr(settings, 'SHOW_DELTA', False)
TEST_MODE = getattr(settings, 'TEST_MODE', False)
HTTP_MONITOR = getattr(settings, 'HTTP_MONITOR', True)
CLOCK_MODE = getattr(settings, 'CLOCK_MODE', 0)
FIFO_LENGTH = getattr(settings, 'FIFO_LENGTH', 30)
BACKGROUND_COLOR = getattr(settings, 'BACKGROUND_COLOR', (0, 0, 0))

# Streaming Settings
WEBP_STREAMING = getattr(settings, 'WEBP_STREAMING', False)
WEBP_LOSSLESS = getattr(settings, 'WEBP_LOSSLESS', False)
WEBP_QUALITY = getattr(settings, 'WEBP_QUALITY', 55)
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
        # Force headless/ascii behavior implicitly if we loaded .npz/.npy
        has_gl = False
        is_headless = True
    else:
        h, w = img0.shape[:2]
        if is_sbs0:
            state.image_size = (w // 2, h)
        else:
            state.image_size = (w, h)

    # 3. Initialize Window
    # If we detected .npz data, display_init will return None (or we skip it)
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

    res0 = loader.load_images(index, *folder_dictionary["Main_and_Float_Folders"])
    fifo.update(index, res0)

    # Initialize these for the loop scope (Prevents 'cur_main' crash if FIFO misses on first frame)
    cur_main, cur_float, cur_m_sbs, cur_f_sbs = None, None, False, False

    # Textures (Only if GL and NOT ASCII dict)
    main_texture = None
    float_texture = None
    if has_gl and not isinstance(res0[0], dict):
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
    max_workers = 2 if cpu_count <= 2 else min(6, cpu_count)

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        next_idx = 1 if index == 0 else (index - 1 if index == png_paths_len - 1 else index + 1)
        future = pool.submit(loader.load_images, next_idx, *folder_dictionary["Main_and_Float_Folders"])
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

                    # --- [NEW] PRE-BAKED ASCII PATH ---
                    if isinstance(m_img, dict):
                        # 1. Merge Layers (Background + Foreground)
                        # Transparency defined as space ' '
                        # Check for 'S' type (Bytes) and decode if necessary for comparison
                        f_is_bytes = False
                        if isinstance(f_img, dict):
                            f_chars = f_img["chars"]
                            if f_chars.dtype.kind == 'S':
                                # Compare against bytes space b' '
                                mask = (f_chars != b' ')
                            else:
                                # Compare against string space ' '
                                mask = (f_chars != ' ')

                            final_chars = m_img["chars"].copy()
                            final_colors = m_img["colors"].copy()
                            final_chars[mask] = f_chars[mask]
                            final_colors[mask] = f_img["colors"][mask]
                        else:
                            # If float is missing/invalid/standard-img, just use main
                            final_chars = m_img["chars"]
                            final_colors = m_img["colors"]

                        # 2. DYNAMIC DOWNSCALING
                        # Get baked dimensions
                        baked_h, baked_w = final_chars.shape
                        target_w = getattr(settings, 'ASCII_WIDTH', baked_w)
                        target_h = getattr(settings, 'ASCII_HEIGHT', baked_h)

                        # Calculate Step (Stride)
                        step_x = max(1, baked_w // target_w)
                        step_y = max(1, baked_h // target_h)

                        if step_x > 1 or step_y > 1:
                            final_chars = final_chars[::step_y, ::step_x]
                            final_colors = final_colors[::step_y, ::step_x]

                        # 3. TYPE CORRECTION & RENDER
                        # If chars are Bytes (S1), cast to Unicode (U1) so we can add to color strings
                        if final_chars.dtype.kind == 'S':
                            final_chars = final_chars.astype(str)

                        # Vectorized string addition: ColorCode + Char
                        color_strings = ANSI_LUT[final_colors]
                        image_grid = np.char.add(color_strings, final_chars)

                        # Join rows
                        rows = ["".join(row) for row in image_grid]
                        ascii_out = "\r\n".join(rows) + RESET_CODE

                        # 4. Output to Exchange
                        exchange.set_frame(ascii_out)
                        successful_display = True
                        last_displayed_index = d_idx

                        # 5. Trigger Next Load
                        next_idx = 1 if index == 0 else (
                            index - 1 if index == png_paths_len - 1 else (index + 1 if index > prev else index - 1))
                        future = pool.submit(loader.load_images, next_idx, *folder_dictionary["Main_and_Float_Folders"])
                        future.add_done_callback(lambda f, i=next_idx: async_cb(f, i))

                        # 6. Timing & Loop (Skip Rendering)
                        now = time.perf_counter()
                        dt = now - frame_start
                        frame_times.append(dt)
                        frame_start = now
                        if FPS:
                            s = (1.0 / FPS) - dt
                            if s > 0: time.sleep(s)

                        # Recalculate FPS immediately for Monitor
                        if len(frame_times) > 1:
                            last_actual_fps = 1.0 / (sum(frame_times) / len(frame_times))

                        # UPDATE MONITOR BEFORE CONTINUING
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

                        # SKIP REMAINDER OF LOOP (GL & Capture)
                        continue
                        # ----------------------------------

                    if has_gl:
                        main_texture = renderer.update_texture(main_texture, m_img)
                        float_texture = renderer.update_texture(float_texture, f_img)

                    successful_display = True
                    last_displayed_index = d_idx
                else:
                    if not is_headless: print(f"[MISS] {index}")
                    fifo_miss_count += 1
                    last_fifo_miss = index
                    # Note: cur_main retains previous value if MISS happens

                next_idx = 1 if index == 0 else (
                    index - 1 if index == png_paths_len - 1 else (index + 1 if index > prev else index - 1))
                future = pool.submit(loader.load_images, next_idx, *folder_dictionary["Main_and_Float_Folders"])
                future.add_done_callback(lambda f, i=next_idx: async_cb(f, i))

            # Render (Skipped if ASCII path taken)
            if has_gl:
                if is_headless: window.use()
                renderer.overlay_images_single_pass(
                    main_texture, float_texture, BACKGROUND_COLOR,
                    main_is_sbs=cur_m_sbs, float_is_sbs=cur_f_sbs
                )

            # Capture (Skipped if ASCII path taken or cur_main is invalid)
            should_capture = False
            if is_headless or not has_gl:
                now = time.time()
                if (index != last_captured_index) and (now - last_server_capture > capture_interval):
                    should_capture = True
                    last_server_capture = now
                    last_captured_index = index

            if should_capture:
                # Safety check: Don't enter here if we are holding a dict
                # (should be caught by continue above, but prevents crashes if flow changes)
                if isinstance(cur_main, dict):
                    pass
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
                                if WEBP_STREAMING:
                                    # --- WEBP ---
                                    pic = webp.WebPPicture.from_numpy(frame)

                                    if WEBP_LOSSLESS:
                                        # Lossless mode ignores quality, sets lossless flag
                                        config = webp.WebPConfig.new(preset=webp.WebPPreset.PHOTO, quality=100)
                                        config.lossless = 1
                                    else:
                                        # Lossy mode
                                        config = webp.WebPConfig.new(preset=webp.WebPPreset.PHOTO, quality=WEBP_QUALITY)
                                        config.lossless = 0

                                    buf = pic.encode(config).buffer()
                                    exchange.set_frame(b'w' + bytes(buf))
                                else:
                                    # --- JPEG ---
                                    enc = jpeg.encode(frame, quality=JPEG_QUALITY, pixel_format=0)
                                    exchange.set_frame(b'j' + enc)

                            elif is_ascii:
                                # Fallback Realtime ASCII (if not using .npz files)
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
                if not HTTP_MONITOR and FRAME_COUNTER_DISPLAY: print(f"{last_actual_fps:.1f} FPS")

            # --- RESTORED MONITOR HOOKS (Standard Path) ---
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
            # ------------------------------

            if not is_headless and has_gl and glfw and glfw.window_should_close(window):
                state.run_mode = False

        if not is_headless and has_gl and glfw:
            glfw.terminate()