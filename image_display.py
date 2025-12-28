#image_display.py
import os
import time
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from collections import deque
import threading

import numpy as np
from turbojpeg import TJPF_RGB
from turbojpeg_loader import get_turbojpeg

import settings
from shared_state import exchange, exchange_web, exchange_ascii
from server_config import get_config, MODE_ASCII, MODE_ASCIIWEB

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

# Initialize Encoders (strict TurboJPEG; will raise if native lib is missing)
jpeg = get_turbojpeg()

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


# Process-compatible function for ASCII string building (CPU-bound work)
def _build_ascii_string_process(final_chars, final_colors, ansi_lut, reset_code):
    """
    CPU-bound ASCII string building in a separate process to avoid GIL.
    This function must be at module level for ProcessPoolExecutor.
    """
    import numpy as np
    # Look up ANSI codes and combine
    color_strings = ansi_lut[final_colors]
    
    # Optimize string building: avoid full array conversion, build strings efficiently
    if final_chars.dtype.kind == 'S':
        # Bytes array: decode row by row to avoid full array conversion
        rows = []
        for i in range(final_chars.shape[0]):
            # Decode bytes row to string once
            row_chars = final_chars[i].tobytes().decode('latin-1')  # Fast decode
            row_colors = color_strings[i]
            # Build string efficiently: join ANSI codes with chars
            row_str = ''.join(c + ch for c, ch in zip(row_colors, row_chars))
            rows.append(row_str)
    else:
        # Already string/unicode array - use numpy char operations (optimized)
        image_grid = np.char.add(color_strings, final_chars)
        # Optimize: use list comprehension but avoid nested joins
        rows = [''.join(row) for row in image_grid]
    
    return "\r\n".join(rows) + reset_code


class FIFOImageBufferPatched(FIFOImageBuffer):
    """Patched version for backward compatibility - base class now has current_depth()."""
    pass


# -----------------------------------------------------------------------------
# WORKER FUNCTION (Runs in Background Threads)
# -----------------------------------------------------------------------------

def load_and_render_frame(loader, index, main_folder, float_folder, source_aspect_ratio=None):
    """
    Loads images AND performs the heavy ASCII merging/string-generation
    in the background thread. Returns the final string if ASCII.
    
    Args:
        loader: ImageLoader instance
        index: Frame index
        main_folder: Main folder path
        float_folder: Float folder path
        source_aspect_ratio: Source image aspect ratio (w/h) for consistent scaling
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

            # --- RESAMPLE USING COVER/CROP (Fill and Trim) ---
            baked_h, baked_w = final_chars.shape
            target_w = max(1, int(getattr(settings, 'ASCII_WIDTH', baked_w)))
            target_h = max(1, int(getattr(settings, 'ASCII_HEIGHT', baked_h)))

            # Scale enough to cover the target grid, then crop the center
            scale_x = target_w / baked_w if baked_w > 0 else 1.0
            scale_y = target_h / baked_h if baked_h > 0 else 1.0
            scale = max(scale_x, scale_y)

            scaled_w = max(target_w, int(round(baked_w * scale)))
            scaled_h = max(target_h, int(round(baked_h * scale)))

            # Resample if needed
            if scaled_w != baked_w or scaled_h != baked_h:
                row_idx = np.linspace(0, baked_h - 1, scaled_h).astype(np.int32)
                col_idx = np.linspace(0, baked_w - 1, scaled_w).astype(np.int32)
                final_chars = final_chars[np.ix_(row_idx, col_idx)]
                final_colors = final_colors[np.ix_(row_idx, col_idx)]

            # Center-crop to the exact output dimensions
            x_off = max(0, (final_chars.shape[1] - target_w) // 2)
            y_off = max(0, (final_chars.shape[0] - target_h) // 2)
            final_chars = final_chars[y_off : y_off + target_h, x_off : x_off + target_w]
            final_colors = final_colors[y_off : y_off + target_h, x_off : x_off + target_w]

            # --- RENDER TO STRING (OPTIMIZED) ---
            # CPU-bound string building - can use process pool for better parallelism
            # Note: Process pool is optional and only used if configured
            # For now, do it in-thread (numpy operations release GIL, so most work is parallel)
            # Process pool can be enabled via USE_PROCESS_POOL_FOR_ASCII setting
            color_strings = ANSI_LUT[final_colors]
            
            # Optimize string building: avoid full array conversion, build strings efficiently
            if final_chars.dtype.kind == 'S':
                # Bytes array: decode row by row to avoid full array conversion
                # Pre-allocate list for rows (faster than list comprehension with join)
                rows = []
                for i in range(final_chars.shape[0]):
                    # Decode bytes row to string once
                    row_chars = final_chars[i].tobytes().decode('latin-1')  # Fast decode
                    row_colors = color_strings[i]
                    # Build string efficiently: join ANSI codes with chars
                    row_str = ''.join(c + ch for c, ch in zip(row_colors, row_chars))
                    rows.append(row_str)
            else:
                # Already string/unicode array - use numpy char operations (optimized)
                image_grid = np.char.add(color_strings, final_chars)
                # Optimize: use list comprehension but avoid nested joins
                rows = [''.join(row) for row in image_grid]
            
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
    last_actual_fps = 0.0  # Initialize to 0, will be calculated from actual frame times

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

    import make_file_lists
    _, main_folder_path, float_folder_path = make_file_lists.initialize_image_lists(clock_source)
    main_folder_count = len(main_folder_path[0])
    float_folder_count = len(float_folder_path[0])
    png_paths_len = len(main_folder_path)

    # 2. Initial Dimension Logic
    temp_loader = ImageLoader()
    temp_loader.set_paths(main_folder_path, float_folder_path)

    img0, is_sbs0 = temp_loader.read_image(main_folder_path[0][0])
    if img0 is None:
        raise RuntimeError("Failed to load initial image.")

    # Store original source dimensions for aspect ratio calculation (ASCII mode)
    source_image_size = None
    source_aspect_ratio = None
    
    # ASCII Dict detection for initialization
    if isinstance(img0, dict):
        h, w = img0['chars'].shape
        state.image_size = (w, h)
        has_gl = False
        is_headless = True
        source_image_size = (w, h)  # Store for aspect ratio
    else:
        h, w = img0.shape[:2]
        if is_sbs0:
            state.image_size = (w // 2, h)
            source_image_size = (w // 2, h)  # Store actual image size (not SBS)
        else:
            state.image_size = (w, h)
            source_image_size = (w, h)  # Store for aspect ratio
    
    # Calculate source aspect ratio from first frame (for ASCII modes)
    # This ensures consistent aspect ratio across all frames
    if source_image_size is not None:
        src_w, src_h = source_image_size
        source_aspect_ratio = src_w / src_h if src_h > 0 else 1.0

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

    def _get_screen_size(window_obj):
        if window_obj is None:
            return None
        if hasattr(window_obj, "size"):
            return window_obj.size
        if glfw:
            try:
                monitor = glfw.get_window_monitor(window_obj) or glfw.get_primary_monitor()
                if monitor:
                    mode = glfw.get_video_mode(monitor)
                    if mode:
                        return (mode.size.width, mode.size.height)
                return glfw.get_framebuffer_size(window_obj)
            except Exception:
                return None
        return None

    screen_size = _get_screen_size(window)
    if screen_size is None and source_image_size is not None:
        src_w, src_h = source_image_size
        if is_ascii:
            screen_size = (
                getattr(settings, "ASCII_WIDTH", src_w),
                getattr(settings, "ASCII_HEIGHT", src_h),
            )
        elif is_headless:
            screen_size = HEADLESS_RES

    if source_image_size is not None:
        src_w, src_h = source_image_size
        if screen_size is not None:
            scr_w, scr_h = screen_size
            print(f"[DISPLAY] Source image: {src_w}x{src_h} | Screen: {scr_w}x{scr_h}")
            
            # Debug: Aspect ratio calculations
            if is_ascii:
                font_ratio = getattr(settings, 'ASCII_FONT_RATIO', 0.5)
                image_aspect = src_w / src_h if src_h > 0 else 1.0
                terminal_aspect = (scr_w * font_ratio) / scr_h if scr_h > 0 else 1.0
                print(f"[DISPLAY] Image aspect ratio: {image_aspect:.4f} | Terminal aspect ratio: {terminal_aspect:.4f} (font_ratio={font_ratio})")
        else:
            print(f"[DISPLAY] Source image: {src_w}x{src_h} | Screen: unknown")

    # 4. Loader & Buffer Init
    index, _ = update_index(png_paths_len, PINGPONG)
    
    update_folder_selection(index, float_folder_count, main_folder_count)

    loader = ImageLoader()
    loader.set_paths(main_folder_path, float_folder_path)
    loader.set_png_paths_len(png_paths_len)

    fifo = FIFOImageBufferPatched(max_size=FIFO_LENGTH)

    # Pre-load using the NEW helper directly for the first frame
    initial_folders = folder_dictionary["Main_and_Float_Folders"]
    res0 = load_and_render_frame(loader, index, *initial_folders, source_aspect_ratio=source_aspect_ratio)
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

    # 5. Thread Pool for I/O-bound work
    # Optimize worker count for I/O-bound image loading
    # I/O-bound tasks can use more threads than CPU cores
    cpu_count = os.cpu_count() or 1
    # Conservative formula that works well across all devices
    # Reverted from aggressive tiered scaling to universal formula
    max_workers = min(8, cpu_count + 2)

    # Process pool for CPU-bound ASCII work (optional, opt-in only)
    # Use ProcessPoolExecutor for ASCII string building to avoid GIL limitations
    # Disabled by default - users can opt-in via settings if needed
    use_process_pool_for_ascii = getattr(settings, 'USE_PROCESS_POOL_FOR_ASCII', False)
    ascii_process_pool = None
    if use_process_pool_for_ascii and is_ascii:
        ascii_process_pool = ProcessPoolExecutor(max_workers=min(4, cpu_count))

    try:
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            next_idx = 1 if index == 0 else (index - 1 if index == png_paths_len - 1 else index + 1)
            # Use the NEW worker function
            folders = folder_dictionary["Main_and_Float_Folders"]
            future = pool.submit(load_and_render_frame, loader, next_idx, *folders, source_aspect_ratio=source_aspect_ratio)
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
                            # Pre-baked ASCII should only occur in ASCII mode, but handle edge cases
                            if is_ascii:
                                # ASCII mode: send to ASCII exchange
                                exchange_ascii.set_frame(m_img)
                            # Legacy exchange for backward compatibility (only if not in web mode)
                            if not is_web:
                                exchange.set_frame(m_img)
                            successful_display = True
                            last_displayed_index = d_idx

                            # Trigger Next Load
                            next_idx = 1 if index == 0 else (
                                index - 1 if index == png_paths_len - 1 else (index + 1 if index > prev else index - 1))
                            folders = folder_dictionary["Main_and_Float_Folders"]
                            future = pool.submit(load_and_render_frame, loader, next_idx, *folders, source_aspect_ratio=source_aspect_ratio)
                            future.add_done_callback(lambda f, i=next_idx: async_cb(f, i))

                            # Timing
                            now = time.perf_counter()
                            dt = now - frame_start
                            frame_times.append(dt)
                            frame_start = now
                            if FPS:
                                s = (1.0 / FPS) - dt
                                if s > 0: time.sleep(s)

                            # Calculate FPS from frame times (handle both single and multiple frames)
                            if len(frame_times) >= 1:
                                if len(frame_times) == 1:
                                    # Single frame: use that frame's time directly
                                    last_actual_fps = 1.0 / dt if dt > 0 else 0.0
                                else:
                                    # Multiple frames: use average frame time
                                    avg_frame_time = sum(frame_times) / len(frame_times)
                                    last_actual_fps = 1.0 / avg_frame_time if avg_frame_time > 0 else 0.0

                            if monitor:
                                fd = folder_dictionary
                                monitor.update({
                                    "index": index,
                                    "displayed": last_displayed_index,
                                    "fps": last_actual_fps,
                                    "fifo_depth": fifo.current_depth(),
                                    "successful_frame": True,
                                    "main_folder": fd["Main_and_Float_Folders"][0],
                                    "float_folder": fd["Main_and_Float_Folders"][1],
                                    "rand_mult": fd.get("rand_mult"),
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
                    folders = folder_dictionary["Main_and_Float_Folders"]
                    future = pool.submit(load_and_render_frame, loader, next_idx, *folders, source_aspect_ratio=source_aspect_ratio)
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
                                # Only HeadlessWindow has .fbo and .size attributes
                                # In windowed mode, window is a raw glfw object without these attributes
                                # This should only execute in headless mode (should_capture check above ensures this)
                                if hasattr(window, 'fbo') and hasattr(window, 'size'):
                                    raw = window.fbo.read(components=3)
                                    w_fbo, h_fbo = window.size
                                    frame = np.frombuffer(raw, dtype=np.uint8).reshape((h_fbo, w_fbo, 3))
                                    # In web mode, resize to HEADLESS_RES
                                    if is_web and frame is not None:
                                        import cv2
                                        frame = cv2.resize(frame, HEADLESS_RES, interpolation=cv2.INTER_LINEAR)
                                else:
                                    # Windowed mode - shouldn't reach here due to should_capture logic
                                    # But add safety check to prevent AttributeError
                                    print("[WARNING] Attempted FBO read in windowed mode - skipping capture")
                            else:
                                # Use HEADLESS_RES only for web rendering
                                # ASCII mode uses its own ASCII_WIDTH/ASCII_HEIGHT sizing.
                                if is_web:
                                    tgt_size = HEADLESS_RES
                                else:
                                    tgt_size = None  # Local mode: full resolution
                                
                                frame = renderer.composite_cpu(
                                    cur_main, cur_float,
                                    main_is_sbs=cur_m_sbs, float_is_sbs=cur_f_sbs,
                                    target_size=tgt_size
                                )

                            if frame is not None:
                                if is_web:
                                    # Web only: use web exchange (and legacy for backward compat)
                                    enc = jpeg.encode(frame, quality=JPEG_QUALITY, pixel_format=TJPF_RGB)
                                    exchange_web.set_frame(b'j' + enc)
                                    exchange.set_frame(b'j' + enc)  # Legacy compatibility
                                elif is_ascii:
                                    text_frame = ascii_converter.to_ascii(frame)
                                    exchange_ascii.set_frame(text_frame)
                                    exchange.set_frame(text_frame)  # Legacy compatibility

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

                # Calculate FPS from frame times (handle both single and multiple frames)
                if len(frame_times) >= 1:
                    if len(frame_times) == 1:
                        # Single frame: use that frame's time directly
                        last_actual_fps = 1.0 / dt if dt > 0 else 0.0
                    else:
                        # Multiple frames: use average frame time
                        avg_frame_time = sum(frame_times) / len(frame_times)
                        last_actual_fps = 1.0 / avg_frame_time if avg_frame_time > 0 else 0.0
                    #print(f"{last_actual_fps:.1f} FPS")

                if monitor:
                    fd = folder_dictionary
                    monitor.update({
                        "index": index,
                        "displayed": last_displayed_index,
                        "fps": last_actual_fps,
                        "fifo_depth": fifo.current_depth(),
                        "successful_frame": successful_display,
                        "main_folder": fd["Main_and_Float_Folders"][0],
                        "float_folder": fd["Main_and_Float_Folders"][1],
                        "rand_mult": fd.get("rand_mult"),
                        "main_folder_count": main_folder_count,
                        "float_folder_count": float_folder_count,
                        "fifo_miss_count": fifo_miss_count,
                        "last_fifo_miss": last_fifo_miss
                    })

                if not is_headless and has_gl and glfw and glfw.window_should_close(window):
                    state.run_mode = False

        if not is_headless and has_gl and glfw:
            glfw.terminate()
        if is_headless and has_gl and window is not None and hasattr(window, "close"):
            try:
                window.close()
            except Exception as e:
                print(f"[DISPLAY] Headless window cleanup failed: {e}")
    finally:
        if ascii_process_pool is not None:
            ascii_process_pool.shutdown(wait=True)

