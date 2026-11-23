#image_display.py
import os
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
import threading
import atexit
from contextlib import nullcontext

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
import index_calculator
from folder_selector import update_folder_selection, folder_dictionary
from display_manager import DisplayState, display_init, _is_wayland_session, _hide_cursor_reliable, _move_wlrctl_offscreen_once
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


from image_loader import ImageLoader, FIFOImageBuffer, TOLERANCE
from image_preparation import advance_tick


class FIFOImageBufferPatched(FIFOImageBuffer):
    """main's display buffer, verbatim.

    Keyed by picture number only: get() returns the nearest loaded picture
    within TOLERANCE, whatever folder it was loaded from, so a folder switch
    or a late load never becomes a hole. The base class gained folder/exact
    matching for the modem source, which keeps using it; the display uses
    main's behaviour, which ran for years without misses.
    """

    def update(self, index, data_tuple):
        with self.lock:
            self.total_updates += 1
            dropped = False
            if len(self.queue) >= self.max_size:
                self.queue.popleft()
                self.dropped_count += 1
                dropped = True
            self.queue.append((index, data_tuple))
            return not dropped

    def get(self, current_index):
        with self.lock:
            while self.queue and abs(self.queue[0][0] - current_index) > TOLERANCE:
                self.queue.popleft()
            if not self.queue: return None
            best_idx, best_data = min(self.queue, key=lambda item: abs(item[0] - current_index))
            if abs(best_idx - current_index) <= TOLERANCE:
                return best_idx, best_data[0], best_data[1], best_data[2], best_data[3]
            return None


# -----------------------------------------------------------------------------
# WORKER FUNCTION (Runs in Background Threads)
# -----------------------------------------------------------------------------

def load_and_render_frame(loader, index, main_folder, float_folder,
                          source_aspect_ratio=None, jpeg_scaling_factor=None,
                          jpeg_min_size=None):
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
    m_img, f_img, m_sbs, f_sbs = loader.load_images(
        index, main_folder, float_folder,
        jpeg_scaling_factor=jpeg_scaling_factor,
        jpeg_min_size=jpeg_min_size)

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
            if getattr(settings, 'ASCII_WEB_SOURCE_SIZE', None):
                target_w, target_h = settings.ASCII_WEB_SOURCE_SIZE

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
            # Look up ANSI codes and combine
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

    # Monitor Counters
    fifo_miss_count = 0
    last_fifo_miss = -1
    last_displayed_index = 0
    shown_index = 0

    # Identify Modes
    is_ascii = getattr(settings, 'ASCII_MODE', False)
    server_mode = getattr(settings, 'SERVER_MODE', False)
    frame_publisher = None
    local_frame_port = getattr(settings, 'LOCAL_FRAME_PORT', None)
    if local_frame_port is not None:
        from local_frame_bridge import LatestFramePublisher
        frame_publisher = LatestFramePublisher(local_frame_port)
        atexit.register(frame_publisher.close)
    is_local_frame_source = frame_publisher is not None
    local_frame_rate = float(getattr(settings, 'LOCAL_FRAME_FPS', FPS) or 15)
    local_frame_period = 1.0 / max(local_frame_rate, .1)
    next_local_frame = 0.0

    def publish_local_frame(frame):
        nonlocal next_local_frame
        if frame_publisher is None:
            return
        now = time.monotonic()
        if now < next_local_frame:
            return
        # Which image this is and the app's clock: a modem sender stamps its
        # packets with them, so the receiver can tell the loop position.
        frame_publisher.publish(frame, {
            'index': shown_index, 'frames': png_paths_len,
            'pingpong': bool(PINGPONG),
            'clocked': not index_calculator.midi_mode,
            'launch_ns': index_calculator.launch_time,
            'shown_ns': time.time_ns()})
        next_local_frame = now + local_frame_period
    
    # Warn if both modes are set (configuration conflict)
    if server_mode and is_ascii:
        print("⚠️  WARNING: Both SERVER_MODE and ASCII_MODE are enabled in settings.py")
        print("   -> ASCII mode will take precedence, web mode will be disabled")
        print("   -> This may indicate a configuration error")
    
    is_web = server_mode and not is_ascii
    is_headless = is_web or is_ascii or is_local_frame_source
    # ASCII has its own rate knob: unlike MJPEG, its small text frames are not
    # constrained by the web capture bandwidth setting.
    capture_rate = (getattr(settings, "ASCII_FPS", 15) if is_ascii else
                    (1.0 / local_frame_period if is_local_frame_source else
                     SERVER_CAPTURE_RATE))
    capture_interval = 1.0 / capture_rate
    last_server_capture = time.time()
    last_captured_index = None
    # Headless web/ASCII loops have no local window between captures. Run them
    # at their output cadence instead of rendering extra frames that are never
    # published.
    loop_fps = FPS
    if (is_web or is_ascii or is_local_frame_source) and capture_rate and capture_rate > 0:
        loop_fps = min(FPS, capture_rate) if FPS and FPS > 0 else capture_rate

    # --- LOGGING ---
    if is_web:
        mode_str = "MJPEG"
        qual_str = JPEG_QUALITY
        print(f"[DISPLAY] Stream Encoder: {mode_str} ({qual_str})")
    # --------------------

    import make_file_lists
    baked = None
    if getattr(settings, 'LAYER_BAKE_DIR', None):
        from pathlib import Path
        if (Path(settings.LAYER_BAKE_DIR) / 'rgb_layers.json').exists():
            from ascii_rgb_layers import RgbWebLayers
            if get_config().get_mode() != MODE_ASCIIWEB:
                raise ValueError('RGB web samples are only supported in asciiweb mode')
            baked = RgbWebLayers(settings.LAYER_BAKE_DIR)
            try:
                baked.validate_settings()
            except Exception:
                baked.close()
                raise
        else:
            from baked_layers import BakedLayers
            baked = BakedLayers(settings.LAYER_BAKE_DIR)
        if is_web and baked.web_size and tuple(baked.web_size) != tuple(HEADLESS_RES):
            raise ValueError('Web bake target differs from HEADLESS_RES; rebuild with --web-size')
        main_folder_path, float_folder_path = baked.main_paths, baked.float_paths
        make_file_lists.initialize_frame_clock(clock_source, baked.frames)
    else:
        _, main_folder_path, float_folder_path = make_file_lists.initialize_image_lists(clock_source)
    main_folder_count = len(main_folder_path[0])
    float_folder_count = len(float_folder_path[0])
    png_paths_len = len(main_folder_path)

    # 2. Initial Dimension Logic
    temp_loader = baked if baked is not None else ImageLoader()
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

    # Half-size TurboJPEG decode emits one-quarter of the pixels. Use it only
    # when the source still covers the web target after scaling. Other modes
    # retain full-resolution decoding.
    web_jpeg_scale = None
    web_jpeg_min_size = None
    if is_web and source_image_size is not None:
        target_w, target_h = HEADLESS_RES
        if (source_image_size[0] >= 2 * target_w and
                source_image_size[1] >= 2 * target_h):
            web_jpeg_scale = (1, 2)
            # JPEG files are SBS: require each decoded half to remain at least
            # as large as the web target, checking every file's header.
            web_jpeg_min_size = (4 * target_w, 2 * target_h)

    rgb_samples = bool(getattr(baked, 'rgb_samples', False))
    if rgb_samples:
        # Source fitting and mirroring were already sampled offline. The tiny
        # live mix must cover the master grid exactly, without fitting it twice.
        state.image_size = (baked.size[0] * settings.ASCII_FONT_RATIO, baked.size[1])
        state.mirror = False

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

    if rgb_samples:
        if not has_gl or renderer.using_legacy_gl():
            baked.close()
            raise RuntimeError('RGB web samples require ModernGL; no lossy uint8 fallback')
        profile = 'gles-mediump' if renderer.ctx.version_code < 330 else 'desktop'
        if profile != baked.manifest['precision']:
            baked.close()
            raise ValueError('RGB web bake GL precision differs; rebuild on this backend')
        print('[ASCII] Compressed independent RGB samples; live grid-only alpha mix')

    def upload_layer(image, texture=None):
        if rgb_samples:
            from ascii_rgb_layers import upload_sample
            return upload_sample(renderer.ctx, image, texture)
        return (renderer.create_texture(image) if texture is None else
                renderer.update_texture(texture, image))

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

    loader = baked if baked is not None else ImageLoader()
    loader.set_paths(main_folder_path, float_folder_path)
    loader.set_png_paths_len(png_paths_len)

    # Pre-load using the NEW helper directly for the first frame
    initial_folders = folder_dictionary["Main_and_Float_Folders"]
    res0 = load_and_render_frame(
        loader, index, *initial_folders,
        source_aspect_ratio=source_aspect_ratio,
        jpeg_scaling_factor=web_jpeg_scale,
        jpeg_min_size=web_jpeg_min_size)

    # The sender consumes the composed source image, not the fullscreen
    # framebuffer. Keep this in source dimensions so monitor aspect ratio,
    # letterboxing, and display scaling never become encoded picture pixels.
    bridge_main, bridge_float, bridge_main_sbs, bridge_float_sbs = res0

    cur_main, cur_float, cur_m_sbs, cur_f_sbs = None, None, False, False

    main_texture = None
    float_texture = None
    # Only create textures if NOT ascii string and GL active
    ascii_grid_capture = None
    ascii_web_size = getattr(settings, 'ASCII_WEB_SOURCE_SIZE', None)
    if (is_ascii and has_gl and (getattr(settings, 'ASCII_CLIENT_SIZING', False) or ascii_web_size)
            and not renderer.using_legacy_gl()):
        from ascii_render import AsciiGridCapture
        ascii_grid_capture = AsciiGridCapture(renderer.ctx,
                                             sizes=(ascii_web_size,) if ascii_web_size else None)
        grid_description = (f'shared {ascii_web_size[0]}x{ascii_web_size[1]} browser master' if ascii_web_size else
                            '30x20, 45x30, 60x40, 90x60')
        print(f'[ASCII] Direct source-layer grids: {grid_description}; '
              f'colour blur={settings.ASCII_COLOR_BLUR}; no post-grid resize; '
              f'truecolour={settings.ASCII_TRUECOLOR}; tint brightness={settings.ASCII_COLOR_BRIGHTNESS}')
    if has_gl and not isinstance(res0[0], str):
        main_texture = upload_layer(res0[0])
        float_texture = upload_layer(res0[1])
    cur_main, cur_float, cur_m_sbs, cur_f_sbs = res0
    shown_index = last_displayed_index = index
    # Startup inspection must not pin a full-resolution pair for the session.
    del res0, img0, temp_loader

    def prepare_picture(wanted, folders):
        return load_and_render_frame(
            loader, wanted, *folders, source_aspect_ratio=source_aspect_ratio,
            jpeg_scaling_factor=web_jpeg_scale, jpeg_min_size=web_jpeg_min_size)

    # --- Loading: main's model, which ran for years without misses ---
    # A FIFO of loaded pictures keyed by picture number, filled by a thread
    # pool that is handed the next picture in the playback direction on every
    # picture change; the display takes the nearest loaded picture within
    # TOLERANCE. (The deadline preparer that replaced it missed ~14 pictures
    # a second on an eMMC tablet and a few a minute on servers.)
    fifo = FIFOImageBufferPatched(max_size=FIFO_LENGTH)
    fifo.update(index, (cur_main, cur_float, cur_m_sbs, cur_f_sbs))

    compensator = RollingIndexCompensator()

    def async_cb(fut, idx):
        try:
            fifo.update(idx, fut.result())
        except Exception as e:
            if monitor: monitor.record_load_error(idx, e)

    # I/O-bound work: more threads than cores, as in main.
    cpu_count = os.cpu_count() or 1
    max_workers = min(8, cpu_count + 2)

    def submit(pool, next_idx):
        folders = folder_dictionary["Main_and_Float_Folders"]
        future = pool.submit(prepare_picture, next_idx, folders)
        future.add_done_callback(lambda f, i=next_idx: async_cb(f, i))

    with (baked if baked is not None else nullcontext()), \
            ThreadPoolExecutor(max_workers=max_workers) as pool:
        next_idx = 1 if index == 0 else (index - 1 if index == png_paths_len - 1 else index + 1)
        submit(pool, next_idx)

        frame_times = deque(maxlen=60)
        ticker_period = (1.0 / loop_fps
                         if loop_fps and loop_fps > 0 else None)
        capture_period = ticker_period if is_headless else None
        next_tick = time.monotonic()

        def pace_iteration(iteration_start):
            """Wait for the next monotonic frame deadline; skip missed ticks."""
            nonlocal next_tick
            if ticker_period is not None:
                now = time.monotonic()
                # A visible window never holds after a late iteration: the
                # image index follows the wall clock, so waiting out a tick
                # that has already passed only drops the images due meanwhile
                # (a skip beats a hold). The swap-time test below cannot tell
                # this on Mesa, where swap_buffers returns at once and the
                # refresh wait lands in the next GL call (the texture upload);
                # holding there cut a Cherry Trail tablet and X11 machines
                # from ~52 fps to ~30 with images missed. Headless output
                # keeps its capture grid.
                next_tick, remaining = advance_tick(
                    next_tick, ticker_period, now,
                    swap_consumed=(not is_headless
                                   or local_swap_seconds >= ticker_period*.5))
                if remaining > 0:
                    time.sleep(remaining)
            return time.perf_counter() - iteration_start

        next_miss_log = 0.0
        first_render = True
        shown_folders = tuple(initial_folders)
        
        # Counter for periodic mouse hiding (every 60 frames ~= 2 seconds at 30fps)
        cursor_hide_counter = 0
        
        # Flag to ensure mouse jiggle only happens once after first frame render
        mouse_jiggled = False

        while (state.run_mode and not is_headless) or is_headless:
            iteration_start = time.perf_counter()
            successful_display = False
            view_dirty = False
            local_swap_seconds = 0.0

            if not is_headless and has_gl and glfw:
                glfw.poll_events()
                if not state.run_mode: break
                if getattr(state, 'needs_redraw', False):
                    view_dirty = True
                    state.needs_redraw = False

            if state.needs_update and not is_headless and has_gl:
                # Clear before reconfiguring so a resize callback raised by
                # macOS during the transition is not discarded afterward.
                state.needs_update = False
                view_dirty = True
                window = display_init(state)
                if window is not None:
                    _hide_cursor_reliable(window)

            prev = index
            index, _ = update_index(png_paths_len, PINGPONG)
            if index != prev:
                update_folder_selection(index, float_folder_count, main_folder_count)
            folders = tuple(folder_dictionary['Main_and_Float_Folders'])

            if index != prev:
                comp_idx = compensator.get_compensated_index(index)
                res = fifo.get(comp_idx)

                if res:
                    d_idx, m_img, f_img, m_sbs, f_sbs = res
                    compensator.update(index, d_idx)
                    previous_main, previous_float = cur_main, cur_float

                    # Update loop pointers. The picture chosen for this index
                    # (nearest within TOLERANCE) is what is shown for it, so
                    # headless capture follows the index as in main.
                    shown_index = index
                    shown_folders = folders
                    cur_main = m_img
                    cur_float = f_img
                    cur_m_sbs = m_sbs
                    cur_f_sbs = f_sbs
                    bridge_main = m_img
                    bridge_float = f_img
                    bridge_main_sbs = m_sbs
                    bridge_float_sbs = f_sbs

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
                        submit(pool, next_idx)

                        # Timing
                        dt = pace_iteration(iteration_start)
                        frame_times.append(dt)

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
                        if m_img is not previous_main:
                            main_texture = upload_layer(m_img, main_texture)
                        if f_img is not previous_float:
                            float_texture = upload_layer(f_img, float_texture)

                    successful_display = True
                    last_displayed_index = d_idx
                else:
                    if not is_headless and time.monotonic() >= next_miss_log:
                        print(f"[MISS] {index}")
                        next_miss_log = time.monotonic()+1.0
                    fifo_miss_count += 1
                    last_fifo_miss = index

                next_idx = 1 if index == 0 else (
                    index - 1 if index == png_paths_len - 1 else (index + 1 if index > prev else index - 1))
                submit(pool, next_idx)

            # Render (GL)
            render_now = has_gl and (first_render or is_local_frame_source or successful_display or
                                    view_dirty or renderer._quad_dirty)
            if render_now and ascii_grid_capture is None:
                if is_headless: window.use()
                renderer.overlay_images_single_pass(
                    main_texture, float_texture, BACKGROUND_COLOR,
                    main_is_sbs=cur_m_sbs, float_is_sbs=cur_f_sbs
                )
                if (frame_publisher is not None and has_gl and
                        time.monotonic() >= next_local_frame):
                    if (bridge_main is not None and
                            not isinstance(bridge_main, (str, dict))):
                        rendered_frame = renderer.composite_cpu(
                            bridge_main, bridge_float,
                            main_is_sbs=bridge_main_sbs,
                            float_is_sbs=bridge_float_sbs,
                            target_size=None)
                        if rendered_frame is not None:
                            publish_local_frame(rendered_frame)

            # Capture (Only for Images/Headless Web)
            should_capture = False
            if ((is_headless and not is_local_frame_source) or not has_gl):
                now = time.time()
                # Headless output is already paced by the monotonic ticker
                # below. Applying the same interval again against wall-clock
                # time can reject every other tick at a matched output rate.
                interval_elapsed = (capture_period is not None or
                                    now - last_server_capture > capture_interval)
                ascii_sizes = None
                if ascii_grid_capture is not None:
                    from ascii_terminal import pictures
                    ascii_sizes = (ascii_web_size,) if ascii_web_size else pictures.requested_scales()
                capture_identity = (index, shown_index, shown_folders, ascii_sizes)
                if capture_identity != last_captured_index and interval_elapsed:
                    should_capture = True
                    last_server_capture = now
                    last_captured_index = capture_identity

            if should_capture and shown_index == index and shown_folders == folders:
                # If m_img is a string, we already handled it.
                if ascii_grid_capture is not None:
                    grids = ascii_grid_capture.capture(
                        main_texture, float_texture, BACKGROUND_COLOR,
                        cur_m_sbs, cur_f_sbs, window, sizes=ascii_sizes)
                    if ascii_web_size:
                        text_frame = ascii_converter.to_ascii(grids[ascii_web_size],
                            width=ascii_web_size[0], height=ascii_web_size[1], sampled=True)
                        exchange_ascii.set_frame(text_frame)
                        exchange.set_frame(text_frame)
                    else:
                        pictures.publish(grids)
                elif isinstance(cur_main, str):
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
                                    if (frame.shape[1], frame.shape[0]) != HEADLESS_RES:
                                        frame = cv2.resize(
                                            frame, HEADLESS_RES,
                                            interpolation=cv2.INTER_LINEAR)
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
                            publish_local_frame(frame)
                            if is_web:
                                # Web only: use web exchange (and legacy for backward compat)
                                enc = jpeg.encode(frame, quality=JPEG_QUALITY, pixel_format=TJPF_RGB)
                                exchange_web.set_frame(b'j' + enc)
                                exchange.set_frame(b'j' + enc)  # Legacy compatibility
                            elif is_ascii:
                                if getattr(settings, 'ASCII_CLIENT_SIZING', False):
                                    from ascii_terminal import pictures
                                    pictures.publish(frame)
                                if ascii_web_size:
                                    text_frame = ascii_converter.to_ascii(frame,
                                        width=ascii_web_size[0], height=ascii_web_size[1],
                                        sampled=has_gl)
                                else:
                                    text_frame = ascii_converter.to_ascii(frame)
                                exchange_ascii.set_frame(text_frame)
                                exchange.set_frame(text_frame)  # Legacy compatibility

                    except Exception as e:
                        print(f"[CAPTURE ERROR] {e}")

            if not is_headless and render_now and glfw:
                swap_started = time.monotonic()
                glfw.swap_buffers(window)
                local_swap_seconds = time.monotonic()-swap_started
                if glfw.window_should_close(window):
                    state.run_mode = False
                
                # Jiggle mouse after first frame render to help compositors grab focus (especially Wayland)
                # This must happen after swap_buffers when window is visible and has focus
                if not mouse_jiggled:
                    from event_handler import jiggle_mouse_for_focus
                    jiggle_mouse_for_focus(window)
                    mouse_jiggled = True
                    # One-time wlrctl move to bottom-left offscreen (Linux/Wayland)
                    _move_wlrctl_offscreen_once(window)
                
                # Periodically hide cursor to ensure it stays hidden across different compositors
                # Every 60 frames (~2 seconds at 30fps)
                cursor_hide_counter += 1
                if cursor_hide_counter >= 60:
                    _hide_cursor_reliable(window)
                    cursor_hide_counter = 0

            if render_now:
                first_render = False

            dt = pace_iteration(iteration_start)
            frame_times.append(dt)

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
                    "local_swap_ms": local_swap_seconds*1000,
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
        if ascii_grid_capture is not None:
            ascii_grid_capture.close()
        if is_headless and has_gl and window is not None and hasattr(window, "close"):
            try:
                window.close()
            except Exception as e:
                print(f"[DISPLAY] Headless window cleanup failed: {e}")
        if frame_publisher is not None:
            frame_publisher.close()
