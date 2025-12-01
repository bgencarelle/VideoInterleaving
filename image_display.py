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

# Ensure PyOpenGL checks are off (not needed with ModernGL, but leaving for safety)
os.environ['PYOPENGL_ERROR_CHECKING'] = '0'

# Only import glfw when we actually do a local window
if not settings.SERVER_MODE:
    import glfw

from settings import (
    FULLSCREEN_MODE,
    PINGPONG,
    FPS,
    FRAME_COUNTER_DISPLAY,
    SHOW_DELTA,
    TEST_MODE,
    HTTP_MONITOR,
    CLOCK_MODE,
    FIFO_LENGTH,
    BACKGROUND_COLOR,
)
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


def cpu_composite_frame(main_img, float_img):
    """
    Pure-CPU compositing for headless mode:

    - Take main_img and float_img (both HxWxC, uint8).
    - Treat them as RGBA if they have 4 channels, or RGB/GRAY with full alpha.
    - Composite:
        BACKGROUND_COLOR  ->  main  ->  float
      using per-pixel alpha.
    - Return a 3-channel BGR frame, same resolution as the source.

    No resizing. Browser handles scaling and letterboxing.
    """
    if main_img is None:
        return None

    # --- Helpers: normalize to RGBA (RGB order) -----------------------------
    def to_rgba(img):
        if img is None:
            return None

        # 4-channel: assume RGBA as produced for ModernGL
        if img.ndim == 3 and img.shape[2] == 4:
            return img

        # 3-channel: assume RGB-like; give it full alpha
        if img.ndim == 3 and img.shape[2] == 3:
            h, w, _ = img.shape
            alpha = np.full((h, w, 1), 255, dtype=np.uint8)
            return np.concatenate([img, alpha], axis=2)

        # grayscale: expand to RGB + alpha
        if img.ndim == 2:
            rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
            h, w, _ = rgb.shape
            alpha = np.full((h, w, 1), 255, dtype=np.uint8)
            return np.concatenate([rgb, alpha], axis=2)

        raise ValueError(f"Unsupported image shape in to_rgba: {img.shape!r}")

    m_rgba = to_rgba(main_img)
    f_rgba = to_rgba(float_img) if float_img is not None else None

    h, w, _ = m_rgba.shape

    # If float exists but size differs, ignore it (cheaper than resizing)
    if f_rgba is not None and f_rgba.shape != m_rgba.shape:
        f_rgba = None

    # --- Build background as RGB in uint16 space ---------------------------
    # BACKGROUND_COLOR is [R, G, B] in 0..255
    bg_r, bg_g, bg_b = BACKGROUND_COLOR
    bg = np.empty((h, w, 3), dtype=np.uint16)
    bg[..., 0] = bg_r
    bg[..., 1] = bg_g
    bg[..., 2] = bg_b

    # Extract main color/alpha in uint16
    m_rgb = m_rgba[..., :3].astype(np.uint16)   # 0..255
    m_a   = m_rgba[..., 3:4].astype(np.uint16)  # 0..255, shape (H, W, 1)

    # main over background: base = m * a + bg * (255 - a)
    inv_m_a = 255 - m_a
    base_rgb = (m_rgb * m_a + bg * inv_m_a + 127) // 255  # +127 for rounding

    # float over base (if present)
    if f_rgba is not None:
        f_rgb = f_rgba[..., :3].astype(np.uint16)
        f_a   = f_rgba[..., 3:4].astype(np.uint16)
        inv_f_a = 255 - f_a
        out_rgb = (f_rgb * f_a + base_rgb * inv_f_a + 127) // 255
    else:
        out_rgb = base_rgb

    # Back to uint8 RGB
    out_rgb = np.clip(out_rgb, 0, 255).astype(np.uint8)

    # OpenCV/JPEG expect BGR, so swap channels
    out_bgr = out_rgb[..., ::-1]

    return out_bgr



def run_display(clock_source=CLOCK_MODE):
    state = DisplayState()
    state.fullscreen = FULLSCREEN_MODE
    last_actual_fps = FPS

    # --- Server capture timing (HEADLESS / SERVER_MODE only) ---
    last_server_capture = time.time()
    capture_rate = getattr(settings, "SERVER_CAPTURE_RATE", FPS or 10)
    capture_interval = 1.0 / capture_rate

    # --- Initialize folders & image size ---
    import calculators
    _, main_folder_path, float_folder_path = calculators.init_all(clock_source)
    main_folder_count = len(main_folder_path[0])
    float_folder_count = len(float_folder_path[0])
    png_paths_len = len(main_folder_path)

    # determine initial image size from first main image
    first_image_path = main_folder_path[0][0]
    img0 = cv2.imread(first_image_path, cv2.IMREAD_UNCHANGED)
    if img0 is None:
        raise RuntimeError(f"Unable to load image: {first_image_path}")
    height, width = img0.shape[:2]
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

    # --- Setup Display / GL usage ---
    # In SERVER_MODE we go pure CPU: no GL, no Xvfb, no moderngl context.
    if settings.SERVER_MODE:
        window = None
        has_gl = False
        print("[HEADLESS] SERVER_MODE=True, using CPU-only compositing (no GL, no Xvfb).")
    else:
        # Local mode: create an actual GL window
        window = display_init(state)
        has_gl = True
        register_callbacks(window, state)

    # initial index & folder selection
    index, _ = update_index(png_paths_len, PINGPONG)
    last_index = index
    update_folder_selection(index, float_folder_count, main_folder_count)

    fifo_buffer = FIFOImageBufferPatched(max_size=FIFO_LENGTH)
    main_folder, float_folder = folder_dictionary["Main_and_Float_Folders"]
    main_image, float_image = image_loader.load_images(index, main_folder, float_folder)
    fifo_buffer.update(index, (main_image, float_image))

    # track current images for CPU compositing
    current_main_img = main_image
    current_float_img = float_image

    # JPEG encode params reused every frame
    encode_params = [
        int(cv2.IMWRITE_JPEG_QUALITY),
        getattr(settings, "JPEG_QUALITY", 80),
    ]

    # If you ever run local GL mode, we can still keep textures alive
    if has_gl:
        import renderer  # ensure GL renderer is available locally
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
    with ThreadPoolExecutor(max_workers=6) as executor:
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

            # --- 1. Event Polling (Local Only) ---
            if not settings.SERVER_MODE and has_gl:
                import glfw
                glfw.poll_events()
                if not state.run_mode:
                    break

            # --- 2. Reinit Check (local, GL) ---
            if state.needs_update and not settings.SERVER_MODE and has_gl:
                display_init(state)
                state.needs_update = False

            # --- 3. Update Index & Images ---
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

                    if SHOW_DELTA:
                        d = index - displayed_index
                        off = compensated - index
                        print(f"[DEBUG] idx={index}, disp={displayed_index}, Δ={d}, offset={off}")

                    if has_gl:
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

            # --- 4. Drawing (local GL only) ---
            if has_gl and not settings.SERVER_MODE:
                window.use()
                renderer.overlay_images_two_pass_like_old(
                    main_texture, float_texture, background_color=BACKGROUND_COLOR
                )

            # --- 5. Server Capture / Streaming (CPU-only in SERVER_MODE) ---
            if settings.SERVER_MODE:
                now = time.time()
                if now - last_server_capture > capture_interval:
                    last_server_capture = now
                    try:
                        frame = cpu_composite_frame(current_main_img, current_float_img)
                        if frame is not None:
                            ok, jpg_bytes = cv2.imencode(".jpg", frame, encode_params)
                            if ok:
                                exchange.set_frame(jpg_bytes.tobytes())
                            else:
                                print("[CAPTURE] cv2.imencode failed (CPU path).")
                    except Exception as e:
                        print(f"[CAPTURE ERROR] {e}")

            # --- 6. Swap Buffers (local GL only) ---
            if not settings.SERVER_MODE and has_gl:
                import glfw
                glfw.swap_buffers(window)

            # --- 7. Timing ---
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

            # rolling FPS print
            if len(frame_times) > 1:
                avg = sum(frame_times) / len(frame_times)
                last_actual_fps = 1.0 / avg if avg > 0 else 0.0
                if not HTTP_MONITOR and FRAME_COUNTER_DISPLAY:
                    print(f"{last_actual_fps:.2f} FPS")
                    if last_actual_fps < FPS - 2:
                        print(f" Frame drop! Target {FPS}, Got {last_actual_fps:.2f}")

            # update HTTP monitor
            if monitor:
                monitor.update(
                    {
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
                    }
                )

            # exit on window close (Local Only)
            if not settings.SERVER_MODE and has_gl:
                import glfw
                if glfw.window_should_close(window):
                    state.run_mode = False

        # local GL teardown
        if not settings.SERVER_MODE and has_gl:
            import glfw
            glfw.terminate()
