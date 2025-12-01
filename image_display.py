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


def cpu_composite_frame(main_img, float_img, out_size):
    """
    CPU fallback for headless mode:

    - Reproduces the old two-pass blending:
        clear to BACKGROUND_COLOR
        draw main with its alpha
        draw float with its alpha on top
    - Preserves aspect ratio.
    - Letter/pillarboxes into out_size.
    - Ensures 3-channel BGR for JPEG encoding.
    """
    if main_img is None:
        return None

    # --- helpers -------------------------------------------------------------
    def ensure_rgba(img):
        """Ensure image is uint8 RGBA (we treat incoming 4ch as RGBA)."""
        if img is None:
            return None

        if img.ndim == 3 and img.shape[2] == 4:
            # Assume already RGBA (this is what ModernGL path uses)
            return img

        if img.ndim == 3 and img.shape[2] == 3:
            # 3-channel: add opaque alpha
            h, w = img.shape[:2]
            alpha = np.full((h, w, 1), 255, dtype=img.dtype)
            return np.concatenate([img, alpha], axis=2)

        if img.ndim == 2:
            # grayscale -> RGB -> add alpha
            rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
            h, w = rgb.shape[:2]
            alpha = np.full((h, w, 1), 255, dtype=rgb.dtype)
            return np.concatenate([rgb, alpha], axis=2)

        # Fallback: try to force into 4 channels via OpenCV
        if img.ndim == 3:
            # try treat as BGR, convert to BGRA, then reorder to RGBA
            bgra = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
            # reorder BGRA -> RGBA
            return bgra[..., [2, 1, 0, 3]]

        raise ValueError("Unsupported image shape in cpu_composite_frame: "
                         f"{img.shape!r}")

    # --- ensure we have RGBA for both passes ---------------------------------
    main_rgba = ensure_rgba(main_img)
    float_rgba = ensure_rgba(float_img) if float_img is not None else None

    h0, w0, _ = main_rgba.shape

    # --- build background in linear-ish RGB ----------------------------------
    # BACKGROUND_COLOR is 0–255 RGB
    bg_r, bg_g, bg_b = [c / 255.0 for c in BACKGROUND_COLOR]
    bg_rgb = np.array([bg_r, bg_g, bg_b], dtype=np.float32)
    bg_rgb = np.broadcast_to(bg_rgb, (h0, w0, 3))

    # convert to float [0,1]
    m_rgb = main_rgba[..., :3].astype(np.float32) / 255.0
    m_a   = main_rgba[..., 3:4].astype(np.float32) / 255.0  # keep as (H,W,1)

    # First pass: main over background
    # out = m.rgb * m.a + bg * (1 - m.a)
    m_comp_rgb = m_rgb * m_a + bg_rgb * (1.0 - m_a)
    m_comp_a   = m_a + (1.0 - m_a) * 0.0  # alpha of main pass (not super important)

    # Second pass: float over main_comp (if present and same size)
    if float_rgba is not None and float_rgba.shape == main_rgba.shape:
        f_rgb = float_rgba[..., :3].astype(np.float32) / 255.0
        f_a   = float_rgba[..., 3:4].astype(np.float32) / 255.0

        # standard "over" compositing:
        # out = f.rgb * f.a + m_comp * (1 - f.a)
        out_rgb = f_rgb * f_a + m_comp_rgb * (1.0 - f_a)
        out_a   = f_a + m_comp_a * (1.0 - f_a)
    else:
        out_rgb = m_comp_rgb
        out_a   = m_comp_a

    # pack back into RGBA uint8
    out_rgba = np.clip(
        np.concatenate([out_rgb, out_a], axis=2) * 255.0,
        0,
        255,
    ).astype(np.uint8)

    # --- convert RGBA -> BGR for OpenCV/JPEG ---------------------------------
    out_bgr = cv2.cvtColor(out_rgba, cv2.COLOR_RGBA2BGR)

    # --- letterbox / pillarbox into target canvas ----------------------------
    out_w, out_h = out_size
    h, w = out_bgr.shape[:2]

    if w == 0 or h == 0 or out_w == 0 or out_h == 0:
        return None

    scale = min(out_w / w, out_h / h)
    new_w = int(w * scale)
    new_h = int(h * scale)

    resized = cv2.resize(out_bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)

    canvas = np.zeros((out_h, out_w, 3), dtype=np.uint8)
    x0 = (out_w - new_w) // 2
    y0 = (out_h - new_h) // 2
    canvas[y0:y0 + new_h, x0:x0 + new_w] = resized

    return canvas



def run_display(clock_source=CLOCK_MODE):
    state = DisplayState()
    state.fullscreen = FULLSCREEN_MODE
    last_actual_fps = FPS

    # --- Server capture timing (HEADLESS / SERVER_MODE only) ---
    last_server_capture = time.time()
    capture_rate = getattr(settings, "SERVER_CAPTURE_RATE", 5)  # encodes/sec
    capture_interval = 1.0 / capture_rate

    # --- Initialize folders & image size ---
    import calculators
    _, main_folder_path, float_folder_path = calculators.init_all(clock_source)
    main_folder_count = len(main_folder_path[0])
    float_folder_count = len(float_folder_path[0])
    png_paths_len = len(main_folder_path)

    # determine initial image size from the first frame
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

    # --- Setup Display (Window or Headless) ---
    window = display_init(state)

    if settings.SERVER_MODE:
        # In SERVER_MODE, window is None when HEADLESS_USE_GL=False
        has_gl = window is not None
    else:
        has_gl = True
        # Local mode: we actually have a window and events
        register_callbacks(window, state)

    if not settings.SERVER_MODE and window is None:
        raise RuntimeError("Local mode requires a GL window, but display_init returned None.")

    # initial index & folder selection
    index, _ = update_index(png_paths_len, PINGPONG)
    last_index = index
    last_captured_index = None  # track what we've already encoded for streaming

    update_folder_selection(index, float_folder_count, main_folder_count)

    fifo_buffer = FIFOImageBufferPatched(max_size=FIFO_LENGTH)
    main_folder, float_folder = folder_dictionary["Main_and_Float_Folders"]
    main_image, float_image = image_loader.load_images(index, main_folder, float_folder)
    fifo_buffer.update(index, (main_image, float_image))

    # track current displayed images (for CPU compositing path)
    current_main_img = main_image
    current_float_img = float_image

    # Push an initial frame into the exchange so /video_feed shows something immediately
    try:
        headless_res = getattr(settings, "HEADLESS_RES", (state.image_size[0], state.image_size[1]))
        initial_frame = cpu_composite_frame(main_image, float_image, headless_res)
        if initial_frame is not None:
            encode_params = [
                int(cv2.IMWRITE_JPEG_QUALITY),
                getattr(settings, "JPEG_QUALITY", 70),
            ]
            ok, jpg_bytes = cv2.imencode(".jpg", initial_frame, encode_params)
            if ok:
                print("[TEST] Pushing initial frame to FrameExchange.")
                exchange.set_frame(jpg_bytes.tobytes())
            else:
                print("[TEST] cv2.imencode failed on initial frame.")
    except Exception as e:
        print(f"[TEST] Initial frame encode error: {e}")

    # create textures *only* if we have GL (Mac/dev)
    if has_gl:
        main_texture = renderer.create_texture(main_image)
        float_texture = renderer.create_texture(float_image)
    else:
        main_texture = None
        float_texture = None

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
    max_workers = min(4, (os.cpu_count() or 1))
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
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
                glfw.poll_events()
                if not state.run_mode:
                    break

            # --- 2. Reinit Check ---
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

            # --- 4. Drawing / compositing ---
            if has_gl and not settings.SERVER_MODE:
                # Local mode (Mac): draw to real window
                renderer.overlay_images_two_pass_like_old(
                    main_texture, float_texture, background_color=BACKGROUND_COLOR
                )

            # --- 5. Server Capture / Streaming (VPS / SERVER_MODE) ---
            if settings.SERVER_MODE:
                now = time.time()

                # Only encode when the frame actually changes, and not more often than SERVER_CAPTURE_RATE
                if (index != last_captured_index) and (now - last_server_capture >= capture_interval):
                    last_server_capture = now
                    last_captured_index = index

                    try:
                        if has_gl:
                            # GL FBO capture path (if you ever enable HEADLESS_USE_GL=True)
                            raw_data = window.fbo.read(components=3)
                            w, h = window.size
                            img = np.frombuffer(raw_data, dtype=np.uint8).reshape((h, w, 3))
                            img = cv2.cvtColor(np.flipud(img), cv2.COLOR_RGB2BGR)
                        else:
                            # CPU compositing path (VPS, no GL)
                            headless_res = getattr(settings, "HEADLESS_RES", state.image_size)
                            img = cpu_composite_frame(current_main_img, current_float_img, headless_res)
                            if img is None:
                                continue

                        encode_params = [
                            int(cv2.IMWRITE_JPEG_QUALITY),
                            getattr(settings, "JPEG_QUALITY", 70),
                        ]
                        ok, jpg_bytes = cv2.imencode(".jpg", img, encode_params)
                        if ok:
                            exchange.set_frame(jpg_bytes.tobytes())
                        else:
                            print("[CAPTURE] cv2.imencode failed.")
                    except Exception as e:
                        print(f"[CAPTURE ERROR] {e}")

            # --- 6. Swap Buffers (Local Only) ---
            if not settings.SERVER_MODE and has_gl:
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

            # rolling FPS
            if len(frame_times) > 1:
                avg = sum(frame_times) / len(frame_times)
                last_actual_fps = 1.0 / avg if avg > 0 else 0.0
                if not HTTP_MONITOR:
                    if FRAME_COUNTER_DISPLAY:
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
            if not settings.SERVER_MODE and has_gl and glfw.window_should_close(window):
                state.run_mode = False

        if not settings.SERVER_MODE and has_gl:
            glfw.terminate()
