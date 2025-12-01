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

os.environ['PYOPENGL_ERROR_CHECKING'] = '0'

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

def sharpen_for_stream(frame_bgr: np.ndarray,
                       amount: float = 0.5,
                       radius: float = 1.0,
                       threshold: int = 0) -> np.ndarray:
    """
    Simple unsharp mask in BGR space.

    amount   – how strong the sharpening is (0.0 disables)
    radius   – blur radius (in pixels, via Gaussian sigma)
    threshold – optional: don't sharpen very low-contrast areas (0 = off)
    """
    if amount <= 0.0:
        return frame_bgr

    # Gaussian blur as the "unsharp" base
    blurred = cv2.GaussianBlur(frame_bgr, ksize=(0, 0),
                               sigmaX=radius, sigmaY=radius)

    # Basic unsharp mask: original*(1+amount) + blurred*(-amount)
    sharpened = cv2.addWeighted(frame_bgr, 1.0 + amount,
                                blurred, -amount, 0)

    if threshold > 0:
        # Optional: avoid sharpening noise
        diff = cv2.absdiff(frame_bgr, blurred)
        mask = (diff < threshold)
        # Where mask is True, keep original pixels
        sharpened = np.where(mask, frame_bgr, sharpened)

    return sharpened


def cpu_composite_frame(main_img, float_img):
    """
    Fast CPU compositing for headless fallback:

    - main_img: RGB / RGBA base
    - float_img: optional RGB / RGBA overlay
    - No resizing: browser does the scaling.
    """
    if main_img is None:
        return None

    # main -> RGB
    if main_img.ndim == 2:
        main_rgb = cv2.cvtColor(main_img, cv2.COLOR_GRAY2RGB)
    elif main_img.ndim == 3 and main_img.shape[2] == 4:
        main_rgb = main_img[..., :3]
    elif main_img.ndim == 3 and main_img.shape[2] == 3:
        main_rgb = main_img
    else:
        raise ValueError(f"Unsupported main_img shape: {main_img.shape!r}")

    base = main_rgb.astype(np.uint16)

    if float_img is None:
        out_rgb = np.clip(base, 0, 255).astype(np.uint8)
        return out_rgb[..., ::-1]

    if float_img.shape[0] != base.shape[0] or float_img.shape[1] != base.shape[1]:
        out_rgb = np.clip(base, 0, 255).astype(np.uint8)
        return out_rgb[..., ::-1]

    # float -> RGB + alpha
    if float_img.ndim == 2:
        float_rgb = cv2.cvtColor(float_img, cv2.COLOR_GRAY2RGB).astype(np.uint16)
        alpha = None
    elif float_img.ndim == 3 and float_img.shape[2] == 4:
        float_rgb = float_img[..., :3].astype(np.uint16)
        alpha = float_img[..., 3:4].astype(np.uint16)
    elif float_img.ndim == 3 and float_img.shape[2] == 3:
        float_rgb = float_img.astype(np.uint16)
        alpha = None
    else:
        raise ValueError(f"Unsupported float_img shape: {float_img.shape!r}")

    if alpha is None:
        out_rgb16 = float_rgb
    else:
        inv_alpha = 255 - alpha
        out_rgb16 = (float_rgb * alpha + base * inv_alpha + 127) // 255

    out_rgb = np.clip(out_rgb16, 0, 255).astype(np.uint8)
    return out_rgb[..., ::-1]


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

    # --- Display / GL init (headless or local) ---
    window = display_init(state)

    if settings.SERVER_MODE:
        has_gl = window is not None
        if has_gl:
            print("[DISPLAY] SERVER_MODE: using headless ModernGL.")
        else:
            print("[DISPLAY] SERVER_MODE: using CPU-only compositing.")
    else:
        if window is None:
            raise RuntimeError("Local mode requires a GL window, but display_init returned None.")
        has_gl = True
        register_callbacks(window, state)

    encode_params = [
        int(cv2.IMWRITE_JPEG_QUALITY), getattr(settings, "JPEG_QUALITY", 80),
        # int(cv2.IMWRITE_JPEG_OPTIMIZE), 1,
        # int(cv2.IMWRITE_JPEG_PROGRESSIVE), 1,   # if you want progressive
    ]

    # --- Initial index & images ---
    index, _ = update_index(png_paths_len, PINGPONG)
    last_index = index
    update_folder_selection(index, float_folder_count, main_folder_count)

    fifo_buffer = FIFOImageBufferPatched(max_size=FIFO_LENGTH)
    main_folder, float_folder = folder_dictionary["Main_and_Float_Folders"]
    main_image, float_image = image_loader.load_images(index, main_folder, float_folder)
    fifo_buffer.update(index, (main_image, float_image))
    # In headless GL mode, tell the renderer what the image size is
    if settings.SERVER_MODE and getattr(settings, "HEADLESS_USE_GL", False):
        h_img, w_img = main_image.shape[0], main_image.shape[1]
        renderer.set_transform_parameters(
            fs_scale=1.0,
            fs_offset_x=0.0,
            fs_offset_y=0.0,
            image_size=(w_img, h_img),
            rotation_angle=0.0,
            mirror_mode=0,
        )

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

    #max_workers = min(6, (os.cpu_count() or 1))

    max_workers = 3
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # schedule first preload
        if index == 0:
            next_index = 1
        elif index == png_paths_len - 1:
            next_index = index - 1
        else:
            next_index = index + 1 if index > last_index else index - 1

        future = executor.submit(
            image_loader.load_images, next_index, main_folder, float_folder
        )
        future.add_done_callback(lambda fut, s_idx=next_index: async_load_callback(fut, s_idx))

        frame_times = deque(maxlen=60)
        frame_start = time.perf_counter()

        while (state.run_mode and not settings.SERVER_MODE) or settings.SERVER_MODE:
            successful_display = False

            # --- 1. Event polling (local only) ---
            if not settings.SERVER_MODE and has_gl:
                glfw.poll_events()
                if not state.run_mode:
                    break

            # --- 2. Reinit (local only) ---
            if state.needs_update and not settings.SERVER_MODE and has_gl:
                display_init(state)
                state.needs_update = False

            # --- 3. Index & images ---
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

                future = executor.submit(
                    image_loader.load_images, next_index, main_folder, float_folder
                )
                future.add_done_callback(lambda fut, s_idx=next_index: async_load_callback(fut, s_idx))

            # --- 4. Drawing ---
            if has_gl:
                # In local mode, glfw has already made the context current.
                # In headless mode, HeadlessWindow.use() binds the FBO.
                if settings.SERVER_MODE:
                    window.use()
                renderer.overlay_images_two_pass_like_old(
                    main_texture,
                    float_texture,
                    background_color=BACKGROUND_COLOR,
                )

            # --- 5. Capture / streaming (SERVER_MODE only) ---
            if settings.SERVER_MODE:
                now = time.time()
                if (index != last_captured_index) and (now - last_server_capture > capture_interval):
                    last_server_capture = now
                    last_captured_index = index

                    try:
                        if has_gl:
                            # Read back from headless FBO
                            window.use()
                            raw = window.fbo.read(components=3)  # RGB, bottom-up
                            w, h = window.size
                            frame = np.frombuffer(raw, dtype=np.uint8).reshape((h, w, 3))
                            frame = cv2.cvtColor(np.flipud(frame), cv2.COLOR_RGB2BGR)
                        else:
                            # CPU fallback
                            frame = cpu_composite_frame(current_main_img, current_float_img)

                        if frame is not None:
                            # OPTIONAL SHARPEN
                            amt = getattr(settings, "STREAM_SHARPEN_AMOUNT", 0.0)
                            if amt > 0.0:
                                radius = getattr(settings, "STREAM_SHARPEN_RADIUS", 1.0)
                                thresh = getattr(settings, "STREAM_SHARPEN_THRESHOLD", 0)
                                frame_to_encode = sharpen_for_stream(frame, amt, radius, thresh)
                            else:
                                frame_to_encode = frame

                            ok, buf = cv2.imencode(".jpg", frame_to_encode, encode_params)
                            if ok:
                                exchange.set_frame(buf.tobytes())
                            else:
                                print("[CAPTURE] cv2.imencode failed.")
                    except Exception as e:
                        print(f"[CAPTURE ERROR] {e}")

            # --- 6. Swap buffers (local only) ---
            if not settings.SERVER_MODE and has_gl:
                glfw.swap_buffers(window)

            # --- 7. Timing / FPS ---
            now = time.perf_counter()
            dt = now - frame_start
            frame_times.append(dt)
            frame_start = now

            if FPS:
                target = 1.0 / FPS
                to_sleep = target - dt
                if to_sleep > 0:
                    time.sleep(to_sleep)

            if len(frame_times) > 1:
                avg = sum(frame_times) / len(frame_times)
                last_actual_fps = 1.0 / avg if avg > 0 else 0.0
                if not HTTP_MONITOR and FRAME_COUNTER_DISPLAY:
                    print(f"{last_actual_fps:.2f} FPS")
                    if last_actual_fps < FPS - 2:
                        print(f" Frame drop! Target {FPS}, Got {last_actual_fps:.2f}")

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

            if not settings.SERVER_MODE and has_gl and glfw.window_should_close(window):
                state.run_mode = False

        if not settings.SERVER_MODE and has_gl:
            glfw.terminate()
