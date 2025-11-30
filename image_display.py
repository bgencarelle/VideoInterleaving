import os, platform, datetime, time

import cv2
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from collections import deque

import settings
from shared_state import exchange

# Ensure PyOpenGL checks are off (not needed with ModernGL, but leaving for safety)
os.environ['PYOPENGL_ERROR_CHECKING'] = '0'
# Only import glfw when we actually do a local window
if not settings.SERVER_MODE:
    import glfw

from settings import (FULLSCREEN_MODE, PINGPONG, FPS, FRAME_COUNTER_DISPLAY, SHOW_DELTA, TEST_MODE, HTTP_MONITOR,
                      CLOCK_MODE, FIFO_LENGTH, BACKGROUND_COLOR)
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


def run_display(clock_source=CLOCK_MODE):
    state = DisplayState()
    state.fullscreen = FULLSCREEN_MODE
    last_actual_fps = FPS

    # --- Server capture timing (HEADLESS / SERVER_MODE only) ---
    capture_rate = getattr(settings, 'SERVER_CAPTURE_RATE', 10)  # default 10 FPS
    capture_interval = 1.0 / capture_rate
    last_server_capture = time.time()

    # --- Initialize folders & image size ---
    import calculators
    _, main_folder_path, float_folder_path = calculators.init_all(clock_source)
    main_folder_count = len(main_folder_path[0])
    float_folder_count = len(float_folder_path[0])
    png_paths_len = len(main_folder_path)

    # determine initial image size from first image
    first_image_path = main_folder_path[0][0]
    img0 = cv2.imread(first_image_path, cv2.IMREAD_UNCHANGED)
    if img0 is None:
        raise RuntimeError(f"Unable to load image: {first_image_path}")
    img_h, img_w = img0.shape[:2]
    state.image_size = (img_w, img_h)

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
        # HEADLESS: configure viewport + transforms so image is centered & aspect-correct
        fbo_w, fbo_h = window.size

        # Full-FBO viewport
        window.ctx.viewport = (0, 0, fbo_w, fbo_h)

        # Compute uniform scale to fit image inside FBO while preserving aspect
        scale_x = fbo_w / img_w
        scale_y = fbo_h / img_h
        fs_scale = min(scale_x, scale_y)

        scaled_w = img_w * fs_scale
        scaled_h = img_h * fs_scale

        # Offset so quad is centered at world origin (0,0)
        offset_x = -scaled_w / 2.0
        offset_y = -scaled_h / 2.0

        renderer.set_transform_parameters(
            fs_scale=fs_scale,
            fs_offset_x=offset_x,
            fs_offset_y=offset_y,
            image_size=(img_w, img_h),
            rotation_angle=0.0,
            mirror_mode=0,
        )

        # Orthographic MVP: map world coords [-fbo_w/2, fbo_w/2] × [-fbo_h/2, fbo_h/2] → NDC [-1,1]^2
        mvp = np.eye(4, dtype=np.float32)
        mvp[0, 0] = 2.0 / fbo_w
        mvp[1, 1] = 2.0 / fbo_h
        renderer.update_mvp(mvp)

        print(f"[HEADLESS] FBO {fbo_w}x{fbo_h}, image {img_w}x{img_h}, scale {fs_scale:.3f}")
    else:
        # Local windowed mode: let your existing callback / transform logic handle it
        register_callbacks(window, state)

    # initial index & folder selection
    index, _ = update_index(png_paths_len, PINGPONG)
    last_index = index
    update_folder_selection(index, float_folder_count, main_folder_count)

    fifo_buffer = FIFOImageBufferPatched(max_size=FIFO_LENGTH)
    main_folder, float_folder = folder_dictionary['Main_and_Float_Folders']
    main_image, float_image = image_loader.load_images(index, main_folder, float_folder)
    fifo_buffer.update(index, (main_image, float_image))

    # Push a test frame straight into the exchange (debug)
    try:
        encode_params = [int(cv2.IMWRITE_JPEG_QUALITY),
                         getattr(settings, 'JPEG_QUALITY', 80)]
        ok, jpg_bytes = cv2.imencode('.jpg', main_image, encode_params)
        if ok:
            print("[TEST] Pushing initial frame to FrameExchange.")
            exchange.set_frame(jpg_bytes.tobytes())
        else:
            print("[TEST] cv2.imencode failed on initial frame.")
    except Exception as e:
        print(f"[TEST] Initial frame encode error: {e}")

    # create textures
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

        frame_times = deque(maxlen=60)
        frame_start = time.perf_counter()

        while state.run_mode:
            successful_display = False

            # --- 1. Event Polling (Local Only) ---
            if not settings.SERVER_MODE:
                glfw.poll_events()
                if not state.run_mode:
                    break

            # --- 2. Reinit Check ---
            if state.needs_update:
                display_init(state)
                state.needs_update = False

            # --- 3. Update Index & Images ---
            prev = index
            index, _ = update_index(png_paths_len, PINGPONG)

            if index != prev:
                last_index = prev
                update_folder_selection(index, float_folder_count, main_folder_count)
                main_folder, float_folder = folder_dictionary['Main_and_Float_Folders']
                compensated = compensator.get_compensated_index(index)
                result = fifo_buffer.get(compensated)

                if result is not None:
                    displayed_index, main_img, float_img = result
                    compensator.update(index, displayed_index)
                    if SHOW_DELTA:
                        d = index - displayed_index
                        off = compensated - index
                        print(f"[DEBUG] idx={index}, disp={displayed_index}, Δ={d}, offset={off}")
                    main_texture = renderer.update_texture(main_texture, main_img)
                    float_texture = renderer.update_texture(float_texture, float_img)
                    successful_display = True
                else:
                    print(f"[MISS] FIFO miss for index {index} at {datetime.datetime.now()}")

                if index == 0:
                    next_index = 1
                elif index == png_paths_len - 1:
                    next_index = index - 1
                else:
                    next_index = index + 1 if index > prev else index - 1
                future = executor.submit(image_loader.load_images, next_index, main_folder, float_folder)
                future.add_done_callback(lambda fut, s_idx=next_index: async_load_callback(fut, s_idx))

            # --- 4. Drawing ---
            if settings.SERVER_MODE:
                window.use()  # Bind the virtual framebuffer

            renderer.overlay_images_two_pass_like_old(
                main_texture, float_texture, background_color=BACKGROUND_COLOR
            )

            # --- 5. Server Capture ---
            if settings.SERVER_MODE:
                now_t = time.time()
                if now_t - last_server_capture > capture_interval:
                    last_server_capture = now_t
                    try:
                        raw_data = window.fbo.read(components=3)
                        w, h = window.size

                        img = np.frombuffer(raw_data, dtype=np.uint8).reshape((h, w, 3))
                        img = cv2.cvtColor(np.flipud(img), cv2.COLOR_RGB2BGR)

                        encode_params = [int(cv2.IMWRITE_JPEG_QUALITY),
                                         getattr(settings, 'JPEG_QUALITY', 80)]
                        ok, jpg_bytes = cv2.imencode('.jpg', img, encode_params)
                        if not ok:
                            print("[CAPTURE] cv2.imencode failed.")
                        else:
                            exchange.set_frame(jpg_bytes.tobytes())
                           # print(f"[CAPTURE] Frame captured and stored at {datetime.datetime.now()}")
                    except Exception as e:
                        print(f"[CAPTURE ERROR] {e}")

            # --- 6. Swap Buffers (Local Only) ---
            if not settings.SERVER_MODE:
                glfw.swap_buffers(window)

            # --- 7. Timing ---
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
                if not HTTP_MONITOR:
                    print(f" {last_actual_fps:.2f} FPS")
                    if last_actual_fps < FPS - 2:
                        print(f" Frame drop! Target {FPS}, Got {last_actual_fps:.2f}")

            if monitor:
                monitor.update({
                    "index": index,
                    "displayed": displayed_index if 'displayed_index' in locals() else index,
                    "delta": index - (displayed_index if 'displayed_index' in locals() else index),
                    "fps": last_actual_fps,
                    "fifo_depth": fifo_buffer.current_depth(),
                    "successful_frame": successful_display,
                    "main_folder": main_folder,
                    "float_folder": float_folder,
                    "rand_mult": folder_dictionary.get('rand_mult'),
                    "main_folder_count": main_folder_count,
                    "float_folder_count": float_folder_count
                })

            if not settings.SERVER_MODE and glfw.window_should_close(window):
                state.run_mode = False

        if not settings.SERVER_MODE:
            glfw.terminate()
