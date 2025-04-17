import os
os.environ["PYOPENGL_ERROR_CHECKING"] = "0"

from concurrent.futures import ThreadPoolExecutor
from collections import deque
import pygame, platform

import calculators
from settings import (
    FULLSCREEN_MODE, PINGPONG, FPS, CLOCK_MODE, FIFO_LENGTH,
    FRAME_COUNTER_DISPLAY, SHOW_DELTA, TEST_MODE, HTTP_MONITOR,
)
from index_calculator  import update_index
from folder_selector   import update_folder_selection, folder_dictionary
import renderer
from event_handler     import event_check
from display_manager   import DisplayState, get_aspect_ratio, display_init

# ───────── optional HTTP monitor ─────────
monitor = None
if TEST_MODE and HTTP_MONITOR:
    from lightweight_monitor import start_monitor
    monitor = start_monitor()

# ───────── helper class ─────────
class RollingIndexCompensator:
    def __init__(self, maxlen=10, correction_factor=0.5):
        self.diffs = deque(maxlen=maxlen)
        self.correction_factor = correction_factor
    def update(self, cur, disp):
        self.diffs.append(cur - disp)
    def get(self, cur):
        if not self.diffs:
            return cur
        avg = sum(self.diffs) / len(self.diffs)
        return cur - round(avg * self.correction_factor)

# ───────── main entry ─────────
def run_display(clock_source=CLOCK_MODE):
    # basic setup
    state = DisplayState(); state.fullscreen = FULLSCREEN_MODE
    _, main_paths, float_paths = calculators.init_all(clock_source)
    print(platform.system(), "midi clock mode is:", clock_source)

    main_cnt  = len(main_paths[0])
    float_cnt = len(float_paths[0])
    png_len   = len(main_paths)
    _, w, h   = get_aspect_ratio(main_paths[0][0])
    state.image_size = (w, h); print("Image size:", state.image_size)

    from image_loader import ImageLoader, FIFOImageBuffer
    class FIFOImageBufferPatched(FIFOImageBuffer):
        def current_depth(self):
            with self.lock: return len(self.queue)

    loader = ImageLoader()
    loader.set_paths(main_paths, float_paths)
    loader.set_png_paths_len(png_len)

    pygame.init(); display_init(state)
    pygame.event.set_grab(False); pygame.mouse.set_visible(False)
    vclock = pygame.time.Clock()

    index, _  = update_index(png_len, PINGPONG)
    update_folder_selection(index, float_cnt, main_cnt)
    fifo      = FIFOImageBufferPatched(max_size=FIFO_LENGTH)

    # first frame
    m_fold, f_fold = folder_dictionary['Main_and_Float_Folders']
    m_img, f_img   = loader.load_images(index, m_fold, f_fold)
    fifo.update(index, (m_img, f_img))
    tex1 = renderer.create_texture(m_img)
    tex2 = renderer.create_texture(f_img)

    # miss counters
    fifo_miss_count = 0
    last_fifo_miss  = -1

    compensator = RollingIndexCompensator()
    displayed_index = index      # seed for first monitor packet

    # async helper
    def async_load_callback(fut, sched_idx):
        try:
            fifo.update(sched_idx, fut.result())
        except Exception as e:
            print("Async load error:", e)
            if monitor: monitor.record_load_error(sched_idx, e)

    with ThreadPoolExecutor(max_workers=4) as pool:
        def queue_next(idx, last):
            if idx == 0:                 return 1
            if idx == png_len - 1:       return idx - 1
            return idx + 1 if idx > last else idx - 1

        nxt = queue_next(index, index)
        pool.submit(loader.load_images, nxt, m_fold, f_fold
                    ).add_done_callback(lambda fut,s=nxt: async_load_callback(fut,s))

        frame_counter = 0; start_mark = pygame.time.get_ticks()

        def maybe_print_fps(fr_cnt, mark):
            if not FRAME_COUNTER_DISPLAY: return fr_cnt, mark
            if (pygame.time.get_ticks()-mark) >= 10_000:
                fps_val = fr_cnt / 10
                print(f"index:{index}  [Display Rate] {fps_val:.2f} fps")
                if fps_val < FPS-2: print("[Warning] frame‑drop suspected")
                return 0, pygame.time.get_ticks()
            return fr_cnt, mark

        while state.run_mode:
            # events + fullscreen toggle
            state.fullscreen = event_check(pygame.event.get(), state)
            if state.needs_update:
                display_init(state); pygame.mouse.set_visible(False)
                state.needs_update = False

            prev = index
            index, _ = update_index(png_len, PINGPONG)
            if index != prev:
                update_folder_selection(index, float_cnt, main_cnt)
                m_fold, f_fold = folder_dictionary['Main_and_Float_Folders']

                cmp_idx  = compensator.get(index)
                hit      = fifo.get(cmp_idx)
                if hit:
                    displayed_index, m_img, f_img = hit
                    if SHOW_DELTA:
                        diff = index - displayed_index
                        off  = cmp_idx - index
                        print(f"[DEBUG] idx={index} disp={displayed_index} Δ={diff} off={off}")
                    compensator.update(index, displayed_index)
                    renderer.update_texture(tex1, m_img)
                    renderer.update_texture(tex2, f_img)
                else:
                    fifo_miss_count += 1
                    last_fifo_miss  = index
                    print(f"[MISS] FIFO miss idx={index} (comp={cmp_idx}) total={fifo_miss_count}")

                nxt = queue_next(index, prev)
                pool.submit(loader.load_images, nxt, m_fold, f_fold
                            ).add_done_callback(lambda fut,s=nxt: async_load_callback(fut,s))

            renderer.overlay_images_two_pass_like_old(tex1, tex2)
            pygame.display.flip()

            vclock.tick(FPS)
            frame_counter += 1
            frame_counter, start_mark = maybe_print_fps(frame_counter, start_mark)
            actual_fps = vclock.get_fps()

            if monitor:
                monitor.update({
                    "index": index,
                    "displayed": displayed_index,
                    "offset": cmp_idx - index,
                    "fps": actual_fps,
                    "fifo_depth": fifo.current_depth(),
                    "successful_frame": (hit is not None),
                    "main_folder": m_fold,
                    "float_folder": f_fold,
                    "rand_mult": folder_dictionary["rand_mult"],
                    "fifo_miss_count": fifo_miss_count,
                    "last_fifo_miss":  last_fifo_miss,
                    "main_folder_count":  main_cnt,
                    "float_folder_count": float_cnt,
                })

    pygame.quit()

if __name__ == "__main__":
    run_display(CLOCK_MODE)
