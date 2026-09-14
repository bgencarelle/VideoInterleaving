"""
mouse_follow.py -- Mouse-following screen capture source with live zoom control for modem_screen.py
"""
import threading
import numpy as np


def mouse_follow_source(initial_width=400, aspect_ratio=4 / 3):
    """
    Grab a fixed-size screen region centered on the mouse cursor.
    Zoom in/out dynamically using '+' / '=' and '-' keys.
    """
    try:
        from mss import mss as _MSS
    except ImportError:
        raise SystemExit("Missing 'mss'. Run: pip install mss")

    try:
        import pyautogui
    except ImportError:
        raise SystemExit("Missing 'pyautogui'. Run: pip install pyautogui")

    try:
        from pynput import keyboard
    except ImportError:
        raise SystemExit("Missing 'pynput'. Run: pip install pynput")

    sct = None
    listener = None
    screen_w, screen_h = pyautogui.size()

    lock = threading.Lock()
    state = {
        'width': int(initial_width),
        'height': int(initial_width / aspect_ratio)
    }

    def _update_zoom(factor):
        with lock:
            new_w = max(20, min(screen_w, int(state['width'] * factor)))
            state['width'] = new_w
            state['height'] = max(15, min(screen_h, int(new_w / aspect_ratio)))

    def _on_press(key):
        try:
            # Handle standard character keys
            if hasattr(key, 'char') and key.char:
                if key.char in ('+', '='):
                    _update_zoom(0.9)  # Zoom in (smaller bounding box)
                elif key.char == '-':
                    _update_zoom(1.1)  # Zoom out (larger bounding box)
            # Handle Virtual Key codes (e.g. Numpad +, Numpad -)
            elif hasattr(key, 'vk') and key.vk:
                if key.vk in (107, 187):  # VK_ADD / VK_OEM_PLUS
                    _update_zoom(0.9)
                elif key.vk in (109, 189):  # VK_SUBTRACT / VK_OEM_MINUS
                    _update_zoom(1.1)
        except Exception:
            pass

    # Start non-blocking keyboard listener
    try:
        listener = keyboard.Listener(on_press=_on_press)
        listener.start()
    except Exception as exc:
        print(f"Warning: Could not start keyboard zoom listener: {exc}")

    def grab():
        nonlocal sct
        # Initialize mss lazily on the capture worker thread
        if sct is None:
            sct = _MSS()

        mx, my = pyautogui.position()

        with lock:
            w = state['width']
            h = state['height']

        # Center bounding box on mouse position
        left = mx - (w // 2)
        top = my - (h // 2)

        # Clamp bounding box inside display boundaries
        left = max(0, min(left, screen_w - w))
        top = max(0, min(top, screen_h - h))

        mon = {
            'left': int(left),
            'top': int(top),
            'width': int(w),
            'height': int(h)
        }

        shot = sct.grab(mon)
        raw = np.frombuffer(shot.raw, np.uint8).reshape(shot.height, shot.width, 4)
        return raw[:, :, 2::-1]  # Convert BGRA to RGB

    def close():
        nonlocal sct, listener
        if listener is not None:
            try:
                listener.stop()
            except Exception:
                pass
            listener = None
        if sct is not None:
            try:
                sct.close()
            except Exception:
                pass
            sct = None

    grab.close = close
    return grab