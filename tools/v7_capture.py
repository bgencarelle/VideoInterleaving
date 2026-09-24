"""Capture sources used by the standalone V7 live sender.

This module contains no modem encoder or decoder.
"""
import argparse
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from urllib.parse import urlsplit

import numpy as np

# Capture sources. Each returns grab() -> RGB uint8 array, any size.
# --------------------------------------------------------------------------

def _read_exact(stream, count):
    """Read exactly `count` bytes, or return None at end of stream.

    subprocess is opened with bufsize=0, so stdout is a raw stream and read(n)
    is free to return fewer bytes than asked. A colour frame here is hundreds
    of kilobytes -- far past a pipe buffer -- so short reads are the norm, not
    an edge case, and treating one as end-of-stream silently yields a frame of
    zeros. scope_screen gets away with the single read because its grey frames
    are ~16 kB.
    """
    chunks = []
    have = 0
    while have < count:
        piece = stream.read(count-have)
        if not piece:
            return None
        chunks.append(piece)
        have += len(piece)
    return b''.join(chunks)


def _region(text):
    if not text:
        return None
    parts = [int(p) for p in text.split(',')]
    if len(parts) != 4 or parts[2] < 1 or parts[3] < 1:
        raise argparse.ArgumentTypeError('Region is left,top,width,height')
    return parts


def _read_ppm(stream):
    """Read one self-describing RGB frame from FFmpeg's image2pipe output."""
    if stream.readline() != b'P6\n':
        raise RuntimeError('FFmpeg capture ended without a complete frame. '
                           'See FFmpeg errors above; check the capture '
                           'device, supported frame rate, and permissions.')
    dimensions = stream.readline()
    while dimensions.startswith(b'#'):
        dimensions = stream.readline()
    w, h = map(int, dimensions.split())
    if w <= 0 or h <= 0 or stream.readline().strip() != b'255':
        raise RuntimeError('Invalid RGB frame header from FFmpeg')
    buf = _read_exact(stream, w*h*3)
    if buf is None:
        raise RuntimeError('FFmpeg capture ended partway through a frame')
    return np.frombuffer(buf, np.uint8).reshape(h, w, 3)


def screen_source(region=None):
    """Grab the screen through mss.

    np.frombuffer wraps the captured bytes instead of copying them; on a Retina
    panel that buffer is tens of megabytes per grab. Capture dominates this
    mode either way -- prefer --source ffmpeg if the rate matters.
    """
    try:
        from mss import MSS as _MSS
    except ImportError:
        from mss import mss as _MSS
    sct = None

    def grab():
        nonlocal sct
        if sct is None:
            sct = _MSS()
        mon = sct.monitors[1] if region is None else {
            'left': region[0], 'top': region[1],
            'width': region[2], 'height': region[3]}
        shot = sct.grab(mon)
        raw = np.frombuffer(shot.raw, np.uint8).reshape(shot.height, shot.width, 4)
        return raw[:, :, 2::-1]          # BGRA -> RGB
    def close():
        if sct is not None:
            sct.close()
    grab.close = close
    return grab


def test_source():
    """Synthetic moving colour target for capture/transport testing."""
    start = time.perf_counter()

    def grab():
        t = time.perf_counter() - start
        yy, xx = np.mgrid[0:240, 0:200]
        cx, cy = 100 + 70*np.sin(t*1.1), 120 + 80*np.cos(t*.7)
        blob = np.exp(-(((xx-cx)/38)**2 + ((yy-cy)/38)**2))
        rgb = np.zeros((240, 200, 3))
        rgb[:, :, 0] = .12 + .80*blob
        rgb[:, :, 1] = .10 + .45*blob*(.5 + .5*np.sin(t))
        rgb[:, :, 2] = .16 + .30*(xx/200)
        return np.uint8(np.clip(rgb, 0, 1)*255)

    return grab


def mouse_follow_source(initial_width=400, aspect_ratio=4/3):
    """Capture a zoomable screen region centered on the mouse cursor."""
    try:
        from mss import MSS as _MSS
        import pyautogui
        from pynput import keyboard
    except ImportError as exc:
        raise SystemExit(f'Mouse-follow capture dependency missing: {exc.name}')
    sct = None
    listener = None
    screen_w, screen_h = pyautogui.size()
    lock = threading.Lock()
    state = {'width': int(initial_width), 'height': int(initial_width/aspect_ratio)}

    def update_zoom(factor):
        with lock:
            width = max(20, min(screen_w, int(state['width'] * factor)))
            state['width'] = width
            state['height'] = max(15, min(screen_h, int(width/aspect_ratio)))

    def on_press(key):
        char = getattr(key, 'char', None)
        if char in ('+', '='):
            update_zoom(.9)
        elif char == '-':
            update_zoom(1.1)

    try:
        listener = keyboard.Listener(on_press=on_press)
        listener.start()
    except Exception:
        listener = None

    def grab():
        nonlocal sct
        if sct is None:
            sct = _MSS()
        mx, my = pyautogui.position()
        with lock:
            width, height = state['width'], state['height']
        left = max(0, min(mx-width//2, screen_w-width))
        top = max(0, min(my-height//2, screen_h-height))
        shot = sct.grab({'left': int(left), 'top': int(top),
                         'width': int(width), 'height': int(height)})
        raw = np.frombuffer(shot.raw, np.uint8).reshape(shot.height, shot.width, 4)
        return raw[:, :, 2::-1]

    def close():
        nonlocal sct, listener
        if listener is not None:
            listener.stop()
            listener = None
        if sct is not None:
            sct.close()
            sct = None
    grab.close = close
    return grab


def ffmpeg_source(spec, fps, region=None, display=None, width=320,
                  scale_flags='neighbor'):
    """Capture through ffmpeg's platform fast path.

    mss goes via CoreGraphics on macOS and costs tens of milliseconds a grab.
    ffmpeg uses avfoundation / x11grab / gdigrab and scales before handing the
    frame over, so Python receives a small RGB array and does no image work.

    PPM frames carry their dimensions in the pipe. Scale by width and preserve
    the actual input aspect, including for cameras unrelated to screen size.
    """
    if shutil.which('ffmpeg') is None:
        raise SystemExit('ffmpeg not found. brew install ffmpeg / apt install ffmpeg')
    w = int(width)
    if w < 1:
        raise ValueError('Capture width must be positive')
    fps = 30 if fps is None else fps
    if scale_flags not in ('neighbor', 'area', 'bilinear', 'bicubic', 'lanczos'):
        raise ValueError(f'Unsupported FFmpeg scale flags: {scale_flags}')

    if spec:
        fmt, src = spec.split(':', 1)
    elif sys.platform == 'darwin':
        fmt, src = 'avfoundation', f'{display if display is not None else 1}:none'
    elif sys.platform.startswith('win'):
        fmt, src = 'gdigrab', 'desktop'
    else:
        fmt, src = 'x11grab', os.environ.get('DISPLAY', ':0.0')

    cmd = ['ffmpeg', '-nostdin', '-loglevel', 'error', '-f', fmt,
           '-framerate', str(int(max(fps, 1)))]
    if fmt == 'x11grab' and region:
        cmd += ['-video_size', f'{region[2]}x{region[3]}',
                '-i', f'{src}+{region[0]},{region[1]}']
    else:
        cmd += ['-i', src]
    cmd += ['-vf', f'scale={w}:-1:flags={scale_flags}', '-pix_fmt', 'rgb24',
            '-fps_mode', 'passthrough',
            '-c:v', 'ppm', '-f', 'image2pipe', '-an', '-sn', '-']
    # Device timestamps are not necessarily a constant-rate timeline. The
    # default sync mode can emit thousands of duplicates to fill their gaps.
    # This pipe needs exactly one image for each input frame.
    errors = tempfile.TemporaryFile()
    try:
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stdin=subprocess.DEVNULL,
                                stderr=errors, bufsize=0, start_new_session=True)
    except BaseException:
        errors.close()
        raise
    closed = threading.Event()
    close_lock = threading.Lock()

    def grab():
        try:
            return _read_ppm(proc.stdout)
        except RuntimeError as exc:
            with close_lock:
                if closed.is_set():
                    raise
                errors.seek(0)
                detail = errors.read().decode('utf-8', errors='replace').strip()
            raise RuntimeError(f'{exc}\n{detail}' if detail else str(exc)) from exc

    def close():
        with close_lock:
            if closed.is_set():
                return
            closed.set()
            try:
                if proc.poll() is None:
                    proc.terminate()
                    try:
                        proc.wait(timeout=2)
                    except subprocess.TimeoutExpired:
                        proc.kill()
                        proc.wait(timeout=2)
            finally:
                errors.close()
        # Closing stdout is left to the capture worker after its read exits.

    grab.proc = proc
    grab.close = close
    grab.paced = True  # the device already paces the pipe; drain it continuously
    return grab


def camera_source(index=0, fps=30, width=320, spec=None,
                  scale_flags='neighbor'):
    """Webcam through ffmpeg, same reasoning as screen capture."""
    if sys.platform == 'darwin':
        spec = spec or f'avfoundation:{index}'
    elif sys.platform.startswith('win'):
        spec = spec or 'dshow:video=Integrated Camera'
    else:
        spec = spec or f'v4l2:/dev/video{index}'
    return ffmpeg_source(spec, fps, width=width, scale_flags=scale_flags)


def screen_capture_source(fps, region=None, display=None, width=320,
                          spec=None, scale_flags='neighbor'):
    """Compatibility adapter for the former modem_screen API."""
    # AVFoundation's device numbering is machine-dependent. The old screen
    # path selected the first screen capture device; do not inherit the
    # camera-oriented default used by the generic FFmpeg helper.
    if display is None and sys.platform == 'darwin':
        display = 0
    return ffmpeg_source(spec, fps, region=region, display=display,
                         width=width, scale_flags=scale_flags)


_LIVE_SCHEMES = frozenset({
    'http', 'https', 'rtsp', 'rtsps', 'rtmp', 'rtmps', 'udp', 'tcp', 'srt',
    'rist', 'rtp', 'rtmpe', 'rtmpt',
})


def _is_stream_url(source):
    return urlsplit(str(source)).scheme.lower() in _LIVE_SCHEMES


def video_source(source, loop=None, realtime=None, width=320,
                 scale_flags='bicubic'):
    """Read a local video file in a real-time loop or a live stream URL.

    Local files loop and are paced with ``-re``. Network URLs are treated as
    live inputs: they are read as delivered, without file-loop or input pacing.
    PPM carries each output frame's dimensions, so this path needs no separate
    ffprobe pass and can handle sources with different aspect ratios.
    """
    if shutil.which('ffmpeg') is None:
        raise SystemExit('ffmpeg not found. brew install ffmpeg / apt install ffmpeg')
    source = os.path.expanduser(str(source))
    is_stream = _is_stream_url(source)
    if not is_stream and not os.path.isfile(source):
        raise SystemExit(f'No such video file: {source}')
    if width < 1:
        raise ValueError('Capture width must be positive')
    if scale_flags not in ('neighbor', 'area', 'bilinear', 'bicubic', 'lanczos'):
        raise ValueError(f'Unsupported FFmpeg scale flags: {scale_flags}')

    if loop is None:
        loop = not is_stream
    if realtime is None:
        realtime = not is_stream

    cmd = ['ffmpeg', '-nostdin', '-loglevel', 'error']
    if loop:
        cmd += ['-stream_loop', '-1']
    if realtime:
        cmd += ['-re']
    if is_stream:
        # Fail a stalled network read instead of leaving the capture worker
        # blocked forever during shutdown or source loss.
        cmd += ['-rw_timeout', '10000000']
    cmd += ['-i', source, '-vf',
            f'scale={int(width)}:-2:flags={scale_flags}',
            '-fps_mode', 'passthrough', '-pix_fmt', 'rgb24',
            '-c:v', 'ppm', '-f', 'image2pipe', '-an', '-sn', '-']

    errors = tempfile.TemporaryFile()
    try:
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE,
                                stderr=errors, bufsize=0)
    except BaseException:
        errors.close()
        raise
    state = {'closed': False}
    close_lock = threading.Lock()

    def grab():
        try:
            return _read_ppm(proc.stdout)
        except RuntimeError as exc:
            with close_lock:
                if state['closed']:
                    raise
                errors.flush()
                errors.seek(0)
                detail = errors.read().decode('utf-8', errors='replace').strip()
            if source in detail:
                detail = detail.replace(source, '<video source>')
            message = f'FFmpeg video source failed: {exc}'
            if detail:
                message += f'\n{detail}'
            raise RuntimeError(message) from exc

    def close():
        with close_lock:
            if state['closed']:
                return
            state['closed'] = True
            try:
                if proc.poll() is None:
                    proc.terminate()
                    try:
                        proc.wait(timeout=2)
                    except subprocess.TimeoutExpired:
                        proc.kill()
                        proc.wait(timeout=2)
            finally:
                errors.close()

    grab.proc = proc
    grab.close = close
    # File input is paced by -re; a live URL is paced by its own arrival rate.
    # Drain either continuously so the newest-frame mailbox stays current.
    grab.paced = True
    return grab


def test_source():
    """A moving target that needs no devices, for checking the chain."""
    start = time.perf_counter()

    def grab():
        t = time.perf_counter()-start
        yy, xx = np.mgrid[0:240, 0:200]
        cx, cy = 100+70*np.sin(t*1.1), 120+80*np.cos(t*.7)
        blob = np.exp(-(((xx-cx)/38)**2 + ((yy-cy)/38)**2))
        rgb = np.zeros((240, 200, 3))
        rgb[:, :, 0] = .12+.80*blob
        rgb[:, :, 1] = .10+.45*blob*(.5+.5*np.sin(t))
        rgb[:, :, 2] = .16+.30*(xx/200)
        return np.uint8(np.clip(rgb, 0, 1)*255)
    return grab


class Throttled:
    """Capture on a background thread at its own rate.

    The encoder asks for a picture once per packet. A capture that is slower
    than the wire must not stall the audio callback, and one that is faster
    must not burn a core producing frames nobody sends. The newest frame wins,
    exactly like the receiver's presentation rule.
    """

    def __init__(self, grab, hz):
        self._grab = grab
        self._latest = None
        self._error = None
        self._updated = None
        self._ready = threading.Event()
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self.grabs = 0
        self._period = 1.0/max(hz, .1)
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        try:
            self._ready.wait()
            self()
        except BaseException:
            self.close()
            raise

    def _run(self):
        try:
            while not self._stop.is_set():
                began = time.perf_counter()
                frame = self._grab()
                if frame is None:
                    raise RuntimeError('Capture produced no frames')
                with self._lock:
                    self._latest = frame
                    self._updated = time.monotonic()
                    self.grabs += 1
                self._ready.set()
                rest = self._period-(time.perf_counter()-began)
                if rest > 0 and not getattr(self._grab, 'paced', False):
                    self._stop.wait(rest)
        except BaseException as exc:
            with self._lock:
                self._error = exc
        finally:
            self._ready.set()
            if hasattr(self._grab, 'close'):
                self._grab.close()
            proc = getattr(self._grab, 'proc', None)
            if proc is not None:
                proc.stdout.close()

    def retune(self, hz):
        """Follow the wire rate once the audio device has reported its own.

        The frame rate is the device's sample rate over the frame length, so it
        is not known until the stream is open -- after the capture thread has
        already started. Rather than requesting a rate to make the number
        predictable, the capture is re-paced to the rate that turned up.
        """
        with self._lock:
            self._period = 1.0/max(hz, .1)

    def __call__(self):
        with self._lock:
            if self._error is not None:
                raise RuntimeError(f'Capture failed: {self._error}') from self._error
            if self._updated is not None and time.monotonic()-self._updated > max(5, 3*self._period):
                raise RuntimeError('Capture stalled: no new frame for over '
                                   f'{max(5, 3*self._period):g} seconds')
            return self._latest

    def close(self):
        self._stop.set()
        proc = getattr(self._grab, 'proc', None)
        if proc is not None:
            if hasattr(self._grab, 'close'):
                self._grab.close()
            else:
                proc.terminate()
                try:
                    proc.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait(timeout=2)
        self._thread.join(timeout=2)


# --------------------------------------------------------------------------
