"""Capture sources used by the standalone V7 live sender.

This module contains no modem encoder or decoder.
"""
import argparse
import os
import re
import signal
import shutil
import socket
import subprocess
import sys
import tempfile
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlsplit

import numpy as np

from local_frame_bridge import read_frame

ROOT = Path(__file__).resolve().parents[1]


def _temporary_error_file():
    directory = ROOT/'tmp'
    directory.mkdir(exist_ok=True)
    return tempfile.TemporaryFile(dir=directory)

# Capture sources return RGB uint8 arrays, or a CapturedFrame when preprocessing
# changes the pixel dimensions but the original source aspect must be retained.
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class CapturedFrame:
    """Pixels plus source geometry when capture preprocessing changes size."""
    rgb: np.ndarray
    source_size: tuple
    prepared: bool = False


class CaptureEndOfStream(Exception):
    """A finite capture source ended cleanly after its final frame."""


def _showinfo_source_size(line):
    """Read the pre-filter width/height printed by FFmpeg's showinfo filter."""
    match = re.search(rb'\bs:(\d+)x(\d+)\b', line)
    if match is None:
        return None
    width, height = (int(part) for part in match.groups())
    return (width, height) if width > 0 and height > 0 else None


def _read_exact(stream, count):
    """Read exactly `count` bytes, or return None at end of stream.

    subprocess is opened with bufsize=0, so stdout is a raw stream and read(n)
    is free to return fewer bytes than asked. A colour frame here is hundreds
    of kilobytes -- far past a pipe buffer -- so short reads are the norm, not
    an edge case, and treating one as end-of-stream silently yields a frame of
    zeros. scope_screen gets away with the single read because its grey frames
    are ~16 kB.
    """
    readinto = getattr(stream, 'readinto', None)
    if readinto is None:
        chunks = []
        have = 0
        while have < count:
            piece = stream.read(count-have)
            if not piece:
                return None
            chunks.append(piece)
            have += len(piece)
        return b''.join(chunks)
    # Read straight into one buffer. A full-size frame (6 MB at 1080p, as
    # --dct-encode captures it) arrives in ~100 pipe-sized pieces; the join
    # above would allocate a large bytes object per piece and copy the frame
    # again. A bytearray also gives the caller a writable array.
    buffer = bytearray(count)
    view = memoryview(buffer)
    have = 0
    while have < count:
        got = readinto(view[have:])
        if not got:
            return None
        have += got
    return buffer


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
        raise RuntimeError('FFmpeg output ended before a complete PPM frame')
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


class LocalModeSource:
    """Launch local mode and receive its newest rendered RGB frame."""

    STARTUP_TIMEOUT = 30.0

    def __init__(self, capture_fps=None):
        self._listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._listener.bind(('127.0.0.1', 0))
        self._listener.listen(1)
        self._listener.settimeout(.2)
        self._condition = threading.Condition()
        self._connection = None
        self._latest = None
        self._closed = False
        self._receiver = threading.Thread(
            target=self._receive, name='v7-local-mode-frames', daemon=True)
        self._receiver.start()
        command = [
            sys.executable, str(ROOT/'main.py'), '--mode', 'local',
            '--local-frame-port', str(self._listener.getsockname()[1]),
            '--local-frame-fps', str(float(capture_fps or 15.0)),
        ]
        try:
            self.process = subprocess.Popen(
                command, cwd=str(ROOT), stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except BaseException:
            self.close()
            raise

    def _receive(self):
        while True:
            with self._condition:
                if self._closed:
                    return
            try:
                connection, _address = self._listener.accept()
            except socket.timeout:
                continue
            except OSError:
                return
            connection.settimeout(2.0)
            with self._condition:
                if self._closed:
                    connection.close()
                    return
                self._connection = connection
            try:
                while True:
                    frame = read_frame(connection)
                    if frame is None:
                        break
                    with self._condition:
                        self._latest = frame
                        self._condition.notify_all()
            except (ConnectionError, OSError, ValueError):
                pass
            finally:
                with self._condition:
                    if self._connection is connection:
                        self._connection = None
                try:
                    connection.close()
                except OSError:
                    pass

    def __call__(self):
        deadline = time.monotonic() + self.STARTUP_TIMEOUT
        with self._condition:
            while self._latest is None and not self._closed:
                if self.process.poll() is not None:
                    raise RuntimeError(
                        'main.py --mode local exited before publishing a frame')
                remaining = deadline-time.monotonic()
                if remaining <= 0:
                    raise TimeoutError(
                        'Timed out waiting for the local-mode video frame')
                self._condition.wait(min(.1, remaining))
            if self._latest is None:
                raise RuntimeError('Local-mode video source has stopped')
            return self._latest.copy()

    def close(self):
        with self._condition:
            if self._closed:
                return
            self._closed = True
            connection = self._connection
            self._condition.notify_all()
        for sock in (connection, self._listener):
            if sock is None:
                continue
            try:
                sock.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
            try:
                sock.close()
            except OSError:
                pass
        process = getattr(self, 'process', None)
        if process is not None and process.poll() is None:
            try:
                process.send_signal(signal.SIGINT)
            except OSError:
                process.terminate()
            try:
                process.wait(timeout=2.0)
            except subprocess.TimeoutExpired:
                process.terminate()
                try:
                    process.wait(timeout=2.0)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=2.0)
        self._receiver.join(timeout=1.0)


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
                  scale_flags='neighbor', output_size=None,
                  preserve_size=False):
    """Capture through ffmpeg's platform fast path.

    mss goes via CoreGraphics on macOS and costs tens of milliseconds a grab.
    ffmpeg uses avfoundation / x11grab / gdigrab and scales before handing the
    frame over, so Python receives a small RGB array and does no image work.

    PPM frames carry their output dimensions in the pipe. The optional fixed
    output size is used by the camera sender to produce the V7 sampling grid in
    one FFmpeg resize; showinfo preserves the pre-scale source geometry for the
    packet aspect code.
    """
    if shutil.which('ffmpeg') is None:
        raise SystemExit('ffmpeg not found. brew install ffmpeg / apt install ffmpeg')
    w = int(width)
    if w < 1:
        raise ValueError('Capture width must be positive')
    fps = 30 if fps is None else fps
    if scale_flags not in ('neighbor', 'area', 'bilinear', 'bicubic', 'lanczos'):
        raise ValueError(f'Unsupported FFmpeg scale flags: {scale_flags}')
    if output_size is not None:
        output_size = tuple(map(int, output_size))
        if len(output_size) != 2 or min(output_size) < 1:
            raise ValueError('Output size must contain two positive dimensions')
        if preserve_size:
            raise ValueError('preserve_size cannot be combined with output_size')

    if spec:
        fmt, src = spec.split(':', 1)
    elif sys.platform == 'darwin':
        fmt, src = 'avfoundation', f'{display if display is not None else 1}:none'
    elif sys.platform.startswith('win'):
        fmt, src = 'gdigrab', 'desktop'
    else:
        fmt, src = 'x11grab', os.environ.get('DISPLAY', ':0.0')

    cmd = ['ffmpeg', '-nostdin', '-loglevel',
           'info' if output_size is not None else 'error', '-f', fmt,
           '-framerate', f'{max(float(fps), 1.0):g}']
    if output_size is not None:
        cmd.append('-nostats')
    if fmt == 'x11grab' and region:
        cmd += ['-video_size', f'{region[2]}x{region[3]}',
                '-i', f'{src}+{region[0]},{region[1]}']
    elif fmt == 'gdigrab' and region:
        cmd += ['-offset_x', str(region[0]), '-offset_y', str(region[1]),
                '-video_size', f'{region[2]}x{region[3]}', '-i', src]
    else:
        cmd += ['-i', src]
    scale = (f'scale={output_size[0]}:{output_size[1]}:flags={scale_flags}'
             if output_size is not None else
             f'scale={w}:-1:flags={scale_flags}')
    video_filter = (f'showinfo=checksum=0,{scale}' if output_size else
                    None if preserve_size else scale)
    # Device timestamps are not necessarily a constant-rate timeline. The
    # default sync mode can emit thousands of duplicates to fill their gaps.
    # This pipe needs exactly one image for each input frame.
    if video_filter is not None:
        cmd += ['-vf', video_filter]
    cmd += ['-pix_fmt', 'rgb24', '-fps_mode', 'passthrough',
            '-c:v', 'ppm', '-f', 'image2pipe', '-an', '-sn', '-']
    errors = _temporary_error_file()
    log_lock = threading.Lock()
    try:
        proc = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stdin=subprocess.DEVNULL,
            stderr=(subprocess.PIPE if output_size is not None else errors),
            bufsize=0, start_new_session=True)
    except BaseException:
        errors.close()
        raise
    closed = threading.Event()
    close_lock = threading.Lock()
    source_size = {'value': None}
    source_size_ready = threading.Event()
    stderr_thread = None

    def save_error_line(line):
        with log_lock:
            errors.write(line)
            errors.flush()

    if output_size is not None:
        def drain_stderr():
            for line in iter(proc.stderr.readline, b''):
                reported_size = _showinfo_source_size(line)
                if reported_size is not None and source_size['value'] is None:
                    source_size['value'] = reported_size
                    source_size_ready.set()
                    continue
                if b'Parsed_showinfo' not in line:
                    save_error_line(line)

        stderr_thread = threading.Thread(target=drain_stderr, daemon=True)
        stderr_thread.start()

    def error_detail():
        with log_lock:
            errors.flush()
            errors.seek(0)
            detail = errors.read().decode('utf-8', errors='replace').strip()
            errors.seek(0, os.SEEK_END)
        return detail

    def grab():
        try:
            frame = _read_ppm(proc.stdout)
            if output_size is None:
                return frame
            if not source_size_ready.wait(5.0):
                raise RuntimeError(
                    'FFmpeg did not report the original capture dimensions')
            return CapturedFrame(frame, source_size['value'], prepared=True)
        except RuntimeError as exc:
            with close_lock:
                if closed.is_set():
                    raise
                detail = error_detail()
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
                if stderr_thread is not None:
                    stderr_thread.join(timeout=2)
                with log_lock:
                    errors.close()
        # Closing stdout is left to the capture worker after its read exits.

    grab.proc = proc
    grab.close = close
    grab.paced = True  # the device already paces the pipe; drain it continuously
    return grab


def camera_source(index=0, fps=30, width=320, spec=None,
                  scale_flags='neighbor', output_size=None,
                  preserve_size=False):
    """Webcam through FFmpeg, optionally scaled to the fixed V7 grid."""
    if sys.platform == 'darwin':
        spec = spec or f'avfoundation:{index}'
    elif sys.platform.startswith('win'):
        spec = spec or 'dshow:video=Integrated Camera'
    else:
        spec = spec or f'v4l2:/dev/video{index}'
    return ffmpeg_source(spec, fps, width=width, scale_flags=scale_flags,
                         output_size=output_size, preserve_size=preserve_size)


def screen_capture_source(fps, region=None, display=None, width=320,
                          spec=None, scale_flags='neighbor',
                          preserve_size=False):
    """Compatibility adapter for the former modem_screen API."""
    # AVFoundation's device numbering is machine-dependent. The old screen
    # path selected the first screen capture device; do not inherit the
    # camera-oriented default used by the generic FFmpeg helper.
    if display is None and sys.platform == 'darwin':
        display = 0
    return ffmpeg_source(spec, fps, region=region, display=display,
                         width=width, scale_flags=scale_flags,
                         preserve_size=preserve_size)


_NETWORK_SCHEMES = frozenset({
    'http', 'https', 'rtsp', 'rtsps', 'rtmp', 'rtmps', 'udp', 'tcp', 'srt',
    'rist', 'rtp', 'rtmpe', 'rtmpt',
})
_LIVE_SCHEMES = _NETWORK_SCHEMES - {'http', 'https'}
_FINITE_VIDEO_SUFFIXES = frozenset({
    '.avi', '.m4v', '.mkv', '.mov', '.mp4', '.mpeg', '.mpg', '.ts', '.webm',
    '.wmv',
})


def _is_stream_url(source):
    return urlsplit(str(source)).scheme.lower() in _NETWORK_SCHEMES


def _is_hls_url(source):
    parsed = urlsplit(str(source))
    return (parsed.scheme.lower() in ('http', 'https') and
            parsed.path.lower().endswith(('.m3u8', '.m3u')))


def _realtime_input_options(source, live):
    """Pace finite files and timestamped HTTP media/HLS to their media clock."""
    if not live or _is_hls_url(source):
        return ['-re']
    parsed = urlsplit(str(source))
    if (parsed.scheme.lower() in ('http', 'https') and
            os.path.splitext(parsed.path)[1].lower() in
            _FINITE_VIDEO_SUFFIXES):
        return ['-re']
    return []


def video_source_loops(source, live=None):
    """Whether the live sender's video-source policy repeats this source."""
    is_stream = _is_stream_url(source)
    is_live = (urlsplit(str(source)).scheme.lower() in _LIVE_SCHEMES
               if live is None else bool(live))
    if is_live and not is_stream:
        raise ValueError('--video-live requires a stream URL')
    return not is_live


HD_HEIGHT = 720


def untagged_hd_matrix(source):
    """'bt709' for a local video file whose stream carries no colour-matrix
    tag and is HD (720 lines or more), else None.

    FFmpeg converts an untagged stream to RGB with the SD (BT.601) matrix at
    any size; HD material is BT.709 by convention, and reading it as BT.601
    shifts saturated colours by up to 25 of 255. Tagged streams are converted
    correctly already. One ffprobe call when the source opens; any failure
    (no ffprobe, a stream URL, an unreadable file) leaves FFmpeg's default.
    """
    source = str(source)
    if _is_stream_url(source) or shutil.which('ffprobe') is None:
        return None
    try:
        probe = subprocess.run(
            ['ffprobe', '-v', 'error', '-select_streams', 'v:0',
             '-show_entries', 'stream=height,color_space',
             '-of', 'default=nw=1', source],
            capture_output=True, text=True, timeout=5)
    except Exception:                     # no probe: keep FFmpeg's default
        return None
    if probe.returncode != 0:
        return None
    fields = dict(line.split('=', 1) for line in probe.stdout.splitlines()
                  if '=' in line)
    try:
        height = int(fields.get('height', ''))
    except ValueError:
        return None
    tagged = fields.get('color_space', 'unknown') not in ('unknown', '')
    return 'bt709' if height >= HD_HEIGHT and not tagged else None


def video_scale_filter(source, width, scale_flags, preserve_size):
    """The -vf value for a video file or stream (None: no filter needed)."""
    matrix = untagged_hd_matrix(source)
    options = []
    if not preserve_size:
        options += [f'{int(width)}:-2', f'flags={scale_flags}']
    if matrix:
        options.append(f'in_color_matrix={matrix}')
    return 'scale='+':'.join(options) if options else None


def video_source(source, loop=None, realtime=None, width=320,
                 scale_flags='bicubic', live=None, preserve_size=False,
                 start=0.0):
    """Read a local video file in a real-time loop or a live stream URL.

    Local files and HTTP(S) media URLs loop and are paced with ``-re`` by
    default. Native live protocols are read as delivered. Set ``live=True``
    for live HLS/HTTP URLs; HLS manifests and direct finite HTTP media remain
    paced to their media timestamps, play once, and end cleanly. PPM carries
    each output frame's dimensions. A local file is probed once for its colour
    matrix tag (see untagged_hd_matrix).

    A positive ``start`` (seconds) reads from that position to the end of the
    file once and then ends cleanly. FFmpeg cannot combine an input seek with
    ``-stream_loop`` (the repeat lands at the wrong place with broken
    timestamps), so FilePlayback reopens a looping reader after that pass.
    """
    start = max(0.0, float(start or 0.0))
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

    is_live = not video_source_loops(source, live)
    parsed_source = urlsplit(source)
    finite_http_media = (
        parsed_source.scheme.lower() in ('http', 'https') and
        os.path.splitext(parsed_source.path)[1].lower() in
        _FINITE_VIDEO_SUFFIXES)
    if loop is None:
        loop = not is_live
    if start > 0:
        loop = False
    if realtime is None:
        realtime = not is_live or finite_http_media

    cmd = ['ffmpeg', '-nostdin', '-loglevel', 'error']
    if loop:
        cmd += ['-stream_loop', '-1']
    if realtime or _is_hls_url(source):
        cmd += ['-re']
    if is_stream:
        # Fail a stalled network read instead of leaving the capture worker
        # blocked forever during shutdown or source loss.
        cmd += ['-rw_timeout', '10000000']
    if start > 0:
        cmd += ['-ss', f'{start:.3f}']
    cmd += ['-i', source]
    video_filter = video_scale_filter(source, width, scale_flags,
                                      preserve_size)
    if video_filter is not None:
        cmd += ['-vf', video_filter]
    cmd += ['-fps_mode', 'passthrough', '-pix_fmt', 'rgb24',
            '-c:v', 'ppm', '-f', 'image2pipe', '-an', '-sn', '-']

    errors = _temporary_error_file()
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
                return_code = proc.poll()
                if return_code is None:
                    try:
                        return_code = proc.wait(timeout=1.0)
                    except subprocess.TimeoutExpired:
                        pass
                if return_code == 0:
                    raise CaptureEndOfStream from exc
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

    def release():
        # For FilePlayback, on the reading thread once its read has returned.
        proc.stdout.close()

    grab.proc = proc
    grab.close = close
    grab.release = release
    # File input is paced by -re; a live URL is paced by its own arrival rate.
    # Drain either continuously so the newest-frame mailbox stays current.
    grab.paced = True
    return grab


def is_file_video_source(source):
    """Whether a video source is a local file (pause/seek/resume apply)."""
    source = str(source or '').strip()
    return bool(source) and not _is_stream_url(source)


def probe_duration(source, run=None):
    """Duration of a local video file in seconds, or None when unknown."""
    if (_is_stream_url(source) or
            not os.path.isfile(os.path.expanduser(str(source))) or
            shutil.which('ffprobe') is None):
        return None
    try:
        probe = (run or subprocess.run)(
            ['ffprobe', '-v', 'error', '-show_entries', 'format=duration',
             '-of', 'default=nw=1:nk=1', os.path.expanduser(str(source))],
            capture_output=True, text=True, timeout=5)
        duration = float((probe.stdout or '').strip().splitlines()[0])
    except Exception:                    # no duration: position only
        return None
    if probe.returncode != 0 or not np.isfinite(duration) or duration <= 0:
        return None
    return duration


def probe_frames(source, run=None):
    """(frame rate, frame count) of a local file's first video stream.

    Either is None when ffprobe does not report it (Matroska, for one, stores
    no frame count). The rate is the stream's average frame rate.
    """
    if (_is_stream_url(source) or
            not os.path.isfile(os.path.expanduser(str(source))) or
            shutil.which('ffprobe') is None):
        return None, None
    try:
        probe = (run or subprocess.run)(
            ['ffprobe', '-v', 'error', '-select_streams', 'v:0',
             '-show_entries', 'stream=avg_frame_rate,nb_frames',
             '-of', 'default=nw=1', os.path.expanduser(str(source))],
            capture_output=True, text=True, timeout=5)
        if probe.returncode != 0:
            return None, None
        fields = dict(line.split('=', 1)
                      for line in (probe.stdout or '').splitlines()
                      if '=' in line)
    except Exception:                    # no probe: position from wall time
        return None, None
    rate = count = None
    try:
        top, _slash, bottom = fields.get('avg_frame_rate', '').partition('/')
        rate = float(top)/float(bottom or 1)
        if not np.isfinite(rate) or rate <= 0:
            rate = None
    except (ValueError, ZeroDivisionError):
        rate = None
    try:
        count = int(fields.get('nb_frames', ''))
        if count <= 0:
            count = None
    except ValueError:
        count = None
    return rate, count


class FilePlayback:
    """Pause, seek and restart for a looping local video file.

    The readers are FFmpeg processes paced by ``-re`` on their own wall
    clock, so a reader cannot be held: stopping the pipe would make it race to
    catch up afterwards. Instead the reader is closed and a new one is opened
    at the wanted position. ``open_reader(start)`` returns a frame reader: a
    looping one for ``start == 0``, and for a later start one that plays to
    the end of the file once and raises CaptureEndOfStream, after which the
    looping reader is opened from the beginning (the usual repeat).

    While paused the call returns the held picture again at a short interval,
    so the capture thread keeps a fresh frame and the sender keeps emitting
    packets of it. A seek made while paused reads one picture at the new
    position and holds that.

    Position is counted from the pictures actually read: the start of the
    current reader plus (frames it has delivered)/``fps``. The readers pass
    every decoded source frame through once (``-fps_mode passthrough``) and
    the capture thread reads them all, so the count is exact in source
    frames, and ``fps`` is the file's frame rate. It is reset by a seek, a
    restart and the end of a pass from a seek position. The looping reader
    repeats the file without a mark in the pipe, so its count wraps every
    ``loop_frames`` (the file's frame count) when that is known, and
    otherwise every ``duration``. Wall time is not used.

    Only when ``fps`` is unknown (no ffprobe, or no rate in the file) does
    the position fall back to the wall time the reader has been delivering.

    A reader's ``close()`` must be safe from another thread and make a blocked
    read fail; its optional ``release()`` is called on the reading thread once
    the read has returned.
    """

    paced = True

    def __init__(self, open_reader, duration=None, start=0.0, reader=None,
                 clock=time.monotonic, idle=0.05, on_close=None, fps=None,
                 loop_frames=None):
        self._open_reader = open_reader
        self.fps = (float(fps) if fps and np.isfinite(fps) and fps > 0
                    else None)
        self._loop_frames = (int(loop_frames) if loop_frames and
                             loop_frames > 0 else None)
        self._frames = 0
        self.duration = (float(duration) if duration and
                         np.isfinite(duration) and duration > 0 else None)
        self._clock = clock
        self._idle = float(idle)
        self._on_close = on_close
        self._cond = threading.Condition()
        self._generation = 0
        self._reader_generation = 0
        self._reader = None
        self._reader_start = 0.0
        self._first_frame_at = None
        self._paused = False
        self._still = False
        self._held = None
        self._closed = False
        self._reading = False
        self._position = self._clamp(start)
        if reader is None:
            reader = open_reader(self._position)
        self._reader = reader
        self._reader_start = self._position

    def _clamp(self, position):
        try:
            position = float(position)
        except (TypeError, ValueError):
            return 0.0
        if not np.isfinite(position) or position <= 0:
            return 0.0
        if self.duration is not None and position >= self.duration:
            return 0.0                   # past the end: from the beginning
        return position

    def _position_locked(self):
        if self._first_frame_at is None:
            return self._position
        looping = self._reader_start == 0
        if self.fps is not None:
            frames = self._frames
            if looping and self._loop_frames:
                return (frames % self._loop_frames)/self.fps
            position = self._reader_start+frames/self.fps
        else:
            position = self._reader_start+max(
                0.0, self._clock()-self._first_frame_at)
        if self.duration is not None:
            if looping:
                position %= self.duration
            else:
                position = min(position, self.duration)
        return position

    @property
    def position(self):
        with self._cond:
            return self._position_locked()

    @property
    def paused(self):
        with self._cond:
            return self._paused

    def status(self):
        with self._cond:
            return {'position': self._position_locked(),
                    'duration': self.duration, 'paused': self._paused}

    def _interrupt_locked(self, position):
        """Retire the current reader; the reading thread reopens at position."""
        self._position = position
        self._first_frame_at = None
        self._generation += 1
        self._cond.notify_all()
        return self._reader

    @staticmethod
    def _close_reader(reader):
        close = getattr(reader, 'close', None)
        if close is not None:
            try:
                close()
            except Exception:
                pass

    @staticmethod
    def _release_reader(reader):
        release = getattr(reader, 'release', None)
        if release is not None:
            try:
                release()
            except Exception:
                pass

    def pause(self):
        with self._cond:
            if self._paused or self._closed:
                return
            self._paused = True
            reader = self._interrupt_locked(self._position_locked())
        self._close_reader(reader)

    def play(self):
        with self._cond:
            if not self._paused or self._closed:
                return
            self._paused = False
            self._still = False
            self._cond.notify_all()

    def seek(self, position):
        with self._cond:
            if self._closed:
                return
            self._still = self._paused
            reader = self._interrupt_locked(self._clamp(position))
        self._close_reader(reader)

    def restart(self):
        self.seek(0.0)

    def __call__(self):
        while True:
            stale = None
            with self._cond:
                if self._closed:
                    raise RuntimeError('video playback is closed')
                generation = self._generation
                reader = self._reader
                if (reader is not None and
                        self._reader_generation != generation):
                    stale, reader, self._reader = reader, None, None
                if self._paused and self._held is None:
                    self._still = True   # paused before any picture: get one
                still = self._still
                waiting = self._paused and not still
                if waiting and stale is None:
                    self._cond.wait(self._idle)
                    if (self._generation == generation and self._paused and
                            not self._still and not self._closed):
                        return self._held
                    continue
                start = self._position
            if stale is not None:
                self._close_reader(stale)
                self._release_reader(stale)
            if waiting:
                continue
            if reader is None:
                reader = self._open_reader(start)
                with self._cond:
                    if self._closed or self._generation != generation:
                        stale = reader
                    else:
                        self._reader = reader
                        self._reader_generation = generation
                        self._reader_start = start
                        self._first_frame_at = None
                        self._frames = 0
                if stale is not None:
                    self._close_reader(stale)
                    self._release_reader(stale)
                    continue
            ended = False
            frame = None
            with self._cond:
                self._reading = True
            try:
                frame = reader()
            except CaptureEndOfStream:
                ended = True
            except Exception:
                with self._cond:
                    self._reading = False
                    interrupted = (self._closed or
                                   self._generation != generation)
                    if interrupted and self._reader is reader:
                        self._reader = None
                if not interrupted:
                    raise
                self._release_reader(reader)
                continue
            ended = ended or frame is None
            with self._cond:
                self._reading = False
                current = (not self._closed and
                           self._generation == generation)
                if ended:
                    if self._reader is reader:
                        self._reader = None
                    if current and start > 0:
                        # The pass from a seek position reached the end of
                        # the file: repeat from the beginning.
                        self._position = 0.0
                        self._first_frame_at = None
                elif current:
                    if self._first_frame_at is None:
                        self._first_frame_at = self._clock()
                    self._frames += 1
                    self._held = frame
                    if still:
                        # One picture at the new position, then hold it.
                        self._still = False
                        self._first_frame_at = None
                        self._generation += 1
            if ended:
                self._close_reader(reader)
                self._release_reader(reader)
                if current and start == 0:
                    # The looping reader does not end by itself.
                    raise CaptureEndOfStream
                continue
            if current:
                return frame

    def close(self):
        with self._cond:
            already = self._closed
            self._closed = True
            self._generation += 1
            reader = self._reader
            reading = self._reading
            if not reading:
                self._reader = None
            self._cond.notify_all()
        if reader is not None:
            self._close_reader(reader)
            if not reading:
                self._release_reader(reader)
        if not already and self._on_close is not None:
            self._on_close()


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
        self._first_frame = None
        self._error = None
        self._ended = False
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
                    if self._first_frame is None:
                        self._first_frame = frame
                    self._updated = time.monotonic()
                    self.grabs += 1
                self._ready.set()
                rest = self._period-(time.perf_counter()-began)
                if rest > 0 and not getattr(self._grab, 'paced', False):
                    self._stop.wait(rest)
        except CaptureEndOfStream:
            with self._lock:
                self._ended = True
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

    @property
    def first_frame(self):
        """The first captured frame, retained for soundtrack-start pairing."""
        with self._lock:
            return self._first_frame

    @property
    def ended(self):
        with self._lock:
            return self._ended

    def __call__(self):
        with self._lock:
            if self._error is not None:
                raise RuntimeError(f'Capture failed: {self._error}') from self._error
            if self._ended:
                return None
            if self._updated is not None and time.monotonic()-self._updated > max(5, 3*self._period):
                raise RuntimeError('Capture stalled: no new frame for over '
                                   f'{max(5, 3*self._period):g} seconds')
            return self._latest

    def close(self):
        self._stop.set()
        proc = getattr(self._grab, 'proc', None)
        close = getattr(self._grab, 'close', None)
        if close is not None:
            close()
        elif proc is not None:
            proc.terminate()
            try:
                proc.wait(timeout=2)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=2)
        self._thread.join(timeout=2)


# --------------------------------------------------------------------------
