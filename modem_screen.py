"""
modem_screen.py -- send a live source over the v3 audio modem.

The bake path (main.py --mode modem) reads pre-rendered composites from disk.
This does the same job from anything live, using the same capture sources as
scope_screen.py:

    python modem_screen.py --source ffmpeg --device "BlackHole 2ch"
    python modem_screen.py --source screen --region 0,0,800,600
    python modem_screen.py --source video --file clip.mp4
    python modem_screen.py --source camera
    python modem_screen.py --source test --write out.wav

Receive with:

    python utilities/modem_v3_check.py live-receive --device "BlackHole 2ch"

The receiver takes no --preset or --profile: the wire is fixed and the profile
is declared in the header.

BE REALISTIC ABOUT THE RESOLUTION. The picture is whatever the profile says:
40x48 colour for 'color-dct' (DCT) or 'color-wavelet' (wavelet) -- each
sampled 2x finer and truncated, so the source bake is 80x96. That
is a silhouette, a face, a moving shape, a lava lamp. It is not a desktop, and
text will not survive.

Unlike scope_screen this wants COLOUR, so ffmpeg is asked for rgb24 rather than
gray. Every source, including cameras, is stretched to the native 80x96 grid.
The source aspect's nearest preset travels in the header and is restored only
for display. --aspect overrides automatic selection; no bars are transmitted.

Frame rate is set by the wire, not by the capture: a packet is 3200 samples,
so the modem consumes one picture every 3200 samples and the capture is
throttled to match. Grabbing faster only wastes CPU; grabbing slower repeats
the last frame rather than stalling the stream.

How many pictures a second that is depends on the rate the output device is
already set to -- 17.34 at 48 kHz, 15.93 at 44.1 kHz -- and no rate is
requested of it. The real figure is printed once the stream is open, and the
capture is re-paced to it then. The receiver does not need to be told either:
it reads the cadence off the preamble.

Screen capture uses FFmpeg's platform backend and scales before piping to Python.
"""
import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import time

import numpy as np
from PIL import Image, ImageOps, ImageEnhance

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from animation_modem import transport3 as V3                      # noqa: E402
from animation_modem.audio_common import (device, pair, pcm,      # noqa: E402
                                          wire_notice)
from animation_modem.core import (REFERENCE_RATE as RATE,        # noqa: E402
                                  SourceCoder, bound_emission)
from animation_modem.wavelet import WaveletCoder                 # noqa: E402
from animation_modem.imaging import (DEFAULT_PROFILE, PROFILES, burn_counters,  # noqa: E402
                                     fit_shapes, image_values,
                                      source_size, ENCODE_FILTERS,
                                     plane_grids, plane_shapes, wire_profiles)
from animation_modem.playback import PacketOutput                 # noqa: E402
from animation_modem.aspect import ASPECT_CHOICES, aspect_code     # noqa: E402


# --------------------------------------------------------------------------
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


def screen_capture_source(fps, region=None, display=None, width=320, spec=None):
    """Keep screen capture and full-resolution scaling outside the audio process."""
    if sys.platform == 'darwin' and spec is None and display is None:
        # AVFoundation indices include cameras and vary with connected devices.
        # Discover the first screen rather than accidentally opening a webcam.
        if shutil.which('ffmpeg') is None:
            raise SystemExit('ffmpeg not found. brew install ffmpeg')
        listing = subprocess.run(
            ['ffmpeg', '-nostdin', '-hide_banner', '-f', 'avfoundation',
             '-list_devices', 'true', '-i', ''],
            capture_output=True, text=True, timeout=15)
        match = re.search(r'\[(\d+)\]\s+Capture screen', listing.stderr)
        if match is None:
            raise SystemExit('No FFmpeg screen device found; specify --display or '
                             '--ffmpeg-input.\n' + listing.stderr)
        display = int(match.group(1))
    return ffmpeg_source(spec, fps, region, display, width)


def ffmpeg_source(spec, fps, region=None, display=None, width=320,
                  video_size=None, raw_output=False, scale_flags=None):
    """Capture through ffmpeg's platform fast path.

    mss goes via CoreGraphics on macOS and costs tens of milliseconds a grab.
    ffmpeg uses avfoundation / x11grab / gdigrab and scales before handing the
    frame over, so Python receives a small RGB array and does no image work.

    PPM frames carry their dimensions in the pipe. Scale by width and preserve
    the actual input aspect, including for cameras unrelated to screen size.
    """
    if shutil.which('ffmpeg') is None:
        raise SystemExit('ffmpeg not found. brew install ffmpeg / apt install ffmpeg')
    w = int(width) if width is not None else None
    if w is not None and w < 1:
        raise ValueError('Capture width must be positive')

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
        if video_size is not None and fmt in ('avfoundation', 'v4l2', 'dshow'):
            cmd += ['-video_size', f'{video_size[0]}x{video_size[1]}']
        cmd += ['-i', src]
    filters = []
    if region and fmt != 'x11grab':
        filters.append(f'crop={region[2]}:{region[3]}:{region[0]}:{region[1]}')
    if w is not None:
        scale = f'scale={w}:-1'
        if scale_flags:
            scale += f':flags={scale_flags}'
        filters.append(scale)
    if filters:
        cmd += ['-vf', ','.join(filters)]
    cmd += ['-pix_fmt', 'rgb24', '-fps_mode', 'passthrough']
    raw_shape = None
    if raw_output and video_size is not None and w is not None:
        out_h = max(2, int(round(w*video_size[1]/video_size[0]))//2*2)
        # The explicit output size makes rawvideo framing unambiguous and
        # avoids the PPM image2pipe muxer/header work for camera frames.
        if filters:
            scale = f'scale={w}:{out_h}'
            if scale_flags:
                scale += f':flags={scale_flags}'
            cmd[cmd.index('-vf')+1] = scale
        else:
            scale = f'scale={w}:{out_h}'
            if scale_flags:
                scale += f':flags={scale_flags}'
            cmd += ['-vf', scale]
        raw_shape = (out_h, w)
        cmd += ['-f', 'rawvideo']
    else:
        cmd += ['-c:v', 'ppm', '-f', 'image2pipe']
    cmd += ['-an', '-sn', '-']
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
            if raw_shape is not None:
                count = raw_shape[0]*raw_shape[1]*3
                data = _read_exact(proc.stdout, count)
                if data is None:
                    raise RuntimeError('FFmpeg capture ended')
                return np.frombuffer(data, np.uint8).reshape(*raw_shape, 3)
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


def _lowest_camera_mode(fmt, source, fps):
    """Return the smallest advertised camera mode supporting ``fps``.

    FFmpeg exposes the mode list for AVFoundation, V4L2 and DirectShow. The
    text differs slightly, so accept both AVFoundation's ``@[15 30]fps`` form
    and backend lines that put the rate after the dimensions.
    """
    if shutil.which('ffmpeg') is None:
        return None
    try:
        probe = subprocess.run(
            ['ffmpeg', '-nostdin', '-hide_banner', '-f', fmt,
             '-list_options', 'true', '-i', source],
            capture_output=True, text=True, timeout=15)
    except (OSError, TypeError, subprocess.SubprocessError):
        return None
    modes = []
    for line in (probe.stderr or '').splitlines():
        match = re.search(r'(\d+)x(\d+)', line)
        if not match:
            continue
        # Do not treat the width and height as frame rates. AVFoundation puts
        # rates in brackets; V4L2/DirectShow commonly put them later on the
        # same line or on a following mode line.
        tail = line[match.end():]
        rates = [float(value) for value in re.findall(r'\d+(?:\.\d+)?', tail)]
        size = (int(match.group(1)), int(match.group(2)))
        if any(abs(rate-fps) < .01 for rate in rates):
            modes.append(size)
    return min(modes, key=lambda size: size[0]*size[1]) if modes else None


def _lowest_camera_mode_any(fmt, source):
    """Return ``(width, height), fps`` for the smallest-rate camera mode.

    When the caller has no capture-rate preference, choose the lowest
    advertised FPS first, then the smallest frame at that rate.  This avoids
    opening a fast/high-resolution sensor mode merely because the encoder is
    slower than the camera.
    """
    if shutil.which('ffmpeg') is None:
        return None
    try:
        probe = subprocess.run(
            ['ffmpeg', '-nostdin', '-hide_banner', '-f', fmt,
             '-list_options', 'true', '-i', source],
            capture_output=True, text=True, timeout=15)
    except (OSError, TypeError, subprocess.SubprocessError):
        return None
    modes = []
    for line in (probe.stderr or '').splitlines():
        match = re.search(r'(\d+)x(\d+)', line)
        if not match:
            continue
        rates = [float(value) for value in
                 re.findall(r'\d+(?:\.\d+)?', line[match.end():])]
        size = (int(match.group(1)), int(match.group(2)))
        modes.extend((rate, size) for rate in rates if rate > 0)
    if not modes:
        return None
    rate, size = min(modes, key=lambda item: (item[0],
                                               item[1][0]*item[1][1]))
    return size, rate


def camera_source(index=0, fps=30, width=320, spec=None,
                  scale_flags=None):
    """Capture the smallest supported mode, then resize before the Python pipe.

    Output scaling preserves the camera aspect. Keep full-resolution RGB traffic
    out of Python's audio-producing process, and do not make the camera deliver
    a larger sensor mode than the modem can use.
    """
    if sys.platform == 'darwin':
        # AVFoundation's input is a video:audio pair.  Leaving the audio side
        # implicit can make FFmpeg select a different format and reject a
        # camera framerate that the video device explicitly advertises.
        spec = spec or f'avfoundation:{index}:none'
        # Select the smallest sensor mode before opening the device.  The
        # later `scale=` filter only limits pipe traffic; without this probe
        # AVFoundation may open a 1080p/portrait mode and do the expensive
        # capture first.  If probing is unavailable, retain FFmpeg's default.
        source_index = spec.split(':', 1)[1].split(':', 1)[0]
        if fps is None:
            selected = _lowest_camera_mode_any(
                'avfoundation', f'{source_index}:none')
            mode, fps = selected if selected else (None, 30)
        else:
            mode = _lowest_camera_mode('avfoundation',
                                       f'{source_index}:none', fps)
        return ffmpeg_source(spec, fps, width=width, video_size=mode,
                             raw_output=True, scale_flags=scale_flags)
    elif sys.platform.startswith('win'):
        spec = spec or 'dshow:video=Integrated Camera'
    else:
        spec = spec or f'v4l2:/dev/video{index}'
    fmt, source = spec.split(':', 1)
    if fps is None:
        selected = _lowest_camera_mode_any(fmt, source)
        mode, fps = selected if selected else (None, 30)
    else:
        mode = _lowest_camera_mode(fmt, source, fps)
    return ffmpeg_source(spec, fps, width=width, video_size=mode,
                         raw_output=True, scale_flags=scale_flags)


def video_source(path, loop=True, realtime=True):
    """Decode a file through ffmpeg, optionally looping.

    `realtime` paces ffmpeg to the wall clock with -re, which is right when the
    packets are going to a device. It is wrong when rendering to a WAV as fast
    as the CPU allows: the renderer outruns the decoder and every grab returns
    the same frame. Rendering therefore reads the pipe sequentially instead,
    one decoded frame per packet, so the clip plays at the wire rate.
    """
    if shutil.which('ffmpeg') is None:
        raise SystemExit('ffmpeg not found. brew install ffmpeg / apt install ffmpeg')
    if not os.path.exists(path):
        raise SystemExit(f'No such file: {path}')
    w = 320
    probe = subprocess.run(
        ['ffprobe', '-v', 'error', '-select_streams', 'v:0', '-show_entries',
         'stream=width,height', '-of', 'csv=p=0', path],
        capture_output=True, text=True)
    try:
        sw, sh = [int(x) for x in probe.stdout.strip().split(',')[:2]]
    except Exception:
        sw, sh = 16, 9
    h = max(2, int(round(w*sh/float(sw)))//2*2)
    state = {'proc': None}

    def start():
        cmd = ['ffmpeg', '-loglevel', 'error']
        if loop:
            cmd += ['-stream_loop', '-1']
        if realtime:
            cmd += ['-re']
        cmd += ['-i', path, '-vf', f'scale={w}:{h}', '-pix_fmt', 'rgb24',
                '-f', 'rawvideo', '-an', '-sn', '-']
        state['proc'] = subprocess.Popen(cmd, stdout=subprocess.PIPE,
                                         stderr=subprocess.DEVNULL, bufsize=0)
    start()
    nbytes = w*h*3
    last = [None]

    def grab():
        buf = _read_exact(state['proc'].stdout, nbytes)
        if buf is None:
            if last[0] is None:
                raise SystemExit(f'ffmpeg produced no frames from {path}')
            return last[0]
        last[0] = np.frombuffer(buf, np.uint8).reshape(h, w, 3)
        return last[0]
    grab.proc = state['proc']
    grab.sequential = True          # one decoded frame per call; do not throttle
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
# Frame preparation
# --------------------------------------------------------------------------

def fitter(profile, rotate=0, mirror=False, letterbox=False, aspect='auto',
           encode_filter='lanczos'):
    """Full source -> native 80x96 image with aspect metadata.

    Striding to roughly 4x the target keeps large desktop captures cheap.
    Measure geometry before that approximation and account for rotation.
    Explicit letterbox=True is retained for old diagnostic callers only;
    streaming always stretches the whole picture.
    """
    # Every codec samples the native grid. Display geometry is metadata, not
    # padding; derive it from the actual capture, including camera renegotiation.
    size = source_size(profile)
    resample = ENCODE_FILTERS[encode_filter]
    target = max(size)*4

    def prepare(raw):
        raw = np.asarray(raw, np.uint8)
        h, w = raw.shape[:2]
        code = aspect_code((h, w) if rotate in (90, 270) else (w, h), aspect)
        step = max(1, min(w//target, h//target))
        if step > 1:
            raw = raw[::step, ::step]
        im = Image.fromarray(raw, 'RGB')
        if rotate:
            im = im.rotate(-rotate, expand=True)
        if mirror:
            im = ImageOps.mirror(im)
        if letterbox:
            im = ImageOps.pad(im, size, resample, color=(1, 1, 1))
        else:
            im = im.resize(size, resample)

        # Color enhancements to sharpen outlines before DCT frequency truncation
        im = ImageEnhance.Color(im).enhance(1.2)
        im = ImageEnhance.Contrast(im).enhance(1.1)
        im = ImageEnhance.Brightness(im).enhance(1.1)
        im.info['aspect_code'] = code
        return im

    return prepare


def build(args):
    """Layout and coder, warning loudly if the picture had to shrink.

    fit_shapes quietly scales the planes down until they fit the wire, so a
    profile that asks for more coefficients than the wire carries still runs --
    it just sends a much smaller picture than the profile names. Nothing says
    so, and silent resolution loss is worse than an error.
    """
    v6_profiles = ('v6-dct', 'v6-wavelet', 'v6-repeat-dct',
                   'v6-repeat-wavelet')
    if args.profile not in PROFILES and args.profile not in v6_profiles:
        raise SystemExit(f'Unknown profile {args.profile}')
    wire = getattr(args, 'wire', 'wide')
    if wire == 'tape' and args.profile not in ('hd-dwt', 'tape-80x60'):
        raise SystemExit('--wire tape requires hd-dwt or tape-80x60')
    if wire == 'v6' and args.profile not in ('v6-dct', 'v6-wavelet'):
        raise SystemExit('--wire v6 requires v6-dct or v6-wavelet')
    if wire == 'v6-repeat' and args.profile not in ('v6-repeat-dct',
                                                    'v6-repeat-wavelet'):
        raise SystemExit('--wire v6-repeat requires a v6-repeat profile')
    if args.profile in v6_profiles:
        repeat = args.profile.startswith('v6-repeat-')
        if wire != ('v6-repeat' if repeat else 'v6'):
            raise SystemExit(f'--profile {args.profile} requires the matching wire')
        from animation_modem.v6 import coder_for as v6_coder_for, full_repeat_coder
        layout = V3.WIRE_V6_REPEAT if repeat else V3.WIRE_V6
        coder = (full_repeat_coder(args.profile.removeprefix('v6-repeat-'))
                 if repeat else v6_coder_for(args.profile.removeprefix('v6-')))
        coder.slots(layout)  # precompute the diversity assignment before live output
        copy_text = ('full coefficient copies' if repeat else
                     'coarse Y/Cb/Cr copies')
        print(f'{args.profile}: {coder.n_orig} analog values + '
              f'{len(coder.copy_of)} {copy_text} -> decodes 80x96 '
              f'@ {layout.fps:.2f} fps.', file=sys.stderr)
        return layout, coder, coder.grids
    
    # v5 profile uses different wire layout
    if args.profile == 'hd-dwt':
        layout = V3.WIRE_TAPE if wire == 'tape' else V3.WIRE_HD
        # wavelet.hd_dwt_coder() is the one constructor, shared with the
        # receiver and tools: both ends must build it identically.
        if wire == 'tape':
            from animation_modem.wavelet import tape_80x96_coder
            coder = tape_80x96_coder()
        else:
            from animation_modem.wavelet import hd_dwt_coder
            coder = hd_dwt_coder()
        shapes = coder.shapes
        # Lay out the slots now (~20 ms, cached after): computed lazily on the
        # first encode it would cost the first live packet its deadline.
        coder.slots(layout)
        if wire == 'tape':
            print(f'tape-80x96: {coder.n_orig} luma-heavy DCT values + '
                  f'{len(coder.copy_of)} opposite-leg copies -> decodes 80x96 '
                  f'@ {layout.fps:.1f} fps.', file=sys.stderr)
        else:
            print(f'hd-dwt: {coder.n_orig} CDF 9/7 values (half-res pyramid of the '
                  f'80x96 grid) + {len(coder.copy_of)} repeat copies -> decodes 80x96 '
                  f'@ {layout.fps:.1f} fps.', file=sys.stderr)
        return layout, coder, shapes
    
    # v3/v4 profiles
    # Fail here rather than on the first encode. The header spends two bits on
    # the profile, so only the entries in PROFILE_CODES can be named on the
    # wire; a profile that exists in PROFILES but has no code would otherwise
    # run all the way to V3.encode before raising, after the device was already
    # open.
    try:
        V3.profile_code(args.profile)
    except ValueError as exc:
        raise SystemExit(str(exc)) from None
    layout = V3.WIRE_TAPE_25 if args.profile == 'tape-80x60' else V3.WIRE
    wanted = plane_shapes(args.profile)
    grids = plane_grids(args.profile)
    shapes = fit_shapes(wanted, layout.capacity)
    if shapes != wanted:
        grids = shapes          # see coder_for: a shrunk corner is not a corner
    if args.profile == 'tape-80x60':
        from animation_modem.wavelet import tape_80x60_coder
        coder = tape_80x60_coder()
        shapes, grids = coder.shapes, coder.grids
    else:
        coder_cls = WaveletCoder if 'wavelet' in args.profile else SourceCoder
        coder = coder_cls(shapes, grids=grids)
    if coder.truncated:
        print(f'{args.profile}: sampling {grids[0][1]}x{grids[0][0]} and sending '
              f'the low-frequency corner in {coder.count} slots. The profile is '
              f'declared in the header, so the receiver needs no argument.',
              file=sys.stderr)
    asked = int(sum(np.prod(s) for s in wanted))
    if coder.count < asked:
        got = (shapes[0][1], shapes[0][0])
        print(f'WARNING: the wire holds {layout.capacity} values, profile '
              f'{args.profile} wants {asked}. Picture shrunk to '
              f'{got[0]}x{got[1]} ({coder.count} coefficients). Raise '
              f'image_symbols in transport3.WIRE, or use a smaller profile.',
              file=sys.stderr)
    return layout, coder, grids


def wire_profile_code(profile):
    """V6 reuses codes 0/1 inside its distinct frame geometry."""
    if profile == 'v6-dct':
        return 0
    if profile == 'v6-wavelet':
        return 1
    if profile in ('v6-repeat-dct', 'v6-repeat-wavelet'):
        return 0 if profile.endswith('dct') else 1
    return V3.profile_code(profile)


def source_for(args, fps):
    region = _region(args.region)
    if args.source == 'mouse-follow':
        from mouse_follow import mouse_follow_source
        return mouse_follow_source(initial_width=args.capture_width)
    if args.source == 'screen':
        return screen_capture_source(args.capture_fps or fps, region,
                                     args.display, args.capture_width,
                                     args.ffmpeg_input)
    if args.source == 'ffmpeg':
        return ffmpeg_source(args.ffmpeg_input, args.capture_fps or fps,
                             region, args.display, args.capture_width)
    if args.source == 'camera':
        return camera_source(args.camera, args.capture_fps or 30,
                             width=args.capture_width, spec=args.ffmpeg_input)
    if args.source == 'video':
        return video_source(args.file, not args.no_loop, realtime=not args.write)
    return test_source()


# --------------------------------------------------------------------------
# Run
# --------------------------------------------------------------------------

def to_wav(args, layout, coder, prepare, grab):
    """Render to a file instead of a device. Needs no audio hardware."""
    import wave
    count = args.frames or int(round(layout.fps*args.seconds))
    # v5 uses stereo like v3/v4
    channels = 2
    with wave.open(args.write, 'wb') as sink:
        sink.setparams((channels, 2, RATE, 0, 'NONE', 'not compressed'))
        # Pace captures at the wire's frame rate: --write records what the
        # encoder would have sent live, so each grab must happen when that
        # packet would air. Unpaced, all 200 grabs land in the first few ms
        # and a screen/test source records one frozen instant. A slow
        # machine simply falls behind (delay <= 0 skips the sleep); it never
        # sleeps into the past or queues stale captures.
        start = time.perf_counter()
        for n in range(count):
            delay = start + n/layout.fps - time.perf_counter()
            if delay > 0:
                time.sleep(delay)
            image = prepare(grab())
            code = image.info.get('aspect_code', 0)
            if args.numbered:
                image = burn_counters(image, n+1, n+1, count)
            if args.profile == 'hd-dwt':
                audio = V3.encode_v5(image_values(image, coder.grids), layout, coder,
                                     n+1, (n % count)+1, count,
                                     stamp_ms=int(n*1000/layout.fps),
                                     headroom=0.95, profile=2, aspect_code=code)
            else:
                audio = V3.encode(image_values(image, coder.grids), layout, coder,
                                  n+1, (n % count)+1, count,
                                  stamp_ms=int(n*1000/layout.fps),
                                  profile=wire_profile_code(args.profile), aspect_code=code)
            audio = bound_emission(audio, args.emit_ceiling, RATE)
            sink.writeframesraw(pcm(audio*args.gain))
    print(f'wrote {count} frames, {count/layout.fps:.1f} s at {layout.fps:.2f} fps '
          f'(captured over {time.perf_counter()-start:.1f} s) -> {args.write}')


def to_device(args, layout, coder, prepare, grab):
    # v5 uses stereo like v3/v4
    channels = args.channels
    sent = misses = 0
    started = time.perf_counter()
    # The send lead reserve() grants must cover the encode, or submit()
    # refuses the packet as stale. A miss never advances next_start, so the
    # next reserve() grants the SAME short lead: an encode slower than
    # --prepare-ms misses every packet forever and nothing is ever sent (this
    # is how hd-dwt stalled while its CDF 9/7 took ~25 ms against 10 ms).
    # After a miss, grow the lead to the slowest encode seen plus a margin.
    prepare_ms = args.prepare_ms
    slowest_ms = 0.0
    # WIRE_HD frame=3488 = 32 * 109, so blocksize=109 gives integer blocks.
    blocksize = 109 if layout.name == 'wire-hd' else 256
    with PacketOutput(device(args.device), channels, args.latency,
                      frame=layout.frame, packet=layout.packet,
                      blocksize=blocksize) as output:
        # The device is open now, so the real wire rate is finally known. A
        # capture paced at the reference frame rate would drift against a
        # 44.1 kHz device by 8% -- one wasted or repeated grab every 12 frames.
        if not args.capture_fps and args.source != 'camera' and hasattr(grab, 'retune'):
            grab.retune(output.fps)
        if not args.quiet:
            print(f'output {output.rate:g} Hz, {output.fps:.2f} fps')
        notice = wire_notice(layout, output.rate)
        if notice:
            print(notice, file=sys.stderr)
        try:
            while not args.frames or sent < args.frames:
                if not output.ready():
                    time.sleep(.002)
                    continue
                slot = output.reserve(prepare_ms, args.receive_margin_ms)
                if slot is None:
                    continue
                began = time.perf_counter()
                image = prepare(grab())
                code = image.info.get('aspect_code', 0)
                if args.numbered:
                    image = burn_counters(image, sent+1, sent+1, 0xffff)
                if args.profile == 'hd-dwt':
                    audio = V3.encode_v5(image_values(image, coder.grids), layout,
                                         coder, (sent+1) & 0xffffffff, (sent % 0xffff)+1,
                                         0xffff, stamp_ms=int(
                                             (slot.target_time_ns//1_000_000) & 0xffffffff),
                                         headroom=0.95, profile=2, aspect_code=code)
                else:
                    audio = V3.encode(image_values(image, coder.grids), layout,
                                      coder, (sent+1) & 0xffffffff, (sent % 0xffff)+1,
                                      0xffff, stamp_ms=int(
                                          (slot.target_time_ns//1_000_000) & 0xffffffff),
                                       profile=wire_profile_code(args.profile), aspect_code=code)
                # RATE, not output.rate. The packet is still on the REFERENCE
                # grid here -- band_limited resamples it to the device further
                # down, holding the carriers at the same hertz -- so the filter
                # has to be designed against the rate the array is actually at.
                # Passing the device rate put the real cutoff at ceiling*48000/
                # device: 7008 Hz instead of 14000 on a 96 kHz device, which
                # removes most of mid-14k's 375-12750 Hz band and decodes
                # nothing. Filtering here and resampling after is the right
                # order anyway: the emitted band then lands where this says.
                audio = bound_emission(audio, args.emit_ceiling, RATE)
                encode_ms = (time.perf_counter()-began)*1000
                accepted = output.submit(audio, slot)
                # Through submit(): it resamples to the device rate BEFORE its
                # deadline check, so that cost is part of the lead too.
                slowest_ms = max(slowest_ms, (time.perf_counter()-began)*1000)
                if accepted:
                    sent += 1
                    if args.log_frames:
                        print(json.dumps({
                            'frame': sent, 'encode_ms': round(encode_ms, 2),
                            'grabs': getattr(grab, 'grabs', None),
                            'deadline_misses': output.deadline_misses,
                            'starvations': output.starvations}), flush=True)
                else:
                    misses += 1
                    wanted = slowest_ms*1.5 + 2
                    if wanted > prepare_ms:
                        prepare_ms = wanted
                        print(f'\nencode took {slowest_ms:.1f} ms; send lead raised '
                              f'to {prepare_ms:.1f} ms', file=sys.stderr)
                if not args.quiet and sent and sent % 30 == 0:
                    elapsed = time.perf_counter()-started
                    print(f'  {sent} queued, {output.completed} played, '
                          f'{output.completed/elapsed:5.2f} fps out, '
                          f'{output.deadline_misses} missed, {output.starvations} starved',
                          end='\r', flush=True)
        except KeyboardInterrupt:
            pass
        finally:
            output.finish()
    elapsed = time.perf_counter()-started
    print(f'\nqueued {sent} packets, played {output.completed} in {elapsed:.1f} s '
          f'({output.completed/max(elapsed,1e-9):.2f} fps), '
          f'{output.deadline_misses} deadline misses')


def parser():
    """Built separately from main() so it can be inspected without running.

    to_device reads args.prepare_ms and args.receive_margin_ms, which were
    never defined here: every test used --write, which never reaches that
    branch, so the live path raised AttributeError on the first packet with
    full test coverage passing. test_modem_screen now walks the module's AST
    for every args.X it reads and asserts the parser defines it.
    """
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--source', choices=('screen', 'ffmpeg', 'video', 'camera', 'test', 'mouse-follow'),
                    default='test',
                    help='screen = FFmpeg screen capture (auto-detected on macOS); '
                         'ffmpeg = explicit platform capture; '
                         'camera = webcam; test = no devices; mouse-follow = dynamic cursor tracking')
    # The profiles the header can name. Every one is DCT/wavelet-sampled; the bake and
    # the wire shapes both come from the profile's geometry.
    ap.add_argument('--profile', choices=list(wire_profiles()) + [
        'v6-dct', 'v6-wavelet', 'v6-repeat-dct', 'v6-repeat-wavelet'],
                    default=DEFAULT_PROFILE,
                    help='Picture geometry. Each samples a grid 2x finer than '
                         'it transmits and sends the low-frequency corner; the '
                         'receiver reads which was sent from the header.')
    ap.add_argument('--wire', choices=('wide', 'tape', 'v6', 'v6-repeat'), default='wide',
                    help='wide uses carriers through 20.25 kHz; tape reallocates '
                         'the complete hd-dwt picture below 12.75 kHz and bounds '
                         'the emitted waveform at 14 kHz (lower frame rate)')
    ap.add_argument('--device', type=device, help='Audio output device')
    ap.add_argument('--channels', type=pair, default=(0, 1))
    ap.add_argument('--latency', default='low')
    ap.add_argument('--prepare-ms', type=float, default=10.0,
                    help='Minimum encoding lead before a send deadline. Raise '
                         'it if --log-frames shows deadline misses.')
    ap.add_argument('--receive-margin-ms', type=float, default=15.0,
                    help='Time after packet completion reserved for the '
                         'receiver to decode and display')
    ap.add_argument('--region', help='Screen region as left,top,width,height')
    ap.add_argument('--display', type=int, help='avfoundation screen index')
    ap.add_argument('--camera', type=int, default=0, help='Camera index')
    ap.add_argument('--ffmpeg-input', metavar='FMT:SRC',
                    help='Override the ffmpeg input, e.g. avfoundation:1:none')
    ap.add_argument('--file', help='Video file for --source video')
    ap.add_argument('--no-loop', action='store_true', help='Play a file once')
    ap.add_argument('--capture-fps', type=int,
                    help='Capture rate. Defaults to 30 fps for cameras and the '
                         'wire rate for other sources. Choose a rate supported '
                         'by your capture device.')
    ap.add_argument('--capture-width', type=int, default=320,
                    help='Width ffmpeg scales to before Python sees the frame')
    ap.add_argument('--rotate', type=int, default=0, choices=(0, 90, 180, 270))
    ap.add_argument('--mirror', action='store_true')
    ap.add_argument('--encode-filter', choices=tuple(ENCODE_FILTERS), default='lanczos',
                    help='Source-to-80x96 resize filter (default: lanczos). '
                         'Applied after capture pre-scaling/striding.')
    ap.add_argument('--aspect', choices=ASPECT_CHOICES, default='auto',
                    help='Display aspect: auto selects the nearest preset from '
                         'each source frame (including cameras), after rotation. '
                         'All sources are stretched to 80x96 without bars.')
    ap.add_argument('--crop', action='store_true',
                    help='Legacy alias for the default full-frame stretch')
    ap.add_argument('--numbered', action='store_true',
                    help='Burn frame counters into the picture')
    ap.add_argument('--gain', type=float, default=1.0,
                    help='Output scale. v3 already normalises to 0.95; above '
                         '1.0 clips.')
    ap.add_argument('--emit-ceiling', type=float, metavar='HZ',
                    help='Hold the emitted spectrum under HZ. top_bin bounds '
                         'the carriers, not the emission: the preamble is '
                         'square-edged and every preset emits out past 23 kHz '
                         'without this. Only needed for a channel with a hard '
                         'ceiling -- one that merely rolls off removes the tail '
                         'itself at no cost. The full-band header rides the top '
                         'carriers, so a low cut costs header diversity.')
    ap.add_argument('--frames', type=int, help='Stop after this many packets')
    ap.add_argument('--seconds', type=float, default=10.0,
                    help='Duration for --write when --frames is not given')
    ap.add_argument('--write', help='Render to a WAV instead of a device')
    ap.add_argument('--log-frames', action='store_true')
    ap.add_argument('--quiet', action='store_true')
    ap.add_argument('--list-devices', action='store_true')
    return ap


def main(argv=None):
    args = parser().parse_args(argv)
    # Keep the selected wire explicit at the CLI boundary; build() also uses a
    # compatibility fallback for older tests/callers that construct Namespace.
    if args.wire == 'tape' and args.profile not in ('hd-dwt', 'tape-80x60'):
        raise SystemExit('--wire tape requires hd-dwt or tape-80x60')
    if args.wire == 'v6' and args.profile not in ('v6-dct', 'v6-wavelet'):
        raise SystemExit('--wire v6 requires v6-dct or v6-wavelet')
    if args.wire == 'v6-repeat' and args.profile not in ('v6-repeat-dct',
                                                          'v6-repeat-wavelet'):
        raise SystemExit('--wire v6-repeat requires a v6-repeat profile')

    if args.list_devices:
        from animation_modem.audio_common import sounddevice
        print(sounddevice().query_devices())
        return
    if args.source == 'video' and not args.file:
        raise SystemExit('--source video needs --file')
    if not np.isfinite(args.gain) or args.gain <= 0:
        raise SystemExit('--gain must be finite and positive')
    if args.crop and not args.quiet:
        print('--crop: full-frame stretching is now the default.')

    # Neither preset nor profile has to be agreed out of band. The wire is
    # fixed and the profile is declared in the header, so nothing is refused.
    layout, coder, shapes = build(args)
    prepare = fitter(args.profile, args.rotate, args.mirror, aspect=args.aspect,
                     encode_filter=args.encode_filter)
    raw = source_for(args, layout.fps)
    # Throttling exists so a slow capture cannot stall the audio callback and a
    # fast one cannot burn a core. Neither applies when rendering to a file, and
    # a throttle there would sample the same frame repeatedly.
    capture_hz = args.capture_fps or (30 if args.source == 'camera' else layout.fps)
    grab = raw if (args.write and getattr(raw, 'sequential', False)) \
        else Throttled(raw, capture_hz)
    print(f'{layout.describe()}')
    # Frame rate here is the reference figure. Live output reprints it from the
    # device's own rate once the stream is open, which is the one that governs.
    picture = source_size(args.profile)
    label = getattr(coder, 'profile_name', args.profile)
    print(f'profile {label}: {picture[0]}x{picture[1]} picture, '
          f'{coder.count} coefficients, '
          f'{layout.fps:.2f} fps at {RATE:g} Hz')
    try:
        if args.write:
            to_wav(args, layout, coder, prepare, grab)
        else:
            to_device(args, layout, coder, prepare, grab)
    finally:
        if hasattr(grab, 'close'):
            grab.close()
        elif getattr(grab, 'proc', None) is not None:
            grab.proc.terminate()


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print('\nStopped.', file=sys.stderr)
