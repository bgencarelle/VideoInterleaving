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

    python utilities/modem_v3_check.py live-receive --preset lean-v3 \
        --profile color-lean

BE REALISTIC ABOUT THE RESOLUTION. The picture is whatever the profile says:
40x48 colour for 'color' and 'color-lean', 48x60 grey for 'mono'. That is a
silhouette, a face, a moving shape, a lava lamp. It is not a desktop, and text
will not survive.

Unlike scope_screen this wants COLOUR, so ffmpeg is asked for rgb24 rather than
gray, and the frame is fitted to the profile's aspect rather than the screen's.
Any chroma plane off the luma aspect ratio gets letterboxed by image_values and
loses 2-4 dB, so the fit happens once, here, on the full-resolution frame.

Frame rate is set by the wire, not by the capture: at lean-v3 a packet is 2768
samples, so the modem consumes 17.34 pictures a second and the capture is
throttled to match. Grabbing faster only wastes CPU; grabbing slower repeats
the last frame rather than stalling the stream.

Screen capture needs `mss` (pip install mss), or use --source ffmpeg, which is
much faster on macOS and does the scaling itself.
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
import threading
import time

import numpy as np
from PIL import Image, ImageOps

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from animation_modem import transport3 as V3                      # noqa: E402
from animation_modem.audio_common import device, pair, pcm        # noqa: E402
from animation_modem.core import RATE, SourceCoder                # noqa: E402
from animation_modem.imaging import (PROFILES, burn_counters,     # noqa: E402
                                     fit_shapes, image_values, plane_shapes)
from animation_modem.playback import PacketOutput                 # noqa: E402


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
    sct = _MSS()
    mon = sct.monitors[1] if region is None else {
        'left': region[0], 'top': region[1],
        'width': region[2], 'height': region[3]}

    def grab():
        shot = sct.grab(mon)
        raw = np.frombuffer(shot.raw, np.uint8).reshape(shot.height, shot.width, 4)
        return raw[:, :, 2::-1]          # BGRA -> RGB
    return grab


def ffmpeg_source(spec, fps, region=None, display=None, width=320):
    """Capture through ffmpeg's platform fast path.

    mss goes via CoreGraphics on macOS and costs tens of milliseconds a grab.
    ffmpeg uses avfoundation / x11grab / gdigrab and scales before handing the
    frame over, so Python receives a small RGB array and does no image work.

    The output size is pinned so each frame is a known number of bytes; reading
    it back from ffmpeg's stderr is fragile.
    """
    if shutil.which('ffmpeg') is None:
        raise SystemExit('ffmpeg not found. brew install ffmpeg / apt install ffmpeg')
    try:
        import mss as _mss
        with (_mss.MSS() if hasattr(_mss, 'MSS') else _mss.mss()) as _s:
            mon = _s.monitors[1]
            sw, sh = mon['width'], mon['height']
    except Exception:
        sw, sh = 1920, 1080
    if region:
        sw, sh = region[2], region[3]
    w = int(width)
    h = max(2, int(round(w*sh/float(sw)))//2*2)

    if spec:
        fmt, src = spec.split(':', 1)
    elif sys.platform == 'darwin':
        fmt, src = 'avfoundation', f'{display if display is not None else 1}:none'
    elif sys.platform.startswith('win'):
        fmt, src = 'gdigrab', 'desktop'
    else:
        fmt, src = 'x11grab', os.environ.get('DISPLAY', ':0.0')

    cmd = ['ffmpeg', '-loglevel', 'error', '-f', fmt,
           '-framerate', str(int(max(fps, 1)))]
    if fmt == 'x11grab' and region:
        cmd += ['-video_size', f'{region[2]}x{region[3]}',
                '-i', f'{src}+{region[0]},{region[1]}']
    else:
        cmd += ['-i', src]
    cmd += ['-vf', f'scale={w}:{h}', '-pix_fmt', 'rgb24',
            '-f', 'rawvideo', '-an', '-sn', '-']
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE,
                            stderr=subprocess.DEVNULL, bufsize=0)
    nbytes = w*h*3
    last = [np.zeros((h, w, 3), np.uint8)]

    def grab():
        buf = _read_exact(proc.stdout, nbytes)
        if buf is None:
            return last[0]
        last[0] = np.frombuffer(buf, np.uint8).reshape(h, w, 3)
        return last[0]
    grab.proc = proc
    grab.size = (w, h)
    return grab


def camera_source(index=0, fps=15, width=320, spec=None):
    """Webcam through ffmpeg, same reasoning as screen capture."""
    if sys.platform == 'darwin':
        spec = spec or f'avfoundation:{index}'
    elif sys.platform.startswith('win'):
        spec = spec or 'dshow:video=Integrated Camera'
    else:
        spec = spec or f'v4l2:/dev/video{index}'
    return ffmpeg_source(spec, fps, width=width)


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
        self._latest = grab()
        if self._latest is None:
            raise SystemExit('Capture produced no frames')
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self.grabs = 0
        self._period = 1.0/max(hz, .1)
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self):
        while not self._stop.is_set():
            began = time.perf_counter()
            try:
                frame = self._grab()
            except Exception:
                break
            with self._lock:
                self._latest = frame
                self.grabs += 1
            rest = self._period-(time.perf_counter()-began)
            if rest > 0:
                self._stop.wait(rest)

    def __call__(self):
        with self._lock:
            return self._latest

    def close(self):
        self._stop.set()
        proc = getattr(self._grab, 'proc', None)
        if proc is not None:
            proc.terminate()


# --------------------------------------------------------------------------
# Frame preparation
# --------------------------------------------------------------------------

def fitter(profile, rotate=0, mirror=False, letterbox=True):
    """Full-resolution RGB array -> a PIL image the profile's exact size.

    Fitting happens once here on the source frame rather than per plane.
    image_values pads each plane independently, so handing it an off-aspect
    picture puts bars in luma AND chroma at different scales.

    The target is tiny -- 40x48 -- and the source may not be. mss hands back
    the whole backing buffer, so on a Retina panel LANCZOS runs from 3840x2400
    straight down to 40 pixels wide. Measured, that costs 75-88 ms per frame
    against a 57.7 ms packet budget: the capture alone cannot keep up with the
    wire. Striding to roughly 4x the target first is a view, so it is free, and
    LANCZOS then filters an image ~600x smaller. Measured 0.35 ms against 75 ms
    for 43.6 dB of agreement with the direct resize -- on a deliberately
    alias-hostile test pattern, and invisible at 40x48.

    ffmpeg sources already scale before Python sees them, so the stride is a
    no-op there.
    """
    size = PROFILES[profile][0]
    target = max(size)*4

    def prepare(raw):
        raw = np.asarray(raw, np.uint8)
        h, w = raw.shape[:2]
        step = max(1, min(w//target, h//target))
        if step > 1:
            raw = raw[::step, ::step]
        im = Image.fromarray(raw, 'RGB')
        if rotate:
            im = im.rotate(-rotate, expand=True)
        if mirror:
            im = ImageOps.mirror(im)
        if letterbox:
            return ImageOps.pad(im, size, Image.LANCZOS, color=(4, 4, 4))
        return ImageOps.fit(im, size, Image.LANCZOS)
    return prepare


def build(args):
    """Layout and coder, warning loudly if the picture had to shrink.

    fit_shapes quietly scales the planes down until they fit the preset, so a
    mismatched pair still runs -- it just sends a much smaller picture than the
    profile names. 'lofi' with 'color' yields 378 coefficients rather than
    2880, and nothing says so. Silent resolution loss is worse than an error.
    """
    if args.profile not in PROFILES:
        raise SystemExit(f'Unknown profile {args.profile}')
    layout = V3.ALL_PRESETS[args.preset]
    wanted = plane_shapes(args.profile)
    shapes = fit_shapes(wanted, layout.capacity)
    coder = SourceCoder(shapes)
    asked = int(sum(np.prod(s) for s in wanted))
    if coder.count < asked:
        got = (shapes[0][1], shapes[0][0])
        print(f'WARNING: preset {args.preset} holds {layout.capacity} values, '
              f'profile {args.profile} wants {asked}. Picture shrunk to '
              f'{got[0]}x{got[1]} ({coder.count} coefficients). Pick a preset '
              f'with more capacity, or a smaller profile.', file=sys.stderr)
    return layout, coder, shapes


def source_for(args, fps):
    region = _region(args.region)
    if args.source == 'screen':
        return screen_source(region)
    if args.source == 'ffmpeg':
        return ffmpeg_source(args.ffmpeg_input, args.capture_fps or fps,
                             region, args.display, args.capture_width)
    if args.source == 'camera':
        return camera_source(args.camera, args.capture_fps or fps,
                             args.capture_width, args.ffmpeg_input)
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
    with wave.open(args.write, 'wb') as sink:
        sink.setparams((2, 2, RATE, 0, 'NONE', 'not compressed'))
        for n in range(count):
            image = prepare(grab())
            if args.numbered:
                image = burn_counters(image, n+1, n+1, count)
            audio = V3.encode(image_values(image, coder.shapes), layout, coder,
                              n+1, (n % count)+1, count,
                              stamp_ms=int(n*1000/layout.fps))
            sink.writeframesraw(pcm(audio*args.gain))
    print(f'wrote {count} frames, {count/layout.fps:.1f} s at {layout.fps:.2f} fps '
          f'-> {args.write}')


def to_device(args, layout, coder, prepare, grab):
    channels = args.channels
    sent = misses = 0
    started = time.perf_counter()
    with PacketOutput(device(args.device), channels, args.latency,
                      frame=layout.frame, packet=layout.packet) as output:
        try:
            while not args.frames or sent < args.frames:
                if not output.ready():
                    time.sleep(.002)
                    continue
                slot = output.reserve(args.prepare_ms, args.receive_margin_ms)
                if slot is None:
                    continue
                began = time.perf_counter()
                image = prepare(grab())
                if args.numbered:
                    image = burn_counters(image, sent+1, sent+1, 0xffff)
                audio = V3.encode(image_values(image, coder.shapes), layout,
                                  coder, (sent+1) & 0xffffffff, (sent % 0xffff)+1,
                                  0xffff, stamp_ms=int(
                                      (slot.target_time_ns//1_000_000) & 0xffffffff))
                encode_ms = (time.perf_counter()-began)*1000
                if output.submit(audio, slot):
                    sent += 1
                    if args.log_frames:
                        print(json.dumps({
                            'frame': sent, 'encode_ms': round(encode_ms, 2),
                            'grabs': getattr(grab, 'grabs', None),
                            'deadline_misses': output.deadline_misses,
                            'starvations': output.starvations}), flush=True)
                else:
                    misses += 1
                if not args.quiet and sent and sent % 30 == 0:
                    elapsed = time.perf_counter()-started
                    print(f'  {sent} packets, {sent/elapsed:5.2f} fps out, '
                          f'{misses} missed, {output.starvations} starved',
                          end='\r', flush=True)
        except KeyboardInterrupt:
            pass
        finally:
            output.finish()
    elapsed = time.perf_counter()-started
    print(f'\nsent {sent} packets in {elapsed:.1f} s '
          f'({sent/max(elapsed,1e-9):.2f} fps), {misses} deadline misses')


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
    ap.add_argument('--source', choices=('screen', 'ffmpeg', 'video', 'camera', 'test'),
                    default='test',
                    help='screen = mss (simple, slow on macOS); ffmpeg = the '
                         'platform fast path; camera = webcam; test = no devices')
    ap.add_argument('--preset', choices=list(V3.ALL_PRESETS), default='lean-v3')
    ap.add_argument('--profile', choices=list(PROFILES), default='color-lean')
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
                    help='Capture rate. Defaults to the wire rate; higher only '
                         'costs CPU because the newest frame wins either way.')
    ap.add_argument('--capture-width', type=int, default=320,
                    help='Width ffmpeg scales to before Python sees the frame')
    ap.add_argument('--rotate', type=int, default=0, choices=(0, 90, 180, 270))
    ap.add_argument('--mirror', action='store_true')
    ap.add_argument('--crop', action='store_true',
                    help='Fill the frame and crop instead of letterboxing')
    ap.add_argument('--numbered', action='store_true',
                    help='Burn frame counters into the picture')
    ap.add_argument('--gain', type=float, default=1.0,
                    help='Output scale. v3 already normalises to 0.95; above '
                         '1.0 clips.')
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

    if args.list_devices:
        from animation_modem.audio_common import sounddevice
        print(sounddevice().query_devices())
        return
    if args.source == 'video' and not args.file:
        raise SystemExit('--source video needs --file')
    if not np.isfinite(args.gain) or args.gain <= 0:
        raise SystemExit('--gain must be finite and positive')

    layout, coder, shapes = build(args)
    prepare = fitter(args.profile, args.rotate, args.mirror, not args.crop)
    raw = source_for(args, layout.fps)
    # Throttling exists so a slow capture cannot stall the audio callback and a
    # fast one cannot burn a core. Neither applies when rendering to a file, and
    # a throttle there would sample the same frame repeatedly.
    grab = raw if (args.write and getattr(raw, 'sequential', False)) \
        else Throttled(raw, args.capture_fps or layout.fps)
    print(f'{layout.describe()}')
    print(f'profile {args.profile}: {PROFILES[args.profile][0][0]}x'
          f'{PROFILES[args.profile][0][1]}, {coder.count} coefficients, '
          f'{layout.fps:.2f} fps')
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
    main()
