"""
scope_out.py -- send 2D vector graphics to an oscilloscope in XY mode
via the sound card.  Left channel = X, right channel = Y.

    pip install numpy sounddevice

Scope setup: XY / "Format XY" mode, both inputs DC-coupled if available,
~200-500 mV/div, start with the system volume low and bring it up.
"""

import re
import sys
import threading

try:
    import settings as settings_mod
except Exception:                    # usable standalone, outside the repo
    settings_mod = None
import numpy as np
import sounddevice as sd

SAMPLE_RATE = 48_000     # fallback only; Scope reads the device's rate at init
FPS = 50
SAMPLES_PER_FRAME = SAMPLE_RATE // FPS   # fallback resolution budget

JUMP_GAIN = 0.12   # <1 -> fewer samples spent on travel moves -> dimmer
SMOOTH = 5         # circular box filter width; tames DAC ringing at corners
LEVEL = 0.9        # peak output amplitude, keep below 1.0


# Names that usually mean an internal loudspeaker.  The XY signal is not audio
# -- it is a full-amplitude sweep -- so sending it to a small driver sounds
# awful and is hard on the tweeter.
_BUILTIN_HINTS = ("built-in", "builtin", "internal", "macbook", "imac",
                  "speakers", "speaker", "hdmi", "displayport", "bcm2835")


def warn_if_builtin(device):
    """Print a warning when the resolved output looks like internal speakers."""
    try:
        info = sd.query_devices(device) if device is not None \
            else sd.query_devices(kind="output")
        name = scrub(str(info.get("name", "")))
    except Exception:
        return
    low = name.lower()
    if any(h in low for h in _BUILTIN_HINTS):
        print(f"[SCOPE] !!  Output is {name!r}, which looks like internal "
              f"speakers.\n"
              f"[SCOPE]     This signal is NOT audio -- it is a full-amplitude "
              f"XY sweep. It will\n"
              f"[SCOPE]     sound like harsh noise and can damage small drivers "
              f"at volume.\n"
              f"[SCOPE]     Use --ask or --device to select your scope "
              f"interface.")


_ANSI_OR_CTRL = re.compile(r"\x1b\[[0-9;?]*[A-Za-z]|\x1b.|[\x00-\x1f\x7f]")


def scrub(text):
    """
    Drop lone surrogates from a string.

    Both sys.argv and PortAudio device names arrive decoded with
    surrogateescape, so a single stray byte anywhere in the command line or in
    a device name becomes something like a lone U+DCC3.  Any later encode()
    raises UnicodeEncodeError, which is how a bad byte in shell history takes
    down startup.  Strip them rather than propagate.
    """
    if not isinstance(text, str):
        return text
    return "".join(c for c in text if not 0xD800 <= ord(c) <= 0xDFFF)


def _clean_input(raw):
    """
    Scrub a line read from a terminal.

    Terminals leave debris in the input buffer -- arrow keys arrive as escape
    sequences, and a stray high byte comes back from input() as a lone
    lone surrogate (U+DCC3 and friends).  Passing that through produced a
    and killed the run, so drop anything that is not printable text.
    """
    if raw is None:
        return ""
    txt = scrub(raw)
    return _ANSI_OR_CTRL.sub("", txt).strip()


def list_output_devices(min_channels=2):
    """[(index, name, host_api, default_rate)] for devices that can do stereo."""
    try:
        apis = sd.query_hostapis()
    except Exception:
        apis = []
    out = []
    for i, d in enumerate(sd.query_devices()):
        if d.get("max_output_channels", 0) < min_channels:
            continue
        ha = d.get("hostapi")
        api = apis[ha]["name"] if apis and isinstance(ha, int) and ha < len(apis) else "?"
        out.append((i, scrub(d.get("name", f"device {i}")), scrub(api),
                    int(d.get("default_samplerate") or 0)))
    return out


def default_output_index():
    try:
        d = sd.default.device
        idx = d[1] if isinstance(d, (list, tuple)) else d
        return idx if isinstance(idx, (int, np.integer)) and idx >= 0 else None
    except Exception:
        return None


def resolve_device(spec):
    """
    Accept a name fragment, an index, or None (= system default).

    An integer indexes the OUTPUT-ONLY list -- the same numbering --ask prints.
    It is deliberately NOT a raw PortAudio index: PortAudio numbers inputs and
    outputs in one sequence, so "1" there is typically a microphone, and the
    two numberings disagreeing is exactly the trap this avoids.
    """
    if spec is None or spec == "":
        return None
    spec = scrub(spec) if isinstance(spec, str) else spec
    if isinstance(spec, str):
        spec = _ANSI_OR_CTRL.sub("", spec).strip()
        if not spec:
            return None
    devs = list_output_devices()

    def listing():
        return "\n".join(f"  [{i}] {n}  ({a}, {r} Hz)"
                         for i, (_gi, n, a, r) in enumerate(devs))

    try:
        k = int(spec)
    except (TypeError, ValueError):
        hits = [d for d in devs if str(spec).lower() in d[1].lower()]
        if not hits:
            raise SystemExit(
                f"No output device matching {spec!r}. Outputs are:\n{listing()}")
        if len(hits) > 1:
            print(f"[AUDIO] {spec!r} matches {len(hits)} devices, "
                  f"using {hits[0][1]!r}")
        return hits[0][0]

    if not 0 <= k < len(devs):
        raise SystemExit(
            f"--device {k} is out of range; there are {len(devs)} outputs.\n"
            f"These indices match what --ask shows:\n{listing()}")
    return devs[k][0]


def choose_device(ask=False, device=None, stream=None):
    """
    Pick an output device.  Returns a PortAudio index, or None for the system
    default.

    The prompt numbers outputs sequentially from 0, and --device uses the same
    numbering, so what you read off the list is what you can pass next time.

    Prompts only when asked AND more than one stereo-capable output exists AND
    there is a terminal to prompt on -- so --ask is safe to leave in a kiosk
    launch script, where it takes the default instead of hanging.
    """
    if device is not None:
        return resolve_device(device)
    devs = list_output_devices()
    if not devs:
        raise RuntimeError("No stereo-capable audio output found. "
                           "On Linux check that libportaudio2 is installed.")
    if not ask or len(devs) == 1:
        return None
    if not sys.stdin.isatty():
        print("[AUDIO] --ask given but no terminal; using system default")
        return None

    dflt = default_output_index()
    print("\nAudio outputs (rate sets samples/trace -- higher is more detail):")
    for n, (gi, name, api, rate) in enumerate(devs):
        mark = "  <- system default" if gi == dflt else ""
        print(f"  [{n}] {name}  ({api}, {rate} Hz){mark}")
    for _attempt in range(3):
        try:
            raw = _clean_input(input("Output [system default]: "))
        except (EOFError, KeyboardInterrupt):
            print()
            return None
        if not raw:
            return None
        try:
            k = int(raw)
        except ValueError:
            hits = [d for d in devs if raw.lower() in d[1].lower()]
            if len(hits) == 1:
                return hits[0][0]
            if len(hits) > 1:
                print(f"[AUDIO] {raw!r} matches {len(hits)} devices, "
                      f"using {hits[0][1]!r}")
                return hits[0][0]
            print(f"[AUDIO] no output matches {raw!r} -- enter a number 0-"
                  f"{len(devs) - 1}, a name, or blank for the default")
            continue
        if 0 <= k < len(devs):
            return devs[k][0]
        print(f"[AUDIO] {k} is out of range -- enter 0-{len(devs) - 1}, "
              "a name, or blank for the default")
    print("[AUDIO] no valid selection; using system default")
    return None


def lowpass_frame(frame, samplerate, cutoff_hz, taper_hz=0.0):
    """
    Band-limit one looping frame, circularly.

    The frame repeats forever, so it is genuinely periodic and an FFT filter is
    exact here -- no edge transient, and the loop stays seamless.  An IIR run
    linearly over the buffer would leave a discontinuity at the wrap.

    cutoff_hz : everything above this is removed.
    taper_hz  : width of a raised-cosine rolloff above the cutoff.  0 gives a
                brick wall, which is harsher than any real filter but shows the
                limit cleanly; a few kHz is closer to real hardware.

    This is the honest test of whether a grid is really being drawn: the
    preview interpolates between samples and models no bandwidth limit, so
    detail can look present that no signal path could carry.
    """
    n = len(frame)
    if not cutoff_hz or cutoff_hz <= 0 or n < 8:
        return np.asarray(frame, dtype=np.float32)
    F = np.fft.rfft(np.asarray(frame, dtype=np.float64), axis=0)
    freqs = np.fft.rfftfreq(n, 1.0 / float(samplerate))
    if taper_hz and taper_hz > 0:
        x = np.clip((cutoff_hz + taper_hz - freqs) / float(taper_hz), 0.0, 1.0)
        gain = 0.5 - 0.5 * np.cos(np.pi * x)          # raised cosine
    else:
        gain = (freqs <= cutoff_hz).astype(np.float64)
    out = np.fft.irfft(F * gain[:, None], n=n, axis=0)
    return np.ascontiguousarray(out, dtype=np.float32)


def from_screen(points, width, height):
    """Pixel coords -> [-1, 1], aspect preserved, origin at centre."""
    p = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    s = max(width, height) / 2.0
    return np.stack([(p[:, 0] - width / 2.0) / s,
                     (p[:, 1] - height / 2.0) / s], axis=1)


def rasterize(polylines, n=SAMPLES_PER_FRAME):
    """
    polylines: list of (K, 2) arrays of points in [-1, 1].
    Returns an (n, 2) float32 frame: one closed loop through every shape,
    sampled at constant arc length so brightness is even.
    """
    pts, is_jump = [], []
    for pl in polylines:
        pl = np.asarray(pl, dtype=np.float64).reshape(-1, 2)
        if len(pl) < 2:
            continue
        if pts:
            is_jump.append(True)              # travel move from previous end
        pts.append(pl)
        is_jump.extend([False] * (len(pl) - 1))

    if not pts:
        # Never park the beam: 0 V on both channels is a stationary
        # full-brightness dot at centre screen, which burns phosphor on
        # analog CRTs.  Idle on a circle so the energy stays spread out.
        th = np.linspace(0, 2 * np.pi, n, endpoint=False)
        return (LEVEL * np.stack([np.cos(th), np.sin(th)], axis=1)).astype(np.float32)

    P = np.vstack(pts)
    P = np.vstack([P, P[0]])                  # close loop back to the start
    is_jump.append(True)

    seg = np.diff(P, axis=0)
    length = np.hypot(seg[:, 0], seg[:, 1])
    w = np.where(np.array(is_jump), length * JUMP_GAIN, length)
    w = np.maximum(w, 1e-9)

    cum = np.concatenate([[0.0], np.cumsum(w)])
    t = np.linspace(0.0, cum[-1], n, endpoint=False)
    i = np.clip(np.searchsorted(cum, t, side="right") - 1, 0, len(w) - 1)
    f = ((t - cum[i]) / w[i])[:, None]
    out = P[i] + f * (P[i + 1] - P[i])

    if SMOOTH > 1:
        k = np.ones(SMOOTH) / SMOOTH
        pad = SMOOTH
        out = np.stack([
            np.convolve(np.r_[c[-pad:], c, c[:pad]], k, mode="same")[pad:-pad]
            for c in out.T
        ], axis=1)

    # Fixed gain only.  Per-frame mean-centring or peak-normalising would
    # re-fit every frame to the screen, destroying registration across a
    # matted sequence: the image would swim as content moves and lurch on
    # folder switches.  Geometry arrives already in canvas coords [-1, 1];
    # the clip guards smoothing overshoot, it is not a scaler.
    out = np.clip(out * LEVEL, -LEVEL, LEVEL)
    return np.ascontiguousarray(out, dtype=np.float32)


def precompensate_hpf(frame, corner_hz, samplerate, max_boost=8.0, level=LEVEL):
    """
    Pre-emphasise the frame to cancel the output's AC coupling.

    Headphone and line outputs are AC coupled: a series capacitor forms a
    single-pole high-pass, typically cornering somewhere around 5-50 Hz.  That
    is fatal here because the VERTICAL SWEEP REPEATS AT THE TRACE RATE -- 30 Hz
    at 30 fps -- so the sweep's fundamental sits right on the corner.  The
    result is not a slight tilt: the picture collapses into a funnel, because
    the slow envelope of both sweeps is attenuated and phase-shifted.

    The frame repeats, so its spectrum is exactly the harmonics of
    samplerate/len(frame), and the correction is exact rather than approximate:
    a single-pole high-pass H(f) = jf/(jf+fc) is inverted by multiplying each
    harmonic by (1 + fc/(jf)).  Bin 0 is genuinely lost -- no amount of
    pre-emphasis restores true DC through a capacitor -- but the image does not
    need DC, only the harmonics.

    ONLY USE THIS IF YOU CAN SEE THE DISTORTION.  A virtual device (BlackHole,
    a loopback, an interface's digital output) has no coupling capacitor, and
    much of this correction is a PHASE shift -- at the fundamental it rotates
    the vertical sweep 45 degrees against the horizontal.  On a path that needs
    it, that cancels; on a path that does not, it shears the picture.  Run
    `python scope_out.py --calibrate` first and leave this off if the test
    square is already square.

    max_boost caps the low-frequency gain so a badly mismatched corner cannot
    blow the amplitude up; the result is renormalised to `level` afterwards, so
    the trade shows up as reduced headroom rather than clipping.
    """
    frame = np.asarray(frame, dtype=np.float64)
    n = len(frame)
    if not corner_hz or corner_hz <= 0 or n < 4:
        return frame.astype(np.float32)
    f = np.fft.rfftfreq(n, d=1.0 / samplerate)
    corr = np.ones(len(f), dtype=complex)
    with np.errstate(divide="ignore", invalid="ignore"):
        corr[1:] = 1.0 + corner_hz / (1j * f[1:])
    mag = np.abs(corr)
    over = mag > max_boost
    corr[over] = corr[over] / mag[over] * max_boost
    corr[0] = 0.0                       # DC cannot pass; do not try

    out = np.empty_like(frame)
    for c in range(frame.shape[1]):
        out[:, c] = np.fft.irfft(np.fft.rfft(frame[:, c]) * corr, n=n)
    peak = np.abs(out).max()
    if peak > 0:
        out *= level / peak
    return out.astype(np.float32)


class BufferedSource:
    """
    Run a sample generator on a worker thread, feeding a ring buffer that the
    audio callback drains.

    Generating inside the callback means the WORST-CASE generation time has to
    fit the block deadline; miss it once and the stream underruns.  With a
    producer thread only the AVERAGE has to keep up, and the ring absorbs the
    jitter -- the same reason the video path prefetches into a FIFO.

    Depth is a real trade-off, not free headroom: every buffered sample is a
    sample of latency between a state change and the beam showing it, and low
    latency is the entire point of realtime mode.  A few blocks is plenty.

    On underflow the last sample is HELD, never zeroed: zero on both channels
    parks the beam at screen centre, which is a bright stationary dot.
    """

    def __init__(self, source, blocksize=256, depth=6, channels=2):
        self.source = source
        self.blocksize = int(blocksize)
        self.capacity = self.blocksize * max(2, int(depth))
        self._buf = np.zeros((self.capacity, channels), dtype=np.float32)
        self._w = 0                  # samples written, monotonic
        self._r = 0                  # samples read, monotonic
        self._last = np.zeros(channels, dtype=np.float32)
        self._lock = threading.Lock()
        self._wake = threading.Event()
        self._stop = threading.Event()
        self.underruns = 0
        self.blocks_made = 0
        self._thread = threading.Thread(target=self._run, daemon=True,
                                        name="scope-source")
        self._thread.start()

    def _run(self):
        while not self._stop.is_set():
            with self._lock:
                free = self.capacity - (self._w - self._r)
            if free < self.blocksize:
                self._wake.wait(0.002)
                self._wake.clear()
                continue
            try:
                chunk = self.source(self.blocksize)
            except Exception:
                chunk = np.tile(self._last, (self.blocksize, 1))
            chunk = np.asarray(chunk, dtype=np.float32)
            with self._lock:
                start = self._w % self.capacity
                end = start + len(chunk)
                if end <= self.capacity:
                    self._buf[start:end] = chunk
                else:
                    cut = self.capacity - start
                    self._buf[start:] = chunk[:cut]
                    self._buf[:end - self.capacity] = chunk[cut:]
                self._w += len(chunk)
                self.blocks_made += 1

    def __call__(self, n):
        """Called from the audio callback: a copy out of the ring, nothing more."""
        out = np.empty((n, 2), dtype=np.float32)
        with self._lock:
            avail = self._w - self._r
            take = min(n, avail)
            if take:
                start = self._r % self.capacity
                end = start + take
                if end <= self.capacity:
                    out[:take] = self._buf[start:end]
                else:
                    cut = self.capacity - start
                    out[:cut] = self._buf[start:]
                    out[cut:take] = self._buf[:end - self.capacity]
                self._r += take
                self._last = out[take - 1].copy()
        if take < n:
            out[take:] = self._last      # hold position; never park at centre
            self.underruns += 1
        self._wake.set()
        return out

    def close(self):
        self._stop.set()
        self._wake.set()
        self._thread.join(timeout=0.5)


class Scope:
    """Continuously loops the current frame out of an audio device.

    device=None targets the system default output.  The sample rate is taken
    from that device's own default: on desktops the default output is a shared
    mixer, and requesting any other rate just makes it resample behind your
    back, which blurs corners -- and probing support is useless because mixers
    accept every rate.  Pass samplerate= explicitly only on an exclusive route
    (raw ALSA hw:, WASAPI exclusive), where higher rates buy real resolution.
    """

    def __init__(self, device=None, samplerate=None, fps=FPS, samples=None,
                 invert_y=True, swap_xy=False, source=None,
                 lowpass_hz=None, lowpass_taper=0.0, blocksize=512):
        """
        samples : path length per trace -- the REAL parameter.  Refresh is not
                  set independently; it falls out as rate/samples, because the
                  DAC consumes samples at a fixed rate and the beam position is
                  those samples.  `fps` is just a convenience for specifying
                  `samples` as rate/fps.  Frames may vary in length at runtime;
                  the callback handles it.
        """
        self.invert_y = invert_y
        self.swap_xy = swap_xy
        if samplerate is None:
            info = sd.query_devices(device, "output")
            samplerate = int(info["default_samplerate"]) or SAMPLE_RATE
        self.source = source
        # Optional band-limit applied to every frame before it reaches the DAC,
        # so hardware sees exactly what the offline comparison shows.
        self.lowpass_hz = lowpass_hz
        self.lowpass_taper = lowpass_taper
        self.samplerate = samplerate
        self.samples_per_frame = (max(64, int(samples)) if samples
                                  else max(64, round(samplerate / fps)))
        self._frame = rasterize([], self.samples_per_frame)  # idle circle, never a parked dot
        self._pending = None
        self._lock = threading.Lock()
        self._pos = 0
        self.frames_drawn = 0     # complete traces emitted
        self.frames_dropped = 0   # indices superseded before they were drawn
        warn_if_builtin(device)
        # An explicit blocksize matters: with latency="low" and blocksize
        # unset, PortAudio picks the smallest buffer the device allows, so the
        # audio thread wakes ~1500 times a second and takes the GIL each time.
        # 512 frames is ~5 ms at 96 kHz -- far below the trace period, and a
        # third the wakeups.
        self.blocksize = int(blocksize or 0)
        self.stream = sd.OutputStream(
            samplerate=samplerate, channels=2, dtype="float32",
            device=device, blocksize=self.blocksize,
            latency="low", callback=self._callback)

    def _callback(self, outdata, frames, time_info, status):
        # Continuous source: content follows the clock in real time, with no
        # frame boundaries to wait for.
        if self.source is not None:
            try:
                buf = self.source(frames)
                outdata[:] = buf[:frames]
                self.frames_drawn += 1
            except Exception:
                outdata[:] = 0.0
            return
        # Swap ONLY at a frame boundary.  Replacing the buffer mid-trace makes
        # the beam jump from its position in one image to the same offset in a
        # different one -- a bright tear on every index change.
        filled = 0
        while filled < frames:
            f = self._frame
            n = len(f)
            take = min(frames - filled, n - self._pos)
            outdata[filled:filled + take] = f[self._pos:self._pos + take]
            self._pos += take
            filled += take
            if self._pos >= n:
                self._pos = 0
                self.frames_drawn += 1
                p = self._pending          # atomic under the GIL; last write wins
                self._pending = None
                if p is not None:
                    self._frame = p

    def show_frame(self, frame):
        """Queue a raw (n, 2) sample frame for the next frame boundary.

        Raster mode bypasses rasterize() because uneven dwell IS the image, so
        arc-length resampling would destroy it.  If an earlier frame is still
        waiting, it is discarded rather than queued: the scope should show the
        index that is current NOW, never fall behind replaying stale ones.
        """
        if self._pending is not None:
            self.frames_dropped += 1
        f = np.ascontiguousarray(frame, dtype=np.float32)
        if self.lowpass_hz:
            f = lowpass_frame(f, self.samplerate, self.lowpass_hz,
                              self.lowpass_taper)
        self._pending = f

    def show(self, polylines):
        """Call this once per drawn frame from your render loop."""
        f = rasterize(polylines, self.samples_per_frame)
        if self.invert_y:
            f[:, 1] *= -1                     # screen y-down -> scope y-up
        if self.swap_xy:
            f = np.ascontiguousarray(f[:, ::-1])
        self.show_frame(f)

    def __enter__(self):
        self.stream.start()
        return self

    def __exit__(self, *exc):
        self.stream.stop()
        self.stream.close()


def calibration_frame(n, level=LEVEL):
    """A square with a centre cross and corner ticks.

    Deliberately a SQUARE and not a picture: AC coupling shows up as the
    vertical sides splaying into a funnel and the horizontal sides bowing,
    which is obvious on a known shape and easy to miss on a face.
    """
    s = 0.75 * level
    box = np.array([[-s, -s], [s, -s], [s, s], [-s, s], [-s, -s]])
    cross_h = np.array([[-s * 0.25, 0.0], [s * 0.25, 0.0]])
    cross_v = np.array([[0.0, -s * 0.25], [0.0, s * 0.25]])
    return rasterize([box, cross_h, cross_v], n)


if __name__ == "__main__":
    import argparse
    import math
    import time

    ap = argparse.ArgumentParser(
        description="Scope output bench. Draws a test pattern so you can set "
                    "up the physical chain before any content is involved.")
    ap.add_argument("--calibrate", action="store_true",
                    help="draw a square instead of the spinning demo, for "
                         "checking AC-coupling distortion")
    ap.add_argument("--dc-comp", type=float, metavar="HZ",
                    help="AC-coupling pre-compensation to try. Leave unset "
                         "first: if the square is square, your output is DC "
                         "coupled and this would only distort it.")
    ap.add_argument("--fps", type=int, default=FPS)
    ap.add_argument("--device")
    ap.add_argument("--ask", action="store_true")
    args = ap.parse_args()

    scope = Scope(fps=args.fps,
                  device=choose_device(ask=args.ask, device=args.device))
    n = scope.samples_per_frame
    print(f"[BENCH] {n} samples/trace @ {scope.samplerate} Hz "
          f"({args.fps} traces/sec)")
    if args.calibrate:
        print("[BENCH] square + centre cross.")
        print("        Sides parallel and corners at 90 deg -> DC coupled, "
              "leave --dc-comp off.")
        print("        Sides splayed into a funnel           -> AC coupled, "
              "raise --dc-comp until they are parallel.")
        if args.dc_comp:
            print(f"[BENCH] pre-compensating at {args.dc_comp:g} Hz")

    with scope:
        t0 = time.time()
        try:
            while True:
                if args.calibrate:
                    frame = calibration_frame(n)
                    if args.dc_comp:
                        frame = precompensate_hpf(frame, args.dc_comp,
                                                  scope.samplerate)
                    scope.show_frame(frame)
                else:
                    a = time.time() - t0
                    th = np.linspace(0, 2 * np.pi, 96, endpoint=False)
                    circle = np.stack([0.6 * np.cos(th), 0.6 * np.sin(th)], 1)
                    circle = np.vstack([circle, circle[:1]])
                    sq = 0.35 * np.array([[-1, -1], [1, -1], [1, 1], [-1, 1], [-1, -1]])
                    c, sn = math.cos(a), math.sin(a)
                    scope.show([circle, sq @ np.array([[c, -sn], [sn, c]])])
                time.sleep(1 / 30)
        except KeyboardInterrupt:
            pass