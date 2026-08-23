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
                 invert_y=True, swap_xy=False, source=None):
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
        self.stream = sd.OutputStream(
            samplerate=samplerate, channels=2, dtype="float32",
            device=device, latency="low", callback=self._callback)

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
        self._pending = np.ascontiguousarray(frame, dtype=np.float32)

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


if __name__ == "__main__":
    import math
    import time

    th = np.linspace(0, 2 * np.pi, 128, endpoint=False)
    circle = np.stack([0.6 * np.cos(th), 0.6 * np.sin(th)], axis=1)
    circle = np.vstack([circle, circle[:1]])
    s = 0.35
    square = np.array([[-s, -s], [s, -s], [s, s], [-s, s], [-s, -s]])

    with Scope() as scope:
        t0 = time.time()
        try:
            while True:
                a = time.time() - t0
                c, sn = math.cos(a), math.sin(a)
                rot = square @ np.array([[c, -sn], [sn, c]])
                scope.show([circle, rot])
                time.sleep(1 / 30)
        except KeyboardInterrupt:
            pass