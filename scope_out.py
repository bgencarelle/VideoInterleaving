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
import time

SCOPE_OUT_API_VERSION = 5  # rotation + mirror output, always-on shaped X trigger

try:
    import settings as settings_mod
except Exception:                    # usable standalone, outside the repo
    settings_mod = None
import numpy as np
try:
    import sounddevice as sd
except Exception as _sd_err:            # deliberately broad -- see below
    # sounddevice calls Pa_Initialize() at IMPORT time, so a headless host
    # fails before this module even finishes loading, and nothing in the
    # project can run -- not even the paths that never touch a device.
    #
    # The exception is NOT reliably ImportError or OSError:
    #   no libportaudio2        -> OSError
    #   no PulseAudio server    -> sounddevice.PortAudioError, which subclasses
    #                              plain Exception and nothing narrower
    #   no ALSA config          -> PortAudioError again, different host error
    # A server typically hits the middle one: PortAudio is installed and works,
    # it just has no session bus to reach a sound server through. Catching
    # narrowly here is how this failed the first time.
    sd = None
    _SD_IMPORT_ERROR = _sd_err
else:
    _SD_IMPORT_ERROR = None


def have_audio():
    """False when PortAudio is absent -- only --device null will work."""
    return sd is not None

SAMPLE_RATE = 48_000     # physical-device fallback; normally read from device
NULL_SAMPLE_RATE = 96_000  # virtual trace budget; no DAC compatibility limit
FPS = 50
SAMPLES_PER_FRAME = SAMPLE_RATE // FPS   # fallback resolution budget

JUMP_GAIN = 0.12   # <1 -> fewer samples spent on travel moves -> dimmer
SMOOTH = 5         # circular box filter width; tames DAC ringing at corners
LEVEL = 0.9        # peak output amplitude, keep below 1.0

# A frame whose peak-to-peak is below this on BOTH channels is not a picture,
# it is a stationary beam.  Full scale is 1.8 peak-to-peak, so this is ~5e-5 of
# the range: far below anything a real image produces, and far above the
# round-off left by the filters a frame has already been through.
PARK_PTP = 1e-4


def beam_is_parked(frame, eps=PARK_PTP):
    """True when every sample in the frame is effectively the same position.

    "Never park the beam" is stated in four places in this codebase and was
    enforced in none of them: rasterize() idles on a circle, BufferedSource holds
    its last sample, StochasticGen substitutes a micro-circle for a one-pixel
    source -- but a frame that collapses anywhere else reached the DAC intact.
    A stationary beam is a full-brightness dot that burns phosphor, so the
    check belongs at the one point every frame passes through, not in each
    producer that might create one.

    ptp per channel rather than diff(): no allocation of an (n, 2) temporary in
    a path that runs once per trace.
    """
    f = np.asarray(frame)
    if len(f) < 2:
        return True
    # ptp propagates NaN and inf, so both are caught from the two reductions we
    # already need -- `not (v >= 0)` is True for NaN and `v == inf` for inf.
    # An isfinite().all() here would be correct too, but it allocates a full
    # boolean array, and this runs on the audio thread once per callback.
    px, py = np.ptp(f[:, 0]), np.ptp(f[:, 1])
    if not (px >= 0.0) or not (py >= 0.0) or px == np.inf or py == np.inf:
        # Not a picture. Treat it as parked so the caller replaces it with
        # something the deflection amplifiers can actually follow.
        return True
    return bool(px < eps and py < eps)


def unpark_frame(frame, phase=0, level=LEVEL, eps=PARK_PTP):
    """Replace a stationary frame with a small circle about the same point.

    Matches StochasticGen's existing one-pixel guard: the radius is small
    enough that the picture is still recognisably "a dot there" rather than
    jumping somewhere else, but the energy is spread over a ring instead of a
    single spot.  ``phase`` advances the start angle so consecutive parked
    frames do not retrace the identical ring.
    """
    f = np.ascontiguousarray(np.asarray(frame, dtype=np.float32)[:, :2])
    n = max(2, len(f))
    radius = level * 0.02
    centre = np.asarray(f[0] if len(f) else (0.0, 0.0), dtype=np.float64)
    if not np.isfinite(centre).all():
        centre = np.zeros(2)          # a NaN park has no position to keep
    # Clamp the CENTRE, not the finished ring. Clipping the ring is what made
    # this function able to produce a parked beam of its own: a park at the
    # -0.936 pedestal that render_yt_grid uses for empty rows is outside
    # +-LEVEL on both axes, so every point of the ring clipped to the same
    # corner and the guard reported success while the dot kept burning.
    centre = np.clip(centre, -(level - radius), level - radius)
    th = 2.0 * np.pi * (np.arange(n) + int(phase)) / n
    out = centre + radius * np.column_stack([np.cos(th), np.sin(th)])
    return np.ascontiguousarray(out[:len(f)] if len(f) >= 2 else out,
                                dtype=np.float32)


def rotate_frame(frame, degrees):
    """Rotate XY samples around the display centre in 90-degree steps.

    Scope mode has no GLFW window, so its transform must happen on the signal
    itself. Keeping the operation here, at the final output boundary, makes the
    same rotation apply to every renderer and to both frame and realtime paths.
    """
    angle = int(degrees) % 360
    if angle % 90:
        raise ValueError("scope rotation must be a multiple of 90 degrees")
    src = np.asarray(frame, dtype=np.float32)
    if angle == 0:
        return np.ascontiguousarray(src)
    out = np.empty_like(src)
    if angle == 90:
        out[:, 0] = -src[:, 1]
        out[:, 1] = src[:, 0]
    elif angle == 180:
        out[:] = -src
    else:  # 270
        out[:, 0] = src[:, 1]
        out[:, 1] = -src[:, 0]
    return np.ascontiguousarray(out)


def mirror_frame(frame, mirror):
    """Flip XY samples left-right about the display centre.

    Unlike rotation, this one belongs at the OUTPUT boundary rather than in
    image space. Rotation at 90/270 would move the fast sweep off X and break
    raster timing and the Y-T trigger, which is why scope_display rotates the
    luminance instead. A mirror only negates X: the sweep stays on X, every
    row keeps its slot and its duration, and the result is bit-for-bit the
    reflection you would get by mirroring the source image first. Doing it
    here means it costs one sign flip instead of a second orientation
    parameter threaded through every renderer.
    """
    src = np.asarray(frame, dtype=np.float32)
    if not mirror:
        return np.ascontiguousarray(src)
    out = src.copy()
    out[:, 0] = -out[:, 0]
    return np.ascontiguousarray(out)


TRIGGER_SHAPES = ("ramp", "step")


def clip_for_trigger(frame, level=LEVEL, floor=-0.98):
    """Keep picture content strictly below the trigger threshold.

    The marker is unique only because nothing else crosses +0.95. Geometry
    obeys that by construction -- it is built inside +-LEVEL -- but a lowpass
    or DC pre-emphasis applied afterwards can ring above it, and a second
    rising crossing per trace is a scope that will not hold a picture still.
    The floor is below -LEVEL because fixed row timing parks empty slots on a
    pedestal at -0.936 that is deliberately outside the picture range.
    """
    out = np.array(frame, dtype=np.float32, copy=True)
    np.clip(out[:, 0], floor, level, out=out[:, 0])
    # Y is bounded too. It cannot forge a trigger -- nothing looks at Y -- but
    # a single +inf from a filter or a division upstream drives the vertical
    # deflection to whatever the amplifier will do, and the marker's own rail
    # is the largest legitimate value on that axis.
    np.clip(out[:, 1], -1.0, 1.0, out=out[:, 1])
    return out


def trigger_frame(frame, trigger_samples=24, trigger_level=0.99,
                  offset=0, period=None, shape="ramp"):
    """Insert one threshold-unique rising edge on X per trace.

    Picture content is limited to LEVEL (0.9) and the marker reaches 0.99, so
    a trigger level around +0.95 sees only this edge. ``offset`` supports
    arbitrary audio callback block boundaries; ``period`` is the trace length
    at which the marker repeats.

    Two shapes, because the marker has to serve two displays at once:

    ``step`` is the original: half the marker parked at -0.99, half at +0.99,
    Y untouched. On a Y-T scope that is ideal. On an XY scope those are two
    STATIONARY samples runs, and a stationary beam is the brightest thing on
    the screen -- two hard dots and a streak across the picture. It is kept
    for scopes whose trigger will not take the ramp.

    ``ramp`` is the default, and the reason the marker can be on all the time.
    An edge trigger fires on the CROSSING; dwelling at the extremes buys
    nothing but brightness. So X sweeps monotonically from -0.99 to +0.99 with
    only a couple of samples of hold at each end for the comparator, and Y is
    pushed to the same rail -- outside the +-LEVEL picture box, the same
    off-screen-excursion trick apply_overscan() uses for travel moves. Set the
    scope so +-0.9 fills the screen and the whole marker deflects past the
    phosphor: identical trigger, no ink.
    """
    if shape not in TRIGGER_SHAPES:
        raise ValueError(f"trigger shape must be one of {TRIGGER_SHAPES}")
    src = np.asarray(frame, dtype=np.float32)
    if src.ndim != 2 or src.shape[1] < 2:
        raise ValueError("trigger source must have shape (samples, 2)")
    count = len(src)
    if count == 0:
        return np.empty((0, 2), dtype=np.float32)
    out = np.ascontiguousarray(src[:, :2]).copy()
    cycle = max(1, int(period if period is not None else count))
    marker = max(4, min(int(trigger_samples), cycle))
    phase = (np.arange(count, dtype=np.int64) + int(offset)) % cycle
    level = min(1.0, max(float(trigger_level), LEVEL + 0.01))

    if shape == "step":
        low = phase < max(2, marker // 2)
        high = (phase >= max(2, marker // 2)) & (phase < marker)
        out[low, 0] = -level
        out[high, 0] = level
        return out

    inside = phase < marker
    if not inside.any():
        return out
    p = phase[inside].astype(np.float64)
    hold = max(1, marker // 8)
    span = max(1, marker - 2 * hold - 1)
    # Monotonic: exactly one crossing of any threshold in (-level, +level).
    t = np.clip((p - hold) / span, 0.0, 1.0)
    out[inside, 0] = (-level + 2.0 * level * t).astype(np.float32)
    out[inside, 1] = np.float32(level)
    return out


# The name this shipped under. Kept so settings.py edits, saved command lines
# and the existing tests that pin the step waveform keep working.
yt_trigger_frame = trigger_frame


def marker_window(count, trigger_level=0.99, shape="ramp"):
    """The marker as its OWN samples, to be prepended to a complete picture.

    trigger_frame() overwrites the head of a trace, which is what the realtime
    path has to do -- it has no frame boundary to prepend at. For a whole
    frame that is the wrong trade: it destroyed the first `count` samples of
    every picture and, with them, the exact frame-boundary handoff that
    scope_display restores into frame[0]. Reserving the marker its own window
    costs 0.75% of the refresh rate instead of 0.75% of the picture, and the
    handoff survives because the picture's first sample is still its own.

    The window is entered and left in ONE sample each. That is deliberate and
    it is the dimmest possible transit: brightness is dwell per unit length, so
    spending more samples getting to the rail would make the move brighter, not
    dimmer -- the same reasoning apply_overscan() documents. Measured against a
    normal raster frame, a transit is ~84x faster than picture ink.
    """
    n = max(2, int(count))
    level = min(1.0, max(float(trigger_level), LEVEL + 0.01))
    out = np.empty((n, 2), dtype=np.float32)
    if shape == "step":
        split = max(1, n // 2)
        out[:split] = (-level, level)
        out[split:] = (level, level)
        return out
    if shape != "ramp":
        raise ValueError(f"trigger shape must be one of {TRIGGER_SHAPES}")
    hold = max(1, n // 8)
    span = max(1, n - 2 * hold - 1)
    t = np.clip((np.arange(n, dtype=np.float64) - hold) / span, 0.0, 1.0)
    out[:, 0] = (-level + 2.0 * level * t).astype(np.float32)
    out[:, 1] = np.float32(level)
    return out


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
    if isinstance(spec, str) and spec.strip().lower() in ("null", "none", "off"):
        return "null"          # handled by Scope, never reaches PortAudio
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
    # Renormalising is only meaningful if something survived DC removal.  A
    # constant frame is ENTIRELY bin 0, so killing that bin leaves nothing but
    # FFT round-off -- and `level / peak` then amplifies that round-off to full
    # scale, turning a parked beam into a full-screen noise smear.  Measured:
    # a constant frame came out at 0.9 peak, and a frame carrying 1e-4 of real
    # motion was given 9000x gain.  Below the floor, hand back the input and
    # let the caller's park guard deal with it.
    if peak < 1e-3:
        return np.asarray(frame, dtype=np.float32)
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
        # NOT zeros: the callback can beat the producer thread to the first
        # block, and holding the "last" sample would then hold (0, 0) -- the
        # parked centre dot this class exists to avoid -- for as long as the
        # cold start lasts.  Start on the idle ring instead.
        self._last = np.zeros(channels, dtype=np.float32)
        if channels >= 2:
            self._last[0] = LEVEL
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


class NullStream:
    """A sound card that isn't there.

    A VPS has no audio device, so sd.OutputStream() fails and scope mode will
    not start at all.  But the samples still have to be GENERATED -- the whole
    point of the server build is to hand the picture to browsers and let each
    client render the audio on its own hardware.

    So this presents the OutputStream interface and drives the callback from a
    software clock instead of a DAC.  Same cadence, same frame boundaries, no
    hardware.  It intentionally does NOT keep the samples: nothing on the
    server listens to them.  What the browser needs is the luminance, and that
    is published separately by the tap.
    """

    def __init__(self, samplerate, blocksize, callback):
        self.samplerate = float(samplerate)
        self.blocksize = int(blocksize or 512)
        self._cb = callback
        self.latency = 0.0
        self._stop = threading.Event()
        self._buf = np.zeros((self.blocksize, 2), dtype=np.float32)
        self._t = None

    def start(self):
        if self._t is not None:
            return
        self._t = threading.Thread(target=self._run, daemon=True,
                                   name="scope-null-clock")
        self._t.start()

    def _run(self):
        period = self.blocksize / self.samplerate
        nxt = time.monotonic()
        while not self._stop.is_set():
            try:
                self._cb(self._buf, self.blocksize, None, None)
            except Exception:
                pass
            nxt += period
            # absolute deadline, not sleep(period): sleeping accumulates drift,
            # and the index clock is derived from how many blocks have gone by
            delay = nxt - time.monotonic()
            if delay > 0:
                self._stop.wait(delay)
            else:
                nxt = time.monotonic()          # fell behind; resynchronise

    def stop(self):
        self._stop.set()

    def close(self):
        self._stop.set()


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
                 invert_y=True, swap_xy=False, source=None, rotation=0,
                 lowpass_hz=None, lowpass_taper=0.0, blocksize=512,
                 yt_mode=None, yt_trigger_us=250.0,
                 yt_trigger_level=0.99, mirror=False,
                 trigger=True, trigger_shape="ramp"):
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
        self.set_rotation(rotation)
        self.set_mirror(mirror)
        _null = (isinstance(device, str) and device.strip().lower()
                 in ("null", "none", "off"))
        if sd is None and not _null:
            raise RuntimeError(
                f"PortAudio is unavailable ({_SD_IMPORT_ERROR}). On a host with "
                "no sound card, run with --device null: the samples are "
                "generated for browsers to render, not played here.")
        if samplerate is None:
            if _null:
                # No device constrains the virtual renderer. Use the same
                # high-resolution budget as the normal 96 kHz scope route:
                # at 30 traces/s this gives the preview 3200 XY positions
                # instead of 1600. Browser-local audio still builds from
                # luminance at its own AudioContext.sampleRate.
                samplerate = NULL_SAMPLE_RATE
            else:
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
        # The marker is no longer a mode. It is a property of the output, on
        # by default, because a ramp-shaped marker parked outside the picture
        # box costs an XY display nothing and is the entire requirement for a
        # single-channel Y-T display. `yt_mode` is the name it shipped under
        # and still wins when passed explicitly.
        self.trigger = bool(trigger if yt_mode is None else yt_mode)
        if trigger_shape not in TRIGGER_SHAPES:
            raise ValueError(f"trigger shape must be one of {TRIGGER_SHAPES}")
        self.trigger_shape = trigger_shape
        self.yt_mode = self.trigger          # legacy attribute name
        self.yt_trigger_us = max(float(yt_trigger_us), 1.0)
        # Bounded above as well as below. --scope-trigger-us only validated
        # "finite and > 0", so 40000 us at 96 kHz asked for 3840 marker samples
        # against a 3200-sample trace and silently replaced the entire picture.
        # An eighth of the trace is already a very generous edge.
        _ceiling = max(4, self.samples_per_frame // 8)
        _asked = max(4, round(float(samplerate) * self.yt_trigger_us
                              / 1_000_000.0))
        self.yt_trigger_samples = min(_asked, _ceiling)
        if self.yt_trigger_samples != _asked:
            print(f"[SCOPE] trigger marker {self.yt_trigger_us:g} us is "
                  f"{_asked} samples of a {self.samples_per_frame}-sample "
                  f"trace; clamped to {self.yt_trigger_samples}.")
            self.yt_trigger_us = (self.yt_trigger_samples * 1_000_000.0
                                  / float(samplerate))
        self.yt_trigger_level = min(
            1.0, max(float(yt_trigger_level), LEVEL + 0.01))
        self._yt_pos = 0
        # Built once. The window is identical every trace, and rebuilding it
        # per callback was ~15 numpy allocations on the audio thread.
        self._marker = marker_window(self.yt_trigger_samples,
                                     self.yt_trigger_level, self.trigger_shape)
        self._frame = rasterize([], self.samples_per_frame)  # idle circle, never a parked dot
        if self.trigger:
            self._frame = np.vstack((self._marker, self._frame))
        self._pending = None
        self._lock = threading.Lock()
        self._pos = 0
        self.frames_drawn = 0     # complete traces emitted
        self.frames_dropped = 0   # indices superseded before they were drawn
        self.beams_unparked = 0   # frames that arrived stationary and were rung
        self.dac_dropouts = 0     # callbacks PortAudio flagged; each one is a
                                  # block of silence it filled in for us, which
                                  # on a scope is a dot at screen centre
        # Held on underflow instead of zeroing.  Seeded from the idle circle so
        # that a failure in the very first callback holds a point on the ring
        # rather than the (0, 0) a zeros() default would give.
        self._last_out = self._frame[0].copy()
        self.null = (isinstance(device, str) and device.strip().lower()
                     in ("null", "none", "off"))
        if not self.null:
            warn_if_builtin(device)
        # An explicit blocksize matters: with latency="low" and blocksize
        # unset, PortAudio picks the smallest buffer the device allows, so the
        # audio thread wakes ~1500 times a second and takes the GIL each time.
        # 512 frames is ~5 ms at 96 kHz -- far below the trace period, and a
        # third the wakeups.
        self.blocksize = int(blocksize or 0)
        if self.null:
            print(f"[SCOPE] no audio device (--device null): generating at "
                  f"{samplerate:.0f} Hz for the browser to render")
            self.stream = NullStream(samplerate, self.blocksize or 512,
                                     self._callback)
            return
        self.stream = sd.OutputStream(
            samplerate=samplerate, channels=2, dtype="float32",
            device=device, blocksize=self.blocksize,
            latency="low", callback=self._callback)

    @property
    def trace_samples(self):
        """Samples the DAC actually consumes per trace, marker included.

        samples_per_frame is the PICTURE budget -- what the renderers are given
        -- and the marker is added to it, so the refresh rate is
        samplerate / trace_samples, not samplerate / samples_per_frame.
        """
        return self.samples_per_frame + (len(self._marker) if self.trigger else 0)

    def _stamp_marker(self, block, pos):
        """Write the marker into a continuous block, in place, no allocation.

        The realtime path has no frame boundary, so the marker repeats on a
        sample counter. At most a couple of windows intersect any one block, so
        this is a slice copy per window rather than the phase-array arithmetic
        trigger_frame() does -- that ran on the audio thread and allocated
        about fifteen arrays per callback.
        """
        n = len(block)
        period = max(1, self.samples_per_frame)
        marker = len(self._marker)
        # A window that began in the previous block and runs into this one.
        if pos < marker:
            k = min(marker - pos, n)
            block[:k] = self._marker[pos:pos + k]
        start = (-pos) % period                  # next window boundary
        while start < n:
            k = min(marker, n - start)
            block[start:start + k] = self._marker[:k]
            start += period

    def _callback(self, outdata, frames, time_info, status):
        # PortAudio fills a missed block with SILENCE, and silence on both
        # channels is a stationary full-brightness dot at screen centre.  We
        # cannot retrieve those samples, but an underrun that is never counted
        # is a bright spot with no explanation; counted, it is a number on the
        # dashboard.  No formatting here -- this is the audio thread.
        #
        # output_underflow only: CallbackFlags is truthy for priming_output
        # too, which every clean stream sets on its first callbacks, and a
        # dashboard that reports dropouts on a healthy stream is worse than one
        # that reports none.
        if status is not None and getattr(status, "output_underflow", False):
            self.dac_dropouts += 1
        # Continuous source: content follows the clock in real time, with no
        # frame boundaries to wait for.
        if self.source is not None:
            try:
                buf = self.source(frames)
                # Mirror before the marker: negating X afterwards would turn
                # the trigger's rising edge into a falling one and the scope
                # would stop locking.
                rendered = mirror_frame(
                    rotate_frame(buf[:frames], self.rotation), self.mirror)
                # Guard BEFORE the marker, for the same reason show_frame does:
                # the marker is motion of its own, so a stream that collapsed
                # upstream -- a starved BufferedSource holding its last sample,
                # most likely -- would look alive by the time it was checked.
                if beam_is_parked(rendered):
                    rendered = unpark_frame(
                        rendered, phase=self.beams_unparked * 7)
                    self.beams_unparked += 1
                if self.trigger:
                    # Realtime is the one path that still OVERWRITES: it is a
                    # continuous stream with no frame boundary to prepend at,
                    # so the marker is stamped on a sample counter instead. It
                    # costs the same handful of samples per period; they just
                    # come out of the sweep rather than being reserved.
                    np.clip(rendered[:, 0], -0.98, LEVEL, out=rendered[:, 0])
                    np.clip(rendered[:, 1], -1.0, 1.0, out=rendered[:, 1])
                    self._stamp_marker(rendered, self._yt_pos)
                    self._yt_pos = ((self._yt_pos + frames)
                                    % self.samples_per_frame)
                outdata[:] = rendered
                self._last_out = rendered[-1].copy()
                self.frames_drawn += 1
            except Exception:
                # HOLD, never zero.  BufferedSource.__call__ already documents why
                # -- zero on both channels parks the beam at screen centre --
                # and this path was the one place that did the opposite.
                outdata[:] = self._last_out
                self.dac_dropouts += 1
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
        if frames:
            self._last_out = outdata[-1].copy()

    def ready(self):
        """True when the last queued frame has been taken by the callback.

        Interlaced fields must be handed over one per trace, in order: queue
        two and the second silently replaces the first, so half the rows are
        never drawn.  Gating on this keeps the producer exactly one trace
        ahead, which is also what stops the chained sweep from breaking.
        """
        return self._pending is None

    def set_rotation(self, degrees):
        """Set output rotation, matching local mode's quarter-turn control."""
        angle = int(degrees) % 360
        if angle % 90:
            raise ValueError("scope rotation must be a multiple of 90 degrees")
        self.rotation = angle
        Scope._output_rotation = angle

    def set_mirror(self, mirror):
        """Set output mirroring, matching local mode's `m` key."""
        self.mirror = bool(mirror)
        Scope._output_mirror = self.mirror

    # --- optional preview tap -------------------------------------------
    # Every render path lands in show_frame(): show() rasterises then calls it,
    # and raster mode calls it directly.  So one hook here catches raster,
    # vector, mix and the lowpass variants without touching any of them.
    #
    # Gated on demand.  _tap_until is pushed forward by each /scope/trace
    # request, so with nobody watching the page this costs one float compare
    # per trace and nothing else.  An installation running unattended for weeks
    # must not pay for a preview no one is looking at.
    _tap_until = 0.0
    _tap = {"seq": 0, "data": None}
    _tap_lock = threading.Lock()
    # How many consecutive traces make ONE PICTURE.  Set to the interlace field
    # count by the display engine.  A trace is a FIELD, not a picture, so a tap
    # that grabs one trace shows every Nth scanline -- 50% of the picture at
    # fields=2, 26% at fields=4.  For anyone driving real hardware that is
    # merely a wrong preview; for the many people whose ONLY display is the web
    # page, it is the whole image being wrong.
    _tap_fields = 1
    _tap_accum = []
    # The LUMINANCE the trace was built from. This is what a browser needs: it
    # renders its own trace, at its own AudioContext rate, on its own DAC.
    # Shipping this instead of PCM is ~19x less bandwidth, because lossy
    # compression is fatal to a waveform (the waveform IS the picture) but
    # nearly free on the luminance (the renderer quantises it to a ~50x66 grid
    # downstream regardless).
    _luma = {"seq": 0, "data": None}
    _luma_lock = threading.Lock()
    _output_rotation = 0
    _output_mirror = False

    @classmethod
    def publish_luma(cls, lum):
        """Called by the display engine with the composited luminance."""
        if lum is None:
            return
        try:
            q = np.clip(np.asarray(lum, dtype=np.float32), 0.0, 1.0)
            if cls._output_rotation:
                q = np.rot90(q, k=cls._output_rotation // 90)
            if cls._output_mirror:
                # The display engine rotates the luminance itself, so the
                # rotation above is normally a no-op -- but mirroring happens
                # at the output boundary, after this luminance was composited.
                # Without the flip here the browser's local render would be
                # the mirror image of what the hardware is drawing.
                q = np.ascontiguousarray(q[:, ::-1])
            with cls._luma_lock:
                cls._luma["seq"] += 1
                cls._luma["data"] = (q * 255.0).astype(np.uint8)
        except Exception:
            pass

    @classmethod
    def read_luma(cls):
        with cls._luma_lock:
            return cls._luma["seq"], cls._luma["data"]

    @classmethod
    def set_tap_fields(cls, fields):
        """Tell the tap how many traces to join before publishing a picture."""
        cls._tap_fields = max(1, int(fields))
        cls._tap_accum = []

    @classmethod
    def want_tap(cls, seconds=3.0):
        """Ask for preview frames for the next few seconds. Returns nothing."""
        cls._tap_until = time.monotonic() + seconds

    @classmethod
    def read_tap(cls):
        """(seq, (n,2) float32 array) of the latest trace, or (0, None)."""
        with cls._tap_lock:
            return cls._tap["seq"], cls._tap["data"]

    def _capture(self, frame):
        """Park a copy of the trace. Deliberately does NO rendering.

        Every sample is kept and nothing is decimated, because dwell is the
        image: brightness comes from how many samples land in a cell, so
        dropping every other one discards exactly the information the picture
        is made of.  A 3200x2 float32 copy is 25 KB and a few microseconds.

        Rendering happens on the HTTP thread instead, so the cost lands on
        whoever is watching rather than on the loop that has to hit a trace
        deadline every 16 ms.
        """
        if len(frame) == 0:
            return
        pts = np.array(frame, dtype=np.float32, copy=True) / max(LEVEL, 1e-9)
        k = Scope._tap_fields
        if k > 1:
            # Join consecutive traces into a whole picture. They are already
            # chained end-to-start, so concatenating is exactly what the beam
            # draws -- no seam to stitch.
            Scope._tap_accum.append(pts)
            if len(Scope._tap_accum) < k:
                return
            pts = np.concatenate(Scope._tap_accum[-k:], axis=0)
            Scope._tap_accum = []
        with Scope._tap_lock:
            Scope._tap["seq"] += 1
            Scope._tap["data"] = pts

    def show_frame(self, frame):
        """Queue a raw (n, 2) sample frame for the next frame boundary.

        Raster mode bypasses rasterize() because uneven dwell IS the image, so
        arc-length resampling would destroy it.  If an earlier frame is still
        waiting, it is discarded rather than queued: the scope should show the
        index that is current NOW, never fall behind replaying stale ones.
        """
        if self._pending is not None:
            self.frames_dropped += 1
        f = mirror_frame(rotate_frame(frame, self.rotation), self.mirror)
        # Before the filters, not after: lowpass ringing and the Y-T marker
        # both add motion of their own, so a picture that collapsed upstream
        # would still look alive by the time it reached the DAC.
        if beam_is_parked(f):
            f = unpark_frame(f, phase=self.beams_unparked * 7)
            self.beams_unparked += 1
        if self.lowpass_hz:
            f = lowpass_frame(f, self.samplerate, self.lowpass_hz,
                              self.lowpass_taper)
        if self.trigger:
            # Clip here, not in each renderer: the filters above are exactly
            # what can push picture content over the trigger threshold, and
            # this is the one place every path has already passed through
            # them.
            #
            # PREPEND, do not overwrite. The marker gets its own samples, so
            # the picture arrives whole and frame[0] -- which scope_display
            # sets to the exact beam handoff point -- is still the first thing
            # drawn after it. The cost moves from 0.75% of the picture to 0.75%
            # of the refresh rate.
            f = np.vstack((self._marker, clip_for_trigger(f)))
        if Scope._tap_until > time.monotonic():
            try:
                self._capture(f)
            except Exception:
                pass                       # a preview must never break audio
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

    # trigger=False: this is a MEASUREMENT pattern. The whole point of the
    # calibration square is that you judge a known shape against itself to
    # find AC-coupling distortion, and a marker stamped over the first samples
    # -- with Y pinned to the rail -- is exactly the kind of thing you would
    # then mistake for the distortion you came to look for.
    scope = Scope(fps=args.fps, trigger=False,
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
