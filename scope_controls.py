"""
scope_controls.py -- live keyboard tuning for scope mode.

Everything the runtime can change without a rebake, adjustable while watching
the actual scope.  Restarting to try a different trim is hopeless when the
thing you are judging is a beam in front of you.

Terminal handling is separated from the key mapping so the mapping can be
tested without a tty.  Works over SSH; silently does nothing when stdin is not
a terminal, so it is safe to leave enabled in a launch script.

Press h for the key map, p to print the current settings as flags you can
paste back into a command line or settings.py.
"""
import os
import sys

# name -> (attribute, step kind, lo, hi, needs_recalibration)
#
# Recalibration matters: trim and density change the GRID, so the calibrated
# grid and levels have to be recomputed.  Gamma and lowpass only reshape the
# existing grid, so they take effect immediately.
SPECS = {
    "trim":     ("trim",     "add", 0.0, 0.60, True),
    "density":  ("density",  "mul", 0.20, 4.0, True),
    "gamma":    ("gamma",    "add", 0.4, 10.0, False),
    "lowpass":  ("lowpass",  "cyc", None, None, False),
    "rows":     ("rows",     "add", 0, 400, True),
}

LOWPASS_CYCLE = [None, 12000.0, 6000.0, 3000.0, 1500.0]

HELP = """
scope live controls
  -  / =    trim        down / up      (dark-cell cutoff; raises kill stray lines)
  ,  / .    density     finer / coarser (samples per cell; below ~0.3 flecks)
  [  / ]    gamma       down / up      (raster dwell / stochastic probability)
  l         lowpass     cycle off -> 12k -> 6k -> 3k -> 1.5k
  v         mode         vector -> raster -> stochastic -> stipple -> fusion
  f         fusion       vrs -> vr -> sv -> sr (in fusion mode)
  w         sweep       alternate -> palindrome -> retrace
  a         autofit     on / off
  p         print current settings as command-line flags
  h         this help
  q         quit
"""


class KeyMap:
    """Pure key -> state change.  No terminal involved, so it is testable."""

    def __init__(self, state):
        self.state = state
        self.dirty = False          # set when the grid must be recalibrated
        self.quit = False
        self.message = ""

    def _bump(self, name, direction):
        attr, kind, lo, hi, recal = SPECS[name]
        if name == "gamma":
            mode = self.state.get("mode")
            fusion = self.state.get("fusion_components", "vrs")
            attr = ("stochastic_gamma"
                    if mode in ("stochastic", "stipple")
                    or (mode == "fusion" and "s" in fusion)
                    else "raster_gamma")
        cur = self.state.get(attr)
        if cur is None and name == "gamma":
            cur = self.state.get("gamma")
        if kind == "cyc":
            try:
                i = LOWPASS_CYCLE.index(cur)
            except ValueError:
                i = 0
            new = LOWPASS_CYCLE[(i + 1) % len(LOWPASS_CYCLE)]
        elif kind == "mul":
            new = min(max((cur or 1.0) * (1.25 if direction > 0 else 0.8), lo), hi)
        else:
            step = 0.02 if name == "trim" else (0.2 if name == "gamma" else 4)
            new = min(max((cur or 0) + step * direction, lo), hi)
            if name == "rows" and new <= 0:
                new = None
        self.state[attr] = new
        if name == "gamma":
            self.state["gamma"] = new
        if recal:
            self.dirty = True
        shown = "auto" if new is None and name == "rows" else (
            "off" if new is None else (f"{new:g}"))
        self.message = f"{name} = {shown}"

    def feed(self, ch):
        s = self.state
        if ch in ("-", "_"):
            self._bump("trim", -1)
        elif ch in ("=", "+"):
            self._bump("trim", +1)
        elif ch == ",":
            self._bump("density", -1)
        elif ch == ".":
            self._bump("density", +1)
        elif ch == "[":
            self._bump("gamma", -1)
        elif ch == "]":
            self._bump("gamma", +1)
        elif ch == "l":
            self._bump("lowpass", +1)
        elif ch == "v":
            if s.get("mode_locked"):
                self.message = "mode cycling unavailable in realtime/mix mode"
                return True
            order = ["vector", "raster", "stochastic", "stipple", "fusion"]
            current = s.get("mode", "raster" if s.get("raster") else "vector")
            i = order.index(current) if current in order else 0
            s["mode"] = order[(i + 1) % len(order)]
            s["raster"] = s["mode"] == "raster"  # legacy state readers
            gamma_key = ("stochastic_gamma"
                         if (s["mode"] in ("stochastic", "stipple")
                             or (s["mode"] == "fusion"
                                 and "s" in s.get("fusion_components", "vrs")))
                         else "raster_gamma")
            if gamma_key in s:
                s["gamma"] = s[gamma_key]
            self.dirty = True
            self.message = "mode = " + s["mode"].upper()
        elif ch == "f":
            if s.get("mode") != "fusion":
                self.message = "fusion combinations are available in FUSION mode"
                return True
            order = ["vrs", "vr", "sv", "sr"]
            current = s.get("fusion_components", "vrs")
            i = order.index(current) if current in order else 0
            s["fusion_components"] = order[(i + 1) % len(order)]
            gamma_key = ("stochastic_gamma"
                         if "s" in s["fusion_components"] else "raster_gamma")
            if gamma_key in s:
                s["gamma"] = s[gamma_key]
            self.dirty = True
            self.message = "fusion = " + s["fusion_components"].upper()
        elif ch == "w":
            order = ["alternate", "palindrome", "retrace"]
            i = order.index(s.get("sweep", "alternate")) if s.get("sweep") in order else 0
            s["sweep"] = order[(i + 1) % len(order)]
            self.message = "sweep = " + s["sweep"]
        elif ch == "a":
            s["autofit"] = not s.get("autofit", True)
            self.dirty = True
            self.message = "autofit = " + ("on" if s["autofit"] else "off")
        elif ch == "p":
            self.message = "\n" + as_flags(s)
        elif ch == "h":
            self.message = HELP
        elif ch in ("q", "\x03", "\x04"):
            self.quit = True
        else:
            return False
        return True


def as_flags(s):
    """Current state as flags you can paste into a command line."""
    mode = s.get("mode", "raster" if s.get("raster") else "vector")
    mix_hz = s.get("mix_hz")
    if mix_hz:
        out = [f"--scope-mix {mix_hz:g}",
               f"--scope-mix-duty {s.get('mix_duty', 0.5):g}"]
    else:
        out = [] if mode == "vector" else [f"--scope-mode {mode}"]
        if mode == "fusion":
            out.append(f"--scope-fusion {s.get('fusion_components', 'vrs')}")
    out.append(f"--scope-trim {s.get('trim', 0.02):g}")
    if mix_hz:
        raster_gamma = s.get("raster_gamma", 2.2)
        stochastic_gamma = s.get("stochastic_gamma", 2.0)
        out.append(f"--scope-gamma {raster_gamma:g}")
        if abs(stochastic_gamma - raster_gamma) > 1e-9:
            out.append(f"--scope-stochastic-gamma {stochastic_gamma:g}")
    elif mode == "fusion":
        raster_gamma = s.get("raster_gamma", 2.2)
        stochastic_gamma = s.get("stochastic_gamma", 2.0)
        out.append(f"--scope-gamma {raster_gamma:g}")
        if abs(stochastic_gamma - raster_gamma) > 1e-9:
            out.append(f"--scope-stochastic-gamma {stochastic_gamma:g}")
    elif mode in ("stochastic", "stipple"):
        gamma = s.get("stochastic_gamma", s.get("gamma", 2.0))
        out.append(f"--scope-gamma {gamma:g}")
        if mode == "stipple":
            out.append(
                f"--scope-stipple-points {int(s.get('stipple_points', 768))}")
    else:
        gamma = s.get("raster_gamma", s.get("gamma", 2.2))
        out.append(f"--scope-gamma {gamma:g}")
    if abs(s.get("density", 1.0) - 1.0) > 1e-6:
        out.append(f"--scope-density {s['density']:g}")
    if s.get("precondition") is not None:
        out.append(f"--scope-precondition {s['precondition']:g}")
    if s.get("rows"):
        out.append(f"--scope-rows {int(s['rows'])}")
    if s.get("sweep", "alternate") != "alternate":
        out.append(f"--scope-sweep {s['sweep']}")
    if s.get("lowpass"):
        out.append(f"--scope-lowpass {s['lowpass']:g}")
    if not s.get("autofit", True):
        out.append("--scope-no-autofit")
    return "  " + " ".join(out)


class Terminal:
    """Non-blocking raw-mode reader.  No-op when stdin is not a terminal."""

    def __init__(self):
        self.enabled = False
        self._saved = None
        try:
            if not sys.stdin.isatty():
                return
            import termios, tty                      # noqa: F401
            self._termios = termios
            self._saved = termios.tcgetattr(sys.stdin.fileno())
            tty.setcbreak(sys.stdin.fileno())
            self.enabled = True
        except Exception:
            self.enabled = False

    def read(self):
        """Every character waiting right now; empty list if none."""
        if not self.enabled:
            return []
        import select
        out = []
        while select.select([sys.stdin], [], [], 0)[0]:
            c = os.read(sys.stdin.fileno(), 1).decode("utf-8", "ignore")
            if not c:
                break
            out.append(c)
        return out

    def restore(self):
        if self._saved is not None:
            try:
                self._termios.tcsetattr(sys.stdin.fileno(),
                                        self._termios.TCSADRAIN, self._saved)
            except Exception:
                pass
            self._saved = None
        self.enabled = False
