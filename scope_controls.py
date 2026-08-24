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
    "gamma":    ("gamma",    "add", 0.4, 5.0, False),
    "lowpass":  ("lowpass",  "cyc", None, None, False),
    "rows":     ("rows",     "add", 0, 400, True),
}

LOWPASS_CYCLE = [None, 12000.0, 6000.0, 3000.0, 1500.0]

HELP = """
scope live controls
  -  / =    trim        down / up      (dark-cell cutoff; raises kill stray lines)
  ,  / .    density     finer / coarser (samples per cell; below ~0.3 flecks)
  [  / ]    gamma       down / up      (contrast of the dwell curve)
  l         lowpass     cycle off -> 12k -> 6k -> 3k -> 1.5k
  v         vector / raster
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
        cur = self.state.get(attr)
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
            s["raster"] = not s.get("raster", False)
            self.dirty = True
            self.message = "mode = " + ("RASTER" if s["raster"] else "VECTOR")
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
    out = ["--scope-raster"] if s.get("raster") else []
    out.append(f"--scope-trim {s.get('trim', 0.02):g}")
    out.append(f"--scope-gamma {s.get('gamma', 2.2):g}")
    if abs(s.get("density", 1.0) - 1.0) > 1e-6:
        out.append(f"--scope-density {s['density']:g}")
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