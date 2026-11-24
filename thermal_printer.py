"""Print what is on screen to an Epson ESC/POS thermal receipt printer.

Set up for the MICROS-branded Epson TM-T88V (C31CA85A6001) with its
Ethernet interface: raw ESC/POS over TCP port 9100, 80 mm paper, 576 dots
across. It is a thermal printer, so it prints dots: by default the window's
own pixels are read back after a frame is drawn, turned grey, fitted to the
paper (a landscape window turns to run along the paper), dithered to black
and white (Floyd-Steinberg) and sent as raster lines (GS v 0). That prints
whatever is drawn, in any local mode.

'text' mode is the alternative for the ASCII display: the characters on
screen in the printer's own font B. A 9 x 18-dot cell is ASCII_FONT_RATIO
(0.5), so the frame keeps its proportions; the black background stays blank
paper, cells too dark to see on screen print blank, the --ascii-bio phrase
is spelled again after any shrinking so words stay whole, and a narrow
frame is magnified to fill the width.

FramePrinter separates when from what: the random timer, P and a double tap
only ask for a print (request/tick); the display loop, which owns the GL
context, takes the request after drawing a frame and hands over the pixels
(or the characters). Converting and sending run on their own thread.
"""
import random
import socket
import threading
import time

import math

import numpy as np

ESC, GS = b'\x1b', b'\x1d'
FONT_B_DOTS = (9, 17)          # TM-T88V font B, dots wide x high
LINE_DOTS = 18                 # 9 x 18 cells: the 0.5 the frame is sampled with


def paper_indices(base, shown, palette_size, invert=False):
    """Indices into GLYPHS for paper: the characters on screen, or with
    ``invert`` the palette cells flipped (bright screen -> blank paper);
    phrase letters (where ``shown`` differs) are kept either way."""
    if not invert:
        return shown
    inverted = (palette_size - 1) - base
    return np.where(shown != base, shown, inverted)


def blank_unseen(indices, colours, palette_size, dark):
    """Blank the cells drawn so dark on screen that they can hardly be seen.
    The screen carries brightness in colour too; paper has only ink, so
    without this the background's faint dots print as full black ones."""
    if colours is None or dark <= 0:
        return indices
    seen = colours.max(axis=-1) >= dark
    return np.where(seen, indices, palette_size - 1)


def paper_cells(rows, cols, columns):
    """Nearest cells for a grid wider than the paper (keeping its shape),
    as (ys, xs); None when it fits."""
    if cols <= columns:
        return None
    new_rows = max(1, round(rows * columns / cols))
    ys = np.minimum(rows - 1, ((np.arange(new_rows) + .5) * rows / new_rows).astype(int))
    xs = np.minimum(cols - 1, ((np.arange(columns) + .5) * cols / columns).astype(int))
    return np.ix_(ys, xs)


def fit_to_paper(indices, columns):
    """Resample a grid wider than the paper; return it with its
    magnification (1-8) so a narrow one fills the paper."""
    cells = paper_cells(*indices.shape, columns)
    if cells is not None:
        return indices[cells], 1
    return indices, max(1, min(8, columns // max(1, indices.shape[1])))


def frame_to_escpos(base, shown, glyphs, palette_size, columns=64, feed_lines=10, invert=False,
                    colours=None, dark=32, respell=None):
    """ESC/POS bytes that print one frame and cut the paper.

    A frame wider than the paper is resampled first and, given ``respell``
    (ascii_converter.apply_bio), its phrase is spelled again on the smaller
    grid, as the screen spells after zooming; resampling the spelled grid
    would cut letters out of the words.
    """
    cells = paper_cells(*base.shape, columns)
    if cells is not None:
        base = base[cells]
        colours = None if colours is None else colours[cells]
        shown = respell(base) if respell is not None else shown[cells]
    paper = paper_indices(base, shown, palette_size, invert)
    if not invert:
        paper = blank_unseen(paper, colours, palette_size, dark)
    grid, scale = fit_to_paper(paper, columns)
    lines = [''.join(row).rstrip() for row in glyphs[grid]]
    text = '\n'.join(lines).encode('cp437', 'replace')
    size = (scale - 1) << 4 | (scale - 1)              # GS ! : width and height multiple
    return b''.join([
        ESC + b'@',                                    # reset to defaults
        ESC + b'a\x01',                                # centre a narrow frame
        ESC + b'M\x01',                                # font B
        ESC + b'3' + bytes([min(255, LINE_DOTS * scale)]),   # row pitch keeps the 0.5 cell
        GS + b'!' + bytes([size]),
        text, b'\n',
        GS + b'!\x00', ESC + b'2',                     # back to normal size and spacing
        feed_and_cut(feed_lines),
    ])


PAPER_DOTS = 576               # TM-T88V on 80 mm paper
# Small raster bands: should the printer reset part way through a print (a
# dense picture draws the most current), only the band it was in can turn
# to garbage; the next band's GS v 0 header starts a fresh picture command.
# With 256-line bands the rest of the picture printed as rolls of text.
RASTER_BAND = 32               # raster lines per GS v 0 command (2.3 KB at 576 dots)


def screen_to_ink(rgb, dots=PAPER_DOTS, rotate=None, ink=.85):
    """A picture (rows top-down, RGB, already upright and cut to the
    picture) as a dithered ink mask ``dots`` wide: True where the printer
    burns a dot, dark as ink, the full width of the paper. ``rotate='auto'``
    turns a landscape picture to run along the paper instead. ``ink`` caps
    the share of dots burnt where the picture is black (PRINTER_INK): full
    black still reads as black on thermal paper, with less heat and
    current."""
    from PIL import Image, ImageOps
    image = Image.fromarray(np.ascontiguousarray(rgb)).convert('L')
    if rotate == 'auto' and image.width > image.height:
        image = image.rotate(90, expand=True)
    height = max(1, round(image.height * dots / image.width))
    image = image.resize((dots, height), Image.LANCZOS)
    image = ImageOps.invert(image)                 # dark -> set bits
    ink = min(1.0, max(0.0, float(ink)))
    if ink < 1.0:
        image = image.point(lambda v: int(v * ink + .5))
    # Floyd-Steinberg dithering to one bit: the set bits are the ink.
    return np.asarray(image.convert('1'), dtype=bool)


def feed_and_cut(lines=10):
    """Feed ``lines`` blank lines past the print, then cut there."""
    return ESC + b'd' + bytes([max(0, min(255, int(lines)))]) + GS + b'V\x00'


def print_speed(level):
    """GS ( K <fn 50>: print speed, 1 (slowest) to 13; None leaves it."""
    if level is None:
        return b''
    return GS + b'(K\x02\x002' + bytes([max(1, min(13, int(level)))])


def ink_to_escpos(ink, feed_lines=10, speed=2):
    """ESC/POS raster (GS v 0) for an ink mask, in small bands, at a low
    print speed (fewer dots heated at once, so less peak current), then a
    feed and a cut."""
    rows, width = ink.shape
    packed = np.packbits(ink, axis=1)              # MSB first: the leftmost dot
    row_bytes = packed.shape[1]
    out = [ESC + b'@', print_speed(speed)]
    for top in range(0, rows, RASTER_BAND):
        band = packed[top:top + RASTER_BAND]
        out.append(GS + b'v0\x00' + bytes([row_bytes & 255, row_bytes >> 8,
                                           len(band) & 255, len(band) >> 8]))
        out.append(band.tobytes())
    out.append(feed_and_cut(feed_lines))
    return b''.join(out)


class DoubleTap:
    """Two presses close in time and place: a double tap on the touchscreen
    (it reaches the window as mouse clicks) or a double click."""

    def __init__(self, seconds=.4, distance=40.0):
        self.seconds, self.distance = seconds, distance
        self._last = None                  # (time, x, y) of an unpaired press

    def press(self, when, x, y):
        last = self._last
        if (last is not None and when - last[0] <= self.seconds
                and math.hypot(x - last[1], y - last[2]) <= self.distance):
            self._last = None              # a third tap starts afresh
            return True
        self._last = (when, x, y)
        return False


class FramePrinter:
    """Prints frames, asked for at random times (tick), on P and on a double
    tap (request); the display loop takes a request once a frame is drawn
    and hands over its pixels (print_screen) or characters (print_text).
    Converting and sending run on their own thread, one print at a time, so
    the display never waits on the printer."""

    def __init__(self, host, port=9100, every=(120.0, 900.0), mode='screen',
                 dots=PAPER_DOTS, rotate=None, columns=64, dark=32, feed_lines=10,
                 speed=2, ink=.85,
                 respell=None, clock=time.monotonic, rng=None, send=None, random_prints=True):
        self.host, self.port = host, int(port)
        self.mode = mode if mode in ('screen', 'text') else 'screen'
        self.dots, self.rotate, self.feed_lines = int(dots), rotate, int(feed_lines)
        self.speed, self.ink = speed, ink
        self.columns, self.dark, self.respell = int(columns), int(dark), respell
        low, high = every
        self.every = (max(1.0, float(low)), max(1.0, float(low), float(high)))
        self.clock = clock
        self.rng = rng or random.Random()
        self._send = send or self._send_tcp
        self._busy = threading.Lock()
        self.random = bool(random_prints)  # False: P only, no random prints
        self.wanted = False
        self.next_at = self.clock() + self.rng.uniform(*self.every)
        self.printed = 0

    # When ---------------------------------------------------------------------
    def request(self):
        """Ask for the next drawn frame (P, a double tap)."""
        self.wanted = True

    def tick(self):
        """Every loop: ask for a frame when the random time has come."""
        if self.random and self.clock() >= self.next_at:
            self.next_at = self.clock() + self.rng.uniform(*self.every)
            self.wanted = True

    def take(self):
        """True once per request, when a print can start now."""
        if not self.wanted or self._busy.locked():
            return False
        self.wanted = False
        return True

    # What ---------------------------------------------------------------------
    def print_screen(self, rgb):
        """The window's pixels (rows top-down, RGB uint8)."""
        size = f"{rgb.shape[1]}x{rgb.shape[0]} px"
        return self._submit(lambda: ink_to_escpos(
            screen_to_ink(rgb, self.dots, self.rotate, self.ink),
            self.feed_lines, self.speed), size)

    def print_text(self, frame):
        """The ASCII display's characters: (base, shown, glyphs,
        palette_size[, colours])."""
        if frame is None:
            return False
        base, shown, glyphs, palette_size = frame[:4]
        colours = frame[4] if len(frame) > 4 else None
        base, shown = base.copy(), shown.copy()
        colours = None if colours is None else colours.copy()
        size = f"{base.shape[1]}x{base.shape[0]} characters"
        return self._submit(lambda: frame_to_escpos(
            base, shown, glyphs, palette_size, self.columns, self.feed_lines,
            colours=colours, dark=self.dark, respell=self.respell), size)

    def _submit(self, build, size):
        if not self._busy.acquire(blocking=False):
            return False

        def run():
            try:
                self._send(build())
                self.printed += 1
                print(f"[PRINTER] Frame {self.printed} printed ({size}) at {self.host}")
            except OSError as exc:
                print(f"⚠️  [PRINTER] {self.host}:{self.port}: {exc}")
            except Exception as exc:               # a bad frame must not stop the display
                print(f"⚠️  [PRINTER] Could not print the frame: {exc}")
            finally:
                self._busy.release()

        threading.Thread(target=run, name='frame-printer', daemon=True).start()
        return True

    def _send_tcp(self, data):
        # A short wait to connect, then a long one to send: a dark picture
        # makes the printer slow itself down, and giving up mid-picture
        # leaves it waiting for the rest, so it prints the next job as
        # garbage. Only a printer gone for minutes is given up on.
        with socket.create_connection((self.host, self.port), timeout=5) as sock:
            sock.settimeout(300)
            sock.sendall(data)
