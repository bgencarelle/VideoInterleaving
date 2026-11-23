"""Bounded, read-only Telnet negotiation and shared terminal-sized pictures."""
from collections import OrderedDict
import threading
import time

import ascii_converter

MAX_COLS = 160
MAX_ROWS = 60
SCALES = ((30, 20), (45, 30), (60, 40), (90, 60))
IAC, DO, DONT, WILL, WONT, SB, SE, NAWS = 255, 253, 254, 251, 252, 250, 240, 31
NEGOTIATE = bytes((IAC, DO, NAWS))
# No linefeeds or autowrap: even the bottom-right cell cannot scroll the screen.
ENTER = b'\x1b[?1049h\x1b[?25l\x1b[?7l\x1b[0m\x1b[40m\x1b[2J\x1b[H'
SYNC_BEGIN = '\x1b[?2026h'
SYNC_END = '\x1b[?2026l'
LEAVE = SYNC_END.encode() + b'\x1b[0m\x1b[?7h\x1b[?25h\x1b[?1049l'


def fitting_grid(size):
    cols, rows = size
    fitting = [(w, h) for w, h in SCALES if w <= cols and h <= rows]
    if fitting:
        return fitting[-1]
    scale = min(cols / 60, rows / 40)
    return max(1, min(cols, round(60 * scale))), max(1, min(rows, round(40 * scale)))


class WindowSize:
    """Incremental Telnet parser; subnegotiations have a fixed memory bound."""
    def __init__(self):
        self.state = 'data'
        self.command = None
        self.option = None
        self.data = bytearray()
        self.overflow = False
        self.size = (60, 24)  # Fits a conventional 80x24 terminal without NAWS.
        self.reported = None

    def feed(self, payload):
        reply = bytearray()
        for byte in payload:
            if self.state == 'data':
                if byte == IAC:
                    self.state = 'iac'
                elif ord('1') <= byte <= ord('4'):
                    # Raw TCP clients do not send NAWS. Allow explicit sizing
                    # without accepting commands or arbitrary escape sequences.
                    self.size = SCALES[byte - ord('1')]
            elif self.state == 'iac':
                if byte in (DO, DONT, WILL, WONT):
                    self.command = byte
                    self.state = 'option'
                elif byte == SB:
                    self.state = 'suboption'
                else:
                    self.state = 'data'
            elif self.state == 'option':
                if self.command == DO:
                    reply.extend((IAC, WONT, byte))
                elif self.command == WILL and byte != NAWS:
                    reply.extend((IAC, DONT, byte))
                self.state = 'data'
            elif self.state == 'suboption':
                self.option = byte
                self.data.clear()
                self.overflow = False
                self.state = 'sub'
            elif self.state == 'sub':
                if byte == IAC:
                    self.state = 'subiac'
                else:
                    self._append(byte)
            elif self.state == 'subiac':
                if byte == IAC:
                    self._append(byte)
                    self.state = 'sub'
                elif byte == SE:
                    if self.option == NAWS and not self.overflow and len(self.data) == 4:
                        cols = int.from_bytes(self.data[:2], 'big')
                        rows = int.from_bytes(self.data[2:], 'big')
                        if cols and rows:
                            self.reported = (cols, rows)
                            self.size = (min(cols, MAX_COLS), min(rows, MAX_ROWS))
                    self.state = 'data'
                else:
                    self.state = 'sub'
        return bytes(reply)

    def _append(self, byte):
        if len(self.data) < 64:
            self.data.append(byte)
        else:
            self.overflow = True


class TerminalPictures:
    """One current set of RGB grids and bounded shared encoded variants."""
    def __init__(self, variants=20):
        self.lock = threading.Lock()
        self.version = 0
        self.frame = None
        self.cache = OrderedDict()
        self.variants = variants
        self.grids = OrderedDict()
        self.requests = {}

    def requested_scales(self):
        """Bounded four-grid demand, expiring after disconnected viewers stop polling."""
        with self.lock:
            now = time.monotonic()
            self.requests = {grid: when for grid, when in self.requests.items()
                             if now - when < 2.0}
            return tuple(sorted(self.requests)) or ((60, 40),)

    def publish(self, frame):
        # FBO readback owns immutable bytes; retain only this current picture.
        with self.lock:
            self.frame = frame
            self.version += 1
            self.cache.clear()
            self.grids.clear()

    def picture(self, size, previous=None):
        cols, rows = size
        if not (1 <= cols <= MAX_COLS and 1 <= rows <= MAX_ROWS):
            raise ValueError('Invalid terminal dimensions')
        with self.lock:
            grid = fitting_grid(size)
            source_grid = grid if grid in SCALES else SCALES[0]
            self.requests[source_grid] = time.monotonic()
            identity = (self.version, size)
            if self.frame is None or identity == previous:
                return identity, None
            if isinstance(self.frame, dict) and source_grid not in self.frame:
                # Ask the producer for an independently sampled target; do not
                # substitute another size or mark this picture as delivered.
                return previous, None
            if size not in self.cache:
                fit_w, fit_h = grid
                left, top = (cols - fit_w) // 2, (rows - fit_h) // 2
                grid = (fit_w, fit_h)
                if grid not in self.grids:
                    # GL sampled each layer at this exact character grid.
                    rgb = (self.frame[source_grid]
                           if isinstance(self.frame, dict) else self.frame)
                    sampled = isinstance(self.frame, dict) and grid in self.frame
                    text = ascii_converter.to_ascii(rgb, width=fit_w, height=fit_h,
                                                    sampled=sampled)
                    self.grids[grid] = text.split('\r\n')
                    if len(self.grids) > self.variants:
                        self.grids.popitem(last=False)
                lines = self.grids[grid]
                # Supported terminals defer painting until the whole picture
                # arrives. Unknown private modes are ignored by other terminals.
                output = [SYNC_BEGIN, '\x1b[40m']
                for row in range(rows):
                    output.append(f'\x1b[{row + 1};1H')
                    output.append('\x1b[37m')
                    if top <= row < top + fit_h:
                        output.extend((' ' * left, lines[row - top],
                                       '\x1b[38;5;235m\x1b[40m', ' ' * (cols - left - fit_w)))
                    else:
                        output.append(' ' * cols)
                output.append('\x1b[0m')
                output.append(SYNC_END)
                payload = ''.join(output)
                self.cache[size] = payload.encode('utf-8')
                if len(self.cache) > self.variants:
                    self.cache.popitem(last=False)
            self.cache.move_to_end(size)
            return identity, self.cache[size]


pictures = TerminalPictures()
