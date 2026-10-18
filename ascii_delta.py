"""Opt-in, per-connection lossless ANSI updates for complete terminal pictures.

Input is the bounded, black-background output of TerminalPictures, not arbitrary
terminal traffic. No quantisation, frame queue or client extension is involved.
The caller must discard this state after clearing/resizing the terminal. A failed
send must close the connection, rather than retry from a now-advanced state.
"""
import re

_CSI = re.compile(r'\x1b\[([0-9;?]*)([A-Za-z])')
_BEGIN = b'\x1b[?2026h'
_END = b'\x1b[?2026l'


def read_picture(payload, size):
    """Return visible (glyph, foreground escape) cells; reject unknown controls."""
    width, height = size
    cells = [None] * (width * height)
    text = payload.decode('utf-8')
    x = y = 0
    foreground = b'\x1b[37m'
    position = 0
    while position < len(text):
        if text[position] == '\x1b':
            match = _CSI.match(text, position)
            if match is None:
                raise ValueError('Unsupported terminal control')
            args, command = match.groups()
            position = match.end()
            if command == 'H':
                coordinates = [int(v) if v else 1 for v in args.split(';')]
                y = coordinates[0] - 1
                x = (coordinates[1] if len(coordinates) == 2 else 1) - 1
            elif command == 'm':
                values = [int(v) if v else 0 for v in args.split(';')]
                if values == [0]:
                    foreground = b'\x1b[37m'
                elif values == [40]:
                    pass
                elif (len(values) == 1 and
                      (30 <= values[0] <= 37 or 90 <= values[0] <= 97)):
                    foreground = match.group().encode()
                elif ((len(values) == 3 and values[:2] == [38, 5]) or
                      (len(values) == 5 and values[:2] == [38, 2])):
                    if not all(0 <= v <= 255 for v in values[2:]):
                        raise ValueError('Invalid terminal colour')
                    foreground = match.group().encode()
                else:
                    raise ValueError('Unsupported terminal attributes')
            elif (command, args) not in (('h', '?2026'), ('l', '?2026')):
                raise ValueError('Unsupported terminal control')
        else:
            glyph = text[position]
            position += 1
            # Application palettes contain only single-column characters. Reject
            # controls; the upstream renderer owns Unicode display-width policy.
            if ord(glyph) < 32 or ord(glyph) == 127:
                raise ValueError('Unsupported terminal glyph')
            if not (0 <= x < width and 0 <= y < height):
                raise ValueError('Picture outside terminal')
            # Foreground on a space is invisible on the fixed black background.
            cells[y * width + x] = (glyph.encode('utf-8'),
                                    None if glyph == ' ' else foreground)
            x += 1
    if any(cell is None for cell in cells):
        raise ValueError('Incomplete terminal picture')
    return cells


class TerminalDelta:
    """Keep only the last delivered screen, never a sequence of stale frames."""
    def __init__(self):
        self.reset()

    def reset(self):
        self.cells = None
        self.size = None

    def encode(self, payload, size):
        try:
            current = read_picture(payload, size)
        except (ValueError, UnicodeError):
            # Fail open to the original complete picture. Forget prior state so
            # the next supported picture is redrawn in full.
            self.reset()
            return payload
        return self.encode_cells(current, size, payload)

    def encode_cells(self, current, size, fallback=None):
        """Encode a validated complete screen; foregrounds on spaces are None."""
        previous = self.cells if self.size == size else None
        width, height = size
        parts = [_BEGIN, b'\x1b[0m\x1b[40m']
        colour = b'\x1b[37m'
        for row in range(height):
            start = row * width
            changed = [column for column in range(width)
                       if previous is None or
                       current[start + column] != previous[start + column]]
            cursor = None
            for column in changed:
                move = f'\x1b[{row + 1};{column + 1}H'.encode()
                if cursor is not None and cursor < column:
                    # Repainting a short unchanged gap can cost less than CUP.
                    gap = []
                    gap_colour = colour
                    for i in range(cursor, column):
                        glyph, tint = current[start + i]
                        if tint is not None and tint != gap_colour:
                            gap.append(tint)
                            gap_colour = tint
                        gap.append(glyph)
                    glyph, tint = current[start + column]
                    gap_cost = sum(map(len, gap))
                    # Account for the colour instruction needed after the gap.
                    if tint is not None:
                        gap_cost += 0 if tint == gap_colour else len(tint)
                    move_cost = len(move)
                    if tint is not None:
                        move_cost += 0 if tint == colour else len(tint)
                    if gap_cost <= move_cost:
                        parts.extend(gap)
                        colour = gap_colour
                    else:
                        parts.append(move)
                elif cursor != column:
                    parts.append(move)
                glyph, tint = current[start + column]
                if tint is not None and tint != colour:
                    parts.append(tint)
                    colour = tint
                parts.append(glyph)
                cursor = column + 1
        parts.append(_END)
        encoded = b''.join(parts)
        self.cells, self.size = current, size
        # Full original pictures remain a safe upper bound on transmitted bytes.
        return encoded if fallback is None or len(encoded) < len(fallback) else fallback
