"""Table-driven truecolour encoding with run-length colour changes.

Channel fragments are built once; the frame loop formats no decimal strings.
This avoids loading a JIT/LLVM stack just to encode terminal text.
"""

_RED = [f'\x1b[38;2;{value};' for value in range(256)]
_GREEN = [f'{value};' for value in range(256)]
_BLUE = [f'{value}m' for value in range(256)]


def encode_rgb_rows(rgb, chars):
    rows = []
    # Convert scalar containers once, avoiding NumPy scalar access per cell.
    for colors, glyphs in zip(rgb.tolist(), chars.tolist()):
        parts = []
        previous = None
        for color, char in zip(colors, glyphs):
            if color != previous:
                red, green, blue = color
                parts.extend((_RED[red], _GREEN[green], _BLUE[blue]))
                previous = color
            parts.append(char)
        rows.append(''.join(parts))
    return '\r\n'.join(rows)
