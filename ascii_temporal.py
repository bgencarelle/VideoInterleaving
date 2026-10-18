"""Experimental native run-coherent temporal colour planning, off by default.

The reference is the CURRENT accepted-heavy picture. The previous picture is
only a candidate source: it never becomes the reference for error bounds.
"""
import ctypes
import math
import re

import numpy as np
from ascii_color_sharing import ColorSharingEngine
from ascii_delta import TerminalDelta, read_picture


class TemporalRunPlanner:
    def __init__(self, tolerance=1.0, edge=6.0):
        if not math.isfinite(tolerance) or not 0 <= tolerance <= 100:
            raise ValueError('tolerance must be finite in 0..100')
        if not math.isfinite(edge) or not 0 <= edge <= 1000:
            raise ValueError('edge must be finite in 0..1000')
        self.tolerance, self.edge = float(tolerance), float(edge)
        self.engine = ColorSharingEngine()
        self.kernel = self.engine.lib.ascii_temporal_plan
        ptr, size, number = ctypes.c_void_p, ctypes.c_size_t, ctypes.c_double
        self.kernel.argtypes = [ptr]*7 + [size, size, number, number]
        self.kernel.restype = ctypes.c_int

    def plan(self, rgb, chars, previous, previous_chars):
        rgb = self.engine._rgb(rgb)
        previous = self.engine._rgb(previous)
        height, width = rgb.shape[:2]
        if height > 60 or width > 160 or previous.shape != rgb.shape:
            raise ValueError('Matched terminal grids up to 160x60 required')
        for value in (chars, previous_chars):
            if (not isinstance(value, np.ndarray) or value.dtype != np.dtype('U1') or
                    value.shape != rgb.shape[:2]):
                raise ValueError('Matched native U1 glyph grids required')
            cp = value.view(np.uint32)
            if np.any((cp < 32) | ((cp >= 127) & (cp < 160)) |
                      ((cp >= 0xd800) & (cp <= 0xdfff)) | (cp > 0x10ffff)):
                raise ValueError('Printable single-column glyphs required')
        chars = np.ascontiguousarray(chars)
        previous_chars = np.ascontiguousarray(previous_chars)
        lab = self.engine.perceptual(rgb)
        old_lab = self.engine.perceptual(previous)
        out = np.empty_like(rgb)
        status = self.kernel(rgb.ctypes.data, lab.ctypes.data, chars.ctypes.data,
            previous.ctypes.data, old_lab.ctypes.data, previous_chars.ctypes.data,
            out.ctypes.data, height, width, self.tolerance, self.edge)
        if status == 1:
            raise MemoryError('Temporal planner allocation failed')
        if status:
            raise ValueError('Invalid native planner dimensions')
        return out


_RGB_ESCAPE = re.compile(rb'\x1b\[38;2;(\d+);(\d+);(\d+)m\Z')


class TemporalTerminalEncoder:
    """Per-client opt-in adapter; choose by actual bytes, not just DAG estimates.

    The header, if supplied, is never colour-approximated. Unsupported colours
    fall back to the original complete picture. As with TerminalDelta, send
    failure must close the connection and a terminal clear must reset state.
    """
    def __init__(self, tolerance=1.0, header_rows=0):
        if not isinstance(header_rows, int) or not 0 <= header_rows <= 60:
            raise ValueError('header_rows must be an integer in 0..60')
        self.planner = TemporalRunPlanner(tolerance=tolerance)
        self.header_rows = header_rows
        self.delta = TerminalDelta()
        self.last_optimized = False

    def reset(self):
        self.delta.reset()
        self.last_optimized = False

    @staticmethod
    def _arrays(cells, width, height):
        colours, glyphs = [], []
        for glyph, tint in cells:
            glyphs.append(glyph.decode())
            if tint is None:
                colours.append((0, 0, 0))
            else:
                match = _RGB_ESCAPE.fullmatch(tint)
                if match is None:
                    raise ValueError('Temporal adapter requires truecolour cells')
                colours.append(tuple(int(v) for v in match.groups()))
        return (np.array(colours, dtype=np.uint8).reshape(height, width, 3),
                np.array(glyphs, dtype='U1').reshape(height, width))

    def encode(self, payload, size, optimize=True):
        self.last_optimized = False
        try:
            current = read_picture(payload, size)
        except (ValueError, UnicodeError):
            self.reset()
            return payload
        width, height = size
        offset = width * self.header_rows
        previous = self.delta.cells if self.delta.size == size else None
        if not optimize or previous is None or height <= self.header_rows:
            self.delta.cells, self.delta.size = current, size
            return payload
        try:
            rgb, chars = self._arrays(current[offset:], width, height-self.header_rows)
            old, old_chars = self._arrays(previous[offset:], width, height-self.header_rows)
        except ValueError:
            self.delta.cells, self.delta.size = current, size
            return payload
        chosen = self.planner.plan(rgb, chars, old, old_chars)
        selected = current[:offset] + [
            (glyph, None if tint is None else b'\x1b[38;2;%d;%d;%dm' % tuple(map(int, c)))
            for (glyph, tint), c in zip(current[offset:], chosen.reshape(-1, 3))]
        exact = TerminalDelta()
        exact.cells, exact.size = previous, size
        exact_payload = exact.encode_cells(current, size, payload)
        selected_payload = self.delta.encode_cells(selected, size)
        if len(exact_payload) <= len(selected_payload):
            self.delta.cells, self.delta.size = current, size
            return exact_payload
        self.last_optimized = True
        return selected_payload
