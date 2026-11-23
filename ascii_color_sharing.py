"""Optional low-memory RGB colour sharing for terminal output.

Build the small C helper explicitly with utilities/build_ascii_color_sharing.py.
Importing this module never compiles code or imports Numba/LLVM. Application
grading/glyph selection precede this engine; only RGB tints are changed.
"""
import ctypes
import hashlib
import math
from pathlib import Path
import platform

import numpy as np

SOURCE = Path(__file__).resolve().parent / 'native' / 'ascii_color_sharing.c'


def library_path():
    identity = SOURCE.read_bytes() + (platform.system()+platform.machine()).encode()
    digest = hashlib.sha256(identity).hexdigest()[:20]
    return SOURCE.parent.parent / '_cache' / 'ascii_color_sharing' / f'{digest}.so'


class ColorSharingEngine:
    """Reentrant native DP/encoder with the prototype's exact NumPy transform."""
    def __init__(self, tolerance=1.0, edge=6.0, tint_gamma=.9, lightness_limit=None,
                 chroma_model='full'):
        for name, value in (('tolerance', tolerance), ('edge', edge)):
            if not math.isfinite(value) or value < 0 or value > 1000:
                raise ValueError(f'{name} must be finite and in 0..1000')
        if not math.isfinite(tint_gamma) or not 0 < tint_gamma <= 10:
            raise ValueError('tint_gamma must be finite and in (0, 10]')
        self.tolerance, self.edge = float(tolerance), float(edge)
        if chroma_model not in ('full','420'):
            raise ValueError("chroma_model must be 'full' or '420'")
        self.chroma_model = chroma_model
        if lightness_limit is not None and (not math.isfinite(lightness_limit) or
                                           not 0 <= lightness_limit <= 100):
            raise ValueError('lightness_limit must be None or finite in 0..100')
        self.lightness_limit = -1. if lightness_limit is None else float(lightness_limit)
        self.scale = np.array([1. if v == 0 else 255*(v/255)**tint_gamma/v
                               for v in range(256)], dtype=np.float64)
        path = library_path()
        if not path.is_file():
            raise RuntimeError('Build colour sharing first: .venv/bin/python '
                               'utilities/build_ascii_color_sharing.py')
        self.lib = ctypes.CDLL(str(path))
        ptr, size, number = ctypes.c_void_p, ctypes.c_size_t, ctypes.c_double
        self.lib.ascii_lift.argtypes = [ptr, ptr, ptr, size]
        self.lib.ascii_lift.restype = None
        self.lib.ascii_model420.argtypes = [ptr, ptr, size, size, number]
        self.lib.ascii_model420.restype = None
        self.lib.ascii_share.argtypes = [ptr, ptr, ptr, size, size, number, number, number, ptr]
        self.lib.ascii_share.restype = ctypes.c_int
        self.lib.ascii_encode.argtypes = [ptr, ptr, ptr, size, size]
        self.lib.ascii_encode.restype = size

    @staticmethod
    def _rgb(rgb):
        if not isinstance(rgb, np.ndarray) or rgb.dtype != np.uint8:
            raise ValueError('RGB must be a uint8 numpy array')
        if rgb.ndim != 3 or rgb.shape[2] != 3:
            raise ValueError('RGB must have shape (height, width, 3)')
        if not 1 <= rgb.shape[0] <= 4096 or not 1 <= rgb.shape[1] <= 4096:
            raise ValueError('RGB dimensions must be in 1..4096')
        return np.ascontiguousarray(rgb)

    def lift(self, rgb):
        rgb = self._rgb(rgb)
        out = np.empty_like(rgb)
        self.lib.ascii_lift(rgb.ctypes.data, self.scale.ctypes.data,
                            out.ctypes.data, rgb.shape[0]*rgb.shape[1])
        return out

    @staticmethod
    def perceptual(rgb):
        # Keep operation order identical to the accepted prototype, including
        # NumPy matmul/cbrt rounding and representative tie-breaking.
        x = np.asarray(rgb, dtype=float)/255
        linear = np.where(x <= .04045, x/12.92, ((x+.055)/1.055)**2.4)
        lms = linear @ np.array([[.4122214708,.5363325363,.0514459929],
                                 [.2119034982,.6806995451,.1073969566],
                                 [.0883024619,.2817188376,.6299787005]]).T
        lab = np.cbrt(lms) @ np.array([[.2104542553,.7936177850,-.0040720468],
                                      [1.9779984951,-2.4285922050,.4505937099],
                                      [.0259040371,.7827717662,-.8086757660]]).T
        return np.ascontiguousarray(100*lab)

    def share_colors(self, rgb):
        """Share already lifted/graded tints without applying the midtone curve."""
        lifted = self._rgb(rgb)
        lab = self.perceptual(lifted)
        model = self.comparison_model(lab)
        out = np.empty_like(lifted)
        height, width = lifted.shape[:2]
        result = self.lib.ascii_share(lifted.ctypes.data, model.ctypes.data,
            out.ctypes.data, height, width, self.tolerance, self.edge,
            self.lightness_limit, lab.ctypes.data)
        if result:
            raise MemoryError('Colour-sharing workspace allocation failed')
        return out

    def comparison_model(self, lab):
        """Coarsen comparison chroma only; full L and original RGB stay intact."""
        if not isinstance(lab,np.ndarray) or lab.dtype!=np.float64 or lab.ndim!=3 or lab.shape[2]!=3:
            raise ValueError('Comparison features must be float64 (height, width, 3)')
        if not 1<=lab.shape[0]<=4096 or not 1<=lab.shape[1]<=4096 or not np.isfinite(lab).all():
            raise ValueError('Comparison features must be finite with dimensions in 1..4096')
        if self.chroma_model=='full':return lab
        lab=np.ascontiguousarray(lab);out=np.empty_like(lab)
        self.lib.ascii_model420(lab.ctypes.data,out.ctypes.data,lab.shape[0],lab.shape[1],self.edge)
        return out

    def share(self, rgb):
        return self.share_colors(self.lift(rgb))

    def encode_colors(self, rgb, chars):
        rgb = self._rgb(rgb)
        if not isinstance(chars, np.ndarray) or chars.dtype != np.dtype('U1'):
            raise ValueError('Characters must be a native U1 numpy array')
        if chars.shape != rgb.shape[:2]:
            raise ValueError('Character and RGB dimensions must match')
        chars = np.ascontiguousarray(chars)
        codepoints = chars.view(np.uint32)
        if np.any(((codepoints >= 0xd800) & (codepoints <= 0xdfff)) |
                  (codepoints > 0x10ffff)):
            raise ValueError('Invalid Unicode code points are not UTF-8 characters')
        height, width = chars.shape
        out = np.empty(height*(width*24+2), dtype=np.uint8)
        length = self.lib.ascii_encode(rgb.ctypes.data, codepoints.ctypes.data,
                                       out.ctypes.data, height, width)
        return out[:length].tobytes().decode('utf-8')

    def encode(self, rgb, chars):
        return self.encode_colors(self.share(rgb), chars)
