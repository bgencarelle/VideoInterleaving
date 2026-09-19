"""Three header bits describe display geometry; coding stays on the 80x96 grid."""
import math
import operator


ASPECT_PRESETS = ('5:6', '1:1', '4:3', '3:2', '16:9', '2.39:1', '3:4', '9:16')
ASPECT_RATIOS = (5/6, 1., 4/3, 3/2, 16/9, 2.39, 3/4, 9/16)
ASPECT_CHOICES = ('auto', 'native', *ASPECT_PRESETS)
FRAME_MASK = (1 << 29) - 1


def aspect_code(size, preset='auto'):
    """Auto selects the closest preset by proportional distortion, per frame.

    Use source dimensions AFTER rotation, BEFORE squeezing or subsampling.
    Three bits cannot describe arbitrary ratios exactly; never imply otherwise.
    """
    if preset == 'native':
        return 0
    if preset != 'auto':
        return ASPECT_PRESETS.index(preset)
    width, height = size
    if width <= 0 or height <= 0:
        raise ValueError('Source dimensions must be positive')
    ratio = width/height
    return min(range(8), key=lambda code: abs(math.log(ratio/ASPECT_RATIOS[code])))


def pack_absolute(absolute, code=0):
    absolute, code = operator.index(absolute), operator.index(code)
    if absolute < 0 or not 0 <= code < 8:
        raise ValueError('Need a nonnegative frame counter and aspect code 0..7')
    return (absolute & FRAME_MASK) | (code << 29)


def unpack_absolute(encoded):
    return encoded & FRAME_MASK, encoded >> 29
