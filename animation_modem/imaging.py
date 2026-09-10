"""Image <-> value-vector conversion for the v2 transport.

These were living in utilities/modem_v2_check.py, which put them out of reach
of modem_display.py -- the runtime cannot import from utilities/. Nothing here
touches the wire format; it is the source-coding half, shared by the tool and
the runtime so both produce byte-identical value vectors.

The normal wire format is fixed: color, 40x48 luma with 20x24 chroma, 2880
values. `plane_shapes` still accepts the other profiles for offline
comparisons, and `fit_shapes` trims a profile into a smaller layout's budget.
"""
import numpy as np
from PIL import Image, ImageOps

# Luma size and chroma size per profile. Every profile is exactly 2880 values.
PROFILES = {
    'color': ((40, 48), (20, 24)),
    'detail': ((48, 56), (8, 12)),
    'mono': ((48, 60), None),
}
DEFAULT_PROFILE = 'color'


def plane_shapes(profile=DEFAULT_PROFILE):
    """Row/column shapes of the planes a profile transmits, luma first."""
    if profile not in PROFILES:
        raise ValueError(f'Unknown profile: {profile}')
    size, chroma = PROFILES[profile]
    return [(size[1], size[0])] + ([(chroma[1], chroma[0])]*2 if chroma else [])


def fit_shapes(shapes, capacity):
    """Shrink plane shapes to fit a layout's value budget, keeping the aspect.

    Only the narrow presets need this: tape-fast must not try to push 2880
    values through 1260 slots. The fixed wide format never triggers it.
    """
    if sum(int(np.prod(s)) for s in shapes) <= capacity:
        return list(shapes)
    rows, cols = shapes[0]
    aspect = cols/rows
    best = None
    for width in range(2, cols+1, 2):
        for height in range(2, rows+1, 2):
            trial = [(height, width)]
            if len(shapes) == 3:
                trial += [(height//2, width//2)]*2
            count = sum(int(np.prod(s)) for s in trial)
            if count <= capacity and min(min(s) for s in trial) > 0:
                score = count/(1 + abs(width/height - aspect))
                if best is None or score > best[0]:
                    best = (score, trial)
    if best is None:
        raise ValueError(f'No plane layout fits {capacity} values')
    return best[1]


def _shapes(spec):
    """Accept a plane-shape list or anything carrying one, such as a SourceCoder."""
    return list(getattr(spec, 'shapes', spec))


def image_values(image, shapes):
    """One finite value per source coefficient, in [-1, 1], luma plane first."""
    shapes = _shapes(shapes)
    rows, cols = shapes[0]
    padded = ImageOps.pad(image.convert('RGB'), (cols, rows),
                          method=Image.Resampling.LANCZOS, color='black')
    planes = padded.convert('YCbCr').split()
    return np.concatenate([
        np.asarray(plane.resize((shape[1], shape[0]), Image.Resampling.BOX)).ravel()
        for plane, shape in zip(planes, shapes)]).astype(float)/127.5 - 1


def values_image(values, shapes):
    """Rebuild an RGB image from a decoded value vector."""
    values = np.asarray(values, float)
    shapes = _shapes(shapes)
    planes, offset = [], 0
    for rows, cols in shapes:
        count = rows*cols
        pixels = np.uint8(np.clip(np.rint((values[offset:offset+count]+1)*127.5), 0, 255))
        planes.append(Image.fromarray(pixels.reshape(rows, cols)))
        offset += count
    if len(planes) == 1:
        return planes[0].convert('RGB')
    return Image.merge('YCbCr', (planes[0], *[
        plane.resize(planes[0].size, Image.Resampling.BILINEAR)
        for plane in planes[1:]])).convert('RGB')
