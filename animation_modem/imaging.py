"""Image <-> value-vector conversion for the modem transport.

These were living in the check utility, which put them out of reach
of modem_display.py -- the runtime cannot import from utilities/. Nothing here
touches the wire format; it is the source-coding half, shared by the tool and
the runtime so both produce byte-identical value vectors.

The wire format is the DCT family: `color-dct` at 2880 values and
`color-wavelet` at 2880 values (same wire shape, wavelet transform),
and `fit_shapes` trims a profile into a smaller layout's budget.

v5: `hd-dwt` samples 80x96 (+ 40x48 chroma), 2-level CDF 9/7 DWT, and sends
the half-resolution pyramid (2880 values, cut at a band boundary) plus 800
repeat copies of its strongest values on diverse carriers -- 3680, the whole
WIRE_HD budget. Same detail as v3; the extra capacity buys robustness, not
resolution. See wavelet.Cdf97Coder.
"""
import numpy as np
from PIL import Image, ImageDraw, ImageOps

# Luma size and chroma size per profile, on the wire.
# v3/v4: 40x48 luma + 20x24 chroma = 2880 slots
# v5: the half-resolution pyramid it carries is 40x48-equivalent too; its
# repeat copies (wavelet.Cdf97Coder) fill the rest of the WIRE_HD budget.
PROFILES = {
    'color-dct': ((40, 48), (20, 24)),
    'color-wavelet': ((40, 48), (20, 24)),
    'hd-dwt': ((40, 48), (20, 24)),
}
# Sampling grid per profile -- the DECODE resolution, always the baked 80x96.
# SourceCoder truncates the grid's transform to the wire shape -- ONE transform,
# and the allocation table built against the grid's frequencies.
# v5: 2-level CDF 9/7 on the same 80x96 grid as v3/v4 (divisible by 8), keeping
# only the first ~3360 packed coefficients; the decoder reconstructs the soft
# full-resolution 80x96 picture (JPEG2000-style truncation).
PROFILE_GRIDS = {'color-dct': ((80, 96), (40, 48)),
                 'color-wavelet': ((80, 96), (40, 48)),
                 'hd-dwt': ((80, 96), (40, 48))}
DEFAULT_PROFILE = 'color-dct'
# v5 wire budget: WIRE_HD's Layout.capacity = header_capacity +
# image_symbols*data_bins*4 = 3680. wavelet.hd_dwt_coder() fills exactly this
# (2880 pyramid values + 800 copies); test_hd_dwt pins it to the layout.
HD_MONO_CAPACITY = 3680


def hd_dwt_shapes():
    """Plane shapes of the pyramid hd-dwt carries (before its repeat copies)."""
    return plane_shapes('hd-dwt')


def wire_profiles():
    """Profiles a transmitter can name. Every one holds its own wire code."""
    return tuple(PROFILES)


def source_size(profile=DEFAULT_PROFILE):
    """(width, height) of the pixels a bake must hold for this profile.

    For a truncating profile that is the SAMPLING grid, not the wire shape:
    color-dct puts 40x48 on the wire but needs 80x96 pixels behind it, and a
    bake at 40x48 gives it nothing to truncate.
    """
    return PROFILE_GRIDS.get(profile, PROFILES[profile])[0]


def plane_grids(profile=DEFAULT_PROFILE):
    """Sampling grids for a profile: its wire shapes unless it truncates."""
    if profile not in PROFILE_GRIDS:
        return plane_shapes(profile)
    size, chroma = PROFILE_GRIDS[profile]
    return [(size[1], size[0])] + ([(chroma[1], chroma[0])]*2 if chroma else [])


def plane_shapes(profile=DEFAULT_PROFILE):
    """Row/column shapes of the planes a profile transmits, luma first."""
    if profile not in PROFILES:
        raise ValueError(f'Unknown profile: {profile}')
    size, chroma = PROFILES[profile]
    return [(size[1], size[0])] + ([(chroma[1], chroma[0])]*2 if chroma else [])


def fit_shapes(shapes, capacity, divisor=2):
    """Shrink plane shapes to fit a layout's value budget, keeping the aspect.

    Widths/heights step by `divisor` so callers can keep planes subdividable:
    CDF 9/7 at `levels` needs every plane halvable cleanly at every level --
    and chroma is half-res, so the fit must step luma dims by
    2**levels * 2 = 8 for levels=2 (see hd_dwt_shapes).
    """
    if sum(int(np.prod(s)) for s in shapes) <= capacity:
        return list(shapes)
    rows, cols = shapes[0]
    aspect = cols/rows
    best = None
    for width in range(divisor, cols+1, divisor):
        for height in range(divisor, rows+1, divisor):
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


# 3x5 digits, small enough to read on a 40x48 transmitted image.
GLYPHS = dict(zip('0123456789AF/', [
    '111101101101111', '010110010010111', '111001111100111',
    '111001111001111', '101101111001001', '111100111001111',
    '111100111101111', '111001001001001', '111101111101111',
    '111101111001111', '010101111101101', '111100110100100',
    '001001010100100']))


def _text(im, text, y):
    strip = Image.new('RGB', (4*len(text), 7), 'black')
    draw = ImageDraw.Draw(strip)
    for i, character in enumerate(text):
        for p, bit in enumerate(GLYPHS[character]):
            if bit == '1':
                draw.point((i*4 + p % 3, 1 + p//3), fill='white')
    if strip.width > im.width:
        strip = strip.resize((im.width, 7), Image.Resampling.NEAREST)
    im.paste(strip, (0, y))


def burn_counters(image, absolute, index, count):
    """Debug overlay in transmitted pixels. Kept out of the transport so the
    wire format carries no notion of it."""
    out = image.convert('RGB').copy()
    _text(out, 'A' + str(absolute).zfill(6), 0)
    _text(out, 'F' + str(index) + '/' + str(count), 7)
    return out
