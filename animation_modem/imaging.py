"""Image <-> value-vector conversion for the v2 transport.

These were living in the check utility, which put them out of reach
of modem_display.py -- the runtime cannot import from utilities/. Nothing here
touches the wire format; it is the source-coding half, shared by the tool and
the runtime so both produce byte-identical value vectors.

The normal wire format is fixed: color, 40x48 luma with 20x24 chroma, 2880
values. `plane_shapes` still accepts the other profiles for offline
comparisons, and `fit_shapes` trims a profile into a smaller layout's budget.
"""
import numpy as np
from PIL import Image, ImageDraw, ImageOps

# Luma size and chroma size per profile. Every profile is exactly 2880 values.
PROFILES = {
    'color': ((40, 48), (20, 24)),
    # Measured on real baked frames: chroma at half luma resolution takes 33%
    # of the coefficient budget and carries 1.24% of the image energy (luma
    # 98.76%). Quartering it frees 720 slots -- 4 whole OFDM symbols -- and the
    # picture gets BETTER, because the surviving chroma coefficients each get
    # more power and stop blotching. 2160 coefficients, +20.8% frame rate, and
    # +0.9 to +5.4 dB depending on how noisy the channel is.
    #
    # Keep any chroma plane on the luma aspect ratio. image_values letterboxes
    # each plane independently, so an off-aspect chroma plane gets colour bars
    # and loses 2-4 dB -- which looks exactly like a chroma-resolution effect
    # and is not one.
    'color-lean': ((40, 48), (10, 12)),
    'detail': ((48, 56), (8, 12)),
    'mono': ((48, 60), None),
    # A BAKE geometry, not a wire profile. convert_to_modem_dct writes slabs at
    # this size; nothing can transmit it, for three separate reasons.
    #
    # It has no wire code. PROFILE_CODES holds four names and the header spends
    # exactly two bits on the field, both nibbles of the flags byte being folder
    # indices and top_bin needing its other six. A fifth name has nowhere to go
    # without growing the header.
    #
    # It does not fit. 11520 values against lean-v3's 2200, so fit_shapes runs
    # and lands on 34x42 luma -- SMALLER than color-lean's 40x48. Selecting it
    # would lower the resolution, silently but for build()'s warning.
    #
    # And the DCT would not rescue it. SourceCoder transmits every coefficient
    # of every plane -- count == sum(prod(shapes)) -- so the transform buys
    # decorrelation for default_allocation, not compression. Sampling 2x finer
    # and sending only the low-frequency corner (same slot count) was measured
    # against this baseline across five kinds of content at 12-30 dB SNR: +0.1
    # to +0.5 dB typical, +1.6 dB best case on a clean smooth gradient, and
    # NEGATIVE on portrait content. Box-downsampling already captures what
    # those coefficients can carry. Raising capacity does not help either --
    # lean-v3's 2200 slots to wide-v3's 3000 moved a 24 dB reconstruction by
    # +0.68 dB on graphics and -0.76 dB on portraits, because the same transmit
    # power spread over more coefficients leaves each one weaker.
    'color-dct': ((80, 96), (40, 48))
}
# Entries that exist as source geometry but cannot go on the wire. Kept as data
# rather than a comment so the code table test can assert the real contract:
# PROFILE_CODES is exactly PROFILES minus this set, and adding a profile
# without deciding which side it falls on fails that test.
BAKE_ONLY = ('color-dct',)
DEFAULT_PROFILE = 'color'


def wire_profiles():
    """Profiles a transmitter can actually name in the header."""
    return tuple(name for name in PROFILES if name not in BAKE_ONLY)


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
                          method=Image.Resampling.NEAREST, color='black')
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


def values_image_dct(values, shapes=None):
    """Reconstructs an 80x96 image from truncated 2D-DCT values and target shapes."""
    if shapes is None:
        shapes = ((48, 40), (24, 20), (24, 20))
    y_shape, cb_shape, cr_shape = shapes[0], shapes[1], shapes[2]

    values = np.asarray(values, dtype=float)
    y_count = y_shape[0] * y_shape[1]
    cb_count = cb_shape[0] * cb_shape[1]
    cr_count = cr_shape[0] * cr_shape[1]

    # Reconstruct Luma
    y_coeffs = values[:y_count].reshape(y_shape) * (127.5 * np.sqrt(80 * 96))
    y_full = np.zeros((96, 80), dtype=float)
    y_full[:y_shape[0], :y_shape[1]] = y_coeffs
    y_plane = np.uint8(np.clip(_idct2(y_full), 0, 255))

    # Reconstruct Chroma
    cb_coeffs = values[y_count:y_count + cb_count].reshape(cb_shape) * (127.5 * np.sqrt(40 * 48))
    cr_coeffs = values[y_count + cb_count:y_count + cb_count + cr_count].reshape(cr_shape) * (127.5 * np.sqrt(40 * 48))

    cb_full, cr_full = np.zeros((48, 40), dtype=float), np.zeros((48, 40), dtype=float)
    cb_full[:cb_shape[0], :cb_shape[1]] = cb_coeffs
    cr_full[:cr_shape[0], :cr_shape[1]] = cr_coeffs

    cb_sub = Image.fromarray(np.uint8(np.clip(_idct2(cb_full), 0, 255)))
    cr_sub = Image.fromarray(np.uint8(np.clip(_idct2(cr_full), 0, 255)))

    cb_plane = cb_sub.resize((80, 96), Image.Resampling.BILINEAR)
    cr_plane = cr_sub.resize((80, 96), Image.Resampling.BILINEAR)

    return Image.merge('YCbCr', (Image.fromarray(y_plane), cb_plane, cr_plane)).convert('RGB')

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


# Add at the bottom of imaging.py
import scipy.fftpack as fftpack


def _dct2(a): return fftpack.dct(fftpack.dct(a.T, norm='ortho').T, norm='ortho')


def _idct2(a): return fftpack.idct(fftpack.idct(a.T, norm='ortho').T, norm='ortho')


def image_values_dct(image):
    high_res = image.convert('RGB').resize((80, 96), Image.Resampling.NEAREST)
    planes = high_res.convert('YCbCr').split()
    y_dct = _dct2(np.asarray(planes[0], dtype=float))
    y_coeffs = y_dct[:48, :40].ravel() / (127.5 * np.sqrt(80 * 96))

    c_norms = []
    for p in planes[1:]:
        p_sub = p.resize((40, 48), Image.Resampling.LANCZOS)
        c_dct = _dct2(np.asarray(p_sub, dtype=float))
        c_norms.append(c_dct[:24, :20].ravel() / (127.5 * np.sqrt(40 * 48)))

    return np.concatenate([y_coeffs, c_norms[0], c_norms[1]])


def image_values_dct(image, shapes=None):
    """Encodes an image to 2D-DCT coefficients matching the target shapes."""
    if shapes is None:
        shapes = ((48, 40), (24, 20), (24, 20))
    y_shape, cb_shape, cr_shape = shapes[0], shapes[1], shapes[2]

    high_res = image.convert('RGB').resize((80, 96), Image.Resampling.LANCZOS)
    planes = high_res.convert('YCbCr').split()

    # 1. Luma Plane DCT
    y_dct = _dct2(np.asarray(planes[0], dtype=float))
    y_coeffs = y_dct[:y_shape[0], :y_shape[1]].ravel()
    y_norm = y_coeffs / (127.5 * np.sqrt(80 * 96))

    # 2. Chroma Planes DCT
    chroma_norms = []
    for plane, c_shape in zip(planes[1:], [cb_shape, cr_shape]):
        p_sub = plane.resize((40, 48), Image.Resampling.LANCZOS)
        c_dct = _dct2(np.asarray(p_sub, dtype=float))
        c_coeffs = c_dct[:c_shape[0], :c_shape[1]].ravel()
        chroma_norms.append(c_coeffs / (127.5 * np.sqrt(40 * 48)))

    return np.concatenate([y_norm, chroma_norms[0], chroma_norms[1]])