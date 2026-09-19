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
from scipy.ndimage import gaussian_filter
from PIL import Image, ImageDraw
from .aspect import aspect_code, ASPECT_RATIOS

# Luma size and chroma size per profile, on the wire.
# v3/v4: 40x48 luma + 20x24 chroma = 2880 slots
# v5: the half-resolution pyramid it carries is 40x48-equivalent too; its
# repeat copies (wavelet.Cdf97Coder) fill the rest of the WIRE_HD budget.
PROFILES = {
    'color-dct': ((40, 48), (20, 24)),
    'color-wavelet': ((40, 48), (20, 24)),
    'hd-dwt': ((40, 48), (20, 24)),
    # 25 fps tape profile: every DCT value is sent twice on diverse stereo
    # slots. 20x15 keeps exact 4:3 luma geometry; coarse chroma suppresses bands.
    'tape-80x60': ((20, 15), (6, 5)),
}
# V6 reuses header codes 0/1 only inside its distinct frame geometry. Keep
# these names out of wire_profiles(): that function is the pinned legacy
# four-code registry, not a list of every experimental source coder.
EXPERIMENTAL_PROFILES = {
    'v6-dct': ((40, 48), (20, 24)),
    'v6-wavelet': ((40, 48), (20, 24)),
}
# Sampling grid per profile -- the DECODE resolution, always the baked 80x96.
# SourceCoder truncates the grid's transform to the wire shape -- ONE transform,
# and the allocation table built against the grid's frequencies.
# v5: 2-level CDF 9/7 on the same 80x96 grid as v3/v4 (divisible by 8), keeping
# only the first ~3360 packed coefficients; the decoder reconstructs the soft
# full-resolution 80x96 picture (JPEG2000-style truncation).
PROFILE_GRIDS = {'color-dct': ((80, 96), (40, 48)),
                 'color-wavelet': ((80, 96), (40, 48)),
                  'hd-dwt': ((80, 96), (40, 48)),
                  'tape-80x60': ((80, 60), (40, 30))}
EXPERIMENTAL_GRIDS = {'v6-dct': ((80, 96), (40, 48)),
                      'v6-wavelet': ((80, 96), (40, 48))}
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
    profiles = {**PROFILES, **EXPERIMENTAL_PROFILES}
    return {**PROFILE_GRIDS, **EXPERIMENTAL_GRIDS}.get(profile,
                                                        profiles[profile])[0]


def plane_grids(profile=DEFAULT_PROFILE):
    """Sampling grids for a profile: its wire shapes unless it truncates."""
    grids = {**PROFILE_GRIDS, **EXPERIMENTAL_GRIDS}
    if profile not in grids:
        return plane_shapes(profile)
    size, chroma = grids[profile]
    return [(size[1], size[0])] + ([(chroma[1], chroma[0])]*2 if chroma else [])


def plane_shapes(profile=DEFAULT_PROFILE):
    """Row/column shapes of the planes a profile transmits, luma first."""
    profiles = {**PROFILES, **EXPERIMENTAL_PROFILES}
    if profile not in profiles:
        raise ValueError(f'Unknown profile: {profile}')
    size, chroma = profiles[profile]
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


ENCODE_FILTERS = {'box': Image.Resampling.BOX,
                  'nearest': Image.Resampling.NEAREST,
                  'lanczos': Image.Resampling.LANCZOS,
                  'bicubic': Image.Resampling.BICUBIC}


def prepare_image(image, preset='auto', encode_filter='lanczos'):
    """Squeeze the whole source to native geometry and retain its display preset."""
    code = aspect_code(image.size, preset)
    image = image.convert('RGB').resize((80, 96), ENCODE_FILTERS[encode_filter])
    image.info['aspect_code'] = code
    return image


def display_image(result, shapes=None, *, scale=1, bounds=None, smooth=False):
    """Shared live/export rendering, after native reconstruction.

    Height sets the export scale; width restores the signalled ratio. Native
    preset zero leaves the unscaled reconstruction pixel-identical. Window
    rendering uses nearest-neighbour to preserve decoded pixel values by default.
    Smooth interpolation is opt-in.
    """
    image = values_image(result.values, result.extra.get('shapes', shapes))
    ratio = result.extra.get('aspect', ASPECT_RATIOS[0])
    height = max(1, round(image.height * scale))
    width = max(1, round(height * ratio))
    if bounds is not None:
        width = min(bounds[0], max(1, round(bounds[1] * ratio)))
        height = min(bounds[1], max(1, round(width / ratio)))
    if (width, height) == image.size:
        return image
    return image.resize((width, height), Image.Resampling.LANCZOS if smooth
                        else Image.Resampling.NEAREST)


def image_values(image, shapes):
    """One finite value per source coefficient, in [-1, 1], luma plane first."""
    shapes = _shapes(shapes)
    rows, cols = shapes[0]
    sampled = image.convert('RGB').resize((cols, rows), Image.Resampling.LANCZOS)
    planes = sampled.convert('YCbCr').split()
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


def stabilize_chroma(values, previous, shapes, pilot_error=None, coverage=None):
    """Blend degraded chroma toward the previous decoded picture.

    Lossy audio codecs can leave a valid header and recognizable luma while
    quantising Cb/Cr into visible bands. Keep current luma exactly; only blend
    chroma when a same-shaped previous frame exists and confidence is low.
    """
    current = np.asarray(values)
    if shapes is None:
        return current
    shape_list = _shapes(shapes)
    if previous is None or current.shape != np.shape(previous) or len(shape_list) < 3:
        return current
    error = 0.0 if pilot_error is None else max(0.0, float(pilot_error))
    seen = 1.0 if coverage is None else np.clip(float(coverage), 0.0, 1.0)
    amount = max((error - .20) / .80, (0.85 - seen) / .85, 0.0)
    amount = float(np.clip(amount * .65, 0.0, .65))
    if amount <= 0.0:
        return current
    result = current.copy()
    old = np.asarray(previous)
    offset = shape_list[0][0] * shape_list[0][1]
    for rows, cols in shape_list[1:]:
        count = rows * cols
        blended = (
            current[offset:offset + count] * (1.0 - amount) +
            old[offset:offset + count] * amount)
        # Codec damage is spatially correlated at the low-resolution chroma
        # planes. Smooth only when confidence is poor; this removes isolated
        # false-color steps without touching luma or clean chroma.
        sigma = min(1.6, 0.35 + 1.8 * amount)
        result[offset:offset + count] = gaussian_filter(
            blended.reshape(rows, cols), sigma=sigma, mode='nearest').ravel()
        offset += count
    return result


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
