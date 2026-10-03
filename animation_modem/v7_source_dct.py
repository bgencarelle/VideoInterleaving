"""Opt-in V7 source-resolution RGB to coder-grid DCT preparation.

This module is independent of the application and live capture stack. It keeps
the current sender interface: one normalized Y/Cb/Cr value per model-grid
sample. The caller remains responsible for aspect metadata and modem encoding.
"""
from functools import lru_cache

import numpy as np
from numba import njit, prange
from scipy.fft import dctn, idctn
from scipy.ndimage import gaussian_filter

from animation_modem.v7_kernels import KernelContext


NEUTRAL_CHROMA = 128.0/255.0
WEIGHTED_AGGREGATIONS = ('weighted-tent', 'weighted-cosine',
                         'weighted-gaussian')
AGGREGATIONS = ('off', 'area-box', *WEIGHTED_AGGREGATIONS)
BAND_PROFILES = ('off', 'mid-luma', 'perceptual-color')
SHARPEN_MODES = ('off', 'taper', 'usm')


def _rgb_float(rgb):
    """Return contiguous float64 gamma-encoded RGB in [0, 1]."""
    data = np.asarray(rgb)
    if data.ndim != 3 or data.shape[2] != 3:
        raise ValueError('source DCT expects an HxWx3 RGB frame')
    if data.dtype == np.uint8:
        result = data.astype(np.float64)/255.0
    else:
        result = np.asarray(data, dtype=np.float64)
        if not np.isfinite(result).all():
            raise ValueError('RGB source contains non-finite values')
        if result.size and result.min() >= 0 and result.max() > 1:
            if result.max() > 255:
                raise ValueError('float RGB values must be in [0, 1] or [0, 255]')
            result = result/255.0
    if not np.isfinite(result).all():
        raise ValueError('RGB source contains non-finite values')
    return np.ascontiguousarray(result)


@lru_cache(maxsize=32)
def _dct_basis(source_count, mode_count):
    """Return the leading orthonormal DCT-II basis vectors in float32."""
    source_count, mode_count = int(source_count), int(mode_count)
    if source_count <= 0 or not 0 < mode_count <= source_count:
        raise ValueError('partial DCT basis dimensions are invalid')
    samples = np.arange(source_count, dtype=np.float64)[:, None]
    modes = np.arange(mode_count, dtype=np.float64)[None, :]
    basis = np.cos(np.pi*(2*samples+1)*modes/(2*source_count))
    basis *= np.sqrt(2.0/source_count)
    basis[:, 0] /= np.sqrt(2.0)
    basis = np.ascontiguousarray(basis, dtype=np.float32)
    basis.setflags(write=False)
    return basis


@lru_cache(maxsize=32)
def _block_dct_basis(source_count, block_size, mode_count):
    """Integrate the native DCT basis over uniform source blocks."""
    source_count, block_size, mode_count = map(
        int, (source_count, block_size, mode_count))
    if source_count % block_size:
        raise ValueError('source dimension must be divisible by block size')
    reduced_count = source_count//block_size
    basis = _dct_basis(source_count, mode_count)
    integrated = basis.reshape(reduced_count, block_size, mode_count).sum(axis=1)
    integrated = np.ascontiguousarray(integrated, dtype=np.float32)
    integrated.setflags(write=False)
    return integrated


@njit(cache=True, parallel=True)
def _block_average_rgb(data, block_size, scale):
    """Average RGB source blocks into normalized float32 pixels."""
    height, width, _channels = data.shape
    reduced_height = height//block_size
    reduced_width = width//block_size
    result = np.empty((reduced_height, reduced_width, 3), dtype=np.float32)
    for y in prange(reduced_height):
        for x in range(reduced_width):
            red = np.float32(0.0)
            green = np.float32(0.0)
            blue = np.float32(0.0)
            for by in range(block_size):
                for bx in range(block_size):
                    source_y = y*block_size+by
                    source_x = x*block_size+bx
                    red += np.float32(data[source_y, source_x, 0])
                    green += np.float32(data[source_y, source_x, 1])
                    blue += np.float32(data[source_y, source_x, 2])
            result[y, x, 0] = red*scale
            result[y, x, 1] = green*scale
            result[y, x, 2] = blue*scale
    return result


def source_fold_dct_coefficients(rgb, grids, positions):
    """Project native RGB onto the compact DCT support required by a Fold.

    ``positions`` are flattened, concatenated coder-grid DCT indices. The
    projection evaluates their enclosing rectangles; the returned vector is
    zero outside those rectangles. Every requested position is computed from
    the native-resolution image with the orthonormal source-frame basis, then
    amplitude-normalized to the corresponding coder grid. This is the input
    expected by ``Fold500.encode_dct_coefficients``.

    The neutral YCbCr conversion is linear except for the tiny legal-range
    chroma clamp at fully saturated red/blue endpoints. That clamp is omitted
    here; its largest per-pixel correction is 1/(2*255) and it is not a
    transport or fold operation.
    """
    data = np.asarray(rgb)
    if data.ndim != 3 or data.shape[2] != 3:
        raise ValueError('fold source projection expects an HxWx3 RGB frame')
    height, width = data.shape[:2]
    if min(height, width) <= 0:
        raise ValueError('fold source projection expects a non-empty frame')
    grids = tuple(tuple(map(int, shape)) for shape in grids)
    if len(grids) != 3 or any(min(shape) <= 0 for shape in grids):
        raise ValueError('Fold 500 source projection expects three positive grids')
    if height < max(rows for rows, _ in grids) or \
            width < max(cols for _, cols in grids):
        raise ValueError('source frame is smaller than the V7 coder grid')

    positions = np.asarray(positions)
    total = sum(rows*cols for rows, cols in grids)
    if positions.ndim != 1 or not np.issubdtype(positions.dtype, np.integer) or \
            np.any(positions < 0) or np.any(positions >= total):
        raise ValueError('Fold source positions must be valid flattened grid indices')
    if not len(positions):
        raise ValueError('Fold source projection needs at least one position')

    offsets = np.cumsum([0] + [rows*cols for rows, cols in grids])
    extents = []
    for plane, (rows, cols) in enumerate(grids):
        local = positions[(positions >= offsets[plane]) &
                          (positions < offsets[plane+1])] - offsets[plane]
        if local.size:
            extents.append((int(local.max()//cols)+1,
                            int((local % cols).max())+1))
        else:
            extents.append((0, 0))
    luma_rows, luma_cols = extents[0]
    if min(luma_rows, luma_cols) <= 0:
        raise ValueError('Fold source positions must include luma coefficients')
    max_rows = max(rows for rows, _ in extents)
    max_cols = max(cols for _, cols in extents)

    if data.dtype == np.uint8:
        rgb01 = np.asarray(data, dtype=np.float32)
        rgb01 *= np.float32(1.0/255.0)
    else:
        rgb01 = np.asarray(data, dtype=np.float32)
        if not np.isfinite(rgb01).all() or rgb01.min() < 0 or rgb01.max() > 1:
            raise ValueError('fold source RGB must be finite and in [0, 1]')
        rgb01 = np.ascontiguousarray(rgb01)

    # Fuse the vertical transforms of R/G/B into one BLAS matrix multiply.
    # A second small multiply returns the leading source-grid DCT rectangles.
    basis_y = _dct_basis(height, max_rows)
    basis_x = _dct_basis(width, max_cols)
    vertical = basis_y.T @ rgb01.reshape(height, width*3)
    horizontal = (vertical.reshape(max_rows, width, 3)
                  .transpose(0, 2, 1).reshape(max_rows*3, width) @ basis_x)
    rgb_coefficients = (horizontal.reshape(max_rows, 3, max_cols)
                        .transpose(0, 2, 1))

    red, green, blue = (rgb_coefficients[..., channel]
                        for channel in range(3))
    values = (
        .299000*red + .587000*green + .114000*blue,
        -.168736*red - .331264*green + .500000*blue,
        .500000*red - .418688*green - .081312*blue,
    )
    # Convert [0, 1] YCbCr to the coder's [-1, 1] values directly in DCT
    # space. Constant offsets affect only the DC coefficient.
    result = np.zeros(total, dtype=np.float32)
    neutral_dc = np.float32(np.sqrt(height*width))
    for plane, ((grid_rows, grid_cols), (rows, cols), coeff) in enumerate(
            zip(grids, extents, values)):
        if not rows or not cols:
            continue
        coeff = np.asarray(coeff[:rows, :cols], dtype=np.float32)*2.0
        if plane == 0:
            coeff[0, 0] -= neutral_dc
        else:
            # Cb/Cr share the full-range 128/255 neutral offset.
            coeff[0, 0] += np.float32(neutral_dc/255.0)
        coeff *= np.float32(np.sqrt(
            (grid_rows*grid_cols)/(height*width)))
        result[offsets[plane]:offsets[plane+1]].reshape(
            grid_rows, grid_cols)[:rows, :cols] = coeff
    return result


class FoldBlockDCTProjector:
    """Reusable layout/basis plan for block-integrated Fold projection."""

    def __init__(self, source_shape, grids, positions, block_size=4):
        height, width = map(int, source_shape[:2])
        if min(height, width) <= 0:
            raise ValueError('fold source projection expects a non-empty frame')
        block_size = int(block_size)
        if block_size <= 0 or height % block_size or width % block_size:
            raise ValueError('source dimensions must be divisible by block size')

        grids = tuple(tuple(map(int, shape)) for shape in grids)
        if len(grids) != 3 or any(min(shape) <= 0 for shape in grids):
            raise ValueError(
                'Fold 500 source projection expects three positive grids')
        positions = np.asarray(positions)
        total = sum(rows*cols for rows, cols in grids)
        if positions.ndim != 1 or \
                not np.issubdtype(positions.dtype, np.integer) or \
                np.any(positions < 0) or np.any(positions >= total):
            raise ValueError(
                'Fold source positions must be valid flattened grid indices')
        if not len(positions):
            raise ValueError('Fold source projection needs at least one position')

        offsets = np.cumsum([0] + [rows*cols for rows, cols in grids])
        extents = []
        for plane, (rows, cols) in enumerate(grids):
            local = positions[(positions >= offsets[plane]) &
                              (positions < offsets[plane+1])] - offsets[plane]
            if local.size:
                extents.append((int(local.max()//cols)+1,
                                int((local % cols).max())+1))
            else:
                extents.append((0, 0))
        if not extents[0][0] or not extents[0][1]:
            raise ValueError(
                'Fold source positions must include luma coefficients')
        max_rows = max(rows for rows, _ in extents)
        max_cols = max(cols for _, cols in extents)
        reduced_height, reduced_width = height//block_size, width//block_size
        if reduced_height < max_rows or reduced_width < max_cols:
            raise ValueError(
                'block-reduced source is smaller than requested DCT support')

        self.source_shape = (height, width)
        self.grids = grids
        self.offsets = offsets
        self.extents = tuple(extents)
        self.total = total
        self.block_size = block_size
        self.reduced_shape = (reduced_height, reduced_width)
        self.max_rows = max_rows
        self.max_cols = max_cols
        self.basis_y = _block_dct_basis(height, block_size, max_rows)
        self.basis_x = _block_dct_basis(width, block_size, max_cols)

    def project(self, rgb):
        """Project one frame using this cached dimension/support plan."""
        data = np.asarray(rgb)
        if data.ndim != 3 or data.shape[2] != 3 or \
                data.shape[:2] != self.source_shape:
            raise ValueError('RGB frame does not match the projector dimensions')
        if data.dtype == np.uint8:
            scale = np.float32(1.0/(self.block_size*self.block_size*255.0))
        else:
            data = np.asarray(data, dtype=np.float32)
            if not np.isfinite(data).all() or data.min() < 0 or data.max() > 1:
                raise ValueError('fold source RGB must be finite and in [0, 1]')
            scale = np.float32(1.0/(self.block_size*self.block_size))
        reduced = _block_average_rgb(data, self.block_size, scale)

        reduced_height, reduced_width = self.reduced_shape
        vertical = self.basis_y.T @ reduced.reshape(
            reduced_height, reduced_width*3)
        horizontal = (
            vertical.reshape(self.max_rows, reduced_width, 3)
            .transpose(0, 2, 1).reshape(self.max_rows*3, reduced_width) @
            self.basis_x)
        rgb_coefficients = (horizontal.reshape(
            self.max_rows, 3, self.max_cols).transpose(0, 2, 1))

        red, green, blue = (rgb_coefficients[..., channel]
                            for channel in range(3))
        values = (
            .299000*red + .587000*green + .114000*blue,
            -.168736*red - .331264*green + .500000*blue,
            .500000*red - .418688*green - .081312*blue,
        )
        result = np.zeros(self.total, dtype=np.float32)
        height, width = self.source_shape
        neutral_dc = np.float32(np.sqrt(height*width))
        for plane, ((grid_rows, grid_cols), (rows, cols), coeff) in enumerate(
                zip(self.grids, self.extents, values)):
            if not rows or not cols:
                continue
            coeff = np.asarray(coeff[:rows, :cols], dtype=np.float32)*2.0
            if plane == 0:
                coeff[0, 0] -= neutral_dc
            else:
                coeff[0, 0] += np.float32(neutral_dc/255.0)
            coeff *= np.float32(np.sqrt(
                (grid_rows*grid_cols)/(height*width)))
            result[self.offsets[plane]:self.offsets[plane+1]].reshape(
                grid_rows, grid_cols)[:rows, :cols] = coeff
        return result


def source_fold_block_dct_coefficients(rgb, grids, positions,
                                       block_size=4):
    """Approximate native Fold coefficients from block-averaged RGB.

    The compiled block-average pass feeds a reusable
    :class:`FoldBlockDCTProjector`, which integrates the original
    full-resolution DCT-II basis over each block. Retained modes therefore
    stay at native-source frequencies; they are not same-index modes of a
    resized image. This is exact for block-constant input and approximates
    native coefficients for general images. High-frequency information within
    each block is lost. Dimensions must be divisible by ``block_size``.

    Streaming callers should construct one projector per source geometry and
    Fold support, then call ``project`` for each frame so validation, extent
    discovery, and basis setup are amortized.
    """
    data = np.asarray(rgb)
    projector = FoldBlockDCTProjector(
        data.shape, grids, positions, block_size=block_size)
    return projector.project(data)


@lru_cache(maxsize=32)
def _box_axis_footprints(source_count, target_count):
    """Return exact source-pixel overlap weights for each target pixel."""
    source_count, target_count = int(source_count), int(target_count)
    if source_count <= 0 or target_count <= 0:
        raise ValueError('area-resize dimensions must be positive')
    if target_count > source_count:
        raise ValueError('area-box resampling does not enlarge an axis')
    scale = source_count/target_count
    width = int(np.ceil(scale))+1
    indexes = np.zeros((target_count, width), dtype=np.int32)
    weights = np.zeros((target_count, width), dtype=np.float64)
    counts = np.zeros(target_count, dtype=np.int32)
    for target in range(target_count):
        left = target*scale
        right = (target+1)*scale
        for source in range(int(np.floor(left)), min(
                source_count, int(np.ceil(right)))):
            overlap = min(right, source+1.0)-max(left, float(source))
            if overlap > 0:
                slot = counts[target]
                indexes[target, slot] = source
                weights[target, slot] = overlap
                counts[target] += 1
    for array in (indexes, weights, counts):
        array.setflags(write=False)
    return indexes, weights, counts


@njit(nogil=True, cache=True)
def _area_box_kernel(plane, y_indexes, y_weights, y_counts,
                     x_indexes, x_weights, x_counts):
    out = np.empty((len(y_counts), len(x_counts)), np.float64)
    for y in range(len(y_counts)):
        for x in range(len(x_counts)):
            total = 0.0
            area = 0.0
            for ky in range(y_counts[y]):
                sy = y_indexes[y, ky]
                wy = y_weights[y, ky]
                for kx in range(x_counts[x]):
                    sx = x_indexes[x, kx]
                    weight = wy*x_weights[x, kx]
                    total += weight*plane[sy, sx]
                    area += weight
            out[y, x] = total/area
    return out


def _area_box_resample(plane, target_shape):
    """Apply separable exact pixel-area averaging to a float plane."""
    plane = np.asarray(plane, dtype=np.float64)
    if plane.ndim != 2 or min(plane.shape) <= 0:
        raise ValueError('area-box resampling expects a non-empty 2-D plane')
    target_rows, target_cols = map(int, target_shape)
    y_indexes, y_weights, y_counts = _box_axis_footprints(
        plane.shape[0], target_rows)
    x_indexes, x_weights, x_counts = _box_axis_footprints(
        plane.shape[1], target_cols)
    return _area_box_kernel(plane, y_indexes, y_weights, y_counts,
                            x_indexes, x_weights, x_counts)


def _axis_frequency_gain(target_count, aggregation):
    """Return a same-mode taper on the target's normalized DCT axis.

    Each gain multiplies one whole-frame DCT coefficient. The operation never
    averages signed coefficients or rescales their mode indices by the
    source/target raster ratio.
    """
    normalized_mode = np.arange(target_count, dtype=np.float64)/target_count
    if aggregation == 'weighted-tent':
        return 1.0-normalized_mode
    if aggregation == 'weighted-cosine':
        return .5+.5*np.cos(np.pi*normalized_mode)
    if aggregation == 'weighted-gaussian':
        return np.exp(-.5*(normalized_mode/.5)**2)
    raise ValueError(f'aggregation {aggregation!r} is not a weighted reducer')


def _aggregate(coefficients, aggregation, target_shape):
    """Select same-index modes, optionally applying separable gain tapers."""
    if aggregation == 'off':
        return coefficients[:target_shape[0], :target_shape[1]]
    if aggregation not in WEIGHTED_AGGREGATIONS:
        raise ValueError(f'aggregation {aggregation!r} is not a weighted reducer')
    target_rows, target_cols = target_shape
    reduced = coefficients[:target_rows, :target_cols].copy()
    row_gain = _axis_frequency_gain(target_rows, aggregation)
    col_gain = _axis_frequency_gain(target_cols, aggregation)
    reduced *= row_gain[:, None]*col_gain[None, :]
    return reduced


def _frequency_weights(shape, plane, profile, display_size,
                       viewing_distance_mm, display_dpi,
                       viewport_aspect=None):
    rows, cols = shape
    u = np.arange(rows, dtype=np.float64)[:, None]
    v = np.arange(cols, dtype=np.float64)[None, :]
    if profile == 'mid-luma':
        if plane != 0:
            return np.ones(shape, dtype=np.float64)
        radius = np.hypot(u/max(rows-1, 1), v/max(cols-1, 1))/np.sqrt(2.0)
        band = np.exp(-.5*((radius-.42)/.15)**2)
        weights = 1.0 + .20*band
    elif profile == 'perceptual-color':
        width, height = map(float, display_size)
        if min(width, height, viewing_distance_mm, display_dpi) <= 0:
            raise ValueError('display geometry must be positive')
        aspect = (cols/rows if viewport_aspect is None
                  else float(viewport_aspect))
        view_width = min(width, height*aspect)
        view_height = view_width/aspect
        mm_per_pixel = 25.4/display_dpi
        angle_x = 2*np.degrees(np.arctan(
            view_width*mm_per_pixel/(2*viewing_distance_mm)))
        angle_y = 2*np.degrees(np.arctan(
            view_height*mm_per_pixel/(2*viewing_distance_mm)))
        cpd_x = (v/2.0)/max(angle_x, 1e-9)
        cpd_y = (u/2.0)/max(angle_y, 1e-9)
        frequency = np.hypot(cpd_x, cpd_y)
        csf = 2.6*(.0192+.114*frequency)*np.exp(-(.114*frequency)**1.1)
        csf /= max(float(np.max(csf)), 1e-12)
        weights = (.78+.42*csf if plane == 0 else .70+.30*csf)
    else:
        return np.ones(shape, dtype=np.float64)
    weights[0, 0] = 1.0
    return weights


@lru_cache(maxsize=64)
def _dct_matrix(source_count, mode_count):
    """Leading orthonormal DCT-II basis vectors (source x modes), float64."""
    samples = np.arange(int(source_count), dtype=np.float64)[:, None]
    modes = np.arange(int(mode_count), dtype=np.float64)[None, :]
    basis = np.cos(np.pi*(2*samples+1)*modes/(2*int(source_count)))
    basis *= np.sqrt(2.0/int(source_count))
    basis[:, 0] /= np.sqrt(2.0)
    basis.setflags(write=False)
    return basis


def _axis_blocks(size, block):
    """Widths of the pre-shrink blocks along one axis: whole ``block``-pixel
    blocks, then the remainder (if any) as a last, narrower block."""
    size, block = int(size), int(block)
    widths = np.full(-(-size//block), block, dtype=np.int64)
    if size % block:
        widths[-1] = size % block
    return widths


# 2:1 decimation filter between the block means and the transform: six
# symmetric taps (a, b, c, c, b, a), designed so what would alias into the
# sent band (above 0.8 of the input's Nyquist) is 74 dB down. Its passband
# droop (to 0.79 at the top of the sent band) is divided out in the analysis.
DECIMATE_TAPS = (0.03537128823198011, 0.15996811301904001, 0.3046605987489799)


def _decimate_gain(omega):
    """The decimation filter's gain at ``omega`` radians per input sample."""
    a, b, c = DECIMATE_TAPS
    return 2*(a*np.cos(2.5*omega)+b*np.cos(1.5*omega)+c*np.cos(.5*omega))


@lru_cache(maxsize=64)
def _axis_analysis(size, block, modes, grid, decimated=0):
    """(samples x modes) analysis of a pre-shrunk plane along one
    ``size``-pixel axis.

    Column k estimates the axis's orthonormal DCT coefficient k, scaled to a
    ``grid``-sample axis (the coder's ``sqrt(grid/size)``). The plane's
    samples are means of ``block``-pixel blocks (a remainder is one narrower
    last block, so nothing is cropped), then ``decimated`` passes (0, 1 or 2)
    of the 2:1 decimation filter. A block of w pixels centred at c holds the mean of
    cos(t*(n+.5)), t = pi*k/size, which is cos(t*c)*sin(w*t/2)/(w*sin(t/2));
    the filter (edges reflected, as the cosines are) multiplies that by its
    gain at t*block (a second pass: at 2*t*block). Each sample's weight is its width and both factors are
    divided out, so in-band amplitudes equal the full-resolution transform's.
    Detail finer than the block means still aliases; that part cannot be
    undone here.
    """
    size, block = int(size), int(block)
    widths = _axis_blocks(size, block).astype(np.float64)
    theta = np.pi*np.arange(1, int(modes), dtype=np.float64)/size
    half = theta[None, :]/2
    decimated = int(decimated)
    if decimated:
        if size % (block << decimated):
            raise ValueError('decimation needs whole block groups')
        droop = np.sin(block*half)/(block*np.sin(half))
        for stage in range(decimated):
            droop = droop*_decimate_gain(theta*(block << stage))[None, :]
        widths = np.full(len(widths) >> decimated, float(block << decimated))
    else:
        droop = np.sin(widths[:, None]*half)/(widths[:, None]*np.sin(half))
    centres = np.cumsum(widths)-widths/2
    matrix = np.empty((len(widths), int(modes)), dtype=np.float64)
    matrix[:, 0] = widths*np.sqrt(float(grid))/size
    matrix[:, 1:] = (widths[:, None]*np.cos(centres[:, None]*theta[None, :]) /
                     droop*np.sqrt(2.0*float(grid))/size)
    matrix.setflags(write=False)
    return matrix


@njit(cache=True, fastmath=True, nogil=True)
def _decimate_axis0(plane, a, b, c):
    """2:1 along axis 0 with taps (a, b, c, c, b, a), edges reflected."""
    rows, cols = plane.shape[0]//2, plane.shape[1]
    last = plane.shape[0]-1
    out = np.empty((rows, cols))
    for i in range(rows):
        i0, i1 = 2*i-2, 2*i-1
        i4, i5 = 2*i+2, 2*i+3
        if i0 < 0:
            i0 = -1-i0
        if i1 < 0:
            i1 = -1-i1
        if i4 > last:
            i4 = 2*last+1-i4
        if i5 > last:
            i5 = 2*last+1-i5
        for j in range(cols):
            out[i, j] = (a*(plane[i0, j]+plane[i5, j]) +
                         b*(plane[i1, j]+plane[i4, j]) +
                         c*(plane[2*i, j]+plane[2*i+1, j]))
    return out


@njit(cache=True, fastmath=True, nogil=True)
def _decimate_axis1(plane, a, b, c):
    """2:1 along axis 1 with taps (a, b, c, c, b, a), edges reflected."""
    rows, cols = plane.shape[0], plane.shape[1]//2
    last = plane.shape[1]-1
    out = np.empty((rows, cols))
    for i in range(rows):
        row = plane[i]
        for j in range(1, cols-1):                 # no edge: vectorises
            out[i, j] = (a*(row[2*j-2]+row[2*j+3]) +
                         b*(row[2*j-1]+row[2*j+2]) +
                         c*(row[2*j]+row[2*j+1]))
        for j in (0, cols-1):
            j0, j1 = 2*j-2, 2*j-1
            j4, j5 = 2*j+2, 2*j+3
            if j0 < 0:
                j0 = -1-j0
            if j1 < 0:
                j1 = -1-j1
            if j4 > last:
                j4 = 2*last+1-j4
            if j5 > last:
                j5 = 2*last+1-j5
            out[i, j] = (a*(row[j0]+row[j5])+b*(row[j1]+row[j4]) +
                         c*(row[2*j]+row[2*j+1]))
    return out


@njit(cache=True, fastmath=True, nogil=True)
def _ycbcr_kernel(red, green, blue, chroma_gain):
    """Y, Cb, Cr planes (JFIF weights) of R, G, B planes in [0, 1]."""
    rows, cols = red.shape
    y = np.empty((rows, cols))
    cb = np.empty((rows, cols))
    cr = np.empty((rows, cols))
    for i in range(rows):
        for j in range(cols):
            r, g, b = red[i, j], green[i, j], blue[i, j]
            y[i, j] = .299000*r+.587000*g+.114000*b
            u = -.168736*r-.331264*g+.5*b
            v = .5*r-.418688*g-.081312*b
            if chroma_gain != 1.0:
                u = min(max(chroma_gain*u, -.5), .5)
                v = min(max(chroma_gain*v, -.5), .5)
            cb[i, j] = .5+u
            cr[i, j] = .5+v
    return y, cb, cr


def _decimate(plane, axis_y, axis_x):
    """The 2:1 decimation filter on the flagged axes."""
    plane = np.ascontiguousarray(plane)
    if axis_x:
        plane = _decimate_axis1(plane, *DECIMATE_TAPS)
    if axis_y:
        plane = _decimate_axis0(plane, *DECIMATE_TAPS)
    return plane


@lru_cache(maxsize=32)
def _direct_plan(height, width, rows, cols, block_y=1, block_x=1,
                 decimated_y=0, decimated_x=0):
    """Cached matrices for DCT truncation of an h×w source to an r×c grid.

    The plane is the source's block means (``block_y``×``block_x`` pixels, 1 =
    the source itself), 2:1 decimated ``decimated_y``/``decimated_x`` times.
    ``C = analysis_y @ plane @ analysis_x`` is the
    source's leading r×c orthonormal DCT, amplitude-normalized to the grid
    (see _axis_analysis). ``synthesis_y @ C @ synthesis_x`` is the grid's
    inverse DCT, and ``left @ plane @ right`` is both steps folded together.
    """
    analysis_y = np.ascontiguousarray(
        _axis_analysis(height, block_y, rows, rows, decimated_y).T)
    analysis_x = np.ascontiguousarray(
        _axis_analysis(width, block_x, cols, cols, decimated_x))
    synthesis_y = np.ascontiguousarray(_dct_matrix(rows, rows))
    synthesis_x = np.ascontiguousarray(_dct_matrix(cols, cols).T)
    left = np.ascontiguousarray(synthesis_y @ analysis_y)
    right = np.ascontiguousarray(analysis_x @ synthesis_x)
    plan = (analysis_y, analysis_x, synthesis_y, synthesis_x, left, right)
    for array in plan:
        array.setflags(write=False)
    return plan


@lru_cache(maxsize=16)
def _tone_lut(brightness, gamma):
    """uint8 code -> toned [0, 1] value: brightness, clip, then gamma."""
    lut = np.clip(np.arange(256, dtype=np.float64)/255.0*brightness, 0.0, 1.0)
    if gamma != 1.0:
        lut = lut**(1.0/gamma)
    lut.setflags(write=False)
    return lut


@lru_cache(maxsize=16)
def _taper_gain(rows, cols, sent_rows, sent_cols, strength):
    """1 + s*b(f) inside the sent corner, 1 elsewhere (see the spec)."""
    u = np.arange(sent_rows, dtype=np.float64)[:, None]/sent_rows
    v = np.arange(sent_cols, dtype=np.float64)[None, :]/sent_cols
    radius = np.hypot(u, v)
    boost = np.where(radius < 1, (27.0/4.0)*radius**2*(1.0-radius), 0.0)
    gain = np.ones((rows, cols), dtype=np.float64)
    gain[:sent_rows, :sent_cols] += strength*boost
    gain.setflags(write=False)
    return gain


# Block sums are accumulated as integers (table values times 2**44): integer
# adds have no multi-cycle dependency chain, which is what bounds a float
# running sum. Rounding is 3e-14 per table value; a block may hold 4,096
# pixels before the sum could overflow.
FIXED_ONE = float(2**44)


@lru_cache(maxsize=16)
def _tone_lut_fixed(brightness, gamma):
    table = np.rint(_tone_lut(brightness, gamma)*FIXED_ONE).astype(np.int64)
    table.setflags(write=False)
    return table


@njit(cache=True, nogil=True)
def _toned_block_means_u8(data, lut, block_y, block_x, out_rows, out_cols):
    """Tone uint8 RGB through the fixed-point ``lut`` and average
    block_y×block_x blocks (the last block on each axis is the remainder, if
    any).

    One pass over the frame; returns contiguous R, G, B mean planes.
    """
    height, width = data.shape[0], data.shape[1]
    full_cols = width//block_x
    tail = width-full_cols*block_x
    sums = np.zeros((3, out_cols), dtype=np.int64)
    result = np.empty((3, out_rows, out_cols), dtype=np.float64)
    for by in range(out_rows):
        y0 = by*block_y
        count_y = min(block_y, height-y0)
        sums[:, :] = 0
        for dy in range(count_y):
            y = y0+dy
            for bx in range(full_cols):
                red = 0
                green = 0
                blue = 0
                x0 = bx*block_x
                for dx in range(block_x):
                    red += lut[data[y, x0+dx, 0]]
                    green += lut[data[y, x0+dx, 1]]
                    blue += lut[data[y, x0+dx, 2]]
                sums[0, bx] += red
                sums[1, bx] += green
                sums[2, bx] += blue
            for x in range(full_cols*block_x, width):
                sums[0, full_cols] += lut[data[y, x, 0]]
                sums[1, full_cols] += lut[data[y, x, 1]]
                sums[2, full_cols] += lut[data[y, x, 2]]
        scale = 1.0/(count_y*block_x*FIXED_ONE)
        for channel in range(3):
            for bx in range(full_cols):
                result[channel, by, bx] = sums[channel, bx]*scale
            if tail:
                result[channel, by, full_cols] = (
                    sums[channel, full_cols]/(count_y*tail*FIXED_ONE))
    return result


@njit(cache=True, nogil=True)
def _block_means_u8(data, gain, block_y, block_x, out_rows, out_cols):
    """_toned_block_means_u8 for a tone that is a plain gain (no clip, no
    gamma): the codes themselves are summed, which needs no table."""
    height, width = data.shape[0], data.shape[1]
    full_cols = width//block_x
    tail = width-full_cols*block_x
    sums = np.zeros((out_cols, 3), dtype=np.int64)
    result = np.empty((3, out_rows, out_cols), dtype=np.float64)
    for by in range(out_rows):
        y0 = by*block_y
        count_y = min(block_y, height-y0)
        sums[:, :] = 0
        for dy in range(count_y):
            row = data[y0+dy]
            for bx in range(full_cols):
                red = 0
                green = 0
                blue = 0
                x0 = bx*block_x
                for dx in range(block_x):
                    red += row[x0+dx, 0]
                    green += row[x0+dx, 1]
                    blue += row[x0+dx, 2]
                sums[bx, 0] += red
                sums[bx, 1] += green
                sums[bx, 2] += blue
            for x in range(full_cols*block_x, width):
                for channel in range(3):
                    sums[full_cols, channel] += row[x, channel]
        scale = gain/(count_y*block_x)
        for channel in range(3):
            for bx in range(full_cols):
                result[channel, by, bx] = sums[bx, channel]*scale
            if tail:
                result[channel, by, full_cols] = (
                    sums[full_cols, channel]*gain/(count_y*tail))
    return result


@lru_cache(maxsize=16)
def _luminance_luts(brightness, gamma):
    """uint8 code -> weighted linear light of the toned value, per channel."""
    linear = _srgb_to_linear(_tone_lut(brightness, gamma))
    luts = np.ascontiguousarray(LUMINANCE_WEIGHTS[:, None]*linear[None, :])
    luts.setflags(write=False)
    return luts


@njit(cache=True, nogil=True)
def _toned_block_means_luminance_u8(data, lut, linear, block_y, block_x,
                                    out_rows, out_cols):
    """_toned_block_means_u8 plus each block's mean linear luminance.

    The luminance is averaged in linear light per pixel. Linearising the
    block's mean code instead reads any block with detail inside it too dark
    (a one-pixel black/white pattern: 0.21 instead of 0.50).
    """
    height, width = data.shape[0], data.shape[1]
    full_cols = width//block_x
    tail = width-full_cols*block_x
    result = np.zeros((3, out_rows, out_cols), dtype=np.float64)
    luminance = np.zeros((out_rows, out_cols), dtype=np.float64)
    for by in range(out_rows):
        y0 = by*block_y
        count_y = min(block_y, height-y0)
        for dy in range(count_y):
            y = y0+dy
            for bx in range(out_cols):
                red = 0.0
                green = 0.0
                blue = 0.0
                light = 0.0
                x0 = bx*block_x
                for dx in range(block_x if bx < full_cols else tail):
                    r = data[y, x0+dx, 0]
                    g = data[y, x0+dx, 1]
                    b = data[y, x0+dx, 2]
                    red += lut[r]
                    green += lut[g]
                    blue += lut[b]
                    light += linear[0, r]+linear[1, g]+linear[2, b]
                result[0, by, bx] += red
                result[1, by, bx] += green
                result[2, by, bx] += blue
                luminance[by, bx] += light
        for bx in range(out_cols):
            scale = 1.0/(count_y*(block_x if bx < full_cols else tail))
            luminance[by, bx] *= scale
            for channel in range(3):
                result[channel, by, bx] *= scale
    return result, luminance


@njit(cache=True, nogil=True)
def _block_means_f64(data, block_y, block_x, out_rows, out_cols):
    height, width = data.shape[0], data.shape[1]
    full_cols = width//block_x
    tail = width-full_cols*block_x
    result = np.zeros((3, out_rows, out_cols), dtype=np.float64)
    for by in range(out_rows):
        y0 = by*block_y
        count_y = min(block_y, height-y0)
        for dy in range(count_y):
            y = y0+dy
            for bx in range(out_cols):
                x0 = bx*block_x
                for dx in range(block_x if bx < full_cols else tail):
                    for channel in range(3):
                        result[channel, by, bx] += data[y, x0+dx, channel]
        for bx in range(out_cols):
            scale = 1.0/(count_y*(block_x if bx < full_cols else tail))
            for channel in range(3):
                result[channel, by, bx] *= scale
    return result


@njit(cache=True, fastmath=True, nogil=True)
def _separable(left, plane, right):
    """``left @ plane @ right`` in one thread, without BLAS.

    The products are small (a few million multiply-adds per frame). OpenBLAS
    and a parallel Numba pool are faster per call, but both leave worker
    threads spinning after it: at the sender's 12 frames/s that cost 3-4x the
    useful CPU (11-14 ms per frame instead of 3-4 ms, 2-core host), taken
    from capture, audio and a receiver on the same machine.
    """
    height, width = plane.shape
    rows = left.shape[0]
    cols = right.shape[1]
    partial = np.zeros((height, cols))
    for i in range(height):
        for k in range(width):
            value = plane[i, k]
            for j in range(cols):
                partial[i, j] += value*right[k, j]
    out = np.zeros((rows, cols))
    for i in range(rows):
        for k in range(height):
            value = left[i, k]
            for j in range(cols):
                out[i, j] += value*partial[k, j]
    return out


PRESHRINK_FACTOR = 4


def _direct_planes(rgb, grids, shapes, brightness, gamma, sharpen,
                   sharpen_strength, clarity, chroma_gain, luminance=False):
    """Validated, pre-shrunk and pixel-enhanced Y[/Cb/Cr] planes in [0, 1].

    With ``luminance`` the result also carries the source's linear luminance
    on the luma grid (see source_luminance), else None.
    """
    if sharpen not in SHARPEN_MODES:
        raise ValueError(f'unknown DCT sharpen mode {sharpen!r}')
    brightness, gamma = float(brightness), float(gamma)
    sharpen_strength, clarity = float(sharpen_strength), float(clarity)
    chroma_gain = float(chroma_gain)
    if not np.isfinite(brightness) or brightness <= 0:
        raise ValueError('brightness must be finite and positive')
    if not np.isfinite(gamma) or gamma <= 0:
        raise ValueError('gamma must be finite and positive')
    if not 0 <= sharpen_strength <= 1 or not 0 <= clarity <= 1:
        raise ValueError('DCT enhancement strengths must be in [0, 1]')
    if not 1 <= chroma_gain <= 1.3:
        raise ValueError('chroma gain must be in [1, 1.3]')
    grids = tuple(tuple(map(int, shape)) for shape in grids)
    shapes = tuple(tuple(map(int, shape)) for shape in shapes)
    if len(grids) != len(shapes) or len(grids) not in (1, 3):
        raise ValueError('V7 source DCT expects one or three matching plane grids')
    if any(g[0] < s[0] or g[1] < s[1] for g, s in zip(grids, shapes)):
        raise ValueError('each transmitted shape must fit its coder grid')

    data = np.asarray(rgb)
    if data.ndim != 3 or data.shape[2] != 3:
        raise ValueError('source DCT expects an HxWx3 RGB frame')
    light = None
    height, width = data.shape[:2]
    if height < max(shape[0] for shape in grids) or \
            width < max(shape[1] for shape in grids):
        raise ValueError('source frame is smaller than the V7 coder grid')
    # Pre-shrink by division: blocks that leave the means near
    # PRESHRINK_FACTOR times the luma grid on each axis (never under 3x unless
    # the source is; 1 = no shrink); a remainder is one narrower last block.
    # An axis with whole block pairs is then decimated 2:1 by DECIMATE_TAPS,
    # so the transform runs on a plane near twice the grid. Detail finer than
    # the block means aliases into the sent band: 54-56 dB below the picture
    # (plain 2x block means: 39-43) and 35-42 dB below the detail in the top
    # half of the sent band (2x: 21-26), against the full-resolution
    # transform.
    luma_rows, luma_cols = grids[0]
    if max(height, width) > 4096*min(luma_rows, luma_cols):
        raise ValueError('source frame is too large for the direct encode')
    block_y = max(1, int(height/(PRESHRINK_FACTOR*luma_rows)+.5))
    block_x = max(1, int(width/(PRESHRINK_FACTOR*luma_cols)+.5))
    rows, cols = -(-height//block_y), -(-width//block_x)
    # (the filter's stopband covers the sent band while the decimated plane
    # is at least 1.75 times the grid)
    decimate_y = height % (2*block_y) == 0 and rows//2 >= 1.75*luma_rows
    decimate_x = width % (2*block_x) == 0 and cols//2 >= 1.75*luma_cols

    if data.dtype == np.uint8:
        # One compiled signature for every capture layout (mss hands over
        # strided BGRA->RGB views); the sender's warm-up compiles this one.
        data = np.ascontiguousarray(data)
        if luminance == 'linear' and len(grids) == 3:
            means, light = _toned_block_means_luminance_u8(
                data, _tone_lut(brightness, gamma),
                _luminance_luts(brightness, gamma), block_y, block_x,
                rows, cols)
        elif gamma == 1.0 and brightness <= 1.0:
            means = _block_means_u8(data, brightness/255.0, block_y, block_x,
                                    rows, cols)
        else:
            means = _toned_block_means_u8(
                data, _tone_lut_fixed(brightness, gamma), block_y, block_x,
                rows, cols)
    else:
        toned = np.clip(_rgb_float(data)*brightness, 0.0, 1.0)
        if gamma != 1.0:
            toned = toned**(1.0/gamma)
        means = _block_means_f64(toned, block_y, block_x, rows, cols)

    # YCbCr is linear, so converting block means equals averaging the
    # per-pixel conversion.
    if decimate_y or decimate_x:
        means = [_decimate(plane, decimate_y, decimate_x) for plane in means]
        rows, cols = means[0].shape
    red, green, blue = (np.ascontiguousarray(plane) for plane in means)
    target = (source_luminance(means, grids[0], light)
              if luminance and len(grids) == 3 else None)
    y, cb, cr = _ycbcr_kernel(red, green, blue, chroma_gain)
    planes = [y]
    blocks = [(block_y, block_x, int(decimate_y), int(decimate_x))]
    if len(grids) == 3:
        # Chroma's grid is half luma's: one more decimation where the plane
        # has whole pairs and stays at least 1.75 times the chroma grid.
        again_y = (decimate_y and height % (4*block_y) == 0 and
                   rows//2 >= 1.75*grids[1][0])
        again_x = (decimate_x and width % (4*block_x) == 0 and
                   cols//2 >= 1.75*grids[1][1])
        if again_y or again_x:
            cb = _decimate(cb, again_y, again_x)
            cr = _decimate(cr, again_y, again_x)
        planes += [cb, cr]
        blocks += [(block_y, block_x, decimate_y+again_y,
                    decimate_x+again_x)]*2
    # Pixel-domain radii are in luma grid pixels at any source size. Spec
    # order: chroma gain (above), clarity, then usm.
    unit = (rows/luma_rows, cols/luma_cols)
    if clarity:
        blurred = gaussian_filter(y, sigma=(6*unit[0], 6*unit[1]),
                                  mode='reflect')
        y = y+clarity*(y-blurred)
    if sharpen == 'usm' and sharpen_strength:
        blurred = gaussian_filter(y, sigma=(.8*unit[0], .8*unit[1]),
                                  mode='reflect')
        y = y+sharpen_strength*(y-blurred)
    planes[0] = np.ascontiguousarray(y)
    taper = sharpen_strength if sharpen == 'taper' else 0.0
    return planes, grids, shapes, taper, target, blocks, (height, width)


# ------------------------------------------------------------ luma adjustment
# Constant-luminance repair (luma adjustment, Strom et al., DCC 2016; the
# HDR chroma-subsampling fix). Y'CbCr from gamma-coded RGB leaves part of a
# saturated pixel's brightness in Cb/Cr; the wire sends chroma at a fraction
# of luma's resolution, so coloured edges lose brightness (dark or bright
# fringes, and much of the ringing at colour edges). The sender knows exactly
# which chroma coefficients the receiver will have, so it picks each luma
# grid value to make the pixel's linear luminance match the source.
LUMINANCE_WEIGHTS = np.array([.2126, .7152, .0722])            # sRGB / BT.709
LUMA_ADJUST_ITERATIONS = 6                                      # safeguarded Newton steps


def _srgb_to_linear(values):
    values = np.clip(values, 0.0, 1.0)
    return np.where(values <= .04045, values/12.92,
                    ((values+.055)/1.055)**2.4)


def _linear_to_srgb(values):
    values = np.clip(values, 0.0, 1.0)
    return np.where(values <= .0031308, values*12.92,
                    1.055*np.maximum(values, 0.0)**(1/2.4)-.055)


def _srgb_to_linear_extended(values):
    """sRGB decoding continued past [0, 1]: odd below black, straight above
    white (so a window's overshoot survives the round trip)."""
    values = np.asarray(values, dtype=np.float64)
    inside = _srgb_to_linear(values)
    below = -_srgb_to_linear(-values)
    above = 1.0+(values-1.0)*(2.4/1.055)
    return np.where(values < 0.0, below, np.where(values > 1.0, above, inside))


SRGB_LUT_SIZE = 4096


def _srgb_table():
    codes = np.linspace(0.0, 1.0, SRGB_LUT_SIZE+1)
    table = np.where(codes <= .04045, codes/12.92,
                     ((codes+.055)/1.055)**2.4)
    table.setflags(write=False)
    return table


SRGB_LUT = _srgb_table()


@njit(cache=True, fastmath=True, nogil=True, inline='always')
def _linear_and_slope(value, table):
    """sRGB code -> (linear light, d linear/d code), by linear interpolation
    in a 4,097-entry table (error below 1e-7); slope 0 where the display
    clips."""
    if value <= 0.0:
        return 0.0, 0.0
    if value >= 1.0:
        return 1.0, 0.0
    position = value*(table.shape[0]-1)
    index = int(position)
    fraction = position-index
    low = table[index]
    step = table[index+1]-low
    return low+fraction*step, step*(table.shape[0]-1)


@njit(cache=True, fastmath=True, nogil=True)
def _luminance_of_means(red, green, blue, table, out):
    rows, cols = red.shape
    for i in range(rows):
        for j in range(cols):
            lr, _ = _linear_and_slope(red[i, j], table)
            lg, _ = _linear_and_slope(green[i, j], table)
            lb, _ = _linear_and_slope(blue[i, j], table)
            out[i, j] = .2126*lr+.7152*lg+.0722*lb
    return out


@njit(cache=True, fastmath=True, nogil=True)
def _luma_adjust_kernel(luma, cb, cr, target, iterations, table):
    """In place: safeguarded Newton per pixel (see luma_adjust)."""
    rows, cols = luma.shape
    for i in range(rows):
        for j in range(cols):
            low, high, value = -1.0, 1.0, luma[i, j]
            blue_shift = cb[i, j]*.5-.5/255.0
            red_shift = cr[i, j]*.5-.5/255.0
            green_shift = -.344136*blue_shift-.714136*red_shift
            red_shift *= 1.402
            blue_shift *= 1.772
            goal = target[i, j]
            for _ in range(iterations):
                y = (value+1.0)*.5
                lr, sr = _linear_and_slope(y+red_shift, table)
                lg, sg = _linear_and_slope(y+green_shift, table)
                lb, sb = _linear_and_slope(y+blue_shift, table)
                error = .2126*lr+.7152*lg+.0722*lb-goal
                if abs(error) < 1e-6:
                    break
                if error > 0:
                    high = value
                else:
                    low = value
                slope = .5*(.2126*sr+.7152*sg+.0722*sb)
                step = value-error/slope if slope > 1e-6 else 2.0
                value = step if low <= step <= high else (low+high)*.5
            luma[i, j] = value
    return luma


def source_luminance(means, luma_grid, luminance=None):
    """Linear luminance of the toned source on the luma grid.

    ``luminance`` is the per-block mean of the source pixels' linear
    luminance (the uint8 encoder's own pass). Without it, the encoder's R, G,
    B block ``means`` (at least twice the luma grid) are linearised instead,
    which is exact only where a block is flat. Either is then area-averaged
    exactly to the grid.
    """
    if luminance is None:
        red, green, blue = (np.ascontiguousarray(plane, dtype=np.float64)
                            for plane in means)
        luminance = _luminance_of_means(red, green, blue, SRGB_LUT,
                                        np.empty(red.shape, np.float64))
    rows, cols = (int(value) for value in luma_grid)
    if luminance.shape == (rows, cols):
        return luminance
    return _area_box_resample(luminance, (rows, cols))


def _shown_rgb(luma, cb, cr):
    """The receiver shader's RGB (before its clip) for grid values."""
    y = (luma+1.0)*.5
    cb = cb*.5-.5/255.0
    cr = cr*.5-.5/255.0
    return np.stack((y+1.402*cr, y-.344136*cb-.714136*cr, y+1.772*cb), -1)


def _shown_luminance(luma, cb, cr):
    """Linear luminance the receiver shows for grid values."""
    return _srgb_to_linear(_shown_rgb(luma, cb, cr)) @ LUMINANCE_WEIGHTS


@lru_cache(maxsize=8)
def _received_chroma_plan(rows, cols, luma_rows, luma_cols):
    """Chroma grid -> its DCT, and that DCT zero-padded -> the luma grid."""
    scale = (luma_rows*luma_cols/(rows*cols))**.25
    plan = (np.ascontiguousarray(_dct_matrix(rows, rows).T),
            np.ascontiguousarray(_dct_matrix(cols, cols)),
            np.ascontiguousarray(_dct_matrix(luma_rows, rows)*scale),
            np.ascontiguousarray(_dct_matrix(luma_cols, cols).T*scale))
    for array in plan:
        array.setflags(write=False)
    return plan


def received_chroma(values, grids, chroma_sent):
    """The receiver's Cb, Cr on the luma grid: only the sent coefficients,
    evaluated by DCT zero-padding (what DCT reconstruction displays)."""
    offsets = np.cumsum([0] + [rows*cols for rows, cols in grids])
    luma_rows, luma_cols = grids[0]
    out = []
    for plane, sent in zip((1, 2), chroma_sent):
        rows, cols = grids[plane]
        analysis_y, analysis_x, synthesis_y, synthesis_x = \
            _received_chroma_plan(rows, cols, luma_rows, luma_cols)
        grid = np.ascontiguousarray(
            values[offsets[plane]:offsets[plane+1]],
            dtype=np.float64).reshape(rows, cols)
        coefficients = _separable(analysis_y, grid, analysis_x)
        coefficients[~np.asarray(sent, bool).reshape(rows, cols)] = 0.0
        out.append(_separable(synthesis_y, coefficients, synthesis_x))
    return out


def luma_adjust(values, grids, chroma_sent, target,
                iterations=LUMA_ADJUST_ITERATIONS):
    """Re-fit the luma grid for the chroma the receiver will actually show.

    ``values`` is the concatenated Y/Cb/Cr grid vector; ``chroma_sent`` one
    mask per chroma grid of the coefficients the wire carries; ``target`` the
    source's linear luminance on the luma grid (source_luminance). Each luma
    value becomes the one, in [-1, 1], whose shown luminance matches the
    target (a few Newton steps from the plain value). Chroma is unchanged, so
    the result encodes like any other values vector.
    """
    values = np.array(values, dtype=np.float64)
    if target is None or len(grids) != 3:
        return values
    rows, cols = grids[0]
    target = np.asarray(target, np.float64).reshape(rows, cols)
    cb, cr = received_chroma(values, grids, chroma_sent)
    luma = np.ascontiguousarray(values[:rows*cols].reshape(rows, cols))
    # Safeguarded Newton on the luma value (numba, per pixel): shown luminance
    # never falls as luma rises (each RGB channel moves by half a luma step
    # until the display clips it), so the root is kept bracketed; a Newton
    # step that leaves the bracket, or a clipped pixel with no slope, bisects.
    _luma_adjust_kernel(luma, np.ascontiguousarray(cb),
                        np.ascontiguousarray(cr), np.ascontiguousarray(target),
                        int(iterations), SRGB_LUT)
    values[:rows*cols] = luma.ravel()
    return values


# Ten alternating projections hold ~90% of the converged gain at ~3 ms.
CLIP_AWARE_ITERATIONS = 10
_CLIP_EDGE = 1e-6


def clip_aware_luma(values, luma_shape, sent, iterations=CLIP_AWARE_ITERATIONS):
    """Re-fit the sent luma coefficients for a receiver that clips to [-1, 1].

    The receiver shows only the sent coefficients, and the display clips at
    black and white. Ringing past those limits is invisible, so it costs
    nothing; ringing inside them is the halo around edges. Alternating
    projections between "only these coefficients" and "matches the target
    where it is inside the range, at or beyond the limit where the target
    sits on it" move the ringing into the clipped region. ``values`` is the
    concatenated grid vector (luma first); ``sent`` marks the luma grid
    coefficients the wire carries. Only those coefficients change, so the
    result encodes exactly like any other values vector (it may leave
    [-1, 1] where that is invisible). Zero iterations returns the input.
    """
    values = np.asarray(values, dtype=np.float64)
    iterations = int(iterations)
    if iterations <= 0:
        return values
    rows, cols = (int(value) for value in luma_shape)
    count = rows*cols
    sent = np.asarray(sent, dtype=bool).reshape(rows, cols)
    luma = values[:count].reshape(rows, cols)
    full = dctn(luma, norm='ortho')
    target = np.clip(luma, -1.0, 1.0)
    upper = target >= 1.0-_CLIP_EDGE
    lower = target <= -1.0+_CLIP_EDGE
    coefficients = np.where(sent, full, 0.0)
    for _ in range(iterations):
        shown = idctn(coefficients, norm='ortho')
        goal = np.where(upper, np.maximum(shown, 1.0),
                        np.where(lower, np.minimum(shown, -1.0), target))
        coefficients = np.where(sent, dctn(goal, norm='ortho'), 0.0)
    result = values.copy()
    result[:count] = idctn(np.where(sent, coefficients, full),
                           norm='ortho').ravel()
    return result


def warmup_direct_dct(grids, shapes):
    """Compile the direct encoder for writable and read-only frames.

    Numba compiles one specialization per array writability. FFmpeg frames
    wrapped from ``bytes`` and ``np.asarray`` of a Pillow image are read-only,
    a warm-up frame from ``np.zeros`` is not; a specialization first met on
    a live frame compiles there (about 2 s, a visible stall).
    """
    rows = max(shape[0] for shape in grids)
    cols = max(shape[1] for shape in grids)
    frame = np.zeros((rows, cols, 3), dtype=np.uint8)
    read_only = frame.copy()
    read_only.setflags(write=False)
    for data in (frame, read_only):
        direct_dct_values(data, grids, shapes)
        direct_dct_values(data, grids, shapes, sharpen='taper')
        direct_dct_values(data, grids, shapes, luminance_out=[])
        direct_dct_values(data, grids, shapes, luminance_out=[],
                          linear_light=True)
        direct_dct_values(data, grids, shapes, brightness=1.05)


@lru_cache(maxsize=8)
def _pixel_plan(rows, cols, grid_rows, grid_cols):
    """rows×cols picture -> the grid values whose DCT is that picture's own
    DCT, zero-padded (the coder's amplitude scale)."""
    scale = (grid_rows*grid_cols/(rows*cols))**.25
    left = np.ascontiguousarray(
        _dct_matrix(grid_rows, rows) @ _dct_matrix(rows, rows).T*scale)
    right = np.ascontiguousarray(
        _dct_matrix(cols, cols) @ _dct_matrix(grid_cols, cols).T*scale)
    for array in (left, right):
        array.setflags(write=False)
    return left, right


# Pixel encode downscales. 'average' is the pixel-domain area average. The
# others downscale inside the transform: the sent rectangle is the source's
# own leading DCT coefficients, each weighted by droop**p, where droop is the
# gain an area average has at that frequency. p = +1 ('soft') is the area
# average without its aliasing; p = 0 ('cut') keeps the coefficients as they
# are, the most detail the rectangle can hold; p = -1 ('crisp') undoes pixel
# repetition, which returns block art (a whole multiple of the rectangle)
# exactly. Without aliasing to mask it, a hard edge rings: flat areas next to
# edges show a faint mesh that grows from soft to crisp.
PIXEL_DETAILS = {'average': None, 'soft': 1.0, 'cut': 0.0, 'crisp': -1.0}


def _area_gain(count, block):
    """Gain of a ``block``-pixel area average (any real block >= 1) on the
    first ``count`` DCT indexes of a count*block-pixel axis."""
    gain = np.ones(int(count), dtype=np.float64)
    if block > 1:
        theta = np.pi*np.arange(1, int(count))/(int(count)*float(block))
        gain[1:] = np.sin(block*theta/2)/(block*np.sin(theta/2))
    return gain


def _pixel_values_in_dct(rgb, grids, shapes, brightness, gamma, power):
    data = np.asarray(rgb)
    height, width = data.shape[:2]
    planes = direct_dct_coefficients(rgb, grids, shapes,
                                     brightness=brightness, gamma=gamma)
    result = np.empty(sum(r*c for r, c in grids), dtype=np.float64)
    offset = 0
    for coefficients, (grid_rows, grid_cols), (rows, cols) in zip(
            planes, grids, shapes):
        kept = np.zeros((grid_rows, grid_cols))
        kept[:rows, :cols] = coefficients[:rows, :cols]*np.outer(
            _area_gain(rows, height/rows)**power,
            _area_gain(cols, width/cols)**power)
        grid = _separable(np.ascontiguousarray(_dct_matrix(grid_rows, grid_rows)),
                          kept,
                          np.ascontiguousarray(_dct_matrix(grid_cols, grid_cols).T))
        result[offset:offset+grid.size] = grid.ravel()
        offset += grid.size
    return result


def pixel_dct_values(rgb, grids, shapes, *, brightness=1.0, gamma=1.0,
                     detail='average'):
    """Pixel encode: the frame area-averaged to the sent rectangles
    (``shapes``: 48x40 luma, 24x20 chroma) and returned as the grid values
    whose DCT is exactly those small pictures' own DCT.

    A wire that carries each plane's full ``shapes`` rectangle (Fold 500)
    then delivers the small picture pixel for pixel: a receiver that
    evaluates the coefficients on that rectangle (the 'pixel' display) gets
    the sender's pixels back, up to channel noise. A source that is a whole
    multiple of the luma rectangle passes without any resampling blur.
    Values are not clipped: the zero-padded picture overshoots at hard edges
    on the coder grid, and clipping that would change the sent pixels.

    ``detail`` other than 'average' downscales inside the transform instead
    (PIXEL_DETAILS); a source smaller than the coder grid always uses the
    average.
    """
    if detail not in PIXEL_DETAILS:
        raise ValueError(f'unknown pixel detail {detail!r}')
    grids = tuple(tuple(map(int, shape)) for shape in grids)
    shapes = tuple(tuple(map(int, shape)) for shape in shapes)
    data = np.asarray(rgb)
    if data.ndim != 3 or data.shape[2] != 3:
        raise ValueError('pixel encode expects an HxWx3 RGB frame')
    if (PIXEL_DETAILS[detail] is not None and
            data.shape[0] >= max(shape[0] for shape in grids) and
            data.shape[1] >= max(shape[1] for shape in grids)):
        return _pixel_values_in_dct(data, grids, shapes, brightness, gamma,
                                    PIXEL_DETAILS[detail])
    rows, cols = shapes[0]
    height, width = data.shape[:2]
    if height < rows or width < cols:
        # Smaller than the sent rectangle: repeat pixels up to it.
        data = np.repeat(np.repeat(data, -(-rows//height), axis=0),
                         -(-cols//width), axis=1)
        height, width = data.shape[:2]
    # Whole blocks that never straddle a sent pixel: the largest block that
    # divides the source pixels per sent pixel (or, when that is not a whole
    # number, the source size) and leaves at least four samples per sent pixel.
    def block_for(size, sent):
        span = size//sent if size % sent == 0 else size
        return next(block for block in range(max(1, size//(4*sent)), 0, -1)
                    if span % block == 0)

    block_y, block_x = block_for(height, rows), block_for(width, cols)
    mean_rows, mean_cols = height//block_y, width//block_x
    if data.dtype == np.uint8:
        data = np.ascontiguousarray(data)
        if gamma == 1.0 and brightness <= 1.0:
            means = _block_means_u8(data, brightness/255.0, block_y, block_x,
                                    mean_rows, mean_cols)
        else:
            means = _toned_block_means_u8(
                data, _tone_lut_fixed(brightness, gamma), block_y, block_x,
                mean_rows, mean_cols)
    else:
        toned = np.clip(_rgb_float(data)*brightness, 0.0, 1.0)
        if gamma != 1.0:
            toned = toned**(1.0/gamma)
        means = _block_means_f64(toned, block_y, block_x, mean_rows,
                                 mean_cols)
    red, green, blue = (plane if plane.shape == (rows, cols) else
                        _area_box_resample(plane, (rows, cols))
                        for plane in means)
    planes = _ycbcr_kernel(np.ascontiguousarray(red),
                           np.ascontiguousarray(green),
                           np.ascontiguousarray(blue), 1.0)[:len(grids)]
    result = np.empty(sum(r*c for r, c in grids), dtype=np.float64)
    offset = 0
    for plane, (grid_rows, grid_cols), (sent_rows, sent_cols) in zip(
            planes, grids, shapes):
        if plane.shape != (sent_rows, sent_cols):
            plane = _area_box_resample(plane, (sent_rows, sent_cols))
        left, right = _pixel_plan(sent_rows, sent_cols, grid_rows, grid_cols)
        grid = _separable(left, np.ascontiguousarray(plane), right)
        result[offset:offset+grid.size] = grid.ravel()
        offset += grid.size
    result *= 2.0
    result -= 1.0
    return result


def direct_dct_coefficients(rgb, grids, shapes, *, brightness=1.0, gamma=1.0,
                            sharpen='off', sharpen_strength=.25, clarity=0.0,
                            chroma_gain=1.0):
    """Direct DCT encode to full-grid coefficients, one array per plane.

    Each array is the orthonormal DCT of that plane's coder-grid values
    (``2*x - 1``), i.e. what ``SourceCoder.forward`` computes from the values
    before keeping its corner; concatenated and flattened it is the vector
    ``Fold500.encode_dct_coefficients`` accepts. This is the entry point for
    coefficient-domain folding: nothing here resamples pixels, and values are
    not clipped (clipping is a pixel-domain operation).
    """
    planes, grids, shapes, taper, _, blocks, source = _direct_planes(
        rgb, grids, shapes, brightness, gamma, sharpen, sharpen_strength,
        clarity, chroma_gain)
    result = []
    for index, (plane, (grid_rows, grid_cols), sent) in enumerate(
            zip(planes, grids, shapes)):
        analysis_y, analysis_x, *_ = _direct_plan(
            *source, grid_rows, grid_cols, *blocks[index])
        coefficients = 2.0*_separable(analysis_y, plane, analysis_x)
        coefficients[0, 0] -= np.sqrt(grid_rows*grid_cols)
        if index == 0 and taper:
            coefficients *= _taper_gain(grid_rows, grid_cols, sent[0],
                                        sent[1], taper)
        result.append(coefficients)
    return result


class KernelFrame:
    """One frame's dealings with a pluggable kernel (animation_modem.v7_kernels).

    Built by ``direct_dct_values`` from the pre-shrunk planes. ``gain`` is the
    kernel's window for a plane; ``filter_target`` puts the same window on the
    luminance luma adjustment aims at; ``post`` is the non-linear refit that
    runs on the final values. A kernel that raises is bypassed for the frame
    and its message kept for the sender to report.
    """

    def __init__(self, selection, grids, shapes, masks, planes):
        self.selection = selection
        self.kernel = selection.kernel
        self.params = selection.params
        self.contexts = []
        masks = list(masks) if masks is not None else [None]*len(grids)
        masks += [None]*(len(grids)-len(masks))
        for index, (grid, sent, mask, plane) in enumerate(
                zip(grids, shapes, masks, planes)):
            if mask is not None:
                mask = np.asarray(mask, bool)
                if mask.size != grid[0]*grid[1]:
                    mask = None
            self.contexts.append(KernelContext(index, grid, sent, mask, plane))

    def gain(self, index):
        try:
            return self.kernel.gain(self.contexts[index], self.params)
        except Exception as exc:
            self.kernel.note_failure(exc)
            return None

    def filter_target(self, target):
        """The luminance goal seen through the luma window.

        The goal is linear light, the window is designed on the picture as the
        wire and the receiver show it (gamma coded), where halos and sharpness
        are judged. So the goal goes to that domain, through the window, and
        back; filtering linear light directly would turn a window that is
        gentle on a dark edge into a harsh one.
        """
        gain = self.gain(0)
        if gain is None or target is None:
            return target
        rows, cols = target.shape
        if (rows, cols) != gain.shape:
            return target
        basis_y = _dct_matrix(rows, rows)
        basis_x = _dct_matrix(cols, cols)
        coded = _linear_to_srgb(target)
        coefficients = basis_y.T @ coded @ basis_x
        return _srgb_to_linear_extended(
            basis_y @ (coefficients*gain) @ basis_x.T)

    def post(self, values, grids):
        """Run the kernel's refit on each plane of a concatenated value vector."""
        if not self.kernel.has_post:
            return values
        values = np.array(values, dtype=np.float64)
        offset = 0
        for index, (rows, cols) in enumerate(grids):
            count = rows*cols
            plane = values[offset:offset+count].reshape(rows, cols)
            try:
                refit = self.kernel.post((plane+1.0)*.5, self.contexts[index],
                                         self.params)
            except Exception as exc:
                self.kernel.note_failure(exc)
            else:
                values[offset:offset+count] = (refit*2.0-1.0).ravel()
            offset += count
        return values


def direct_dct_values(rgb, grids, shapes, *, brightness=1.0, gamma=1.0,
                      sharpen='off', sharpen_strength=.25, clarity=0.0,
                      chroma_gain=1.0, luminance_out=None,
                      linear_light=False, kernel=None, kernel_masks=None,
                      kernel_frame_out=None):
    """Direct DCT encode: native RGB frame to the sender's coder-grid values.

    Stages (see the direct-encode spec): tone per pixel at full resolution,
    fused with a block average to about four times the luma grid; 2:1
    decimation filter; YCbCr on the result; pixel-domain enhancement; DCT
    truncation to each grid with the pre-shrink's droop divided out; optional
    taper on the luma coefficients; inverse grid DCT; ``clip(2x - 1)``.
    Returns the concatenated Y/Cb/Cr values in [-1, 1]. A list passed as
    ``luminance_out`` receives the source's linear luminance on the luma grid
    (the target for luma_adjust): the linearised pre-shrunk means, or with
    ``linear_light`` the mean of every source pixel's linear luminance.

    ``kernel`` (a ``v7_kernels.KernelSelection``) puts a pluggable window on
    each plane's coefficients; ``kernel_masks`` names, per plane, the
    coefficients the wire carries (the transmitted rectangle when absent). A
    list passed as ``kernel_frame_out`` receives the frame's KernelFrame, whose
    ``post`` the caller runs after luma adjustment.
    """
    planes, grids, shapes, taper, target, blocks, source = _direct_planes(
        rgb, grids, shapes, brightness, gamma, sharpen, sharpen_strength,
        clarity, chroma_gain,
        luminance=(luminance_out is not None and
                   ('linear' if linear_light else True)))
    frame = None
    if kernel is not None:
        frame = KernelFrame(kernel, grids, shapes, kernel_masks, planes)
        if kernel_frame_out is not None:
            kernel_frame_out.append(frame)
    if luminance_out is not None:
        luminance_out.append(frame.filter_target(target)
                             if frame is not None else target)
    result = np.empty(sum(r*c for r, c in grids), dtype=np.float64)
    offset = 0
    for index, (plane, (grid_rows, grid_cols), sent) in enumerate(
            zip(planes, grids, shapes)):
        (analysis_y, analysis_x, synthesis_y, synthesis_x,
         left, right) = _direct_plan(*source, grid_rows, grid_cols,
                                     *blocks[index])
        window = _taper_gain(grid_rows, grid_cols, sent[0], sent[1],
                             taper) if index == 0 and taper else None
        shaped = frame.gain(index) if frame is not None else None
        if shaped is not None:
            window = shaped if window is None else window*shaped
        if window is not None:
            coefficients = _separable(analysis_y, plane, analysis_x)
            coefficients *= window
            grid = _separable(synthesis_y, coefficients, synthesis_x)
        else:
            grid = _separable(left, plane, right)
        count = grid_rows*grid_cols
        result[offset:offset+count] = grid.ravel()
        offset += count
    result *= 2.0
    result -= 1.0
    np.clip(result, -1.0, 1.0, out=result)
    return result


def source_dct_values(rgb, grids, shapes, *, brightness=1.0, gamma=1.0,
                      clip_values=True, sharpen='off', sharpen_strength=.25,
                      clarity=0.0, chroma_gain=1.0, aggregation='off',
                      band_profile='off', display_size=(1080, 900),
                      viewing_distance_mm=600.0, display_dpi=96.0):
    """Analyze a native RGB frame and return coder-grid values plus diagnostics.

    Each full-resolution float Y/Cb/Cr plane is orthonormally transformed. The
    direct candidate retains its low-frequency coder-grid corner. Optional
    weighted modes apply same-index spectral gain windows; see ``_aggregate``.
    They never average signed DCT coefficients. The resulting coefficients are
    amplitude-normalized and inverse transformed to the existing V7 spatial-
    value interface. The normal sender then performs its ordinary DCT and
    rank/fold work exactly once.
    """
    if sharpen not in SHARPEN_MODES:
        raise ValueError(f'unknown DCT sharpen mode {sharpen!r}')
    if aggregation not in AGGREGATIONS:
        raise ValueError(f'unknown coefficient aggregation {aggregation!r}')
    if band_profile not in BAND_PROFILES:
        raise ValueError(f'unknown DCT band profile {band_profile!r}')
    brightness, gamma = float(brightness), float(gamma)
    sharpen_strength, clarity = float(sharpen_strength), float(clarity)
    chroma_gain = float(chroma_gain)
    if not np.isfinite(brightness) or brightness <= 0:
        raise ValueError('brightness must be finite and positive')
    if not np.isfinite(gamma) or gamma <= 0:
        raise ValueError('gamma must be finite and positive')
    if not 0 <= sharpen_strength <= 1 or not 0 <= clarity <= 1:
        raise ValueError('DCT enhancement strengths must be in [0, 1]')
    if not 1 <= chroma_gain <= 1.3:
        raise ValueError('chroma gain must be in [1, 1.3]')

    rgb01 = _rgb_float(rgb)
    height, width = rgb01.shape[:2]
    grids = tuple(tuple(map(int, shape)) for shape in grids)
    shapes = tuple(tuple(map(int, shape)) for shape in shapes)
    if len(grids) != len(shapes) or len(grids) not in (1, 3):
        raise ValueError('V7 source DCT expects one or three matching plane grids')
    if height < max(shape[0] for shape in grids) or \
            width < max(shape[1] for shape in grids):
        raise ValueError('source frame is smaller than the V7 coder grid')
    if any(gr[0] < sh[0] or gr[1] < sh[1]
           for gr, sh in zip(grids, shapes)):
        raise ValueError('each transmitted shape must fit its coder grid')

    toned = np.clip(rgb01*brightness, 0.0, 1.0)
    rgb_clip_fraction = float(np.count_nonzero(toned != rgb01*brightness)/toned.size)
    if gamma != 1.0:
        toned = toned**(1.0/gamma)
    red, green, blue = (toned[..., index] for index in range(3))
    y = .299000*red + .587000*green + .114000*blue
    cb_unclipped = (NEUTRAL_CHROMA-.168736*red-.331264*green+.5*blue)
    cr_unclipped = (NEUTRAL_CHROMA+.5*red-.418688*green-.081312*blue)
    cb = np.clip(cb_unclipped, 0.0, 1.0)
    cr = np.clip(cr_unclipped, 0.0, 1.0)
    chroma_clipped = int(np.count_nonzero(cb != cb_unclipped) +
                         np.count_nonzero(cr != cr_unclipped))
    source_chroma_count = 2*height*width

    if chroma_gain != 1.0:
        cb_raw = NEUTRAL_CHROMA+(cb-NEUTRAL_CHROMA)*chroma_gain
        cr_raw = NEUTRAL_CHROMA+(cr-NEUTRAL_CHROMA)*chroma_gain
        chroma_clipped += int(np.count_nonzero(cb_raw != np.clip(cb_raw, 0, 1))+
                              np.count_nonzero(cr_raw != np.clip(cr_raw, 0, 1)))
        cb, cr = np.clip(cb_raw, 0, 1), np.clip(cr_raw, 0, 1)

    if sharpen == 'usm' and sharpen_strength:
        sy = .8*height/grids[0][0]
        sx = .8*width/grids[0][1]
        y = y+sharpen_strength*(y-gaussian_filter(y, sigma=(sy, sx),
                                                 mode='reflect'))
    if clarity:
        sy = 6.0*height/grids[0][0]
        sx = 6.0*width/grids[0][1]
        y = y+clarity*(y-gaussian_filter(y, sigma=(sy, sx), mode='reflect'))

    planes = (y,) if len(grids) == 1 else (y, cb, cr)
    reconstructed = []
    for plane_index, (plane, grid, sent_shape) in enumerate(
            zip(planes, grids, shapes)):
        grid_rows, grid_cols = grid
        if aggregation == 'area-box':
            # This is an explicit spatial reduction followed by the target-grid
            # DCT. Unlike signed-frequency averaging, it cannot mix unrelated
            # whole-frame DCT modes or cancel the image's AC energy.
            reduced = _area_box_resample(plane, (grid_rows, grid_cols))
            if band_profile == 'off' and not (
                    sharpen == 'taper' and sharpen_strength):
                # The ordinary coder will perform the target-grid DCT. Avoid
                # an identity DCT/IDCT pair here when no spectral shaping uses
                # the coefficients in this preparation stage.
                reconstructed.append(2.0*reduced-1.0)
                continue
            coefficients = dctn(reduced, norm='ortho')
        else:
            coefficients = dctn(plane, norm='ortho')
            coefficients = _aggregate(
                coefficients, aggregation, (grid_rows, grid_cols)).copy()
            coefficients *= np.sqrt((grid_rows*grid_cols)/(height*width))
        if sharpen == 'taper' and plane_index == 0 and sharpen_strength:
            sent_rows, sent_cols = sent_shape
            u = np.arange(sent_rows, dtype=np.float64)[:, None]
            v = np.arange(sent_cols, dtype=np.float64)[None, :]
            radius = np.hypot(u/sent_rows, v/sent_cols)
            taper = np.zeros_like(radius)
            active = radius < 1
            taper[active] = (27.0/4.0)*radius[active]**2*(1-radius[active])
            coefficients[:sent_rows, :sent_cols] *= (
                1.0+sharpen_strength*taper)
        if band_profile != 'off':
            coefficients *= _frequency_weights(
                grid, plane_index, band_profile, display_size,
                viewing_distance_mm, display_dpi,
                viewport_aspect=width/height)
        reconstructed.append(2.0*idctn(coefficients, norm='ortho')-1.0)

    raw_values = np.concatenate([plane.ravel() for plane in reconstructed])
    outside = np.maximum(np.abs(raw_values)-1.0, 0.0)
    grid_clip_count = int(np.count_nonzero(outside))
    values = np.clip(raw_values, -1.0, 1.0) if clip_values else raw_values
    stats = {
        'source_size': [int(width), int(height)],
        'tone_rgb_clip_fraction': rgb_clip_fraction,
        'source_chroma_clip_fraction': chroma_clipped/source_chroma_count,
        'grid_clip_fraction': grid_clip_count/len(raw_values),
        'grid_clip_max_excess': float(outside.max(initial=0.0)),
        'grid_value_min': float(raw_values.min(initial=0.0)),
        'grid_value_max': float(raw_values.max(initial=0.0)),
        'clip_values': bool(clip_values),
        'aggregation': aggregation,
        'band_profile': band_profile,
        'sharpen': sharpen,
        'sharpen_strength': sharpen_strength,
        'clarity': clarity,
        'chroma_gain': chroma_gain,
    }
    if clip_values and not np.isfinite(values).all():
        raise FloatingPointError('source DCT produced non-finite coder values')
    if not clip_values and not np.isfinite(values).all():
        raise FloatingPointError('source DCT produced non-finite coder values')
    return values, stats
