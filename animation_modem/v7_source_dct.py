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


@lru_cache(maxsize=32)
def _direct_plan(height, width, rows, cols):
    """Cached matrices for DCT truncation of an h×w plane to an r×c grid.

    ``C = analysis_y @ plane @ analysis_x`` is the plane's leading r×c
    orthonormal DCT, amplitude-normalized to the grid (the coder's own
    ``sqrt(rc/hw)``). ``synthesis_y @ C @ synthesis_x`` is the grid's inverse
    DCT, and ``left @ plane @ right`` is both steps folded together.
    """
    norm = (rows*cols/(height*width))**.25
    analysis_y = np.ascontiguousarray(_dct_matrix(height, rows).T*norm)
    analysis_x = np.ascontiguousarray(_dct_matrix(width, cols)*norm)
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


@njit(cache=True, nogil=True)
def _toned_block_means_u8(data, lut, block_y, block_x, out_rows, out_cols):
    """Tone uint8 RGB through ``lut`` and average block_y×block_x blocks.

    One pass over the frame; returns contiguous R, G, B mean planes.
    """
    result = np.zeros((3, out_rows, out_cols), dtype=np.float64)
    scale = 1.0/(block_y*block_x)
    for by in range(out_rows):
        for dy in range(block_y):
            y = by*block_y+dy
            for bx in range(out_cols):
                red = 0.0
                green = 0.0
                blue = 0.0
                x0 = bx*block_x
                for dx in range(block_x):
                    red += lut[data[y, x0+dx, 0]]
                    green += lut[data[y, x0+dx, 1]]
                    blue += lut[data[y, x0+dx, 2]]
                result[0, by, bx] += red
                result[1, by, bx] += green
                result[2, by, bx] += blue
        for channel in range(3):
            for bx in range(out_cols):
                result[channel, by, bx] *= scale
    return result


@njit(cache=True, nogil=True)
def _block_means_f64(data, block_y, block_x, out_rows, out_cols):
    result = np.zeros((3, out_rows, out_cols), dtype=np.float64)
    scale = 1.0/(block_y*block_x)
    for by in range(out_rows):
        for dy in range(block_y):
            y = by*block_y+dy
            for bx in range(out_cols):
                x0 = bx*block_x
                for dx in range(block_x):
                    for channel in range(3):
                        result[channel, by, bx] += data[y, x0+dx, channel]
        for channel in range(3):
            for bx in range(out_cols):
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


def _direct_planes(rgb, grids, shapes, brightness, gamma, sharpen,
                   sharpen_strength, clarity, chroma_gain):
    """Validated, pre-shrunk and pixel-enhanced Y[/Cb/Cr] planes in [0, 1]."""
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
    height, width = data.shape[:2]
    if height < max(shape[0] for shape in grids) or \
            width < max(shape[1] for shape in grids):
        raise ValueError('source frame is smaller than the V7 coder grid')
    # Pre-shrink by division: the largest whole blocks that keep the plane
    # at least twice the luma grid on each axis (1 = no shrink).
    luma_rows, luma_cols = grids[0]
    block_y = max(1, height//(2*luma_rows))
    block_x = max(1, width//(2*luma_cols))
    rows, cols = height//block_y, width//block_x

    if data.dtype == np.uint8:
        # One compiled signature for every capture layout (mss hands over
        # strided BGRA->RGB views); the sender's warm-up compiles this one.
        data = np.ascontiguousarray(data)
        means = _toned_block_means_u8(
            data, _tone_lut(brightness, gamma), block_y, block_x, rows, cols)
    else:
        toned = np.clip(_rgb_float(data)*brightness, 0.0, 1.0)
        if gamma != 1.0:
            toned = toned**(1.0/gamma)
        means = _block_means_f64(toned, block_y, block_x, rows, cols)

    # YCbCr is linear, so converting block means equals averaging the
    # per-pixel conversion.
    red, green, blue = means
    y = .299000*red + .587000*green + .114000*blue
    planes = [y]
    if len(grids) == 3:
        cb = .5-.168736*red-.331264*green+.5*blue
        cr = .5+.5*red-.418688*green-.081312*blue
        if chroma_gain != 1.0:
            cb = np.clip(.5+chroma_gain*(cb-.5), 0.0, 1.0)
            cr = np.clip(.5+chroma_gain*(cr-.5), 0.0, 1.0)
        planes += [cb, cr]
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
    return planes, grids, shapes, taper


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
    planes, grids, shapes, taper = _direct_planes(
        rgb, grids, shapes, brightness, gamma, sharpen, sharpen_strength,
        clarity, chroma_gain)
    rows, cols = planes[0].shape
    result = []
    for index, (plane, (grid_rows, grid_cols), sent) in enumerate(
            zip(planes, grids, shapes)):
        analysis_y, analysis_x, *_ = _direct_plan(
            rows, cols, grid_rows, grid_cols)
        coefficients = 2.0*_separable(analysis_y, plane, analysis_x)
        coefficients[0, 0] -= np.sqrt(grid_rows*grid_cols)
        if index == 0 and taper:
            coefficients *= _taper_gain(grid_rows, grid_cols, sent[0],
                                        sent[1], taper)
        result.append(coefficients)
    return result


def direct_dct_values(rgb, grids, shapes, *, brightness=1.0, gamma=1.0,
                      sharpen='off', sharpen_strength=.25, clarity=0.0,
                      chroma_gain=1.0):
    """Direct DCT encode: native RGB frame to the sender's coder-grid values.

    Stages (see the direct-encode spec): tone per pixel at full resolution
    through a 256-entry table, fused with a block average that leaves the
    plane at least twice the luma grid; YCbCr on the block means; pixel-
    domain enhancement; DCT truncation to each grid; optional taper on the
    luma coefficients; inverse grid DCT; ``clip(2x - 1)``. Returns the
    concatenated Y/Cb/Cr values in [-1, 1].
    """
    planes, grids, shapes, taper = _direct_planes(
        rgb, grids, shapes, brightness, gamma, sharpen, sharpen_strength,
        clarity, chroma_gain)
    rows, cols = planes[0].shape
    result = np.empty(sum(r*c for r, c in grids), dtype=np.float64)
    offset = 0
    for index, (plane, (grid_rows, grid_cols), sent) in enumerate(
            zip(planes, grids, shapes)):
        (analysis_y, analysis_x, synthesis_y, synthesis_x,
         left, right) = _direct_plan(rows, cols, grid_rows, grid_cols)
        if index == 0 and taper:
            coefficients = _separable(analysis_y, plane, analysis_x)
            coefficients *= _taper_gain(grid_rows, grid_cols, sent[0],
                                        sent[1], taper)
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
