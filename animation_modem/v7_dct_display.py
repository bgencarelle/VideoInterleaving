"""Direct DCT reconstruction of decoded V7 planes at an arbitrary size.

The decoder's planes are band-limited: each is an inverse DCT of a low-
frequency block of coefficients on its coder grid. Displaying one at another
size is then exact by evaluating the same cosine series on the new pixel
grid, which is what padding the spectrum and running a full inverse DCT at
display size computes. Here only the occupied part of the spectrum is
evaluated, as two separable products with cached float32 bases, in one thread
(no FFT or BLAS pools left spinning in a receiver that also plays audio).
"""
from functools import lru_cache
import math

import numpy as np
from numba import njit
from scipy.fft import dctn

# Coefficients below this fraction of a plane's largest one are rounding
# residue of the decoder's inverse DCT (float32 planes), not signal.
SUPPORT_RELATIVE_FLOOR = 1e-6


@lru_cache(maxsize=64)
def _basis(size, modes):
    """Orthonormal DCT-II synthesis basis (size x modes), float32."""
    samples = np.arange(size, dtype=np.float64)[:, None]
    orders = np.arange(modes, dtype=np.float64)[None, :]
    basis = np.cos(np.pi*(2*samples+1)*orders/(2*size))*math.sqrt(2.0/size)
    basis[:, 0] /= math.sqrt(2.0)
    basis = np.ascontiguousarray(basis, dtype=np.float32)
    basis.setflags(write=False)
    return basis


@lru_cache(maxsize=64)
def _basis_rows(size, modes):
    rows = np.ascontiguousarray(_basis(size, modes).T)
    rows.setflags(write=False)
    return rows


@njit(cache=True, fastmath=True, nogil=True)
def _synthesize(left, coefficients, right_rows):
    """``left @ coefficients @ right_rows`` for a tall/wide output.

    ``left`` is H x r, ``coefficients`` r x c, ``right_rows`` c x W. The
    output row loop is blocked four modes at a time so each output row is
    read and written c/4 times instead of c.
    """
    height = left.shape[0]
    rows = coefficients.shape[0]
    cols = coefficients.shape[1]
    width = right_rows.shape[1]
    partial = np.zeros((height, cols), dtype=np.float32)
    for y in range(height):
        for u in range(rows):
            weight = left[y, u]
            for v in range(cols):
                partial[y, v] += weight*coefficients[u, v]
    out = np.zeros((height, width), dtype=np.float32)
    for y in range(height):
        row = out[y]
        v = 0
        while v+4 <= cols:
            a0 = partial[y, v]
            a1 = partial[y, v+1]
            a2 = partial[y, v+2]
            a3 = partial[y, v+3]
            r0 = right_rows[v]
            r1 = right_rows[v+1]
            r2 = right_rows[v+2]
            r3 = right_rows[v+3]
            for x in range(width):
                row[x] += a0*r0[x]+a1*r1[x]+a2*r2[x]+a3*r3[x]
            v += 4
        while v < cols:
            weight = partial[y, v]
            basis_row = right_rows[v]
            for x in range(width):
                row[x] += weight*basis_row[x]
            v += 1
    return out


def spectral_support(coefficients, floor=SUPPORT_RELATIVE_FLOOR):
    """Rows and columns that hold every coefficient above the floor."""
    magnitude = np.abs(coefficients)
    peak = float(magnitude.max(initial=0.0))
    if peak == 0.0:
        return 1, 1
    occupied = magnitude > peak*floor
    rows = int(np.flatnonzero(occupied.any(axis=1))[-1])+1
    cols = int(np.flatnonzero(occupied.any(axis=0))[-1])+1
    return rows, cols


def reconstruct_plane(plane, target_shape):
    """Evaluate a decoded plane's DCT series on a target_shape pixel grid.

    Equivalent to zero-padding (or truncating) the plane's orthonormal DCT
    to the target size, scaling by sqrt(target area / source area) and
    inverse transforming there.
    """
    plane = np.asarray(plane, dtype=np.float64)
    if plane.ndim != 2 or min(plane.shape) <= 0:
        raise ValueError('DCT reconstruction needs a non-empty 2-D plane')
    target_height, target_width = (int(value) for value in target_shape)
    if target_height <= 0 or target_width <= 0:
        raise ValueError('target dimensions must be positive')
    source_height, source_width = plane.shape
    coefficients = dctn(plane, norm='ortho')
    rows, cols = spectral_support(coefficients)
    rows = min(rows, target_height)
    cols = min(cols, target_width)
    block = coefficients[:rows, :cols]*math.sqrt(
        (target_height*target_width)/(source_height*source_width))
    return _synthesize(_basis(target_height, rows),
                       np.ascontiguousarray(block, dtype=np.float32),
                       _basis_rows(target_width, cols))


def viewport_shapes(plane_shapes, viewport_size):
    """Luma at the viewport's (width, height); chroma at the same scale."""
    target_width, target_height = (int(value) for value in viewport_size)
    if target_width <= 0 or target_height <= 0:
        raise ValueError('viewport dimensions must be positive')
    luma_height, luma_width = plane_shapes[0]
    scale_x = target_width/luma_width
    scale_y = target_height/luma_height
    return ((target_height, target_width),) + tuple(
        (max(1, round(rows*scale_y)), max(1, round(cols*scale_x)))
        for rows, cols in plane_shapes[1:])


def reconstruct_planes(planes, viewport_size):
    """Decoded Y[/Cb/Cr] planes straight to a viewport-size display."""
    shapes = tuple(np.shape(plane) for plane in planes)
    return tuple(reconstruct_plane(plane, shape) for plane, shape in zip(
        planes, viewport_shapes(shapes, viewport_size)))


# ------------------------------------------------------------- edge-consistent
# Consistent reconstruction (Gerchberg-Papoulis with a total-variation prior):
# the receiver looks for the picture with the sharpest, flattest regions whose
# DCT still equals exactly the coefficients it received. It alternates a short
# Chambolle TV (ROF) denoise, the black/white clip, and putting the received
# coefficients back, on a working grid EDGE_FACTOR times the coder grid. The
# transmitted information is unchanged; the missing high frequencies are filled
# in to suit flat-shaded pictures (ringing and the dotted ripple go away, edges
# sharpen). The default works on the coder grid itself with four rounds (about
# 1.4 ms per picture); EDGE_HIGH works at twice the grid (about 4 ms) for a
# little more. Real modem with luma adjustment, clean channel, SSIMULACRA2 over
# no reconstruction (cartoon / robot+test card / photos): default +2.7 / +1.2
# / +1.6, high +3.1 / +1.4 / +2.7.
EDGE_FACTOR = 1
EDGE_ROUNDS = 4
EDGE_HIGH = {'factor': 2, 'rounds': 4}
# Chambolle iterations per round (warm-started). Two measure the same as ten
# through the real modem (clean cartoon / robot+card / photos +2.2 / +1.2 /
# +2.3 against +2.2 / +1.2 / +1.9 at 75 % strength) for a third less time.
EDGE_INNER = 2
EDGE_WEIGHT = .05               # TV strength (code units; values span [-1, 1])


@njit(cache=True, fastmath=True, nogil=True)
def _divergence(px, py, out):
    """out = div p (backward differences; Chambolle's discretisation)."""
    height, width = px.shape
    for i in range(height):
        out[i, 0] = -px[i, 0]
        for j in range(1, width):
            out[i, j] = px[i, j-1]-px[i, j]
    for j in range(width):
        out[0, j] -= py[0, j]
    for i in range(1, height):
        for j in range(width):
            out[i, j] += py[i-1, j]-py[i, j]


@njit(cache=True, fastmath=True, nogil=True)
def _edge_rounds(x, values, known, analysis, synthesis_rows, rounds, inner,
                 weight, trust, room):
    """In place: ``rounds`` of TV prox + clip + data consistency on ``x``.

    ``analysis`` is H x r (orthonormal DCT basis columns), ``synthesis_rows``
    c x W; ``values``/``known`` are the r x c received block (scaled to the
    working grid) and its mask. ``room`` (r x c) is how far each received
    coefficient may lie from its value: 0 puts it back exactly, more lets
    the picture keep any value within that distance. Single thread; inner
    loops run over contiguous rows so they vectorise.
    """
    height, width = x.shape
    rows, cols = values.shape
    px = np.zeros((height, width), dtype=np.float32)
    py = np.zeros((height, width), dtype=np.float32)
    out = np.empty((height, width), dtype=np.float32)
    gx = np.empty(width, dtype=np.float32)
    gy = np.empty(width, dtype=np.float32)
    partial = np.empty((height, cols), dtype=np.float32)
    coefficients = np.empty((rows, cols), dtype=np.float32)
    tau = np.float32(.25)
    scale = tau/np.float32(weight)
    one = np.float32(1)
    for _ in range(rounds):
        # Chambolle's dual iteration for min |grad u| + |u - x|^2/(2 weight).
        for _ in range(inner):
            _divergence(px, py, out)
            for i in range(height):
                for j in range(width):
                    out[i, j] += x[i, j]
            for i in range(height):
                for j in range(width-1):
                    gx[j] = out[i, j+1]-out[i, j]
                gx[width-1] = 0
                if i < height-1:
                    for j in range(width):
                        gy[j] = out[i+1, j]-out[i, j]
                else:
                    for j in range(width):
                        gy[j] = 0
                for j in range(width):
                    norm = one+scale*np.sqrt(gx[j]*gx[j]+gy[j]*gy[j])
                    px[i, j] = (px[i, j]-tau*gx[j])/norm
                    py[i, j] = (py[i, j]-tau*gy[j])/norm
        # The TV result, clipped to the displayable range.
        _divergence(px, py, out)
        for i in range(height):
            for j in range(width):
                x[i, j] = min(one, max(-one, x[i, j]+out[i, j]))
        # Put the received coefficients back: x += synthesis(known * (values -
        # analysis(x))).
        for i in range(height):
            row = x[i]
            for v in range(cols):
                basis_row = synthesis_rows[v]
                total = np.float32(0)
                for j in range(width):
                    total += row[j]*basis_row[j]
                partial[i, v] = total
        coefficients[:, :] = 0
        for i in range(height):
            for u in range(rows):
                a = analysis[i, u]
                for v in range(cols):
                    coefficients[u, v] += a*partial[i, v]
        for u in range(rows):
            for v in range(cols):
                if known[u, v]:
                    # Back inside what was received: exactly the value, or
                    # the nearest point of its bounds.
                    seen = coefficients[u, v]
                    target = min(max(seen, values[u, v]-room[u, v]),
                                 values[u, v]+room[u, v])
                    coefficients[u, v] = np.float32(trust)*(target-seen)
                else:
                    coefficients[u, v] = np.float32(0)
        partial[:, :] = 0
        for i in range(height):
            for u in range(rows):
                a = analysis[i, u]
                for v in range(cols):
                    partial[i, v] += a*coefficients[u, v]
        for i in range(height):
            row = x[i]
            for v in range(cols):
                weight_v = partial[i, v]
                basis_row = synthesis_rows[v]
                for j in range(width):
                    row[j] += weight_v*basis_row[j]
    return x


def edge_consistent_plane(plane, factor=EDGE_FACTOR, rounds=EDGE_ROUNDS,
                          inner=EDGE_INNER, weight=EDGE_WEIGHT, trust=1.0,
                          strength=1.0, room=None):
    """A decoded plane rebuilt ``factor`` times larger by consistent
    reconstruction (see above). The received coefficients are those of the
    plane's spectral support above the rounding floor; their values are kept
    exactly. ``strength`` (0 to 1) mixes the result with the plain
    reconstruction: both hold the received coefficients, so every mix does
    too; lower values keep more of the natural texture and more of the
    ripple. Returns float32, ``factor`` x the plane's shape.

    ``room`` is what the decoder knows about the plane's coefficients, when
    it says: a plane-shaped array, infinite where a coefficient was not
    sent and otherwise how far the true value may lie from the decoded one
    (0: exact; half a stair for a nested-fold host). Which coefficients
    were received is then taken from it instead of guessed from the values,
    and each is held within its bounds instead of at its value."""
    plane = np.asarray(plane, dtype=np.float64)
    if plane.ndim != 2 or min(plane.shape) <= 0:
        raise ValueError('edge reconstruction needs a non-empty 2-D plane')
    factor = int(factor)
    if factor < 1:
        raise ValueError('edge reconstruction factor must be at least 1')
    source_height, source_width = plane.shape
    height, width = source_height*factor, source_width*factor
    coefficients = dctn(plane, norm='ortho')
    if room is not None:
        room = np.asarray(room, dtype=np.float64)
        if room.shape != plane.shape:
            raise ValueError('coefficient room must have the plane\'s shape')
        sent = np.isfinite(room)
        rows = int(np.nonzero(sent.any(axis=1))[0][-1])+1 if sent.any() else 1
        cols = int(np.nonzero(sent.any(axis=0))[0][-1])+1 if sent.any() else 1
        block = coefficients[:rows, :cols]
        known = np.ascontiguousarray(sent[:rows, :cols])
        bounds = np.ascontiguousarray(
            np.where(known, room[:rows, :cols], 0.0)*factor, dtype=np.float32)
    else:
        rows, cols = spectral_support(coefficients)
        block = coefficients[:rows, :cols]
        peak = float(np.abs(coefficients).max(initial=0.0))
        known = np.abs(block) > peak*SUPPORT_RELATIVE_FLOOR
        bounds = np.zeros((rows, cols), dtype=np.float32)
    values = np.ascontiguousarray(block*factor, dtype=np.float32)
    analysis = _basis(height, rows)
    synthesis_rows = _basis_rows(width, cols)
    strength = float(strength)
    if not 0.0 <= strength <= 1.0:
        raise ValueError('edge reconstruction strength must be from 0 to 1')
    x = _synthesize(analysis, np.where(known, values, 0).astype(np.float32),
                    synthesis_rows)
    if strength == 0.0:
        return x
    plain = x.copy() if strength < 1.0 else None
    x = _edge_rounds(x, values, np.ascontiguousarray(known), analysis,
                     synthesis_rows, int(rounds), int(inner), float(weight),
                     float(trust), bounds)
    if plain is not None:
        x *= np.float32(strength)
        plain *= np.float32(1.0-strength)
        x += plain
    return x



# ---------------------------------------------------------------- guided chroma
# The wire carries colour at half the luma resolution per axis, so colour
# edges are twice as soft as the brightness edges they belong to and bleed
# across them. Within a small neighbourhood colour is close to a linear
# function of brightness, and that relation can be measured in the band both
# planes were sent in. Applying it to the full-band luma predicts the chroma
# detail that was not sent (the guided filter used for joint upsampling; the
# same idea as a codec's chroma-from-luma, here without side information).
# Every received chroma coefficient is then put back exactly, so only
# coefficients the wire did not carry are filled in. The slope is scaled by
# the local squared correlation of the two planes: where colour does not
# follow brightness (fabric folds, shading) nothing is carried over, which
# is what keeps photographs from getting worse. Clean channel, nine pictures
# at their own layouts, against the source: aspect-fold-500 chroma PSNR
# +0.43 dB (every picture +0.1 to +1.4), mean CIEDE2000 -0.22;
# aspect-mono-500 +0.28 dB (+0.03 to +1.2), -0.20. About 2 ms.
CHROMA_RADIUS = 3               # neighbourhood half-width, luma grid samples
CHROMA_EPSILON = 3e-3           # luma variance (code units^2) below which
#                                 the relation is not trusted
CHROMA_GAIN_LIMIT = .5          # largest |d chroma / d luma|


def guided_chroma_plane(luma, chroma, radius=CHROMA_RADIUS,
                        epsilon=CHROMA_EPSILON, gain_limit=CHROMA_GAIN_LIMIT):
    """A decoded chroma plane rebuilt on the luma plane's grid with detail
    predicted from luma. Its DCT equals the received chroma coefficients
    wherever the wire carried one. Returns float32, the luma plane's shape."""
    from scipy.fft import idctn
    from scipy.ndimage import uniform_filter
    luma = np.asarray(luma, dtype=np.float64)
    chroma = np.asarray(chroma, dtype=np.float64)
    if luma.ndim != 2 or chroma.ndim != 2 or min(chroma.shape) <= 0:
        raise ValueError('guided chroma needs non-empty 2-D planes')
    height, width = luma.shape
    if chroma.shape[0] > height or chroma.shape[1] > width:
        raise ValueError('the chroma plane must not exceed the luma plane')
    coefficients = dctn(chroma, norm='ortho')
    rows, cols = spectral_support(coefficients)
    peak = float(np.abs(coefficients).max(initial=0.0))
    known = np.zeros((height, width), dtype=bool)
    known[:rows, :cols] = (np.abs(coefficients[:rows, :cols]) >
                           peak*SUPPORT_RELATIVE_FLOOR)
    received = np.zeros((height, width))
    received[:rows, :cols] = coefficients[:rows, :cols]*math.sqrt(
        (height*width)/(chroma.shape[0]*chroma.shape[1]))
    received[~known] = 0.0
    plain = idctn(received, norm='ortho')
    # Luma limited to the same coefficients: the band both planes share.
    luma_band = idctn(np.where(known, dctn(luma, norm='ortho'), 0.0),
                      norm='ortho')
    size = 2*int(radius)+1

    def box(values):
        return uniform_filter(values, size, mode='nearest')

    mean_luma, mean_chroma = box(luma_band), box(plain)
    variance = box(luma_band*luma_band)-mean_luma*mean_luma
    chroma_variance = box(plain*plain)-mean_chroma*mean_chroma
    covariance = box(luma_band*plain)-mean_luma*mean_chroma
    slope = np.clip(covariance/(variance+epsilon), -gain_limit, gain_limit)
    slope *= np.clip(covariance*covariance/(
        (variance+epsilon)*(chroma_variance+.25*epsilon)), 0.0, 1.0)
    # Only the luma detail outside the shared band is carried over, and only
    # into coefficients the wire did not carry: flat luma, or a slope of
    # zero, returns the chroma exactly as sent.
    detail = dctn(box(slope)*(luma-luma_band), norm='ortho')
    detail[known] = 0.0
    return np.ascontiguousarray(plain+idctn(detail, norm='ortho'),
                                dtype=np.float32)
