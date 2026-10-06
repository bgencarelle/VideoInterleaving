"""Noise-matched nested fold (Shannon-Kotel'nikov staircase) for V7 luma slots.

This is the repaired form of ``tools/v7_fold_surface.py``.  The earlier
surface made four design errors, each measured in ``docs/V7_SK_VERDICT.md``:

* the *new* (least important) coordinate was the coarse strip index and the
  *existing* coordinate was squeezed into the strip, so every added layer cost
  the already-carried detail 20*log10(strips) dB;
* strips were reconstructed at compressed-domain midpoints instead of source
  centroids, so a 3-strip extra coordinate decoded to 0 or +-5.8 sigma;
* strip width was unrelated to the slot's measured noise and had no guard
  band, so layers were either far too coarse or slipped;
* it nested inside the 500 production guests (already the fine half of the
  production fold) instead of using the 1,420 unfolded luma slots.

Here one wire slot carries a host coefficient ``a`` (unit variance) on a
uniform staircase whose step is ``kappa`` times that slot's own noise, and up
to two guests inside the stair:

    level 0   s = a
    level 1   s = step*round(a/step) + alpha*step*c(b1)
    level 2   s = step*round(a/step) + alpha*step*(m*delta + alpha2*delta*c(b2))
              m = round(c(b1)/delta),  delta = 2/L

``c`` is a training-fitted compander into [-1, 1].  ``alpha = 1/2 - z/kappa``
leaves a guard of ``z`` noise standard deviations each side of a stair.  All
reconstruction points are conditional means fitted on training pictures only.
There is no per-frame side information: both ends hold one frozen table.

Numerical kernels are Numba-compiled; Python is orchestration.
"""
import hashlib
import math

import numpy as np
from numba import njit

LUT = 513                      # receiver conditional-mean table size
COMPANDER = 1025               # compander table size
B_LIMIT = 16.0                 # compander domain, guest standard deviations


@njit(cache=True)
def dct_matrix(n):
    """Orthonormal DCT-II analysis matrix (rows are basis functions)."""
    out = np.empty((n, n))
    for k in range(n):
        scale = math.sqrt((1.0 if k == 0 else 2.0)/n)
        for i in range(n):
            out[k, i] = scale*math.cos(math.pi*(i+.5)*k/n)
    return out


@njit(cache=True)
def area_matrix(source, target):
    """Exact box (area) resampling weights, target x source."""
    out = np.zeros((target, source))
    ratio = source/target
    for t in range(target):
        lo, hi = t*ratio, (t+1)*ratio
        first, last = int(math.floor(lo)), min(source, int(math.ceil(hi)))
        for s in range(first, last):
            overlap = min(hi, s+1.0)-max(lo, float(s))
            if overlap > 0:
                out[t, s] = overlap/ratio
    return out


@njit(cache=True)
def separable(left, plane, right):
    return np.dot(np.dot(left, plane), right.T)


@njit(cache=True)
def fit_compander(samples, power):
    """Monotone map of [-B_LIMIT, B_LIMIT] onto [-1, 1], slope ~ pdf**power.

    power = 1/3 minimises additive-noise error referred to the source
    (Bennett); the fitted table is symmetric and strictly increasing.
    Returns (forward table on a uniform |b| grid, mean square of c(b)).
    """
    half = COMPANDER
    density = np.zeros(half)
    for x in samples:
        position = min(abs(x), B_LIMIT)/B_LIMIT*(half-1)
        i = int(position)
        if i >= half-1:
            density[half-1] += 1.0
        else:
            f = position-i
            density[i] += 1.0-f
            density[i+1] += f
    # Light smoothing and a floor keep the inverse well conditioned in the
    # sparse tail without inventing probability mass near zero.
    smooth = density.copy()
    for _ in range(8):
        previous = smooth.copy()
        for i in range(half):
            lo, hi = max(i-1, 0), min(i+1, half-1)
            smooth[i] = .25*previous[lo]+.5*previous[i]+.25*previous[hi]
    floor = 1e-4*smooth.max()
    table = np.zeros(half)
    for i in range(1, half):
        slope = .5*((smooth[i-1]+floor)**power+(smooth[i]+floor)**power)
        table[i] = table[i-1]+slope
    table /= table[half-1]
    total = 0.0
    for x in samples:
        total += compress(table, x)**2
    return table, total/max(len(samples), 1)


@njit(cache=True)
def compress(table, x):
    half = len(table)
    position = min(abs(x), B_LIMIT)/B_LIMIT*(half-1)
    i = min(int(position), half-2)
    f = position-i
    value = table[i]*(1.0-f)+table[i+1]*f
    return value if x >= 0 else -value


@njit(cache=True)
def laplace_centroid(index, step):
    """Mean of a unit-variance Laplacian inside staircase cell ``index``."""
    if index == 0:
        return 0.0
    beta = math.sqrt(.5)
    lo = (abs(index)-.5)*step
    decay = math.exp(-step/beta)
    value = lo+beta-step*decay/(1.0-decay)
    return value if index > 0 else -value


@njit(cache=True)
def symbol_power(step):
    """Mean square of step*round(a/step) for a unit-variance Laplacian."""
    beta = math.sqrt(.5)
    total = 0.0
    for k in range(1, 4000):
        lo = (k-.5)*step
        if lo > 40.0:
            break
        mass = .5*(math.exp(-lo/beta)-math.exp(-(lo+step)/beta))
        total += 2.0*mass*(k*step)**2
    return total


@njit(cache=True)
def encode_slot(a, b1, b2, level, step, alpha, levels, alpha2, table, mid_gain=1.0):
    """One wire symbol in host standard deviations, before power scaling."""
    if level == 0:
        return a
    stair = step*np.round(a/step)
    c1 = compress(table, b1)
    if level == 1:
        return stair+alpha*step*c1
    delta = 2.0/levels
    limit = (levels-1)//2
    m = min(max(np.round(c1*mid_gain/delta), -limit), limit)
    return stair+alpha*step*(m*delta+alpha2*delta*compress(table, b2))


@njit(cache=True)
def split_slot(received, level, step, alpha, levels, alpha2):
    """(stair index, mid index, fine coordinate in [-1, 1]) of one symbol."""
    k = np.round(received/step)
    t = min(max((received-k*step)/(alpha*step), -1.0), 1.0)
    if level == 1:
        return k, 0.0, t
    delta = 2.0/levels
    limit = (levels-1)//2
    m = min(max(np.round(t/delta), -limit), limit)
    fine = min(max((t-m*delta)/(alpha2*delta), -1.0), 1.0)
    return k, m, fine


@njit(cache=True)
def lut_index(t):
    return int(min(max(np.round((t+1.0)*.5*(LUT-1)), 0), LUT-1))


@njit(cache=True)
def encode_frame(a, b1, b2, level, step, scale, alpha, levels, alpha2, table, mid_gain=1.0):
    """Wire-normalised symbols (each slot at unit design power)."""
    out = np.empty(len(a))
    for i in range(len(a)):
        out[i] = scale[i]*encode_slot(a[i], b1[i], b2[i], level[i], step[i],
                                      alpha[level[i]], levels, alpha2, table, mid_gain)
    return out


@njit(cache=True)
def decode_frame(received, level, step, scale, alpha, levels, alpha2,
                 shrink, lut1, mid, lut2):
    """Conditional-mean host and guests from received wire symbols."""
    n = len(received)
    a, b1, b2 = np.zeros(n), np.zeros(n), np.zeros(n)
    limit = (levels-1)//2
    for i in range(n):
        value = received[i]/scale[i]
        if level[i] == 0:
            a[i] = value*shrink[i]
            continue
        k, m, fine = split_slot(value, level[i], step[i], alpha[level[i]],
                                levels, alpha2)
        a[i] = laplace_centroid(int(k), step[i])
        if level[i] == 1:
            b1[i] = lut1[lut_index(fine)]
        else:
            b1[i] = mid[int(m)+limit]
            b2[i] = lut2[lut_index(fine)]
    return a, b1, b2


@njit(cache=True)
def fit_tables(b1, b2, level, step, scale, noise, alpha, levels, alpha2,
               table, draws, seed, mid_gain=1.0):
    """Receiver conditional means at the design noise, training data only.

    ``b1``/``b2`` are frames x slots.  Noise is simulated on the fine
    coordinate of every training guest; the host stair is taken as decoded
    because slips are rare by construction and are scored, not hidden.
    """
    np.random.seed(seed)
    limit = (levels-1)//2
    sums = np.zeros((2, LUT))
    counts = np.zeros((2, LUT))
    mid_sum, mid_count = np.zeros(levels), np.zeros(levels)
    frames, slots = b1.shape
    for f in range(frames):
        for i in range(slots):
            if level[i] == 0:
                continue
            clean = encode_slot(0.0, b1[f, i], b2[f, i], level[i], step[i],
                                alpha[level[i]], levels, alpha2, table, mid_gain)
            for _ in range(draws):
                value = clean+np.random.standard_normal()*noise[i]/scale[i]
                # Slips of the host stair wrap the residual: model them.
                _, m, fine = split_slot(value, level[i], step[i],
                                        alpha[level[i]], levels, alpha2)
                j = lut_index(fine)
                if level[i] == 1:
                    sums[0, j] += b1[f, i]
                    counts[0, j] += 1.0
                else:
                    mid_sum[int(m)+limit] += b1[f, i]
                    mid_count[int(m)+limit] += 1.0
                    sums[1, j] += b2[f, i]
                    counts[1, j] += 1.0
    out = np.zeros((2, LUT))
    for r in range(2):
        # Kernel-smoothed conditional mean; empty bins inherit neighbours.
        width = 6
        for j in range(LUT):
            top, bottom = 0.0, 0.0
            for d in range(-3*width, 3*width+1):
                q = j+d
                if 0 <= q < LUT:
                    weight = math.exp(-.5*(d/width)**2)
                    top += weight*sums[r, q]
                    bottom += weight*counts[r, q]
            out[r, j] = top/bottom if bottom > 0 else 0.0
        for j in range(LUT//2+1):               # enforce odd symmetry
            value = .5*(out[r, LUT-1-j]-out[r, j])
            out[r, LUT-1-j], out[r, j] = value, -value
    centroids = np.zeros(levels)
    for m in range(levels):
        centroids[m] = mid_sum[m]/mid_count[m] if mid_count[m] > 0 else 0.0
    for m in range(limit+1):
        value = .5*(centroids[levels-1-m]-centroids[m])
        centroids[levels-1-m], centroids[m] = value, -value
    return out[0], centroids, out[1]


@njit(cache=True)
def reverse_water_fill(variance, bits):
    """Gaussian rate-distortion: (water level, distortion, active count)."""
    lam = np.sort(variance)[::-1]
    n = len(lam)
    logs = np.log2(lam)
    prefix = 0.0
    theta, active = lam[0], 0
    for k in range(1, n+1):
        prefix += logs[k-1]
        # all of the first k active at level theta: bits = .5*(prefix-k*log2 theta)
        candidate = 2.0**((prefix-2.0*bits)/k)
        if candidate <= lam[k-1] and (k == n or candidate >= lam[k]):
            theta, active = candidate, k
            break
        if k == n:
            theta, active = candidate, n
    distortion = 0.0
    for i in range(n):
        distortion += min(theta, lam[i])
    return theta, distortion, active


class NestedFold:
    """Frozen table for a nested fold over ``slots`` wire slots.

    ``host_sd``: standard deviation of each host coefficient (most important
    slot first).  ``noise``: design noise of each slot in units of that slot's
    design signal standard deviation.  ``linear`` slots stay unfolded, the
    next ``slots-linear-triple`` carry one guest and the last ``triple``
    carry two.  Guests are numbered in the order they should be filled:
    ``guest_slot[j]``, ``guest_layer[j]``.
    """

    def __init__(self, noise, linear, triple, kappa, guard=3.0, levels=3,
                 kappa2=None, guard2=None, compander_power=1/3, mid_gain=1.0):
        noise = np.ascontiguousarray(noise, float)
        slots = len(noise)
        if not 0 <= linear <= slots or not 0 <= triple <= slots-linear:
            raise ValueError('zone sizes outside the slot count')
        if levels % 2 != 1 or levels < 3:
            raise ValueError('mid levels must be odd and at least three')
        self.slots, self.linear, self.triple = slots, int(linear), int(triple)
        self.kappa, self.guard, self.levels = float(kappa), float(guard), int(levels)
        self.kappa2 = float(kappa2 if kappa2 is not None else kappa)
        self.guard2 = float(guard2 if guard2 is not None else guard)
        self.compander_power = float(compander_power)
        self.mid_gain = float(mid_gain)
        self.noise = noise
        self.level = np.zeros(slots, np.int64)
        self.level[linear:] = 1
        if triple:
            self.level[slots-triple:] = 2
        # Triple stairs must contain `levels` mid cells, each with its own
        # guard: the stair is kappa2 noise units per mid cell.
        per_noise = np.where(self.level == 2, self.kappa2*self.levels, self.kappa)
        alpha1 = .5-self.guard/self.kappa
        alpha_triple = .5-self.guard/(self.kappa2*self.levels)
        self.alpha2 = .5-self.guard2/(self.kappa2*alpha_triple*2)
        if min(alpha1, alpha_triple, self.alpha2) <= 0:
            raise ValueError('guard leaves no room inside the stair')
        self.alpha = np.array([0., alpha1, alpha_triple])
        # Solve step and power scale together: step is in host-sd units and
        # the emitted symbol is scaled to unit design power.
        self.step = np.ones(slots)
        self.scale = np.ones(slots)
        self.shrink = 1.0/(1.0+noise*noise)
        self._per_noise = per_noise
        self.table = None

    def _solve(self, guest_power):
        for i in range(self.slots):
            if self.level[i] == 0:
                continue
            step = self._per_noise[i]*self.noise[i]
            for _ in range(40):
                inner = self.alpha[self.level[i]]*step
                power = symbol_power(step)+inner*inner*guest_power[self.level[i]]
                # scale*sqrt(power) = 1 and noise in host units = noise/scale
                updated = self._per_noise[i]*self.noise[i]*math.sqrt(max(power, 1e-12))
                if abs(updated-step) < 1e-12:
                    break
                step = updated
            self.step[i] = step
            self.scale[i] = 1.0/math.sqrt(max(power, 1e-12))

    def guest_order(self):
        """(slot, layer) of every guest, best-protected class first."""
        single = np.flatnonzero(self.level == 1)
        triple = np.flatnonzero(self.level == 2)
        slot = np.concatenate((single, triple, triple))
        layer = np.concatenate((np.zeros(len(single), np.int64),
                                np.zeros(len(triple), np.int64),
                                np.ones(len(triple), np.int64)))
        return slot, layer

    def fit(self, guests_normalised, draws=4, seed=20261006):
        """Fit compander and receiver tables.  ``guests_normalised``:
        training frames x guest count, each guest in its own standard
        deviations, columns in ``guest_order``."""
        g = np.ascontiguousarray(guests_normalised, float)
        self.table, c_power = fit_compander(g.ravel(), self.compander_power)
        delta = 2.0/self.levels
        # Mean square of the triple residual: mid cell centre plus inner part.
        mid_power = 0.0
        slot, layer = self.guest_order()
        b1, b2 = self.unpack(g)
        triple = self.level == 2
        if triple.any():
            limit = (self.levels-1)//2
            total, count = 0.0, 0
            for value, inner in zip(b1[:, triple].ravel(), b2[:, triple].ravel()):
                c1 = float(compress(self.table, value))
                m = min(max(round(c1*self.mid_gain/delta), -limit), limit)
                total += (m*delta+self.alpha2*delta*float(compress(self.table, inner)))**2
                count += 1
            mid_power = total/count
        self._solve(np.array([0., c_power, mid_power]))
        self.lut1, self.mid, self.lut2 = fit_tables(
            b1, b2, self.level, self.step, self.scale, self.noise, self.alpha,
            self.levels, self.alpha2, self.table, draws, seed, self.mid_gain)
        return self

    def unpack(self, guests):
        """Guest columns in fill order -> per-slot (b1, b2) arrays."""
        guests = np.atleast_2d(guests)
        slot, layer = self.guest_order()
        b1 = np.zeros((len(guests), self.slots))
        b2 = np.zeros((len(guests), self.slots))
        b1[:, slot[layer == 0]] = guests[:, layer == 0]
        b2[:, slot[layer == 1]] = guests[:, layer == 1]
        return b1, b2

    def pack(self, b1, b2):
        slot, layer = self.guest_order()
        return np.where(layer == 0, b1[slot], b2[slot])

    @property
    def guests(self):
        return int(np.sum(self.level == 1)+2*np.sum(self.level == 2))

    def encode(self, hosts_normalised, guests_normalised):
        b1, b2 = self.unpack(guests_normalised)
        return encode_frame(np.ascontiguousarray(hosts_normalised, float),
                            b1[0], b2[0], self.level, self.step, self.scale,
                            self.alpha, self.levels, self.alpha2, self.table, self.mid_gain)

    def decode(self, received):
        a, b1, b2 = decode_frame(np.ascontiguousarray(received, float),
                                 self.level, self.step, self.scale, self.alpha,
                                 self.levels, self.alpha2, self.shrink,
                                 self.lut1, self.mid, self.lut2)
        return a, self.pack(b1, b2)

    def digest(self):
        digest = hashlib.sha256()
        for array in (self.noise, self.level, self.step, self.scale, self.alpha,
                      self.table, self.lut1, self.mid, self.lut2):
            digest.update(np.ascontiguousarray(array).tobytes())
        digest.update(f'{self.kappa}:{self.guard}:{self.levels}:{self.kappa2}:'
                      f'{self.guard2}:{self.alpha2}:{self.mid_gain}'.encode())
        return digest.hexdigest()


# ----------------------------------------------------------- soft decoding
SOFT_POINTS = 49               # guest grid across the stair residual
SOFT_REACH = 2                 # stairs each side of the nearest one


@njit(cache=True)
def expand_table(table):
    """Inverse compander on a uniform grid of c in [0, 1] -> |b|."""
    half = len(table)
    out = np.empty(half)
    j = 0
    for i in range(half):
        c = i/(half-1)
        while j < half-2 and table[j+1] < c:
            j += 1
        span = table[j+1]-table[j]
        f = (c-table[j])/span if span > 0 else 0.0
        out[i] = (j+min(max(f, 0.0), 1.0))/(half-1)*B_LIMIT
    return out


@njit(cache=True)
def expand(inverse, c):
    half = len(inverse)
    position = min(abs(c), 1.0)*(half-1)
    i = min(int(position), half-2)
    f = position-i
    value = inverse[i]*(1.0-f)+inverse[i+1]*f
    return value if c >= 0 else -value


@njit(cache=True)
def residual_density(guests, table):
    """Training density of the compressed guest on the soft grid."""
    out = np.zeros(SOFT_POINTS)
    for b in guests:
        c = compress(table, b)
        position = (c+1.0)*.5*(SOFT_POINTS-1)
        i = min(int(position), SOFT_POINTS-2)
        f = position-i
        out[i] += 1.0-f
        out[i+1] += f
    total = out.sum()
    for i in range(SOFT_POINTS):
        out[i] = (out[i]+1e-4*total/SOFT_POINTS)/(total*(1+1e-4))
    return out


@njit(cache=True)
def soft_decode_frame(received, level, step, scale, alpha, sigma, density, inverse):
    """Posterior-mean host and guest of every slot at its own noise.

    ``sigma[i]`` is the noise the receiver measured for slot i, in units of
    that slot's design signal.  Nothing is thresholded: as noise grows the
    stair posterior spreads, the host estimate slides from the stair centroid
    to a plain shrunk reading and the guest estimate fades to zero.  Folded
    slots carry one guest (level 1).
    """
    n = len(received)
    a, b = np.zeros(n), np.zeros(n)
    beta = math.sqrt(.5)
    grid = np.empty(SOFT_POINTS)
    value_of = np.empty(SOFT_POINTS)
    for j in range(SOFT_POINTS):
        grid[j] = -1.0+2.0*j/(SOFT_POINTS-1)
        value_of[j] = expand(inverse, grid[j])
    for i in range(n):
        value = received[i]/scale[i]
        noise = max(sigma[i]/scale[i], 1e-9)
        if level[i] == 0:
            a[i] = value/(1.0+noise*noise)
            continue
        inner = alpha[1]*step[i]
        nearest = np.round(value/step[i])
        # The grid cannot resolve noise finer than its own spacing.
        spread = math.sqrt(noise*noise+(inner*2.0/(SOFT_POINTS-1))**2/3.0)
        total = host = guest = 0.0
        for d in range(-SOFT_REACH, SOFT_REACH+1):
            k = nearest+d
            lo = (abs(k)-.5)*step[i]
            if k == 0:
                prior = 1.0-math.exp(-.5*step[i]/beta)
            else:
                prior = .5*(math.exp(-lo/beta)-math.exp(-(lo+step[i])/beta))
            centroid = laplace_centroid(int(k), step[i])
            base = value-k*step[i]
            for j in range(SOFT_POINTS):
                miss = (base-inner*grid[j])/spread
                weight = prior*density[j]*math.exp(-.5*miss*miss)
                total += weight
                host += weight*centroid
                guest += weight*value_of[j]
        if total > 1e-300:
            a[i] = host/total
            b[i] = guest/total
        else:
            a[i] = value/(1.0+noise*noise)
    return a, b


@njit(cache=True)
def hard_decode_frame(received, level, step, scale, alpha, inverse):
    """Nearest-stair decode, for the sender's own trial of a level."""
    n = len(received)
    a, b = np.zeros(n), np.zeros(n)
    for i in range(n):
        value = received[i]/scale[i]
        if level[i] == 0:
            a[i] = value
            continue
        k = np.round(value/step[i])
        a[i] = laplace_centroid(int(k), step[i])
        t = min(max((value-k*step[i])/(alpha[1]*step[i]), -1.0), 1.0)
        b[i] = expand(inverse, t)
    return a, b


@njit(cache=True)
def self_convolved(density):
    """Density of the mean of two independent compressed guests."""
    out = np.zeros(SOFT_POINTS)
    for i in range(SOFT_POINTS):
        for j in range(SOFT_POINTS):
            position = .5*(i+j)
            lo = int(position)
            f = position-lo
            out[lo] += density[i]*density[j]*(1.0-f)
            if lo+1 < SOFT_POINTS:
                out[lo+1] += density[i]*density[j]*f
    return out/out.sum()


@njit(cache=True)
def soft_decode_pair(left, right, level, step, scale, alpha, sigma_left, sigma_right,
                     density, inverse):
    """Joint posterior-mean decode of two channels that carry the same hosts
    on the same stairs and different guests.

    Both readings vote on the stair; each guest is read from its own channel
    under that shared vote.  A channel that is missing or far noisier simply
    carries less weight: with one channel erased this is ``soft_decode_frame``.
    Returns (host, left guest, right guest).
    """
    n = len(left)
    a, bl, br = np.zeros(n), np.zeros(n), np.zeros(n)
    beta = math.sqrt(.5)
    grid = np.empty(SOFT_POINTS)
    value_of = np.empty(SOFT_POINTS)
    for j in range(SOFT_POINTS):
        grid[j] = -1.0+2.0*j/(SOFT_POINTS-1)
        value_of[j] = expand(inverse, grid[j])
    for i in range(n):
        vl, vr = left[i]/scale[i], right[i]/scale[i]
        nl = max(sigma_left[i]/scale[i], 1e-9)
        nr = max(sigma_right[i]/scale[i], 1e-9)
        wl, wr = 1.0/(nl*nl), 1.0/(nr*nr)
        if level[i] == 0:
            a[i] = (vl*wl+vr*wr)/(1.0+wl+wr)
            continue
        inner = alpha[1]*step[i]
        blur = (inner*2.0/(SOFT_POINTS-1))**2/3.0
        sl, sr = math.sqrt(nl*nl+blur), math.sqrt(nr*nr+blur)
        # Centre the stair search on the more trustworthy reading.
        nearest = np.round((vl if nl <= nr else vr)/step[i])
        total = host = guest_l = guest_r = 0.0
        best = -1e300
        logs = np.empty((2*SOFT_REACH+1, 3))
        for d in range(-SOFT_REACH, SOFT_REACH+1):
            k = nearest+d
            lo = (abs(k)-.5)*step[i]
            if k == 0:
                prior = 1.0-math.exp(-.5*step[i]/beta)
            else:
                prior = .5*(math.exp(-lo/beta)-math.exp(-(lo+step[i])/beta))
            like_l = part_l = like_r = part_r = 0.0
            # Likelihoods are kept relative to each channel's best point so a
            # very noisy channel cannot underflow the product.
            base_l, base_r = vl-k*step[i], vr-k*step[i]
            floor_l = floor_r = 1e300
            for j in range(SOFT_POINTS):
                ml = (base_l-inner*grid[j])/sl
                mr = (base_r-inner*grid[j])/sr
                floor_l = min(floor_l, ml*ml)
                floor_r = min(floor_r, mr*mr)
            for j in range(SOFT_POINTS):
                ml = (base_l-inner*grid[j])/sl
                mr = (base_r-inner*grid[j])/sr
                el = density[j]*math.exp(-.5*(ml*ml-floor_l))
                er = density[j]*math.exp(-.5*(mr*mr-floor_r))
                like_l += el
                part_l += el*value_of[j]
                like_r += er
                part_r += er*value_of[j]
            index = d+SOFT_REACH
            logs[index, 0] = math.log(max(prior*like_l*like_r, 1e-300))-.5*(floor_l+floor_r)
            logs[index, 1] = part_l/max(like_l, 1e-300)
            logs[index, 2] = part_r/max(like_r, 1e-300)
            best = max(best, logs[index, 0])
        for d in range(-SOFT_REACH, SOFT_REACH+1):
            index = d+SOFT_REACH
            weight = math.exp(logs[index, 0]-best)
            total += weight
            host += weight*laplace_centroid(int(nearest+d), step[i])
            guest_l += weight*logs[index, 1]
            guest_r += weight*logs[index, 2]
        a[i] = host/total
        bl[i] = guest_l/total
        br[i] = guest_r/total
    return a, bl, br


@njit(cache=True)
def stair_values(hosts, level, step):
    """Each host as a slip-free receiver decodes it (stair centroids)."""
    out = hosts.copy()
    for i in range(len(hosts)):
        if level[i] > 0:
            out[i] = laplace_centroid(int(np.round(hosts[i]/step[i])), step[i])
    return out
