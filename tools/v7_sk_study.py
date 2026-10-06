"""Does folding more picture coordinates into V7's luma slots buy resolution?

Slot-level study at the *measured* precision of the real V7 wire, with the
rate-distortion bound that no mapping can beat.  Wire confirmation of the
winning table is ``tools/v7_sk_wire.py``.

    .venv/bin/python tools/v7_sk_study.py --noise tmp/sk/wire/NOISE.npz --out tmp/sk/study

Numerical kernels are Numba-compiled (see ``tools/v7_sk_fold.py``).
"""
import argparse
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
from numba import njit
from PIL import Image

from tools import v7_sk_fold as sk

LATTICE = (96, 80)             # production luma sampling lattice
BIG = 3                        # reference lattice is BIG times finer per axis
LAYOUT = '3:4'
# Fallback wire-domain noise standard deviations (reference-face packet).
# Prefer --noise with the per-slot profile from tools/v7_sk_wire.py measure.
MEASURED_NOISE = {'clean': .0103, 'hiss-60': .0127, 'hiss-50': .0257,
                  'hiss-45': .0431, 'hiss-40': .0754}
BAND = 2.0                     # cycles per picture height
RHO_MIN, GAIN_RANGE, RESIDUAL_MAX = .8, (.5, 1.5), .5


# ------------------------------------------------------------------ kernels
@njit(cache=True)
def luma(rgb):
    h, w = rgb.shape[:2]
    out = np.empty((h, w))
    for y in range(h):
        for x in range(w):
            out[y, x] = (.299*rgb[y, x, 0]+.587*rgb[y, x, 1]+.114*rgb[y, x, 2])/255.0*2.0-1.0
    return out


@njit(cache=True)
def smooth_power(square, radius):
    """Mean of per-position mean squares in a (2r+1)^2 frequency window.
    The lowest frequencies, where the spectrum is steep, are left alone."""
    rows, cols = square.shape
    out = square.copy()
    for u in range(rows):
        for v in range(cols):
            if u < 3 and v < 3:
                continue
            total, count = 0.0, 0
            for du in range(-radius, radius+1):
                for dv in range(-radius, radius+1):
                    p, q = u+du, v+dv
                    if 0 <= p < rows and 0 <= q < cols and not (p < 3 and q < 3):
                        total += square[p, q]
                        count += 1
            out[u, v] = total/count
    return out


@njit(cache=True)
def production_fold_encode(hosts, guests, step, limit, mu):
    """Production companded fold symbols at unit design power (folding.py)."""
    amp = .4*step
    power = 1.0+step*step/12.0+amp*amp*.12
    out = np.empty(len(hosts))
    for i in range(len(hosts)):
        c = math.copysign(math.log1p(mu*min(abs(guests[i]), limit)/limit)/math.log1p(mu), guests[i])
        out[i] = (step*np.round(hosts[i]/step)+amp*c)/math.sqrt(power)
    return out


@njit(cache=True)
def production_fold_decode(received, noise, step, limit, mu):
    amp = .4*step
    power = 1.0+step*step/12.0+amp*amp*.12
    carried = amp*amp*.12
    n = len(received)
    out_h, out_g = np.empty(n), np.empty(n)
    for i in range(n):
        symbol = received[i]*math.sqrt(power)
        sigma = noise[i]*math.sqrt(power)
        k = np.round(symbol/step)
        residual = min(max((symbol-step*k)/amp, -1.0), 1.0)*carried/(carried+sigma*sigma)
        out_h[i] = step*k
        out_g[i] = math.copysign(limit*math.expm1(abs(residual)*math.log1p(mu))/mu, residual)
    return out_h, out_g


@njit(cache=True)
def band_statistics(truth, decoded, band, bands):
    """Per radial band: correlation, signed transfer, unexplained residual."""
    tt, td, dd = np.zeros(bands), np.zeros(bands), np.zeros(bands)
    for i in range(len(truth)):
        b = band[i]
        if b < bands:
            tt[b] += truth[i]*truth[i]
            td[b] += truth[i]*decoded[i]
            dd[b] += decoded[i]*decoded[i]
    rho, gain, residual = np.zeros(bands), np.zeros(bands), np.ones(bands)
    for b in range(bands):
        if tt[b] > 0:
            gain[b] = td[b]/tt[b]
            if dd[b] > 0:
                rho[b] = td[b]/math.sqrt(tt[b]*dd[b])
            residual[b] = math.sqrt(max(dd[b]-gain[b]*gain[b]*tt[b], 0.0)/tt[b])
    return rho, gain, residual


@njit(cache=True)
def equivalent_count(sorted_energy, error):
    """Coefficients an ideal noise-free radial truncation needs for this error."""
    tail = 0.0
    for k in range(len(sorted_energy)-1, -1, -1):
        if tail+sorted_energy[k] > error:
            # linear interpolation inside coefficient k
            return k+1-(error-tail)/sorted_energy[k]
        tail += sorted_energy[k]
    return 0.0


@njit(cache=True)
def water_filled_bits(noise_power, total_power):
    """Capacity of parallel slots if the same total power were re-divided."""
    lo, hi = 0.0, noise_power.max()+total_power
    for _ in range(200):
        level = .5*(lo+hi)
        used = 0.0
        for n in noise_power:
            used += max(level-n, 0.0)
        if used > total_power:
            hi = level
        else:
            lo = level
    bits = 0.0
    for n in noise_power:
        if lo > n:
            bits += .5*math.log2(lo/n)
    return bits


@njit(cache=True)
def entropy_power_ratio(samples):
    """Entropy power of unit-variance samples relative to a Gaussian."""
    bins, limit = 4001, 40.0
    hist = np.zeros(bins)
    total, square = 0, 0.0
    for x in samples:
        square += x*x
        j = int((min(max(x, -limit), limit)+limit)/(2*limit)*(bins-1)+.5)
        hist[j] += 1.0
        total += 1
    width = 2*limit/(bins-1)
    h = 0.0
    for j in range(bins):
        if hist[j] > 0:
            p = hist[j]/total
            h -= p*math.log(p/width)
    return math.exp(2*h)/(2*math.pi*math.e)/(square/total)


# ------------------------------------------------------------------ geometry
class Geometry:
    """A profile's luma slots, radial guest order and reference lattice.

    ``aspect-fold-500`` and ``aspect-mono-500`` are the production profiles;
    ``stereo-slices`` is one channel of the two-channel profile, whose
    signature slots lie outside its 988 luma slots.
    """

    def __init__(self, profile='aspect-fold-500', layout=LAYOUT):
        from tools.v7_analog_frame import ProductionFrameWire
        from aspect_fold import layout_size
        self.profile, self.layout = profile, layout
        self.mono = profile == 'aspect-mono-500'
        self.slices = profile == 'stereo-slices'
        rows, cols = LATTICE
        self.rows, self.cols = rows, cols
        if self.slices:
            from types import SimpleNamespace
            from slice_wire import SliceWire
            from tools.v7_color_wire import TARGET
            from animation_modem import v7
            self.base = v7.load_model(TARGET, 'box')
            self.wire = SliceWire(layout)
            model = self.wire.model_for(self.base, layout)
            slots = self.wire.slots(layout)
            self.slots = slots
            part = slots.parts[0]
            self.model_index = np.arange(part.start, part.stop)
            self.host_position = np.asarray(slots.base[0])
            self.power = (np.asarray(model.gain)**2*np.asarray(model.lam))[self.model_index]
            self.fold_hosts = np.zeros(0, int)
            self.fold_guests = np.zeros(0, int)
            self.signature = 0
            codec = SimpleNamespace(pattern=np.zeros(0))
        else:
            wire = ProductionFrameWire(profile, layout)
            model, codec = wire.model, wire.codec
            # Importance order of the profile: the mono wire re-ranks with
            # chroma weighted up, so its sent set and order differ from stereo's.
            order = (np.asarray(wire.wire.coefficient_order(model)) if self.mono
                     else np.asarray(model.order))
            rank = np.empty(len(order), int)
            rank[order] = np.arange(len(order))
            sent = np.unique(np.concatenate(model.rank_tables))
            sent = sent[sent >= 0]
            luma_slots = sent[np.asarray(model.plane)[sent] == 0]
            self.model_index = luma_slots[np.argsort(rank[luma_slots])]   # important first
            self.host_position = np.asarray(model.coder.positions)[self.model_index]
            self.power = (np.asarray(model.gain)**2*np.asarray(model.lam))[self.model_index]
            slot_of = {int(index): slot for slot, index in enumerate(self.model_index)}
            self.fold_hosts = np.array([slot_of[int(index)] for index in codec.hosts])
            self.fold_guests = np.asarray(codec.guests)
            self.signature = int(codec.signature)
            self.wire = wire
        width, height = layout_size(layout)
        self.aspect = (width, height)
        u, v = np.mgrid[:rows, :cols]
        key = (u*u*width*width+v*v*height*height).ravel()
        radial = np.lexsort((v.ravel(), u.ravel(), key))
        kept = np.zeros(rows*cols, bool)
        kept[self.host_position] = True
        self.guest_position = radial[~kept[radial]]
        self.radial = radial
        # Reference lattice (BIG x finer); positions of the small lattice in it.
        self.big = (rows*BIG, cols*BIG)
        U, V = np.mgrid[:self.big[0], :self.big[1]]
        self.big_frequency = np.sqrt(U*U*width*width+V*V*height*height).ravel()/(2*width)
        self.big_radial = np.argsort(self.big_frequency, kind='stable')
        self.embed = (u*self.big[1]+v).ravel()
        self.band = np.int64(self.big_frequency//BAND)
        self.model, self.codec = model, codec


def portrait(path, aspect=(3, 4)):
    """Centre crop to ``aspect`` (width, height) as uint8 RGB."""
    with Image.open(path) as image:
        rgb = np.asarray(image.convert('RGB'))
    h, w = rgb.shape[:2]
    width, height = aspect
    scale = min(w//width, h//height)
    target, tall = scale*width, scale*height
    x, y = (w-target)//2, (h-tall)//2
    return np.ascontiguousarray(rgb[y:y+tall, x:x+target])


class Analyzer:
    def __init__(self, shape):
        self.cache, self.shape = {}, shape

    def __call__(self, rgb):
        plane = luma(rgb)
        key = plane.shape
        if key not in self.cache:
            rows, cols = self.shape
            self.cache[key] = (sk.dct_matrix(rows)@sk.area_matrix(key[0], rows),
                               sk.dct_matrix(cols)@sk.area_matrix(key[1], cols))
        left, right = self.cache[key]
        # Coefficients in 96x80-lattice units whatever the analysis lattice.
        return sk.separable(left, plane, right)*math.sqrt(
            LATTICE[0]*LATTICE[1]/(self.shape[0]*self.shape[1]))


def corpus(folder):
    files = sorted(Path(folder).glob('*.png'))+sorted(Path(folder).glob('*.jpg'))
    if len(files) < 8:
        raise SystemExit(f'need at least eight pictures in {folder}')
    return files[0::2], files[1::2]            # training, held out


class Statistics:
    """Per-position mean and standard deviation, training pictures only."""

    def __init__(self, coefficients):
        stack = np.stack(coefficients)
        self.mean = np.zeros(stack.shape[1:])
        self.mean[0, 0] = stack[:, 0, 0].mean()
        square = np.mean((stack-self.mean)**2, axis=0)
        self.variance = np.maximum(smooth_power(square, 1), 1e-12)
        self.sd = np.sqrt(self.variance)


# ------------------------------------------------------------------- schemes
SIGNATURE_SYMBOL = 3.0/math.sqrt(1+1/12+.16*.12)     # production pattern level


def loudness(symbols):
    """Noise multiplier of a packet relative to the measured profile.

    The sender holds every packet at one audio level, so a picture whose slot
    values are large is turned down and its noise, counted in slot units,
    rises in proportion.  Calibrated on the real wire (video and Kodak
    frames): about 1.2 times the symbol RMS, for quiet pictures as well as loud.
    """
    return max(.5, 1.2*float(np.sqrt(np.mean(np.square(symbols)))))


class Linear:
    """Maps a 96x80 luma coefficient plane to unit-power wire symbols.

    Every scheme is ``decode(encode(plane) + noise)``: the same two calls
    serve the slot-level study and the real wire.
    """
    name = 'linear 1,920'

    def __init__(self, geometry, stats, noise):
        self.g, self.s, self.noise = geometry, stats, np.asarray(noise, float)
        p = geometry.host_position
        self.host_sd, self.host_mean = stats.sd.ravel()[p], stats.mean.ravel()[p]

    def carried(self):
        return len(self.g.host_position)

    def hosts(self, plane):
        return (plane.ravel()[self.g.host_position]-self.host_mean)/self.host_sd

    def plane(self, hosts, positions=None, values=None):
        out = np.zeros(self.g.rows*self.g.cols)
        out[self.g.host_position] = hosts*self.host_sd+self.host_mean
        if positions is not None:
            out[positions] = values
        return out.reshape(self.g.rows, self.g.cols)

    def encode(self, plane):
        return self.hosts(plane)

    def decode(self, received):
        return self.plane(received/(1+self.noise**2))

    def run(self, plane, seed):
        rng = np.random.default_rng(seed)
        symbols = self.encode(plane)
        return self.decode(symbols+loudness(symbols)*self.noise*rng.standard_normal(len(self.noise)))


class Production(Linear):
    """Production Fold 500 slots and mapping; statistics retrained."""
    name = 'production fold 500'
    limit, mu, step = 12.0, 4.0, 1.0

    def __init__(self, geometry, stats, noise):
        super().__init__(geometry, stats, noise)
        usable = len(geometry.fold_hosts)-geometry.signature
        self.h, self.q = geometry.fold_hosts[:usable], geometry.fold_guests[:usable]
        self.reserved = geometry.fold_hosts[usable:]
        self.pattern = np.asarray(geometry.codec.pattern, float)

    def carried(self):
        return len(self.g.host_position)-len(self.reserved)+len(self.q)

    def guests(self, plane):
        return plane.ravel()[self.q]/self.s.sd.ravel()[self.q]

    def encode(self, plane):
        symbols = self.hosts(plane)
        symbols[self.h] = production_fold_encode(symbols[self.h], self.guests(plane),
                                                 self.step, self.limit, self.mu)
        symbols[self.reserved] = SIGNATURE_SYMBOL*self.pattern
        return symbols

    def decode(self, received):
        estimate = received/(1+self.noise**2)
        host, guest = production_fold_decode(received[self.h], self.noise[self.h],
                                             self.step, self.limit, self.mu)
        estimate[self.h] = host
        estimate[self.reserved] = 0.0                # signature slots carry no picture
        return self.plane(estimate, self.q, guest*self.s.sd.ravel()[self.q])


class PriorSurface(Production):
    """The earlier fold-native serpentine surface inside the 484 guests,
    using ``tools/v7_fold_surface.py``'s own encode/decode functions."""

    def __init__(self, geometry, stats, noise, inputs, outputs, strips):
        super().__init__(geometry, stats, noise)
        self.inputs, self.outputs, self.strips = inputs, outputs, strips
        self.name = f'earlier surface {inputs}to{outputs}, {strips} strips'
        self.groups = len(self.h)//outputs
        self.count = self.groups*outputs
        taken = np.zeros(geometry.rows*geometry.cols, bool)
        taken[geometry.host_position] = True
        taken[geometry.fold_guests] = True
        extra = geometry.radial[~taken[geometry.radial]]
        extra = extra[:self.groups*(inputs-outputs)].reshape(self.groups, inputs-outputs)
        self.positions = np.concatenate(
            (self.q[:self.count].reshape(self.groups, outputs), extra), 1)

    def carried(self):
        return super().carried()+self.groups*(self.inputs-self.outputs)

    def _compress(self, u):
        return np.sign(u)*np.log1p(self.mu*np.minimum(abs(u), self.limit)/self.limit)/np.log1p(self.mu)

    def _expand(self, c):
        return np.sign(c)*self.limit*np.expm1(abs(c)*np.log1p(self.mu))/self.mu

    def encode(self, plane):
        from tools.v7_fold_surface import surface_encode
        symbols = self.hosts(plane)
        sd = self.s.sd.ravel()
        source = plane.ravel()[self.positions]/sd[self.positions]
        residual = np.empty(len(self.h))
        residual[:self.count] = (surface_encode(.5+.5*self._compress(source),
                                                self.outputs, self.strips)*2-1).ravel()
        residual[self.count:] = self._compress(self.guests(plane)[self.count:])
        amp = .4*self.step
        power = 1+self.step**2/12+amp*amp*.12
        symbols[self.h] = (self.step*np.round(symbols[self.h]/self.step)+amp*residual)/math.sqrt(power)
        symbols[self.reserved] = SIGNATURE_SYMBOL*self.pattern
        return symbols

    def decode(self, received):
        from tools.v7_fold_surface import surface_decode
        estimate = received/(1+self.noise**2)
        amp = .4*self.step
        power = 1+self.step**2/12+amp*amp*.12
        symbol = received[self.h]*math.sqrt(power)
        k = np.round(symbol/self.step)
        back = np.clip((symbol-self.step*k)/amp, -1, 1)
        decoded = surface_decode(back[:self.count].reshape(self.groups, self.outputs)/2+.5,
                                 self.inputs, self.strips)*2-1
        estimate[self.h] = self.step*k
        estimate[self.reserved] = 0.0
        sd = self.s.sd.ravel()
        positions = np.concatenate((self.positions.ravel(), self.q[self.count:]))
        values = np.concatenate(((self._expand(np.clip(decoded, -1, 1))*sd[self.positions]).ravel(),
                                 self._expand(back[self.count:])*sd[self.q[self.count:]]))
        return self.plane(estimate, positions, values)


class Nested(Linear):
    """Noise-matched nested fold over the production luma slots.  The 16
    production signature slots stay reserved, so wire resources match."""

    def __init__(self, geometry, stats, noise, training, linear, triple, kappa,
                 guard=3.0, levels=3, kappa2=None, guard2=None, name=None, mid_gain=1.0,
                 guest_order='radial'):
        super().__init__(geometry, stats, noise)
        self.reserved = geometry.fold_hosts[len(geometry.fold_hosts)-geometry.signature:]
        self.active = np.setdiff1d(np.arange(len(self.noise)), self.reserved)
        self.pattern = np.asarray(geometry.codec.pattern, float)
        self.fold = sk.NestedFold(self.noise[self.active], linear, triple, kappa, guard,
                                  levels, kappa2, guard2, mid_gain=mid_gain)
        candidates = geometry.guest_position
        if isinstance(guest_order, np.ndarray):
            # An explicit list of guest positions, in fill order.
            candidates, guest_order = guest_order, 'given'
        if guest_order in ('axes', 'rows', 'columns'):
            # Line-oriented supports: uncarried coefficients nearest the
            # frequency axes first ('rows': horizontal lines only, 'columns':
            # vertical lines only), then by radius.
            width, height = geometry.aspect
            u = (candidates//geometry.cols)*width
            v = (candidates % geometry.cols)*height
            off = {'axes': np.minimum(u, v), 'rows': v, 'columns': u}[guest_order]
            candidates = candidates[np.lexsort((u*u+v*v, off))]
        if guest_order == 'variance':
            # Strongest uncarried coefficients first, by training statistics.
            candidates = candidates[np.argsort(-stats.variance.ravel()[candidates], kind='stable')]
        self.q = candidates[:self.fold.guests]
        self.parameters = dict(linear=linear, triple=triple, kappa=kappa, guard=guard,
                               levels=levels, kappa2=kappa2, guard2=guard2, mid_gain=mid_gain,
                               guest_order=guest_order)
        if guest_order == 'given':
            del self.parameters['guest_order']
        self.name = name or (f'nested fold: {linear} linear, '
                             f'{len(self.active)-linear-triple} double, {triple} triple')
        self.fold.fit(np.stack([self.guests(plane) for plane in training]))

    def carried(self):
        return len(self.active)+self.fold.guests

    def guests(self, plane):
        return plane.ravel()[self.q]/self.s.sd.ravel()[self.q]

    def encode(self, plane):
        symbols = np.empty(len(self.noise))
        symbols[self.active] = self.fold.encode(self.hosts(plane)[self.active], self.guests(plane))
        symbols[self.reserved] = SIGNATURE_SYMBOL*self.pattern
        return symbols

    def decode(self, received):
        estimate = np.zeros(len(self.noise))
        estimate[self.active], guests = self.fold.decode(received[self.active])
        return self.plane(estimate, self.q, guests*self.s.sd.ravel()[self.q])


# ------------------------------------------------------------------- scoring
def score(geometry, reference_big, decoded):
    """Scores of one decoded 96x80 plane against the fine reference."""
    full = np.zeros(geometry.big[0]*geometry.big[1])
    full[geometry.embed] = decoded.ravel()
    truth = reference_big.ravel()
    # Brightness (DC) is not picture detail: leave it out of energy ratios.
    error = float(np.sum((full-truth)[1:]**2))
    energy = truth[geometry.big_radial]**2
    energy[0] = 0.0
    bands = int(geometry.band.max())+1
    rho, gain, residual = band_statistics(truth, full, geometry.band, bands)
    passed = ((rho >= RHO_MIN) & (gain >= GAIN_RANGE[0]) & (gain <= GAIN_RANGE[1]) &
              (residual <= RESIDUAL_MAX))
    return {'error': error, 'energy': float(energy.sum()),
            'psnr': 10*math.log10(4*LATTICE[0]*LATTICE[1]/max(error, 1e-30)),
            'equivalent': float(equivalent_count(energy, error)),
            'passed': passed, 'rho': rho}


def summarise(geometry, rows, need):
    """Aggregate per-picture scores; ``need`` pictures must pass each band."""
    passed = np.stack([r['passed'] for r in rows])
    count = passed.sum(0)
    limit = 0
    for b in range(passed.shape[1]):
        if count[b] < need:
            break
        limit = b+1
    return {'pictures': len(rows),
            'mean_error': float(np.mean([r['error'] for r in rows])),
            'psnr_db': float(np.mean([r['psnr'] for r in rows])),
            'relative_error': float(np.mean([r['error']/r['energy'] for r in rows])),
            'equivalent_coefficients': float(np.exp(np.mean(np.log([max(r['equivalent'], 1) for r in rows])))),
            'resolved_cycles': limit*BAND,
            'band_pass_counts': count[:int(64//BAND)].tolist(),
            'median_rho': np.median(np.stack([r['rho'] for r in rows]), 0)[:int(64//BAND)].round(3).tolist()}


def evaluate(geometry, scheme, planes, references, seeds=(11, 12)):
    rows = []
    for index, (plane, reference) in enumerate(zip(planes, references)):
        for seed in seeds:
            rows.append(score(geometry, reference, scheme.run(plane, 1000*seed+index)))
    # Average the noise draws per picture before pass counting.
    merged = []
    for i in range(0, len(rows), len(seeds)):
        group = rows[i:i+len(seeds)]
        merged.append({'error': np.mean([r['error'] for r in group]), 'energy': group[0]['energy'],
                       'psnr': np.mean([r['psnr'] for r in group]),
                       'equivalent': np.mean([r['equivalent'] for r in group]),
                       'passed': np.all([r['passed'] for r in group], 0),
                       'rho': np.mean([r['rho'] for r in group], 0)})
    return merged


# -------------------------------------------------------------------- bounds
def ensemble_tail(geometry, variance_big):
    radial_var = variance_big.ravel()[geometry.big_radial][1:]
    return np.concatenate((np.cumsum(radial_var[::-1])[::-1], [0.]))


def ensemble_equivalent(tail, error):
    """Radially ordered coefficients whose loss equals this mean error."""
    return int(np.searchsorted(-tail, -error))


def bounds(geometry, variance_big, noise_sd, baseline_equivalent, entropy_ratio=1.0):
    """What any mapping could do with these slots at this noise."""
    lam = np.sort(variance_big.ravel()[1:])[::-1]            # detail only
    power = geometry.power
    # ``noise_sd`` is per slot, in units of that slot's design signal sd.
    bits_actual = float(np.sum(.5*np.log2(1+1/noise_sd**2)))
    bits_flat = float(water_filled_bits(np.ascontiguousarray(noise_sd**2*power), float(power.sum())))
    tail = ensemble_tail(geometry, variance_big)
    pass_bits = .5*math.log2(1/(1-RHO_MIN**2))             # Gaussian R(D) at the pass threshold
    out = {'slots': len(power), 'bits_per_packet_luma': bits_actual,
           'bits_per_coefficient_at_pass_threshold': pass_bits,
           'coefficients_at_pass_threshold_at_most': int(bits_actual/pass_bits),
           'bits_if_slot_power_were_redivided': bits_flat,
           'median_slot_snr_db': float(np.median(-20*np.log10(noise_sd)))}
    for label, bits in (('actual_power', bits_actual), ('redivided_power', bits_flat)):
        theta, distortion, active = sk.reverse_water_fill(lam, bits)
        equivalent = ensemble_equivalent(tail, distortion)
        out[label] = {'water_level': float(theta), 'distortion': float(distortion),
                      'relative_error': float(distortion/lam.sum()),
                      'coefficients_above_water': int(active),
                      'equivalent_coefficients': equivalent,
                      'linear_resolution_bound_vs_baseline': float(math.sqrt(equivalent/baseline_equivalent))}
    # Optimistic outer bound: marginal non-Gaussianity exploited perfectly
    # (Shannon lower bound with the measured entropy-power ratio).
    theta, _, _ = sk.reverse_water_fill(lam*entropy_ratio, bits_actual)
    distortion = float(np.sum(np.where(lam*entropy_ratio > theta, theta, lam)))
    equivalent = ensemble_equivalent(tail, distortion)
    out['marginal_entropy_outer_bound'] = {
        'entropy_power_ratio': entropy_ratio, 'distortion': distortion,
        'equivalent_coefficients': equivalent,
        'linear_resolution_bound_vs_baseline': float(math.sqrt(equivalent/baseline_equivalent))}
    return out


def design_search(geometry, stats, noise, training, references, grid):
    """Training-set error and resolved band of every table in a declared grid."""
    trace = []
    need = len(training)-1
    for parameters in grid:
        try:
            scheme = Nested(geometry, stats, noise, training, **parameters)
        except ValueError:
            continue
        summary = summarise(geometry, evaluate(geometry, scheme, training, references, seeds=(5,)), need)
        trace.append(dict(parameters, relative_error=summary['relative_error'],
                          resolved_cycles=summary['resolved_cycles'], carried=scheme.carried()))
    return trace


def pick(trace, objective, ceiling=None):
    keys = ('linear', 'triple', 'kappa', 'guard', 'kappa2', 'guard2')
    rows = [r for r in trace if ceiling is None or r['relative_error'] <= ceiling] or trace
    if objective == 'error':
        best = min(rows, key=lambda r: r['relative_error'])
    else:
        best = min(rows, key=lambda r: (-r['resolved_cycles'], r['relative_error']))
    return {k: best[k] for k in keys if k in best}


def grid_for(slots):
    grid = []
    for linear in (0, 16, 64, 208, 400, 700, 1000, 1420):
        for kappa in (10, 12, 14, 17, 20, 24, 30, 36, 44):
            for guard in (1.5, 2.0, 2.5, 3.0):
                grid.append(dict(linear=linear, triple=0, kappa=kappa, guard=guard))
                for triple in (300, 700, 1200, slots-linear):
                    if triple > slots-linear:
                        continue
                    for kappa2 in (8, 11, 15, 20):
                        grid.append(dict(linear=linear, triple=triple, kappa=kappa, guard=guard,
                                         kappa2=kappa2, guard2=guard))
    return grid


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('--corpus', default='tmp/sk/corpus')
    parser.add_argument('--out', default='tmp/sk/study')
    parser.add_argument('--conditions', nargs='+', default=list(MEASURED_NOISE))
    parser.add_argument('--noise', help='per-slot wire noise from tools/v7_sk_wire.py measure')
    args = parser.parse_args(argv)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    geometry = Geometry()
    train_files, test_files = corpus(args.corpus)
    small, big = Analyzer(LATTICE), Analyzer(geometry.big)
    train = [small(portrait(f)) for f in train_files]
    test = [small(portrait(f)) for f in test_files]
    train_big = [big(portrait(f)) for f in train_files]
    test_big = [big(portrait(f)) for f in test_files]
    stats = Statistics(train)
    stats_big = Statistics(test_big)                  # bound: the scored pictures' own spectrum
    tail = ensemble_tail(geometry, stats_big.variance)
    need = len(test)-1 if len(test) >= 12 else len(test)
    normalised = np.concatenate([(p/stats_big.sd).ravel()[1:] for p in test_big])
    report = {'layout': LAYOUT, 'lattice': LATTICE, 'training': [f.name for f in train_files],
              'held_out': [f.name for f in test_files], 'band_cycles': BAND,
              'criterion': {'rho_min': RHO_MIN, 'gain': GAIN_RANGE, 'residual_max': RESIDUAL_MAX,
                            'pictures_needed': need},
              'source_entropy_power_vs_gaussian': float(entropy_power_ratio(normalised)),
              'conditions': {}}
    for condition in args.conditions:
        if args.noise:
            noise = np.load(args.noise)[condition]
        else:
            noise = MEASURED_NOISE[condition]/np.sqrt(geometry.power)
        sigma = float(np.sqrt(np.mean(noise**2*geometry.power)))
        schemes = [Linear(geometry, stats, noise), Production(geometry, stats, noise),
                   PriorSurface(geometry, stats, noise, 3, 2, 3),
                   PriorSurface(geometry, stats, noise, 7, 3, 3)]
        slots = len(noise)-geometry.signature
        trace = design_search(geometry, stats, noise, train, train_big, grid_for(slots))
        reference = summarise(geometry, evaluate(geometry, schemes[1], train, train_big, seeds=(5,)),
                              len(train)-1)
        # Three frozen picks from one training-only search.  The resolution
        # pick may not be worse than production in overall training error.
        schemes.append(Nested(geometry, stats, noise, train, name='nested fold (least error)',
                              **pick(trace, 'error')))
        schemes.append(Nested(geometry, stats, noise, train, name='nested fold (most resolved)',
                              **pick(trace, 'resolved', reference['relative_error'])))
        schemes.append(Nested(geometry, stats, noise, train, name='nested fold (most coefficients)',
                              **pick([r for r in trace if r['linear'] <= 16 and
                                      r['triple'] == slots-r['linear']], 'error')))
        results = {}
        for scheme in schemes:
            summary = summarise(geometry, evaluate(geometry, scheme, test, test_big), need)
            summary['luma_coefficients_carried'] = scheme.carried()
            if isinstance(scheme, Nested):
                summary['parameters'] = scheme.parameters
            results[scheme.name] = summary
        base = results['production fold 500']
        for summary in results.values():
            summary['ensemble_equivalent_coefficients'] = ensemble_equivalent(tail, summary['mean_error'])
        for summary in results.values():
            summary['linear_resolution_vs_production'] = {
                'by_equivalent_coefficients': math.sqrt(summary['equivalent_coefficients']/base['equivalent_coefficients']),
                'by_resolved_cycles': (summary['resolved_cycles']/base['resolved_cycles']
                                       if base['resolved_cycles'] else None),
                'by_ensemble_equivalent': math.sqrt(summary['ensemble_equivalent_coefficients']/base['ensemble_equivalent_coefficients']),
                'by_carried_coefficients': math.sqrt(summary['luma_coefficients_carried']/base['luma_coefficients_carried'])}
        report['conditions'][condition] = {
            'wire_noise_sd': sigma, 'schemes': results,
            'bound': bounds(geometry, stats_big.variance, noise, base['ensemble_equivalent_coefficients'],
                            report['source_entropy_power_vs_gaussian']),
            'search': sorted(trace, key=lambda r: r['relative_error'])[:12],
            'search_tables': len(trace)}
        print(condition, json.dumps({k: (round(v['psnr_db'], 2), round(v['equivalent_coefficients']),
                                         v['resolved_cycles'], v['luma_coefficients_carried'])
                                     for k, v in results.items()}), flush=True)
        print('  bound', json.dumps(report['conditions'][condition]['bound']), flush=True)
    (out/'STUDY.json').write_text(json.dumps(report, indent=1)+'\n')
    return report


if __name__ == '__main__':
    main()
