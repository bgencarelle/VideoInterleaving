"""Shared modem engine: wire geometry, source coding and packet demodulation.

This is everything both sides need once timing is known. The v2 transmitter and
its correlation-bank receiver have been removed; transport3 is the only
transport. What remains here is version-independent:

  Layout            carriers, pilots, header placement, frame geometry
  SourceCoder       DCT with a power allocation, and the Wiener inverse
  decode_packet     OFDM demodulation, channel equalisation, header CRC
  resample_packet   fractional resampling used by the v3 timing fit

transport3 owns the preamble, acquisition, encoding and the Receiver. It
imports from here rather than the other way round, so this module has no
knowledge of how a packet was found.

The file keeps its name to avoid churning every import in the tree.

Of the three things that used to have to match at both endpoints, two no longer
do. Profile travels in two spare header bits. Preset is identified by decoding
against candidates and letting the header CRC pick, since it cannot be
signalled -- you need the layout to know where the header is. The allocation
table is still shared state: it is not on the wire and there is nothing to
detect it from, so a custom --allocation must match at both ends.
"""
import struct
import zlib
import time
from functools import lru_cache, cached_property
from fractions import Fraction
from dataclasses import dataclass, field

import numpy as np
from scipy.fft import dctn, idctn, rfft
from scipy.signal import resample_poly, firwin, butter, sosfiltfilt, filtfilt

REFERENCE_RATE = 48000
RATE = REFERENCE_RATE
N, CP = 128, 16
SYMBOL = N + CP
LOW_BIN = 3
SYNC_LEN = 288
GUARD = 32
HEADER_BYTES = 16
HEADER_SLOTS = 80


@dataclass(frozen=True)
class Layout:
    """Everything the wire format needs, derived from two choices."""
    top_bin: int = 54
    image_symbols: int = 15
    name: str = 'wide'
    progressive: bool = False
    header_width: int = 20
    header_split: bool = False
    spread_carriers: bool = False
    orthogonal_training: bool = False
    dense_header: bool = False

    def __post_init__(self):
        if not (isinstance(self.top_bin, int) and 10 <= self.top_bin <= 63
                and isinstance(self.image_symbols, int) and self.image_symbols > 0):
            raise ValueError('Layout requires top_bin 10..63 and positive image_symbols')

    @cached_property
    def carriers(self):
        return np.arange(1 if self.progressive else LOW_BIN, self.top_bin + 1)

    @cached_property
    def pilots(self):
        if self.progressive:
            base = np.array([3, 5, 21, 45])
            if self.top_bin >= base[-1]:
                return base
            upper = np.round(base[2:] * self.top_bin / base[-1]).astype(int)
            pilots = np.unique(np.concatenate([base[:2], upper]))
            pilots = pilots[(pilots >= self.carriers.min()) &
                            (pilots <= self.top_bin)]
            if len(pilots) < 4:
                spare = np.setdiff1d(self.carriers, pilots)
                need = min(4 - len(pilots), len(spare))
                pick = spare[np.linspace(0, len(spare) - 1, need).round().astype(int)]
                pilots = np.unique(np.concatenate([pilots, pick]))
            return pilots
        c = self.carriers
        if len(c) < 12:
            return c[np.linspace(0, len(c) - 1, 4).round().astype(int)]
        return c[np.linspace(3, len(c) - 4, 4).round().astype(int)]

    @cached_property
    def data_bins(self):
        return np.array([k for k in self.carriers if k not in set(self.pilots.tolist())])

    @cached_property
    def header_bins(self):
        want = min(max(1, self.header_width), len(self.data_bins))
        bins = self.data_bins
        if not self.spread_carriers:
            return bins[:want]
        middle_out = np.argsort(np.abs(np.arange(len(bins)) - (len(bins) - 1) / 2),
                                kind='stable')
        return np.sort(bins[np.sort(middle_out[:want])])

    @cached_property
    def header_lanes(self):
        return len(self.header_bins) * (2 if self.header_split else 1)

    @cached_property
    def header_symbols(self):
        return -(-HEADER_SLOTS // self.header_lanes)

    @cached_property
    def symbols(self):
        return 2 + self.header_symbols + self.image_symbols

    @cached_property
    def packet(self):
        return SYNC_LEN + self.symbols * SYMBOL

    @cached_property
    def frame(self):
        return self.packet + GUARD

    def fps_at(self, rate):
        return rate / self.frame

    def band_at(self, rate):
        c = self.carriers
        return (float(c[0] * rate / N), float(c[-1] * rate / N))

    @cached_property
    def fps(self):
        return self.fps_at(REFERENCE_RATE)

    @cached_property
    def band(self):
        return self.band_at(REFERENCE_RATE)

    @cached_property
    def wire_magic(self):
        if self.dense_header:
            return b'V4'
        return b'V3' if self.progressive else b'V2'

    @cached_property
    def spare_bins(self):
        taken = set(self.header_bins.tolist())
        return np.array([b for b in self.data_bins if b not in taken])

    @cached_property
    def header_capacity(self):
        return self.header_symbols * len(self.spare_bins) * 4 if self.dense_header else 0

    @cached_property
    def capacity(self):
        return self.header_capacity + self.image_symbols * len(self.data_bins) * 4

    @cached_property
    def max_speed(self):
        return N / (2 * self.top_bin)

    def describe(self, rate=REFERENCE_RATE):
        low, high = self.band_at(rate)
        return (f'{self.name}: {low:.0f}-{high:.0f} Hz, '
                f'{self.fps_at(rate):.2f} fps, {self.capacity} values, '
                f'{len(self.data_bins)} data bins, up to {self.max_speed:.2f}x speed '
                f'(at {rate:g} Hz)')


PRESETS = {
    'wide': Layout(top_bin=54, image_symbols=15, name='wide', progressive=True),
    'tape': Layout(top_bin=27, image_symbols=35, name='tape'),
    'tape-fast': Layout(top_bin=27, image_symbols=15, name='tape-fast'),
    'narrow': Layout(top_bin=21, image_symbols=44, name='narrow'),
    'lofi': Layout(top_bin=10, image_symbols=24, name='lofi'),
}


@lru_cache(maxsize=32)
def phases(layout, seed=52001):
    rng = np.random.default_rng(seed + layout.top_bin * 97 + layout.image_symbols)
    return np.exp(1j * rng.uniform(-np.pi, np.pi, (layout.symbols, len(layout.carriers), 2)))


def default_allocation(shapes, count):
    weights = []
    for rows, cols in shapes:
        fy = np.arange(rows)[:, None] / max(rows - 1, 1)
        fx = np.arange(cols)[None, :] / max(cols - 1, 1)
        weights.append((1.0 / (1.0 + 12 * np.hypot(fy, fx))).ravel())
    table = np.concatenate(weights)
    return np.resize(table, count)


class SourceCoder:
    def __init__(self, shapes, allocation=None, grids=None):
        self.shapes = list(shapes)
        self.grids = list(grids) if grids is not None else list(shapes)
        if len(self.grids) != len(self.shapes) or any(
                g[0] < s[0] or g[1] < s[1]
                for g, s in zip(self.grids, self.shapes)):
            raise ValueError('Each grid must be at least as large as its shape')
        self.count = int(sum(np.prod(s) for s in self.shapes))
        self.source_count = int(sum(np.prod(g) for g in self.grids))
        self.truncated = self.grids != self.shapes

        sigma = (self._allocation() if allocation is None
                 else np.asarray(allocation, float))
        if sigma.shape != (self.count,) or not np.all(np.isfinite(sigma) & (sigma > 0)):
            raise ValueError('Allocation must be one positive weight per value')

        gains = np.sqrt(sigma / np.mean(sigma))
        self.gains = gains / np.sqrt(np.mean(gains**2))
        self.variance = sigma**2

    def _allocation(self):
        if not self.truncated:
            return default_allocation(self.shapes, self.count)
        weights = []
        for (rows, cols), (grid_rows, grid_cols) in zip(self.shapes, self.grids):
            fy = np.arange(rows)[:, None] / max(grid_rows - 1, 1)
            fx = np.arange(cols)[None, :] / max(grid_cols - 1, 1)
            weights.append((1.0 / (1.0 + 12 * np.hypot(fy, fx))).ravel())
        return np.resize(np.concatenate(weights), self.count)

    def _split(self, values, shapes):
        out, offset = [], 0
        for shape in shapes:
            n = int(np.prod(shape))
            out.append(values[offset:offset + n].reshape(shape))
            offset += n
        return out

    def forward(self, values):
        planes = self._split(np.asarray(values, float), self.grids)
        kept = []
        for plane, (rows, cols) in zip(planes, self.shapes):
            kept.append(dctn(plane, norm='ortho')[:rows, :cols].ravel())
        return np.concatenate(kept) * self.gains

    def inverse(self, sent, reliability=None, noise_variance=None):
        """Wire slots in, pixels out. Dropped coefficients come back as zero,
        which is the least-energy completion and what truncation implies."""
        if reliability is None:
            coeffs = np.asarray(sent, float) / self.gains
        else:
            gain = self.gains * np.asarray(reliability)

            # Cap noise variance impact on clean, equalized channels so static
            # EQ cuts don't trigger additive noise suppression on active carriers.
            n_var = np.asarray(noise_variance, float)
            n_var_capped = np.minimum(n_var, self.variance * 0.1)

            denominator = gain ** 2 * self.variance + n_var_capped
            coeffs = np.divide(np.asarray(sent) * gain * self.variance, denominator,
                               out=np.zeros(self.count), where=denominator > 1e-20)

        out = []
        for corner, (rows, cols), grid in zip(
                self._split(coeffs, self.shapes), self.shapes, self.grids):
            if (rows, cols) == tuple(grid):
                out.append(idctn(corner, norm='ortho').ravel())
            else:
                full = np.zeros(grid)
                full[:rows, :cols] = corner
                out.append(idctn(full, norm='ortho').ravel())
        return np.concatenate(out)

@lru_cache(maxsize=32)
def coefficient_slots(layout, shapes):
    count = sum(h * w for h, w in shapes)
    if not layout.progressive:
        return np.arange(count)
    ranks = []
    for h, w in shapes:
        yy, xx = np.mgrid[:h, :w]
        ranks.extend(np.hypot(yy / h, xx / w).ravel())
    source_order = np.argsort(ranks, kind='stable')
    low_to_high = _slot_order(layout)[:count]
    mapping = np.empty(count, dtype=int)
    mapping[source_order] = low_to_high
    mapping.setflags(write=False)
    return mapping


@lru_cache(maxsize=32)
def _slot_order(layout):
    lanes = len(layout.data_bins)
    symbols, carriers = [], []
    if layout.dense_header:
        spare = np.flatnonzero(np.isin(layout.data_bins, layout.spare_bins))
        symbols.append(np.repeat(np.arange(layout.header_symbols), len(spare)))
        carriers.append(np.tile(spare, layout.header_symbols))
    symbols.append(np.repeat(np.arange(layout.image_symbols), lanes)
                   + layout.header_symbols)
    carriers.append(np.tile(np.arange(lanes), layout.image_symbols))
    symbol = np.repeat(np.concatenate(symbols), 4)
    carrier = np.repeat(np.concatenate(carriers), 4)
    channel = np.tile(np.array([0, 0, 1, 1]), len(symbol) // 4)
    part = np.tile(np.array([0, 1, 0, 1]), len(symbol) // 4)
    if layout.spread_carriers:
        rank = np.empty(lanes, int)
        rank[np.argsort(np.abs(np.arange(lanes) - (lanes - 1) / 2), kind='stable')] = \
            np.arange(lanes)
        return np.lexsort((rank[carrier], part, channel, symbol))
    return np.lexsort((part, channel, symbol, carrier))


@lru_cache(maxsize=4)
def _sinc_weight_table(taps=8, phases=4096):
    offsets = np.arange(-taps + 1, taps + 1, dtype=float)
    fraction = np.arange(phases, dtype=float)[:, None] / phases
    weights = (np.sinc(fraction - offsets[None, :]) *
               np.sinc((fraction - offsets[None, :]) / taps))
    weights /= weights.sum(axis=1, keepdims=True)
    return weights


@lru_cache(maxsize=32)
def _sample_walk(length, taps):
    return np.arange(length, dtype=np.float64), np.arange(-taps + 1, taps + 1)


def _sample_at(samples, position, taps=8):
    _, offsets = _sample_walk(0, taps)
    base = np.floor(position).astype(np.intp)
    table = _sinc_weight_table(taps)
    phase = np.minimum(((position - base) * len(table)).astype(np.intp), len(table) - 1)
    index = np.clip(base[:, None] + offsets, 0, len(samples) - 1)
    return np.einsum('ij,ijc->ic', table[phase], samples[index]).astype(np.float32)


@lru_cache(maxsize=32)
def _body_walk(layout):
    return (SYNC_LEN + np.arange(layout.symbols)[:, None] * SYMBOL +
            CP - 4 + np.arange(N)[None, :]).ravel()


def emit_ratio(rate, reference=REFERENCE_RATE, limit=16):
    rate = int(round(rate))
    if rate <= reference:
        return None
    ratio = Fraction(rate, int(reference)).limit_denominator(limit)
    return None if ratio == 1 else (ratio.numerator, ratio.denominator)


@lru_cache(maxsize=8)
def _emit_filter(up, down, top_bin=54, guard=1.185):
    edge = top_bin / N
    width = max(edge * (guard - 1), .02)
    taps = int(2 * np.ceil(4 / (width / max(up, down))) + 1)
    return firwin(taps, 1 / max(up, down), window=('kaiser', 8.6))


def band_limited(packet, rate, reference=REFERENCE_RATE, top_bin=54, pad=64):
    packet = np.asarray(packet, np.float32)
    ratio = emit_ratio(rate, reference)
    if ratio is None:
        return packet
    up, down = ratio
    want = int(round(len(packet) * up / down))
    quiet = np.zeros((pad, packet.shape[1]), np.float32)
    out = resample_poly(np.concatenate([quiet, packet, quiet]), up, down,
                        axis=0, window=_emit_filter(up, down, top_bin))
    begin = int(round(pad * up / down))
    out = out[begin:begin + want]
    if len(out) < want:
        out = np.concatenate([out, np.zeros((want - len(out), packet.shape[1]))])
    peak, want_peak = float(np.max(np.abs(out))), float(np.max(np.abs(packet)))
    if peak > 0 and want_peak > 0:
        out = out * (want_peak / peak)
    return np.asarray(out, np.float32)


EMIT_TAPS = 63


def bound_emission(packet, ceiling_hz, rate=REFERENCE_RATE, taps=EMIT_TAPS):
    packet = np.asarray(packet, np.float32)
    if not ceiling_hz or ceiling_hz >= rate / 2:
        return packet
    peak = float(np.max(np.abs(packet)))
    window = firwin(int(taps) | 1, ceiling_hz, fs=rate, window=('kaiser', 8.6))
    out = filtfilt(window, [1.0], packet, axis=0)
    got = float(np.max(np.abs(out)))
    if peak > 0 and got > 0:
        out = out * (peak / got)
    return np.asarray(out, np.float32)


def emit_length(samples, rate, reference=REFERENCE_RATE):
    ratio = emit_ratio(rate, reference)
    return samples if ratio is None else int(round(samples * ratio[0] / ratio[1]))


def resample_packet(samples, rate, length, taps=8, offset=0.0, fast=False):
    if fast:
        walk, _ = _sample_walk(length, taps)
        return _sample_at(samples, offset + walk * (1.0 + rate), taps)
    else:
        position = offset + np.arange(length) * (1.0 + rate)
        base = np.floor(position).astype(int)
        offsets = np.arange(-taps + 1, taps + 1)
        index = np.clip(base[:, None] + offsets[None, :], 0, len(samples) - 1)
        delta = (position - base)[:, None] - offsets[None, :]
        weights = np.sinc(delta) * np.sinc(delta / taps)
        weights /= weights.sum(axis=1, keepdims=True)
    return np.einsum('ij,ijc->ic', weights, samples[index]).astype(np.float32)


HEADER_GAIN = 1.6
IMAGE_GAIN = .7

HEADER_FORMAT = '>2sBBIHHI'
FOLDER_LIMIT = 16
PROFILE_CODES = ('color', 'color-lean', 'color-dct', 'mono')
TOP_BIN_MASK = 0x3f


def profile_code(name):
    try:
        return PROFILE_CODES.index(name)
    except ValueError:
        raise ValueError(f'Profile {name!r} has no wire code. '
                         f'Known: {", ".join(PROFILE_CODES)}') from None


def profile_name(code):
    code = int(code) & 3
    return PROFILE_CODES[code] if code < len(PROFILE_CODES) else None


def pack_folders(face, float_folder):
    return ((int(face) % FOLDER_LIMIT) << 4) | (int(float_folder) % FOLDER_LIMIT)


def pack_header(flags, top_bin, absolute, index, count, stamp_ms, magic=b'V2',
                profile=0):
    if not (0 <= absolute <= 0xffffffff and 1 <= index <= count <= 0xffff):
        raise ValueError('Frame/index/count outside the header ranges')
    if not 0 <= top_bin <= TOP_BIN_MASK:
        raise ValueError(f'top_bin must fit six bits, 0..{TOP_BIN_MASK}')
    if not 0 <= profile <= 3:
        raise ValueError('Profile code must fit the two spare header bits')
    raw = struct.pack(HEADER_FORMAT, magic, flags & 0xff,
                      (top_bin & TOP_BIN_MASK) | (int(profile) << 6),
                      absolute, index, count, stamp_ms & 0xffffffff)
    return raw + struct.pack('>I', zlib.crc32(raw))


@dataclass
class Decoded:
    status: str
    values: np.ndarray | None = None
    absolute: int | None = None
    index: int | None = None
    count: int | None = None
    stamp_ms: int | None = None
    flags: int = 0
    rate_error: float = 0.0
    rate_confidence: float = 0.0
    pilot_error: float | None = None
    coverage: float | None = None
    identity: str = 'unknown'
    tier: str = 'none'
    extra: dict = field(default_factory=dict)

    @property
    def face_folder(self):
        return None if self.identity != 'verified_header' else (self.flags >> 4) & 0xf

    @property
    def float_folder(self):
        return None if self.identity != 'verified_header' else self.flags & 0xf

    @property
    def source_index(self):
        return None if self.index is None else self.index - 1


@lru_cache(maxsize=32)
def _decode_tables(layout):
    carriers = layout.carriers
    carrier_set = set(carriers.tolist())
    indices = tuple(np.searchsorted(carriers, bins) for bins in
                    (layout.data_bins, layout.pilots, layout.header_bins))
    unused = np.array([k for k in range(1, N // 2 + 1)
                       if k not in carrier_set][:8], dtype=int)
    return (*indices, unused, phases(layout).conj())


def _channel_equalizer(spectrum, layout):
    carriers = layout.carriers
    received = spectrum[:, carriers, :]
    _, _, _, unused, inverse_phase = _decode_tables(layout)
    if layout.orthogonal_training:
        a0 = inverse_phase[0, :, 0].conj()
        a1 = inverse_phase[1, :, 0].conj()
        t = np.empty((len(carriers), 2, 2), complex)
        t[:, 0, 0], t[:, 1, 0] = a0, a0
        t[:, 0, 1], t[:, 1, 1] = a1, -a1
        det = t[:, 0, 0] * t[:, 1, 1] - t[:, 0, 1] * t[:, 1, 0]
        tinv = np.empty_like(t)
        tinv[:, 0, 0], tinv[:, 1, 1] = t[:, 1, 1], t[:, 0, 0]
        tinv[:, 0, 1], tinv[:, 1, 0] = -t[:, 0, 1], -t[:, 1, 0]
        tinv = tinv / det[:, None, None]
        r = np.stack([received[0], received[1]], axis=-1)
        h = np.einsum('krs,ksc->krc', r, tinv, optimize=False)
    else:
        h = np.stack([received[0] * inverse_phase[0, :, 0, None],
                      received[1] * inverse_phase[1, :, 1, None]], axis=-1)
    noise = max(float(np.mean(np.abs(spectrum[:, unused])**2)), 1e-12)
    hH = h.conj().transpose(0, 2, 1)
    gram = np.einsum('kij,kjl->kil', hH, h, optimize=False)
    gram[:, 0, 0] += 4 * noise
    gram[:, 1, 1] += 4 * noise
    det = gram[:, 0, 0] * gram[:, 1, 1] - gram[:, 0, 1] * gram[:, 1, 0]
    adj = np.empty_like(gram)
    adj[:, 0, 0], adj[:, 1, 1] = gram[:, 1, 1], gram[:, 0, 0]
    adj[:, 0, 1], adj[:, 1, 0] = -gram[:, 0, 1], -gram[:, 1, 0]
    inverse = np.einsum('kij,kjl->kil', adj / det[:, None, None], hH, optimize=False)
    weights = np.clip(np.real(np.einsum('kij,kji->ki', inverse, h, optimize=False)), 0, 1)
    variance = .5 * noise * np.sum(np.abs(inverse)**2, axis=-1) / IMAGE_GAIN**2
    coherence = abs(np.sum(h[1:] * h[:-1].conj())) / max(
        np.sqrt(np.sum(abs(h[1:])**2) * np.sum(abs(h[:-1])**2)), 1e-20)
    return inverse, weights, variance, float(coherence), _channel_skew(h, carriers)


def _equalise(body, layout):
    spectrum = rfft(body, n=N, axis=1, workers=1)
    inverse, weights, variance, coherence, skew = _channel_equalizer(spectrum, layout)
    equal = np.einsum('kij,skj->ski', inverse, spectrum[:, layout.carriers, :]) * _decode_tables(layout)[4]
    return equal, weights, variance, coherence, skew


def _channel_skew(h, carriers):
    if len(carriers) < 4 or not np.all(np.isfinite(h)):
        return None
    step = np.einsum('krc,krc->kr', h[1:], h[:-1].conj(), optimize=False)
    totals = step.sum(axis=0)
    both = step[:, 0] * step[:, 1].conj()
    weight = np.abs(both)
    if not np.all(np.isfinite(totals)) or weight.sum() <= 0:
        return None
    scale = N / (2 * np.pi)
    skew = -float(np.angle(totals[0] * totals[1].conj())) * scale
    each = -np.angle(both) * scale
    spread = float(np.sqrt(np.average((each - skew)**2, weights=weight)))
    energy = float(np.sum(np.abs(h)**2))
    mixed = float(np.sum(np.abs(h[:, 0, 1])**2) + np.sum(np.abs(h[:, 1, 0])**2))
    return {'skew_samples': skew, 'skew_spread': spread,
            'crosstalk': mixed / energy if energy else 0.}


def decode_packet(samples, layout, coder, *, body=None, coders=None):
    carriers = layout.carriers
    data, pilots, header, _, _ = _decode_tables(layout)
    if body is None:
        body = samples[SYNC_LEN:layout.packet].reshape(layout.symbols, SYMBOL, 2)[:, CP - 4:CP - 4 + N]

    equal, weights, variance, coherence, skew = _equalise(body, layout)

    timing_drift = 0.0
    clock_errors = []

    # Pilot tracking & phase slope correction (anchored at DC)
    for channel in range(2):
        keep = weights[pilots, channel] > 0.3
        if np.count_nonzero(keep) >= 2:
            bins = layout.pilots[keep].astype(float)
            pilot_values = equal[2:, pilots[keep], channel]
            angles = np.angle(pilot_values)

            if np.all(np.abs(pilot_values) > 0.15):
                initial = np.unwrap(angles[0])
                angles = np.unwrap(angles, axis=0)
                angles += (initial - angles[0])[None, :]
            else:
                angles = np.unwrap(angles, axis=1)

            w = weights[pilots[keep], channel]**2
            s2 = (w * bins * bins).sum()

            if s2 > 1e-12:
                txy = (angles * w * bins).sum(1)
                slope = txy / s2
                offset = np.zeros_like(slope)

                max_slope = np.pi / (N / 2)
                slope = np.clip(slope, -max_slope, max_slope)

                if len(slope) >= 4:
                    drift = abs(float(np.median(slope[-3:]) - np.median(slope[:3]))) * N / (2 * np.pi)
                    timing_drift = max(timing_drift, drift)

                    axis = np.arange(len(slope), dtype=float)
                    axis -= axis.mean()
                    axis_norm = np.einsum('i,i->', axis, axis, optimize=False)

                    if axis_norm > 0:
                        gradient = float(np.einsum('i,i->', axis, slope, optimize=False) / axis_norm)
                        residual = slope - slope.mean() - gradient * axis
                        if np.sqrt(np.mean(residual**2)) * N / (2 * np.pi) < 0.25:
                            clock_errors.append(gradient * N / (2 * np.pi * SYMBOL))

                equal[2:, :, channel] *= np.exp(-1j * (slope[:, None] * carriers + offset[:, None]))

    # Compute phase-normalized pilot error (resilient to static EQ amplitude cuts)
    pilot_norm = equal[2:, pilots] / np.maximum(np.abs(equal[2:, pilots]), 1e-6)
    pilot_error = float(np.sqrt(np.mean(np.abs(pilot_norm - 1)**2)))

    end = 2 + layout.header_symbols
    fields = None

    if layout.header_split:
        picks = [np.stack([equal[2:end, header, 0], equal[2:end, header, 1]],
                          axis=-1).reshape(layout.header_symbols, -1)]
    else:
        picks = [equal[2:end, header].mean(axis=-1),
                 equal[2:end, header, 0], equal[2:end, header, 1]]

    for pick in picks:
        flat = pick.ravel()[:HEADER_SLOTS] / HEADER_GAIN
        if len(flat) < HEADER_SLOTS:
            continue
        bits = np.stack([flat.real > 0, flat.imag > 0], axis=-1).ravel()
        raw = np.packbits(bits).tobytes()

        if zlib.crc32(raw[:HEADER_BYTES]) == struct.unpack('>I', raw[HEADER_BYTES:])[0]:
            magic, hflags, packed, absolute, index, count, stamp = struct.unpack(
                HEADER_FORMAT, raw[:HEADER_BYTES])
            top, code = packed & TOP_BIN_MASK, packed >> 6
            expected_magic = layout.wire_magic
            if magic == expected_magic and top == layout.top_bin and 1 <= index <= count:
                fields = (hflags, absolute, index, count, stamp, code)
                break

    coverage = float(np.mean(weights[data] >= 0.30))
    usable = (coverage >= 0.08 and pilot_error < 2.0 and
              coherence > (0.40 if layout.top_bin <= 13 else 0.25))

    if fields is None and not usable:
        return Decoded('lost', pilot_error=pilot_error, coverage=coverage)

    values_parts, weight_parts, noise_parts = [], [], []

    if layout.header_capacity:
        spare = np.searchsorted(carriers, layout.spare_bins)
        early = equal[2:end, spare] / IMAGE_GAIN
        shape = (layout.header_symbols, len(spare), 2, 2)
        values_parts.append(np.stack([early.real, early.imag], axis=-1).ravel())
        weight_parts.append(np.broadcast_to(weights[spare][None, :, :, None], shape).ravel())
        noise_parts.append(np.broadcast_to(variance[spare][None, :, :, None], shape).ravel())

    block = equal[end:, data] / IMAGE_GAIN
    shape = (layout.image_symbols, len(data), 2, 2)
    values_parts.append(np.stack([block.real, block.imag], axis=-1).ravel())
    weight_parts.append(np.broadcast_to(weights[data][None, :, :, None], shape).ravel())
    noise_parts.append(np.broadcast_to(variance[data][None, :, :, None], shape).ravel())

    sent = np.concatenate(values_parts)
    per = np.concatenate(weight_parts)
    coverage = float(np.mean(per >= 0.30))
    per_noise = np.concatenate(noise_parts)

    declared = profile_name(fields[5]) if fields is not None else None
    picture = coder
    if fields is not None and coders:
        picture = coders.get(fields[5], coder)

    slots = coefficient_slots(layout, tuple(picture.shapes))
    values = picture.inverse(sent[slots], per[slots], per_noise[slots])

    tier = ('best' if coverage >= 0.90 else 'better' if coverage >= 0.65
            else 'good' if coverage >= 0.30 else 'poor')

    if fields is None:
        return Decoded('picture_only' if usable else 'lost',
                       values=values if usable else None, pilot_error=pilot_error,
                       coverage=coverage, tier=tier if usable else 'none',
                       extra={'timing_drift_samples': timing_drift,
                              'clock_error': float(np.median(clock_errors)) if clock_errors else None,
                              'profile': declared, 'shapes': tuple(picture.grids),
                              **(skew or {})})

    flags, absolute, index, count, stamp, _ = fields
    return Decoded('received' if pilot_error < 0.35 else 'degraded', values=values,
                   absolute=absolute, index=index, count=count, stamp_ms=stamp,
                   flags=flags, pilot_error=pilot_error, coverage=coverage,
                   identity='verified_header', tier=tier,
                   extra={'timing_drift_samples': timing_drift,
                          'clock_error': float(np.median(clock_errors)) if clock_errors else None,
                          'profile': declared, 'shapes': tuple(picture.grids),
                          **(skew or {})})
@lru_cache(maxsize=8)
def _sync_filter(cutoff, rate=REFERENCE_RATE):
    return firwin(33, cutoff, fs=rate)


@lru_cache(maxsize=4)
def _conditioning_filter(cutoff, rate=REFERENCE_RATE):
    return butter(4, cutoff, btype='highpass', fs=rate, output='sos')


def _recovery_quality(result):
    if result.values is None or not np.isfinite(result.values).all():
        return (False, False, -float('inf'))
    return (True, result.identity == 'verified_header',
            -float(result.pilot_error if result.pilot_error is not None else np.inf))