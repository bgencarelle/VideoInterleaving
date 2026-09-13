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

The file keeps its name to avoid churning every import in the tree. Static
preset/profile/allocation must still match at both endpoints.
"""
import struct
import zlib
import time
from functools import lru_cache, cached_property
from fractions import Fraction
from dataclasses import dataclass, field

import numpy as np
from scipy.fft import dctn, idctn, rfft
from scipy.signal import resample_poly, firwin, butter, sosfiltfilt

RATE = 48000
N, CP = 128, 16
SYMBOL = N + CP
LOW_BIN = 3
SYNC_LEN = 288
GUARD = 32                 # idle tail; not required for EOF decoding
HEADER_BYTES = 16          # + CRC32 = 20 bytes = 160 bits = 80 QPSK
HEADER_SLOTS = 80          # placed identically on BOTH channels, for diversity
# Preamble scales searched at acquisition. Wide enough for half speed and
# double speed; the picture itself dies above ~1.19x from aliasing anyway.


@dataclass(frozen=True)
class Layout:
    """Everything the wire format needs, derived from two choices."""
    top_bin: int = 54          # highest carrier; 54 -> 20.25 kHz, 27 -> 10.1 kHz
    image_symbols: int = 15    # more symbols -> bigger picture, lower frame rate
    name: str = 'wide'
    progressive: bool = False
    header_width: int = 20     # header carriers; wider -> fewer header symbols
    header_split: bool = False # halve header symbols by sending different
                               # halves on each channel instead of the same
                               # bits twice, trading diversity for frame rate
    orthogonal_training: bool = False  # drive both channels in both training
                               # symbols instead of one at a time; same peak,
                               # twice the energy, half the estimator error

    def __post_init__(self):
        if not (isinstance(self.top_bin,int) and 10<=self.top_bin<=63
                and isinstance(self.image_symbols,int) and self.image_symbols>0):
            raise ValueError('Layout requires top_bin 10..63 and positive image_symbols')

    @cached_property
    def carriers(self):
        return np.arange(1 if self.progressive else LOW_BIN, self.top_bin + 1)

    @cached_property
    def pilots(self):
        if self.progressive:
            # Two low-band pilots survive roll-off; bins 1 and 2 carry images.
            return np.array([3, 5, 21, 45])
        c = self.carriers
        if len(c)<12:
            return c[np.linspace(0,len(c)-1,4).round().astype(int)]
        return c[np.linspace(3, len(c)-4, 4).round().astype(int)]

    @cached_property
    def data_bins(self):
        return np.array([k for k in self.carriers if k not in set(self.pilots.tolist())])

    @cached_property
    def header_bins(self):
        """The lowest data carriers -- the ones that survive tape and roll-off.

        Widening this trades robustness for frame rate: the header moves onto
        higher carriers, which tape treats worse, but needs fewer symbols.
        """
        return self.data_bins[:min(max(1, self.header_width), len(self.data_bins))]

    @cached_property
    def header_lanes(self):
        """Independent header slots per symbol. Two channels when split."""
        return len(self.header_bins)*(2 if self.header_split else 1)

    @cached_property
    def header_symbols(self):
        return -(-HEADER_SLOTS//self.header_lanes)

    @cached_property
    def symbols(self):
        return 2 + self.header_symbols + self.image_symbols

    @cached_property
    def packet(self):
        return SYNC_LEN + self.symbols*SYMBOL

    @cached_property
    def frame(self):
        return self.packet + GUARD

    @cached_property
    def fps(self):
        return RATE/self.frame

    @cached_property
    def band(self):
        c = self.carriers
        return (float(c[0]*RATE/N), float(c[-1]*RATE/N))

    @cached_property
    def capacity(self):
        """Real image values carried per packet."""
        return self.image_symbols*len(self.data_bins)*4

    @cached_property
    def max_speed(self):
        """Playback speed above which the top carrier folds past Nyquist."""
        return (RATE/2)/self.band[1]

    def describe(self):
        return (f'{self.name}: {self.band[0]:.0f}-{self.band[1]:.0f} Hz, '
                f'{self.fps:.2f} fps, {self.capacity} values, '
                f'{len(self.data_bins)} data bins, up to {self.max_speed:.2f}x speed')


PRESETS = {
    # today's band and cadence, for a cable or a digital loopback
    'wide': Layout(top_bin=54, image_symbols=15, name='wide', progressive=True),
    # cassette: 10 kHz ceiling, full picture, slower
    'tape': Layout(top_bin=27, image_symbols=35, name='tape'),
    # cassette: 10 kHz ceiling, keeps the frame rate, smaller picture
    'tape-fast': Layout(top_bin=27, image_symbols=15, name='tape-fast'),
    # worn deck or acoustic coupling
    'narrow': Layout(top_bin=21, image_symbols=44, name='narrow'),
    # Very limited-bandwidth tape: deliberately coarse, but below 3.75 kHz.
    'lofi': Layout(top_bin=10, image_symbols=24, name='lofi'),
}


@lru_cache(maxsize=32)
def phases(layout, seed=52001):
    rng = np.random.default_rng(seed + layout.top_bin*97 + layout.image_symbols)
    return np.exp(1j*rng.uniform(-np.pi, np.pi, (layout.symbols, len(layout.carriers), 2)))


# --------------------------------------------------------------------------
# Source coding: DCT with a fixed power allocation
# --------------------------------------------------------------------------

def default_allocation(shapes, count):
    """Fallback table when no bake statistics are available.

    A separable low-frequency emphasis. Replace with measured coefficient
    energies from the actual bake -- the image library never changes, so the
    table is a constant and costs no header bits.
    """
    weights = []
    for rows, cols in shapes:
        fy = np.arange(rows)[:, None]/max(rows-1, 1)
        fx = np.arange(cols)[None, :]/max(cols-1, 1)
        weights.append((1.0/(1.0 + 12*np.hypot(fy, fx))).ravel())
    table = np.concatenate(weights)
    return np.resize(table, count)


class SourceCoder:
    """Forward and inverse DCT with allocation, over a list of plane shapes."""

    def __init__(self, shapes, allocation=None):
        self.shapes = list(shapes)
        self.count = int(sum(np.prod(s) for s in self.shapes))
        sigma = (default_allocation(self.shapes, self.count)
                 if allocation is None else np.asarray(allocation, float))
        if sigma.shape != (self.count,) or not np.all(np.isfinite(sigma) & (sigma > 0)):
            raise ValueError('Allocation must be one positive weight per value')
        # Power proportional to standard deviation is the analog optimum.
        gains = np.sqrt(sigma/np.mean(sigma))
        self.gains = gains/np.sqrt(np.mean(gains**2))
        self.variance = sigma**2

    def _split(self, values):
        out, offset = [], 0
        for shape in self.shapes:
            n = int(np.prod(shape))
            out.append(values[offset:offset+n].reshape(shape))
            offset += n
        return out

    def forward(self, values):
        coeffs = np.concatenate([dctn(p, norm='ortho').ravel()
                                 for p in self._split(np.asarray(values, float))])
        return coeffs*self.gains

    def inverse(self, sent, reliability=None, noise_variance=None):
        if reliability is None:
            coeffs = np.asarray(sent, float)/self.gains
        else:
            gain = self.gains*np.asarray(reliability)
            denominator = gain**2*self.variance + np.asarray(noise_variance)
            coeffs = np.divide(np.asarray(sent)*gain*self.variance, denominator,
                               out=np.zeros(self.count), where=denominator > 1e-20)
        return np.concatenate([idctn(p, norm='ortho').ravel()
                               for p in self._split(coeffs)])


@lru_cache(maxsize=32)
def coefficient_slots(layout, shapes):
    """Source coefficient -> wire slot, coarse image first in audio frequency.

    Sort all planes together by normalized spatial frequency. Place the three
    DC terms first, then progressively finer features on ascending carriers.
    Each carrier spans all image symbols and both stereo channels.
    """
    count = sum(h*w for h,w in shapes)
    if not layout.progressive:
        return np.arange(count)
    ranks = []
    for h,w in shapes:
        yy,xx = np.mgrid[:h,:w]
        ranks.extend(np.hypot(yy/h,xx/w).ravel())
    source_order = np.argsort(ranks, kind='stable')
    slots = np.arange(layout.capacity).reshape(
        layout.image_symbols, len(layout.data_bins), 2, 2)
    low_to_high = slots.transpose(1,0,2,3).ravel()[:count]
    mapping = np.empty(count, dtype=int)
    mapping[source_order] = low_to_high
    mapping.setflags(write=False)
    return mapping


# --------------------------------------------------------------------------
# Preamble templates and resampling (legacy full-bank helper retained)
# --------------------------------------------------------------------------

@lru_cache(maxsize=4)
def _sinc_weight_table(taps=8, phases=4096):
    """Quantized windowed-sinc weights for the live packet path."""
    offsets = np.arange(-taps+1, taps+1, dtype=float)
    fraction = np.arange(phases, dtype=float)[:, None]/phases
    weights = (np.sinc(fraction-offsets[None, :]) *
               np.sinc((fraction-offsets[None, :])/taps))
    weights /= weights.sum(axis=1, keepdims=True)
    return weights


# One fixed fractional-delay table serves every speed and starting phase.
# Only the affine sample walk changes; no cache keyed by noisy speed estimates.
@lru_cache(maxsize=32)
def _sample_walk(length, taps):
    return np.arange(length, dtype=np.float64), np.arange(-taps+1, taps+1)


def _sample_at(samples, position, taps=8):
    """Read a recovered-clock sample walk using fixed fractional-delay weights."""
    _, offsets = _sample_walk(0, taps)
    base = np.floor(position).astype(np.intp)
    table = _sinc_weight_table(taps)
    phase = np.minimum(((position-base)*len(table)).astype(np.intp), len(table)-1)
    index = np.clip(base[:, None]+offsets, 0, len(samples)-1)
    return np.einsum('ij,ijc->ic', table[phase], samples[index]).astype(np.float32)


@lru_cache(maxsize=32)
def _body_walk(layout):
    """Only the useful FFT windows; skip sync, guards and unused CP samples."""
    return (SYNC_LEN + np.arange(layout.symbols)[:, None]*SYMBOL +
            CP-4 + np.arange(N)[None, :]).ravel()


def resample_packet(samples, rate, length, taps=8, offset=0.0, fast=False):
    """Resample one packet, using a cheap path for normal small clock error.

    The live path and its local timing updates use a fixed fractional-delay
    table. Cold acquisition retains the full-precision reference path.
    """
    if fast:
        walk, _ = _sample_walk(length, taps)
        return _sample_at(samples, offset+walk*(1.0+rate), taps)
    else:
        position = offset + np.arange(length)*(1.0+rate)
        base = np.floor(position).astype(int)
        offsets = np.arange(-taps+1, taps+1)
        index = np.clip(base[:, None]+offsets[None, :], 0, len(samples)-1)
        delta = (position-base)[:, None]-offsets[None, :]
        weights = np.sinc(delta)*np.sinc(delta/taps)
        weights /= weights.sum(axis=1, keepdims=True)
    return np.einsum('ij,ijc->ic', weights, samples[index]).astype(np.float32)


# --------------------------------------------------------------------------
# Encode / decode
# --------------------------------------------------------------------------

HEADER_GAIN = 1.6          # the header must outlive the picture, not precede it
IMAGE_GAIN = .7


HEADER_FORMAT = '>2sBBIHHI'      # magic, flags, top_bin, absolute, index, count, stamp
# The flags byte was transmitted and CRC-protected but always zero. It now
# carries the selected folder pair, four bits each: face in the high nibble,
# float in the low one. Free -- the byte was already on the wire and the header
# did not grow.
#
# Four bits holds 16. Past that the index wraps, so with more than 16 folders
# of a layer the reported number is the folder modulo 16 and several folders
# share a number. That is a reporting field, not a selection input: nothing
# decodes from it and the picture is unaffected either way.
FOLDER_LIMIT = 16


def pack_folders(face, float_folder):
    """Pack a folder pair into the flags byte, wrapping past FOLDER_LIMIT."""
    return ((int(face) % FOLDER_LIMIT) << 4) | (int(float_folder) % FOLDER_LIMIT)


def pack_header(flags, top_bin, absolute, index, count, stamp_ms, magic=b'V2'):
    if not (0 <= absolute <= 0xffffffff and 1 <= index <= count <= 0xffff):
        raise ValueError('Frame/index/count outside the header ranges')
    raw = struct.pack(HEADER_FORMAT, magic, flags & 0xff, top_bin & 0xff,
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
        """Selected face folder, from the flags byte. None without identity.

        Modulo 16 -- see FOLDER_LIMIT. With a longer folder list this is the
        low four bits of the real index, not the index itself.
        """
        return None if self.identity != 'verified_header' else (self.flags >> 4) & 0xf

    @property
    def float_folder(self):
        """Selected float folder, from the flags byte. None without identity.

        Modulo 16 -- see FOLDER_LIMIT.
        """
        return None if self.identity != 'verified_header' else self.flags & 0xf

    @property
    def source_index(self):
        """Zero-based index into the bake.

        The wire carries it one-based so that a zeroed or erased header fails
        the `1 <= index <= count` guard rather than decoding as frame zero.
        Everything downstream wants the bake's own numbering, so convert once
        here instead of leaving a -1 for every caller to remember.
        """
        return None if self.index is None else self.index - 1


@lru_cache(maxsize=32)
def _decode_tables(layout):
    """Fixed wire geometry and phase inverses, shared by every packet."""
    carriers = layout.carriers
    carrier_set = set(carriers.tolist())
    indices = tuple(np.searchsorted(carriers, bins) for bins in
                    (layout.data_bins, layout.pilots, layout.header_bins))
    unused = np.array([k for k in range(1, N//2+1)
                       if k not in carrier_set][:8], dtype=int)
    return (*indices, unused, phases(layout).conj())


def _channel_equalizer(spectrum, layout):
    carriers = layout.carriers
    received = spectrum[:, carriers, :]
    _, _, _, unused, inverse_phase = _decode_tables(layout)
    if layout.orthogonal_training:
        # Both channels drive both training symbols, so each received symbol is
        # a mixture: R = H @ T per carrier, and H = R @ inv(T).
        #
        # The transmitted rows are [1,1] and [1,-1], but BOTH channels carry
        # channel 0's phase -- see the encoder. That matters: phases() draws an
        # independent phase per channel, which would otherwise destroy the
        # orthogonality and leave T near-singular wherever the two phases
        # nearly cancel. Measured, naive [1,1]/[1,-1] drops |det| as low as
        # 0.042 and makes the estimate 20x worse than v2's scheme; locking the
        # phases holds |det| at exactly 2.
        a0 = inverse_phase[0, :, 0].conj()
        a1 = inverse_phase[1, :, 0].conj()
        t = np.empty((len(carriers), 2, 2), complex)
        t[:, 0, 0], t[:, 1, 0] = a0, a0
        t[:, 0, 1], t[:, 1, 1] = a1, -a1
        det = t[:, 0, 0]*t[:, 1, 1] - t[:, 0, 1]*t[:, 1, 0]
        tinv = np.empty_like(t)
        tinv[:, 0, 0], tinv[:, 1, 1] = t[:, 1, 1], t[:, 0, 0]
        tinv[:, 0, 1], tinv[:, 1, 0] = -t[:, 0, 1], -t[:, 1, 0]
        tinv = tinv/det[:, None, None]
        r = np.stack([received[0], received[1]], axis=-1)   # (carrier, rx, symbol)
        h = np.einsum('krs,ksc->krc', r, tinv, optimize=False)
    else:
        h = np.stack([received[0]*inverse_phase[0, :, 0, None],
                      received[1]*inverse_phase[1, :, 1, None]], axis=-1)
    noise = max(float(np.mean(np.abs(spectrum[:, unused])**2)), 1e-12)
    # Keep these tiny products out of matmul/BLAS; do not suppress FP errors.
    hH = h.conj().transpose(0, 2, 1)
    gram = np.einsum('kij,kjl->kil', hH, h, optimize=False)
    gram[:, 0, 0] += 4*noise
    gram[:, 1, 1] += 4*noise
    det = gram[:, 0, 0]*gram[:, 1, 1] - gram[:, 0, 1]*gram[:, 1, 0]
    adj = np.empty_like(gram)
    adj[:, 0, 0], adj[:, 1, 1] = gram[:, 1, 1], gram[:, 0, 0]
    adj[:, 0, 1], adj[:, 1, 0] = -gram[:, 0, 1], -gram[:, 1, 0]
    inverse = np.einsum('kij,kjl->kil', adj/det[:, None, None], hH, optimize=False)
    weights = np.clip(np.real(np.einsum('kij,kji->ki', inverse, h, optimize=False)), 0, 1)
    variance = .5*noise*np.sum(np.abs(inverse)**2, axis=-1)/IMAGE_GAIN**2
    coherence = abs(np.sum(h[1:]*h[:-1].conj())) / max(
        np.sqrt(np.sum(abs(h[1:])**2)*np.sum(abs(h[:-1])**2)), 1e-20)
    return inverse, weights, variance, float(coherence)


def _equalise(body, layout):
    spectrum = rfft(body, n=N, axis=1, workers=1)
    inverse, weights, variance, coherence = _channel_equalizer(spectrum, layout)
    equal = np.einsum('kij,skj->ski', inverse, spectrum[:, layout.carriers, :])*_decode_tables(layout)[4]
    return equal, weights, variance, coherence


def decode_packet(samples, layout, coder, *, body=None):
    carriers = layout.carriers
    data, pilots, header, _, _ = _decode_tables(layout)
    if body is None:
        body = samples[SYNC_LEN:layout.packet].reshape(layout.symbols, SYMBOL, 2)[:, CP-4:CP-4+N]
    equal, weights, variance, coherence = _equalise(body, layout)

    timing_drift = 0.0
    clock_errors = []
    for channel in range(2):
        keep = weights[pilots, channel] > .6
        if np.count_nonzero(keep) >= 2:
            bins = layout.pilots[keep].astype(float)
            pilot_values = equal[2:, pilots[keep], channel]
            angles = np.angle(pilot_values)
            if np.all(np.abs(pilot_values) > .25):
                # Adjacent symbols give a much smaller phase increment than
                # widely spaced carriers during flutter. Keep that continuity
                # instead of choosing a fresh, possibly aliased slope each row.
                initial = np.unwrap(angles[0])
                angles = np.unwrap(angles, axis=0)
                angles += (initial-angles[0])[None, :]
            else:
                # Missing pilots cannot anchor a phase history. Preserve the
                # independent-symbol fallback for header erasure/dropouts.
                angles = np.unwrap(angles, axis=1)
            w = weights[pilots[keep], channel]**2
            s0, s1, s2 = w.sum(), (w*bins).sum(), (w*bins*bins).sum()
            det = s0*s2 - s1*s1
            if det:
                ty, txy = (angles*w).sum(1), (angles*w*bins).sum(1)
                slope = (s0*txy - s1*ty)/det
                offset = (s2*ty - s1*txy)/det
                # Existing pilot estimates also tell the next acquisition
                # whether the cached speed needs refinement. No extra FFT.
                if len(slope) >= 4:
                    drift = abs(float(np.median(slope[-3:])-np.median(slope[:3]))) * N/(2*np.pi)
                    timing_drift = max(timing_drift, drift)
                    # Signed phase slope per symbol measures relative clock
                    # error. Reject nonlinear/noisy fits rather than steering
                    # the recovered clock with an unreliable pilot.
                    axis = np.arange(len(slope), dtype=float)
                    axis -= axis.mean()
                    gradient = float(np.einsum('i,i->', axis, slope, optimize=False) / np.einsum('i,i->', axis, axis, optimize=False))
                    residual = slope-slope.mean()-gradient*axis
                    if np.sqrt(np.mean(residual**2))*N/(2*np.pi) < .15:
                        clock_errors.append(gradient*N/(2*np.pi*SYMBOL))
                equal[2:, :, channel] *= np.exp(-1j*(slope[:, None]*carriers + offset[:, None]))
    pilot_error = float(np.sqrt(np.mean(np.abs(equal[2:, pilots] - 1)**2)))

    end = 2 + layout.header_symbols
    fields = None
    # Both channels carry the same symbols, so their mean is a genuine
    # diversity combine; fall back to each channel alone if one is corrupted.
    if layout.header_split:
        # Each channel carries a different half, so there is no diversity to
        # combine: interleave the two lanes back into one slot sequence.
        picks = [np.stack([equal[2:end, header, 0], equal[2:end, header, 1]],
                          axis=-1).reshape(layout.header_symbols, -1)]
    else:
        picks = [equal[2:end, header].mean(axis=-1),
                 equal[2:end, header, 0], equal[2:end, header, 1]]
    for pick in picks:
        flat = pick.ravel()[:HEADER_SLOTS]/HEADER_GAIN
        if len(flat) < HEADER_SLOTS:
            continue
        bits = np.stack([flat.real > 0, flat.imag > 0], axis=-1).ravel()
        raw = np.packbits(bits).tobytes()
        if zlib.crc32(raw[:HEADER_BYTES]) == struct.unpack('>I', raw[HEADER_BYTES:])[0]:
            magic, hflags, top, absolute, index, count, stamp = struct.unpack(
                HEADER_FORMAT, raw[:HEADER_BYTES])
            expected_magic = b'V3' if layout.progressive else b'V2'
            if magic == expected_magic and top == layout.top_bin and 1 <= index <= count:
                fields = (hflags, absolute, index, count, stamp)
                break

    # Do not reconstruct an image that the checks have already rejected.
    coverage = float(np.mean(weights[data] >= .55))
    usable = (coverage >= .1 and pilot_error < 1.5 and
              coherence > (.65 if layout.top_bin <= 13 else .4))
    if fields is None and not usable:
        return Decoded('lost', pilot_error=pilot_error, coverage=coverage)
    block = equal[end:, data]/IMAGE_GAIN
    sent = np.stack([block.real, block.imag], axis=-1).ravel()
    per = np.broadcast_to(weights[data][None, :, :, None],
                          (layout.image_symbols, len(data), 2, 2)).ravel()
    coverage = float(np.mean(per >= .55))
    per_noise = np.broadcast_to(variance[data][None, :, :, None],
                               (layout.image_symbols, len(data), 2, 2)).ravel()
    slots = coefficient_slots(layout, tuple(coder.shapes))
    values = coder.inverse(sent[slots], per[slots], per_noise[slots])
    tier = ('best' if coverage >= .95 else 'better' if coverage >= .7
            else 'good' if coverage >= .35 else 'poor')
    if fields is None:
        usable = coverage >= .1 and pilot_error < 1.5 and coherence > (.65 if layout.top_bin<=13 else .4)
        return Decoded('picture_only' if usable else 'lost',
                       values=values if usable else None, pilot_error=pilot_error,
                       coverage=coverage, tier=tier if usable else 'none',
                       extra={'timing_drift_samples': timing_drift,
                              'clock_error': float(np.median(clock_errors)) if clock_errors else None})
    flags, absolute, index, count, stamp = fields
    return Decoded('received' if pilot_error < .15 else 'degraded', values=values,
                   absolute=absolute, index=index, count=count, stamp_ms=stamp,
                   flags=flags, pilot_error=pilot_error, coverage=coverage,
                   identity='verified_header', tier=tier,
                   extra={'timing_drift_samples': timing_drift,
                              'clock_error': float(np.median(clock_errors)) if clock_errors else None})


@lru_cache(maxsize=8)
def _sync_filter(cutoff):
    return firwin(33,cutoff,fs=RATE)


@lru_cache(maxsize=4)
def _conditioning_filter(cutoff):
    return butter(4, cutoff, btype='highpass', fs=RATE, output='sos')


def _recovery_quality(result):
    if result.values is None or not np.isfinite(result.values).all():
        return (False, False, -float('inf'))
    return (True, result.identity == 'verified_header',
            -float(result.pilot_error if result.pilot_error is not None else np.inf))

