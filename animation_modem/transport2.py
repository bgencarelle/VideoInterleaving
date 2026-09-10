"""V2 analog image transport with frame-local timing and graded recovery.

SEARCH finds a preamble with a decimated coarse bank and a small timing fit.
BODY waits only for that packet, resamples once, and emits immediately. Prior
rate is a hint, not a prerequisite; FPS, wall time and inter-packet spacing are
not receiver inputs. The capture rate is 48 kHz, but playback rate is measured.

Image coefficients use channel-aware Wiener reconstruction, not division by
weak allocation gains. Identity requires CRC; coherent training can still
recover a picture without identity. No pixels are copied from previous frames.
The narrowest preset also matches the surviving low-band part of the existing
wideband preamble. Static preset/profile/allocation must match both endpoints.
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
SYNC_SCALES = np.round(np.exp(np.linspace(np.log(.5), np.log(2.0), 29)), 6)


def _sync_waveform(seed=41015):
    rng = np.random.default_rng(seed)
    spectrum = np.zeros(N + 1, complex)          # irfft(n=2N) wants N+1 bins
    spectrum[6:109] = np.exp(1j*rng.uniform(-np.pi, np.pi, 103))
    wave = np.fft.irfft(spectrum, n=2*N)
    return wave*(.65/np.max(np.abs(wave)))


SYNC = _sync_waveform()
SYNC_ENERGY = float(np.dot(SYNC, SYNC))


@dataclass(frozen=True)
class Layout:
    """Everything the wire format needs, derived from two choices."""
    top_bin: int = 54          # highest carrier; 54 -> 20.25 kHz, 27 -> 10.1 kHz
    image_symbols: int = 15    # more symbols -> bigger picture, lower frame rate
    name: str = 'wide'
    progressive: bool = False

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
        """The lowest data carriers -- the ones that survive tape and roll-off."""
        return self.data_bins[:min(20, len(self.data_bins))]

    @cached_property
    def header_symbols(self):
        return -(-HEADER_SLOTS//len(self.header_bins))

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

def _scaled_sync(scale):
    """Preamble with received duration `scale` times its nominal duration."""
    if scale == 1.0:
        return SYNC
    length = int(round(len(SYNC)*scale))
    return np.interp(np.arange(length)/scale, np.arange(len(SYNC)), SYNC)


SYNC_BANK = [(float(s), _scaled_sync(float(s))) for s in SYNC_SCALES]
SYNC_BANK = [(s, t, float(np.dot(t, t))) for s, t in SYNC_BANK]


def find_preamble(samples, limit=None, bank=None):
    """Locate the preamble and the source's speed in one pass.

    Returns (position, rate, score). `rate` is received duration over nominal
    duration minus one, so 0.5x playback gives +1.0.
    """
    best = (None, 0.0, 0.0)
    for scale, template, energy in (bank or SYNC_BANK):
        for channel in range(samples.shape[1]):
            score = sync_correlation(samples[:, channel], limit, template, energy)
            if not len(score):
                continue
            at = int(np.argmax(score))
            if score[at] > best[2]:
                best = (at, scale - 1.0, float(score[at]))
    return best


@lru_cache(maxsize=4)
def _sinc_weight_table(taps=8, phases=4096):
    """Quantized windowed-sinc weights for the live packet path."""
    offsets = np.arange(-taps+1, taps+1, dtype=float)
    fraction = np.arange(phases, dtype=float)[:, None]/phases
    weights = (np.sinc(fraction-offsets[None, :]) *
               np.sinc((fraction-offsets[None, :])/taps))
    weights /= weights.sum(axis=1, keepdims=True)
    return weights


# The timing check/preamble gives us one scale for the complete packet.  Keep
# the integer sample walk and quantised fractional phases cached so the live
# decoder does not rebuild arange/floor/sinc arrays for every packet.
_PLAN_SCALE_Q = 65536
_PLAN_PHASE_Q = 4096


@lru_cache(maxsize=32)
def _resample_plan(length, scale_key, phase_key, taps):
    scale = float(scale_key) / _PLAN_SCALE_Q
    phase = float(phase_key) / _PLAN_PHASE_Q
    position = phase + np.arange(length, dtype=np.float64) * scale
    base = np.floor(position).astype(np.int32)
    frac = np.minimum((position - base) * _PLAN_PHASE_Q,
                      _PLAN_PHASE_Q - 1).astype(np.intp)
    offsets = np.arange(-taps + 1, taps + 1, dtype=np.int32)
    # Relative base positions are independent of the packet's absolute start.
    return base, frac, offsets


def resample_packet(samples, rate, length, taps=8, offset=0.0, fast=False):
    """Resample one packet, using a cheap path for normal small clock error.

    The live path uses a cached fractional-delay sinc table; the uncached path
    is retained for the timing fitter, where each call uses a different delay.
    """
    if fast:
        absolute = int(np.floor(offset))
        fraction = float(offset - absolute)
        phase_key = int(round(fraction * _PLAN_PHASE_Q))
        if phase_key >= _PLAN_PHASE_Q:
            absolute += 1
            phase_key = 0
        scale_key = int(round((1.0 + rate) * _PLAN_SCALE_Q))
        base, phase, offsets = _resample_plan(length, scale_key, phase_key, taps)
        index = np.clip(absolute + base[:, None] + offsets[None, :],
                        0, len(samples)-1)
        weights = _sinc_weight_table(taps)[phase]
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


def encode(values, layout, coder, absolute, index, count, stamp_ms=0, flags=0):
    """Build one packet from already-prepared image values in [-1, 1]."""
    if len(values)!=coder.count or not np.isfinite(values).all():
        raise ValueError('Expected one finite value per source coefficient')
    if len(values) > layout.capacity:
        raise ValueError(f'{len(values)} values exceed capacity {layout.capacity}')
    carriers = layout.carriers
    data = np.searchsorted(carriers, layout.data_bins)
    pilots = np.searchsorted(carriers, layout.pilots)
    header = np.searchsorted(carriers, layout.header_bins)
    grid = np.zeros((layout.symbols, len(carriers), 2), complex)
    grid[0, :, 0] = 1
    grid[1, :, 1] = 1

    raw = pack_header(flags, layout.top_bin, absolute, index, count, stamp_ms,
                      magic=b'V3' if layout.progressive else b'V2')
    bits = np.unpackbits(np.frombuffer(raw, np.uint8)).reshape(HEADER_SLOTS, 2)
    qpsk = ((bits[:, 0]*2.-1) + 1j*(bits[:, 1]*2.-1))/np.sqrt(2)
    room = layout.header_symbols*len(header)
    spread = np.resize(qpsk, room).reshape(layout.header_symbols, len(header))
    for s in range(layout.header_symbols):
        # Identical on both channels: stereo repetition is the diversity that
        # keeps identity alive when one side of the tape is weaker.
        grid[2+s, header, 0] = spread[s]*HEADER_GAIN
        grid[2+s, header, 1] = spread[s]*HEADER_GAIN

    room = layout.image_symbols*len(data)*4
    sent = np.zeros(room)
    sent[coefficient_slots(layout, tuple(coder.shapes))] = coder.forward(values)
    block = sent.reshape(layout.image_symbols, len(data), 2, 2)
    grid[2+layout.header_symbols:, data, :] = (block[..., 0] + 1j*block[..., 1])*IMAGE_GAIN
    grid[2:, pilots, :] = 1

    spectrum = np.zeros((layout.symbols, N//2+1, 2), complex)
    spectrum[:, carriers, :] = grid*phases(layout)
    wave = np.fft.irfft(spectrum, n=N, axis=1)
    wave = np.concatenate([wave[:, -CP:], wave], axis=1).reshape(-1, 2)
    out = np.zeros((layout.frame, 2), np.float32)
    out[16:16+len(SYNC), 0] = SYNC
    out[SYNC_LEN:layout.packet] = wave
    # Preserve training/body ratios so the receiver's channel estimate undoes
    # this scale. DCT DC terms can otherwise clip badly on PCM/audio output.
    peak = float(np.max(np.abs(out)))
    if peak > .95:
        out *= .95/peak
    return out


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


def _equalise(body, layout):
    carriers = layout.carriers
    spectrum = rfft(body, n=N, axis=1, workers=1)
    received = spectrum[:, carriers, :]
    _, _, _, unused, inverse_phase = _decode_tables(layout)
    h = np.stack([received[0]*inverse_phase[0, :, 0, None],
                  received[1]*inverse_phase[1, :, 1, None]], axis=-1)
    noise = max(float(np.mean(np.abs(spectrum[:, unused])**2)), 1e-12)
    hH = h.conj().transpose(0, 2, 1)
    gram = hH @ h
    gram[:, 0, 0] += 4*noise
    gram[:, 1, 1] += 4*noise
    det = gram[:, 0, 0]*gram[:, 1, 1] - gram[:, 0, 1]*gram[:, 1, 0]
    adj = np.empty_like(gram)
    adj[:, 0, 0], adj[:, 1, 1] = gram[:, 1, 1], gram[:, 0, 0]
    adj[:, 0, 1], adj[:, 1, 0] = -gram[:, 0, 1], -gram[:, 1, 0]
    inverse = (adj/det[:, None, None]) @ hH
    weights = np.clip(np.real(np.diagonal(inverse @ h, axis1=1, axis2=2)), 0, 1)
    equal = np.einsum('kij,skj->ski', inverse, received)*inverse_phase
    variance = .5*noise*np.sum(np.abs(inverse)**2, axis=-1)/IMAGE_GAIN**2
    coherence = abs(np.sum(h[1:]*h[:-1].conj())) / max(
        np.sqrt(np.sum(abs(h[1:])**2)*np.sum(abs(h[:-1])**2)), 1e-20)
    return equal, weights, variance, float(coherence)


def decode_packet(samples, layout, coder):
    carriers = layout.carriers
    data, pilots, header, _, _ = _decode_tables(layout)
    body = samples[SYNC_LEN:layout.packet].reshape(layout.symbols, SYMBOL, 2)[:, CP-4:CP-4+N]
    equal, weights, variance, coherence = _equalise(body, layout)

    timing_drift = 0.0
    for channel in range(2):
        keep = weights[pilots, channel] > .6
        if np.count_nonzero(keep) >= 2:
            bins = layout.pilots[keep].astype(float)
            angles = np.unwrap(np.angle(equal[2:, pilots[keep], channel]), axis=1)
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
                equal[2:, :, channel] *= np.exp(-1j*(slope[:, None]*carriers + offset[:, None]))
    pilot_error = float(np.sqrt(np.mean(np.abs(equal[2:, pilots] - 1)**2)))

    end = 2 + layout.header_symbols
    fields = None
    # Both channels carry the same symbols, so their mean is a genuine
    # diversity combine; fall back to each channel alone if one is corrupted.
    for pick in (equal[2:end, header].mean(axis=-1),
                 equal[2:end, header, 0], equal[2:end, header, 1]):
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
                       extra={'timing_drift_samples': timing_drift})
    flags, absolute, index, count, stamp = fields
    return Decoded('received' if pilot_error < .15 else 'degraded', values=values,
                   absolute=absolute, index=index, count=count, stamp_ms=stamp,
                   flags=flags, pilot_error=pilot_error, coverage=coverage,
                   identity='verified_header', tier=tier,
                   extra={'timing_drift_samples': timing_drift})


def sync_correlation(x, limit=None, template=None, energy=None):
    from scipy.signal import correlate
    if template is None:
        template, energy = SYNC, SYNC_ENERGY
    x = np.asarray(x, float)
    n = len(x) - len(template) + 1
    if limit is not None:
        n = max(0, min(n, limit))
    if n <= 0:
        return np.empty(0)
    peak = float(np.max(np.abs(x)))
    if peak == 0:
        return np.zeros(n)
    x = x/peak
    cumulative = np.empty(len(x)+1)
    cumulative[0] = 0.
    np.cumsum(x*x, out=cumulative[1:])
    window = cumulative[len(template):] - cumulative[:-len(template)]
    floor = 64*np.finfo(float).eps*float(np.max(window))
    corr = correlate(x[:n+len(template)-1], template, mode='valid', method='auto')
    score = np.zeros(n)
    ok = window[:n] > floor
    score[ok] = np.abs(corr[ok])/np.sqrt(window[:n][ok]*energy)
    return np.minimum(score, 1.0)


@lru_cache(maxsize=128)
def _template(scale):
    ratio = Fraction(float(scale)).limit_denominator(512)
    return resample_poly(SYNC, ratio.numerator, ratio.denominator)


@lru_cache(maxsize=8)
def _coarse_bank(min_speed, max_speed):
    scales = np.geomspace(1/max_speed, 1/min_speed,
                         int(np.ceil(np.log(max_speed/min_speed)/.045))+1)
    return [(s, resample_poly(_template(float(s)), 1, 8)) for s in scales]


@lru_cache(maxsize=8)
def _sync_filter(cutoff):
    return firwin(33,cutoff,fs=RATE)


def _fit_sync(samples, at, scale, reach=.03, cutoff=None, iterations=7):
    """Refine only the preamble, not five complete image decodes."""
    radius = 12
    left = max(0, int(at)-radius)
    local = samples[left: min(len(samples), int(at+280*scale)+radius)]
    best = (0., float(at-left), scale)
    for s in (np.linspace(scale*(1-reach), scale*(1+reach), 13) if reach else ()):
        template = _scaled_sync(float(s))
        energy = float(template @ template)
        for c in range(2):
            scores = sync_correlation(local[:, c], 2*radius+1, template, energy)
            if len(scores):
                p = int(np.argmax(scores))
                if scores[p] > best[0]:best = (float(scores[p]), float(p), float(s))
    _, pos, s = best
    reference=SYNC
    def filtered(z):
        if cutoff is None:return z
        return np.stack([np.convolve(z[:,c],_sync_filter(cutoff),'same') for c in range(z.shape[1])],axis=1)
    if cutoff is not None:
        reference=np.convolve(SYNC,_sync_filter(cutoff),'same')
    energy=float(reference @ reference)
    # Two-parameter timing loop over a single preamble. Project out gain, then
    # fit timing error and dilation together; no general-purpose optimizer and
    # no repeated full-frame inverse FFTs.
    for _ in range(iterations):
        z=filtered(resample_packet(local,s-1,len(SYNC),offset=pos))
        scores=abs(reference @ z)/np.sqrt(energy*np.maximum((z*z).sum(0),1e-20))
        c=int(np.argmax(scores));z=z[:,c].astype(float)
        gain=float(reference @ z)/energy
        derivative=(filtered(resample_packet(local,s-1,len(SYNC),offset=pos+.05))[:,c]
                    -filtered(resample_packet(local,s-1,len(SYNC),offset=pos-.05))[:,c])/.1
        j=np.stack([derivative,derivative*np.arange(len(SYNC))/1000],axis=1)
        j-=reference[:,None]*(reference @ j)[None,:]/energy
        gram=j.T @ j + np.eye(2)*1e-12
        step=np.linalg.solve(gram,j.T @ (z-gain*reference))
        step=np.clip(step,[-1.,-s*2],[1.,s*2])
        pos=max(0,pos-step[0]);s-=step[1]/1000
        if np.max(abs(step))<1e-4:break
    z=filtered(resample_packet(local,s-1,len(SYNC),offset=pos))
    scores=abs(reference @ z)/np.sqrt(energy*np.maximum((z*z).sum(0),1e-20))
    return float(left+pos),float(s),float(scores.max())


@lru_cache(maxsize=4)
def _conditioning_filter(cutoff):
    return butter(4, cutoff, btype='highpass', fs=RATE, output='sos')


def _condition(samples, cutoff):
    # Applied only to buffered search/packet windows; zero phase avoids adding
    # a causal filter's group delay to the playback-rate estimate. No lookahead
    # beyond the already available samples or the current packet is required.
    return sosfiltfilt(_conditioning_filter(cutoff), samples, axis=0)


def _recovery_quality(result):
    if result.values is None or not np.isfinite(result.values).all():
        return (False, False, -float('inf'))
    return (True, result.identity == 'verified_header',
            -float(result.pilot_error if result.pilot_error is not None else np.inf))


@lru_cache(maxsize=8)
def _channel_training(layout):
    """Known preamble/training prefix and a cached 128-tap stereo FIR fit.

    No image, header identity or previous decoded frame is used as training.
    Ignore the first 128 received samples to reduce previous-packet tail bias.
    """
    taps = 128
    spectrum = np.zeros((2,N//2+1,2),complex)
    ph = phases(layout)
    spectrum[0,layout.carriers,0] = ph[0,:,0]
    spectrum[1,layout.carriers,1] = ph[1,:,1]
    wave = np.fft.irfft(spectrum,n=N,axis=1)
    wave = np.concatenate([wave[:,-CP:],wave],axis=1).reshape(-1,2)
    reference = np.zeros((SYNC_LEN+2*SYMBOL,2))
    reference[16:16+len(SYNC),0] = SYNC
    reference[SYNC_LEN:] = wave
    padded = np.pad(reference,((taps-1,0),(0,0)))
    design = np.lib.stride_tricks.sliding_window_view(padded,taps,axis=0)
    design = design[:,:,::-1].reshape(len(reference),-1)[taps:]
    inverse = np.linalg.solve(design.T @ design + np.eye(2*taps)*1e-5,design.T)
    return design,inverse


def _undo_channel_memory(samples,layout):
    """Optional same-packet inverse for EQ tails longer than the cyclic prefix.

    Fit the stereo impulse response from known samples. Regularize spectral
    inversion to limit boost at weak frequencies; normal decoding subsequently
    estimates residual channel error. Caller keeps this only if recovery improves.
    """
    taps = 128
    design,inverse = _channel_training(layout)
    observed = samples[taps:SYNC_LEN+2*SYMBOL]
    coefficients = inverse @ observed
    residual = observed-design @ coefficients
    fit_error = float(np.mean(residual**2)/max(np.mean(observed**2),1e-20))
    if not np.isfinite(fit_error) or fit_error > .1:
        return None,fit_error
    impulse = coefficients.reshape(2,taps,2).transpose(2,0,1)
    size = 1 << (len(samples)+2*taps-1).bit_length()
    channel = np.fft.rfft(impulse,n=size,axis=-1).transpose(2,0,1)
    adjoint = channel.conj().transpose(0,2,1)
    gram = adjoint @ channel
    power = float(np.mean(np.abs(channel)**2))
    regularizer = max(power*max(1e-4,fit_error),1e-12)
    equalizer = np.linalg.solve(gram+np.eye(2)[None]*regularizer,adjoint)
    signal = np.fft.rfft(samples,n=size,axis=0)
    restored = np.fft.irfft(np.einsum('kij,kj->ki',equalizer,signal),n=size,axis=0)
    return restored[:len(samples)],fit_error


class Receiver:
    """Incremental SEARCH -> BODY -> emit. No frame-spacing or wall-clock input.

    A previous scale is only an acquisition hint. Every packet contains its own
    rate reference; cold acquisition is decimated and bounded. No next packet or
    artificial EOF padding is needed. Memory is bounded even for giant WAV reads.
    """
    def __init__(self, layout, coder, threshold=.4, rate_window=None,
                 min_speed=.25, max_speed=2.0, recovery=True, fast=False):
        if not (0 < min_speed <= 1 <= max_speed and min_speed >= .25 and max_speed <= 2):
            raise ValueError('Supported search range: .25 <= min_speed <= 1 <= max_speed <= 2')
        if coder.count > layout.capacity:raise ValueError('Source coder exceeds layout capacity')
        self.layout, self.coder, self.threshold = layout, coder, threshold
        self.recovery = bool(recovery)
        self.fast = bool(fast)
        self.min_speed, self.max_speed = min_speed, max_speed
        self.bank = _coarse_bank(min_speed, max_speed)
        self.lengths=np.array([len(t) for _,t in self.bank])
        self.templates=np.stack([np.pad(t,(0,int(self.lengths.max())-len(t))) for _,t in self.bank])
        self.energies=np.sum(self.templates**2,axis=1)
        self.keep = int(np.ceil(280/min_speed))+32
        self.sync_cutoff=(3000 if layout.progressive else
                          layout.band[1] if layout.top_bin<=13 else None)
        self.reset()

    def reset(self, preserve_timing=False):
        # A capture queue drop creates a gap, but it does not change the
        # device/tape rate. Live callers can discard the partial packet and
        # retain the last measured rate so reacquisition stays on the cheap
        # locked path instead of restarting the wide cold search.
        rate = self.rate if preserve_timing else 0.
        confidence = self.confidence if preserve_timing else 0.
        # Retain storage on reset; sample validity is bounded by the view.
        if not hasattr(self, '_storage'):
            self._storage = np.empty((32768, 2), np.float32)
        self._write_end = 0
        self.buffer = self._storage[:0]
        self.offset = 0
        self.rate, self.confidence = rate, confidence
        self.pending = None
        self.search_after = 272
        self.acquire_ms = 0.
        self.acquisition_path = 'raw'
        self.locked_packets = 0
        self._track_needed = False

    def _fit(self, x, at, scale, reach=.03):
        # Preserve all available timing information on clean audio. Low-band
        # fitting is a fallback for roll-off, never forced on a clean signal.
        iterations = 3 if self.confidence else 7
        raw = _fit_sync(x, at, scale, reach=reach, iterations=iterations)
        if raw[2] >= .85 or self.sync_cutoff is None:
            return raw
        narrowed = _fit_sync(x, at, scale, reach=reach, cutoff=self.sync_cutoff,
                             iterations=iterations)
        return narrowed if narrowed[2] >= self.threshold else raw

    def _acquire(self):
        x = self.buffer[:max(2048, 2*self.keep)]
        if not np.any(x):return None
        self.acquisition_path = 'raw'
        found = self._acquire_window(x)
        if found is None and len(x) >= 272:
            # The preamble begins above 281 Hz even at quarter speed. This
            # search-only cutoff can reject hum without filtering image data.
            found = self._acquire_window(_condition(x,120))
            if found is not None:self.acquisition_path = 'conditioned'
        if found is not None:
            speed = 1/found[1]
            if not self.min_speed*.98 <= speed <= self.max_speed*1.02:
                return None
        return found

    def _acquire_window(self, x):
        scale = 1+self.rate
        if self.confidence and len(x)<np.ceil(272*scale):return None
        t = SYNC if abs(scale-1)<1e-6 else _template(float(scale))
        # Earliest peak, never the strongest peak in the entire file.
        for c in range(2):
            scores = sync_correlation(x[:,c], template=t, energy=float(t@t))
            hits = np.flatnonzero(scores >= max(self.threshold,.55))
            if len(hits):
                first = int(hits[0]);at = first+int(np.argmax(scores[first:first+8]))
                if scores[at]>.999 and scale==1:return float(at),1.,float(scores[at])
                # Once locked, a strong correlation with the previous packet's
                # rate is enough. A short PLL fit is only needed for cold
                # acquisition or when the tape speed has moved enough to
                # weaken this cheap test.
                locked_cutoff = .42 if self.fast else .60
                if self.confidence and scores[at] >= locked_cutoff:
                    # Update fractional start and dilation from known samples.
                    # At most three local steps, no scale bank or trial image decodes.
                    if not self._track_needed and scores[at] >= .995:
                        self.locked_packets += 1
                        return float(at), float(scale), float(scores[at])
                    tracked = _fit_sync(x, at, scale, reach=0, iterations=3)
                    if tracked[2] >= max(self.threshold, scores[at]-.05):
                        self.locked_packets += 1
                        return tracked
                fit = self._fit(x,at,scale,reach=.004 if self.confidence else .03)
                if fit[2]>=self.threshold:return fit
        if len(x)<256:return None
        coarse = np.asarray(resample_poly(x,1,8,axis=0),dtype=float)
        candidates=[]
        # All coarse correlations share one window/energy pass. Recomputing
        # normalization and dispatching scipy.correlate for each scale/channel
        # costs more than the arithmetic, especially when listening to hiss.
        n=len(coarse);width=self.templates.shape[1]
        windows=np.lib.stride_tricks.sliding_window_view(
            np.pad(coarse,((0,width-1),(0,0))),width,axis=0)
        correlation=np.einsum('sk,pck->spc',self.templates,windows)
        cumulative=np.concatenate([np.zeros((1,2)),np.cumsum(coarse**2,axis=0)])
        start=np.arange(n)[None,:];end=start+self.lengths[:,None]
        energy=cumulative[np.minimum(end,n)]-cumulative[start]
        scores=abs(correlation)/np.sqrt(np.maximum(energy,1e-20)*self.energies[:,None,None])
        floor=64*np.finfo(float).eps*np.maximum(energy.max(axis=1,keepdims=True),1e-20)
        scores=np.where((end<=n)[:,:,None] & (energy>floor),np.minimum(scores,1),0)
        peaks=np.argmax(scores,axis=1)
        for k,(s,t) in enumerate(self.bank):
            for c in range(2):
                at=int(peaks[k,c]);score=float(scores[k,at,c])
                if score > min(.8,max(.6,4/np.sqrt(len(t)))):
                    p=at*8;end=p+round(len(SYNC)*s);tail=max(8,round(16*s))
                    # The existing wire preamble is followed by a quiet
                    # guard. Reject stationary hiss cheaply before fitting
                    # timing; do not spend a PLL fit on every noise peak.
                    if end+tail<=len(x):
                        signal=float(np.mean(x[p:end,c]**2))
                        quiet=float(np.mean(x[end:end+tail,c]**2))
                        if signal>4*max(quiet,1e-15):
                            candidates.append((score,p,float(s)))
        # Verify the strongest coarse hypotheses in chronological order.
        for _,at,s in sorted(sorted(candidates,reverse=True)[:12],key=lambda p:p[1]):
            fit=self._fit(x,at,s)
            if fit[2]>=self.threshold:return fit
        return None

    def feed(self, block):
        block=np.asarray(block,np.float32)
        if block.ndim!=2 or block.shape[1]!=2 or not np.isfinite(block).all():
            raise ValueError('Receiver requires finite stereo samples')
        out=[]
        # Fixed internal ingestion makes results independent of the caller's
        # chunk size and prevents a huge WAV block from skipping early packets.
        for start in range(0,len(block),256):
            self._append(block[start:start+256])
            out.extend(self._drain())
        return out

    def _append(self, block):
        count = len(self.buffer)
        need = count + len(block)
        if self._write_end + len(block) > len(self._storage):
            if need > len(self._storage):
                storage = np.empty((max(need, 2*len(self._storage)), 2), np.float32)
                storage[:count] = self.buffer
                self._storage = storage
            else:
                self._storage[:count] = self.buffer.copy()
            self._write_end = count
        begin = self._write_end-count
        self._storage[self._write_end:self._write_end+len(block)] = block
        self._write_end += len(block)
        self.buffer = self._storage[begin:self._write_end]

    def flush(self):
        return self._drain(final=True)

    def _drain(self, final=False):
        out=[]
        while True:
            if self.pending is None:
                if len(self.buffer)<self.search_after and not final:break
                if len(self.buffer)<128:break
                searching=time.perf_counter()
                acquired=self._acquire()
                self.acquire_ms+=(time.perf_counter()-searching)*1000
                if acquired is None:
                    self.search_after=len(self.buffer)+128
                    if len(self.buffer)>self.keep:self._drop(len(self.buffer)-self.keep)
                    break
                at,scale,score=acquired
                self.search_after=272
                self.pending=(at-16*scale,scale,score)
            begin,scale,score=self.pending
            end=begin+(self.layout.packet-1)*scale+1
            if len(self.buffer)<end-(2 if final else 0):break
            started=time.perf_counter()
            if scale==1 and begin==int(begin) and begin>=0:
                straight=self.buffer[int(begin):int(begin)+self.layout.packet]
            else:
                straight=resample_packet(self.buffer,scale-1,self.layout.packet,offset=begin,
                                         fast=self.fast)
            result=decode_packet(straight,self.layout,self.coder)
            input_path = 'raw'
            if self.recovery and (result.identity != 'verified_header' or result.pilot_error > .05):
                # Rate correction has returned the carriers to nominal Hz.
                # Compare the same packet with rumble removed; keep raw on ties.
                alternative = decode_packet(_condition(straight,120),self.layout,self.coder)
                if _recovery_quality(alternative) > _recovery_quality(result):
                    result, input_path = alternative, 'conditioned'
            if self.recovery and (result.identity != 'verified_header' or result.pilot_error > .15):
                restored,fit_error = _undo_channel_memory(straight,self.layout)
                if restored is not None:
                    alternative = decode_packet(restored,self.layout,self.coder)
                    if _recovery_quality(alternative) > _recovery_quality(result):
                        result,input_path = alternative,'eq_corrected'
                        result.extra['channel_fit_error'] = fit_error
            result.rate_error=scale-1
            result.rate_confidence=score
            result.extra.update(sync_score=score, at=self.offset+begin,
                                input_path=input_path, acquisition_path=self.acquisition_path,
                                playback_speed=1/scale,
                                decode_ms=(time.perf_counter()-started)*1000,
                                acquire_ms=self.acquire_ms)
            result.extra['receive_cpu_ms']=result.extra['decode_ms']+self.acquire_ms
            self.pending=None
            if result.values is not None:
                self._track_needed = (result.extra.get('timing_drift_samples', 0.) > .15
                                      or result.pilot_error > .05)
                self.rate,self.confidence=scale-1,score
                out.append(result)
                self.acquire_ms=0.
                self._drop(max(1,int(np.floor(begin+self.layout.packet*scale))))
            else:
                self.confidence=0.
                self.rate=0.
                self._drop(max(1,int(begin+16*scale+1)))
        return out

    def _drop(self,n):
        n=max(1,min(int(n),len(self.buffer)))
        self.buffer=self.buffer[n:]
        self.offset+=n
        self.search_after=max(272,self.search_after-n)
