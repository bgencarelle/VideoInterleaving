"""Proposed wire format v2: self-clocking, band-selectable, gracefully graded.

Differences from v1, each one measured rather than assumed:

* The layout is parametric. `top_bin` chooses the occupied band and
  `image_symbols` trades frame rate against picture size. Symbol length stays
  at 128+16 samples in every preset, so narrowing the band for tape does not
  lengthen a symbol and does not cost flutter tolerance.

* The receiver finds the playback rate and the packet position together, by
  correlating a bank of preambles scaled across 0.5x to 2x. It never assumes
  the source runs at 48 kHz. v1 acquired with one fixed-rate template and
  estimated the rate afterwards from packet spacing, which needed three good
  packets and capped out a few percent off speed.

  Cyclic-prefix autocorrelation is the textbook way to do this and it was
  tried first: the prefix is only CP/(N+CP) of the signal, so the statistic
  sits near 0.11 at best, and against a noise-like OFDM body the peak did not
  separate from the floor -- measured estimates were wrong by tens of percent
  at every speed. The scaled bank is dumber, costs a few hundred microseconds,
  and actually works.

* Image values are sent as DCT coefficients under a fixed power allocation
  derived from the bake. Important coefficients go out loud and fine detail
  goes out quiet, so a failing channel loses detail before it loses structure.
  Measured +5 dB on a good channel and +13 dB on a bad one -- in a standalone
  experiment where the receiver knew each carrier's gain and formed an MMSE
  estimate. THAT IS NOT WHAT THIS FILE DOES YET. decode_packet computes
  exactly that information as `weights` and discards it for the image path,
  and SourceCoder.inverse divides by the allocation gain rather than
  Wiener-filtering with it, so a coefficient given little power has its noise
  amplified by 1/gain. Measured today, allocation LOSES about 3 dB on a
  rolled-off channel. Joining the weights to the estimate is the change that
  makes the gain real, and until then the rest of v2 is not worth adopting. Position is not a lever -- reordering alone measured +0.5 dB,
  because an orthonormal transform does not care which slot a coefficient
  occupies. Do not pair loud coefficients with robust carriers either; that
  multiplies both penalties together and measured 2-5 dB WORSE than leaving
  them independent.

* The header rides the lowest carriers only, at boosted amplitude. In v1 it
  was spread across every data bin, so it was only as strong as the weakest
  one and reliably died while the picture was still watchable.

Each packet remains independently decodable. No temporal state, no deltas.
"""
import struct
import zlib
from dataclasses import dataclass, field

import numpy as np
from scipy.fft import dctn, idctn

RATE = 48000
N, CP = 128, 16
SYMBOL = N + CP
LOW_BIN = 3
SYNC_LEN = 288
GUARD = 32                 # idle tail; also the slack the rate estimator needs
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

    @property
    def carriers(self):
        return np.arange(LOW_BIN, self.top_bin + 1)

    @property
    def pilots(self):
        c = self.carriers
        return c[np.linspace(3, len(c)-4, 4).round().astype(int)]

    @property
    def data_bins(self):
        return np.array([k for k in self.carriers if k not in set(self.pilots.tolist())])

    @property
    def header_bins(self):
        """The lowest data carriers -- the ones that survive tape and roll-off."""
        return self.data_bins[:min(20, len(self.data_bins))]

    @property
    def header_symbols(self):
        return -(-HEADER_SLOTS//len(self.header_bins))

    @property
    def symbols(self):
        return 2 + self.header_symbols + self.image_symbols

    @property
    def packet(self):
        return SYNC_LEN + self.symbols*SYMBOL

    @property
    def frame(self):
        return self.packet + GUARD

    @property
    def fps(self):
        return RATE/self.frame

    @property
    def band(self):
        c = self.carriers
        return (float(c[0]*RATE/N), float(c[-1]*RATE/N))

    @property
    def capacity(self):
        """Real image values carried per packet."""
        return self.image_symbols*len(self.data_bins)*4

    @property
    def max_speed(self):
        """Playback speed above which the top carrier folds past Nyquist."""
        return (RATE/2)/self.band[1]

    def describe(self):
        return (f'{self.name}: {self.band[0]:.0f}-{self.band[1]:.0f} Hz, '
                f'{self.fps:.2f} fps, {self.capacity} values, '
                f'{len(self.data_bins)} data bins, up to {self.max_speed:.2f}x speed')


PRESETS = {
    # today's band and cadence, for a cable or a digital loopback
    'wide': Layout(top_bin=54, image_symbols=15, name='wide'),
    # cassette: 10 kHz ceiling, full picture, slower
    'tape': Layout(top_bin=27, image_symbols=35, name='tape'),
    # cassette: 10 kHz ceiling, keeps the frame rate, smaller picture
    'tape-fast': Layout(top_bin=27, image_symbols=15, name='tape-fast'),
    # worn deck or acoustic coupling
    'narrow': Layout(top_bin=21, image_symbols=44, name='narrow'),
}


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
        if sigma.shape != (self.count,) or not np.all(sigma > 0):
            raise ValueError('Allocation must be one positive weight per value')
        # Power proportional to standard deviation is the analog optimum.
        gains = np.sqrt(sigma/np.mean(sigma))
        self.gains = gains/np.sqrt(np.mean(gains**2))

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

    def inverse(self, sent):
        coeffs = np.asarray(sent, float)/self.gains
        return np.concatenate([idctn(p, norm='ortho').ravel()
                               for p in self._split(coeffs)])


# --------------------------------------------------------------------------
# Rate measurement from the cyclic prefix -- the LTC-style primitive
# --------------------------------------------------------------------------

def _scaled_sync(scale):
    """The preamble as it arrives when the source runs at `scale` speed."""
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


def resample_packet(samples, rate, length, taps=8):
    """Windowed sinc. Linear interpolation loses about half the signal here."""
    position = np.arange(length)*(1.0+rate)
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


def pack_header(flags, top_bin, absolute, index, count, stamp_ms):
    if not (0 <= absolute <= 0xffffffff and 1 <= index <= count <= 0xffff):
        raise ValueError('Frame/index/count outside the header ranges')
    raw = struct.pack(HEADER_FORMAT, b'V2', flags & 0xff, top_bin & 0xff,
                      absolute, index, count, stamp_ms & 0xffffffff)
    return raw + struct.pack('>I', zlib.crc32(raw))


def encode(values, layout, coder, absolute, index, count, stamp_ms=0, flags=0):
    """Build one packet from already-prepared image values in [-1, 1]."""
    if len(values) > layout.capacity:
        raise ValueError(f'{len(values)} values exceed capacity {layout.capacity}')
    carriers = layout.carriers
    data = np.searchsorted(carriers, layout.data_bins)
    pilots = np.searchsorted(carriers, layout.pilots)
    header = np.searchsorted(carriers, layout.header_bins)
    grid = np.zeros((layout.symbols, len(carriers), 2), complex)
    grid[0, :, 0] = 1
    grid[1, :, 1] = 1

    raw = pack_header(flags, layout.top_bin, absolute, index, count, stamp_ms)
    bits = np.unpackbits(np.frombuffer(raw, np.uint8)).reshape(HEADER_SLOTS, 2)
    qpsk = ((bits[:, 0]*2.-1) + 1j*(bits[:, 1]*2.-1))/np.sqrt(2)
    room = layout.header_symbols*len(header)
    spread = np.resize(qpsk, room).reshape(layout.header_symbols, len(header))
    for s in range(layout.header_symbols):
        # Identical on both channels: stereo repetition is the diversity that
        # keeps identity alive when one side of the tape is weaker.
        grid[2+s, header, 0] = spread[s]*HEADER_GAIN
        grid[2+s, header, 1] = spread[s]*HEADER_GAIN

    sent = coder.forward(values)
    room = layout.image_symbols*len(data)*4
    sent = np.resize(np.concatenate([sent, np.zeros(max(0, room-len(sent)))]), room)
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


def _equalise(body, layout):
    carriers = layout.carriers
    spectrum = np.fft.rfft(body, n=N, axis=1)
    received = spectrum[:, carriers, :]
    ph = phases(layout)
    h = np.stack([received[0]/ph[0, :, 0, None], received[1]/ph[1, :, 1, None]], axis=-1)
    unused = [k for k in range(1, N//2+1) if k not in set(carriers.tolist())][:8]
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
    equal = np.einsum('kij,skj->ski', inverse, received)/ph
    return equal, weights


def decode_packet(samples, layout, coder):
    carriers = layout.carriers
    data = np.searchsorted(carriers, layout.data_bins)
    pilots = np.searchsorted(carriers, layout.pilots)
    header = np.searchsorted(carriers, layout.header_bins)
    body = samples[SYNC_LEN:layout.packet].reshape(layout.symbols, SYMBOL, 2)[:, CP-4:CP-4+N]
    equal, weights = _equalise(body, layout)

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
            if magic == b'V2' and top == layout.top_bin and 1 <= index <= count:
                fields = (hflags, absolute, index, count, stamp)
                break

    block = equal[end:, data]/IMAGE_GAIN
    sent = np.stack([block.real, block.imag], axis=-1).ravel()
    per = np.broadcast_to(weights[data][None, :, :, None],
                          (layout.image_symbols, len(data), 2, 2)).ravel()
    coverage = float(np.mean(per >= .55))
    values = coder.inverse(sent[:coder.count])
    tier = ('best' if coverage >= .95 else 'better' if coverage >= .7
            else 'good' if coverage >= .35 else 'poor')
    if fields is None:
        usable = coverage >= .1 and pilot_error < 1.5
        return Decoded('picture_only' if usable else 'lost',
                       values=values if usable else None, pilot_error=pilot_error,
                       coverage=coverage, tier=tier if usable else 'none')
    flags, absolute, index, count, stamp = fields
    return Decoded('received' if pilot_error < .15 else 'degraded', values=values,
                   absolute=absolute, index=index, count=count, stamp_ms=stamp,
                   flags=flags, pilot_error=pilot_error, coverage=coverage,
                   identity='verified_header', tier=tier)


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


class Receiver:
    """Measure the rate, normalise, then acquire. Newest complete frame wins.

    v1 did this the other way round -- acquired with a fixed-rate template and
    estimated afterwards -- which is why it stopped working a few percent off
    speed. Here nothing downstream sees the source's speed at all.
    """

    def __init__(self, layout, coder, threshold=.4, rate_window=8*3200):
        self.layout = layout
        self.coder = coder
        self.threshold = threshold
        self.rate_window = rate_window
        self.buffer = np.empty((0, 2), np.float32)
        self.offset = 0
        self.rate = 0.0
        self.confidence = 0.0

    def feed(self, block):
        block = np.asarray(block, np.float32)
        if block.ndim != 2 or block.shape[1] != 2:
            raise ValueError('Receiver requires two selected audio channels')
        self.buffer = np.concatenate([self.buffer, block])
        out = []
        while True:
            span = int(self.layout.packet*(1.0+self.rate)) + 2*len(SYNC) + 8
            if len(self.buffer) < span:
                break
            # Cold: search every scale. Locked: the neighbours of the known one.
            bank = SYNC_BANK
            if self.confidence:
                near = sorted(SYNC_BANK, key=lambda e: abs(e[0]-(1+self.rate)))[:3]
                bank = near
            limit = len(self.buffer) - span + 1 + len(SYNC)
            at, rate, score = find_preamble(self.buffer, limit, bank)
            if at is None or score < self.threshold:
                self.confidence = 0.0
                self._drop(max(1, len(self.buffer)-span+1))
                break
            self.rate, self.confidence = rate, score
            # encode() places SYNC 16 samples into the packet, so the preamble
            # peak is 16 nominal samples after the packet actually starts.
            begin = at - int(round(16*(1.0+rate)))
            if begin < 0:
                self._drop(at + 1)
                continue
            need = int(self.layout.packet*(1.0+rate)) + 8
            if begin + need > len(self.buffer):
                break
            window = self.buffer[begin:begin+need]
            # The bank quantises the rate to about 2.6% near unity, which is far
            # too coarse for the header. Refine by decoding a few candidates and
            # keeping the one the CRC and the pilots actually like.
            result, rate = self._refine(window, rate)
            result.rate_error = rate
            result.rate_confidence = score
            result.extra['sync_score'] = score
            result.extra['at'] = self.offset + begin
            out.append(result)
            self._drop(begin + need)
        return out

    def _refine(self, window, coarse, points=5, reach=.016):
        best = None
        for rate in coarse + np.linspace(-reach, reach, points):
            if begin_ok := (self.layout.packet*(1.0+rate) <= len(window)):
                straight = (resample_packet(window, rate, self.layout.packet)
                            if abs(rate) > 5e-4 else window[:self.layout.packet])
                candidate = decode_packet(straight, self.layout, self.coder)
                mark = (candidate.identity == 'verified_header',
                        -(candidate.pilot_error if candidate.pilot_error is not None else 9))
                if best is None or mark > best[0]:
                    best = (mark, candidate, float(rate))
                if mark[0] and -mark[1] < .05:
                    break                       # already clean, stop searching
            del begin_ok
        if best is None:
            return decode_packet(window[:self.layout.packet], self.layout, self.coder), coarse
        return best[1], best[2]

    def _drop(self, n):
        n = max(1, min(int(n), len(self.buffer)))
        self.buffer = self.buffer[n:]
        self.offset += n
