"""Shared modem engine: wire geometry, source coding and packet demodulation.

This is everything both sides need once timing is known. transport3 is the
only transport -- the pre-v3 wire is gone, so everything here is
version-independent:

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
from itertools import combinations
from functools import lru_cache, cached_property
from fractions import Fraction
from dataclasses import dataclass, field

import numpy as np
from scipy.fft import dctn, idctn, rfft
from scipy.signal import resample_poly, firwin, butter, sosfiltfilt, filtfilt

REFERENCE_RATE = 48000
# The rate the sample geometry was designed around, and the only rate this
# package ever *writes down*: it is stamped into WAV headers and used to turn a
# sample count into a nominal duration for reporting. Nothing requests it from
# hardware. The wire format is defined in SAMPLES -- 2768 of them per lean-v3
# frame -- so RECEIVING is indifferent to the capture clock: a faster one just
# oversamples, and the preamble's edge intervals report the cadence.
#
# SENDING is not symmetric, which is why this constant also caps the emitted
# band. Playing the sample array out of a faster device scales the carriers up
# with the clock -- 40.5 kHz at 96 kHz, 81 kHz at 192 kHz -- which is above
# both a cheap DAC's reconstruction filter and any sane receiver's Nyquist. So
# a transmitter resamples to its device instead, holding the emitted band at
# 375-20250 Hz whatever the device is set to. See `band_limited`.
RATE = REFERENCE_RATE      # historical name, kept for existing imports
N, CP = 128, 16
SYMBOL = N + CP
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
    progressive: bool = True
    header_width: int = 20     # header carriers; wider -> fewer header symbols
    header_split: bool = False # halve header symbols by sending different
                               # halves on each channel instead of the same
                               # bits twice, trading diversity for frame rate
    spread_carriers: bool = False  # place the coarsest coefficients across the
                               # whole band, strongest on the middle carriers,
                               # instead of stacking them on the lowest one
    orthogonal_training: bool = False  # drive both channels in both training
                               # symbols instead of one at a time; same peak,
                               # twice the energy, half the estimator error
    dense_header: bool = False # carry image on the data carriers the header
                               # symbols leave idle. The header only occupies
                               # header_width of the band, so the rest of every
                               # header symbol was silence -- 30 of 50 carriers
                               # x 4 symbols on lean-v3, 480 slots, +21.8%.
                               # Free in level terms: the preamble holds the
                               # packet peak either way, so nothing renormalises

    def __post_init__(self):
        if not (isinstance(self.top_bin,int) and 10<=self.top_bin<=63
                and isinstance(self.image_symbols,int) and self.image_symbols>0):
            raise ValueError('Layout requires top_bin 10..63 and positive image_symbols')

    @cached_property
    def carriers(self):
        # Progressive placement: bin 1 is the bottom carrier. The two low-band
        # pilots survive any real channel; bins 1 and 2 carry images.
        return np.arange(1, self.top_bin + 1)

    @cached_property
    def pilots(self):
        # Two low-band pilots survive roll-off; bins 1 and 2 carry images.
        base = np.array([3, 5, 21, 45])
        if self.top_bin >= base[-1]:
            return base
        # A narrower progressive layout cannot use the wide-band pilots:
        # bins 21 and 45 fall outside its carriers entirely, and indexing
        # with them raised IndexError from encode(). Keep the low pair,
        # which is the point of the progressive placement, and fold the
        # upper pair proportionally into whatever band remains.
        upper = np.round(base[2:]*self.top_bin/base[-1]).astype(int)
        pilots = np.unique(np.concatenate([base[:2], upper]))
        pilots = pilots[(pilots >= self.carriers.min()) &
                        (pilots <= self.top_bin)]
        if len(pilots) < 4:
            spare = np.setdiff1d(self.carriers, pilots)
            need = min(4-len(pilots), len(spare))
            pick = spare[np.linspace(0, len(spare)-1, need).round().astype(int)]
            pilots = np.unique(np.concatenate([pilots, pick]))
        return pilots

    @cached_property
    def data_bins(self):
        return np.array([k for k in self.carriers if k not in set(self.pilots.tolist())])

    @cached_property
    def header_bins(self):
        """Where the frame header rides: the lowest data carriers, always.

        The header is deliberately independent of ``spread_carriers``, which
        places the IMAGE coefficients. Colour immunity to EQ comes from
        spreading the image planes; the header gains nothing from riding high,
        and loses a lot. A channel that cuts the top of the band (a narrow path)
        or playback faster than the aliasing speed takes the top carriers out
        first, and a header living up there goes with them -- measured: a
        full-band header loses identity entirely under a 15 kHz lowpass or 2x
        playback, while a low-band header keeps it.

        The old reason for riding mid-band was that bin 1 sits on the declared
        bottom edge, so a bottom roll-off killed a header concentrated there.
        With ``header_width`` spanning many carriers that is no longer a single
        point of failure: measured, a 375 Hz highpass still verifies.

        Widening trades header-symbol count for frame rate: fewer header
        symbols and a higher frame rate, but the header reaches toward the top.
        """
        want = min(max(1, self.header_width), len(self.data_bins))
        return self.data_bins[:want]

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

    def fps_at(self, rate):
        """Frames per second on a clock actually running at `rate`.

        Frame length is a sample count, so the frame rate is whatever the
        hardware clock makes of it: 17.34 fps at 48 kHz, 15.93 at 44.1 kHz.
        Neither is more correct than the other, and neither is requested.
        """
        return rate/self.frame

    def band_at(self, rate):
        """Carrier edges in hertz on a clock actually running at `rate`."""
        c = self.carriers
        return (float(c[0]*rate/N), float(c[-1]*rate/N))

    @cached_property
    def fps(self):
        """Frame rate at the reference clock. For labels, not for scheduling."""
        return self.fps_at(REFERENCE_RATE)

    @cached_property
    def band(self):
        """Carrier edges at the reference clock. For labels, not for filters."""
        return self.band_at(REFERENCE_RATE)

    @cached_property
    def wire_magic(self):
        """Two header bytes that say which wire generation this is.

        Dense layouts need their own, and it is not decoration. lean-v3 and
        lean-v3-dense have the same packet length, the same top_bin and a
        bit-identical header -- only the image carriers differ -- so the CRC
        cannot tell them apart. A receiver trying candidates would verify the
        header of a dense packet against the sparse layout and reconstruct the
        picture at the wrong geometry: garbage, silently, which is exactly what
        the magic exists to prevent. It also stops a transmitter from before
        dense_header being mistaken for one after it.
        """
        if self.dense_header:
            return b'V4'
        return b'V3'

    @cached_property
    def spare_bins(self):
        """Data carriers a header symbol does not use. Idle unless dense."""
        taken = set(self.header_bins.tolist())
        return np.array([b for b in self.data_bins if b not in taken])

    @cached_property
    def header_capacity(self):
        """Image values reclaimed from the header symbols' idle carriers."""
        return self.header_symbols*len(self.spare_bins)*4 if self.dense_header else 0

    @cached_property
    def capacity(self):
        """Real image values carried per packet."""
        return self.header_capacity + self.image_symbols*len(self.data_bins)*4

    @cached_property
    def max_speed(self):
        """Playback speed above which the top carrier folds past Nyquist.

        Nyquist and the carrier both scale with the sample clock, so this ratio
        is pure geometry -- N/(2*top_bin) -- and identical at every rate. It was
        written as (RATE/2)/band[1], which looked rate-dependent and was not.
        """
        return N/(2*self.top_bin)

    def describe(self, rate=REFERENCE_RATE):
        low, high = self.band_at(rate)
        return (f'{self.name}: {low:.0f}-{high:.0f} Hz, '
                f'{self.fps_at(rate):.2f} fps, {self.capacity} values, '
                f'{len(self.data_bins)} data bins, up to {self.max_speed:.2f}x speed '
                f'(at {rate:g} Hz)')


# The v2 wire presets (wide, tape, tape-fast, narrow, lofi) are gone, and so
# is the v3 preset table: every live layout is progressive, and there is
# exactly one of them -- transport3.WIRE. Layouts here are constructed
# directly (tests, probes) so geometry variants stay expressible without a
# table.


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
    """Forward and inverse DCT with allocation, over a list of plane shapes.

    `grids` are the SAMPLING grids; `shapes` the low-frequency corner of each
    that actually goes on the wire. They are equal by default, which is the
    original behaviour: every coefficient of every plane is transmitted, and
    the DCT buys decorrelation for `default_allocation` rather than
    compression.

    Passing larger `grids` samples finer and truncates -- same slot count, more
    source pixels behind it. The transform still happens EXACTLY ONCE, here.
    That is the whole reason this lives in the coder: `forward` runs `dctn` on
    whatever it is handed, so pre-transforming the values outside and passing
    coefficients in transforms them twice. Measured, a double transform smears
    the energy from 0.1% of slots back across 100% of them, which destroys the
    match between the allocation table and the data: a picture that reads 39 dB
    on a clean channel collapses to 8 dB PSNR at -45 dBFS noise, against 33 dB
    for the same geometry transformed once.

    Sampling finer is worth little even done correctly -- +0.1 to +0.5 dB
    typical against box-downsampling at the same slot count, negative on some
    content -- because box-downsampling already captures what those
    coefficients can carry. It is offered because it is nearly free once the
    transform is in the right place, not because it is a resolution win.
    """

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
        # Amplitude, so a truncated packet is still a valid packet at the small
        # size. An ortho DCT of an MxN plane, truncated to mxn and inverted at
        # mxn, comes back scaled by sqrt(mn/MN) -- half amplitude for a 2x grid.
        # Undoing it here means a receiver that knows nothing about grids
        # reconstructs the correctly-exposed low-passed picture rather than a
        # dim one, which is what makes truncation shareable across geometry.
        sigma = (self._allocation() if allocation is None
                 else np.asarray(allocation, float))
        if sigma.shape != (self.count,) or not np.all(np.isfinite(sigma) & (sigma > 0)):
            raise ValueError('Allocation must be one positive weight per value')
        # Power proportional to standard deviation is the analog optimum.
        gains = np.sqrt(sigma/np.mean(sigma))
        self.gains = gains/np.sqrt(np.mean(gains**2))
        self.variance = sigma**2

    def _allocation(self):
        """Weight by spatial frequency ON THE SAMPLING GRID.

        A kept coefficient at index i of a plane sampled twice as finely sits
        at HALF the normalised frequency of index i on the small plane, so the
        table has to be built against the grid or it spends power as though the
        truncated corner were the whole band. With grids == shapes this is
        exactly `default_allocation`.
        """
        if not self.truncated:
            return default_allocation(self.shapes, self.count)
        weights = []
        for (rows, cols), (grid_rows, grid_cols) in zip(self.shapes, self.grids):
            fy = np.arange(rows)[:, None]/max(grid_rows-1, 1)
            fx = np.arange(cols)[None, :]/max(grid_cols-1, 1)
            weights.append((1.0/(1.0 + 12*np.hypot(fy, fx))).ravel())
        return np.resize(np.concatenate(weights), self.count)

    def _split(self, values, shapes):
        out, offset = [], 0
        for shape in shapes:
            n = int(np.prod(shape))
            out.append(values[offset:offset+n].reshape(shape))
            offset += n
        return out

    def forward(self, values):
        """Pixels in (source_count of them), wire slots out (count of them)."""
        planes = self._split(np.asarray(values, float), self.grids)
        kept = []
        for plane, (rows, cols) in zip(planes, self.shapes):
            kept.append(dctn(plane, norm='ortho')[:rows, :cols].ravel())
        return np.concatenate(kept)*self.gains

    def inverse(self, sent, reliability=None, noise_variance=None):
        """Wire slots in, pixels out. Dropped coefficients come back as zero,
        which is the least-energy completion and what truncation implies."""
        if reliability is None:
            coeffs = np.asarray(sent, float) / self.gains
        else:
            rel = np.asarray(reliability, float)
            noise_var = np.asarray(noise_variance, float)

            # `rel` is the residual bias of the MMSE equaliser: the slot arrives
            # as sent ~= gains*rel*x + n. It must be used as the TRUE forward
            # gain here, not floored to a nominal value. Flooring it at 0.25
            # told the estimator a carrier attenuated to rel=0.02 by an EQ cut
            # still had 0.25 of its amplitude, so the de-bias divided by 12x
            # too much and that slot came back ~12x small. Because Y, Cb and Cr
            # slots sit on different carriers, an EQ tilt shrank the planes by
            # different factors -- which is a colour cast, not a noise floor.
            # The floor here is only a division guard; a carrier that is truly
            # dead now decays to zero (grey) instead of to a wrong colour.
            rel_floor = np.maximum(rel, 1e-6)
            gain = self.gains * rel_floor

            denominator = gain ** 2 * self.variance + noise_var
            coeffs = np.divide(np.asarray(sent) * gain * self.variance, denominator,
                               out=np.zeros(self.count), where=denominator > 1e-20)

            # Ensure the DC components of Luma (0), Cb, and Cr are never
            # collapsed by Wiener dampening if the carrier is active.
            dc_slots = [0]
            if len(self.shapes) == 3:
                luma_count = int(np.prod(self.shapes[0]))
                cb_count = int(np.prod(self.shapes[1]))
                dc_slots.extend([luma_count, luma_count + cb_count])

            for dc_idx in dc_slots:
                if dc_idx < len(coeffs) and rel[dc_idx] > 0.5:
                    # Same de-bias as above, or the plane DCs are rescued on a
                    # different scale from their own AC terms -- again a cast.
                    coeffs[dc_idx] = sent[dc_idx] / gain[dc_idx]

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
    """Source coefficient -> wire slot, all layouts progressive.

    Coefficients are ranked by normalized spatial frequency so the coarse image
    lands where the band is healthiest. With spread_carriers the carrier varies
    fastest: the coarsest coefficients go out across ALL carriers rather than
    stacking on one, and carriers are visited middle-out so the highest-energy
    terms sit where no channel rolls off. Same slots, same capacity, different
    mapping.
    """
    count = sum(h*w for h,w in shapes)
    ranks = []
    for h,w in shapes:
        yy,xx = np.mgrid[:h,:w]
        ranks.extend(np.hypot(yy/h,xx/w).ravel())
    source_order = np.argsort(ranks, kind='stable')
    low_to_high = _slot_order(layout)[:count]
    mapping = np.empty(count, dtype=int)
    mapping[source_order] = low_to_high
    mapping.setflags(write=False)
    return mapping


@lru_cache(maxsize=32)
def _slot_order(layout):
    """Wire slots from coarsest-friendly to harshest, over every carrying cell.

    A "cell" is one (symbol, data carrier) that carries image. With
    dense_header there are two blocks of them -- the header symbols' idle
    carriers first, then the image symbols -- laid out in the flat array in
    exactly that order, which is what `encode` and `decode_packet` scatter into
    and gather from.

    Sorting rather than transposing is what lets one expression cover both
    blocks, which are not the same width and so cannot be one rectangular
    array. The keys reproduce the old transposes exactly, and a test asserts
    that for every preset and profile: lexsort takes its LAST key as primary,
    so spread walks symbol, then channel, then real/imaginary, then carrier
    (coarse coefficients across the whole band), and the default walks carrier
    outermost (coarse coefficients stacked on the lowest carrier).
    """
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
    channel = np.tile(np.array([0, 0, 1, 1]), len(symbol)//4)
    part = np.tile(np.array([0, 1, 0, 1]), len(symbol)//4)
    if layout.spread_carriers:
        # Middle-out, so the strongest terms sit where no channel rolls off.
        rank = np.empty(lanes, int)
        rank[np.argsort(np.abs(np.arange(lanes)-(lanes-1)/2), kind='stable')] = \
            np.arange(lanes)
        return np.lexsort((rank[carrier], part, channel, symbol))
    return np.lexsort((part, channel, symbol, carrier))


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


def emit_ratio(rate, reference=REFERENCE_RATE, limit=16):
    """up/down for a transmitter on `rate`, or None when nothing is needed.

    Only rates ABOVE the reference produce a ratio. Below it, the natural
    sample array already sits inside the reference band -- 18.6 kHz at
    44.1 kHz -- and resampling up to reclaim the last 1.6 kHz would push the
    top carrier from 0.84 of that device's Nyquist to 0.92, which is the wrong
    direction for the reconstruction filter on a cheap DAC. Quiet and low is
    the safe failure here.

    The ratio is deliberately coarse. Exact rationals are brutal off the
    44.1 kHz family -- 88.2 kHz is 147/80, and a polyphase filter with a
    transition band that narrow relative to 147x needs ~15000 taps, about
    5 ms per frame on a desktop and far worse on a Pi. 11/6 costs 1129 taps
    and lands 0.23% off. That residual does not have to be corrected, because
    the receiver measures the scale that actually arrives and reports it: it is
    a real 1.0023x transmission, not an error, and the decoder's range is
    +/-15%. Rates that ARE clean multiples (96, 192 kHz) stay exact.
    """
    rate = int(round(rate))
    if rate <= reference:
        return None
    ratio = Fraction(rate, int(reference)).limit_denominator(limit)
    return None if ratio == 1 else (ratio.numerator, ratio.denominator)


@lru_cache(maxsize=8)
def _emit_filter(up, down, top_bin=54, guard=1.185):
    """Anti-image filter with the carriers well clear of its transition band.

    resample_poly's default is ~20*up taps, whose transition band is wide
    enough to start attenuating the top carrier at 20.25 kHz. The passband has
    to reach the carriers and the stopband has to start by the reference
    Nyquist; `guard` is that ratio, 24000/20250, and is not a free parameter.
    """
    edge = top_bin/N                       # carrier top, in reference Nyquist
    width = max(edge*(guard-1), .02)       # transition available, normalised
    taps = int(2*np.ceil(4/(width/max(up, down)))+1)
    return firwin(taps, 1/max(up, down), window=('kaiser', 8.6))


def band_limited(packet, rate, reference=REFERENCE_RATE, top_bin=54, pad=64):
    """Hold a transmitter's carriers at the reference band, whatever its clock.

    This is the one place the two directions differ. A receiver can be clocked
    anywhere, because oversampling costs nothing and the preamble reports the
    cadence. A transmitter cannot: the band it emits is its own clock times the
    carrier geometry, so a device sitting at 96 kHz would put the picture at
    750-40500 Hz, where a low-end DAC's reconstruction filter removes it and no
    receiver below 81 kHz can hear it at all.

    Resampling the packet to the device's rate holds the emitted band at
    375-20250 Hz and the frame at 17.34 fps at every device rate. Note what is
    NOT happening: the device is not asked to change rate, and nothing is
    requested of it. The signal is adapted to the hardware, which is the whole
    point -- the alternative is demanding 48 kHz and failing on anything else.

    Zero padding either side keeps the resampler's transient out of the
    preamble; the frame already opens with 16 idle samples and closes with
    GUARD, so the trimmed result is sample-accurate at both ends.
    """
    packet = np.asarray(packet, np.float32)
    ratio = emit_ratio(rate, reference)
    if ratio is None:
        return packet
    up, down = ratio
    want = int(round(len(packet)*up/down))
    quiet = np.zeros((pad, packet.shape[1]), np.float32)
    out = resample_poly(np.concatenate([quiet, packet, quiet]), up, down,
                        axis=0, window=_emit_filter(up, down, top_bin))
    begin = int(round(pad*up/down))
    out = out[begin:begin+want]
    if len(out) < want:                    # never hand back a short frame
        out = np.concatenate([out, np.zeros((want-len(out), packet.shape[1]))])
    # Restore the level `encode` set. Interpolation reconstructs intersample
    # peaks the original array only implied: measured +2.3 dB here, which
    # would clip a packet encoded to 0.95 and cost far more than the resampling
    # bought. Scaling the whole packet is safe for the same reason encode can
    # normalise at all -- the training symbols carry the scale, so the
    # receiver's channel estimate divides it straight back out.
    peak, want_peak = float(np.max(np.abs(out))), float(np.max(np.abs(packet)))
    if peak > 0 and want_peak > 0:
        out = out*(want_peak/peak)
    return np.asarray(out, np.float32)


EMIT_TAPS = 63             # see bound_emission; bounded by CP, not by choice


def bound_emission(packet, ceiling_hz, rate=REFERENCE_RATE, taps=EMIT_TAPS):
    """Hold the EMITTED spectrum under `ceiling_hz`, not just the carriers.

    top_bin bounds where the information is. It does not bound what goes out
    of the DAC, because the preamble is biphase-mark -- square edges, harmonics
    all the way up. Measured: tape-v3 puts its carriers at 375-10125 Hz and
    still emits 99.99% of its energy out to 23152 Hz, and lean-v3 to 23090 Hz.
    Every preset has essentially the same tail, because it is the same preamble.

    So a channel with a hard ceiling needs this as well as a low top_bin. A
    channel that merely ROLLS OFF does not: it removes the tail itself, and the
    decoder does not miss it -- a 14 kHz low-pass on the receive side costs
    nothing measurable, and even 8 kHz still decodes 4/4.

    The filter has to be GENTLE, which is the counter-intuitive part. A sharper
    one rings longer, and anything longer than the cyclic prefix (CP = 16
    samples) smears across the symbol boundary and breaks orthogonality.
    Measured at top_bin 37 with the cut at 14 kHz, reconstruction error rose
    with sharpness rather than falling: 255 taps 0.036, 511 taps 0.045, 1023
    taps 0.050, 2047 taps 0.060. Going the other way, at top_bin 34 -- 63 taps
    0.0062, 31 taps 0.0027, 17 taps 0.0010 -- but the gentler the filter the
    more of the tail survives, so 63 taps is the knee: 99.99% of the energy
    lands at 13973 Hz for a cut of 14000.

    The carriers need room under the cut for the same reason. At a 14 kHz cut,
    top_bin 37 (13875 Hz, 125 Hz of guard) costs 0.036 while top_bin 34
    (12750 Hz, 1250 Hz of guard) costs 0.0062. Pair this with a preset whose
    top_bin leaves that room -- `mid-14k` is built for exactly this cut.
    """
    packet = np.asarray(packet, np.float32)
    if not ceiling_hz or ceiling_hz >= rate/2:
        return packet
    peak = float(np.max(np.abs(packet)))
    window = firwin(int(taps) | 1, ceiling_hz, fs=rate, window=('kaiser', 8.6))
    out = filtfilt(window, [1.0], packet, axis=0)
    got = float(np.max(np.abs(out)))
    if peak > 0 and got > 0:
        # Same argument as band_limited: the training symbols carry the scale,
        # so the receiver's channel estimate divides a whole-packet gain out.
        out = out*(peak/got)
    return np.asarray(out, np.float32)


def emit_length(samples, rate, reference=REFERENCE_RATE):
    """How many samples `band_limited` will return for a length of `samples`."""
    ratio = emit_ratio(rate, reference)
    return samples if ratio is None else int(round(samples*ratio[0]/ratio[1]))


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

# Picture geometry, in the two spare bits of the top_bin byte. top_bin is
# validated to 10..63 by Layout, so bits 6 and 7 of that byte have always been
# zero: claiming them costs nothing and the header does not grow. The flags
# byte had no room -- both its nibbles are folder indices.
#
# THIS TUPLE IS WIRE ORDER, and it is a WIRE BREAK: the pre-DCT set
# ('color', 'color-lean', 'color-dct', 'mono') is gone, so an old recording
# whose header names code 0 or 3 now decodes as a different (or no) geometry.
# That is deliberate -- see the purge; the magic is not bumped because nothing
# old is worth keeping. Order after code 1 is reserved for future DCT variants.
PROFILE_CODES = ('color-dct', 'lean-dct')
# Two header bits hold four codes; two are spent, two reserved. The DCT grid
# travels in the header like every other profile, the receiver reads it, and
# there is no shared state for the two ends to disagree about.
TOP_BIN_MASK = 0x3f


def profile_code(name):
    """Wire code for a profile name, for the transmitter to declare."""
    try:
        return PROFILE_CODES.index(name)
    except ValueError:
        raise ValueError(f'Profile {name!r} has no wire code. '
                         f'Known: {", ".join(PROFILE_CODES)}') from None


def profile_name(code):
    """Profile a received code names, or None if this build has no such code."""
    code = int(code) & 3
    return PROFILE_CODES[code] if code < len(PROFILE_CODES) else None


def pack_folders(face, float_folder):
    """Pack a folder pair into the flags byte, wrapping past FOLDER_LIMIT."""
    return ((int(face) % FOLDER_LIMIT) << 4) | (int(float_folder) % FOLDER_LIMIT)


def pack_header(flags, top_bin, absolute, index, count, stamp_ms, magic=b'V3',
                profile=0):
    if not (0 <= absolute <= 0xffffffff and 1 <= index <= count <= 0xffff):
        raise ValueError('Frame/index/count outside the header ranges')
    if not 0 <= top_bin <= TOP_BIN_MASK:
        # Masking to 0xff would have silently handed bit 6 to the profile.
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
    # The MMSE equaliser for a 2x2 channel has a closed form -- gram is the
    # per-carrier Hermitian product, its inverse is the adjugate over the
    # determinant, and the pre-multiply against hH falls out by hand. Expanding
    # it here instead of through einsum keeps the (K,2,2) gram/inverse
    # intermediates out of the per-packet path: on the M4 those cost more in
    # dispatch than in arithmetic, and the cheap hardware this targets loses
    # the same ratio again. The products are identical to the matrix route.
    h00, h01, h10, h11 = h[:, 0, 0], h[:, 0, 1], h[:, 1, 0], h[:, 1, 1]
    gram00 = h00.conj()*h00 + h10.conj()*h10 + 4*noise
    gram10 = h01.conj()*h00 + h11.conj()*h10
    gram11 = h01.conj()*h01 + h11.conj()*h11 + 4*noise
    det = gram00*gram11 - gram10*gram10.conj()
    inv00 = (gram11*h00.conj() - gram10.conj()*h10.conj())/det
    inv01 = (gram11*h01.conj() - gram10.conj()*h11.conj())/det
    inv10 = (-gram10*h00.conj() + gram00*h10.conj())/det
    inv11 = (-gram10*h01.conj() + gram00*h11.conj())/det
    weights = np.clip(np.stack([np.real(inv00*h00 + inv01*h10),
                                np.real(inv10*h01 + inv11*h11)], axis=-1), 0, 1)
    variance = .5*noise*np.stack([np.abs(inv00)**2 + np.abs(inv01)**2,
                                  np.abs(inv10)**2 + np.abs(inv11)**2],
                                 axis=-1)/IMAGE_GAIN**2
    coherence = abs(np.sum(h[1:]*h[:-1].conj())) / max(
        np.sqrt(np.sum(abs(h[1:])**2)*np.sum(abs(h[:-1])**2)), 1e-20)
    return inv00, inv01, inv10, inv11, weights, variance, \
        float(coherence), _channel_skew(h, carriers)


def _equalise(body, layout):
    spectrum = rfft(body, n=N, axis=1, workers=1)
    return _equalise_spectrum(spectrum, layout)


def _equalise_spectrum(spectrum, layout):
    """The apply step of ``_equalise``, on a spectrum the caller may have
    corrected first (see ``_refine_drift``)."""
    inv00, inv01, inv10, inv11, weights, variance, coherence, skew = \
        _channel_equalizer(spectrum, layout)
    spec = spectrum[:, layout.carriers, :]
    equal = np.empty_like(spec)
    equal[..., 0] = inv00*spec[..., 0] + inv01*spec[..., 1]
    equal[..., 1] = inv10*spec[..., 0] + inv11*spec[..., 1]
    equal *= _decode_tables(layout)[4]
    return equal, weights, variance, coherence, skew


def _channel_skew(h, carriers):
    """Inter-channel delay and phase, out of the channel estimate for free.

    `h` is the 2x2 matrix per carrier the equaliser already solves for, indexed
    (carrier, receive, transmit). The relationship BETWEEN the two receive
    channels is the row-wise inner product, h[:,0,:] . conj(h[:,1,:]): the
    transmitted symbols cancel, leaving only what the path did to one channel
    against the other. Across carriers its phase is a straight line whose slope
    is a delay and whose intercept is a frequency-independent phase.

    That is head azimuth on tape, a long or swapped leg on a cable, a deliberate
    widener in a processor -- and it costs nothing, because the estimate is
    already computed and then inverted away. Only the reading was missing.

    Note this is NOT recoverable further downstream: by the time the pilot loop
    runs, the 2x2 inverse has divided the relationship out and both channels
    read zero residual. It has to be taken here or not at all.

    Reported per packet:
      skew_samples    receive channel 0 late against channel 1, in samples,
                      signed. Sub-sample values are normal and meaningful.
      skew_phase_deg  the part no delay explains. +/-180 means one leg is
                      polarity-inverted.
      skew_spread     rms of the fit residual, in samples. Small means one
                      clean mechanical relationship across the band; large
                      means the line does not describe the path, so the other
                      two numbers should not be trusted.
    """
    # A dead or silent leg makes the 2x2 solve produce non-finite entries.
    # There is no relationship between one channel and nothing, so say so
    # rather than reporting a number derived from an inf.
    if len(carriers) < 4 or not np.all(np.isfinite(h)):
        return None
    # Phase step per carrier, per receive channel, summed over the transmit
    # streams. Same adjacent-carrier product `coherence` uses just below, and
    # the reason to phrase it that way rather than reading the diagonal is that
    # it stays correct when the path mixes the channels: the sum over c is
    # basis-free, where "the diagonal" stops meaning anything once the matrix
    # is not diagonal. A step of -2*pi*tau/N per bin is a delay of tau samples.
    step = np.einsum('krc,krc->kr', h[1:], h[:-1].conj(), optimize=False)
    # A pure delay puts the SAME phase on every carrier's step, so summing a
    # channel's steps before comparing the two is a coherent average rather
    # than a convenience: it averages the noise down inside each term instead
    # of multiplying two noisy numbers together. Measured at -26 dBFS noise
    # that is 0.25 samples of error against 0.29 for the per-carrier product.
    totals = step.sum(axis=0)
    both = step[:, 0]*step[:, 1].conj()
    weight = np.abs(both)
    if not np.all(np.isfinite(totals)) or weight.sum() <= 0:
        return None
    scale = N/(2*np.pi)
    skew = -float(np.angle(totals[0]*totals[1].conj()))*scale
    # Per-carrier readings of the same quantity. If the path really is one
    # delay, they agree; if this number is large the line does not describe
    # what is happening and `skew_samples` should not be believed.
    each = -np.angle(both)*scale
    spread = float(np.sqrt(np.average((each-skew)**2, weights=weight)))
    # Off-diagonal energy: 0 is clean stereo, 0.5 is a path that has collapsed
    # the two channels into one. Free from the same matrix, and the cheapest
    # way to see a mono sum or a dead leg before the picture falls apart.
    energy = float(np.sum(np.abs(h)**2))
    mixed = float(np.sum(np.abs(h[:, 0, 1])**2) + np.sum(np.abs(h[:, 1, 0])**2))
    return {'skew_samples': skew, 'skew_spread': spread,
            'crosstalk': mixed/energy if energy else 0.}


def _resolve_header(raw, layout):
    """CRC + structure check on a hard-decoded header word -> fields or None.

    The CRC32 sits stored plain beside the 16 payload bytes, so a single bit
    error -- in the data or in the stored CRC itself -- fails the check
    wholesale. There is no redundancy budget for forward error correction: the
    header is 20 bytes out of a 2880-value budget. The fallback is to borrow
    soft information instead (see _correct_header), not to spend more wire.
    """
    if zlib.crc32(raw[:HEADER_BYTES]) != struct.unpack('>I', raw[HEADER_BYTES:])[0]:
        return None
    magic, hflags, packed, absolute, index, count, stamp = struct.unpack(
        HEADER_FORMAT, raw[:HEADER_BYTES])
    top, code = packed & TOP_BIN_MASK, packed >> 6
    if magic != layout.wire_magic or top != layout.top_bin or not 1 <= index <= count:
        return None
    return (hflags, absolute, index, count, stamp, code)


# How many of the least-reliable header bits the tolerance search is willing to
# flip at all. It bounds the flip space: r corrections over this many candidate
# positions is sum_{k<=r} C(HC, k) CRC32 checks, all C-speed, and is only ever
# paid when the straight CRC failed.
HEADER_FLIP_CANDIDATES = 14


def _correct_header(raw, flat, tolerance, layout):
    """Correct up to `tolerance` hard-decision errors using soft reliability.

    A hard bit that actually flipped is almost certainly one of the weakest
    |value| components sitting near the decision line, so only the
    HEADER_FLIP_CANDIDATES least reliable bits are ever candidates. Running the
    flip combinations through the CRC turns the all-or-nothing CRC from a pure
    error *detector* into an outward FEC check: the correction is accepted only
    when the CRC verifies the corrected word end to end. Clean packets never pay
    the search (their straight CRC passed); it exists to turn ``lost`` verdicts
    into verified identities where the hard decision was marginal.
    """
    reliability = np.stack([np.abs(flat.real), np.abs(flat.imag)], axis=-1).ravel()
    candidates = np.argsort(reliability, kind='stable')[:HEADER_FLIP_CANDIDATES]
    depth_max = min(tolerance, HEADER_FLIP_CANDIDATES - 1)
    for depth in range(1, depth_max + 1):
        for combo in combinations(candidates, depth):
            word = bytearray(raw)
            for p in combo:
                word[p >> 3] ^= 1 << (7 - (p & 7))
            fields = _resolve_header(bytes(word), layout)
            if fields:
                return fields
    return None


# A packet whose timing drifts more than this many samples across the frame
# gets the header-as-pilots refit below. Clean wires measure ~2e-8, so this
# stays far below any real wow while never firing on a clean packet (which
# then takes the byte-identical old path and pays nothing).
DRIFT_REFIT = 0.25


def _refine_drift(spectrum, layout, fields):
    """Second pass over a drifted packet, using its header as extra pilots.

    Why a second pass is needed at all: the two training symbols disagree
    under drift -- row 1 arrives rotated by 2*pi*k*drift/N per carrier -- and
    the joint channel solve blends them like a two-path fade. Measured on a
    +-0.25 sample/symbol drift that costs up to 30% of a top carrier's
    AMPLITUDE, which no phase rotation of the body can restore: the estimate
    itself is wrong. The pilot tracker's slope fit also aliases once the
    inter-pilot phase exceeds pi, so late symbols keep most of their drift.

    The fix uses only facts the wire already carries, on the spectrum that
    already exists (no new FFTs -- the re-solve below is cheap per-carrier
    algebra, and the body is never resampled):
      * the header bits, once the CRC has verified them, are known QPSK on
        every header carrier for the early symbols -- where drift is small
        enough that no angle wraps;
      * the four pilots are known (=1) on every symbol;
      * the drift phase is one global ramp rate in the symbol number.

    Anchor fits on the header rows give the rate; dividing it back out of
    training row 1 in the frequency domain re-solves the channel with both
    training rows in agreement; the body's drift is then a clean ramp through
    the origin, corrected with the global rate plus a per-row offset snapped
    from the pilots. Returns the refined (equal, weights, variance,
    pilot_error), or None if the header will not re-verify from the corrected
    symbols -- the caller then keeps its first result, so a failed refit can
    only cost the improvement, never the packet.
    """
    carriers = layout.carriers
    data, pilots, header, _, _ = _decode_tables(layout)
    end = 2 + layout.header_symbols
    pb = layout.pilots.astype(float)
    hb = layout.header_bins.astype(float)

    # The header word the CRC already verified, back as the QPSK the
    # transmitter put on the wire (encode's non-split placement). Skipped
    # entirely for split headers, which interleave the channels differently.
    flags, absolute, index, count, stamp, code = fields
    raw = pack_header(flags, layout.top_bin, absolute, index, count, stamp,
                      magic=layout.wire_magic, profile=code)
    bits = np.unpackbits(np.frombuffer(raw, np.uint8)).reshape(HEADER_SLOTS, 2)
    spread = np.resize(((bits[:, 0]*2.-1) + 1j*(bits[:, 1]*2.-1))/np.sqrt(2),
                       layout.header_symbols*len(header)).reshape(
                           layout.header_symbols, len(header))

    # Rate from anchor fits on the header rows, on the FIRST (pre-tracker)
    # equalisation. 24 known points per row (20 header + 4 pilots); drift is
    # small here so carrier unwrapping is safe.
    equal0, weights0, _, _, _ = _equalise_spectrum(spectrum, layout)
    ms = np.arange(2, end)
    slopes = []
    for s in range(layout.header_symbols):
        slope = 0.
        for ch in range(2):
            known = np.concatenate([
                equal0[2+s, header, ch]/(HEADER_GAIN*spread[s]),
                equal0[2+s, pilots, ch]])
            bins = np.concatenate([hb, pb])
            order = np.argsort(bins)
            ang = np.unwrap(np.angle(known[order]))
            w = np.concatenate([weights0[header, ch],
                                weights0[pilots, ch]])[order]
            A = np.vstack([bins[order], np.ones_like(bins[order])]).T
            sol = np.linalg.lstsq(A*w[:, None], ang*w, rcond=None)[0]
            slope += float(sol[0])/2
        slopes.append(slope)
    rate = float(np.linalg.lstsq(
        np.vstack([ms, np.ones_like(ms)]).T, np.array(slopes),
        rcond=None)[0][0])

    # Re-solve the channel with row 1's drift phase divided out, so both
    # training rows describe the same channel again.
    spec2 = spectrum.copy()
    spec2[1, carriers, :] *= np.exp(-1j*rate*carriers)[:, None]
    equal, weights, variance, _, _ = _equalise_spectrum(spec2, layout)

    # The body's drift is now a ramp through the origin: one global rate in
    # the symbol number, plus a per-row offset snapped from the pilots.
    rows = np.arange(2, layout.symbols)
    for ch in range(2):
        pred = rate*rows[:, None]*pb[None, :]
        meas = np.angle(equal[2:, pilots, ch])
        snap = meas + 2*np.pi*np.round((pred-meas)/(2*np.pi))
        w = weights[pilots, ch]**2
        doff = (w*(snap-pred)).sum(1)/max(w.sum(), 1e-12)
        equal[2:, :, ch] *= np.exp(
            -1j*(rate*rows[:, None]*carriers[None, :] + doff[:, None]))

    # The corrected symbols must still say what the CRC verified.
    pick = equal[2:end, header].mean(axis=-1)
    flat = pick.ravel()[:HEADER_SLOTS]/HEADER_GAIN
    hbits = np.stack([flat.real > 0, flat.imag > 0], axis=-1).ravel()
    raw2 = np.packbits(hbits).tobytes()
    if _resolve_header(raw2, layout) is None and \
            _correct_header(raw2, flat, 2, layout) is None:
        return None
    pilot_error = float(np.sqrt(np.mean(np.abs(equal[2:, pilots]-1)**2)))
    return equal, weights, variance, pilot_error


def decode_packet(samples, layout, coder, *, body=None, coders=None,
                  header_tolerance=2):
    """`coders` maps a profile code to the coder that reconstructs it.

    The transmitter declares its picture geometry in the header, so a receiver
    given this mapping follows whatever arrives instead of having to be told
    out of band. The header is verified before any reconstruction happens, so
    selecting on it costs nothing: no demodulation is repeated.

    Without the mapping, or for a packet whose header did not verify, `coder`
    is used as before -- a picture with no confirmed geometry is reconstructed
    on the caller's assumption, and `extra['profile']` stays None to say so.
    """
    carriers = layout.carriers
    data, pilots, header, _, _ = _decode_tables(layout)
    if body is None:
        body = samples[SYNC_LEN:layout.packet].reshape(layout.symbols, SYMBOL, 2)[:, CP - 4:CP - 4 + N]

    # 1. Primary channel equalization (computes complex gain, weights, and noise variance)
    # The spectrum is kept: a drifted packet's refit below re-solves the
    # channel from it rather than paying a second FFT.
    spectrum = rfft(body, n=N, axis=1, workers=1)
    equal, weights, variance, coherence, skew = _equalise_spectrum(
        spectrum, layout)

    timing_drift = 0.0
    clock_errors = []

    # 2. Pilot tracking & phase slope correction (robust against EQ attenuation)
    for channel in range(2):
        # Lower weight threshold from 0.6 to 0.4 so pilots in heavily rolled-off HF bands are kept
        keep = weights[pilots, channel] > 0.4
        if np.count_nonzero(keep) >= 2:
            bins = layout.pilots[keep].astype(float)
            pilot_values = equal[2:, pilots[keep], channel]
            angles = np.angle(pilot_values)

            if np.all(np.abs(pilot_values) > 0.15):
                # Phase continuity tracking across symbols
                initial = np.unwrap(angles[0])
                angles = np.unwrap(angles, axis=0)
                angles += (initial - angles[0])[None, :]
            else:
                angles = np.unwrap(angles, axis=1)

            w = weights[pilots[keep], channel] ** 2
            s0, s1, s2 = w.sum(), (w * bins).sum(), (w * bins * bins).sum()
            det = s0 * s2 - s1 * s1

            if abs(det) > 1e-12:
                ty, txy = (angles * w).sum(1), (angles * w * bins).sum(1)
                slope = (s0 * txy - s1 * ty) / det
                offset = (s2 * ty - s1 * txy) / det

                # Clamp phase slopes to prevent runaway Phase-Locked-Loop fits in deep EQ nulls
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
                        if np.sqrt(np.mean(residual ** 2)) * N / (2 * np.pi) < 0.25:
                            clock_errors.append(gradient * N / (2 * np.pi * SYMBOL))

                # Apply phase correction
                equal[2:, :, channel] *= np.exp(-1j * (slope[:, None] * carriers + offset[:, None]))

    pilot_error = float(np.sqrt(np.mean(np.abs(equal[2:, pilots] - 1) ** 2)))

    # 3. Header Extraction & CRC Verification
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
        fields = _resolve_header(raw, layout)
        if fields is None and header_tolerance:
            fields = _correct_header(raw, flat, header_tolerance, layout)
        if fields:
            break

    # 3b. Header-as-pilots refit on drifted packets. The gate is the timing
    # drift the tracker already measured, so clean packets take the
    # byte-identical path above and pay nothing; and a refit that cannot
    # re-verify the header changes nothing below, by construction.
    if (fields is not None and not layout.header_split
            and timing_drift > DRIFT_REFIT):
        refined = _refine_drift(spectrum, layout, fields)
        if refined is not None:
            equal, weights, variance = refined[:3]
            pilot_error = refined[3]

    # 3c. Clipping discount. Symmetric clipping preserves zero crossings, so
    # the header's signs verify untouched -- but the image amplitudes inherit
    # the odd-harmonic spray, heaviest at the low bins, and the Wiener trusts
    # those carriers at face value. Detect flat tops in the time samples
    # (clean packets sit ~2e-4 near peak, clipped ~3.6e-3); where found,
    # meter the distortion from the low pilots (known symbols) and add it to
    # the noise variance there, so contaminated slots soften instead of
    # blowing up. Clean packets skip everything below untouched.
    if fields is not None and body.size:
        peak = float(np.max(np.abs(body)))
        if peak > 0 and float(np.mean(np.abs(body) > 0.98*peak)) > 1e-3:
            low_pilots = pilots[layout.pilots <= 10]
            if len(low_pilots):
                extra_var = float(np.mean(
                    np.abs(equal[2:, low_pilots]-1)**2))
                low = carriers <= 12
                if variance.ndim == 2:
                    variance[low, :] += extra_var
                else:
                    variance[low] += extra_var

    # 4. Usability Check & Payload Reconstruction
    coverage = float(np.mean(weights[data] >= 0.45))
    usable = (coverage >= 0.08 and pilot_error < 2.0 and
              coherence > (0.50 if layout.top_bin <= 13 else 0.30))

    if fields is None and not usable:
        return Decoded('lost', pilot_error=pilot_error, coverage=coverage)

    # 4b. Payload assembly. sent/per/per_noise are filled once into their
    # final (symbol, carrier, channel, IQ) layout instead of building broadcast
    # views that ravel-copy and concatenate -- the same bytes with far fewer
    # per-packet temporaries. Every part spans (symbols, bins, channels) with
    # the I/Q pair interleaved, so filling the re-shaped views writes exactly
    # the layout the coefficient gather expects.
    total = layout.capacity
    sent = np.empty(total, dtype=float)
    per = np.empty(total, dtype=float)
    per_noise = np.empty(total, dtype=float)
    off = 0

    if layout.header_capacity:
        spare = np.searchsorted(carriers, layout.spare_bins)
        hs, n = layout.header_symbols, len(spare)
        early = equal[2:end, spare] / IMAGE_GAIN
        v = sent[off:off + 4*hs*n].reshape(hs, n, 2, 2)
        v[..., 0] = early.real
        v[..., 1] = early.imag
        p = per[off:off + 4*hs*n].reshape(hs, n, 2, 2)
        p[...] = weights[spare][None, :, :, None]
        pn = per_noise[off:off + 4*hs*n].reshape(hs, n, 2, 2)
        pn[...] = variance[spare][None, :, :, None]
        off += 4*hs*n

    block = equal[end:, data] / IMAGE_GAIN
    s, b = layout.image_symbols, len(data)
    v = sent[off:off + 4*s*b].reshape(s, b, 2, 2)
    v[..., 0] = block.real
    v[..., 1] = block.imag
    p = per[off:off + 4*s*b].reshape(s, b, 2, 2)
    p[...] = weights[data][None, :, :, None]
    pn = per_noise[off:off + 4*s*b].reshape(s, b, 2, 2)
    pn[...] = variance[data][None, :, :, None]

    coverage = float(np.mean(per >= 0.45))

    # Guard against exact zeros only. An absolute floor here is not scale-free:
    # on a quiet or attenuated channel the real post-equaliser variance sits far
    # below 1e-4, so the floor dominated the Wiener denominator and applied an
    # extra, carrier-dependent shrink on top of the de-bias above.
    per_noise[per_noise < 1e-20] = 1e-20

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
    return Decoded('received' if pilot_error < 0.20 else 'degraded', values=values,
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

