"""V7-only source coding, sample timing, and output shaping helpers.

This module intentionally contains no legacy wire encoder or decoder.
"""
from fractions import Fraction
from functools import lru_cache

import numpy as np
from scipy.fft import dctn, idctn
from scipy.signal import resample_poly, firwin, filtfilt

REFERENCE_RATE = 48000

RATE = REFERENCE_RATE
N, CP = 128, 16

SYMBOL = N + CP

SYNC_LEN = 288

GUARD = 32

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

def _sinc_weight_table(taps=8, phases=4096):
    """Quantized windowed-sinc weights for the live packet path."""
    offsets = np.arange(-taps+1, taps+1, dtype=float)
    fraction = np.arange(phases, dtype=float)[:, None]/phases
    weights = (np.sinc(fraction-offsets[None, :]) *
               np.sinc((fraction-offsets[None, :])/taps))
    weights /= weights.sum(axis=1, keepdims=True)
    return weights

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

EMIT_TAPS = 63

def bound_emission(packet, ceiling_hz, rate=REFERENCE_RATE, taps=EMIT_TAPS):
    """Hold the EMITTED spectrum under `ceiling_hz`, not just the carriers.

    top_bin bounds where the information is. It does not bound what goes out
    of the DAC, because the preamble is biphase-mark -- square edges, harmonics
    all the way up. Measured: earlier tape profiles puts its carriers at 375-10125 Hz and
    still emits 99.99% of its energy out to 23152 Hz, and earlier lean profiles to 23090 Hz.
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

def speed_length(samples, rate, speed=1.0, reference=REFERENCE_RATE):
    """Output length for a reference packet played at ``speed``."""
    speed = float(speed)
    if not np.isfinite(speed) or speed <= 0:
        raise ValueError('speed must be finite and positive')
    return max(1, int(round(float(samples)*float(rate)/(reference*speed))))

def speed_resample(samples, rate, speed=1.0, reference=REFERENCE_RATE):
    """Time-compress a reference packet for a real output sample clock.

    ``band_limited`` intentionally preserves the wire duration.  V7 speed
    profiles need the opposite operation: the same reference packet must
    occupy fewer seconds while retaining its sample-clock carrier geometry.
    The output is exact-length so PacketOutput can schedule it safely.
    """
    samples = np.asarray(samples, np.float32)
    target = speed_length(len(samples), rate, speed, reference)
    if target == len(samples):
        return np.ascontiguousarray(samples)
    ratio = Fraction(target, len(samples)).limit_denominator(256)
    out = resample_poly(samples, ratio.numerator, ratio.denominator, axis=0)
    if len(out) < target:
        out = np.concatenate((out, np.zeros((target-len(out), *out.shape[1:]),
                                             dtype=out.dtype)))
    elif len(out) > target:
        out = out[:target]
    src_peak = float(np.max(np.abs(samples)))
    out_peak = float(np.max(np.abs(out)))
    if src_peak > 0 and out_peak > src_peak:
        out = out*(src_peak/out_peak)
    return np.ascontiguousarray(out, dtype=np.float32)
