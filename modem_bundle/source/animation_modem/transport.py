"""Experimental frame-local analog OFDM image link. Wire format v1."""
import statistics
import struct
import time
import zlib
from collections import deque
from dataclasses import dataclass

import numpy as np
from PIL import Image, ImageDraw, ImageOps
from scipy.signal import correlate
from scipy.ndimage import gaussian_filter

RATE, FPS, FRAME = 48000, 15, 3200
N, CP, SYMBOL = 128, 16, 144
WIDTH, HEIGHT = 40, 48
PILOTS = np.array([6, 19, 35, 50])
BINS = np.array([k for k in range(3, 55) if k not in PILOTS])
ALL = np.arange(3, 55)
PLOC = np.searchsorted(ALL, PILOTS)
DLOC = np.searchsorted(ALL, BINS)
SYNC_LEN = 288
BODY_LEN = 19 * SYMBOL  # two training, two header, fifteen image symbols
PACKET = SYNC_LEN + BODY_LEN
_rng = np.random.default_rng(41015)
PHASE = np.exp(1j * _rng.uniform(-np.pi, np.pi, (19, 52, 2)))
_s = np.zeros(129, complex)
_s[6:109] = np.exp(1j * _rng.uniform(-np.pi, np.pi, 103))
SYNC = np.fft.irfft(_s, n=256)
SYNC *= .65 / np.max(np.abs(SYNC))
SYNC_ENERGY = np.dot(SYNC, SYNC)
RESAMPLE_TAPS = 8  # Lanczos half-width for speed correction
# A source running off-speed compresses or stretches the preamble too, so a
# single template stops correlating well before the demodulator gives up.
# These cover cassette-grade error; the tracked estimate refines it after lock.
SYNC_SCALES = (-.024, -.016, -.008, 0., .008, .016, .024)


def _scaled_sync(scale):
    if not scale:
        return SYNC
    length = int(round(len(SYNC)/(1.0+scale)))
    return np.interp(np.arange(length)*(1.0+scale), np.arange(len(SYNC)), SYNC)


SYNC_BANK = [(scale, _scaled_sync(scale)) for scale in SYNC_SCALES]
SYNC_BANK = [(scale, template, float(np.dot(template, template)))
             for scale, template in SYNC_BANK]

# Exactly 2,880 real image values in every profile; modem timing is unchanged.
PROFILES = {
    'color': ((40, 48), (20, 24)),
    'detail': ((48, 56), (8, 12)),
    'mono': ((48, 60), None),
}
PROFILE_PHASES = {'color': PHASE}
for _name, _seed in [('detail', 41016), ('mono', 41017)]:
    PROFILE_PHASES[_name] = np.exp(1j*np.random.default_rng(_seed).uniform(-np.pi,np.pi,(19,52,2)))


def plane_shapes(profile):
    size, chroma = PROFILES[profile]
    return [(size[1], size[0])] + ([(chroma[1], chroma[0])]*2 if chroma else [])

GLYPHS = dict(zip('0123456789AF/', [
 '111101101101111','010110010010111','111001111100111',
 '111001111001111','101101111001001','111100111001111',
 '111100111101111','111001001001001','111101111101111',
 '111101111001111','010101111101101','111100110100100',
 '001001010100100']))


def _text(im, text, y):
    strip = Image.new('RGB', (4 * len(text), 7), 'black')
    d = ImageDraw.Draw(strip)
    for i, c in enumerate(text):
        for p, bit in enumerate(GLYPHS[c]):
            if bit == '1':
                d.point((i * 4 + p % 3, 1 + p // 3), fill='white')
    if strip.width > im.width:
        strip = strip.resize((im.width, 7), Image.Resampling.NEAREST)
    im.paste(strip, (0, y))


def prepare_image(source, absolute, index, count, numbered=False, profile="color"):
    """Aspect-preserving fit; counters are actual transmitted pixels."""
    im = ImageOps.pad(source.convert('RGB'), PROFILES[profile][0],
                      method=Image.Resampling.LANCZOS, color='black')
    if numbered:
        _text(im, 'A' + str(absolute).zfill(6), 0)
        _text(im, 'F' + str(index) + '/' + str(count), 7)
    return im


def image_values(im, profile="color"):
    y, cb, cr = im.convert('YCbCr').split()
    chroma = PROFILES[profile][1]
    planes = [y]
    if chroma:
        planes += [cb.resize(chroma, Image.Resampling.BOX), cr.resize(chroma, Image.Resampling.BOX)]
    return np.concatenate([np.asarray(p).ravel() for p in planes]).astype(float) / 127.5 - 1


def values_image(values, profile="color"):
    q = np.uint8(np.clip(np.rint((values + 1) * 127.5), 0, 255))
    size, chroma = PROFILES[profile]
    planes = []
    offset = 0
    for shape in plane_shapes(profile):
        count = int(np.prod(shape))
        planes.append(Image.fromarray(q[offset:offset+count].reshape(shape)))
        offset += count
    if not chroma:
        return planes[0].convert('RGB')
    return Image.merge('YCbCr', (planes[0],
        planes[1].resize(size, Image.Resampling.BILINEAR),
        planes[2].resize(size, Image.Resampling.BILINEAR))).convert('RGB')


def encode(source, absolute, index, count, numbered=False, profile="color", *, target_time_ns=None):
    if not (0 <= absolute <= 0xffffffff and 1 <= index <= count <= 0xffffffff):
        raise ValueError('Frame/index/count outside unsigned 32-bit range')
    start = time.perf_counter()
    if profile not in PROFILES:
        raise ValueError(f'Unknown profile: {profile}')
    width, height = PROFILES[profile][0]
    im = prepare_image(source, absolute, index, count, numbered, profile)
    # 20 bytes + CRC32 = 192 bits = two QPSK symbols, repeated across stereo.
    if target_time_ns is None:
        raw = struct.pack('>4sIIIBBBB', b'SI01' if profile == 'color' else b'SI02', absolute, index, count,
                          width, height, FPS, int(numbered))
    else:
        # ST v1: profile/flags replace redundant width/height/FPS fields.
        # Same 24-byte CRC-protected stereo-repeated header; no image capacity loss.
        magic = b'ST' + bytes([list(PROFILES).index(profile), int(numbered)])
        raw = struct.pack('>4sIIII', magic, absolute, index, count,
                          (int(target_time_ns)//1_000_000) & 0xffffffff)
    raw += struct.pack('>I', zlib.crc32(raw))
    bits = np.unpackbits(np.frombuffer(raw, np.uint8)).reshape(2, 48, 2)
    qpsk = ((bits[..., 0].astype(float)*2-1) + 1j*(bits[..., 1].astype(float)*2-1)) / np.sqrt(2)
    carriers = np.zeros((19, 52, 2), complex)
    carriers[0, :, 0] = 1
    carriers[1, :, 1] = 1
    carriers[2:4, DLOC, :] = qpsk[..., None]
    v = image_values(im, profile).reshape(15, 48, 2, 2)
    carriers[4:, DLOC, :] = (v[..., 0] + 1j * v[..., 1]) * .7
    carriers[2:, PLOC, :] = 1
    spectrum = np.zeros((19, 65, 2), complex)
    spectrum[:, ALL, :] = carriers * PROFILE_PHASES[profile]
    wave = np.fft.irfft(spectrum, n=N, axis=1)
    wave = np.concatenate([wave[:, -CP:], wave], axis=1).reshape(-1, 2)
    out = np.zeros((FRAME, 2), np.float32)
    out[16:272, 0] = SYNC
    out[SYNC_LEN:PACKET] = wave
    return out, im, (time.perf_counter()-start)*1000


@dataclass
class Result:
    frame: int | None
    status: str
    image: Image.Image | None = None
    index: int | None = None
    count: int | None = None
    pilot_error: float | None = None
    decode_ms: float = 0
    sample: int = 0
    identity: str = 'unknown'
    pixel_coverage: float | None = None
    training_coherence: float | None = None
    sync_score: float | None = None
    profile: str | None = None
    profile_identity: str = 'unknown'
    target_time_ms32: int | None = None
    target_time_ns: int | None = None
    decode_error_ms: float | None = None
    rate_error: float = 0.0


def conceal_values(values, weights, profile="color"):
    """Fill unreliable samples from spatial neighbours in THIS image only."""
    result = values.copy()
    offset = 0
    for shape in plane_shapes(profile):
        count = int(np.prod(shape))
        p = result[offset:offset+count].reshape(shape)
        w = weights[offset:offset+count].reshape(shape)
        reliable = w >= .55
        if not np.all(reliable):
            numerator = gaussian_filter(np.clip(p,-1,1)*reliable, 1.5)
            denominator = gaussian_filter(reliable.astype(float), 1.5)
            # Neutral luminance/chroma where no useful spatial neighbours exist.
            fill = numerator / np.maximum(denominator, 1e-8)
            p[~reliable] = fill[~reliable]
        offset += count
    return result


def decode_packet(samples):
    start = time.perf_counter()
    # Start inside the cyclic prefix to leave room for acquisition/filter delay.
    body = samples[SYNC_LEN:PACKET].reshape(19, SYMBOL, 2)[:, CP-4:CP-4+N]
    spectrum = np.fft.rfft(body, n=N, axis=1)
    received = spectrum[:, ALL, :]
    # Each profile has distinct training phases. Correct de-rotation produces
    # the smooth channel response; this allows frame-local profile inference
    # even when the digital header is erased. CRC still verifies the selection.
    fits = []
    for name, phases in PROFILE_PHASES.items():
        matrix = np.stack([received[0] / phases[0, :, 0, None],
                           received[1] / phases[1, :, 1, None]], axis=-1)
        score = float(abs(np.sum(matrix[:-1].conj()*matrix[1:])) /
                      max(np.sqrt(np.sum(abs(matrix[:-1])**2)*np.sum(abs(matrix[1:])**2)),1e-12))
        fits.append((score, name, matrix))
    fits.sort(key=lambda x: x[0], reverse=True)
    coherence, profile, h = fits[0]
    profile_margin = coherence-fits[1][0]
    phases = PROFILE_PHASES[profile]
    width, height = PROFILES[profile][0]
    # Unused bins provide a frame-local estimate of noise/leakage. Regularization
    # prevents deep notches from amplifying it into full-scale coloured speckles.
    noise = max(float(np.mean(np.abs(spectrum[:, [1, 2, 57, 58, 59, 60, 61, 62]])**2)), 1e-12)
    # V diag(s/(s^2+4n)) U^H == (H^H H + 4n I)^-1 H^H; 2x2 closed form, no SVD.
    hH = h.conj().transpose(0, 2, 1)
    gram = hH @ h
    gram[:, 0, 0] += 4*noise
    gram[:, 1, 1] += 4*noise
    det = gram[:,0,0]*gram[:,1,1] - gram[:,0,1]*gram[:,1,0]
    adj = np.empty_like(gram)
    adj[:,0,0], adj[:,1,1] = gram[:,1,1], gram[:,0,0]
    adj[:,0,1], adj[:,1,0] = -gram[:,0,1], -gram[:,1,0]
    inv = (adj / det[:, None, None]) @ hH
    response = inv @ h
    weights = np.clip(np.real(np.diagonal(response,axis1=1,axis2=2)),0,1)
    equal = np.einsum('kij,skj->ski', inv, received) / phases
    # Track within-frame timing/phase using four pilots on each channel.
    for channel in range(2):
        keep = weights[PLOC, channel] > .6
        if np.count_nonzero(keep) >= 2:
            bins = PILOTS[keep].astype(float)
            angles = np.unwrap(np.angle(equal[2:, PLOC[keep], channel]), axis=1)
            w = weights[PLOC[keep], channel]**2
            # Weighted line fit in closed form; identical to the pinv solution.
            s0, s1, s2 = w.sum(), (w*bins).sum(), (w*bins*bins).sum()
            determinant = s0*s2 - s1*s1
            ty, txy = (angles*w).sum(1), (angles*w*bins).sum(1)
            slope = (s0*txy - s1*ty) / determinant
            offset = (s2*ty - s1*txy) / determinant
            phase = slope[:,None]*ALL + offset[:,None]
            equal[2:,:,channel] *= np.exp(-1j*phase)
    pe = float(np.sqrt(np.mean(np.abs(equal[2:, PLOC] - 1)**2)))
    candidates = [(equal[2:4, DLOC]*weights[DLOC]).sum(axis=-1) / np.maximum(weights[DLOC].sum(axis=-1),1e-9),
                  equal[2:4, DLOC, 0], equal[2:4, DLOC, 1]]
    header = None
    target_time_ms32 = None
    for symbols in candidates:
        bits = np.stack([symbols.real > 0, symbols.imag > 0], axis=-1)
        raw = np.packbits(bits.ravel()).tobytes()
        if zlib.crc32(raw[:20]) == struct.unpack('>I', raw[20:])[0]:
            if raw[:2] == b'ST':
                magic, absolute, index, count, target = struct.unpack('>4sIIII', raw[:20])
                if (magic[2] == list(PROFILES).index(profile) and magic[3] in (0, 1)
                        and 1 <= index <= count):
                    header = (magic, absolute, index, count)
                    target_time_ms32 = target
                    break
            else:
                fields = struct.unpack('>4sIIIBBBB', raw[:20])
                if (fields[0] == (b'SI01' if profile == 'color' else b'SI02') and fields[4:7] == (width, height, FPS)
                        and 1 <= fields[2] <= fields[3] and fields[7] in (0, 1)):
                    header = fields
                    break
    v = equal[4:, DLOC] / .7
    values = np.stack([v.real, v.imag], axis=-1).ravel()
    pixel_weights = np.broadcast_to(weights[DLOC][None,:,:,None],(15,48,2,2)).ravel()
    coverage = float(np.mean(pixel_weights >= .55))
    im = values_image(conceal_values(values, pixel_weights, profile), profile)
    if header is None:
        # A recognizable training pattern is required: don't turn arbitrary
        # noise into a claimed recovered frame. Absolute identity stays unknown.
        salvage = coherence >= .5 and profile_margin >= .15 and coverage >= .1
        return Result(None, 'partial_header_unknown' if salvage else 'corrupt_header',
                      image=im if salvage else None, pilot_error=pe,
                      decode_ms=(time.perf_counter()-start)*1000,
                      pixel_coverage=coverage, training_coherence=coherence,
                      profile=profile if salvage else None, profile_identity="estimated_training" if salvage else "unknown")
    return Result(header[1], 'received' if pe < .15 else 'degraded', im,
                  header[2], header[3], pe, (time.perf_counter()-start)*1000,
                  identity='verified_header', pixel_coverage=coverage, training_coherence=coherence,
                  profile=profile, profile_identity='verified_header', target_time_ms32=target_time_ms32)


def resample_packet(samples, rate):
    """Undo a constant playback-speed error before demodulating.

    `rate` is the received duration divided by the nominal duration, minus one:
    negative when the source played fast. Tape and turntable speed error walks
    each symbol off the decoder's fixed window, and once that walk exceeds the
    cyclic prefix the window straddles two symbols and no amount of pilot phase
    correction helps. Correcting the whole packet first keeps every symbol
    inside its guard.

    Linear interpolation is not good enough here: the carriers reach 20.25 kHz
    against a 24 kHz Nyquist, where it loses about half the signal. A windowed
    sinc keeps that under two percent.
    """
    position = np.arange(PACKET)*(1.0+rate)
    base = np.floor(position).astype(int)
    taps = np.arange(-RESAMPLE_TAPS+1, RESAMPLE_TAPS+1)
    index = np.clip(base[:, None]+taps[None, :], 0, len(samples)-1)
    offsets = (position-base)[:, None]-taps[None, :]
    weights = np.sinc(offsets)*np.sinc(offsets/RESAMPLE_TAPS)
    weights /= weights.sum(axis=1, keepdims=True)
    return np.einsum('ij,ijc->ic', weights, samples[index]).astype(np.float32)


def sync_correlation(samples, limit=None, template=None, energy=None):
    """Stable local normalization, including near-silent stereo channels.

    Energy uses a per-call cumulative sum (bounded length, peak-normalized), not
    a running one across the stream, so the precision loss that motivated the
    old O(N*len(SYNC)) moving sum does not apply. `limit` restricts the
    correlation to the offsets the caller will actually inspect; the reliability
    floor still uses every window's energy, so scores are unchanged.
    """
    if template is None:
        template, energy = SYNC, SYNC_ENERGY
    x = np.asarray(samples, dtype=np.float64)
    n = len(x) - len(template) + 1
    if limit is not None:
        n = max(0, min(n, limit))
    if n <= 0:
        return np.empty(0, dtype=float)
    peak = float(np.max(np.abs(x)))
    if peak == 0:
        return np.zeros(n)
    x = x / peak
    cumulative = np.empty(len(x) + 1)
    cumulative[0] = 0.0
    np.cumsum(x * x, out=cumulative[1:])
    window = cumulative[len(template):] - cumulative[:-len(template)]
    # Ignore numerical-floor windows rather than amplify their residuals.
    floor = 64 * np.finfo(float).eps * float(np.max(window))
    m = n
    corr = correlate(x[:m + len(template) - 1], template, mode='valid', method='direct')
    score = np.zeros(m)
    reliable = window[:m] > floor
    score[reliable] = np.abs(corr[reliable]) / np.sqrt(window[:m][reliable] * energy)
    return np.minimum(score, 1.0)  # Round-off only; normalization is local.


class Receiver:
    """Chunk-independent acquisition; missing identities are explicitly estimates.

    Reacquires each frame. No temporal image state. Small clock differences are
    handled by per-frame acquisition and pilot correction, not a full resampler.
    """
    # Below this the existing per-symbol pilot phase fit absorbs the drift on
    # its own, and correcting would only resample for no reason.
    RATE_DEADBAND = .0005

    def __init__(self, threshold=.45, track_rate=True):
        self.buffer = np.empty((0, 2), np.float32)
        self.offset = 0
        self.expected = None
        self.next_sample = None
        self.threshold = threshold
        self.unidentified_next = 0
        self.track_rate = track_rate
        self.rate = 0.
        self.rate_history = deque(maxlen=9)
        self.last_sync = None

    def _observe_rate(self, absolute_sample):
        """Packets leave the transmitter exactly FRAME samples apart, so the
        measured spacing between sync locks is the source's speed error."""
        if self.last_sync is not None:
            gap = absolute_sample - self.last_sync
            periods = round(gap/FRAME)
            # Ignore spacings that are not a plausible whole number of packets.
            if periods >= 1 and abs(gap - periods*FRAME) < FRAME*.04:
                self.rate_history.append(gap/(periods*FRAME) - 1)
        self.last_sync = absolute_sample
        if len(self.rate_history) >= 3:
            estimate = statistics.median(self.rate_history)
            self.rate = estimate if abs(estimate) >= self.RATE_DEADBAND else 0.

    def _discard(self, n):
        self.buffer = self.buffer[n:]
        self.offset += n

    def feed(self, data):
        data = np.asarray(data, np.float32)
        if data.ndim != 2 or data.shape[1] != 2:
            raise ValueError('Receiver requires two selected audio channels')
        if not np.isfinite(data).all():
            raise ValueError('Audio contains NaN or infinity')
        self.buffer = np.concatenate([self.buffer, data])
        results = []
        while len(self.buffer) >= PACKET:
            limit = len(self.buffer) - PACKET + 17
            # Only cold acquisition pays for the bank. Once the spacing between
            # locks has settled, an on-speed source is rate 0 and still wants
            # exactly one template -- test convergence, not the estimate.
            searching = self.track_rate and len(self.rate_history) < 3
            bank = (SYNC_BANK if searching
                    else [min(SYNC_BANK, key=lambda e: abs(e[0]-self.rate))])
            score = None
            for scale, template, energy in bank:
                for c in range(2):
                    one = sync_correlation(self.buffer[:, c], limit, template, energy)
                    score = one if score is None else np.maximum(score, one)
            hits = np.flatnonzero(score >= self.threshold)
            if not len(hits):
                self._discard(max(1, len(self.buffer)-PACKET+1))
                break
            first = int(hits[0])
            peak = first + int(np.argmax(score[first:first+17]))
            begin = peak - 16
            if begin < 0:
                self._discard(peak + 256)
                continue
            # A slowed source stretches the packet past PACKET samples.
            span = PACKET + int(np.ceil(PACKET*max(self.rate, 0.))) + 2
            if begin + span > len(self.buffer):
                break
            absolute_sample = self.offset + begin
            if self.next_sample is None:
                while self.unidentified_next + FRAME <= absolute_sample:
                    results.append(Result(None, 'unidentified', sample=self.unidentified_next))
                    self.unidentified_next += FRAME
            while self.next_sample is not None and self.next_sample < absolute_sample - FRAME//2:
                results.append(Result(self.expected, 'missing', sample=self.next_sample, identity='estimated'))
                self.expected = (self.expected+1) & 0xffffffff
                self.next_sample += FRAME
            if self.track_rate:
                self._observe_rate(absolute_sample)
            window = self.buffer[begin:begin+span]
            result = decode_packet(resample_packet(window, self.rate)
                                   if self.rate else window[:PACKET])
            result.rate_error = self.rate
            result.sync_score = float(score[peak])
            result.sample = absolute_sample
            if result.frame is not None:
                if self.next_sample is None:
                    result.status = 'acquired_' + result.status
                elif result.frame != self.expected:
                    result.status = 'reacquired_' + result.status
                self.expected = (result.frame + 1) & 0xffffffff
                self.next_sample = absolute_sample + FRAME
            elif self.expected is not None:
                result.frame = self.expected
                result.identity = 'estimated'
                self.expected = (self.expected+1) & 0xffffffff
                self.next_sample = absolute_sample + FRAME
            results.append(result)
            self.unidentified_next = absolute_sample + FRAME
            self._discard(begin + PACKET)
        # Emit failures during silence, without waiting for another good header.
        end = self.offset + len(self.buffer)
        if self.next_sample is None:
            while end >= self.unidentified_next + FRAME:
                results.append(Result(None, 'unidentified', sample=self.unidentified_next))
                self.unidentified_next += FRAME
        while self.next_sample is not None and end >= self.next_sample + FRAME + 32:
            results.append(Result(self.expected, 'missing', sample=self.next_sample, identity='estimated'))
            self.expected = (self.expected+1) & 0xffffffff
            self.next_sample += FRAME
        return results
