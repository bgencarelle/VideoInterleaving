"""Experimental frame-local analog OFDM image link. Wire format v1."""
import struct
import time
import zlib
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
    u, singular, vh = np.linalg.svd(h)
    inv = (vh.conj().transpose(0,2,1) * (singular/(singular**2+4*noise))[:,None,:]) @ u.conj().transpose(0,2,1)
    response = inv @ h
    weights = np.clip(np.real(np.diagonal(response,axis1=1,axis2=2)),0,1)
    equal = np.einsum('kij,skj->ski', inv, received) / phases
    # Track within-frame timing/phase using four pilots on each channel.
    for channel in range(2):
        keep = weights[PLOC, channel] > .6
        if np.count_nonzero(keep) >= 2:
            bins = PILOTS[keep]
            angles = np.unwrap(np.angle(equal[2:, PLOC[keep], channel]), axis=1)
            w = weights[PLOC[keep], channel]**2
            design = np.stack([bins, np.ones(len(bins))], axis=1)
            fit = np.linalg.pinv(design*w[:,None]) * w[None,:]
            coeff = angles @ fit.T
            phase = coeff[:,0,None]*ALL + coeff[:,1,None]
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


def sync_correlation(samples):
    """Stable local normalization, including near-silent stereo channels.

    Subtracting cumulative energies loses precision after louder audio. FFT
    correlation also leaves residuals in silent windows. Together these can
    create impossible scores and false locks. Direct local sums avoid both.
    """
    x = np.asarray(samples, dtype=np.float64)
    if len(x) < len(SYNC):
        return np.empty(0, dtype=float)
    peak = float(np.max(np.abs(x)))
    if peak == 0:
        return np.zeros(len(x)-len(SYNC)+1)
    x = x / peak
    corr = correlate(x, SYNC, mode='valid', method='direct')
    energy = correlate(x*x, np.ones(len(SYNC)), mode='valid', method='direct')
    score = np.zeros_like(energy)
    # Ignore numerical-floor windows rather than amplify their residuals.
    reliable = energy > 64*np.finfo(float).eps*float(np.max(energy))
    score[reliable] = np.abs(corr[reliable]) / np.sqrt(energy[reliable]*SYNC_ENERGY)
    return np.minimum(score, 1.0)  # Round-off only; normalization is local.


class Receiver:
    """Chunk-independent acquisition; missing identities are explicitly estimates.

    Reacquires each frame. No temporal image state. Small clock differences are
    handled by per-frame acquisition and pilot correction, not a full resampler.
    """
    def __init__(self, threshold=.45):
        self.buffer = np.empty((0, 2), np.float32)
        self.offset = 0
        self.expected = None
        self.next_sample = None
        self.threshold = threshold
        self.unidentified_next = 0

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
            scores = []
            for c in range(2):
                scores.append(sync_correlation(self.buffer[:, c])[:limit])
            score = np.maximum(*scores)
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
            if begin + PACKET > len(self.buffer):
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
            result = decode_packet(self.buffer[begin:begin+PACKET])
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
