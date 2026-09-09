"""Stateful, sample-continuous audio-path emulation before demodulation."""
from dataclasses import asdict, dataclass
import numpy as np
from scipy.signal import butter, sosfilt
from .transport import RATE


@dataclass
class Settings:
    lowpass_hz: float = 0
    highpass_hz: float = 0
    noise_dbfs: float | None = None
    gain_db: float = 0
    right_gain_db: float = 0
    crosstalk: float = 0
    clip: float = 0
    bits: int = 0
    dropout_ms: float = 0
    dropout_every: float = 2
    seed: int = 2026

    def validate(self):
        for key, value in asdict(self).items():
            if value is not None and not np.isfinite(value):
                raise ValueError(f'{key} must be finite')
        for name in ['lowpass_hz', 'highpass_hz']:
            if not 0 <= getattr(self, name) < RATE/2:
                raise ValueError(f'{name} must be 0 (off) or below {RATE/2} Hz')
        if self.lowpass_hz and self.highpass_hz >= self.lowpass_hz:
            raise ValueError('High-pass cutoff must be below low-pass cutoff')
        if self.noise_dbfs is not None and not -160 <= self.noise_dbfs <= 0:
            raise ValueError('Noise level must be between -160 and 0 dBFS RMS')
        if not all(-80 <= g <= 40 for g in [self.gain_db, self.right_gain_db]):
            raise ValueError('Gain must be between -80 and +40 dB')
        if not 0 <= self.crosstalk <= .5:
            raise ValueError('Crosstalk must be 0 to 0.5 (0.5 is mono summing)')
        if not 0 <= self.clip <= 1:
            raise ValueError('Clip must be 0 (off) or a peak threshold up to 1')
        if self.bits != 0 and not 2 <= self.bits <= 24:
            raise ValueError('Bits must be 0 (off) or 2 to 24')
        if self.dropout_every <= 0 or not 0 <= self.dropout_ms < self.dropout_every*1000:
            raise ValueError('Dropout duration must be nonnegative and shorter than its positive period')
        if self.seed < 0:
            raise ValueError('Seed must be nonnegative')
        return self


PRESETS = {
    'clean': {},
    'mild': dict(lowpass_hz=18000, noise_dbfs=-55, right_gain_db=-2, crosstalk=.03),
    'rough': dict(lowpass_hz=18000, noise_dbfs=-48, right_gain_db=-3,
                  crosstalk=.07, clip=.12),
    'hostile': dict(highpass_hz=300, lowpass_hz=8000, noise_dbfs=-30,
                    right_gain_db=-12, crosstalk=.2, clip=.08,
                    dropout_ms=40, dropout_every=2),
}


def add_arguments(parser):
    g = parser.add_argument_group('Audio-path emulation (before decoding)')
    g.add_argument('--emulate', choices=PRESETS, default='clean')
    g.add_argument('--lowpass-hz', type=float, help='4th-order low-pass cutoff; 0 disables')
    g.add_argument('--highpass-hz', type=float, help='4th-order high-pass cutoff; 0 disables')
    g.add_argument('--noise-dbfs', type=float, help='White noise RMS dB relative to amplitude 1 (not SNR)')
    g.add_argument('--gain-db', type=float, help='Gain on both channels, dB')
    g.add_argument('--right-gain-db', type=float, help='Additional right-channel gain, dB')
    g.add_argument('--crosstalk', type=float, help='Symmetric mixing fraction 0..0.5; 0.5 collapses stereo')
    g.add_argument('--clip', type=float, help='Hard clipping peak, e.g. 0.12; 0 disables')
    g.add_argument('--bits', type=int, help='Full-scale quantization bits, 2..24; 0 disables')
    g.add_argument('--dropout-ms', type=float, help='Periodic silence duration, milliseconds; 0 disables')
    g.add_argument('--dropout-every', type=float, help='Silence period in seconds (default 2)')
    g.add_argument('--seed', type=int, help='Repeatable noise seed (default 2026)')


def settings_from_args(args):
    values = dict(PRESETS[args.emulate])
    for key in Settings.__dataclass_fields__:
        value = getattr(args, key)
        if value is not None:
            values[key] = value
    return Settings(**values).validate()


class Emulator:
    def __init__(self, settings):
        self.settings = settings.validate()
        self.rng = np.random.default_rng(settings.seed)
        self.sample = 0
        filters = []
        if settings.highpass_hz:
            filters.append(butter(4, settings.highpass_hz, btype='highpass', fs=RATE, output='sos'))
        if settings.lowpass_hz:
            filters.append(butter(4, settings.lowpass_hz, fs=RATE, output='sos'))
        self.sos = np.concatenate(filters) if filters else None
        self.zi = np.zeros((len(self.sos), 2, 2)) if filters else None

    def process(self, data):
        s = self.settings
        x = np.asarray(data, dtype=np.float64).copy()
        if x.ndim != 2 or x.shape[1] != 2 or not np.isfinite(x).all():
            raise ValueError('Emulator requires finite stereo samples')
        if not len(x):
            return x.astype(np.float32)
        x *= 10**(s.gain_db/20)
        x[:,1] *= 10**(s.right_gain_db/20)
        if s.crosstalk:
            c=s.crosstalk
            # Explicit stereo sums avoid a BLAS call for this two-channel mix.
            # Some macOS Accelerate builds emit spurious matmul FP warnings.
            left, right = x[:, 0], x[:, 1]
            x = np.column_stack(((1-c)*left+c*right, c*left+(1-c)*right))
        if self.sos is not None:
            x, self.zi = sosfilt(self.sos, x, axis=0, zi=self.zi)
        if s.noise_dbfs is not None:
            x += self.rng.standard_normal(x.shape) * 10**(s.noise_dbfs/20)
        if s.clip:
            x = np.clip(x, -s.clip, s.clip)
        if s.bits:
            scale = 2**(s.bits-1)
            x = np.clip(np.rint(x*scale), -scale, scale-1)/scale
        if s.dropout_ms:
            period = max(1, round(s.dropout_every*RATE))
            duration = max(1, round(s.dropout_ms*RATE/1000))
            indices = np.arange(self.sample, self.sample+len(x))
            # First dropout starts after one full period, not on acquisition.
            missing = (indices >= period) & (indices % period < duration)
            x[missing] = 0
        self.sample += len(x)
        return x.astype(np.float32)
