import argparse
import wave
import numpy as np
from .core import PRESETS, REFERENCE_RATE, N


def device(value):
    return int(value) if isinstance(value, str) and value.isdecimal() else value


def pair(value):
    try:
        p = tuple(int(x)-1 for x in value.split(','))
        if len(p) != 2 or min(p) < 0 or p[0] == p[1]:
            raise ValueError()
        return p
    except ValueError:
        raise argparse.ArgumentTypeError('Use two distinct 1-based channels, e.g. 1,2 or 3,4')


def sounddevice():
    try:
        import sounddevice as sd
        return sd
    except (ImportError, OSError) as exc:
        raise RuntimeError('Live audio requires sounddevice and PortAudio. See README.md.') from exc


def pcm(data):
    return np.rint(np.clip(data, -1, 1)*32767).astype('<i2').tobytes()


def wire_notice(layout, rate, reference=REFERENCE_RATE):
    """Report the band a transmitter is actually putting on the wire.

    Receiving is indifferent to the clock; transmitting is not, because the
    band emitted is the device's clock times the carrier geometry. Above the
    reference rate `band_limited` resamples to hold the carriers where they
    belong, so this reports a held band and the small rate offset the coarse
    resampling ratio leaves behind. Below it, no resampling happens and the
    band simply scales down with the clock, which is the safe direction.

    Either way the top carrier stays under 20.3 kHz, so any device from
    44.1 kHz up can carry it. Returns None at the reference rate.
    """
    from .core import emit_ratio
    if rate == reference:
        return None
    ratio = emit_ratio(rate, reference)
    if ratio is None:
        low, high = layout.band_at(rate)
        return (f'NOTE: output device is at {rate:g} Hz, below the '
                f'{reference:g} Hz reference. Carriers scale down with it, to '
                f'{low:.0f}-{high:.0f} Hz at {layout.fps_at(rate):.2f} fps. '
                f'Further inside the DAC\'s passband, so nothing to do.')
    up, down = ratio
    # Resampling by up/down plays the geometry as though the clock were this,
    # which is the reference rate up to the ratio's rounding error.
    effective = rate*down/up
    low, high = layout.band_at(effective)
    return (f'NOTE: output device is at {rate:g} Hz, above the {reference:g} Hz '
            f'reference. Packets are resampled {up}/{down} so the carriers stay '
            f'at {low:.0f}-{high:.0f} Hz instead of riding the clock up to '
            f'{layout.band_at(rate)[1]:.0f} Hz, where a low-end DAC would drop '
            f'them. Frame rate {layout.fps_at(effective):.2f} fps'
            + ('.' if round(effective) == reference else
               f'; the ratio lands {100*abs(effective-reference)/reference:.2f}% '
               f'off, which the receiver measures and reports rather than '
               f'corrects.'))


def wav_rate(path):
    """The rate the file says it was recorded at. Read, never imposed."""
    with wave.open(str(path), 'rb') as w:
        return w.getframerate()


def wav_blocks(path, channels=(0, 1), block=256):
    """Stereo float blocks from any sample rate.

    The rate used to be checked against 48 kHz and anything else rejected. That
    check bought nothing once timing came from preamble edges: a 44.1 kHz
    recording of a v3 signal decodes as a 1.088x transport, which is inside the
    supported range and exactly what the file contains. Pair this with
    `wav_rate` when the caller wants the result in seconds or hertz.
    """
    with wave.open(str(path), 'rb') as w:
        if w.getsampwidth() != 2 or w.getnchannels() <= max(channels):
            raise ValueError('WAV input must be 16-bit PCM with the selected channels')
        while raw := w.readframes(block):
            yield np.frombuffer(raw, '<i2').reshape(-1, w.getnchannels())[:, channels].astype(np.float32)/32768


def route(data, channels):
    out = np.zeros((len(data), max(channels)+1), np.float32)
    out[:, channels] = data
    return out


# Data occupies bins 3..54 only. Everything outside carries no information, but
# it does inflate the peak and window energy that sync_correlation normalises
# by, which is what pushes the score under threshold on an analog source. The
# demodulator's FFT already rejects it; the acquisition detector does not.
BAND = (600.0, 22000.0)          # comfortably outside 1125 Hz .. 20250 Hz


class InputFilter:
    """Stateful band-pass applied to received audio before acquisition.

    Removes turntable rumble, mains hum and out-of-band hiss. Costs about
    0.4 dB on a clean source and makes full-scale rumble a non-event.
    """

    def __init__(self, band=BAND, order=4, rate=REFERENCE_RATE):
        from scipy.signal import butter
        high, low = band
        if not (0 < high < low < rate/2):
            raise ValueError(f'Input band must satisfy 0 < high < low < {rate/2}')
        carriers = PRESETS['wide'].carriers
        # Carrier frequencies follow the clock the audio is actually arriving
        # on, so the guard band has to be computed against that same clock.
        lowest, highest = carriers[0]*rate/N, carriers[-1]*rate/N
        if high > lowest or low < highest:
            raise ValueError(f'Input band must span the carriers '
                             f'({lowest:.0f}..{highest:.0f} Hz)')
        self.rate = rate
        self.sos = np.concatenate([
            butter(order, high, btype='highpass', fs=rate, output='sos'),
            butter(order, low, btype='lowpass', fs=rate, output='sos')])
        self.zi = np.zeros((len(self.sos), 2, 2))

    def process(self, data):
        from scipy.signal import sosfilt
        data = np.asarray(data, np.float32)
        if not len(data):
            return data
        out, self.zi = sosfilt(self.sos, data, axis=0, zi=self.zi)
        return out.astype(np.float32)


def band(value):
    try:
        high, low = (float(v) for v in value.split(','))
    except ValueError:
        raise argparse.ArgumentTypeError('Use HIGHPASS,LOWPASS in Hz, e.g. 600,22000')
    return (high, low)
