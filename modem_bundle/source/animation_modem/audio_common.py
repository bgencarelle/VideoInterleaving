import argparse
import wave
import numpy as np
from .transport2 import PRESETS, RATE, N


def device(value):
    return int(value) if value and value.isdecimal() else value


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


def wav_blocks(path, channels=(0, 1), block=256):
    with wave.open(str(path), 'rb') as w:
        if w.getframerate() != RATE or w.getsampwidth() != 2 or w.getnchannels() <= max(channels):
            raise ValueError('WAV input must be 48 kHz, 16-bit PCM with the selected channels')
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

    def __init__(self, band=BAND, order=4):
        from scipy.signal import butter
        high, low = band
        if not (0 < high < low < RATE/2):
            raise ValueError(f'Input band must satisfy 0 < high < low < {RATE/2}')
        carriers = PRESETS['wide'].carriers
        lowest, highest = carriers[0]*RATE/N, carriers[-1]*RATE/N
        if high > lowest or low < highest:
            raise ValueError(f'Input band must span the carriers '
                             f'({lowest:.0f}..{highest:.0f} Hz)')
        self.sos = np.concatenate([
            butter(order, high, btype='highpass', fs=RATE, output='sos'),
            butter(order, low, btype='lowpass', fs=RATE, output='sos')])
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
