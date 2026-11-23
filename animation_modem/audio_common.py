"""Small audio adapters shared by the V7 sender and output path."""
import argparse
import numpy as np

REFERENCE_RATE = 48000
N = 128


def device(value):
    return int(value) if isinstance(value, str) and value.isdecimal() else value


def pair(value):
    try:
        channels = tuple(int(x) - 1 for x in value.split(','))
        if len(channels) != 2 or min(channels) < 0 or channels[0] == channels[1]:
            raise ValueError()
        return channels
    except ValueError:
        raise argparse.ArgumentTypeError(
            'Use two distinct 1-based channels, e.g. 1,2 or 3,4')


def sounddevice():
    try:
        import sounddevice as sd
        return sd
    except (ImportError, OSError) as exc:
        raise RuntimeError(
            'Live audio requires sounddevice and PortAudio. See README.md.') from exc


def pcm(data):
    return np.rint(np.clip(data, -1, 1) * 32767).astype('<i2').tobytes()


def route(data, channels):
    out = np.zeros((len(data), max(channels) + 1), np.float32)
    out[:, channels] = data
    return out


def input_pair(channels, available):
    """Resolve logical L/R channels, retaining automatic mono-input support."""
    if channels is None:
        return (0,) if available == 1 else (0, 1)
    channels = tuple(channels)
    if len(channels) != 2 or min(channels) < 0 or len(set(channels)) != 2:
        raise ValueError('Input channels must be two distinct positive channels')
    if max(channels) >= available:
        raise ValueError('Selected input channels unavailable on this device')
    return channels
