import argparse
import wave
import numpy as np
from .transport import RATE


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
