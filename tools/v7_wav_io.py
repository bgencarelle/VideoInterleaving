"""Streaming WAV files in logical modem-channel order (before device routing)."""
from pathlib import Path

import soundfile as sf


class WavWriter:
    """Float WAV preserves the samples submitted to PortAudio without clipping."""

    def __init__(self, path, rate, channels=2, append=False):
        path = Path(path).expanduser()
        self.rate = int(round(rate))
        self.channels = int(channels)
        if append:
            self.file = sf.SoundFile(str(path), mode='r+')
            if (self.file.samplerate != self.rate or
                    self.file.channels != self.channels):
                self.file.close()
                raise ValueError('Recording rate/channels changed after reconnect; '
                                 'start a new recording')
            self.file.seek(0, sf.SEEK_END)
        else:
            self.file = sf.SoundFile(str(path), mode='w', samplerate=self.rate,
                                     channels=self.channels, format='WAV',
                                     subtype='FLOAT')

    def write(self, samples):
        self.file.write(samples)

    def close(self):
        self.file.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def wav_blocks(path, size=1024):
    """Yield float32 blocks; libsndfile supports PCM and float recordings."""
    with sf.SoundFile(str(Path(path).expanduser())) as source:
        while True:
            block = source.read(size, dtype='float32', always_2d=True)
            if not len(block):
                break
            yield block


def wav_info(path):
    info = sf.info(str(Path(path).expanduser()))
    return info.samplerate, info.channels, info.frames
