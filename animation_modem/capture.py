"""Input rate negotiation and continuous conversion to the modem sample rate."""
from contextlib import contextmanager
from fractions import Fraction

import numpy as np
from scipy.signal import firwin, upfirdn

from .transport2 import RATE

# PortAudio tests specific settings; it does not enumerate every supported rate.
STANDARD_RATES = (768000, 705600, 384000, 352800, 192000, 176400,
                  96000, 88200, 64000, 48000, 44100, 32000, 24000,
                  22050, 16000, 12000, 11025, 8000)


@contextmanager
def open_input(sd, device, channels, default_rate):
    """Use the highest tested rate that both opens and starts successfully."""
    rates = sorted(set(STANDARD_RATES) | {float(default_rate)}, reverse=True)
    failures = []
    stream = None
    for rate in rates:
        if rate <= 0 or not np.isfinite(rate):
            continue
        candidate = None
        try:
            sd.check_input_settings(device=device, channels=channels,
                                    dtype='float32', samplerate=rate)
            candidate = sd.InputStream(device=device, channels=channels,
                                       dtype='float32', samplerate=rate,
                                       blocksize=0, latency='low')
            candidate.start()
        except sd.PortAudioError as exc:
            if candidate is not None:
                candidate.close()
            failures.append(f'{rate:g} Hz: {exc}')
            continue
        except BaseException:
            # No context manager has been entered yet: interrupted startup
            # must still release the device handle.
            if candidate is not None:
                candidate.close()
            raise
        stream = candidate
        break
    if stream is None:
        raise RuntimeError('No usable input sample rate. ' + '; '.join(failures))
    try:
        yield stream
    finally:
        try:
            stream.stop()
        finally:
            stream.close()


class CaptureResampler:
    """Causal polyphase FIR with retained history and rational sample phase.

    Complete denominator-sized groups are processed. At 44.1 kHz that holds
    fewer than 147 input samples; integer downsampling holds fewer than one
    output sample. Filter delay is reported separately.
    """
    def __init__(self, input_rate, channels):
        if not np.isfinite(input_rate) or input_rate <= 0:
            raise ValueError('Input sample rate must be positive and finite')
        ratio = (Fraction(RATE) / Fraction(str(input_rate))).limit_denominator(10000)
        if abs(float(ratio) * input_rate - RATE) > .001:
            raise ValueError('Unsupported input sample-rate ratio')
        self.up, self.down = ratio.numerator, ratio.denominator
        self.channels = channels
        self.input_rate = input_rate
        if self.up == self.down:
            self.taps = np.ones(1)
            self.history_length = 0
        else:
            scale = max(self.up, self.down)
            self.taps = firwin(20*scale+1, 1/scale, window=('kaiser', 5.0)) * self.up
            needed = (len(self.taps)-1+self.up-1)//self.up
            self.history_length = ((needed+self.down-1)//self.down)*self.down
        self.filter_delay_ms = (len(self.taps)-1)/(2*self.up*input_rate)*1000
        self.max_buffer_ms = (self.down-1)/input_rate*1000
        self.reset()

    def reset(self):
        self.history = np.zeros((self.history_length, self.channels), np.float32)
        self.pending = np.empty((0, self.channels), np.float32)

    def process(self, block):
        block = np.asarray(block, np.float32)
        if block.ndim != 2 or block.shape[1] != self.channels or not np.isfinite(block).all():
            raise ValueError('Capture samples must be finite with the configured channel count')
        if self.up == self.down:
            return block
        self.pending = np.concatenate((self.pending, block))
        count = len(self.pending)//self.down*self.down
        if not count:
            return np.empty((0, self.channels), np.float32)
        joined = np.concatenate((self.history, self.pending[:count]))
        self.pending = self.pending[count:].copy()
        converted = upfirdn(self.taps, joined, self.up, self.down, axis=0)
        start = self.history_length*self.up//self.down
        output = converted[start:start+count*self.up//self.down]
        self.history = joined[-self.history_length:].copy()
        return output.astype(np.float32)
