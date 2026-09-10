"""Device-default capture and continuous conversion to the modem sample rate."""
from contextlib import contextmanager
from fractions import Fraction

import numpy as np
from scipy.signal import firwin, upfirdn

from .transport2 import RATE

@contextmanager
def open_input(sd, device, channels, default_rate):
    """Open once at the device-reported default rate, without probing rates."""
    rate = float(default_rate)
    if not np.isfinite(rate) or rate <= 0:
        raise ValueError('Device reported an invalid default sample rate')
    stream = sd.InputStream(device=device, channels=channels,
                            dtype='float32', samplerate=rate,
                            blocksize=0)
    try:
        stream.start()
    except BaseException:
        stream.close()
        raise
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


class BufferedInput:
    """Read independently of decoding; retain at most 150 ms of fresh audio.

    Sequence gaps mark discarded blocks, so filters and acquisition reset at
    the exact discontinuity. The buffer has capacity, not a prefill delay.
    """
    def __init__(self, stream, blocksize, max_seconds=.15):
        from collections import deque
        import threading
        self.stream, self.blocksize = stream, blocksize
        self.capacity = max(1, int(max_seconds*stream.samplerate/blocksize))
        self.queue = deque()
        self.lock = threading.Lock()
        self.stop = threading.Event()
        self.error = None
        self.sequence = 0
        self.expected = 0
        self.thread = threading.Thread(target=self._run, daemon=True)

    def _poll(self):
        if self.stream.read_available < self.blocksize:
            return False
        audio, overflow = self.stream.read(self.blocksize)
        # Own samples even if a backend reuses its read buffer.
        with self.lock:
            if len(self.queue) >= self.capacity:
                self.queue.popleft()
            self.queue.append((self.sequence, audio.copy(), overflow))
            self.sequence += 1
        return True

    def _run(self):
        try:
            while not self.stop.is_set():
                if not self._poll():self.stop.wait(.002)
        except BaseException as exc:
            self.error = exc

    def read(self):
        with self.lock:
            if not self.queue:
                if self.error is not None:raise self.error
                return None
            sequence, audio, overflow = self.queue.popleft()
        skipped = sequence-self.expected
        self.expected = sequence+1
        return audio, overflow, skipped

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *args):
        self.stop.set()
        self.thread.join(timeout=1.0)
        if self.thread.is_alive():
            # Unblock a driver read before its owning stream is closed.
            self.stream.abort()
            self.thread.join(timeout=1.0)
            if self.thread.is_alive():
                raise RuntimeError('Audio input reader did not stop')
