"""Line the two input channels up before decoding.

The packet header, clock word and metadata are sent identically on both
channels, so a receiver that hears one channel a few samples behind the
other (a dragged track in an editor, tape azimuth, a cable or interface
skew) sees them smeared or comb-filtered in their sum.  This sits in front
of everything else in the live receiver: it measures the offset between the
channels by cross-correlation over the last fraction of a second
(dominated by the headers, which are the loudest common part), and delays
whichever channel is ahead by that many samples.

A new offset is taken only when two measurements in a row agree and the
correlation peak stands clear of the rest, so the picture data's own
inter-channel differences (stereo M/S, two independent wires) do not move
it.  Polarity is left to the leveller, which already handles an inverted
channel.
"""
import numpy as np

WINDOW_SECONDS = .2      # audio each measurement looks at (>= 2 packets)
EVERY_SECONDS = .25      # how often the offset is measured until locked
LOCKED_SECONDS = 2.0     # how often once several measurements agree
LOCK_AFTER = 4           # agreeing measurements that count as locked
MAX_SECONDS = .02         # largest offset searched (20 ms: a dragged track)
MIN_CLARITY = 8.0        # peak against the median correlation magnitude
AGREE = 2                # measurements that must agree before a change


def measure(left, right, max_lag):
    """(offset, clarity): how many samples ``right`` lags ``left`` (negative:
    it leads), by cross-correlation."""
    n = len(left)
    size = 1 << int(np.ceil(np.log2(n+max_lag+1)))
    a = np.fft.rfft(left-np.mean(left), size)
    b = np.fft.rfft(right-np.mean(right), size)
    # Plain correlation: the loud common part (the headers) dominates. A
    # phase-only (whitened) correlation gives quiet content an equal say,
    # and quiet content common to both channels at zero lag outvoted a real
    # 1,110-sample offset on a recording.
    corr = np.fft.irfft(b*np.conj(a), size)
    lags = np.concatenate((corr[:max_lag+1], corr[-max_lag:]))
    offsets = np.concatenate((np.arange(max_lag+1), np.arange(-max_lag, 0)))
    mag = np.abs(lags)
    best = int(np.argmax(mag))
    clarity = float(mag[best]/max(np.median(mag), 1e-12))
    return int(offsets[best]), clarity


class ChannelAligner:
    """Delays the leading channel of a stereo stream by the measured offset."""

    def __init__(self, rate):
        self.rate = float(rate)
        self.window = int(round(WINDOW_SECONDS*self.rate))
        self.every = int(round(EVERY_SECONDS*self.rate))
        self.every_locked = int(round(LOCKED_SECONDS*self.rate))
        self.max_lag = max(1, int(round(MAX_SECONDS*self.rate)))
        self.offset = 0                  # applied: right lags left by this
        self.candidate, self.agreed = 0, 0
        self.history = np.zeros((0, 2), np.float32)
        self.since = 0
        self.pending = np.zeros((0, 2), np.float32)   # delay line
        self.last = None                  # (offset, clarity) of last measurement

    def _remeasure(self):
        if len(self.history) < self.window:
            return
        seg = self.history[-self.window:].astype(float)
        if np.max(np.abs(seg)) <= 1e-5:
            return
        offset, clarity = measure(seg[:, 0], seg[:, 1], self.max_lag)
        self.last = (offset, clarity)
        if clarity < MIN_CLARITY:
            return
        if offset == self.candidate:
            self.agreed += 1
        else:
            self.candidate, self.agreed = offset, 1
        if self.agreed >= AGREE and offset != self.offset:
            self.offset = offset

    def process(self, block):
        """The block with the channels lined up (same length, same dtype);
        mono or wider input passes through unchanged."""
        block = np.asarray(block)
        if block.ndim != 2 or block.shape[1] != 2 or not len(block):
            return block
        # Measure on what arrives, before alignment.
        block32 = np.asarray(block, np.float32)
        self.history = np.concatenate((self.history, block32))[-self.window:]
        self.since += len(block)
        locked = self.agreed >= LOCK_AFTER and self.candidate == self.offset
        if self.since >= (self.every_locked if locked else self.every):
            self.since = 0
            self._remeasure()
        # Delay line: the leading channel waits |offset| samples.
        if not self.offset:
            self.pending = self.pending[:0]
            return block32.astype(block.dtype, copy=False)
        joined = np.concatenate((self.pending, block32))
        lead = 0 if self.offset > 0 else 1       # right lags: left leads
        delay = abs(self.offset)
        out = np.empty((len(block), 2), np.float32)
        # The lagging channel is taken as it comes (the newest samples).
        out[:, 1-lead] = joined[len(joined)-len(block):, 1-lead]
        start = len(joined)-len(block)-delay
        if start >= 0:
            out[:, lead] = joined[start:start+len(block), lead]
        else:
            out[:, lead] = 0.0
            take = len(block)+start
            if take > 0:
                out[-start:, lead] = joined[:take, lead]
        # Keep the newest ``delay`` samples (all of them while the line is
        # still filling) for the next block.
        self.pending = (joined[max(len(joined)-delay, 0):] if delay else joined[:0])
        return out.astype(block.dtype, copy=False)
