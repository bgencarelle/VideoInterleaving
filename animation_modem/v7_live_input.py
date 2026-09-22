"""Live V7 receive input: polarity-corrected storage, a minimal rolling
buffer, and an honest incoming frame rate.

No audio device, threads or UI live here; tools/v7_live.py feeds captured
blocks in and decodes what take() hands back.  Keeping the policy here makes
it testable without sounddevice.

Sample rate.  Everything here is measured in captured samples.  The capture
rate need not be 48 kHz: a frame is PULSE_FRAME x scale samples, where the
scale (capture rate / (48 kHz x playback speed)) is read off each header, so
frame-sized lengths below all scale with it.  Only incoming_fps() needs the
real rate, to convert samples to seconds.

Buffer policy.  A latest-only pulse decode needs one complete frame plus the
next frame's header (the commit boundary), so two frames of audio at the
current playback speed plus a small guard always suffice.  Display latency is
set by that header wait, not by how much history is kept, so keeping more only
costs CPU.  While unlocked (start-up, after repeated failures) the frame length
is unknown, so the buffer allows the slowest accepted speed instead.
"""
from collections import deque

import numpy as np

from animation_modem import transport3 as PULSE
from animation_modem.v7 import (META_SYMBOL, PULSE_FRAME, PULSE_MAX_SCALE,
                                PULSE_MIN_SCALE, RATE, leg_polarity,
                                pulse_frame_starts)

GUARD_FRAMES = .25          # header, timing tolerance and block granularity
UNLOCK_AFTER = 3            # failed decodes before the frame length is forgotten
RATE_WINDOW_S = 2.0         # averaging window for displayed rates
RATE_STALE_S = 1.0          # a rate with no event this long reads 0
# A header is found only if SYNC_LEN + META_SYMBOL + 32 samples follow its
# scan point; successive incoming scans overlap by a little more than that.
_HEADER_OVERLAP = PULSE.SYNC_LEN + META_SYMBOL + 64
# Two hits closer than half the shortest accepted frame are the same header.
_SAME_HEADER = PULSE_FRAME*PULSE_MIN_SCALE/2


def windowed_rate(times, now, window=RATE_WINDOW_S, stale=RATE_STALE_S):
    """Events per second over the last `window` seconds of `times`.

    Uses the span between the first and last event in the window, so it is
    exact for a steady stream, and reads 0 once no event arrived for `stale`
    seconds (instead of freezing at the last value).
    """
    times = list(times)          # one atomic copy: another thread may append
    if not times or now - times[-1] > stale:
        return 0.0
    recent = [t for t in times if times[-1] - t <= window]
    if len(recent) < 2 or recent[-1] <= recent[0]:
        return 0.0
    return (len(recent)-1)/(recent[-1]-recent[0])


class LiveInput:
    """Rolling live input.  Feed blocks with add(); take() returns audio to
    decode exactly when a new frame header has arrived (a new frame is then
    complete), so decode cycles follow the wire rather than the capture block
    size; call decoded() after each decode."""

    def __init__(self, decode_history=1, decode_batch=1, rate=RATE):
        self.decode_history = max(1, int(decode_history))
        self.decode_batch = max(1, int(decode_batch))
        self.rate = rate
        self.polarity = 1          # right-leg sign applied to stored input
        self.scale = None          # scale of the latest header (1/playback speed)
        self.total = 0             # samples ever stored (absolute clock)
        self._blocks = []
        self._judged_to = 0        # absolute sample up to which polarity was judged
        self._scan_from = 0        # absolute sample where the next header scan starts
        self._pending = 0          # headers arrived since the last take()
        self._headers = deque(maxlen=256)     # absolute header positions
        self._header_walls = deque(maxlen=256)

    # ------------------------------------------------------------ buffer
    def span(self):
        """Samples in one frame at the current speed (slowest speed if unknown)."""
        return PULSE_FRAME*(self.scale if self.scale else PULSE_MAX_SCALE)

    def _scaled(self, samples):
        """A nominal (48 kHz, 1x) length at the current frame scale."""
        return samples*(self.scale if self.scale else PULSE_MAX_SCALE)

    def cap(self):
        return int(self.span()*(1 + self.decode_history + GUARD_FRAMES))

    def buffered(self):
        return sum(len(block) for block in self._blocks)

    def reset(self):
        """Discontinuity (dropped input block): never stitch across it."""
        self._blocks.clear()
        self._judged_to = self._scan_from = self.total
        self._pending = 0
        self.scale = None
        self._headers.clear()
        self._header_walls.clear()

    def add(self, block):
        """Store one captured block (a private copy), polarity-corrected."""
        if self.polarity < 0 and block.ndim == 2 and block.shape[1] == 2:
            block[:, 1] *= -1
        self._blocks.append(block)
        self.total += len(block)

    def take(self, now):
        """Audio to decode now, or None until a new header has arrived.

        Every call judges polarity on the newest audio, scans only the audio
        not scanned yet for headers, and trims the buffer to two frames plus
        the guard, so memory and per-call work stay bounded with no signal.
        """
        if not self._blocks:
            return None
        audio = (self._blocks[0] if len(self._blocks) == 1
                 else np.concatenate(self._blocks))
        self._judge_polarity(audio)
        self._pending += self._count_headers(audio, now)
        if (self._headers and
                self.total - self._headers[-1] > 2*self._scaled(PULSE_FRAME)):
            self.scale = None                  # lost the stream: unlock
        cap = self.cap()
        if len(audio) > cap:
            audio = audio[-cap:]
        self._blocks = [audio]
        if self._pending < self.decode_batch:
            return None
        self._pending = 0
        return audio

    def decoded(self):
        """After a decode keep one frame plus the guard: it holds the header
        that starts the next frame."""
        if not self._blocks:
            return
        keep = int(self.span()*(self.decode_history + GUARD_FRAMES))
        audio = self._blocks[-1] if len(self._blocks) == 1 else np.concatenate(self._blocks)
        self._blocks = [audio[-keep:]] if len(audio) > keep else [audio]

    # ------------------------------------------------------------ polarity
    def _judge_polarity(self, audio):
        """Stored audio is already corrected, so a strongly negative L/R
        correlation over the newest frame means the correction is now wrong
        (a rewire or new tape): toggle it and re-flip the audio not judged
        before -- all of it at start-up, at least that newest frame.  Audio
        already judged correct is left alone."""
        frame = int(self.span())
        if len(audio) >= frame//2 and leg_polarity(audio[-frame:], 1) < 0:
            self.polarity = -self.polarity
            judged = self._judged_to - (self.total - len(audio))
            audio[max(min(judged, len(audio) - frame), 0):, 1] *= -1
        self._judged_to = self.total

    # ------------------------------------------------------------ incoming
    def _count_headers(self, audio, now):
        """New headers in the audio not scanned yet.  A header is accepted
        only once enough audio follows it for the decoder to use it; one
        closer to the end is left for the next (overlapping) scan."""
        start = self.total - len(audio)
        begin = max(self._scan_from - start, 0)
        # The decoder's own header scan needs SYNC_LEN + META_SYMBOL + 32
        # samples after a header's scan point regardless of scale, and the
        # header itself grows with scale: honour whichever is longer.
        overlap = max(self._scaled(_HEADER_OVERLAP), _HEADER_OVERLAP)
        found = 0
        for frame_start, scale, _ in pulse_frame_starts(audio[begin:]):
            position = start + begin + frame_start
            if position > self.total - overlap:
                break
            if self._headers and position - self._headers[-1] < _SAME_HEADER:
                continue
            self._headers.append(position)
            self._header_walls.append(now)
            self.scale = float(scale)
            found += 1
        self._scan_from = max(self._scan_from, int(self.total - overlap))
        return found

    def incoming_fps(self, now):
        """Readable frames per second arriving on the input.

        Measured on the audio sample clock between detected headers, so it
        is the wire rate as captured (12.245 fps at 1x, scaled by playback
        speed) and drops when headers are unreadable.  Reads 0 once no header
        was seen for RATE_STALE_S of wall time.
        """
        walls, headers = list(self._header_walls), list(self._headers)
        if not walls or not headers or now - walls[-1] > RATE_STALE_S:
            return 0.0
        return windowed_rate([p/self.rate for p in headers], headers[-1]/self.rate)
