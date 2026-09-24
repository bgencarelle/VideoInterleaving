"""Live V7 receive input: polarity-corrected storage, a minimal rolling
buffer, and an honest incoming frame rate.

No audio device, threads or UI live here; tools/v7_live.py feeds captured
blocks in and decodes what take() hands back.  Keeping the policy here makes
it testable without sounddevice.

Sample rate. Everything here is measured in captured samples. The scale read
from a header is capture_rate / (48 kHz x playback_speed); its accepted bounds
are multiplied by capture_rate / 48 kHz, preserving the same playback-speed
range on every device. Frame-sized lengths below use the raw captured-sample
scale. Only incoming_fps() needs the real rate to convert samples to seconds.

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
from animation_modem.v7 import (META_SYMBOL, PULSE_FRAME, RATE, leg_polarity,
                                pulse_frame_starts,
                                pulse_sample_scale_bounds)

GUARD_FRAMES = .25          # header, timing tolerance and block granularity
UNLOCK_AFTER = 3            # failed decodes before the frame length is forgotten
RATE_WINDOW_S = 2.0         # averaging window for displayed rates
RATE_STALE_S = 1.0          # a rate with no event this long reads 0
# A header is found only if SYNC_LEN + META_SYMBOL + 32 samples follow its
# scan point; successive incoming scans overlap by a little more than that.
_HEADER_OVERLAP = PULSE.SYNC_LEN + META_SYMBOL + 64
# Input leveler. The pulse detector's edge hysteresis and the EOF marker floor
# are absolute levels that assume a preamble near PREAMBLE_AMPLITUDE, so the
# input must be levelled *before* headers are searched: scanning the raw
# capture made every input more than ~20 dB below line level undetectable.
LEVEL_TARGET = PULSE.PREAMBLE_AMPLITUDE   # loudest 0.5% of a frame is set here
LEVEL_PERCENTILE = 99.5
GAIN_MIN, GAIN_MAX = .5, 32.0
GAIN_RISE = 1.5                           # per frame; reductions are immediate


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
        self.rate = float(rate)
        self.min_scale, self.max_scale = pulse_sample_scale_bounds(self.rate)
        # Two hits closer than half the shortest accepted frame are the same.
        self._same_header = PULSE_FRAME*self.min_scale/2
        self.polarity = 1          # right-leg sign applied to stored input
        # Raw captured-sample scale: capture_rate/(48 kHz x playback speed).
        self.scale = None
        self.total = 0             # samples ever stored (absolute clock)
        self._blocks = []
        self._judged_to = 0        # absolute sample up to which polarity was judged
        self._scan_from = 0        # absolute sample where the next header scan starts
        self._pending = 0          # headers arrived since the last take()
        # Absolute frame start, measured scale and pulse confidence. The live
        # decoder consumes these anchors so it need not scan this same window
        # for headers a second time.
        self._headers = deque(maxlen=256)
        self._header_walls = deque(maxlen=256)
        # Leveler gain applied to header scans; the receiver decodes with the
        # same gain so its own anchor re-check sees the same levels.
        self.gain = 1.0
        self._leveled_to = 0       # absolute sample of the last level update

    # ------------------------------------------------------------ buffer
    def span(self):
        """Samples in one frame at the current speed (slowest speed if unknown)."""
        return PULSE_FRAME*(self.scale if self.scale else self.max_scale)

    def _scaled(self, samples):
        """A nominal (48 kHz, 1x) length at the current frame scale."""
        return samples*(self.scale if self.scale else self.max_scale)

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
        self._update_level(audio)
        self._pending += self._count_headers(audio, now)
        if (self._headers and
                self.total - self._headers[-1][0] >
                2*self._scaled(PULSE_FRAME)):
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

    def pulse_starts(self, audio):
        """Pulse anchors in `audio`, as ``(start, scale, confidence)``.

        `take()` returns a suffix of the absolute input history. Convert the
        already measured header positions into that suffix's sample clock.
        The decoder locally rechecks these anchors after its gain adjustment.
        """
        start = self.total - len(audio)
        return tuple((position-start, scale, confidence)
                     for position, scale, confidence in self._headers
                     if start <= position < self.total)

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

    # ------------------------------------------------------------ level
    def _update_level(self, audio):
        """Once per frame of new audio, move the gain toward setting the
        loudest 0.5% of the newest frame at LEVEL_TARGET: slow rise, immediate
        reduction. It runs whether or not a header has been found yet -- that
        is what lets a quiet input be raised far enough for its headers to
        become detectable. A reset keeps the gain: an input gap does not
        change the input level."""
        frame = int(self._scaled(PULSE_FRAME) if self.scale
                    else PULSE_FRAME*self.rate/RATE)
        if self.total-self._leveled_to < frame or len(audio) < frame:
            return
        self._leveled_to = self.total
        peak = float(np.percentile(np.abs(audio[-frame:]), LEVEL_PERCENTILE))
        desired = float(np.clip(LEVEL_TARGET/max(peak, 1e-6),
                                GAIN_MIN, GAIN_MAX))
        self.gain = (min(desired, self.gain*GAIN_RISE)
                     if desired > self.gain else desired)

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
        for frame_start, scale, confidence in pulse_frame_starts(
                audio[begin:]*np.float32(self.gain), sample_rate=self.rate):
            position = start + begin + frame_start
            if position > self.total - overlap:
                break
            if (self._headers and
                    position - self._headers[-1][0] < self._same_header):
                continue
            self._headers.append((position, float(scale), float(confidence)))
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
        walls = list(self._header_walls)
        headers = [header[0] for header in self._headers]
        if not walls or not headers or now - walls[-1] > RATE_STALE_S:
            return 0.0
        return windowed_rate([p/self.rate for p in headers], headers[-1]/self.rate)
