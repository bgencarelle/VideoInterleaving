import argparse
import wave
import numpy as np
from .core import Layout, REFERENCE_RATE, N
from scipy.signal import butter, sosfilt

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


class InputLevel:
    """Bring each received channel to the level the decoder's thresholds expect.

    Acquisition uses an ABSOLUTE threshold: `edge_intervals` triggers a Schmitt
    at +/-0.066, 12% of the preamble's nominal 0.55. A quiet input never crosses
    it; measured, a signal at 0.12 of nominal decodes every frame and one at
    0.06 finds zero edges and acquires nothing at all -- not a degraded picture,
    silence.

    PER CHANNEL, against what the preamble should be. The preamble is
    transmitted IDENTICALLY on both channels, so whatever difference in level
    arrives between them is the recording, not the signal: one tape track
    biased hotter than the other, one leg of a cable padded. A single gain for
    the pair leaves that difference in place, and measured it costs the quieter
    channel entirely -- at 1.0/0.03 the loud channel shows 389 preamble edges
    and the quiet one ZERO. Acquisition still succeeds off the loud channel, so
    nothing looks wrong, but the both-channel preamble is half of v3's
    acquisition margin and a dropout on the survivor then has nothing to fall
    back to.

    Correcting it does not destroy anything the decoder wanted. Amplitude
    between the channels is not information here; the transmitter sent them
    equal. The inter-channel PHASE is information -- it is head azimuth, and
    `_channel_skew` reports it -- and a real gain per channel does not touch
    phase. Crosstalk is reported after this correction rather than before,
    which is the more useful of the two.

    The preamble holds the packet peak in about three frames in four, so a
    slow per-channel peak IS a preamble measurement, without needing to find
    the preamble first.

    Two brakes. `balance_db` caps how far apart the two gains may go, so a dead
    channel is never amplified into loud hiss that manufactures false edges.
    And the gain is near-constant across a packet -- `step_db` a block, about
    0.2 dB a packet -- because a packet's channel estimate comes from the
    training symbols at its front, so a gain that drifts within one scales the
    coefficients against an estimate taken at a different level.

    And one return spring. While the raw signal is healthy on its own (inside
    the window and under the ceiling) the gain therefore relaxes toward unity
    at the same slow step: gain differs from 1.0 only while the wire is
    unusable. A weak signal keeps its lift, a dead leg stays frozen, a hot
    input stays under the limiter -- those are the cases that need the gain.
    """

    def __init__(self, target=.7, ceiling=.95, floor=1e-4, step_db=.02,
                 jump_db=6., release=.995, limits=(1e-3, 1e3), window=(.25, .98),
                 balance_db=20.):
        if not 0 < target < ceiling <= 1:
            raise ValueError('Need 0 < target < ceiling <= 1')
        if not 0 < window[0] < target < window[1]:
            raise ValueError('Target must sit inside the do-nothing window')
        self.target, self.ceiling, self.floor = target, ceiling, floor
        self.window = window
        self.step = 10**(step_db/20)
        self.jump = 10**(jump_db/20)
        self.release = release
        self.low, self.high = limits
        self.balance = 10**(balance_db/20)
        self.gain = np.ones(2)
        self.peak = np.zeros(2)
        self.limited = 0          # blocks the limiter had to pull down

    def process(self, audio):
        audio = np.asarray(audio, np.float32)
        if audio.ndim != 2 or audio.shape[1] != 2 or not len(audio):
            return audio
        # Fast attack, slow release, tracked separately for each channel.
        self.peak = np.maximum(np.max(np.abs(audio), axis=0), self.peak*self.release)
        alive = self.peak > self.floor
        scaled = self.peak*self.gain
        # Leave a level that is already fine exactly alone. `encode` normalises
        # a packet to 0.95, so a healthy input is already where it belongs and
        # any gain would only add drift: measured, levelling a nominal signal
        # unconditionally moved reconstruction error from 0.0000 to 0.0016.
        settled = (scaled >= self.window[0]) & (scaled <= self.window[1])
        wanted = self.target/np.maximum(self.peak, self.floor)
        ratio = wanted/self.gain
        far = (ratio > self.jump) | (ratio < 1/self.jump)
        moved = np.where(far, wanted,          # nothing decoding; do not crawl
                         self.gain*np.clip(ratio, 1/self.step, self.step))
        self.gain = np.where(alive & ~settled, moved, self.gain)
        self.gain = np.clip(self.gain, self.low, self.high)
        # Never boost one channel more than balance_db past the other: a dead
        # leg would otherwise be lifted until its own noise floor triggers the
        # Schmitt, inventing edges where there is no preamble at all.
        ceiling_gain = self.gain.min()*self.balance
        self.gain = np.minimum(self.gain, ceiling_gain)
        # Relax to unity while the raw signal is healthy on its own. Gain is
        # only ever needed while the wire is unusable (too weak to acquire,
        # or being limited); a healthy raw peak means neither, so a stale
        # lift or limiter pull-down unwinds instead of ratcheting forever.
        # Gated on this block's own peak, not the tracked one (which decays
        # slowly by design and would stall the return for seconds after one
        # hot transient), and under the ceiling so it can never fight the
        # limiter below: clipped-flat audio peaks at 1.0 and stays governed
        # there. Pausing on odd blocks only modulates the return speed, never
        # its direction, so there is nothing here that can oscillate.
        instant = np.max(np.abs(audio), axis=0)
        healthy = (instant >= self.window[0]) & (instant <= self.ceiling)
        toward = np.where(self.gain > 1.0,
                          np.maximum(1.0, self.gain/self.step),
                          np.minimum(1.0, self.gain*self.step))
        self.gain = np.where(healthy, toward, self.gain)
        out = audio*self.gain
        # Feed-forward limit. Pulling the gain down immediately, and keeping
        # it down, beats clipping: a clipped OFDM symbol is broadband
        # distortion across every carrier at once. The pull-down also feeds
        # back into the leveller's next settled/healthy checks, so a quiet
        # input that was lifted too far converges back to the target instead
        # of sitting permanently clipped. It can only pull gain down (never
        # up), so it cannot ratchet the leveller upward.
        top = np.max(np.abs(out), axis=0)
        hot = top > self.ceiling
        if np.any(hot):
            self.gain = np.where(hot, self.gain*self.ceiling/np.maximum(top, 1e-12),
                                 self.gain)
            self.limited += 1
            out = audio*self.gain
        return out.astype(np.float32)



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


# Data occupies bins 1..54 only. Everything outside carries no information, but
# the lowest bin sits one bin from DC: a program tone, turntable rumble or a low
# EQ shelf below the band leaks into it with a phase that advances symbol to
# symbol. The training cannot predict a rotating leak the way it predicts a
# static EQ, so the bottom carrier's channel estimate is poisoned and identity
# is lost. Guarding one carrier above the band bottom removes the whole class,
# at the cost of the bottom carrier -- which the per-carrier equalizer already
# restores, the same way it restores a few dB of tone control.
GUARD_BINS = 1

# transport3 owns the wire, but it imports this module, so the guard band
# cannot be read off transport3.WIRE. This is the live wire spelled out as a
# bare Layout instead; modem_tests/test_guard_band.py pins it to transport3's
# WIRE so the two cannot drift apart silently.
WIRE = Layout(top_bin=54, image_symbols=14, name='wire',
              progressive=True, orthogonal_training=True,
              spread_carriers=True, dense_header=True, header_width=20)


class InputFilter:
    """Stateful guard band applied to received audio before acquisition.

    A high-pass one carrier above the band bottom, plus a low-pass one carrier
    above the top: out-of-band energy carries no picture, and below the bottom
    it actively corrupts the channel estimate. Order 4, not sharper -- measured,
    a 6th- or 8th-order rolloff disperses enough group delay at the front of
    the packet to move the detected preamble against the body and lose the
    header, and a causal rolloff *at* the edge does the same.
    """

    def __init__(self, rate=REFERENCE_RATE, band=None, order=4):
        from scipy.signal import butter
        carriers = WIRE.carriers
        # Carrier frequencies follow the clock the audio is actually arriving
        # on, so the guard has to be computed against that same clock.
        lowest, highest = carriers[0]*rate/N, carriers[-1]*rate/N
        if band is None:
            high = lowest + GUARD_BINS*rate/N
            low = highest + rate/N
        else:
            high, low = band
        if not (0 < high < low < rate/2):
            raise ValueError(f'Input band must satisfy 0 < high < low < {rate/2}')
        self.rate, self.high, self.low = rate, high, low
        self.sos = np.concatenate([
            butter(order, high, btype='highpass', fs=rate, output='sos'),
            butter(order, low, btype='lowpass', fs=rate, output='sos')])
        self.zi = np.zeros((len(self.sos), 2, 2))

    def reset(self):
        self.zi = np.zeros_like(self.zi)

    def process(self, data):
        from scipy.signal import sosfilt
        data = np.asarray(data, np.float32)
        if not len(data):
            return data
        out, self.zi = sosfilt(self.sos, data, axis=0, zi=self.zi)
        return out.astype(np.float32)

    def filtfilt(self, data):
        """Zero-phase form for a whole recording, where one exists.

        The receiver's timing is edge-counted, so a causal rolloff anywhere near
        an edge can move the detected preamble against the body; zero-phase has
        no such cost. That makes it the right guard for the offline `read` path,
        which is where recordings of real tape and processor chains are decoded.
        """
        from scipy.signal import sosfiltfilt
        data = np.asarray(data, np.float32)
        if not len(data):
            return data
        return sosfiltfilt(self.sos, data, axis=0).astype(np.float32)


def band(value):
    try:
        high, low = (float(v) for v in value.split(','))
    except ValueError:
        raise argparse.ArgumentTypeError('Use HIGHPASS,LOWPASS in Hz, e.g. 600,22000')
    return (high, low)

