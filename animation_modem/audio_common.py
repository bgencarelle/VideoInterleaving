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
    """Lift a too-quiet input up to the level the decoder's thresholds expect.

    Acquisition uses an ABSOLUTE threshold: `edge_intervals` triggers a Schmitt
    at +/-0.066, 12% of the preamble's nominal 0.55. A quiet input never crosses
    it; measured, 0.12 of nominal decodes every frame and 0.06 acquires nothing
    at all. The leveller's only job is to lift such a signal back into the
    decodable range. It is one-way: it never reduces a hot input. Hot input is
    the limiter's job, and the two are decoupled so they cannot pump each other.

    PER CHANNEL, against what the preamble should be. The preamble is
    transmitted IDENTICALLY on both channels, so whatever level difference
    arrives between them is the recording, not the signal. A single pair gain
    leaves that difference in place and costs the quieter channel entirely.
    Amplitude between channels is not information (inter-channel PHASE is --
    head azimuth -- and a real gain does not touch it), so a per-channel lift
    is safe.

    The lift engages only below the do-nothing floor `window[0]`, tracked on a
    slow per-channel peak (fast attack, slow release). A channel already at or
    above that floor is left exactly alone, and a stale lift unwinds toward
    unity once the raw channel is no longer weak. `balance_db` caps how far the
    two gains may separate, so a dead leg is not lifted into loud hiss that
    manufactures false edges, and `step_db`/`jump_db` keep the gain near-constant
    within a packet -- its channel estimate comes from the training symbols at
    its front.

    Hot input goes through a feed-forward limiter that scales the block down to
    `ceiling` without touching the levelling gain, so a hot transient can never
    ratchet the leveller and there is no return spring for the two to fight.
    """

    def __init__(self, target=.7, ceiling=.95, floor=1e-4, step_db=.02,
                 jump_db=6., release=.995, limits=(1.0, 1e3), window=(.25, .98),
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
        self.gain = np.ones(2)      # leveller: lift-only, >= 1.0
        self.limit = np.ones(2)     # limiter: hot side, <= 1.0, independent
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
        # Lift-only, with a dead-zone. Engage only while the post-gain level is
        # still below the do-nothing floor: once a weak channel reaches the
        # window it freezes there, so the tracked peak's slow release cannot
        # make the gain creep. The target is floored at unity, so the leveller
        # never reduces a hot channel -- that is the limiter's job.
        lifting = alive & (scaled < self.window[0])
        wanted = np.maximum(self.target/np.maximum(self.peak, self.floor), 1.0)
        ratio = wanted/self.gain
        far = ratio > self.jump
        moved = np.where(far, wanted, self.gain*np.clip(ratio, 1/self.step, self.step))
        self.gain = np.where(lifting, moved, self.gain)
        self.gain = np.clip(self.gain, self.low, self.high)
        # Never boost one channel more than balance_db past the other: a dead
        # leg would otherwise be lifted until its own noise floor triggers the
        # Schmitt, inventing edges where there is no preamble at all.
        ceiling_gain = self.gain.min()*self.balance
        self.gain = np.minimum(self.gain, ceiling_gain)
        # Unwind a stale lift once the RAW channel is no longer weak. This only
        # ever walks a lift (> 1.0) back to unity; the limiter below never
        # touches this gain, so there is no hot-side spring to pump against it.
        toward = np.where(self.gain > 1.0,
                          np.maximum(1.0, self.gain/self.step), self.gain)
        self.gain = np.where(alive & (self.peak >= self.window[0]), toward, self.gain)
        out = audio*self.gain
        # Limiter (hot side), its own gain, decoupled from the leveller. Instant
        # attack pulls `limit` down to hold the block at the ceiling; it only
        # releases back toward unity while the tracked (sustained) input peak is
        # under the ceiling, so a still-hot signal stays uniformly attenuated
        # instead of breathing block-to-block. `limit` persists, so the cut is
        # uniform across a packet -- and it never touches the levelling gain, so
        # the two cannot pump each other.
        top = np.max(np.abs(out), axis=0)
        hot = top > self.ceiling
        attack = np.minimum(self.limit, self.ceiling/np.maximum(top, 1e-12))
        release = np.minimum(1.0, self.limit*self.step)
        self.limit = np.where(hot, attack,
                              np.where(self.peak <= self.ceiling, release, self.limit))
        self.limited += int(np.any(hot))
        out = out*self.limit
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

