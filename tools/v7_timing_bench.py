#!/usr/bin/env python3
"""V7 timing bench: time and pitch variation on the real wire.

Encodes a short Aspect Fold 500 packet stream, passes it through one channel
condition, decodes it, and reports packets decoded and the mean luma PSNR of
the decoded pictures against a clean decode (capped at 60 dB). Two decode paths are measured:

- ``stream``: decode_pulse_stream over the whole capture (recordings, WAVs);
- ``live``: one latest-only decode per arriving packet, as the live receiver
  calls it, on a rolling window.

Channel conditions (synthetic, deterministic, no extra dependencies):
speed (resampled: time and pitch together), flutter (random speed wobble),
splice (chunks cut or repeated, as digital time-stretchers do), pitch
(resampled, then splices restore the duration: pitch without tempo), tempo
(splices only: tempo without pitch), and noise. With ``--ffmpeg`` the
ffmpeg ``atempo`` time-stretcher and an ``asetrate``/``atempo`` pitch shift
are added.

    python tools/v7_timing_bench.py            # synthetic conditions
    python tools/v7_timing_bench.py --ffmpeg   # plus ffmpeg stretch/pitch
"""
import argparse
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for extra in (ROOT, ROOT/'test_modem_v7'):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))

from animation_modem import v7                                          # noqa: E402

RATE = v7.RATE
PACKETS = 12


# ------------------------------------------------------------------ channels
def resample(audio, factor):
    """Play ``factor`` times faster: time and pitch scale together."""
    positions = np.arange(0, len(audio)-1, factor)
    base = np.arange(len(audio))
    return np.column_stack([np.interp(positions, base, audio[:, ch])
                            for ch in range(audio.shape[1])]).astype(np.float32)


def flutter(audio, rms, band=(20, 300), seed=11):
    """Random speed wobble (RMS fraction of speed), band-limited."""
    from scipy.signal import butter, sosfilt
    rng = np.random.default_rng(seed)
    wobble = sosfilt(butter(2, band, btype='bandpass', fs=RATE, output='sos'),
                     rng.standard_normal(len(audio)))
    wobble *= rms/np.sqrt(np.mean(wobble*wobble))
    positions = np.clip(np.arange(len(audio)) + np.cumsum(wobble),
                        0, len(audio)-1)
    base = np.arange(len(audio))
    return np.column_stack([np.interp(positions, base, audio[:, ch])
                            for ch in range(audio.shape[1])]).astype(np.float32)


def splice(audio, change, chunk=480, seed=5, crossfade=48):
    """Cut (change > 0) or repeat (change < 0) chunks, as WSOLA stretchers do.

    ``change`` is the fractional duration change to make (0.1 removes 10%).
    Cuts land at random positions, ``chunk`` samples each, with a short
    crossfade like a time-stretcher's overlap-add.
    """
    rng = np.random.default_rng(seed)
    total = int(round(abs(change)*len(audio)))
    count = max(1, total//chunk) if total else 0
    # One cut per equal segment, at a random place inside it, so cuts never
    # overlap.
    segment = len(audio)//max(count, 1)
    margin = chunk+2*crossfade
    positions = [k*segment+int(rng.integers(margin, segment-margin))
                 for k in range(count)]
    pieces, last = [], 0
    for position in positions:
        if change > 0:                       # drop [position, position+chunk)
            pieces.append(audio[last:position])
            last = position+chunk
        else:                                # repeat [position-chunk, position)
            pieces.append(audio[last:position])
            pieces.append(audio[position-chunk:position])
            last = position
    pieces.append(audio[last:])
    out = pieces[0]
    for piece in pieces[1:]:
        fade = min(crossfade, len(out), len(piece))
        ramp = np.linspace(0, 1, fade, dtype=np.float32)[:, None]
        joined = out[-fade:]*(1-ramp) + piece[:fade]*ramp
        out = np.concatenate((out[:-fade], joined, piece[fade:]))
    return out.astype(np.float32)


def wsola(audio, stretch, window=1024, tolerance=256):
    """Time-stretch by ``stretch`` (output/input duration) keeping pitch.

    A plain WSOLA: Hann-windowed frames overlap-added at half-window hops,
    each analysis frame shifted within +-``tolerance`` to the position whose
    waveform best continues the output (as SoundTouch and ffmpeg atempo do).
    """
    audio = np.asarray(audio, np.float32)
    hop = window//2
    analysis_hop = hop/stretch
    mono = audio.mean(axis=1)
    taper = np.hanning(window).astype(np.float32)[:, None]
    frames = int((len(audio)-window-2*tolerance)/analysis_hop)
    out = np.zeros((frames*hop+window, audio.shape[1]), np.float32)
    norm = np.zeros(len(out), np.float32)
    previous = None
    for k in range(frames):
        nominal = int(round(k*analysis_hop))+tolerance
        start = nominal
        if previous is not None:
            # Best continuation of the previous frame's natural successor.
            target = mono[previous+hop:previous+hop+window]
            lo, hi = nominal-tolerance, nominal+tolerance
            region = mono[lo:hi+window]
            scores = np.correlate(region, target, mode='valid')
            start = lo+int(np.argmax(scores))
        out[k*hop:k*hop+window] += audio[start:start+window]*taper
        norm[k*hop:k*hop+window] += taper[:, 0]
        previous = start
    return (out/np.maximum(norm, 1e-3)[:, None]).astype(np.float32)


def wsola_pitch(audio, factor, window=1024, tolerance=256):
    """Pitch by ``factor`` with the duration kept (resample, then WSOLA)."""
    return wsola(resample(audio, factor), factor, window, tolerance)


def pitch(audio, factor, chunk=480, seed=5):
    """Pitch up by ``factor`` with the duration restored by repeated chunks."""
    raised = resample(audio, factor)
    return splice(raised, 1/factor-1, chunk=chunk, seed=seed)


def noise(audio, snr_db, seed=3):
    rng = np.random.default_rng(seed)
    level = np.sqrt(np.mean(audio**2))*10**(-snr_db/20)
    return (audio + rng.standard_normal(audio.shape)*level).astype(np.float32)


def ffmpeg_filter(audio, af):
    from scipy.io import wavfile
    with tempfile.TemporaryDirectory() as folder:
        source, target = os.path.join(folder, 'a.wav'), os.path.join(folder, 'b.wav')
        wavfile.write(source, RATE, np.asarray(audio, np.float32))
        subprocess.run(['ffmpeg', '-loglevel', 'error', '-y', '-i', source,
                        '-af', af, '-ar', str(RATE), '-c:a', 'pcm_f32le',
                        target], check=True)
        return wavfile.read(target)[1].astype(np.float32)


def conditions(use_ffmpeg=False):
    cases = [
        ('clean', lambda a: a),
        ('speed 1.02 (resampled)', lambda a: resample(a, 1.02)),
        ('speed 0.97 (resampled)', lambda a: resample(a, .97)),
        ('flutter 0.1%', lambda a: flutter(a, .001)),
        ('flutter 0.3%', lambda a: flutter(a, .003)),
        ('noise 24 dB', lambda a: noise(a, 24)),
        ('tempo +3% (splices)', lambda a: splice(a, .03)),
        ('tempo +10% (splices)', lambda a: splice(a, .10)),
        ('tempo -10% (repeats)', lambda a: splice(a, -.10)),
        ('pitch +2% (tempo kept)', lambda a: pitch(a, 1.02)),
        ('pitch +6% (tempo kept)', lambda a: pitch(a, 1.06)),
        ('pitch -6% (tempo kept)', lambda a: pitch(a, .94)),
        ('tempo +10% + noise 24 dB', lambda a: noise(splice(a, .10), 24)),
        ('WSOLA pitch x1.25', lambda a: wsola_pitch(a, 1.25)),
        ('WSOLA pitch x1.5', lambda a: wsola_pitch(a, 1.5)),
        ('WSOLA pitch x0.8', lambda a: wsola_pitch(a, .8)),
    ]
    if use_ffmpeg and shutil.which('ffmpeg'):
        cases += [
            ('ffmpeg atempo 1.05', lambda a: ffmpeg_filter(a, 'atempo=1.05')),
            ('ffmpeg atempo 1.10', lambda a: ffmpeg_filter(a, 'atempo=1.10')),
            ('ffmpeg atempo 0.90', lambda a: ffmpeg_filter(a, 'atempo=0.90')),
            ('ffmpeg pitch +6%', lambda a: ffmpeg_filter(
                a, 'asetrate=50880,aresample=48000,atempo=0.94340')),
            ('ffmpeg pitch -6%', lambda a: ffmpeg_filter(
                a, 'asetrate=45120,aresample=48000,atempo=1.06383')),
        ]
    return cases


# ---------------------------------------------------------------- the wire
def encode(packets=PACKETS):
    """(wire, base model, layout model, audio, source values) for a test card."""
    from aspect_fold import AspectFoldWire
    from common import TARGET
    import tone_code
    base = v7.load_model(TARGET, 'box')
    wire = AspectFoldWire('16:9')
    from PIL import Image
    with Image.open(v7.REFERENCE_FIXTURE) as image:
        values = v7.image_values(v7.prepare_image(image.convert('RGB'), 'box'),
                                 v7.V7_GRIDS, 'box')
    model, coeffs = wire.encode_coefficients(base, values, 3)
    audio = np.concatenate([tone_code.add_tone_code(v7.encode_pulse_frame_coeffs(
        model, coeffs, counter, aspect_code=3, source_index=counter-1,
        pilot_tones=False,
        pulse_profile_code=wire.pulse_profile_code),
        counter, tone_code.encode_status(wire.status_mode))
        for counter in range(1, packets+1)])
    return wire, base, model, audio.astype(np.float32), values


class _Capture:
    """Attach each decoded packet's equaliser output (for unfolding)."""

    def __init__(self):
        self.eq = None
        self.real = (v7._equalize_numba, v7._equalize_numpy, v7.decode_frame)

    def __enter__(self):
        real_numba, real_numpy, real_decode = self.real

        def numba(*args, **kwargs):
            out = real_numba(*args, **kwargs)
            self.eq = (out[0].copy(), out[1].copy())
            return out

        def numpy(*args, **kwargs):
            out = real_numpy(*args, **kwargs)
            self.eq = (out[0].copy(), out[1].copy())
            return out

        def decode(*args, **kwargs):
            self.eq = None
            result = real_decode(*args, **kwargs)
            if result is not None and self.eq is not None:
                result.diag['fold_eq'] = self.eq
            return result
        v7._equalize_numba, v7._equalize_numpy, v7.decode_frame = (
            numba, numpy, decode)
        return self

    def __exit__(self, *exc):
        (v7._equalize_numba, v7._equalize_numpy,
         v7.decode_frame) = self.real


def _psnr(values, reference):
    luma = slice(0, 96*80)
    error = np.mean((np.asarray(values)[luma]-reference[luma])**2)
    return float(10*np.log10(4/max(error, 1e-12)))


def _score(wire, model, results, reference):
    """(packets decoded, mean luma PSNR of every decoded packet)."""
    good = [result for result in results
            if result.status != 'lost' and 'fold_eq' in result.diag]
    if not good:
        return 0, float('nan')
    quality = [min(_psnr(wire.values(model, result), reference), 60.0)
               for result in good]
    return len(good), float(np.mean(quality))


def decode_stream(wire, model, audio, reference):
    import tone_code
    with _Capture(), tone_code.coded_pilot_timing():
        results, _ = v7.decode_pulse_stream(
            model, audio, models={model.encoding_type: model},
            state=v7.PulseState(tail_memory=False), sample_rate=RATE,
            pilot_timing='tone-seeded')
    return _score(wire, model, results, reference)


def decode_live(wire, model, audio, reference):
    """Latest-only decodes on a rolling window, one per half packet period."""
    import tone_code
    window = int(2.5*v7.PULSE_FRAME)
    step = v7.PULSE_FRAME//2
    decoded = {}
    with _Capture(), tone_code.coded_pilot_timing():
        state = v7.PulseState(tail_memory=False)
        for end in range(window//2, len(audio)+step, step):
            chunk = audio[max(0, end-window):end]
            if len(chunk) < v7.PULSE_FRAME:
                continue
            results, _ = v7.decode_pulse_stream(
                model, chunk, latest_only=True,
                models={model.encoding_type: model}, state=state,
                sample_rate=RATE, pilot_timing='tone-seeded')
            for result in results:
                key = result.diag.get('source_index')
                if result.status != 'lost' and key is not None:
                    decoded.setdefault(key, result)
    return _score(wire, model, list(decoded.values()), reference)


def run(use_ffmpeg=False, packets=PACKETS, paths=('stream', 'live')):
    wire, _base, model, audio, _values = encode(packets)
    import tone_code
    with _Capture(), tone_code.coded_pilot_timing():
        results, _ = v7.decode_pulse_stream(
            model, audio, models={model.encoding_type: model},
            state=v7.PulseState(tail_memory=False), sample_rate=RATE,
            pilot_timing='tone-seeded')
    reference = wire.values(model, results[-1])
    rows = []
    for label, channel in conditions(use_ffmpeg):
        received = channel(audio)
        row = {'condition': label}
        if 'stream' in paths:
            row['stream'] = decode_stream(wire, model, received, reference)
        if 'live' in paths:
            row['live'] = decode_live(wire, model, received, reference)
        rows.append(row)
    return packets, rows


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--ffmpeg', action='store_true',
                        help='also run ffmpeg time-stretch and pitch-shift cases')
    parser.add_argument('--packets', type=int, default=PACKETS)
    args = parser.parse_args(argv)
    packets, rows = run(args.ffmpeg, args.packets)
    print(f'{"condition":28s} {"stream":>16s} {"live":>16s}   '
          f'(packets decoded/{packets}, mean luma PSNR vs clean decode)')
    for row in rows:
        cells = []
        for path in ('stream', 'live'):
            count, quality = row[path]
            cells.append(f'{count:2d} {quality:6.2f} dB' if count else
                         f'{count:2d}     --   ')
        print(f'{row["condition"]:28s} {cells[0]:>16s} {cells[1]:>16s}')


if __name__ == '__main__':
    main()
