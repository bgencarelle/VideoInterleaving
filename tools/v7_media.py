"""Damage models for the V7 bench: MP3 (LAME via ffmpeg), ATRAC (atracdenc +
ffmpeg decode, at 44.1 kHz like MiniDisc), tape-style companding noise
reduction, and wrappers around the repo's existing tape simulators.

External tools: ffmpeg with libmp3lame, and atracdenc
(https://github.com/dcherednik/atracdenc) found via $ATRACDENC or PATH.
Synthetic diagnostics only; real media captures remain authoritative.

NR models are deliberately simple, documented approximations -- not a
Dolby/dbx implementation. They exist to exercise level-dependent,
time-varying treble gain (Dolby B-like) and wideband 2:1 companding with
pre-emphasis (dbx-like), each played back without matching decode or with
a level mistrack.
"""
import os
import shutil
import subprocess
import sys
import tempfile
import wave
from pathlib import Path

import numpy as np
from scipy.signal import butter, lfilter, resample_poly, sosfilt

sys.path.insert(0, str(Path(__file__).resolve().parent))
import bench_v6_tape_lift as lift                                       # noqa: E402
import test_tape_matrix as tm                                           # noqa: E402

RATE = 48000
ATRAC = os.environ.get('ATRACDENC') or shutil.which('atracdenc') or 'atracdenc'


def _write(path, audio, rate):
    pcm = np.clip(np.rint(np.asarray(audio)*32767), -32768, 32767).astype('<i2')
    with wave.open(str(path), 'wb') as w:
        w.setnchannels(2); w.setsampwidth(2); w.setframerate(rate); w.writeframes(pcm.tobytes())


def _read(path):
    with wave.open(str(path), 'rb') as w:
        rate = w.getframerate()
        raw = np.frombuffer(w.readframes(w.getnframes()), '<i2').reshape(-1, w.getnchannels())
    return raw.astype(float)/32768, rate


def _fit(out, n):
    return np.pad(out, ((0, max(0, n-len(out))), (0, 0)))[:n].astype(np.float32)


def mp3(kbps, vbr=None):
    enc = ['-q:a', str(vbr)] if vbr is not None else ['-b:a', f'{kbps}k']
    def run(audio):
        with tempfile.TemporaryDirectory() as d:
            a, b, c = Path(d)/'a.wav', Path(d)/'b.mp3', Path(d)/'c.wav'
            _write(a, audio, RATE)
            subprocess.run(['ffmpeg', '-v', 'error', '-y', '-i', a, '-c:a', 'libmp3lame',
                            *enc, b], check=True)
            subprocess.run(['ffmpeg', '-v', 'error', '-y', '-i', b, '-ar', str(RATE), '-ac', '2', c], check=True)
            out, _ = _read(c)
        return _fit(out, len(audio))
    return run


def atrac(mode):
    """mode: 'sp' (ATRAC1 292k), 'lp2' (ATRAC3 132k), 'lp4' (ATRAC3 66k joint),
    'plus' (ATRAC3plus as produced by atracdenc, 352.8k)."""
    codec, ext = {'sp': ('atrac1', 'aea'), 'lp2': ('atrac3', 'oma'), 'lp4': ('atrac3_lp4', 'oma'),
                  'plus': ('atrac3plus', 'oma')}[mode]
    def run(audio):
        with tempfile.TemporaryDirectory() as d:
            a, b, c = Path(d)/'a.wav', Path(d)/f'b.{ext}', Path(d)/'c.wav'
            _write(a, resample_poly(audio, 147, 160, axis=0), 44100)       # MiniDisc runs at 44.1 kHz
            subprocess.run([ATRAC, '-e', codec, '-i', a, '-o', b], check=True, capture_output=True)
            if mode == 'sp':
                subprocess.run([ATRAC, '-d', '-i', b, '-o', c], check=True, capture_output=True)
            else:
                subprocess.run(['ffmpeg', '-v', 'error', '-y', '-i', b, c], check=True)
            out, rate = _read(c)
        return _fit(resample_poly(out, 160, 147, axis=0), len(audio))
    return run


def _envelope(x, attack_ms, release_ms):
    a = np.exp(-1/(attack_ms*1e-3*RATE)); r = np.exp(-1/(release_ms*1e-3*RATE))
    out = np.empty_like(x); s = 0.0
    for i, v in enumerate(x):
        s = a*s + (1-a)*v if v > s else r*s + (1-r)*v
        out[i] = s
    return out


def dolby_b_like(audio, decode=False, mistrack_db=0.0, dolby_level_dbfs=-12.0):
    """Sliding treble shelf: up to +10 dB above ~1.5 kHz for quiet treble,
    fading to 0 dB as treble approaches Dolby level. decode=True applies the
    inverse, computed with a detector level offset of mistrack_db."""
    hp = butter(1, 1500, btype='highpass', fs=RATE, output='sos')
    out = np.empty_like(np.asarray(audio, float))
    for ch in range(audio.shape[1]):
        x = np.asarray(audio[:, ch], float)
        hf = sosfilt(hp, x)
        env = np.sqrt(_envelope(hf*hf, 2, 60)) + 1e-9
        lvl = 20*np.log10(env) - dolby_level_dbfs + (mistrack_db if decode else 0)
        boost_db = 10*np.clip(-lvl/30, 0, 1)
        g = 10**(boost_db/20) - 1
        out[:, ch] = x - hf*g/(1+g) if decode else x + hf*g
    return out.astype(np.float32)


def dbx_like(audio, ref_dbfs=-15.0):
    """Encode-only 2:1 compander around ref level, RMS detector, with the
    usual treble pre-emphasis (first-order shelf, ~+10 dB at 10 kHz)."""
    b, a = butter(1, [1000, 12000], btype='bandpass', fs=RATE)
    out = np.empty_like(np.asarray(audio, float))
    for ch in range(audio.shape[1]):
        x = np.asarray(audio[:, ch], float)
        x = x + 2.2*lfilter(b, a, x)                                   # pre-emphasis
        rms = np.sqrt(_envelope(x*x, 5, 80)) + 1e-9
        lvl = 20*np.log10(rms) - ref_dbfs
        out[:, ch] = x*10**(-lvl/2/20)                                  # 2:1 toward ref
    peak = np.abs(out).max()
    return (out*(0.9/peak if peak > 0.9 else 1)).astype(np.float32)


def nr_dolby_enc_only(audio):
    return dolby_b_like(audio)


def nr_dolby_mistrack(db):
    return lambda audio: dolby_b_like(dolby_b_like(audio), decode=True, mistrack_db=db)


def chain(*fns):
    def run(audio):
        for f in fns:
            audio = f(audio)
        return audio
    return run


# ------------------------------------------------------------ repo simulators
def tape_matrix(name):
    """A tools/test_tape_matrix.py case (96 kHz model), applied at 48 kHz."""
    case = next(c for c in tm.CASES if c.name == name)
    def run(audio):
        up = resample_poly(audio, 2, 1, axis=0)
        return resample_poly(tm.impair(up, case), 1, 2, axis=0).astype(np.float32)
    return run


def lift_case(name, seconds):
    """A tools/bench_v6_tape_lift.py case with its seeded absolute timeline."""
    index = [c.name for c in lift.CASES].index(name)
    case = lift.CASES[index]
    events = lift.lift_events(case, seconds+2, seed=100+index)
    return (lambda audio: lift.impair(audio, case, events, 200+index)), events


def mono_sum(audio):
    m = np.asarray(audio).mean(axis=1, keepdims=True)
    return np.repeat(m, 2, axis=1).astype(np.float32)
