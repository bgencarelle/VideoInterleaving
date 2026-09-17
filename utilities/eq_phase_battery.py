#!/usr/bin/env python3
"""EQ + phase battery against a real rendered WAV.

Loads a deterministic modem WAV (render one first, see below), applies each
transformation, decodes with the stock pulse Receiver, and reports
verified/total + tiers -- the same demod path `read --wav` uses. Shared
helpers here (load_wav, minphase_eq) are used by the deck/issue/clip
batteries.

Render the input first (12-16 deterministic frames):

    .venv/bin/python main.py --mode modem --modem-dir images_modem \\
        --modem-pair 1,0 --modem-wav eqtest.wav --modem-frames 16

Run (from the repo root; optional WAV path argument, default eqtest.wav):

    .venv/bin/python utilities/eq_phase_battery.py [wav]
"""
import sys
from pathlib import Path

import numpy as np
from scipy.signal import butter, resample, sosfilt

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from animation_modem import transport3 as V3
from utilities.modem_v3_check import coder_for

RATE = 48000
CENTERS = [31.5, 63, 125, 250, 500, 1000, 2000, 4000, 8000, 16000]
TREBLE = [-12, -8, -4, 0, 2, 2, 0, -4, -10, -16]
TILT12 = [-6, -4, -2, 0, 2, 4, 6, 8, 10, 12]

lay = V3.WIRE
coder, grids = coder_for('color-dct', None, lay)


def load_wav(path):
    import wave
    with wave.open(str(path)) as w:
        assert w.getframerate() == RATE and w.getnchannels() == 2
        raw = np.frombuffer(w.readframes(w.getnframes()), '<i2')
    return (raw.reshape(-1, 2).astype(float)/32768).astype(np.float32)


def minphase_eq(wire, db):
    n = len(wire)
    nfft = 1 << int(np.ceil(np.log2(n + 8192)))
    f = np.fft.rfftfreq(nfft, 1 / RATE)
    db = np.asarray(db, float)
    lm = np.interp(f, CENTERS, db, left=db[0], right=db[-1]) / 20 * np.log(10)
    c = np.fft.irfft(lm, n=nfft)
    m = nfft // 2
    cm = np.zeros(nfft)
    cm[0] = c[0]
    cm[1:m] = 2 * c[1:m]
    cm[m] = c[m]
    H = np.exp(np.fft.rfft(cm))
    out = np.fft.irfft(np.fft.rfft(wire, n=nfft, axis=0) * H[:, None],
                       n=nfft, axis=0)
    return out[:n].astype(np.float32)


def frac_delay(wire, samples):
    n = len(wire)
    k = np.arange(n)
    return np.stack([np.interp(k-samples, k, wire[:, c])
                     for c in range(2)], axis=1).astype(np.float32)


def decode(wire):
    rx = V3.Receiver(lay, coder)
    out = []
    for i in range(0, len(wire), 1024):
        out += rx.feed(np.asarray(wire[i:i+1024], np.float32))
    out += rx.flush()
    got = [r for r in out if r.values is not None]
    ver = sum(1 for r in out if r.identity == 'verified_header')
    tiers = {}
    for r in got:
        tiers[r.tier] = tiers.get(r.tier, 0)+1
    return len(got), ver, tiers


CASES = [
    ('clean', lambda w: w),
    ('lp15000', lambda w: sosfilt(
        butter(10, 15000/(RATE/2), 'low', output='sos'), w, axis=0).astype(np.float32)),
    ('lp10000', lambda w: sosfilt(
        butter(10, 10000/(RATE/2), 'low', output='sos'), w, axis=0).astype(np.float32)),
    ('hp300', lambda w: sosfilt(
        butter(4, 300/(RATE/2), 'high', output='sos'), w, axis=0).astype(np.float32)),
    ('hp375-steep', lambda w: sosfilt(
        butter(10, 375/(RATE/2), 'high', output='sos'), w, axis=0).astype(np.float32)),
    ('treble-16', lambda w: minphase_eq(w, TREBLE)),
    ('tilt+12', lambda w: minphase_eq(w, TILT12)),
    ('notch1k', lambda w: minphase_eq(w, [0, 0, 0, -10, 0, 0, 0, 0, 0, 0])),
    ('noise-45', lambda w: (w+np.random.default_rng(7).normal(
        0, 10**(-45/20), w.shape)).astype(np.float32)),
    ('noise-40', lambda w: (w+np.random.default_rng(7).normal(
        0, 10**(-40/20), w.shape)).astype(np.float32)),
    ('treble+noise45', lambda w: (minphase_eq(w, TREBLE)+np.random.default_rng(7).normal(
        0, 10**(-45/20), w.shape)).astype(np.float32)),
    ('speed1.1', lambda w: resample(w, int(len(w)/1.1), axis=0).astype(np.float32)),
    ('speed0.9', lambda w: resample(w, int(len(w)/0.9), axis=0).astype(np.float32)),
    ('delay+1.7', lambda w: frac_delay(w, 1.7)),
    ('delay+5', lambda w: frac_delay(w, 5.0)),
]


def main(argv=None):
    wav = argv[1] if argv and len(argv) > 1 else 'eqtest.wav'
    base = load_wav(wav)
    n = len(base)//lay.frame
    print(f'{len(base)} samples (~{n} frames), peak '
          f'{np.max(np.abs(base)):.3f}\n')
    for tag, fn in CASES:
        got, ver, tiers = decode(fn(base))
        print(f'{tag:14s} pictures {got:2d}/{n}  verified {ver:2d}/{n}  '
              f'tiers {tiers}')


if __name__ == '__main__':
    main(sys.argv)
