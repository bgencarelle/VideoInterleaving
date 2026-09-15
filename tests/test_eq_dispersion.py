"""Why EQ breaks colour on decode: the channel is longer than the cyclic prefix.

Run:  python tests/test_eq_dispersion.py

Each case filters a known packet, decodes it, and fits one scalar gain per
plane (Y, Cb, Cr). A flat channel gives 1.000/1.000/1.000. A *spread* between
the planes is a colour cast: the three planes live on different carriers, so
anything that biases carriers unevenly tints the picture.

The decisive case is the allpass. It changes the magnitude at NO frequency --
0.00 dB everywhere, in band and out -- and it still destroys the picture. So
the fault is not equalisation in the magnitude sense at all. It is impulse
response LENGTH against CP = 16 samples (0.33 ms at 48 kHz).
"""
import os
import sys

import numpy as np
from scipy.signal import butter, sosfilt, lfilter, firwin

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from animation_modem.core import PRESETS, SourceCoder, N, CP, SYNC_LEN, decode_packet
from animation_modem.transport3 import encode

LAYOUT = PRESETS['wide']
SHAPES = [(40, 48), (10, 12), (10, 12)]          # color-lean


def _plane(rows, cols, seed):
    y, x = np.mgrid[0:rows, 0:cols]
    return (np.sin(x/cols*3 + seed)*np.cos(y/rows*2 + seed))*0.6


def _reference():
    coder = SourceCoder(SHAPES)
    vals = np.concatenate([_plane(*s, i).ravel() for i, s in enumerate(SHAPES)])
    packet = encode(vals, LAYOUT, coder, absolute=1, index=1, count=1, profile=1)
    return coder, vals, packet


def plane_gains(values, truth):
    sizes = [int(np.prod(s)) for s in SHAPES]
    edges = np.cumsum([0] + sizes)
    out = []
    for i in range(len(SHAPES)):
        a, b = edges[i], edges[i+1]
        out.append(float(np.dot(values[a:b], truth[a:b])/np.dot(truth[a:b], truth[a:b])))
    return out


def measure(signal, coder, truth, offset=0):
    span = LAYOUT.symbols*(N + CP)
    body = signal[offset+SYNC_LEN:offset+SYNC_LEN+span]
    body = body.reshape(LAYOUT.symbols, N + CP, 2)[:, CP:, :]
    decoded = decode_packet(None, LAYOUT, coder, body=body)
    if decoded.values is None:
        return None
    gains = plane_gains(decoded.values, truth)
    psnr = 10*np.log10(4/np.mean((decoded.values - truth)**2))
    spread = 20*np.log10(max(gains)/min(gains)) if min(gains) > 0 else float('nan')
    return gains, spread, psnr


def allpass(f0, rate=48000):
    """Unity magnitude at every frequency; only the phase moves."""
    _, a = butter(2, f0, btype='low', fs=rate)
    return lambda x: lfilter(a[::-1], a, x, axis=0)


def shelf(f0, db, rate=48000):
    g = 10**(db/20)
    sos = butter(2, f0, btype='low', fs=rate, output='sos')
    return lambda x: x + (g - 1)*sosfilt(sos, x, axis=0)


def highpass(f0, rate=48000):
    sos = butter(2, f0, btype='high', fs=rate, output='sos')
    return lambda x: sosfilt(sos, x, axis=0)


def short_fir(taps=13, f0=300, rate=48000):
    """Same kind of low-cut, but an impulse response shorter than CP."""
    h = firwin(taps, f0, fs=rate, pass_zero=False)
    return lambda x: lfilter(h, [1], x, axis=0)


def main():
    coder, truth, packet = _reference()

    cases = [
        ('clean',                          None),
        ('flat gain -20 dB',               lambda x: x*0.1),
        ('HF shelf 8 kHz -12 dB',          lambda x: x + (10**(-12/20)-1)*sosfilt(
                                               butter(2, 8000, btype='high', fs=48000,
                                                      output='sos'), x, axis=0)),
        ('low shelf 120 Hz -3 dB',         shelf(120, -3)),
        ('low shelf 120 Hz -12 dB',        shelf(120, -12)),
        ('low shelf 120 Hz -24 dB',        shelf(120, -24)),
        ('highpass 200 Hz',                highpass(200)),
        ('ALLPASS 60 Hz  (0.0 dB)',        allpass(60)),
        ('ALLPASS 120 Hz (0.0 dB)',        allpass(120)),
        ('ALLPASS 300 Hz (0.0 dB)',        allpass(300)),
        ('low cut 300 Hz, 13-tap FIR',     short_fir()),
    ]

    print(f'{"channel":32s} {"Y":>7s} {"Cb":>7s} {"Cr":>7s} {"spread":>8s} {"PSNR":>7s}')
    print('-'*72)
    for name, fn in cases:
        signal = packet if fn is None else fn(packet).astype(np.float32)
        got = measure(signal, coder, truth)
        if got is None:
            print(f'{name:32s}    LOST')
            continue
        (y, cb, cr), spread, psnr = got
        print(f'{name:32s} {y:+7.3f} {cb:+7.3f} {cr:+7.3f} {spread:7.2f}d {psnr:6.1f}d')

    # Steady state, to rule out a start-up transient: five packets back to back,
    # measured on the fifth. Identical numbers mean the damage is ISI, not a
    # settling artefact, so no lead-in or gentler ramp can help.
    print('\nsteady state (5th of 5 packets back to back):')
    for name, fn in (('ALLPASS 120 Hz', allpass(120)), ('highpass 200 Hz', highpass(200))):
        stream = fn(np.tile(packet, (5, 1))).astype(np.float32)
        (y, cb, cr), spread, psnr = measure(stream, coder, truth, offset=4*len(packet))
        print(f'{name:32s} {y:+7.3f} {cb:+7.3f} {cr:+7.3f} {spread:7.2f}d {psnr:6.1f}d')


if __name__ == '__main__':
    main()
