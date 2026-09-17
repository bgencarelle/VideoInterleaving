#!/usr/bin/env python3
"""Decode CPU breakdown: where does the ~1ms/packet go?

Times each stage separately on the real path: acquisition, sampling, demod
rfft, equalizer solve, pilot tracking, header, gather+inverse(DCT),
values_image. Reports mean ms/packet + share. Re-run on any machine whose
CPU table you want to compare (numbers do not transfer across hardware;
verify counts should).

Run from the repo root:

    .venv/bin/python utilities/cpu_breakdown.py
"""
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from animation_modem import core as C
from animation_modem import transport3 as V3
from animation_modem.imaging import image_values, values_image
from modem_bake import ModemLibrary
from utilities.modem_v3_check import coder_for

REPO = Path(__file__).resolve().parent.parent
lay = V3.WIRE
N = 30


def bench(tag, fn, T):
    fn()
    t0 = time.perf_counter()
    for _ in range(N):
        fn()
    T[tag] = (time.perf_counter()-t0)*1000/N


def main(argv=None):
    from scipy.fft import idctn
    coder, grids = coder_for('color-dct', None, lay)
    lib = ModemLibrary(REPO/'images_modem')
    vals = image_values(lib.composite(0, 1, 0), grids)
    packet = np.asarray(V3.encode(vals, lay, coder, 1, 1, 1), np.float32)
    body = packet[V3.SYNC_LEN:V3.SYNC_LEN+lay.symbols*V3.SYMBOL].reshape(
        lay.symbols, V3.SYMBOL, 2)[:, V3.CP-4:V3.CP-4+V3.N].astype(np.float32)
    T = {}
    pre = packet[:len(V3.PREAMBLE)+64][:, 0]
    bench('acquire(measure_pulses)', lambda: V3.measure_pulses(pre), T)
    bench('resample(_sample_at)', lambda: V3._sample_at(
        packet, V3._body_walk(lay).astype(float)), T)
    spec = np.fft.rfft(body, n=V3.N, axis=1)
    bench('demod-rfft(20x128)', lambda: np.fft.rfft(body, n=V3.N, axis=1), T)
    bench('equalizer-solve', lambda: C._channel_equalizer(spec, lay), T)
    bench('decode_packet(full)', lambda: C.decode_packet(
        None, lay, coder, body=body), T)
    r0 = C.decode_packet(None, lay, coder, body=body)
    bench('gather+inverse', lambda: coder.inverse(
        np.zeros(coder.count)), T)
    bench('idctn-96x80', lambda: idctn(np.random.default_rng(0).normal(
        size=(96, 80)), norm='ortho'), T)
    img = values_image(r0.values, grids)
    bench('values_image', lambda: values_image(r0.values, grids), T)
    tot = T['decode_packet(full)']
    print(f'{"stage":22s} {"ms/pkt":>8s} {"share":>6s}')
    for k, v in sorted(T.items(), key=lambda kv: -kv[1]):
        share = f'{100*v/tot:.0f}%' if k != 'decode_packet(full)' else 'TOTAL'
        print(f'{k:22s} {v:8.3f} {share:>6s}')


if __name__ == '__main__':
    main(sys.argv)
