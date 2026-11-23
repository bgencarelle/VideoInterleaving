"""CPU actually burned per call at the live cadence (one call every 82 ms),
every thread of the process included.  Run on the machine that will send or
receive: if a BLAS-backed step burns several times its wall time, its worker
threads are spinning (see AGENTS.md, compiled arithmetic).

    .venv/bin/python tools/v7_cpu_per_call.py
"""
import sys, time, os
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT/'test_modem_v7'))
import numpy as np, nested_fold as NF
from animation_modem import v7_source_dct as S
fold = NF.table('mono', '3:4'); rng = np.random.default_rng(1)
plane = np.zeros(NF.LUMA); plane[fold.host_position] = fold.host_mean+fold.host_sd*rng.laplace(scale=.7, size=fold.slots)
r = {0: (fold.encode(plane, 0, 0, 3), .5*fold.noise)}
luma = fold.decode(r, 0, 3); room = fold.room(0, .5*fold.noise)
R, Rt, C, Ct = fold._transforms
x = luma.reshape(96, 80).copy()
def cpu():  # user+system of the whole process, every thread
    t = os.times(); return t.user+t.system
def measure(name, f, n=40):
    f(); wall = 0.0; c0 = cpu()
    for _ in range(n):
        t = time.perf_counter(); f(); wall += time.perf_counter()-t
        time.sleep(.082)
    print(f'{name:44s} wall {wall/n*1e3:6.2f} ms   cpu burned {(cpu()-c0)/n*1e3:6.2f} ms per call')
measure('idle (sleep only)', lambda: None)
measure('smoothing, 16 passes (BLAS inside numba)', lambda: fold.clean(luma, room))
measure('one transform pair via np.dot (BLAS)', lambda: Rt @ x @ C)
measure('one transform pair via _separable (no BLAS)', lambda: S._separable(Rt, x, C))
measure('stair decode', lambda: fold.decode(r, 0, 3))
