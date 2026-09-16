import sys, numpy as np, wave
w = wave.open(sys.argv[1], 'rb')
x = np.frombuffer(w.readframes(w.getnframes()), '<i2').reshape(-1, 2)[:, 0].astype(float)
S = np.abs(np.fft.rfft(x*np.hanning(len(x))))**2
f = np.fft.rfftfreq(len(x), 1/w.getframerate()); c = np.cumsum(S)/np.sum(S)
print(sys.argv[1], [round(f[np.searchsorted(c, q)]) for q in (.99, .999, .9999)])
