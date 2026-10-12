"""Receiver-side cancellation of the chain's own distortion (fixes list item 28).

A packet's body is OFDM; the transmission chain (a tape driven hot, a clipping
sound card, an overdriven amplifier) bends it, and the bend spreads every
carrier's energy onto the others.  The receiver knows part of what was sent
exactly (the scattered pilots), so on those carriers whatever arrives beyond
the linear channel is the chain's distortion plus noise.  From that it fits a
small model of the bend, predicts the bend's products from its own estimate
of the packet, subtracts them and decodes again.

The bend model is general, not tape-specific: a sum of odd piecewise-linear
hinges of the signal after one of a few pre-emphasis curves (none, then shelves
like a deck's record equalisation), followed by the chain's own playback
response, which the channel estimate already measures.  The candidate with
the smallest pilot residual wins.

The rebuild uses the carriers the receiver trusts most first (ranked by their
pilots' measured error), widening the trusted set each round: a rebuild made
from badly hit carriers would predict the wrong distortion.

All per-packet arithmetic is compiled; the DFTs are small precomputed
matrices (one OFDM symbol is 128 samples, 65 bins).
"""
import math

import numpy as np
from numba import njit

# Hinge knees in units of the emphasised signal's RMS: the bend is fitted
# as a sum of piecewise-linear odd pieces, which follows saturation (flat
# below a knee, bending above it) far better than a polynomial.
ORDERS = np.array([.5, 1.0, 1.5, 2.0, 2.5, 3.0])
# Pre-emphasis candidates: (time constant in microseconds, shelf in dB);
# (0, 0) is a flat chain.
EMPHASIS = ((0.0, 0.0), (50.0, 6.0), (70.0, 10.0), (120.0, 12.0))
TIERS = 3
# The correction is kept only if it removes at least this share of the
# pilots' residual energy; a clean or codec-coloured packet is left alone.
MIN_GAIN = .15


def matrices(n, bins, rate, emphasis=EMPHASIS):
    """Synthesis (bins -> n real samples, as irfft), analysis (as rfft) and
    the emphasis responses on the bins, built once."""
    k = np.arange(bins)
    t = np.arange(n)
    weight = np.full(bins, 2.0)
    weight[0] = 1.0
    if n % 2 == 0 and bins == n//2+1:
        weight[-1] = 1.0
    synth = (weight[None, :]*np.exp(2j*np.pi*np.outer(t, k)/n))/n
    analysis = np.exp(-2j*np.pi*np.outer(k, t)/n)
    freq = k*rate/n
    resp = []
    for tc, shelf in emphasis:
        if tc <= 0:
            resp.append(np.ones(bins, complex))
            continue
        t1 = tc*1e-6
        t2 = t1/10**(shelf/20)
        w = 2j*np.pi*freq
        resp.append((1+w*t1)/(1+w*t2))
    resp = np.array(resp)
    return (np.ascontiguousarray(synth), np.ascontiguousarray(analysis),
            np.ascontiguousarray(resp), np.ascontiguousarray(1/resp))


@njit(cache=True)
def _solve2(h, z):
    det = h[0, 0]*h[1, 1]-h[0, 1]*h[1, 0]
    if abs(det) < 1e-12:
        return 0j, 0j
    return ((h[1, 1]*z[0]-h[0, 1]*z[1])/det, (-h[1, 0]*z[0]+h[0, 0]*z[1])/det)


@njit(cache=True)
def _solve_small(a, b, m):
    """Least squares normal equations (m x m complex), with a small ridge."""
    out = np.zeros(m, np.complex128)
    aa = a.copy()
    bb = b.copy()
    ridge = 1e-9
    for i in range(m):
        ridge = max(ridge, 1e-9*abs(aa[i, i]))
    for i in range(m):
        aa[i, i] += ridge
    # Gaussian elimination with partial pivoting.
    for i in range(m):
        p = i
        for r in range(i+1, m):
            if abs(aa[r, i]) > abs(aa[p, i]):
                p = r
        if abs(aa[p, i]) < 1e-300:
            return out
        if p != i:
            for c in range(m):
                aa[i, c], aa[p, c] = aa[p, c], aa[i, c]
            bb[i], bb[p] = bb[p], bb[i]
        for r in range(i+1, m):
            f = aa[r, i]/aa[i, i]
            for c in range(i, m):
                aa[r, c] -= f*aa[i, c]
            bb[r] -= f*bb[i]
    for i in range(m-1, -1, -1):
        s = bb[i]
        for c in range(i+1, m):
            s -= aa[i, c]*out[c]
        out[i] = s/aa[i, i]
    return out


@njit(cache=True)
def _features(xest, ch, phase, scale, resp, inv, gain, synth, analysis, orders,
              symbols, lo, hi, n, feats):
    """feats[p, s, b]: hinge p of the emphasised channel ch (the part of the
    signal beyond orders[p] times its RMS, with its sign: piecewise-linear
    pieces of any odd bend), back on the received bins through the measured
    playback response (bins lo..hi-1: the body's carriers)."""
    m = len(orders)
    sign = 1.0 if ch == 0 else -1.0
    xs = np.empty(hi-lo, np.complex128)
    u = np.empty((symbols, n))
    power = 0.0
    for s in range(symbols):
        for b in range(lo, hi):
            xs[b-lo] = (xest[s, b, 0]+sign*xest[s, b, 1])/math.sqrt(2.0)*phase[s, b]*resp[b]
        for t in range(n):
            acc = 0.0
            for b in range(lo, hi):
                v = synth[t, b]*xs[b-lo]
                acc += v.real
            u[s, t] = acc*scale
            power += u[s, t]*u[s, t]
    rms = math.sqrt(power/(symbols*n)) if power > 0 else 1.0
    pw = np.empty((m, n))
    for s in range(symbols):
        for t in range(n):
            x = u[s, t]
            mag = abs(x)
            for p in range(m):
                over = mag-orders[p]*rms
                pw[p, t] = (over if x >= 0 else -over) if over > 0 else 0.0
        for b in range(lo, hi):
            back = np.conj(phase[s, b])/scale*inv[b]*gain[s, b, ch]
            for p in range(m):
                acc = 0j
                for t in range(n):
                    acc += analysis[b, t]*pw[p, t]
                feats[p, s, b] = acc*back


@njit(cache=True)
def cancel_packet(Z, H, pilots, is_pilot, is_data, erased, phase, scale,
                  resp, inv, synth, analysis, orders, tiers, min_gain):
    """Z with the chain's predicted distortion removed, or Z unchanged.
    Returns (Z', residual energy before, after, chosen emphasis index)."""
    symbols, bins = Z.shape[0], Z.shape[1]
    n = synth.shape[0]
    m = len(orders)
    K = resp.shape[0]
    # Measured playback response of each channel (left: M+S, right: M-S).
    gain = np.zeros((symbols, bins, 2), np.complex128)
    for s in range(symbols):
        for b in range(bins):
            gain[s, b, 0] = (H[s, b, 0, 0]+H[s, b, 0, 1])/math.sqrt(2.0)
            gain[s, b, 1] = (H[s, b, 1, 0]-H[s, b, 1, 1])/math.sqrt(2.0)
    # Pilot residual: what arrives beyond the linear channel.
    resid = np.zeros((symbols, bins, 2), np.complex128)
    before = 0.0
    for s in range(symbols):
        if erased[s]:
            continue
        for b in range(bins):
            if not is_pilot[s, b]:
                continue
            for c in range(2):
                r = Z[s, b, c]-(H[s, b, c, 0]*pilots[s, b, 0]+H[s, b, c, 1]*pilots[s, b, 1])
                resid[s, b, c] = r
                before += r.real*r.real+r.imag*r.imag
    if before <= 0.0:
        return Z.copy(), before, before, -1
    # Trust per bin: pilot residual against the channel's gain, interpolated.
    pb = np.zeros(bins)
    pc = np.zeros(bins)
    for s in range(symbols):
        if erased[s]:
            continue
        for b in range(bins):
            if is_pilot[s, b]:
                g = abs(gain[s, b, 0])**2+abs(gain[s, b, 1])**2+1e-30
                pb[b] += (abs(resid[s, b, 0])**2+abs(resid[s, b, 1])**2)/g
                pc[b] += 1.0
    known_b = np.zeros(bins, np.int64)
    known_v = np.zeros(bins)
    nk = 0
    for b in range(bins):
        if pc[b] > 0:
            known_b[nk] = b
            known_v[nk] = pb[b]/pc[b]
            nk += 1
    trust = np.full(bins, np.inf)
    for b in range(bins):
        if nk == 0:
            break
        if b <= known_b[0]:
            trust[b] = known_v[0]
        elif b >= known_b[nk-1]:
            trust[b] = known_v[nk-1]
        else:
            for i in range(nk-1):
                if known_b[i] <= b <= known_b[i+1]:
                    f = (b-known_b[i])/(known_b[i+1]-known_b[i])
                    trust[b] = known_v[i]*(1-f)+known_v[i+1]*f
                    break
    used = np.zeros(bins, np.bool_)
    for s in range(symbols):
        for b in range(bins):
            if is_data[s, b]:
                used[b] = True
    order = np.argsort(trust)
    ranked = np.empty(bins, np.int64)
    count = 0
    for i in range(bins):
        if used[order[i]]:
            ranked[count] = order[i]
            count += 1
    tier_of = np.full(bins, tiers, np.int64)
    for i in range(count):
        tier_of[ranked[i]] = (i*tiers)//max(count, 1)
    lo, hi = bins, 0
    for s in range(symbols):
        for b in range(bins):
            if is_data[s, b] or is_pilot[s, b]:
                lo = min(lo, b)
                hi = max(hi, b+1)
    current = Z.copy()
    feats = np.zeros((m, symbols, bins), np.complex128)
    chosen = -1
    best_after = before
    for rnd in range(tiers):
        # This round's estimate of the packet: known pilots everywhere,
        # data from the trusted tiers only.
        xest = np.zeros((symbols, bins, 2), np.complex128)
        for s in range(symbols):
            for b in range(bins):
                if is_pilot[s, b]:
                    xest[s, b, 0] = pilots[s, b, 0]
                    xest[s, b, 1] = pilots[s, b, 1]
                elif is_data[s, b] and tier_of[b] <= rnd and not erased[s]:
                    xest[s, b, 0], xest[s, b, 1] = _solve2(H[s, b], current[s, b])
        candidates = range(K) if rnd == 0 else range(chosen, chosen+1)
        round_best = np.inf
        round_pred = np.zeros((symbols, bins, 2), np.complex128)
        for k in candidates:
            pred = np.zeros((symbols, bins, 2), np.complex128)
            err = 0.0
            for ch in range(2):
                _features(xest, ch, phase, scale, resp[k], inv[k], gain, synth, analysis,
                          orders, symbols, lo, hi, n, feats)
                A = np.zeros((m, m), np.complex128)
                y = np.zeros(m, np.complex128)
                for s in range(symbols):
                    if erased[s]:
                        continue
                    for b in range(bins):
                        if not is_pilot[s, b]:
                            continue
                        for i in range(m):
                            fi = feats[i, s, b]
                            y[i] += np.conj(fi)*resid[s, b, ch]
                            for j in range(m):
                                A[i, j] += np.conj(fi)*feats[j, s, b]
                coef = _solve_small(A, y, m)
                for s in range(symbols):
                    for b in range(bins):
                        acc = 0j
                        for i in range(m):
                            acc += coef[i]*feats[i, s, b]
                        pred[s, b, ch] = acc
                        if is_pilot[s, b] and not erased[s]:
                            r = resid[s, b, ch]-acc
                            err += r.real*r.real+r.imag*r.imag
            if err < round_best:
                round_best = err
                round_pred[:, :, :] = pred
                if rnd == 0:
                    chosen = k
        best_after = round_best
        for s in range(symbols):
            for b in range(bins):
                for c in range(2):
                    current[s, b, c] = Z[s, b, c]-round_pred[s, b, c]
    if best_after > (1.0-min_gain)*before:
        return Z.copy(), before, best_after, -1
    return current, before, best_after, chosen
