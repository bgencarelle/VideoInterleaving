#!/usr/bin/env python3
# usage: .venv/bin/python tools/v7_reshape_guests.py ORIGINAL.npz OUT.npz
"""Reselect nested guests to fill outward from the hosts evenly (no thin strips
along the first rows/columns). Hosts, details, slots and bands unchanged;
guest spread from a smooth fit of the table's own guest spreads; sectors by
the builder's formula."""
import sys, math, numpy as np
src, out = sys.argv[1], sys.argv[2]
d = dict(np.load(src))
def sector(g, C, w, h, n):
    u, v = g//C, g % C
    return np.minimum((np.arctan2(v*h, u*w)/(math.pi/2)*n).astype(np.int64), n-1)
for key in [k for k in d if k.endswith('/guest_position')]:
    pre = key[:-len('guest_position')]; fam, layout = pre.split('/')[:2]
    w, h = map(int, layout.split(':'))
    R, C = map(int, d[pre+'luma_grid']); n = d[pre+'law'].shape[0]
    gp, gs = d[key], d[pre+'guest_sd']; legs, count = gp.shape
    taken = np.zeros(R*C, bool); taken[d[pre+'host_position']] = True
    if pre+'ride' in d:
        taken[d[pre+'detail_position'][d[pre+'ride'] > 0]] = True
    # smooth model of guest spread against position (fitted on this table)
    g = gp.ravel(); u, v = g//C, g % C
    feats = lambda u, v: np.stack([np.ones(len(u)), np.log1p(u), np.log1p(v), np.log1p(u)*np.log1p(v),
                                   np.log1p(u)**2, np.log1p(v)**2], 1)
    b, *_ = np.linalg.lstsq(feats(u, v), np.log(gs.ravel()), rcond=None)
    free = np.flatnonzero(~taken); fu, fv = free//C, free % C
    rho = np.hypot((fu+.5)/R, (fv+.5)/C)
    order = free[np.argsort(rho, kind='stable')][:legs*count]
    if legs == 2:
        # Two-channel tables: the first channel's guests carry the riding
        # details (nested_fold.DETAIL_GUESTS), so the second channel takes
        # the nearest positions as one solid region; the first channel's
        # positions (used only with detail guests off) are the next ring.
        new = np.stack([order[count:2*count], order[:count]])
    else:
        new = order[None, :count]
    nu, nv = new//C, new % C
    nsd = np.exp(feats(nu.ravel(), nv.ravel())@b).reshape(new.shape)
    moved = np.mean([len(set(new[l]) - set(gp[l]))/count for l in range(legs)])
    d[key] = new.astype(gp.dtype); d[pre+'guest_sd'] = nsd
    d[pre+'guest_sector'] = sector(new, C, w, h, n).astype(d[pre+'guest_sector'].dtype)
    print(pre, 'guests per leg', count, 'moved %.0f%%' % (100*moved))
np.savez_compressed(out, **d)
