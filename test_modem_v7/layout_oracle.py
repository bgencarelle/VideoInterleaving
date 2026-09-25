#!/usr/bin/env python3
"""Slot-layout ideas measured with an oracle channel (no modem): per-slot
AWGN at a given channel SNR, SoftCast gains, MMSE receive, still picture with
every tier delivered, 2,880 slots. Fast, and good for ranking ideas; confirm
anything promising with the real-modem scripts.

Experiments (--experiments, default: all)
  ringing   today's 80x96 corners (nearest and box) against sampling a finer
            192x160 grid, with and without a smooth taper before the cutoff
  luma      choose the 2,880 coefficients by variance across all planes
            ('radial': more luma, much less colour) and weight luma harder
  fold      2:1 folding on top of 'radial', hosts in luma vs in colour
  bad       how much luma folding costs when the channel is worse than the
            SNR the fold step was chosen for

    python test_modem_v7/layout_oracle.py FRAME... [--experiments ringing luma] [--quick]
"""
import argparse
import json
from pathlib import Path

import numpy as np

from common import (REPO, Grids, box_values, contact_sheet, fit_variances, load_frames,
                    reference, score, v7, v7_values)

SLOTS = 2880
FINE = ((192, 160), (96, 80), (96, 80))


class Space:
    """A sampling grid with fitted statistics and the test frames' coefficients."""

    def __init__(self, grids, sample, train, test):
        self.grid = Grids(grids)
        self.sample = sample
        self.mu, self.lam = fit_variances(self.grid, sample, train)
        self.coeffs = [self.grid.forward(sample(frame)) for frame in test]
        self.train_coeffs = [self.grid.forward(sample(frame)) for frame in train[:2]]


class Scheme:
    def __init__(self, space, select='radial', luma_weight=1.0, taper=None, fold=0,
                 hosts='luma'):
        s = self.s = space
        self.fold, self.hosts = fold, hosts
        plane = s.grid.plane
        weighted = s.lam*np.where(plane == 0, 1.0, 1.0/luma_weight)
        kept = np.zeros(len(weighted), bool)
        if select == 'rect':                                            # today's corners
            kept[s.grid.corner_positions(v7.V7_SHAPES)] = True
        else:
            kept[np.argsort(-weighted, kind='stable')[:SLOTS]] = True
        k = np.flatnonzero(kept)
        self.kept = k[np.argsort(-weighted[k], kind='stable')]         # rank order
        d = np.flatnonzero(~kept & (plane == 0))                        # guests are luma
        self.guests = d[np.argsort(-s.lam[d], kind='stable')]
        self.w = np.ones(len(weighted))
        if taper is not None:                                           # raised-cosine prefilter
            for p in range(len(s.grid.grids)):
                m = plane == p
                cut = s.grid.rad[kept & m].max() if np.any(kept & m) else 1
                t = np.clip((s.grid.rad[m]/cut - taper)/(1.0001 - taper), 0, 1)
                self.w[m] = .5 + .5*np.cos(np.pi*t)
        self.counts = [int(np.sum(plane[self.kept] == p)) for p in range(3)]

    def transmit(self, c, snr_db, D=1.0, seed=7):
        s = self.s
        rng = np.random.default_rng(seed)
        sigma = 10**(-snr_db/20)
        lam = s.lam*self.w**2
        x = (c - s.mu)*self.w
        kept = self.kept
        if self.fold:
            pool = kept[s.grid.plane[kept] == 0] if self.hosts == 'luma' else \
                kept[s.grid.plane[kept] > 0]
            hosts = pool[-self.fold:]
            lin = kept[~np.isin(kept, hosts)]
        else:
            hosts, lin = kept[:0], kept
        gain = lam[kept]**-.25
        gain /= np.sqrt(np.mean(gain**2*lam[kept]))                    # unit mean slot power
        gain = dict(zip(kept, gain))
        est = s.mu.copy()
        g = np.array([gain[i] for i in lin])
        y = g*x[lin] + sigma*rng.standard_normal(len(lin))
        est[lin] += g*lam[lin]/(g*g*lam[lin] + sigma**2)*y
        if self.fold:
            guests = self.guests[:len(hosts)]
            budget = np.array([gain[i] for i in hosts])**2*lam[hosts]
            h = x[hosts]/np.sqrt(lam[hosts]); u = (c - s.mu)[guests]/np.sqrt(s.lam[guests])
            U = 2.5; beta = .8*D/(2*U)
            sym = D*np.round(h/D) + beta*np.clip(u, -U, U)
            scale = np.sqrt(budget/(1 + D*D/12 + beta*beta))
            r = sym + sigma/scale*rng.standard_normal(len(hosts))
            k = np.round(r/D)
            est[hosts] += np.sqrt(lam[hosts])*D*k
            noise = (sigma/scale/beta)**2
            est[guests] += np.sqrt(s.lam[guests])*(r - D*k)/beta/(1 + noise)
        return est

    def picture(self, coeffs):
        return self.s.grid.picture(coeffs)

    def best_step(self, snr_db, train_refs):
        def psnr(D):
            return np.mean([score(r, self.picture(self.transmit(c, snr_db, D, seed=3)))['psnr_y']
                            for r, c in zip(train_refs, self.s.train_coeffs)])
        return max(np.geomspace(.1, 8, 16), key=psnr)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('frames', nargs='+', type=Path)
    ap.add_argument('--experiments', nargs='+', default=['ringing', 'luma', 'fold', 'bad'],
                    choices=['ringing', 'luma', 'fold', 'bad'])
    ap.add_argument('--snr', type=float, nargs='+', default=[50, 30, 25])
    ap.add_argument('--quick', action='store_true', help='smoke run: one scored frame, 50 dB')
    ap.add_argument('--out', type=Path, default=REPO/'tmp'/'test_modem_v7'/'layout_oracle')
    args = ap.parse_args(argv)
    train, test = load_frames(args.frames)
    if args.quick:
        test, args.snr = test[:1], [50]
    args.out.mkdir(parents=True, exist_ok=True)
    refs = [reference(frame) for frame in test]
    train_refs = [reference(frame) for frame in train[:2]]
    box = Space(v7.V7_GRIDS, lambda im: box_values(im, v7.V7_GRIDS), train, test)
    fine = Space(FINE, lambda im: box_values(im, FINE), train, test)
    rows, sheet = [], [('source', refs[0])]

    def run(label, scheme, snrs=None, step_db=None):
        for snr in snrs or args.snr:
            D = scheme.best_step(step_db or snr, train_refs) if scheme.fold else 1.0
            scores = [score(ref, scheme.picture(scheme.transmit(c, snr, D)))
                      for ref, c in zip(refs, scheme.s.coeffs)]
            row = {'snr_db': snr, 'scheme': label, 'kept_Y_Cb_Cr': scheme.counts,
                   **{k: round(float(np.mean([s[k] for s in scores])), 2) for k in scores[0]}}
            if step_db:
                row['step_chosen_for_db'] = step_db
            rows.append(row)
            print(json.dumps(row), flush=True)
            if snr == (snrs or args.snr)[0] and len(sheet) < 9:
                sheet.append((f'{label} @{snr:g}dB',
                              scheme.picture(scheme.transmit(scheme.s.coeffs[0], snr, D))))

    if 'ringing' in args.experiments:
        nearest = Space(v7.V7_GRIDS, lambda im: v7_values(im, 'nearest'), train, test)
        run('today: nearest 80x96 corners', Scheme(nearest, 'rect'))
        run('box 80x96 corners', Scheme(box, 'rect'))
        run('fine 192x160 corners (brick wall)', Scheme(fine, 'rect'))
        for a in (.85, .7, .5):
            run(f'fine corners, taper from {a:g} of cutoff', Scheme(fine, 'rect', taper=a))
        for a in (.85, .7):
            run(f'fine radial, taper from {a:g} of cutoff', Scheme(fine, 'radial', taper=a))
    if 'luma' in args.experiments:
        run('box 80x96 radial (luma first)', Scheme(box, 'radial'))
        for weight in (1, 2, 4, 8, 16):
            run(f'fine radial, luma x{weight}', Scheme(fine, 'radial', luma_weight=weight))
    if 'fold' in args.experiments:
        for weight in (1, 4):
            for hosts, folds in (('luma', (500, 1000)), ('chroma', (500,))):
                for M in folds:
                    run(f'fine radial x{weight}, fold {M} in {hosts}',
                        Scheme(fine, 'radial', luma_weight=weight, fold=M, hosts=hosts))
    if 'bad' in args.experiments:
        base = Scheme(fine, 'radial')
        low = [25, 20, 15, 10] if not args.quick else [20]
        run('fine radial, no fold', base, snrs=low)
        for M in (500, 1000):
            for design in (30, 25):
                run(f'fine radial, luma fold {M}', Scheme(fine, 'radial', fold=M),
                    snrs=low, step_db=design)
    contact_sheet(args.out/'layouts.png', sheet)
    (args.out/'results.json').write_text(json.dumps(rows, indent=1) + '\n')


if __name__ == '__main__':
    main()
