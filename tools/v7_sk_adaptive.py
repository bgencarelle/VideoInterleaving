"""Backward-adaptive nested fold: guests scaled by picture activity that both
ends read from the decoded host stairs, so no side information is sent.

Natural pictures differ widely in how much fine detail they hold.  A fixed
table gives every picture the same guest range; here the guest scale follows
the picture.  The sender measures activity on the host values *as the receiver
will decode them* (stair centroids), so both ends compute the same scale unless
a stair slips, and one slip among hundreds of ring hosts moves it negligibly.
"""
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from numba import njit

from tools import v7_sk_fold as sk, v7_sk_study as study


@njit(cache=True)
def sector_activity(values, sector, member, sectors):
    """Root mean square of ring hosts in each orientation sector."""
    total, count = np.zeros(sectors), np.zeros(sectors)
    for i in range(len(values)):
        if member[i]:
            total[sector[i]] += values[i]*values[i]
            count[sector[i]] += 1.0
    out = np.ones(sectors)
    for s in range(sectors):
        if count[s] > 0:
            # Floor: a sector with no detail (pure one-direction lines) is
            # treated as very quiet, not as zero.
            out[s] = max(math.sqrt(total[s]/count[s]), .05)
    return out


@njit(cache=True)
def fit_line(x, y):
    """Least-squares (intercept, slope)."""
    n = len(x)
    mx, my = x.mean(), y.mean()
    sxx = sxy = 0.0
    for i in range(n):
        sxx += (x[i]-mx)**2
        sxy += (x[i]-mx)*(y[i]-my)
    slope = sxy/sxx if sxx > 0 else 0.0
    return my-slope*mx, slope


class Adaptive(study.Nested):
    """Nested fold whose guest scale follows decoded-host activity."""

    def __init__(self, geometry, stats, noise, training, sectors=3, ring=.5,
                 bands=2, limits=(.25, 4.0), passes=3, **parameters):
        self.sectors, self.ring, self.bands, self.limits = sectors, ring, bands, limits
        self.law = None
        name = parameters.pop('name', None)
        super().__init__(geometry, stats, noise, training, name=name, **parameters)
        g = geometry
        width, height = g.aspect
        rows, cols = g.rows, g.cols

        def sector_of(position):
            u, v = position//cols, position % cols
            angle = np.arctan2(v*height, u*width)                  # 0 = vertical detail
            return np.minimum((angle/(math.pi/2)*sectors).astype(np.int64), sectors-1)

        host = g.host_position[self.active]
        u, v = host//cols, host % cols
        radius = np.sqrt(u*u*width*width+v*v*height*height)
        folded = self.fold.level > 0
        threshold = np.quantile(radius[folded], 1-ring) if folded.any() else np.inf
        self.member = folded & (radius >= threshold)
        self.host_sector = sector_of(host)
        self.guest_sector = sector_of(self.q)
        self.guest_band = np.minimum(np.arange(len(self.q))*bands//max(len(self.q), 1), bands-1)
        # Fit the scale law and the fold tables together: steps depend on
        # guest power, activity depends on steps.  Both ends use the final ones.
        for _ in range(passes):
            xs = [[[] for _ in range(bands)] for _ in range(sectors)]
            ys = [[[] for _ in range(bands)] for _ in range(sectors)]
            for plane in training:
                activity = self.activity(self.sent_hosts(plane))
                raw = study.Nested.guests(self, plane)
                for s in range(sectors):
                    for b in range(bands):
                        pick = (self.guest_sector == s) & (self.guest_band == b)
                        if pick.sum() >= 8:
                            xs[s][b].append(math.log(activity[s]))
                            ys[s][b].append(.5*math.log(np.mean(raw[pick]**2)+1e-12))
            law = np.zeros((sectors, bands, 2))
            for s in range(sectors):
                for b in range(bands):
                    if len(xs[s][b]) >= 4:
                        law[s, b] = fit_line(np.array(xs[s][b]), np.array(ys[s][b]))
            self.law = law
            self.fold.fit(np.stack([self.guests(plane) for plane in training]))

    def sent_hosts(self, plane):
        """Host values exactly as a slip-free receiver decodes them."""
        hosts = self.hosts(plane)[self.active]
        return self.fold.decode(self.fold.encode(hosts, np.zeros(self.fold.guests)))[0]

    def activity(self, decoded_hosts):
        return sector_activity(np.ascontiguousarray(decoded_hosts), self.host_sector,
                               self.member, self.sectors)

    def scale(self, activity):
        if self.law is None:
            return np.ones(len(self.q))
        law = self.law[self.guest_sector, self.guest_band]
        value = np.exp(law[:, 0]+law[:, 1]*np.log(activity[self.guest_sector]))
        return np.clip(value, *self.limits)

    def guests(self, plane):
        raw = study.Nested.guests(self, plane)
        if self.law is None:
            return raw
        return raw/self.scale(self.activity(self.sent_hosts(plane)))

    def decode(self, received):
        estimate = np.zeros(len(self.noise))
        hosts, guests = self.fold.decode(received[self.active])
        estimate[self.active] = hosts
        scale = self.scale(self.activity(hosts))
        return self.plane(estimate, self.q, guests*scale*self.s.sd.ravel()[self.q])



class Leveled(study.Linear):
    """Loudness-normalised wrapper: one frozen table for quiet and loud pictures.

    The sender divides the picture's detail by a factor from a fixed ladder so
    its slot values sit at the level the table was designed for, and writes the
    ladder index into the amplitude of the 16 reserved signature slots.  The
    sender picks the index by trial decode (see ``level``).  The
    receiver reads the index there and multiplies the decoded detail back.
    """
    # Ladder of 15 levels, a factor 1.5 apart (about 1/3 to 90 times the
    # design level).  The level is written as *which* of 15 orthogonal sign
    # patterns the 16 signature slots carry (rows 1..15 of a Hadamard matrix
    # over the profile's own pattern; row 0 is the stock fold's signature).
    # Reading it is a 16-slot correlation, so it survives noise far beyond
    # the point where the picture itself has faded, and no gain error or
    # level imbalance can shift it.
    RATIO, LOWEST, HIGHEST = 1.5, -3, 11

    def __init__(self, base, name=None):
        super().__init__(base.g, base.s, base.noise)
        self.base, self.name = base, name or 'levelled '+base.name
        self.reserved, self.pattern = base.reserved, base.pattern
        self.mean = base.s.mean

    def carried(self):
        return self.base.carried()

    def level(self, plane):
        """Ladder index the sender chooses by trying each one.

        Analysis by synthesis: for every index the sender encodes, adds noise
        at the level that packet will meet (fixed pseudo-random draw), decodes
        as the receiver will and keeps the index with the least error.  For
        ordinary pictures this lands near the loudness rule; for narrow-band
        line patterns, whose energy is all in guests, it picks the level at
        which those guests are neither clipped nor buried.
        """
        rng = np.random.default_rng(20261006)
        draw = rng.standard_normal(len(self.noise))
        best, chosen = np.inf, 0
        for index in range(self.LOWEST, self.HIGHEST+1):
            factor = self.RATIO**index
            symbols = self.base.encode(self.mean+(plane-self.mean)/factor)
            received = symbols+study.loudness(symbols)*self.noise*draw
            decoded = self.mean+(self.base.decode(received)-self.mean)*factor
            error = float(np.sum((decoded-plane)**2))
            if error < best:
                best, chosen = error, index
        return chosen

    def encode(self, plane):
        index = self.level(plane)
        symbols = self.base.encode(self.mean+(plane-self.mean)/self.RATIO**index)
        symbols[self.reserved] *= self.marks(index)
        return symbols

    @staticmethod
    def hadamard():
        h = np.ones((1, 1))
        while len(h) < 16:
            h = np.block([[h, h], [h, -h]])
        return h

    def marks(self, index):
        """Sign pattern over the signature slots that carries a ladder index."""
        return self.hadamard()[index-self.LOWEST+1]

    def correlations(self, received):
        """Signature amplitude seen along each of the 16 patterns."""
        seen = received[self.reserved]*self.pattern/study.SIGNATURE_SYMBOL
        return self.hadamard()@seen/len(seen)

    def read_level(self, received):
        return int(np.argmax(self.correlations(received)[1:]))+self.LOWEST

    def signature_score(self, received):
        """About 1 for a packet of this fold, about 0 for anything else."""
        return float(np.max(self.correlations(received)[1:]))

    def decode(self, received):
        index = self.read_level(received)
        return self.mean+(self.base.decode(received)-self.mean)*self.RATIO**index



class Soft(Leveled):
    """Levelled fold with a soft receiver: one table for every link.

    The receiver measures each packet's noise on the signature slots and
    decodes every slot by posterior mean at that noise
    (``v7_sk_fold.soft_decode_frame``), optionally slot by slot with
    ``confidence`` from the equaliser.  There is no table switch and no mode
    threshold: quality follows the link.  Folded slots carry one guest.
    """

    def __init__(self, base, name=None, training=None):
        super().__init__(base, name or 'soft '+base.name)
        fold = base.fold
        if np.any(fold.level > 1):
            raise ValueError('the soft receiver takes single-guest folds')
        guests = np.concatenate([base.guests(plane) for plane in training])
        self.density = sk.residual_density(np.ascontiguousarray(guests), fold.table)
        self.inverse = sk.expand_table(fold.table)

    def packet_noise(self, received, index):
        """Measured noise of this packet relative to the design profile."""
        expected = study.SIGNATURE_SYMBOL*self.pattern*self.marks(index)
        measured = float(np.sqrt(np.mean((received[self.reserved]-expected)**2)))
        design = float(np.sqrt(np.mean(self.noise[self.reserved]**2)))
        return max(measured/max(design, 1e-12), .5)

    def decode(self, received, confidence=None):
        base, fold = self.base, self.base.fold
        index = self.read_level(received)
        relative = self.packet_noise(received, index)
        sigma = relative*self.noise
        if confidence is not None:
            # An erased or doubtful slot is one with very large noise.
            sigma = sigma/np.maximum(np.asarray(confidence, float), 1e-3)
        hosts, guests = sk.soft_decode_frame(
            np.ascontiguousarray(received[base.active]), fold.level, fold.step, fold.scale,
            fold.alpha, np.ascontiguousarray(sigma[base.active]), self.density, self.inverse)
        estimate = np.zeros(len(self.noise))
        estimate[base.active] = hosts
        slot, _ = fold.guest_order()
        scale = base.scale(base.activity(hosts))
        plane = base.plane(estimate, base.q, guests[slot]*scale*base.s.sd.ravel()[base.q])
        return self.mean+(plane-self.mean)*self.RATIO**index


GRID = [dict(linear=linear, triple=triple, kappa=kappa, guard=guard, guard2=guard,
             kappa2=20, levels=3, mid_gain=1.5, guest_order='variance')
        for linear in (0, 64, 208, 600, 1000) for triple in (0, 300, 700)
        for kappa in (14, 20, 30) for guard in (1.0, 1.5)]


def main(argv=None):
    """Training-only choice of one adaptive table per noise condition."""
    import argparse
    import json
    from pathlib import Path
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('--corpus', default='tmp/sk/corpus')
    parser.add_argument('--noise', default='tmp/sk/wire/NOISE.npz')
    parser.add_argument('--out', default='tmp/sk/study/ADAPTIVE.json')
    parser.add_argument('--profile', default='aspect-fold-500', choices=('aspect-fold-500', 'aspect-mono-500'))
    args = parser.parse_args(argv)
    geometry = study.Geometry(args.profile)
    train_files, test_files = study.corpus(args.corpus)
    small, big = study.Analyzer(study.LATTICE), study.Analyzer(geometry.big)
    train = [small(study.portrait(f)) for f in train_files]
    train_big = [big(study.portrait(f)) for f in train_files]
    stats = study.Statistics(train)
    noise_file = np.load(args.noise)
    report = {'training': [f.name for f in train_files], 'held_out_not_read': [f.name for f in test_files],
              'grid_tables': len(GRID), 'conditions': {}}
    for condition in noise_file.files:
        noise, trace = noise_file[condition], []
        for parameters in GRID:
            try:
                scheme = Adaptive(geometry, stats, noise, train, passes=2, **parameters)
            except ValueError:
                continue
            rows = study.evaluate(geometry, scheme, train, train_big, seeds=(5,))
            trace.append((float(np.mean([r['error']/r['energy'] for r in rows])), parameters))
        trace.sort(key=lambda row: row[0])
        report['conditions'][condition] = {'parameters': trace[0][1], 'training_relative_error': trace[0][0],
                                           'runner_up': trace[1][1]}
        print(condition, trace[0], flush=True)
    Path(args.out).write_text(json.dumps(report, indent=1)+'\n')


if __name__ == '__main__':
    main()
