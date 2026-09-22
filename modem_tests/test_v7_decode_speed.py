"""V7 decode optimisations must not change results.

The batched fade_and_noise() is checked against the original per-symbol loop,
precomputed block priors against the per-frame computation they replaced, and
the cached sinc table must stay shared and read-only.
"""
import unittest

import numpy as np

from animation_modem import v7, v7_core

TARGET = .1521/np.sqrt(1 + 10**(v7.CLOCK_REL_DB/10))


def fade_and_noise_reference(Z, H):
    """The original per-symbol, per-channel loop (pre-vectorisation)."""
    freq = v7.BINS*v7.RATE/v7.N/1000
    noise = np.zeros((v7.F, 2))
    for s in range(v7.F):
        pil = v7.PILOT_BY_SYMBOL[s]
        pvals = v7.PILOT_VALUES_BY_SYMBOL[s]
        for ch in range(2):
            pred = np.einsum('bi,bi->b', H[s, pil, ch], pvals)
            obs = Z[s, pil, ch]
            mag = np.abs(pred)
            ok = mag > 0.3*mag.max()
            if ok.sum() < 2:
                continue
            rho = obs[ok]/pred[ok]; fb = np.asarray(pil)[ok]*v7.RATE/v7.N/1000; w = mag[ok]
            if ok.sum() >= 3 and np.ptp(fb) > 3:
                A = np.stack([np.ones(ok.sum()), -fb], 1)*w[:, None]
                normal = A.T @ A + 1e-10*np.eye(2)
                u, v = np.linalg.solve(normal, A.T @ (np.log(np.abs(rho)+1e-12)*w))
                v = max(v, 0.0)
            else:
                u, v = float(np.average(np.log(np.abs(rho)+1e-12), weights=w)), 0.0
            a = float(np.angle(np.sum(rho*w)))
            H[s, v7.BINS, ch, :] *= np.exp(u - v*freq + 1j*a)[:, None]
            pred2 = np.einsum('bi,bi->b', H[s, pil, ch], pvals)
            noise[s, ch] = np.sum(np.abs(obs-pred2)**2)/max(len(pil)-3, 1)
    # Unchanged tail: +-1 symbol smoothing, frame-median and pilot floors.
    sm = np.empty_like(noise)
    sm[0] = (noise[0]+noise[1])/3
    sm[-1] = (noise[-2]+noise[-1])/3
    sm[1:-1] = (noise[:-2]+noise[1:-1]+noise[2:])/3
    for ch in range(2):
        noise[:, ch] = np.maximum(
            np.maximum(sm[:, ch], np.median(sm[:, ch])),
            1e-5*v7.PILOT_AMP**2*np.mean(np.abs(H[:, v7.BINS, ch])**2))
    return H, noise


class V7DecodeSpeedTests(unittest.TestCase):
    def _channels(self, seed):
        rng = np.random.default_rng(seed)
        shape = (v7.F, 65, 2, 2)
        H = (rng.standard_normal(shape) + 1j*rng.standard_normal(shape))*.5
        H[:, :, 0, 0] += 1; H[:, :, 1, 1] += 1
        Z = rng.standard_normal((v7.F, 65, 2)) + 1j*rng.standard_normal((v7.F, 65, 2))
        return Z, H

    def test_batched_fade_and_noise_matches_loop(self):
        for seed in range(6):
            Z, H = self._channels(seed)
            if seed == 5:
                H[3, :, 1, :] = 0          # a silent channel: no usable pilots
            ref_H, ref_noise = fade_and_noise_reference(Z, H.copy())
            new_H, new_noise = v7.fade_and_noise(Z, H.copy())
            with self.subTest(seed=seed):
                np.testing.assert_allclose(new_H, ref_H, rtol=1e-10, atol=1e-12)
                np.testing.assert_allclose(new_noise, ref_noise, rtol=1e-10, atol=1e-12)

    def test_precomputed_block_priors_match_per_frame_computation(self):
        model = v7.load_model(TARGET, 'nearest')
        self.assertEqual(len(model.block_prior_tables), v7.TAIL_PHASES)
        for phase, idx in enumerate(model.rank_tables):
            tx_var = np.array([np.mean(np.where(r >= 0, model.gain[np.maximum(r, 0)]**2 *
                                                model.lam[np.maximum(r, 0)], 0)) for r in idx])
            valid = v7.BLOCK_GROUP_INDEX >= 0
            expected = np.where(valid, tx_var[np.maximum(v7.BLOCK_GROUP_INDEX, 0)], 0.0)
            np.testing.assert_array_equal(model.block_prior_tables[phase], expected)

    def test_sinc_table_is_cached_and_read_only(self):
        first = v7_core._sinc_weight_table(16)
        self.assertIs(first, v7_core._sinc_weight_table(16))
        self.assertFalse(first.flags.writeable)
        np.testing.assert_allclose(first.sum(axis=1), 1.0)


if __name__ == '__main__':
    unittest.main()
