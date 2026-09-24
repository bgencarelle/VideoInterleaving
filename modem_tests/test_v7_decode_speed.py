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


def channel_joint_reference(Z, iters=2, force_float32=False):
    """Original per-pilot-bin channel fit, retained as a numeric oracle."""
    if force_float32:
        sv, bv, pv = v7.PILOT_SV, v7.PILOT_BV, v7.PILOT_PV32
        dtype, cdtype = np.float32, np.complex64
        basis = v7._BASIS32
        phase = np.complex64(2j*np.pi/v7.N)
        frequency = bv.astype(np.float32)
        threshold, eps, tolerance = (np.float32(1e-9), np.float32(1e-10),
                                     np.float32(1e-4))
    else:
        sv, bv, pv = v7.PILOT_SV, v7.PILOT_BV, v7.PILOT_PV
        dtype, cdtype = np.float64, np.complex128
        basis = v7._BASIS
        phase = 2j*np.pi/v7.N
        frequency = bv
        threshold, eps, tolerance = 1e-9, 1e-10, 1e-4

    H = np.empty((v7.F, 65, 2, 2), dtype=cdtype)
    for c in range(2):
        y = np.asarray([Z[s, b, c] for s, b in v7.PILOT_OBS],
                       dtype=cdtype)
        theta = np.zeros(len(v7.KNOTS), dtype=dtype)
        for _ in range(iters):
            delta = basis @ theta
            rot = np.exp(phase*frequency*delta[sv]).astype(cdtype)
            h = np.empty((len(v7.PILOT_BINS), 2), dtype=cdtype)
            for b in v7.PILOT_BINS:
                mask = v7.PILOT_MASK_BY_BIN[b]
                A = pv[mask]*rot[mask, None]
                gram = A.conj().T @ A
                rhs = A.conj().T @ y[mask]
                h[v7.PILOT_BIN_INDEX[mask][0]] = v7._solve_2x2_vec(
                    gram, rhs)
            pred = rot*np.einsum(
                'ij,ij->i', h[v7.PILOT_BIN_INDEX], pv)
            ok = np.abs(pred) > threshold
            ph = np.angle(y[ok]/pred[ok]).astype(dtype, copy=False)
            w = np.abs(pred[ok]).astype(dtype, copy=False)
            J = (2*np.pi*frequency[ok]/v7.N)[:, None]*basis[sv[ok]]
            JW = J*w[:, None]
            normal = JW.T @ JW + eps*np.eye(JW.shape[1], dtype=dtype)
            step = np.linalg.solve(normal, JW.T @ (ph*w))
            theta += step
            if np.max(np.abs(step)) < tolerance:
                break
        delta = basis @ theta
        for k in range(2):
            v = h[:, k]
            hk = (np.interp(v7.BINS, v7.PILOT_BINS, v.real) +
                  1j*np.interp(v7.BINS, v7.PILOT_BINS, v.imag)).astype(cdtype)
            H[:, v7.BINS, c, k] = hk[None, :]*np.exp(
                phase*v7.BINS[None, :].astype(dtype)*delta[:, None]).astype(cdtype)
    return H


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

    def test_batched_channel_fit_matches_per_bin_reference(self):
        for seed in range(3):
            rng = np.random.default_rng(seed)
            Z64 = (rng.standard_normal((v7.F, 65, 2)) +
                   1j*rng.standard_normal((v7.F, 65, 2)))
            for force_float32 in (False, True):
                Z = Z64.astype(np.complex64) if force_float32 else Z64
                expected = channel_joint_reference(
                    Z, force_float32=force_float32)
                actual = v7.channel_joint(Z, force_float32=force_float32)
                with self.subTest(seed=seed, force_float32=force_float32):
                    if force_float32:
                        np.testing.assert_allclose(
                            actual[:, v7.BINS], expected[:, v7.BINS],
                            rtol=2e-5, atol=2e-6)
                    else:
                        np.testing.assert_allclose(
                            actual[:, v7.BINS], expected[:, v7.BINS],
                            rtol=1e-11, atol=1e-12)

    def test_precomputed_block_priors_match_per_frame_computation(self):
        model = v7.load_model(TARGET, 'nearest')
        self.assertEqual(len(model.block_prior_tables), v7.TAIL_PHASES)
        for phase, idx in enumerate(model.rank_tables):
            tx_var = np.array([np.mean(np.where(r >= 0, model.gain[np.maximum(r, 0)]**2 *
                                                model.lam[np.maximum(r, 0)], 0)) for r in idx])
            valid = v7.BLOCK_GROUP_INDEX >= 0
            expected = np.where(valid, tx_var[np.maximum(v7.BLOCK_GROUP_INDEX, 0)], 0.0)
            np.testing.assert_array_equal(model.block_prior_tables[phase], expected)

    def test_numba_equalizer_matches_numpy_across_rank_phases(self):
        model = v7.load_model(TARGET, 'nearest')
        rng = np.random.default_rng(20260924)
        H = (rng.standard_normal((v7.F, 65, 2, 2)) +
             1j*rng.standard_normal((v7.F, 65, 2, 2)))*.35
        H[:, :, 0, 0] += 1
        H[:, :, 1, 1] += 1
        Z = (rng.standard_normal((v7.F, 65, 2)) +
             1j*rng.standard_normal((v7.F, 65, 2)))
        noise = rng.uniform(.01, .2, size=(v7.F, 2))

        for counter in range(v7.TAIL_PHASES):
            expected = v7._equalize_numpy(model, Z, H, noise, counter)
            actual = v7._equalize_numba(model, Z, H, noise, counter)
            with self.subTest(counter=counter):
                np.testing.assert_allclose(
                    actual[0], expected[0], rtol=1e-9, atol=1e-11)
                np.testing.assert_allclose(
                    actual[1], expected[1], rtol=1e-9, atol=1e-11)
                np.testing.assert_array_equal(actual[2], expected[2])

    def _tone_frame(self, model, seed, tone_amp=1.0, empty_amp=.01,
                    coherent=True, muted=()):
        """Z with bins 1/3 carrying tones as the receiver sees them."""
        rng = np.random.default_rng(seed)
        Z = (rng.standard_normal((v7.F, 65, 2)) +
             1j*rng.standard_normal((v7.F, 65, 2)))
        Z[:, (0, 2), :] *= empty_amp
        wobble = np.cumsum(rng.normal(0, .15, v7.F))
        for b in v7.PILOT_TONE_BINS:
            nominal = 2*np.pi*b*(v7.SYM*np.arange(v7.F)+wobble)/v7.N
            angle = nominal if coherent else rng.uniform(-np.pi, np.pi, v7.F)
            receive = model.scale*model.phase[:, b]*np.conj(v7.EARLY[b])
            for ch in range(2):
                Z[:, b, ch] = tone_amp*np.exp(1j*angle)/receive
        Z[list(muted)] = 0
        return Z

    def test_numba_tone_timing_matches_numpy_in_every_outcome(self):
        model = v7.load_model(TARGET, 'nearest')
        cases = {
            'detected': dict(),
            'narrow_track': dict(empty_amp=.25),
            'incoherent': dict(coherent=False),
            'no_tone': dict(tone_amp=1e-7),
            'muted_symbols': dict(muted=(5, 6, 7, 8)),
        }
        seen = set()
        for name, options in cases.items():
            for seed in range(3):
                Z = self._tone_frame(model, seed, **options)
                expected_track, expected = v7._pilot_tone_timing_numpy(
                    Z, model, 3)
                actual_track, actual = v7.pilot_tone_timing(Z, model, 3)
                seen.add(expected.get('reason', 'detected') +
                         ('/narrow' if expected.get('narrow_track') else ''))
                with self.subTest(case=name, seed=seed):
                    self.assertEqual(expected.keys(), actual.keys())
                    for key, value in expected.items():
                        if isinstance(value, (str, bool)) or value is None:
                            self.assertEqual(actual[key], value)
                        else:
                            np.testing.assert_allclose(
                                actual[key], value, rtol=1e-9, atol=1e-11)
                    self.assertEqual(expected_track is None,
                                     actual_track is None)
                    if expected_track is not None:
                        for key in ('per_symbol', 'knot_fit'):
                            np.testing.assert_allclose(
                                actual_track[key], expected_track[key],
                                rtol=1e-9, atol=1e-11)
        # Every branch of the estimator must have been exercised.
        self.assertTrue({'detected', 'detected/narrow', 'tone_coherence',
                         'tone_level'} <= seen, seen)

    def test_numba_channel_fit_matches_numpy_seeded_and_replaced(self):
        rng = np.random.default_rng(11)
        for seed in range(3):
            Z, _ = self._channels(seed)
            if seed == 2:
                Z[5:9] = 0                  # a digital mute inside the frame
            delta = rng.normal(0, .3, v7.F)
            for options in (dict(), dict(tone_delta=delta),
                            dict(tone_delta=delta, tone_replaced=True)):
                expected = v7._channel_joint_batched_numpy(
                    Z, 2, False, **options)
                actual = v7._channel_joint_batched(Z, 2, False, **options)
                with self.subTest(seed=seed, options=sorted(options)):
                    np.testing.assert_allclose(
                        actual[:, v7.BINS], expected[:, v7.BINS],
                        rtol=1e-11, atol=1e-12)

    def test_numba_channel_residuals_match_numpy(self):
        for seed in range(4):
            Z, H = self._channels(seed)
            if seed == 3:
                Z[2:4] = 0
            with self.subTest(seed=seed):
                self.assertAlmostEqual(
                    v7._channel_pilot_residual(Z, H),
                    v7._channel_pilot_residual_numpy(Z, H), places=12)
                self.assertAlmostEqual(
                    v7._channel_timing_residual(Z, H),
                    v7._channel_timing_residual_numpy(Z, H), places=10)

    def test_zero_phasor_carries_no_phase(self):
        zeros = np.array([complex(0., 0.), complex(0., -0.),
                          complex(-0., 0.), complex(-0., -0.)])
        np.testing.assert_array_equal(v7._phase_or_zero(zeros), 0.0)
        np.testing.assert_allclose(v7._phase_or_zero(np.array([1j, -1+0j])),
                                   [np.pi/2, np.pi])

    def test_compiled_sample_reads_match_numpy(self):
        rng = np.random.default_rng(3)
        for dtype in (np.float32, np.float64):
            for taps in (4, 16):
                samples = rng.standard_normal((4000, 2)).astype(dtype)
                # interior, fractional and both clamped edges
                positions = np.concatenate([
                    rng.uniform(-10, 4010, 500), [0., 3999., 1.5, 2000.25]])
                with self.subTest(dtype=dtype.__name__, taps=taps):
                    np.testing.assert_array_equal(
                        v7_core._sample_at(samples, positions, taps),
                        v7_core._sample_at_numpy(samples, positions, taps))

    def test_compiled_pulse_search_matches_numpy(self):
        from animation_modem import transport3
        model = v7.load_model(TARGET, 'nearest')
        values = np.zeros(model.coder.source_count)
        wire = v7.encode_pulse_stream(model, [values]*3, pilot_tones=True,
                                      eof_marker=True)
        rng = np.random.default_rng(4)
        signals = {
            'wire': v7._mono(wire),
            'slow': v7._mono(v7.speed_pulse_stream(wire, .8)),   # refinement
            'noise': rng.normal(0, .2, 12000),
            'silence': np.zeros(4000),
        }
        for name, signal in signals.items():
            for dtype in (np.float32, np.float64):
                for start in (0, 1111, 4100):
                    segment = signal[start:].astype(dtype)
                    expected = transport3.measure_pulses_numpy(segment, .25, 8)
                    actual = transport3.measure_pulses(segment, .25, 8)
                    with self.subTest(signal=name, dtype=dtype.__name__,
                                      start=start):
                        self.assertEqual(expected is None, actual is None)
                        if expected is not None:
                            np.testing.assert_allclose(actual, expected,
                                                       rtol=0, atol=1e-12)

    def test_mono_mix_is_the_channel_mean(self):
        rng = np.random.default_rng(5)
        for dtype in (np.float32, np.float64):
            stereo = rng.standard_normal((999, 2)).astype(dtype)
            np.testing.assert_array_equal(v7._mono(stereo), stereo.mean(axis=1))
            np.testing.assert_array_equal(v7._mono(stereo[:, :1]), stereo[:, 0])

    def test_compiled_leg_polarity_matches_numpy_correlation(self):
        rng = np.random.default_rng(6)
        for sign in (1, -1):
            common = rng.standard_normal(3920)
            stereo = np.column_stack((common+.3*rng.standard_normal(3920)+.2,
                                      sign*common+.3*rng.standard_normal(3920)))
            left = stereo[:, 0]-stereo[:, 0].mean()
            right = stereo[:, 1]-stereo[:, 1].mean()
            expected = np.dot(left, right)/np.sqrt(np.dot(left, left)*np.dot(right, right))
            powers = v7._leg_correlation_sums(stereo.astype(np.float32))
            self.assertAlmostEqual(
                powers[2]/np.sqrt(powers[0]*powers[1]), expected, places=5)
            self.assertEqual(v7.leg_polarity(stereo.astype(np.float32)), sign)

    def test_sinc_table_is_cached_and_read_only(self):
        first = v7_core._sinc_weight_table(16)
        self.assertIs(first, v7_core._sinc_weight_table(16))
        self.assertFalse(first.flags.writeable)
        np.testing.assert_allclose(first.sum(axis=1), 1.0)


if __name__ == '__main__':
    unittest.main()
