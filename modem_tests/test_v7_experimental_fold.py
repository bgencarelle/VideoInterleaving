"""The luma-fold prototype (test_modem_v7/, tools/v7_live.py --experimental-fold).

The fold is experimental and lives outside animation_modem, but the live
tools can load it, so its safety properties are pinned here:
- noiseless round trip;
- signature detection, and pass-through of normal packets and of packets
  folded with another table;
- the confidence fallback and the noise gate;
- fail-closed table loading (pins, canonical form, model identity);
- the receiver hooks are restored, including when a receive run fails.
"""
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
for extra in (ROOT/'test_modem_v7', ROOT/'tools'):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))

from animation_modem import v7                                         # noqa: E402
import folding                                                          # noqa: E402
from folding import SIGNATURE_STEPS, table_text                          # noqa: E402
import live_fold                                                        # noqa: E402
from live_fold import LiveFold                                          # noqa: E402

TARGET = .1521/np.sqrt(1 + 10**(v7.CLOCK_REL_DB/10))


class ExperimentalFoldTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = v7.load_model(TARGET, 'box')
        cls.fold = LiveFold(500)
        cls.codec = cls.fold.codec(cls.model)
        with Image.open(v7.REFERENCE_FIXTURE) as image:
            cls.values = v7.image_values(v7.prepare_image(image.convert('RGB'), 'box'),
                                         v7.V7_GRIDS, 'box')
        cls.folded = cls.codec.encode(cls.values)

    def received(self, values, noise=None, conf=None):
        """What the equaliser hands over for `values` on a perfect channel,
        optionally with symbol noise on the host slots and lowered confidence."""
        c = self.codec
        coeffs = c.grid.forward(values)[c.kept]
        xhat = coeffs - self.model.mu
        if noise is not None:
            xhat = xhat.copy()
            xhat[c.hosts] += noise*c.sd_host/np.sqrt(c.power)
        confidence = np.ones(len(coeffs)) if conf is None else conf
        return coeffs, xhat, confidence

    # ------------------------------------------------------------ round trip
    def test_noiseless_round_trip(self):
        c = self.codec
        full = c.decode(*self.received(self.folded))
        truth = c.grid.forward(self.values)
        data = slice(0, c.M - c.signature)                       # not signature slots
        host_err = np.abs(full[c.kept[c.hosts[data]]] - truth[c.kept[c.hosts[data]]])
        self.assertTrue(np.all(host_err <= c.D/2*c.sd_host[data] + 1e-9))
        u = truth[c.guests[data]]/c.sd_guest[data]
        inside = np.abs(u) < folding.U_CLIP
        np.testing.assert_allclose(full[c.guests[data]][inside], truth[c.guests[data]][inside],
                                   atol=1e-9)
        self.assertAlmostEqual(c.last_score, 1.0, places=6)
        self.assertLess(c.last_noise, 1e-6)
        # Everything outside the fold is untouched.
        others = np.setdiff1d(np.arange(len(c.kept)), c.hosts)
        np.testing.assert_allclose(full[c.kept[others]], truth[c.kept[others]], atol=1e-9)

    # --------------------------------------------------------- pass-through
    def test_normal_packet_passes_through(self):
        c = self.codec
        coeffs, xhat, conf = self.received(self.values)
        full = c.decode(coeffs, xhat, conf)
        np.testing.assert_array_equal(full, c.plain(coeffs))
        self.assertLess(c.last_score, .5)

    def test_packet_folded_with_another_table_passes_through(self):
        other = LiveFold(1000).codec(self.model)
        coeffs, xhat, conf = self.received(other.encode(self.values))
        full = self.codec.decode(coeffs, xhat, conf)
        np.testing.assert_array_equal(full, self.codec.plain(coeffs))
        self.assertLess(self.codec.last_score, .5)

    # ------------------------------------------------------------- fallbacks
    def test_low_confidence_host_is_read_plain_and_its_guest_dropped(self):
        c = self.codec
        coeffs, xhat, conf = self.received(self.folded)
        weak = c.hosts[:10]
        conf = conf.copy(); conf[weak] = c.conf_min/2
        full = c.decode(coeffs, xhat, conf)
        mu = self.model.mu[weak]
        np.testing.assert_allclose(full[c.kept[weak]],
                                   mu + (coeffs[weak] - mu)*np.sqrt(c.power))
        np.testing.assert_array_equal(full[c.guests[:10]], 0.0)

    def test_noise_gate_and_guest_weighting(self):
        c = self.codec
        rng = np.random.default_rng(3)
        clean = c.decode(*self.received(self.folded))
        # Moderate noise: still unfolded, guests weighted by beta^2/(beta^2+n^2).
        moderate = .1*c.D*rng.standard_normal(c.M)
        full = c.decode(*self.received(self.folded, noise=moderate))
        self.assertLess(c.last_noise, c.noise_max)
        shrink = c.beta**2/(c.beta**2 + (c.last_noise*c.D)**2)
        self.assertLess(shrink, 1.0)
        self.assertGreater(np.corrcoef(full[c.guests[:-c.signature]],
                                       clean[c.guests[:-c.signature]])[0, 1], .5)
        # Heavy noise: the packet is not unfolded at all.
        heavy = .6*c.D*rng.standard_normal(c.M)
        coeffs, xhat, conf = self.received(self.folded, noise=heavy)
        full = c.decode(coeffs, xhat, conf)
        self.assertGreater(c.last_noise, c.noise_max)
        np.testing.assert_array_equal(full[c.guests], 0.0)
        mu = self.model.mu[c.hosts[:-c.signature]]
        np.testing.assert_allclose(full[c.kept[c.hosts[:-c.signature]]],
                                   mu + (coeffs[c.hosts[:-c.signature]] - mu)*np.sqrt(c.power))

    def test_signature_amplitude(self):
        c = self.codec
        coeffs = c.grid.forward(self.folded)[c.kept]
        symbol = ((coeffs - self.model.mu)[c.hosts]/c.sd_host*np.sqrt(c.power))
        np.testing.assert_allclose(symbol[-c.signature:], SIGNATURE_STEPS*c.D*c.pattern,
                                   atol=1e-9)

    # ------------------------------------------------------------ fail closed
    def test_shipped_tables_are_pinned_and_describe_themselves(self):
        for slots in (500, 1000):
            fold = LiveFold(slots)
            with self.subTest(M=slots):
                self.assertEqual(fold.table['M'], slots)
                self.assertEqual(fold.table['encode_filter'], 'box')
                self.assertEqual(fold.table['model_tables_sha256'], v7.MODEL_TABLES_SHA256)
                fold.check(self.model)

    def test_unpinned_or_edited_table_is_refused(self):
        (ROOT/'tmp').mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=ROOT/'tmp') as scratch:
            table = dict(self.fold.table, D=self.fold.table['D']*1.01)
            live_fold.table_path(500, scratch).write_text(table_text(table))
            with self.assertRaisesRegex(ValueError, 'does not match the pinned'):
                LiveFold(500, table_dir=scratch)
            with self.assertRaisesRegex(ValueError, 'no pinned fold table'):
                LiveFold(700, table_dir=scratch)
            # Same content, not canonical bytes: refused as well.
            path = live_fold.table_path(500, scratch)
            path.write_text(json.dumps(self.fold.table))
            pins = {500: live_fold.hashlib.sha256(path.read_bytes()).hexdigest()}
            with self.assertRaisesRegex(ValueError, 'canonical'):
                LiveFold(500, table_dir=scratch, pins=pins)

    def test_other_models_are_refused_and_shown_unfolded(self):
        nearest = v7.load_model(TARGET, 'nearest')
        with self.assertRaisesRegex(ValueError, "'nearest'"):
            self.fold.check(nearest)
        coeffs, xhat, conf = self.received(self.folded)

        class Result:                       # a decoded packet from another model
            diag = {'fold_eq': (xhat, conf)}
        Result.coeffs = coeffs
        np.testing.assert_allclose(self.fold.values(nearest, Result),
                                   v7.values_from(nearest, coeffs))

    # --------------------------------------------------------------- hooks
    def test_receiver_hooks_unfold_a_real_packet_and_are_restored(self):
        originals = (v7._equalize_numba, v7._equalize_numpy, v7.decode_frame)
        audio = v7.encode_pulse_stream(self.model, [self.folded]*4, 1, [0]*4)
        self.fold.install()
        try:
            self.assertIsNot(v7.decode_frame, originals[2])
            results, _ = v7.decode_pulse_stream(self.model, audio)
        finally:
            self.fold.uninstall()
        self.assertEqual((v7._equalize_numba, v7._equalize_numpy, v7.decode_frame), originals)
        self.assertTrue(results)
        c = self.codec
        full = c.grid.forward(self.fold.values(self.model, results[-1]))
        truth = c.grid.forward(self.values)
        data = c.guests[:-c.signature]
        self.assertGreater(c.last_score, .9)
        self.assertGreater(np.corrcoef(full[data], truth[data])[0, 1], .9)

    def test_live_receiver_restores_hooks_when_the_run_fails(self):
        import v7_live
        originals = (v7._equalize_numba, v7._equalize_numpy, v7.decode_frame)
        seen = []

        def failing_run(args, fold):
            seen.append(v7.decode_frame is not originals[2])
            raise RuntimeError('audio device vanished')
        real, v7_live._run_receive = v7_live._run_receive, failing_run
        try:
            args = v7_live.parser().parse_args(
                ['receive', '--device', '0', '--headless', '--experimental-fold', '500'])
            with self.assertRaises(RuntimeError):
                v7_live.run_receive(args)
        finally:
            v7_live._run_receive = real
        self.assertEqual(seen, [True])
        self.assertEqual((v7._equalize_numba, v7._equalize_numpy, v7.decode_frame), originals)


if __name__ == '__main__':
    unittest.main()
