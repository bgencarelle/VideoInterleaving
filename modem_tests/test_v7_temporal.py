"""Temporal fusion: a held picture is averaged, a moving one is not."""
import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
for extra in (ROOT/'test_modem_v7', ROOT/'tools'):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))

from animation_modem import v7                                           # noqa: E402
from animation_modem.v7_temporal import (MAX_PACKETS, TemporalFusion,    # noqa: E402
                                         fused_packet)
from common import TARGET                                                # noqa: E402
from live_fold import LiveFold                                           # noqa: E402
from tools import v7_live                                                # noqa: E402


def _observe(model, truth, noise_scale, rng, tail_slice=0):
    """An equaliser output for ``truth`` (deviation from the mean) at white
    noise ``noise_scale`` x the model deviation, as an LMMSE estimate."""
    lam = np.asarray(model.lam, float)
    noise = noise_scale**2*lam
    measured = truth+rng.normal(size=len(lam))*np.sqrt(noise)
    conf = lam/(lam+noise)
    seen = np.zeros(len(lam), bool)
    ranks = model.rank_tables[tail_slice % v7.TAIL_PHASES]
    seen[ranks[ranks >= 0]] = True
    return np.where(seen, measured*conf, 0.0), np.where(seen, conf, 0.0), seen


def _test_frame():
    """Bars, a ramp and fine stripes: energy in every band the wire sends."""
    from PIL import Image
    y, x = np.mgrid[0:384, 0:320]
    frame = np.zeros((384, 320, 3), np.uint8)
    frame[..., 0] = 60+140*((x//40) % 2)
    frame[..., 1] = (y*255)//383
    frame[..., 2] = 128+100*np.sin(x/3.0)*np.sin(y/5.0)
    return Image.fromarray(frame, 'RGB')


class TemporalFusionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = v7.load_model(TARGET, 'box')
        cls.lam = np.asarray(cls.model.lam, float)

    def _picture(self, seed):
        return np.random.default_rng(seed).normal(
            size=len(self.lam))*np.sqrt(self.lam)

    def _error(self, xhat, conf, seen, truth):
        value = xhat[seen]/np.maximum(conf[seen], 1e-9)
        return float(np.mean((value-truth[seen])**2/self.lam[seen]))

    def test_first_packet_is_returned_as_decoded(self):
        rng = np.random.default_rng(1)
        xhat, conf, _ = _observe(self.model, self._picture(2), .3, rng)
        fused_xhat, fused_conf, _ = TemporalFusion().update(
            self.model, xhat, conf)
        np.testing.assert_array_equal(fused_xhat, xhat)
        np.testing.assert_array_equal(fused_conf, conf)

    def test_held_picture_error_falls_and_confidence_rises(self):
        rng = np.random.default_rng(3)
        truth = self._picture(4)
        fusion = TemporalFusion()
        single = fused = None
        for packet in range(16):
            xhat, conf, seen = _observe(self.model, truth, .3, rng)
            fused_xhat, fused_conf, known = fusion.update(
                self.model, xhat, conf)
            single = self._error(xhat, conf, seen, truth)
            fused = self._error(fused_xhat, fused_conf, known, truth)
        self.assertLess(fused, .4*single)
        self.assertGreater(fusion.last_gain, 2.0)
        self.assertLessEqual(fusion.last_gain, MAX_PACKETS+1)
        self.assertGreater(float(np.mean(fused_conf[seen])),
                           float(np.mean(conf[seen])))

    def test_new_picture_is_not_mixed_with_the_old_one(self):
        rng = np.random.default_rng(5)
        fusion = TemporalFusion()
        for _ in range(8):
            fusion.update(self.model,
                          *_observe(self.model, self._picture(6), .1, rng)[:2])
        xhat, conf, _ = _observe(self.model, self._picture(7), .1, rng)
        fused_xhat, fused_conf, _ = fusion.update(self.model, xhat, conf)
        np.testing.assert_array_equal(fused_xhat, xhat)
        np.testing.assert_array_equal(fused_conf, conf)
        self.assertEqual(fusion.last_gain, 1.0)

    def test_steady_motion_is_followed(self):
        """A picture that keeps changing a little: the fused error stays at
        or below the single-packet error (no lag worse than the noise)."""
        rng = np.random.default_rng(8)
        truth = self._picture(9)
        step = .1*self._picture(10)
        fusion = TemporalFusion()
        single, fused = [], []
        for packet in range(20):
            truth = truth+step
            xhat, conf, seen = _observe(self.model, truth, .05, rng)
            fused_xhat, fused_conf, known = fusion.update(
                self.model, xhat, conf)
            if packet >= 4:
                single.append(self._error(xhat, conf, seen, truth))
                fused.append(self._error(fused_xhat, fused_conf, known, truth))
        self.assertLess(np.mean(fused), 1.1*np.mean(single))

    def test_key_or_model_change_starts_again(self):
        rng = np.random.default_rng(11)
        fusion = TemporalFusion()
        truth = self._picture(12)
        for _ in range(4):
            fusion.update(self.model, *_observe(self.model, truth, .3, rng)[:2],
                          key='forward')
        xhat, conf, _ = _observe(self.model, truth, .3, rng)
        fused_xhat, _, _ = fusion.update(self.model, xhat, conf, key='reverse')
        np.testing.assert_array_equal(fused_xhat, xhat)

    def test_unseen_coefficients_keep_the_packet_values(self):
        rng = np.random.default_rng(13)
        truth = self._picture(14)
        fusion = TemporalFusion()
        coeffs = None
        for packet in range(9):
            xhat, conf, seen = _observe(self.model, truth, .3, rng, packet)
            decoded = np.asarray(self.model.mu, float)+np.where(
                seen, xhat, 7.0)
            coeffs, _, _ = fused_packet(fusion, self.model, decoded, xhat, conf)
        np.testing.assert_array_equal(coeffs[~seen], decoded[~seen])

    def test_folded_guests_come_back_on_a_held_noisy_picture(self):
        """Through the real fold codec: symbol noise that makes one packet
        drop its guests averages down until they are read again."""
        codec = LiveFold(500).codec(self.model)
        rng = np.random.default_rng(15)
        values = np.clip(rng.normal(size=sum(
            rows*cols for rows, cols in v7.V7_GRIDS))*.3, -1, 1)
        sent = codec.encode_coefficients(values)-np.asarray(self.model.mu)
        fusion = TemporalFusion()
        single_slots = fused_slots = None
        for packet in range(16):
            xhat, conf, seen = _observe(self.model, sent, .22, rng)
            decoded = np.asarray(self.model.mu, float)+xhat
            codec.decode(decoded, xhat, conf, metadata_confirmed=True)
            single_noise = codec.last_noise
            coeffs, fused_xhat, fused_conf = fused_packet(
                fusion, self.model, decoded, xhat, conf)
            codec.decode(coeffs, fused_xhat, fused_conf,
                         metadata_confirmed=True)
            fused_noise = codec.last_noise
        self.assertGreater(single_noise, codec.guest_noise_max)
        self.assertLess(fused_noise, codec.guest_noise_max)

    def _held_stream_error(self, results, shown, values):
        """Mean squared luma error of the last packets, plain and fused."""
        fusion = TemporalFusion()
        plain, fused = [], []
        for result in results:
            if result.status not in ('received', 'verified'):
                fusion.reset()
                continue
            plain.append(float(np.mean(
                (shown(result)-values)[:96*80]**2)))
            result.diag['temporal_fusion'] = fusion
            fused.append(float(np.mean(
                (shown(result)-values)[:96*80]**2)))
            del result.diag['temporal_fusion']
        self.assertGreaterEqual(len(plain), 12)
        self.assertEqual(plain[0], fused[0])
        return np.mean(plain[-4:]), np.mean(fused[-4:])

    def test_real_fold_500_packets_through_noise(self):
        from tools.v7_wire_profile import WireProfile
        profile = WireProfile('default')
        rng = np.random.default_rng(16)
        values = v7_live._picture_values(
            self.model, _test_frame(), 'box', brightness=1.0)[0]
        audio = profile.encode(self.model, [values]*17)
        # The wire sits about 3 dB higher since fixes list item 28; the noise
        # follows it so the picture is as noisy as the test intends.
        audio = audio+rng.normal(0, 10**(-31/20), audio.shape)
        results, _ = profile.decode(self.model, audio)
        plain, fused = self._held_stream_error(
            results, lambda result: profile.values(self.model, result),
            values)
        self.assertLess(fused, .8*plain)

    def test_real_aspect_fold_packets_through_noise(self):
        import tone_code
        from aspect_fold import AspectFoldWire
        wire = AspectFoldWire('auto')
        rng = np.random.default_rng(17)
        values = v7_live._picture_values(
            self.model, _test_frame(), 'box', brightness=1.0)[0]
        model, coeffs = wire.encode_coefficients(self.model, values, 3)
        audio = np.concatenate([tone_code.add_tone_code(
            v7.encode_pulse_frame_coeffs(
                model, coeffs, counter, aspect_code=3,
                source_index=counter-1, pilot_tones=False,
                pulse_profile_code=wire.pulse_profile_code),
            counter, tone_code.encode_status(wire.status_mode))
            for counter in range(1, 18)])
        audio = audio+rng.normal(0, 10**(-34/20), audio.shape)
        profile = v7_live._AdaptiveProfileDecoder(
            v7_live._experimental_fold(500), self.model)
        profile.install()
        try:
            with tone_code.coded_pilot_timing():
                profile.active_mode = profile.dispatch_mode = (
                    profile.aspect_mode)
                results = v7.decode_pulse_stream(
                    self.model, audio.astype(np.float32),
                    state=v7.PulseState(), sample_rate=v7.RATE,
                    pilot_timing='tone-seeded')[0]
        finally:
            profile.uninstall()
        plain, fused = self._held_stream_error(
            results, lambda result: profile.values(self.model, result),
            values)
        self.assertLess(fused, .8*plain)

    def test_real_aspect_mono_packets_through_noise(self):
        from aspect_mono import AspectMonoWire
        wire = AspectMonoWire(self.model)
        rng = np.random.default_rng(18)
        values = v7_live._picture_values(
            self.model, _test_frame(), 'box', brightness=1.0)[0]
        model = wire._packet_model(self.model, 3)
        audio = wire.encode(self.model, [values]*17, aspect_codes=[3]*17)
        audio = audio+rng.normal(0, 10**(-34/20), audio.shape)
        with wire.receiving():
            results = v7.decode_pulse_stream(
                model, audio, sample_rate=v7.RATE,
                pilot_timing='tone-seeded',
                state=v7.PulseState(tail_memory=False))[0]
        plain, fused = self._held_stream_error(
            results, lambda result: wire.values(model, result), values)
        self.assertLess(fused, .9*plain)

    def test_cli_option_is_off_by_default(self):
        parser = v7_live.parser()
        self.assertEqual(parser.parse_args(
            ['receive', '--device', 'null']).temporal_fusion, 'off')
        self.assertEqual(parser.parse_args(
            ['receive', '--device', 'null', '--temporal-fusion',
             'held']).temporal_fusion, 'held')


if __name__ == '__main__':
    unittest.main()
