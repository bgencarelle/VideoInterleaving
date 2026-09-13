"""V3 receive contract: countable preamble, edge-gated acquisition, coasting.

These check the properties v3 actually claims, not just that it round-trips:
the preamble has two interval widths, speed comes out of edge timing alone,
correlation is a fallback rather than the normal path, and level is normalised
in both directions.
"""
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
from PIL import Image

from animation_modem import transport2 as v2
from animation_modem import transport3 as v3
from animation_modem.imaging import (DEFAULT_PROFILE, fit_shapes, image_values,
                                     plane_shapes)
from utilities.modem_v3_check import main


class V3Base(unittest.TestCase):
    def setUp(self):
        self.layout = v2.PRESETS['wide']
        self.shapes = fit_shapes(plane_shapes(DEFAULT_PROFILE), self.layout.capacity)
        self.coder = v2.SourceCoder(self.shapes)
        yy, xx = np.mgrid[:48, :40]
        self.image = Image.fromarray(
            np.uint8(np.stack([xx*6, yy*5, (xx+yy)*2], axis=-1)))
        self.values = image_values(self.image, self.shapes)

    def packet(self, n=1, count=20):
        return v3.encode(self.values, self.layout, self.coder, n, n, count)

    def decode(self, audio, chunk=256, rx=None):
        rx = rx or v3.Receiver(self.layout, self.coder)
        got = []
        for i in range(0, len(audio), chunk):
            got += rx.feed(audio[i:i+chunk])
        return got + rx.flush()

    def warp(self, audio, speed):
        k = np.arange(len(audio))
        return np.stack([np.interp(np.arange(int(len(audio)/speed))*speed, k, audio[:, c])
                         for c in range(2)], axis=1).astype(np.float32)


class PreambleTests(V3Base):
    def test_preamble_has_exactly_two_interval_widths(self):
        """The whole premise: countable like LTC, not a noise burst."""
        gaps = np.diff(v3.edge_intervals(v3.PREAMBLE))
        self.assertEqual(sorted(set(gaps.tolist())), [v3.SHORT, v3.LONG])

    def test_v2_preamble_is_not_countable(self):
        """Guards the claim that this is a real change, not a relabelling."""
        gaps = np.diff(v3.edge_intervals(v2.SYNC, hysteresis=.08))
        self.assertGreater(len(set(gaps.tolist())), 2)

    def test_preamble_present_on_both_channels(self):
        audio = self.packet()
        for channel in (0, 1):
            self.assertGreater(float(np.max(np.abs(audio[16:272, channel]))), .1)

    def test_long_interval_is_twice_the_short_one(self):
        self.assertEqual(v3.LONG, 2*v3.SHORT)


class SpeedTests(V3Base):
    def test_speed_from_edges_alone(self):
        """No correlation, no template bank: intervals give the rate.

        `scale` is received duration over nominal, matching transport2, so
        playing back at 2x yields 0.5 -- not 2.0.
        """
        for speed in (.5, .75, .9, 1.0, 1.25, 1.6, 2.0):
            with self.subTest(speed=speed):
                got = v3.measure_speed(np.asarray(self.warp(self.packet(), speed)[:, 0]))
                self.assertIsNotNone(got)
                self.assertAlmostEqual(got[1], 1/speed, delta=.01/speed)

    def test_ofdm_body_does_not_poison_the_estimate(self):
        """A whole-buffer histogram is dragged by body zero crossings."""
        audio = np.asarray(self.packet()[:, 0])
        got = v3.measure_speed(audio)
        self.assertIsNotNone(got)
        self.assertAlmostEqual(got[1], 1.0, delta=.01)

    def test_noise_alone_yields_no_speed(self):
        noise = np.random.default_rng(0).normal(0, .2, 4096)
        self.assertIsNone(v3.measure_speed(noise))

    def test_rejects_speeds_outside_the_configured_range(self):
        audio = np.asarray(self.warp(self.packet(), 1.9)[:, 0])
        self.assertIsNone(v3.measure_speed(audio, min_speed=.9, max_speed=1.1))


class ReceiveTests(V3Base):
    def test_single_packet_round_trip(self):
        got = self.decode(self.packet())
        self.assertEqual([r.absolute for r in got], [1])
        self.assertEqual(got[0].identity, 'verified_header')
        self.assertLess(np.sqrt(np.mean((got[0].values-self.values)**2)), 1e-2)

    def test_exact_last_packet_without_guard_or_padding(self):
        got = self.decode(self.packet()[:self.layout.packet])
        self.assertEqual([r.absolute for r in got], [1])

    def test_sequence_keeps_frame_identity(self):
        audio = np.concatenate([self.packet(n) for n in range(1, 9)])
        got = self.decode(audio)
        self.assertEqual([r.absolute for r in got], list(range(1, 9)))

    def test_coasts_once_locked(self):
        """Contiguous frames must not be re-searched packet by packet."""
        rx = v3.Receiver(self.layout, self.coder)
        self.decode(np.concatenate([self.packet(n) for n in range(1, 11)]), rx=rx)
        self.assertGreaterEqual(rx.locked_packets, 8)
        self.assertLessEqual(rx.edge_hits, 2)

    def test_correlation_is_not_the_normal_path(self):
        rx = v3.Receiver(self.layout, self.coder)
        self.decode(np.concatenate([self.packet(n) for n in range(1, 11)]), rx=rx)
        self.assertEqual(rx.correlation_hits, 0)

    def test_survives_off_speed_playback(self):
        for speed in (.8, .95, 1.1):
            with self.subTest(speed=speed):
                audio = self.warp(np.concatenate(
                    [self.packet(n) for n in range(1, 5)]), speed)
                got = self.decode(audio)
                self.assertEqual([r.absolute for r in got], [1, 2, 3, 4])

    def test_recovers_after_a_dropout(self):
        audio = np.concatenate([
            self.packet(1), self.packet(2),
            np.zeros((12000, 2), np.float32),
            self.packet(3), self.packet(4)]).astype(np.float32)
        got = self.decode(audio)
        self.assertEqual([r.absolute for r in got], [1, 2, 3, 4])

    def test_chunk_size_does_not_change_results(self):
        audio = np.concatenate([self.packet(n) for n in range(1, 5)])
        a = [r.absolute for r in self.decode(audio, chunk=256)]
        b = [r.absolute for r in self.decode(audio, chunk=1024)]
        c = [r.absolute for r in self.decode(audio, chunk=4099)]
        self.assertEqual(a, b)
        self.assertEqual(a, c)

    def test_reports_which_route_found_each_packet(self):
        got = self.decode(np.concatenate([self.packet(n) for n in range(1, 5)]))
        for r in got:
            self.assertIn(r.extra['acquisition_path'],
                          ('edge', 'coast', 'correlation'))

    def test_silence_decodes_nothing_and_stays_cheap(self):
        rx = v3.Receiver(self.layout, self.coder)
        got = self.decode(np.zeros((48000, 2), np.float32), rx=rx)
        self.assertEqual(got, [])
        self.assertEqual(rx.correlation_hits, 0)

    def test_rejects_malformed_input(self):
        rx = v3.Receiver(self.layout, self.coder)
        for bad in (np.zeros((10, 3), np.float32), np.zeros(10, np.float32),
                    np.full((10, 2), np.nan, np.float32)):
            with self.subTest(shape=np.shape(bad)):
                with self.assertRaises(ValueError):
                    rx.feed(bad)


class LevelTests(V3Base):
    def test_encode_normalises_up_not_only_down(self):
        """v2 only ever attenuated, leaving headroom permanently unused."""
        self.assertAlmostEqual(float(np.max(np.abs(self.packet()))), .95, delta=1e-3)

    def test_v3_is_hotter_than_v2_for_the_same_image(self):
        a = v2.encode(self.values, self.layout, self.coder, 1, 1, 20)
        b = self.packet()
        self.assertGreater(float(np.max(np.abs(b))), float(np.max(np.abs(a))))

    def test_rejects_wrong_value_count(self):
        with self.assertRaises(ValueError):
            v3.encode(self.values[:-1], self.layout, self.coder, 1, 1, 20)

    def test_rejects_non_finite_values(self):
        bad = self.values.copy()
        bad[0] = np.nan
        with self.assertRaises(ValueError):
            v3.encode(bad, self.layout, self.coder, 1, 1, 20)


class HeaderTests(V3Base):
    def test_folder_pair_survives_the_flags_byte(self):
        audio = v3.encode(self.values, self.layout, self.coder, 5, 5, 20,
                          flags=v2.pack_folders(3, 7))
        got = self.decode(audio)
        self.assertEqual((got[0].face_folder, got[0].float_folder), (3, 7))

    def test_source_index_is_zero_based(self):
        got = self.decode(v3.encode(self.values, self.layout, self.coder, 9, 4, 20))
        self.assertEqual((got[0].index, got[0].source_index), (4, 3))


class CliTests(V3Base):
    def test_write_then_read_round_trip(self):
        with tempfile.TemporaryDirectory() as d:
            wav = Path(d)/'v3.wav'
            main(['write', '--frames', '4', '--out', str(wav)])
            self.assertTrue(wav.exists())
            import contextlib
            import io
            out = io.StringIO()
            with contextlib.redirect_stdout(out):
                main(['read', '--wav', str(wav)])
            records = [json.loads(l) for l in out.getvalue().splitlines() if l.startswith('{')]
            self.assertEqual([r['frame'] for r in records], [1, 2, 3, 4])
            self.assertTrue(all(r['identity'] == 'verified_header' for r in records))


class InteropTests(V3Base):
    def test_v2_receiver_does_not_acquire_v3_audio(self):
        """Different preamble: the versions must not half-decode each other."""
        rx = v2.Receiver(self.layout, self.coder, recovery=False, fast=True)
        got = []
        audio = np.concatenate([self.packet(n) for n in range(1, 4)])
        for i in range(0, len(audio), 256):
            got += rx.feed(audio[i:i+256])
        self.assertEqual([r for r in got if r.identity == 'verified_header'], [])

    def test_v3_receiver_does_not_acquire_v2_audio(self):
        audio = np.concatenate([
            v2.encode(self.values, self.layout, self.coder, n, n, 20)
            for n in range(1, 4)])
        got = self.decode(audio.astype(np.float32))
        self.assertEqual([r for r in got if r.identity == 'verified_header'], [])


if __name__ == '__main__':
    unittest.main()
