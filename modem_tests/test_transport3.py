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

from animation_modem import transport3 as v3
from animation_modem.imaging import (DEFAULT_PROFILE, fit_shapes, image_values,
                                     plane_shapes)
from utilities.modem_v3_check import main


class V3Base(unittest.TestCase):
    def setUp(self):
        self.layout = v3.WIRE
        self.shapes = fit_shapes(plane_shapes(DEFAULT_PROFILE), self.layout.capacity)
        self.coder = v3.SourceCoder(self.shapes)
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

    def test_rejects_scales_outside_the_configured_range(self):
        """Bounds are on scale, which is what edge timing actually measures.

        A 1.9x packet has scale ~0.53. Asking only for 0.91..1.11 -- the window
        a 44.1 kHz capture of a normal-speed signal would sit in -- must reject
        it, and must do so without any reference to a sample rate.
        """
        audio = np.asarray(self.warp(self.packet(), 1.9)[:, 0])
        self.assertIsNone(v3.measure_speed(audio, min_scale=1/1.1, max_scale=1/.9))


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
        """The pre-DCT encoder only ever attenuated, leaving headroom unused."""
        self.assertAlmostEqual(float(np.max(np.abs(self.packet()))), .95, delta=1e-3)

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
                          flags=v3.pack_folders(3, 7))
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


if __name__ == '__main__':
    unittest.main()


class HeaderTrimTests(V3Base):
    """header_split halves header symbols; header_width must stay conservative."""

    def trim(self, **kw):
        from animation_modem import core as t2
        return v3.Layout(top_bin=54, image_symbols=15, name='trim',
                         progressive=True, **kw)

    def roundtrip(self, layout, speed=1.0, noise=.004, frames=4):
        coder = v3.SourceCoder(self.shapes)
        audio = np.concatenate([
            v3.encode(self.values, layout, coder, n+1, 1, frames)
            for n in range(frames)]).astype(np.float64)
        if speed != 1.0:
            k = np.arange(len(audio))
            audio = np.stack([np.interp(np.arange(int(len(audio)/speed))*speed, k, audio[:, c])
                              for c in range(2)], axis=1)
        audio = (audio+np.random.default_rng(0).normal(0, noise, audio.shape)).astype(np.float32)
        rx = v3.Receiver(layout, coder)
        got = []
        for i in range(0, len(audio), 1024):
            got += rx.feed(audio[i:i+1024])
        got += rx.flush()
        return sum(1 for g in got if g.identity == 'verified_header')

    def test_defaults_match_the_declared_header_width(self):
        base = self.trim()
        self.assertEqual(base.header_width, 20)
        self.assertFalse(base.header_split)
        self.assertEqual(base.header_symbols, 4)

    def test_split_halves_the_header_symbols(self):
        self.assertEqual(self.trim(header_split=True).header_symbols, 2)

    def test_split_raises_the_frame_rate(self):
        self.assertGreater(self.trim(header_split=True).fps, self.trim().fps)

    def test_split_still_decodes(self):
        self.assertEqual(self.roundtrip(self.trim(header_split=True)), 4)

    def test_split_keeps_the_slow_end_of_the_speed_range(self):
        for speed in (.25, .5, .75):
            with self.subTest(speed=speed):
                self.assertEqual(self.roundtrip(self.trim(header_split=True),
                                                speed=speed), 4)

    def test_widening_header_bins_costs_the_fast_end(self):
        """Guards the finding: header_width, not header_split, breaks fast play.

        The header lives on the lowest carriers precisely because those survive
        a speed shift. Widening moves it onto bins that do not.
        """
        wide = self.roundtrip(self.trim(header_width=40, header_split=True),
                              speed=1.35, noise=.02)
        narrow = self.roundtrip(self.trim(header_split=True),
                                speed=1.35, noise=.02)
        self.assertGreater(narrow, wide)


class TrainingTests(V3Base):
    """Orthogonal training: same peak, twice the energy, better estimate."""

    def trim(self, **kw):
        return v3.Layout(top_bin=54, image_symbols=15, name='t',
                         progressive=True, **kw)

    def decode_at(self, layout, noise, speed=1.0, frames=4):
        coder = v3.SourceCoder(self.shapes)
        audio = np.concatenate([
            v3.encode(self.values, layout, coder, n+1, 1, frames)
            for n in range(frames)]).astype(np.float64)
        if speed != 1.0:
            k = np.arange(len(audio))
            audio = np.stack([np.interp(np.arange(int(len(audio)/speed))*speed, k, audio[:, c])
                              for c in range(2)], axis=1)
        audio = (audio+np.random.default_rng(3).normal(0, noise, audio.shape)).astype(np.float32)
        rx = v3.Receiver(layout, coder)
        got = []
        for i in range(0, len(audio), 1024):
            got += rx.feed(audio[i:i+1024])
        got += rx.flush()
        good = [g for g in got if g.values is not None]
        err = (np.mean([np.mean((np.clip(g.values, -1, 1)-self.values)**2) for g in good])
               if good else 1.0)
        return sum(1 for g in got if g.identity == 'verified_header'), err

    def test_both_channels_carry_both_training_symbols(self):
        """The old mono-lane training left one channel silent; orthogonality fixes it."""
        coder = v3.SourceCoder(self.shapes)
        for layout, expect_silent in ((self.trim(), True),
                                      (self.trim(orthogonal_training=True), False)):
            with self.subTest(orthogonal=not expect_silent):
                audio = v3.encode(self.values, layout, coder, 1, 1, 4)
                body = audio[v3.SYNC_LEN:layout.packet].reshape(
                    layout.symbols, v3.SYMBOL, 2)
                quiet = min(float(np.sqrt(np.mean(body[s, :, c]**2)))
                            for s, c in ((0, 1), (1, 0)))
                self.assertEqual(quiet < 1e-9, expect_silent)

    def test_orthogonal_training_does_not_raise_the_peak(self):
        coder = v3.SourceCoder(self.shapes)
        a = v3.encode(self.values, self.trim(), coder, 1, 1, 4)
        b = v3.encode(self.values, self.trim(orthogonal_training=True), coder, 1, 1, 4)
        self.assertAlmostEqual(float(np.max(np.abs(a))), float(np.max(np.abs(b))), delta=1e-3)

    def test_orthogonal_training_improves_noise_tolerance(self):
        plain, plain_err = self.decode_at(self.trim(), .05)
        ortho, ortho_err = self.decode_at(self.trim(orthogonal_training=True), .05)
        self.assertLess(ortho_err, plain_err)

    def test_orthogonal_training_round_trips_cleanly(self):
        got, err = self.decode_at(self.trim(orthogonal_training=True), .004)
        self.assertEqual(got, 4)
        self.assertLess(err, 1e-2)

    def test_orthogonal_training_keeps_the_speed_range(self):
        for speed in (.25, .5, 1.0, 1.35):
            with self.subTest(speed=speed):
                got, _ = self.decode_at(self.trim(orthogonal_training=True), .02, speed)
                self.assertEqual(got, 4)

    def test_the_wire_exposes_it(self):
        self.assertTrue(v3.WIRE.orthogonal_training)


class BandTests(V3Base):
    """A layout must work inside the band it declares, at both edges."""

    def build(self, **kw):
        return v3.Layout(top_bin=54, image_symbols=15, name='b',
                         progressive=True, orthogonal_training=True, **kw)

    def filtered(self, layout, lo=0., hi=None, noise=.005, frames=3):
        from scipy.signal import butter, sosfiltfilt
        coder = v3.SourceCoder(self.shapes)
        audio = np.concatenate([
            v3.encode(self.values, layout, coder, n+1, 1, frames)
            for n in range(frames)]).astype(float)
        if hi and hi < v3.REFERENCE_RATE/2*.97:
            audio = sosfiltfilt(butter(10, hi/(v3.REFERENCE_RATE/2), 'low', output='sos'),
                                audio, axis=0)
        if lo > 20:
            audio = sosfiltfilt(butter(10, lo/(v3.REFERENCE_RATE/2), 'high', output='sos'),
                                audio, axis=0)
        audio = (audio+np.random.default_rng(0).normal(0, noise, audio.shape)
                 ).astype(np.float32)
        rx = v3.Receiver(layout, coder)
        got = []
        for i in range(0, len(audio), 1024):
            got += rx.feed(audio[i:i+1024])
        got += rx.flush()
        return (sum(1 for g in got if g.values is not None),
                sum(1 for g in got if g.identity == 'verified_header'))

    def edges(self, layout):
        # Band limits are a property of a clock, so these are the limits at the
        # reference clock -- the one the impairment filters below run at.
        return layout.band_at(v3.REFERENCE_RATE)

    def test_spreading_takes_power_off_the_lowest_carrier(self):
        """Stacked on carrier 0, the whole picture rides the band's bottom edge."""
        def share(layout):
            coder = v3.SourceCoder(self.shapes)
            sent = np.zeros(layout.capacity)
            sent[v3.coefficient_slots(layout, tuple(coder.shapes))] = \
                coder.forward(self.values)
            block = sent.reshape(layout.image_symbols, len(layout.data_bins), 2, 2)
            power = np.sum(block**2, axis=(0, 2, 3))
            return float(power[0]/power.sum())
        self.assertGreater(share(self.build()), .9)
        self.assertLess(share(self.build(spread_carriers=True)), .1)

    def test_spread_layout_survives_its_own_declared_band(self):
        layout = self.build(spread_carriers=True)
        lo, hi = self.edges(layout)
        pics, verified = self.filtered(layout, lo, hi)
        self.assertEqual(pics, 3)
        self.assertEqual(verified, 3)

    def test_unspread_layout_loses_identity_at_its_own_bottom_edge(self):
        """Guards the finding, so the default is not mistaken for safe.

        The picture decodes, but identity is fragile: the header rides the
        band's bottom edge and a deck that rolls off there strips it. The
        tolerance engine (header_tolerance=2) rescues at most a single
        marginal header on this wire (verified 1 of 3); the finding that the
        unspread default is NOT robust at its own bottom edge still holds.
        """
        layout = self.build()
        lo, hi = self.edges(layout)
        pics, verified = self.filtered(layout, lo, hi)
        self.assertEqual(pics, 3)
        self.assertLess(verified, 3)

    def test_header_rides_the_lowest_data_carriers_even_when_spread(self):
        """The header is decoupled from spread_carriers: it needs the safe end
        of the band whether or not the IMAGE planes are spread."""
        plain, spread = self.build(), self.build(spread_carriers=True)
        for layout in (plain, spread):
            with self.subTest(spread=layout.spread_carriers):
                self.assertTrue(np.array_equal(
                    layout.header_bins,
                    layout.data_bins[:len(layout.header_bins)]))
                self.assertEqual(int(layout.header_bins.min()),
                                 int(layout.data_bins.min()))



    def test_pilots_stay_inside_the_carrier_range(self):
        """Regression: pilots were hardcoded [3,5,21,45] for any progressive
        layout, so top_bin < 45 raised IndexError from encode()."""
        for top_bin in range(10, 64):
            with self.subTest(top_bin=top_bin):
                layout = v3.Layout(top_bin=top_bin, image_symbols=8, name='p',
                                   progressive=True)
                self.assertTrue((layout.pilots >= layout.carriers.min()).all())
                self.assertTrue((layout.pilots <= top_bin).all())

    def test_narrow_progressive_layout_encodes(self):
        layout = v3.Layout(top_bin=27, image_symbols=35, name='narrow',
                           progressive=True, orthogonal_training=True)
        coder = v3.SourceCoder(v3.fit_shapes(self.shapes, layout.capacity)
                               if hasattr(v3, 'fit_shapes') else self.shapes)
        v3.encode(np.zeros(coder.count), layout, coder, 1, 1, 4)


class LeanChromaTests(V3Base):
    """Chroma at half luma resolution costs a third of the budget for ~1% of
    the image energy. The lean profile quarters it."""

    def shapes_for(self, profile):
        from animation_modem.imaging import plane_shapes
        return plane_shapes(profile)

    def test_lean_profile_is_a_quarter_of_the_chroma(self):
        full = self.shapes_for('color-dct')
        lean = self.shapes_for('lean-dct')
        self.assertEqual(full[0], lean[0])                      # luma unchanged
        self.assertEqual(int(np.prod(lean[1]))*4, int(np.prod(full[1])))

    def test_lean_profile_frees_four_symbols(self):
        import numpy as np
        full = sum(int(np.prod(s)) for s in self.shapes_for('color-dct'))
        lean = sum(int(np.prod(s)) for s in self.shapes_for('lean-dct'))
        self.assertEqual(full, 2880)
        self.assertEqual(lean, 2160)

    def test_chroma_planes_keep_the_luma_aspect(self):
        """Off-aspect chroma gets letterboxed by image_values and loses 2-4 dB,
        which is easy to misread as a chroma-resolution result."""
        from animation_modem.imaging import PROFILES
        for name in ('color-dct', 'lean-dct'):
            with self.subTest(profile=name):
                size, chroma = PROFILES[name]
                self.assertAlmostEqual(size[0]/size[1], chroma[0]/chroma[1], places=3)

    def test_lean_profile_round_trips(self):
        from animation_modem.imaging import plane_shapes
        layout = v3.WIRE
        shapes = plane_shapes('lean-dct')
        coder = v3.SourceCoder(shapes)
        self.assertLessEqual(coder.count, layout.capacity)
        values = image_values(self.image, shapes)
        audio = v3.encode(values, layout, coder, 1, 1, 4)
        rx = v3.Receiver(layout, coder)
        got = []
        for i in range(0, len(audio), 256):
            got += rx.feed(audio[i:i+256])
        got += rx.flush()
        self.assertEqual([g.absolute for g in got], [1])
        self.assertEqual(got[0].identity, 'verified_header')
        self.assertLess(np.sqrt(np.mean((got[0].values-values)**2)), 1e-2)
