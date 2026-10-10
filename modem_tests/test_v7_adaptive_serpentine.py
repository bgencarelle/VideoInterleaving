import unittest
from pathlib import Path
import sys
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from tools.v7_adaptive_serpentine import (
    FACTOR_TO_PILOT_LEVEL,
    HOST_LEVEL,
    PILOT_LEVEL_TO_FACTOR,
    factor_for_pilot_level,
    host_factor_for_ratio,
    install,
    pilot_level_for_factor,
    pilot_pattern,
    select_guest_factor,
)
from tools.v7_sk_fold import (
    adaptive_guest_factor_from_peak,
    normalised_guests,
    serpentine_decode_frame,
    serpentine_encode_frame,
)


class AdaptiveSerpentinePrimitiveTests(unittest.TestCase):
    def _select(self, peak, legs=2):
        flat = np.full(8, peak, dtype=np.float64)
        positions = np.tile(np.array([[0, 1]], dtype=np.int64), (legs, 1))
        sd = np.ones((legs, 2), dtype=np.float64)
        scales = np.ones((legs, 2), dtype=np.float64)
        return select_guest_factor(flat, positions, sd, scales, host_factor_for_ratio(1.5))

    def test_smallest_candidate_and_both_range_endpoints(self):
        self.assertEqual(self._select(0.0), 1.0)
        self.assertEqual(self._select(10.0), 1.0)
        self.assertEqual(self._select(11.0), 2.0)
        self.assertEqual(self._select(21.0), 2.0)
        self.assertEqual(self._select(22.0), 4.0)
        self.assertEqual(self._select(80.0), 8.0)
        self.assertEqual(self._select(128.0), 16.0)
        with self.assertRaises(OverflowError):
            self._select(256.0)

    def test_selection_uses_all_stereo_guest_legs_and_does_not_mutate_tables(self):
        flat = np.array([1.0, 2.0, 80.0, 3.0])
        positions = np.array([[0, 1], [2, 3]], dtype=np.int64)
        sd = np.ones((2, 2))
        scales = np.ones((2, 2))
        sd_before, scales_before = sd.copy(), scales.copy()
        self.assertEqual(select_guest_factor(
            flat, positions, sd, scales, host_factor_for_ratio(1.5)), 8.0)
        np.testing.assert_array_equal(sd, sd_before)
        np.testing.assert_array_equal(scales, scales_before)
        self.assertEqual(HOST_LEVEL, -1)

    def test_factor_one_is_bit_identical_and_guest_scaling_never_changes_hosts(self):
        flat = np.array([2.25, -4.5], dtype=np.float64)
        positions = np.array([1], dtype=np.int64)
        guest_sd = np.array([1.25], dtype=np.float64)
        guest_slot = np.array([0], dtype=np.int64)
        guest_scale = np.array([0.75], dtype=np.float64)
        host_factor = 2/3
        host = np.array([1.25], dtype=np.float64)
        level = np.ones(1, dtype=np.int64)
        step = np.array([0.25], dtype=np.float64)
        scale = np.ones(1, dtype=np.float64)
        offset = np.zeros(1, dtype=np.float64)
        table = np.linspace(0.0, 1.0, 1025)

        base_guest = normalised_guests(flat, positions, guest_sd, guest_slot,
                                       guest_scale, host_factor, 1)
        factor_one_guest = normalised_guests(flat, positions, guest_sd, guest_slot,
                                             guest_scale, host_factor*1.0, 1)
        base = serpentine_encode_frame(host, base_guest, level, step, scale,
                                       table, offset)
        factor_one = serpentine_encode_frame(host, factor_one_guest, level, step,
                                             scale, table, offset)
        np.testing.assert_array_equal(factor_one, base)

        factor_two_guest = normalised_guests(flat, positions, guest_sd, guest_slot,
                                             guest_scale, host_factor*2.0, 1)
        factor_two = serpentine_encode_frame(host, factor_two_guest, level, step,
                                             scale, table, offset)
        inverse = np.linspace(0.0, 16.0, 1025)
        host_one, guest_one = serpentine_decode_frame(
            base, level, step, scale, inverse, offset)
        host_two, guest_two = serpentine_decode_frame(
            factor_two, level, step, scale, inverse, offset)
        np.testing.assert_array_equal(host_one, host_two)
        np.testing.assert_array_equal(host_one, host)
        np.testing.assert_allclose(guest_two*2.0, guest_one, rtol=0.0, atol=1e-12)

    def test_factor_selection_keeps_endpoint_headroom_and_rejects_overflow(self):
        limit = 16.0*(1.0-1e-6)
        self.assertEqual(float(adaptive_guest_factor_from_peak(limit)), 1.0)
        self.assertEqual(float(adaptive_guest_factor_from_peak(limit+1e-8)), 2.0)
        self.assertEqual(float(adaptive_guest_factor_from_peak(16.0*16.01)), 0.0)

    def test_nondefault_host_step_factor_is_preserved_in_install(self):
        with patch('tools.v7_serpentine_codec.install') as serpentine_install:
            install(step_factor=0.25)
        serpentine_install.assert_called_once_with(
            out=None, fixed_level=-1, step_factor=0.25, guest_factor=1.0,
            exact_inverse=False, adaptive_guest_range=True)

    def test_rejects_nonfinite_values_and_bad_geometry(self):
        with self.assertRaises(ValueError):
            select_guest_factor(np.array([np.inf]), np.array([[0]], dtype=np.int64),
                                np.array([[1.0]]), np.array([[1.0]]), 2/3)
        with self.assertRaises(ValueError):
            select_guest_factor(np.array([1.0]), np.array([[0]], dtype=np.int64),
                                np.array([[0.0]]), np.array([[1.0]]), 2/3)
        with self.assertRaises(ValueError):
            select_guest_factor(np.array([1.0]), np.array([[2]], dtype=np.int64),
                                np.array([[1.0]]), np.array([[1.0]]), 2/3)
        with self.assertRaises(OverflowError):
            self._select(256.0)
        self.assertAlmostEqual(host_factor_for_ratio(1.5), 2/3)
        with self.assertRaises(ValueError):
            host_factor_for_ratio(float('nan'))

    def test_explicit_pilot_mapping_and_dither_polarity(self):
        self.assertEqual(PILOT_LEVEL_TO_FACTOR,
                         {-3: 1.0, -2: 2.0, -1: 4.0, 0: 8.0, 1: 16.0})
        for factor, level in FACTOR_TO_PILOT_LEVEL.items():
            self.assertEqual(pilot_level_for_factor(factor), level)
            self.assertEqual(factor_for_pilot_level(level), factor)
        hadamard = np.ones((16, 16), dtype=np.float64)
        for level in PILOT_LEVEL_TO_FACTOR:
            plain = pilot_pattern(level, hadamard, False)
            dithered = pilot_pattern(level, hadamard, True)
            np.testing.assert_array_equal(dithered, -plain)
        # The stock row and unused nested rows are not adaptive guest codes.
        for reserved in (2, 3, 11):
            with self.assertRaises(ValueError):
                factor_for_pilot_level(reserved)
            with self.assertRaises(ValueError):
                pilot_pattern(reserved, hadamard)
        with self.assertRaises(ValueError):
            pilot_level_for_factor(3.0)
        for invalid in (1.5, float('nan'), float('inf')):
            with self.assertRaises(ValueError):
                factor_for_pilot_level(invalid)

    def test_actual_mono_decode_holds_last_good_on_damaged_or_reserved_pilot(self):
        test_modem = Path(__file__).resolve().parents[1]/'test_modem_v7'
        if str(test_modem) not in sys.path:
            sys.path.insert(0, str(test_modem))
        import nested_fold as nf

        codec = nf.NestedMonoCodec.__new__(nf.NestedMonoCodec)
        codec.adaptive_guest_range = True
        codec.signature_index = np.arange(16, dtype=np.int64)
        codec.signature_sd = np.ones(16, dtype=np.float64)
        codec._unit_factor = np.ones(16, dtype=np.float64)
        codec.tail_slice = 3
        codec.last_room = 'previous room'
        codec.last_nested = True
        codec.recognition = nf.Recognition()
        codec._adaptive_last_full = np.array([0.25, -0.5], dtype=np.float64)
        previous = codec._adaptive_last_full.copy()

        for damaged_signature in (np.zeros(16), nf.HADAMARD[15].copy()):
            got = codec.decode(np.zeros(1), damaged_signature, np.ones(16))
            np.testing.assert_array_equal(got, previous)
            self.assertIsNot(got, codec._adaptive_last_full)
            self.assertFalse(codec.last_nested)
            self.assertIsNone(codec.tail_slice)

        codec._adaptive_last_full = None
        self.assertIsNone(codec.decode(np.zeros(1), np.zeros(16), np.ones(16)))

    def _slice_wire_fixture(self):
        test_modem = Path(__file__).resolve().parents[1]/'test_modem_v7'
        if str(test_modem) not in sys.path:
            sys.path.insert(0, str(test_modem))
        import nested_fold as nf

        class DecodeReached(Exception):
            pass

        class FoldStub:
            noise = np.ones(8, dtype=np.float64)

            def __init__(self):
                self.guest_factor = None
                self.readings = None

            def soften(self, readings, phase):
                self.phase = phase
                return {'soft': 'stub'}

            def decode(self, readings, level, phase, averaged=False, soft=None,
                       guest_factor=1.0):
                self.guest_factor = guest_factor
                self.readings = readings
                self.level, self.phase = level, phase
                return np.zeros(4)

        ready = SimpleNamespace(lam=np.ones(8), amplitude=1.0, design=1.0,
                                grids=((2, 2), (1, 1), (1, 1)))
        fold = FoldStub()
        wire_class = nf._slice_wire_class()
        wire = wire_class.__new__(wire_class)
        wire.adaptive_guest_range = True
        wire.last_nested = False
        wire.last_room = None
        wire._adaptive_last_values = np.full(4, 0.375)
        wire.recognition = nf.Recognition()
        wire.choose = lambda halves: halves
        wire.fold = lambda layout: fold
        wire.prepared = lambda layout: ready
        def stop_after_decode(*args, **kwargs):
            raise DecodeReached
        wire.held = SimpleNamespace(update=stop_after_decode)
        halves = []
        for kind, level, score in (('left', -1, 0.9), ('right', -1, 0.85)):
            signature = (score, level, 0.05, 0.1, True, None)
            halves.append(SimpleNamespace(
                kind=kind, layout='4:3', signature=signature, counter=3,
                steady=1.0,
                raw=((np.ones(8), np.ones(8)),)))
        return nf, wire, fold, halves, DecodeReached

    def test_actual_stereo_values_valid_pilots_agree_and_decode_both_legs(self):
        nf, wire, fold, halves, DecodeReached = self._slice_wire_fixture()
        factor_level = pilot_level_for_factor(8.0)
        for half in halves:
            half.signature = (half.signature[0], factor_level, *half.signature[2:])
        with self.assertRaises(DecodeReached):
            wire.values(halves)
        self.assertTrue(wire.last_nested)
        self.assertEqual(fold.guest_factor, 8.0)
        self.assertEqual(fold.level, -1)
        self.assertEqual(set(fold.readings), {0, 1})
        self.assertNotIn('sum', fold.readings)

    def test_actual_stereo_values_partial_leg_and_bad_pilots_hold_last_good(self):
        _, wire, fold, halves, DecodeReached = self._slice_wire_fixture()
        halves[0].signature = (0.9, pilot_level_for_factor(4.0), 0.05, 0.1, True, None)
        halves[1].signature = (0.1, -1, 0.8, 0.2, False, None)
        with self.assertRaises(DecodeReached):
            wire.values(halves)
        self.assertEqual(fold.guest_factor, 4.0)
        self.assertEqual(set(fold.readings), {0})
        self.assertNotIn('sum', fold.readings)

        previous = wire._adaptive_last_values.copy()
        for signatures in (
            ((0.9, 11, 0.05, 0.1, False, None),
             (0.1, 0, 0.8, 0.2, False, None)),
            ((0.1, -1, 0.8, 0.2, False, None),
             (0.2, 0, 0.8, 0.1, False, None)),
        ):
            for half, signature in zip(halves, signatures):
                half.signature = signature
            fold.readings = None
            got = wire.values(halves)
            np.testing.assert_array_equal(got, previous)
            self.assertFalse(wire.last_nested)
            self.assertIsNone(fold.readings)  # no damaged/reserved pilot reached decode


if __name__ == '__main__':
    unittest.main()
