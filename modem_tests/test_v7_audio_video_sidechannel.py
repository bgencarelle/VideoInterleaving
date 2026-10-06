"""Checks for audio-carried V7 picture-detail payloads."""
import unittest

import numpy as np
from PIL import Image

from animation_modem import v7
from tools.v7_audio_video_sidechannel import (
    DETAIL_GRID, active_audio_starts, apply_detail_payload,
    decode_phase_qim, encode_phase_qim, make_detail_payload,
    plan_payload_schedule, pulse_sweep_reference, pulse_sweep_spec,
    unpack_detail_payload)


class V7AudioVideoSidechannelTests(unittest.TestCase):
    def test_residual_payload_round_trips_and_crc_rejects_damage(self):
        rng = np.random.default_rng(17)
        source = rng.integers(0, 256, (480, 640), dtype=np.uint8)
        base = Image.fromarray(rng.integers(
            0, 256, (96, 80, 3), dtype=np.uint8), mode='RGB')
        payload = make_detail_payload(source, base)
        residual = unpack_detail_payload(payload)

        self.assertEqual(len(payload), (DETAIL_GRID[0]*DETAIL_GRID[1])//8+2)
        self.assertEqual(residual.shape, DETAIL_GRID)
        enhanced = apply_detail_payload(base, residual, (640, 480))
        self.assertEqual(enhanced.size, (640, 480))
        corrupted = bytearray(payload)
        corrupted[0] ^= 1
        self.assertIsNone(unpack_detail_payload(corrupted))

    def test_phase_qim_transmits_supplied_bits_without_bin_side_metadata(self):
        sample_count = 12*v7.PULSE_FRAME
        audio = pulse_sweep_reference(sample_count)
        block_samples, _, starts, _ = pulse_sweep_spec(sample_count)
        bits = np.random.default_rng(23).integers(0, 2, 1201, dtype=np.uint8)
        encoded = encode_phase_qim(
            audio, starts, block_samples, bins_per_block=16,
            bits_per_component=4, coset_periods=64, payload_bits=bits)
        decoded = decode_phase_qim(
            encoded['audio'], starts, block_samples, bins_per_block=16,
            bits_per_component=4, coset_periods=64)

        np.testing.assert_array_equal(decoded[:len(bits)], bits)
        for start in starts:
            original = np.fft.rfft(audio[start:start+block_samples])
            modified = np.fft.rfft(
                encoded['audio'][start:start+block_samples])
            np.testing.assert_allclose(
                np.abs(modified), np.abs(original), rtol=1e-5, atol=1e-5)

    def test_activity_gate_and_payload_schedule_are_receiver_reproducible(self):
        audio = np.zeros(12*v7.PULSE_FRAME, dtype=np.float32)
        audio[0:960] = .1
        audio[1200:2160] = .1
        audio[3600:4560] = .1
        active = active_audio_starts(audio, rms_threshold=.01)
        np.testing.assert_array_equal(active, [0, 1200, 3600])

        schedule = plan_payload_schedule(
            np.array([0, 2, 4, 6, 8, 10]), frame_count=3,
            bits_per_frame=10, packet_samples=4, block_capacity=8,
            block_samples=1, sample_rate=1)
        np.testing.assert_array_equal(schedule['starts'], [0, 2, 4, 6, 8, 10])
        np.testing.assert_array_equal(schedule['frame_ready_seconds'], [3, 7, 11])

    def test_reference_sweep_has_complete_pulses_only(self):
        samples = 12*v7.PULSE_FRAME
        pulse_samples, period_samples, starts, frequencies = pulse_sweep_spec(
            samples)
        sweep = pulse_sweep_reference(samples)
        self.assertEqual(len(starts), 39)
        self.assertEqual(len(frequencies), 39)
        self.assertLessEqual(starts[-1]+pulse_samples, samples)
        self.assertGreater(starts[-1]+period_samples+pulse_samples, samples)
        self.assertEqual(sweep.shape, (samples,))
        np.testing.assert_array_equal(sweep, pulse_sweep_reference(samples))


if __name__ == '__main__':
    unittest.main()
