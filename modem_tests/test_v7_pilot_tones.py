"""Evaluation-only low-bin V7 timing references."""
from pathlib import Path
import unittest
from unittest import mock

import numpy as np
from PIL import Image
from scipy.signal import butter, resample_poly, sosfilt

from animation_modem import v7
from tools import v7_live


FIXTURE = Path(__file__).parent / 'fixtures/v7_reference_face.png'
TARGET = .1521/np.sqrt(1 + 10**(v7.CLOCK_REL_DB/10))


class V7PilotToneTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = v7.build_model(FIXTURE, TARGET, encode_filter='nearest')
        with Image.open(FIXTURE) as image:
            cls.values = v7.image_values(
                v7.prepare_image(image, 'nearest'),
                cls.model.coder.grids, 'nearest')

    def test_encoder_is_off_by_default_and_tones_are_m_only(self):
        plain = v7.encode_pulse_stream(self.model, [self.values]*4)
        explicit_off = v7.encode_pulse_stream(
            self.model, [self.values]*4, pilot_tones=False)
        np.testing.assert_array_equal(plain, explicit_off)

        toned = v7.encode_pulse_stream(
            self.model, [self.values]*4, pilot_tones=True)
        delta = toned-plain
        np.testing.assert_allclose(delta[:, 0], delta[:, 1], atol=1e-7)
        self.assertGreater(float(np.sqrt(np.mean(delta**2))), 0)

        # The intended tones are exact-bin references and should not leak into
        # the existing 4-34 data carriers at the useful body FFT windows.
        packet = delta[:v7.PULSE_FRAME, 0]
        leakage = []
        for symbol in range(v7.F):
            start = (v7.PULSE.SYNC_LEN + symbol*v7.SYM + v7.WIN)
            spectrum = np.fft.rfft(packet[start:start+v7.N])
            data_energy = np.sum(np.abs(spectrum[v7.BINS])**2)
            tone_energy = np.sum(np.abs(spectrum[list(v7.PILOT_TONE_BINS)])**2)
            leakage.append(data_energy/max(tone_energy, 1e-30))
        self.assertLess(max(leakage), 1e-12)

    def test_phase_offsets_and_packet_continuity_follow_known_template(self):
        plain = v7.encode_pulse_stream(self.model, [self.values]*4)
        with_tones = v7.encode_pulse_stream(
            self.model, [self.values]*4, pilot_tones=True)
        split_batches = np.concatenate((
            v7.encode_pulse_stream(self.model, [self.values]*2,
                                   start_counter=1, pilot_tones=True),
            v7.encode_pulse_stream(self.model, [self.values]*2,
                                   start_counter=3, pilot_tones=True)))
        np.testing.assert_array_equal(split_batches, with_tones)
        for counter in range(1, 5):
            start = (counter-1)*v7.PULSE_FRAME
            plain_packet = plain[start:start+v7.PULSE_FRAME]
            tone_packet = with_tones[start:start+v7.PULSE_FRAME]
            body = plain_packet[v7.PULSE.SYNC_LEN:
                                v7.PULSE.SYNC_LEN+v7.FRAME]
            body_rms = float(np.sqrt(np.mean(body*body)))
            amplitude = np.sqrt(2)*body_rms*10**(
                v7.PILOT_TONE_REL_DB/20)
            absolute = ((counter-1)*v7.PULSE_FRAME +
                        np.arange(v7.PULSE_FRAME))
            expected = sum(
                amplitude*np.cos(
                    2*np.pi*k*absolute/v7.N + v7.PILOT_TONE_PHASES[k])
                for k in v7.PILOT_TONE_BINS)
            actual = tone_packet[:, 0]-plain_packet[:, 0]
            np.testing.assert_allclose(actual, expected, atol=2e-7, rtol=2e-6)

    def test_phase_track_recovers_known_fast_flutter(self):
        sample_rate = 96_000
        count = 10
        wire = v7.encode_pulse_stream(
            self.model, [self.values]*count, pilot_tones=True)
        source = resample_poly(wire, 2, 1, axis=0).astype(np.float32)
        source_indexes = np.arange(len(source), dtype=float)
        time = source_indexes/sample_rate
        speed_error = (.001*np.sin(2*np.pi*25*time) +
                       .0005*np.sin(2*np.pi*60*time+.4))
        position = source_indexes+np.cumsum(speed_error)
        warped = np.column_stack([
            np.interp(position, source_indexes, source[:, channel],
                      left=0, right=0)
            for channel in range(2)]).astype(np.float32)

        starts = v7.pulse_frame_starts(warped, sample_rate=sample_rate)
        self.assertGreaterEqual(len(starts), count-1)
        errors = []
        correlations = []
        for counter, ((start, _, _), (following, _, _)) in enumerate(
                zip(starts, starts[1:]), 1):
            frame_scale = (following-start)/v7.PULSE_FRAME
            indexes = (start+v7.PULSE.SYNC_LEN*frame_scale +
                       np.arange(v7.FRAME)*frame_scale)
            body = v7._sample_at(warped, indexes, taps=4).astype(np.float32)
            windows = body.reshape(v7.F, v7.SYM, 2)[:,
                                   v7.WIN:v7.WIN+v7.N, :]
            Z = (np.fft.rfft(windows, axis=1)/self.model.scale *
                 np.conj(self.model.phase)[:, :, None] *
                 v7.EARLY[None, :, None])
            estimate, metrics = v7.pilot_tone_timing(Z, self.model, counter)
            self.assertIsNotNone(estimate, metrics)

            centers = (start+(v7.PULSE.SYNC_LEN+
                              np.arange(v7.F)*v7.SYM+v7.WIN+
                              (v7.N-1)/2)*frame_scale)
            actual_source = np.interp(centers, source_indexes, position)
            linear_source = (np.interp(start, source_indexes, position) +
                             (np.interp(following, source_indexes, position)-
                              np.interp(start, source_indexes, position))*
                             (centers-start)/(following-start))
            truth = (actual_source-linear_source)/(sample_rate/v7.RATE)
            truth -= np.median(truth)
            recovered = estimate['per_symbol']
            errors.extend((recovered-truth).tolist())
            correlations.append(float(np.corrcoef(recovered, truth)[0, 1]))

            joint_estimate, joint_metrics = v7.pilot_tone_timing(
                Z, self.model, counter, estimator='joint')
            self.assertIsNotNone(joint_estimate, joint_metrics)
            self.assertEqual(joint_metrics['tone_bins_used'], [1, 3])
            self.assertLess(
                joint_metrics['dual_tone_timing_disagreement_samples'], .5)
            errors.extend((joint_estimate['per_symbol']-truth).tolist())
            correlations.append(float(np.corrcoef(
                joint_estimate['per_symbol'], truth)[0, 1]))

        self.assertLess(float(np.sqrt(np.mean(np.square(errors)))), .25)
        self.assertGreater(float(np.mean(correlations)), .7)

        baseline, _ = v7.decode_pulse_stream(
            self.model, warped, sample_rate=sample_rate)
        joint, _ = v7.decode_pulse_stream(
            self.model, warped, sample_rate=sample_rate,
            pilot_timing='tone-joint')
        self.assertGreaterEqual(
            sum(result.status != 'lost' for result in joint),
            sum(result.status != 'lost' for result in baseline))
        self.assertGreater(sum(
            result.diag.get('pilot_timing', {}).get('mode_applied') ==
            'tone-joint' for result in joint), 0)

    def test_joint_tone_gate_survives_one_contaminated_empty_bin(self):
        wire = v7.encode_pulse_stream(
            self.model, [self.values]*4, pilot_tones=True)
        body = wire[v7.PULSE.SYNC_LEN:v7.PULSE.SYNC_LEN+v7.FRAME]
        windows = body.reshape(v7.F, v7.SYM, 2)[:,
                               v7.WIN:v7.WIN+v7.N, :]
        Z = (np.fft.rfft(windows, axis=1)/self.model.scale *
             np.conj(self.model.phase)[:, :, None] *
             v7.EARLY[None, :, None])

        for guard_bin in (0, 2):
            with self.subTest(guard_bin=guard_bin):
                contaminated = Z.copy()
                contaminated[:, guard_bin, :] = 100
                track, metrics = v7.pilot_tone_timing(
                    contaminated, self.model, 1, estimator='joint')
                self.assertIsNotNone(track, metrics)
                self.assertEqual(metrics['tone_bins_used'], [1, 3])
                guard_index = (0, 1)[(0, 2).index(guard_bin)]
                self.assertGreater(metrics['empty_bin_levels'][guard_index],
                                   100*metrics['empty_bin_level'])

    def test_tone_phase_can_recover_a_mistimed_metadata_symbol(self):
        wire = v7.encode_pulse_stream(
            self.model, [self.values]*3, pilot_tones=True)
        body = wire[v7.PULSE.SYNC_LEN:v7.PULSE.SYNC_LEN+v7.FRAME]
        actual_start = v7.PULSE.SYNC_LEN+v7.FRAME
        # Move the FFT window far enough before the metadata prefix to corrupt
        # fields, while keeping the continuous low-bin reference observable.
        bad_start = actual_start-18
        offset, metrics = v7._pilot_metadata_offset(
            body, wire, bad_start, 1.0, self.model, 1)
        self.assertIsNotNone(offset, metrics)
        self.assertLess(offset, 0)

        bad = v7.decode_metadata(self.model, wire, bad_start, 1, None)
        corrected = v7.decode_metadata(
            self.model, wire, bad_start-offset, 1, None)
        self.assertNotEqual(getattr(bad, 'source_index', None), 0)
        self.assertIsNotNone(corrected)
        self.assertEqual(corrected.source_index, 0)

    def test_timing_aided_retry_recovers_metadata_in_previous_lowpass_case(self):
        count = 40
        packets = [v7.encode_pulse_frame(
            self.model, self.values, counter, source_index=counter-1)
            for counter in range(1, count+1)]
        toned = np.concatenate([
            v7._add_pilot_tones(packet, counter, gate_preamble=True)
            for counter, packet in enumerate(packets, 1)])
        audio = resample_poly(toned, 2, 1, axis=0).astype(np.float32)
        audio = sosfilt(
            butter(4, 4000, btype='lowpass', fs=96000, output='sos'),
            audio, axis=0).astype(np.float32)

        baseline, _ = v7.decode_pulse_stream(
            self.model, audio, sample_rate=96000)
        retried, _ = v7.decode_pulse_stream(
            self.model, audio, sample_rate=96000,
            pilot_timing='tone-joint')
        baseline_valid = sum(bool(result.diag.get('metadata_valid'))
                             for result in baseline)
        retried_valid = sum(bool(result.diag.get('metadata_valid'))
                            for result in retried)
        recovered = sum(bool(result.diag.get('metadata_pilot_retry', {}).get('valid'))
                        for result in retried)
        self.assertGreater(retried_valid, baseline_valid)
        self.assertGreaterEqual(recovered, 1)
        self.assertGreaterEqual(
            sum(result.status != 'lost' for result in retried),
            sum(result.status != 'lost' for result in baseline))

    def test_receiver_modes_fall_back_without_tones_and_decode_with_tones(self):
        plain = v7.encode_pulse_stream(self.model, [self.values]*6)
        no_tone, _ = v7.decode_pulse_stream(
            self.model, plain, pilot_timing='tone-joint')
        self.assertEqual(len(no_tone), 5)
        self.assertTrue(all(
            not result.diag['pilot_timing']['detected'] and
            result.diag['pilot_timing']['mode_applied'] == 'baseline'
            for result in no_tone))

        toned = v7.encode_pulse_stream(
            self.model, [self.values]*6, pilot_tones=True)
        baseline, _ = v7.decode_pulse_stream(self.model, toned)
        for mode in ('tone-seeded', 'tone-joint', 'tone-replaced'):
            with self.subTest(mode=mode):
                experimental, _ = v7.decode_pulse_stream(
                    self.model, toned, pilot_timing=mode)
                self.assertEqual(len(experimental), len(baseline))
                self.assertEqual(
                    sum(result.status != 'lost' for result in experimental),
                    sum(result.status != 'lost' for result in baseline))
                if mode == 'tone-joint':
                    self.assertTrue(all(
                        result.diag['pilot_timing']['mode_applied'] in
                        ('baseline', mode)
                        for result in experimental))
                else:
                    self.assertTrue(all(
                        result.diag['pilot_timing']['detected']
                        for result in experimental))

    def test_default_receiver_image_is_unchanged_at_capture_speeds(self):
        count = 8
        for speed in (1.0, 1.5, 2.0):
            with self.subTest(speed=speed):
                decoded_by_tone = {}
                for enabled in (False, True):
                    wire = v7.encode_pulse_stream(
                        self.model, [self.values]*count,
                        pilot_tones=enabled)
                    audio = v7.speed_pulse_stream(
                        wire, speed, rate=96_000)
                    results, _ = v7.decode_pulse_stream(
                        self.model, audio, sample_rate=96_000)
                    decoded_by_tone[enabled] = results
                plain = decoded_by_tone[False]
                toned = decoded_by_tone[True]
                self.assertEqual(len(plain), len(toned))
                self.assertEqual(
                    [result.status for result in plain],
                    [result.status for result in toned])

                def mean_rmse(results):
                    values = [v7.values_from(self.model, result.coeffs)
                              for result in results
                              if result.status != 'lost']
                    return float(np.mean([
                        np.sqrt(np.mean((value-self.values)**2))
                        for value in values]))

                self.assertLessEqual(
                    abs(mean_rmse(plain)-mean_rmse(toned)), .0005)

    def test_raw_tone_speed_matches_pulse_speed_from_half_to_one_and_half_x(self):
        sample_rate = 96_000
        count = 9
        wire = v7.encode_pulse_stream(
            self.model, [self.values]*count, pilot_tones=True)
        for speed in (.5, 1.0, 1.5):
            with self.subTest(speed=speed):
                audio = v7.speed_pulse_stream(
                    wire, speed, rate=sample_rate)
                starts = v7.pulse_frame_starts(
                    audio, sample_rate=sample_rate)
                self.assertEqual(len(starts), count)
                differences = []
                for (start, _, _), (following, _, _) in zip(
                        starts, starts[1:]):
                    scale = (following-start)/v7.PULSE_FRAME
                    estimate = v7.pilot_tone_speed(
                        audio, sample_rate, start, scale)
                    self.assertTrue(estimate['detected'], estimate)
                    differences.append(abs(estimate['difference_pct']))
                self.assertLess(max(differences), .1)
                if speed == 1.0:
                    decoded, _ = v7.decode_pulse_stream(
                        self.model, audio, sample_rate=sample_rate,
                        pilot_speed_diagnostics=True)
                    speed_diag = decoded[0].diag['pilot_tone_speed']
                    self.assertTrue(speed_diag['detected'], speed_diag)
                    self.assertLess(abs(speed_diag['difference_pct']), .1)

    def test_live_sender_pilot_flag_is_opt_in(self):
        off = v7_live.parser().parse_args([
            'send', '--source', 'test', '--device', 'null'])
        on = v7_live.parser().parse_args([
            'send', '--source', 'test', '--device', 'null', '--pilot-tones'])
        self.assertFalse(off.pilot_tones)
        self.assertTrue(on.pilot_tones)
        receive = v7_live.parser().parse_args([
            'receive', '--device', 'null', '--pilot-timing', 'tone-joint'])
        self.assertEqual(receive.pilot_timing, 'tone-joint')
        self.assertEqual(receive.tone_equalization, 'off')
        tone_eq_receive = v7_live.parser().parse_args([
            'receive', '--device', 'null', '--tone-equalization', 'm-reference'])
        self.assertEqual(tone_eq_receive.tone_equalization, 'm-reference')
        pulse_receive = v7_live.parser().parse_args([
            'receive', '--device', 'null', '--pulse-timing', 'pulse-warp'])
        self.assertEqual(pulse_receive.pulse_timing, 'pulse-warp')

    def test_application_packet_forwards_pilot_flag(self):
        import modem_v7_display

        class Library:
            def composite(self, *_args, **_kwargs):
                image = Image.new('RGB', (80, 96))
                image.info['source_dimensions'] = (80, 96)
                return image

        output = np.zeros((v7.PULSE_FRAME, 2), np.float32)
        with mock.patch('modem_v7_display._source_values',
                        return_value=self.values), \
                mock.patch('modem_v7_display._v7.encode_pulse_frame',
                           return_value=output) as encode:
            modem_v7_display.packet(
                Library(), self.model, 1, 0, (0, 0, 0), pilot_tones=True)
        self.assertTrue(encode.call_args.kwargs['pilot_tones'])

    def test_tone_replacement_wins_for_known_per_symbol_timing_track(self):
        symbols = np.arange(v7.F)
        true_delta = 1.5*np.sin(2*np.pi*5*symbols/v7.F+.2)
        channel = np.zeros((65, 2, 2), complex)
        channel[:, 0, 0] = 1+.2j
        channel[:, 0, 1] = .2-.1j
        channel[:, 1, 0] = .4+.1j
        channel[:, 1, 1] = 1-.3j
        Z = np.zeros((v7.F, 65, 2), complex)
        for index, (symbol, bin_index) in enumerate(v7.PILOT_OBS):
            rotation = np.exp(2j*np.pi*bin_index*true_delta[symbol]/v7.N)
            Z[symbol, bin_index, :] = rotation*(
                channel[bin_index]@v7.PILOT_PV[index])
        for bin_index in v7.PILOT_TONE_BINS:
            nominal = (v7.PULSE.SYNC_LEN +
                       symbols*v7.SYM + v7.WIN)
            physical = 8*np.exp(1j*(
                v7.PILOT_TONE_PHASES[bin_index] +
                2*np.pi*bin_index*nominal/v7.N +
                2*np.pi*bin_index*true_delta/v7.N))
            Z[:, bin_index, :] = (
                physical[:, None]/
                (self.model.scale*self.model.phase[:, bin_index, None]) *
                v7.EARLY[bin_index])

        _, timing = v7.channel_joint(
            Z, pilot_timing='tone-replaced', model=self.model, counter=1,
            return_timing_diag=True)
        self.assertEqual(timing['mode_applied'], 'tone-replaced')
        self.assertLess(timing['tone_pilot_residual'],
                        .5*timing['baseline_pilot_residual'])

    def test_unknown_timing_mode_fails_fast(self):
        with self.assertRaisesRegex(ValueError, 'unknown pilot timing mode'):
            v7.decode_pulse_stream(self.model, np.zeros((10, 2)),
                                   pilot_timing='guess')


if __name__ == '__main__':
    unittest.main()
