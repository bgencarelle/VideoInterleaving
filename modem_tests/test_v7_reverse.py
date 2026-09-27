"""V7 reverse pulse acquisition and EOF-validated packet normalization."""
import unittest

import numpy as np
from PIL import Image, ImageOps
from scipy.signal import resample_poly

from animation_modem import transport3, v7
from animation_modem.v7_live_input import DirectionStreak, LiveInput


TARGET = .1521/np.sqrt(1 + 10**(v7.CLOCK_REL_DB/10))


class V7ReverseTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = v7.load_model(TARGET, 'nearest')
        with Image.open(v7.REFERENCE_FIXTURE) as image:
            cls.image = image.convert('RGB')
            cls.mirror = ImageOps.mirror(cls.image)
        cls.values = [
            v7.image_values(v7.prepare_image(image, 'nearest'),
                            cls.model.coder.grids, 'nearest')
            for image in (cls.image, cls.mirror)]

    def stream(self, count=4):
        images = [self.values[index % 2] for index in range(count)]
        return np.concatenate([
            v7.encode_pulse_frame(
                self.model, values, counter=index+1, aspect_code=6,
                source_index=index, eof_marker=True)
            for index, values in enumerate(images)])

    def test_bidirectional_matcher_matches_reference_and_separates_words(self):
        wire = self.stream(3)
        for name, signal, expected in (
                ('forward', v7._mono(wire), 1),
                ('reverse', v7._mono(wire[::-1].copy()), -1)):
            compiled = transport3.measure_pulses_both(signal)
            reference = transport3.measure_pulses_both_numpy(signal)
            with self.subTest(direction=name):
                self.assertIsNotNone(compiled)
                self.assertIsNotNone(reference)
                self.assertEqual(compiled[3], expected)
                np.testing.assert_allclose(compiled, reference, rtol=0, atol=1e-12)
                forced = transport3.measure_pulses_both(
                    signal, direction='forward' if expected > 0 else 'reverse')
                self.assertEqual(forced[3], expected)
                opposite = transport3.measure_pulses_both(
                    signal, direction='reverse' if expected > 0 else 'forward')
                self.assertIsNone(opposite)
        self.assertIsNone(transport3.measure_pulses_both_numpy(
            np.zeros(8000, dtype=np.float64)))
        rng = np.random.default_rng(771)
        self.assertIsNone(transport3.measure_pulses_both(
            rng.normal(0, .2, 20000).astype(np.float32)))

    def test_bidirectional_matcher_tolerates_signed_zero_dropout_edges(self):
        signal = np.asarray([.5, 0.0, -0.0, 0.0, -0.0, -.5],
                            dtype=np.float32)
        self.assertIsNone(transport3.measure_pulses_both(signal))

    def test_packet_hits_have_capture_order_and_correct_direction(self):
        wire = self.stream(4)
        for direction, audio in ((1, wire), (-1, wire[::-1].copy())):
            hits = v7.pulse_frame_hits(audio, direction='auto')
            with self.subTest(direction=direction):
                self.assertEqual(len(hits), 4)
                self.assertTrue(all(hit[3] == direction for hit in hits))
                np.testing.assert_allclose(
                    [hit[0] for hit in hits], np.arange(4)*v7.PULSE_FRAME,
                    atol=.2)

    def test_auto_acquisition_tracks_packet_by_packet_turnarounds(self):
        packets = [v7.encode_pulse_frame(
            self.model, self.values[index % 2], counter=index+1,
            aspect_code=6, source_index=index, eof_marker=True)
            for index in range(6)]
        directions = (1, 1, -1, -1, 1, 1)
        mixed = np.concatenate([
            packet if direction > 0 else packet[::-1]
            for packet, direction in zip(packets, directions)])
        hits = v7.pulse_frame_hits(mixed)
        self.assertEqual([hit[3] for hit in hits], list(directions))
        np.testing.assert_allclose(
            [hit[0] for hit in hits], np.arange(6)*v7.PULSE_FRAME,
            atol=.2)

    def test_reverse_input_finds_complete_packets_at_the_trailing_preamble(self):
        reverse = self.stream(4)[::-1].copy().astype(np.float32)
        live = LiveInput(rate=v7.RATE)
        state = v7.PulseState(tail_memory=False)
        found = []
        decoded_sources = []
        block = 1024
        for start in range(0, len(reverse), block):
            live.add(reverse[start:start+block].copy())
            audio = live.take(start/v7.RATE)
            if audio is None:
                continue
            base = live.total-len(audio)
            hits = live.pulse_hits(audio)
            found.extend((round(base+hit[0]), hit[3]) for hit in hits)
            if hits:
                packet_start, scale, _, direction = max(
                    hits, key=lambda hit: hit[0])
                self.assertEqual(direction, -1)
                results, info = v7.decode_reverse_packet(
                    self.model, audio, packet_start, scale, state=state,
                    sample_rate=v7.RATE)
                self.assertEqual(info['eof_markers_validated'], 1)
                self.assertEqual(len(results), 1)
                decoded_sources.append(results[0].diag['source_index'])
            live.decoded()
        unique = sorted(set(found))
        self.assertEqual([direction for _, direction in unique], [-1]*4)
        np.testing.assert_allclose(
            [position for position, _ in unique],
            np.arange(4)*v7.PULSE_FRAME, atol=1.0)
        self.assertEqual(decoded_sources, [3, 2, 1, 0])

    def test_reverse_packets_decode_in_capture_order_with_frame_identity(self):
        wire = self.stream(4)
        forward, _ = v7.decode_pulse_stream(
            self.model, wire, frame_boundary='eof',
            state=v7.PulseState(tail_memory=False))
        reverse = wire[::-1].copy()
        hits = v7.pulse_frame_hits(reverse)
        state = v7.PulseState(tail_memory=False)
        decoded = []
        for packet_start, scale, confidence, direction in hits:
            self.assertEqual(direction, -1)
            results, info = v7.decode_reverse_packet(
                self.model, reverse, packet_start, scale, state=state,
                sample_rate=v7.RATE, pilot_timing='baseline')
            self.assertEqual(info['eof_markers_validated'], 1)
            self.assertEqual(len(results), 1)
            result = results[0]
            self.assertTrue(result.diag['metadata_valid'])
            self.assertEqual(result.diag['playback_direction'], -1)
            decoded.append(result)
        self.assertEqual([r.diag['source_index'] for r in decoded], [3, 2, 1, 0])
        self.assertEqual(len(forward), len(decoded))
        for backwards, forwards in zip(decoded, reversed(forward)):
            np.testing.assert_allclose(backwards.coeffs, forwards.coeffs,
                                       rtol=0, atol=2e-3)

    def test_unity_gain_decode_does_not_mutate_capture_buffers(self):
        packet = self.stream(1).astype(np.float32)
        original = packet.copy()
        forward_hit = v7.pulse_frame_hits(packet)[0]
        forward, _ = v7.decode_pulse_stream(
            self.model, packet, latest_only=True, frame_boundary='eof',
            pulse_starts=[forward_hit[:3]])
        self.assertEqual(len(forward), 1)
        np.testing.assert_array_equal(packet, original)

        reverse = original[::-1]
        reverse_before = reverse.copy()
        reverse_hit = v7.pulse_frame_hits(reverse)[0]
        decoded, _ = v7.decode_reverse_packet(
            self.model, reverse, reverse_hit[0], reverse_hit[1],
            state=v7.PulseState(tail_memory=False), sample_rate=v7.RATE)
        self.assertEqual(len(decoded), 1)
        np.testing.assert_array_equal(reverse, reverse_before)

    def test_fresh_receiver_holds_unverifiable_loop_field_tail_slices(self):
        for counter in (5, 6):
            packet = v7.encode_pulse_frame(
                self.model, self.values[0], counter, aspect_code=6,
                source_index=4, eof_marker=True)
            reverse = packet[::-1].copy()
            hit = v7.pulse_frame_hits(reverse)[0]
            results, info = v7.decode_reverse_packet(
                self.model, reverse, hit[0], hit[1], state=v7.PulseState(),
                sample_rate=v7.RATE, pilot_timing='baseline')
            with self.subTest(tail_slice=counter % v7.TAIL_PHASES):
                self.assertEqual(info['eof_markers_validated'], 1)
                self.assertEqual(len(results), 1)
                self.assertFalse(results[0].diag['metadata_valid'])
                self.assertFalse(results[0].diag['displayable'])
                self.assertEqual(
                    results[0].diag['reverse_rejected'],
                    'metadata_not_independently_valid')

    def test_reverse_eof_damage_does_not_commit_or_authorize_a_neighbor(self):
        wire = self.stream(3)
        damaged = wire.copy()
        damaged[-v7.EOF_MARKER_LENGTH:] = 0
        reverse = damaged[::-1].copy()
        hits = v7.pulse_frame_hits(reverse)
        results, info = v7.decode_reverse_packet(
            self.model, reverse, hits[0][0], hits[0][1],
            state=v7.PulseState(), sample_rate=v7.RATE)
        self.assertEqual(results, [])
        self.assertEqual(info.get('eof_markers_validated', 0), 0)

    def test_provisional_reverse_metadata_does_not_update_tail_memory(self):
        state = v7.PulseState()
        for counter, source_index in ((1, 0), (5, 1)):
            packet = v7.encode_pulse_frame(
                self.model, self.values[0], counter, aspect_code=6,
                source_index=source_index, eof_marker=True)
            reverse = packet[::-1].copy()
            hit = v7.pulse_frame_hits(reverse)[0]
            results, _ = v7.decode_reverse_packet(
                self.model, reverse, hit[0], hit[1], state=state,
                sample_rate=v7.RATE)
            self.assertEqual(len(results), 1)
            if counter == 1:
                self.assertFalse(results[0].diag['metadata_provisional'])
                prior_values = state.tail._values.copy()
                prior_age = state.tail._age.copy()
            else:
                self.assertTrue(results[0].diag['metadata_provisional'])
                self.assertFalse(results[0].diag['displayable'])
                np.testing.assert_array_equal(state.tail._values, prior_values)
                np.testing.assert_array_equal(state.tail._age, prior_age)

    def test_direction_overrides_filter_the_other_pulse_word(self):
        wire = self.stream(2)
        self.assertEqual(len(v7.pulse_frame_hits(wire, direction='forward')), 2)
        self.assertEqual(v7.pulse_frame_hits(wire, direction='reverse'), [])
        reverse = wire[::-1].copy()
        self.assertEqual(len(v7.pulse_frame_hits(reverse, direction='reverse')), 2)
        self.assertEqual(v7.pulse_frame_hits(reverse, direction='forward'), [])

    def test_varispeed_capture_rates_and_polarity(self):
        wire = self.stream(3)
        cases = []
        for rate, speed in ((44100, 1.0), (48000, .8), (96000, 1.5)):
            sped = v7.speed_pulse_stream(wire, speed)
            captured = resample_poly(sped, rate, v7.RATE, axis=0)
            cases.append((rate, speed, captured[::-1].copy()))
        for rate, speed, audio in cases:
            for polarity in (1, -1):
                reverse = (audio*polarity).astype(np.float32)
                hits = v7.pulse_frame_hits(reverse, sample_rate=rate)
                with self.subTest(rate=rate, speed=speed, polarity=polarity):
                    self.assertEqual(len(hits), 3)
                    self.assertTrue(all(hit[3] == -1 for hit in hits))
                    state = v7.PulseState()
                    results = []
                    for start, scale, _, _ in hits:
                        decoded, _ = v7.decode_reverse_packet(
                            self.model, reverse, start, scale, state=state,
                            sample_rate=rate, pilot_timing='baseline')
                        results.extend(decoded)
                    self.assertGreaterEqual(
                        sum(bool(r.diag.get('metadata_valid')) for r in results),
                        2)

    def test_reverse_slow_playback_below_quarter_speed(self):
        wire = self.stream(3)
        for speed in (.2, .1, .05, .025, .01):
            reverse = v7.speed_pulse_stream(
                wire, speed, rate=v7.RATE)[::-1].copy()
            margin = int(np.ceil(v7.REVERSE_PACKET_MARGIN/speed))+8
            silence = np.zeros((margin, 2), dtype=np.float32)
            reverse = np.concatenate((silence, reverse, silence))
            hits = v7.pulse_frame_hits(reverse, sample_rate=v7.RATE)
            with self.subTest(speed=speed):
                self.assertEqual(len(hits), 3)
                self.assertTrue(all(hit[3] == -1 for hit in hits))
                np.testing.assert_allclose(
                    [hit[1] for hit in hits], 1/speed, rtol=.002)
                state = v7.PulseState(tail_memory=False)
                sources = []
                for start, scale, _, _ in hits:
                    decoded, info = v7.decode_reverse_packet(
                        self.model, reverse, start, scale, state=state,
                        sample_rate=v7.RATE, pilot_timing='baseline')
                    self.assertEqual(info['eof_markers_validated'], 1)
                    self.assertEqual(len(decoded), 1)
                    self.assertEqual(decoded[0].status, 'received')
                    sources.append(decoded[0].diag['source_index'])
                self.assertEqual(sources, [2, 1, 0])

    def test_slowest_reverse_scale_fits_live_input_buffer_and_decodes(self):
        speed = .01
        wire = self.stream(3)
        reverse = v7.speed_pulse_stream(
            wire, speed, rate=v7.RATE)[::-1].copy()
        margin = int(np.ceil(v7.REVERSE_PACKET_MARGIN/speed))+8
        silence = np.zeros((margin, 2), dtype=np.float32)
        capture = np.concatenate((silence, reverse, silence))*np.asarray(
            [1, -1], dtype=np.float32)
        live = LiveInput(rate=v7.RATE, direction='auto')
        state = v7.PulseState(tail_memory=False)
        sources = []
        last_arrival = None
        block_size = 1024
        for offset in range(0, len(capture), block_size):
            live.add(capture[offset:offset+block_size].copy())
            audio = live.take(offset/v7.RATE)
            if audio is None:
                continue
            hits = [hit for hit in live.pulse_hits(audio) if hit[3] < 0]
            if hits:
                packet_start, scale, _, direction = max(
                    hits, key=lambda hit: hit[0])
                audio_start = live.total-len(audio)
                arrival = int(round(audio_start+packet_start))
                if arrival != last_arrival:
                    state.set_playback_direction(direction)
                    decoded, info = v7.decode_reverse_packet(
                        self.model, audio, packet_start, scale, state=state,
                        sample_rate=v7.RATE, pilot_timing='baseline')
                    self.assertEqual(info['eof_markers_validated'], 1)
                    self.assertEqual(len(decoded), 1)
                    self.assertEqual(decoded[0].status, 'received')
                    sources.append(decoded[0].diag['source_index'])
                    last_arrival = arrival
            live.decoded()
        self.assertEqual(live.max_scale, 100.0)
        self.assertEqual(live.polarity, -1)
        self.assertEqual(sources, [2, 1, 0])

    def test_direction_confirmation_requires_two_distinct_valid_arrivals(self):
        streak = DirectionStreak()
        self.assertEqual(streak.observe(100, -1, True), (None, False))
        self.assertEqual(streak.observe(100, -1, True), (None, False))
        self.assertEqual(streak.observe(200, -1, True), (-1, True))
        self.assertEqual(streak.observe(300, 1, True), (-1, False))
        self.assertEqual(streak.observe(400, 1, False), (-1, False))
        self.assertEqual(streak.observe(500, 1, True), (-1, False))
        self.assertEqual(streak.observe(600, 1, True), (1, True))

    def test_direction_switch_resets_tail_but_not_loop_lock(self):
        state = v7.PulseState()
        state.lock.values[v7.TAIL_SLICE_P] = 37
        state.last_verified = object()
        state.set_playback_direction(1)
        changed = state.set_playback_direction(-1)
        self.assertTrue(changed)
        self.assertIsNone(state.last_verified)
        self.assertIsNone(state.tail._values)
        self.assertEqual(state.lock.values[v7.TAIL_SLICE_P], 37)


if __name__ == '__main__':
    unittest.main()
