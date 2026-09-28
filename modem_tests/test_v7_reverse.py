"""V7 reverse pulse acquisition and EOF-validated packet normalization."""
import sys
import unittest
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps
from scipy.signal import resample_poly

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from animation_modem import transport3, v7                             # noqa: E402
from animation_modem.imaging import values_image                        # noqa: E402
from animation_modem.v7_live_input import (DirectionStreak, LiveInput,  # noqa: E402
                                           select_packet_hit)
from tools.v7_wire_profile import WireProfile                          # noqa: E402


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

    def test_schmitt_interval_codes_identify_all_profile_modes_both_directions(self):
        profiles = tuple(transport3.PROFILE_PREAMBLE_BITS)
        gap_words = {
            code: tuple(int(gap == transport3.SHORT)
                        for gap in transport3._PROFILE_GAPS[index])
            for index, code in enumerate(profiles)
        }
        reversed_gap_words = {
            code: tuple(int(gap == transport3.SHORT)
                        for gap in transport3._PROFILE_REVERSED_GAPS[index])
            for index, code in enumerate(profiles)
        }
        distances = [
            sum(a != b for a, b in zip(left, right))
            for index, code in enumerate(profiles)
            for other in profiles[index+1:]
            for left, right in (
                (gap_words[code], gap_words[other]),
                (gap_words[code], reversed_gap_words[other]),
                (reversed_gap_words[code], gap_words[other]),
            )
        ]
        self.assertEqual(min(distances), 6)

        for profile_code in transport3.PROFILE_PREAMBLE_BITS:
            preamble = transport3.profile_preamble(profile_code)
            packet = np.zeros(320, dtype=np.float32)
            packet[16:16+len(preamble)] = preamble
            for direction, capture in (
                    (1, packet), (-1, packet[::-1].copy())):
                hit = transport3.measure_pulses_profile_both(capture)
                with self.subTest(profile=profile_code, direction=direction):
                    self.assertIsNotNone(hit)
                    self.assertEqual(hit[3], direction)
                    self.assertEqual(hit[4], profile_code)
                    self.assertAlmostEqual(hit[1], 1.0, places=5)

    def test_profile_preamble_changes_keep_decoded_picture_equivalent(self):
        encode = lambda code: v7.encode_pulse_frame(
            self.model, self.values[0], 1, aspect_code=6, source_index=0,
            eof_marker=True, pulse_profile_code=code)
        reference_packet = encode(1)
        reference_results, reference_info = v7.decode_pulse_stream(
            self.model, reference_packet, frame_boundary='eof',
            state=v7.PulseState(tail_memory=False))
        self.assertEqual(len(reference_results), 1, reference_info)
        reference_coeffs = reference_results[0].coeffs.copy()
        reference_image = np.asarray(values_image(
            v7.values_from(self.model, reference_coeffs),
            self.model.coder.grids))

        for profile_code in transport3.PROFILE_PREAMBLE_BITS:
            candidate = encode(profile_code)
            results, info = v7.decode_pulse_stream(
                self.model, candidate, frame_boundary='eof',
                state=v7.PulseState(tail_memory=False))
            with self.subTest(profile=profile_code):
                self.assertEqual(len(results), 1, info)
                self.assertEqual(info['eof_markers_validated'], 1)
                np.testing.assert_allclose(
                    results[0].coeffs, reference_coeffs, rtol=0, atol=1.2e-3)
                image = np.asarray(values_image(
                    v7.values_from(self.model, results[0].coeffs),
                    self.model.coder.grids))
                self.assertLessEqual(
                    int(np.max(np.abs(image.astype(np.int16)-
                                      reference_image.astype(np.int16)))), 2)

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

    def test_mirrored_turnaround_keeps_both_words(self):
        # A reverse-to-forward turn on a packet boundary (a ping-pong loop, or
        # a tape reversing exactly there) puts the reversed preamble and the
        # forward one SYNC_LEN apart. Both are genuine packets.
        packet = self.stream(1)
        for speed in (.96, .98, 1.0, 1.02, 1.04):
            sped = v7.speed_pulse_stream(packet, speed, rate=v7.RATE)
            capture = np.concatenate((sped[::-1], sped)).astype(np.float32)
            with self.subTest(speed=speed):
                hits = v7.pulse_frame_hits(capture, sample_rate=v7.RATE)
                self.assertEqual([hit[3] for hit in hits], [-1, 1])
                self.assertAlmostEqual(hits[0][0], 0.0, delta=1.0)
                self.assertAlmostEqual(hits[1][0], len(sped), delta=1.0)
                mono = v7._mono(capture)
                window = mono[len(sped)-1000:len(sped)+1000]
                compiled = transport3.measure_pulses_both(window)
                reference = transport3.measure_pulses_both_numpy(window)
                self.assertEqual(compiled[3], -1)
                self.assertEqual(reference[3], -1)
        # Words that share their edges stay ambiguous.
        forward = (100.0, 1.0, 1.0, 1)
        self.assertTrue(transport3._opposite_words_overlap(
            forward, (100.0+len(transport3.PREAMBLE)-1, 1.0, 1.0, -1)))
        self.assertFalse(transport3._opposite_words_overlap(
            forward, (100.0-transport3.SYNC_LEN, 1.0, 1.0, -1)))

    def test_turnaround_hit_choice(self):
        forward_header = (5000.0, 1.0, 1.0, 1)
        reversed_packet = (1080.0, 1.0, 1.0, -1)
        # The forward hit only starts a packet; the reversed one is complete.
        self.assertEqual(select_packet_hit(
            (forward_header, reversed_packet), audio_start=0),
            reversed_packet)
        # ... unless that reversed packet was already decoded.
        self.assertEqual(select_packet_hit(
            (reversed_packet, forward_header), audio_start=10,
            decoded_through=1090), forward_header)
        # Steady forward and reverse streams take the newest hit.
        older_forward = (1080.0, 1.0, 1.0, 1)
        self.assertEqual(select_packet_hit((older_forward, forward_header)),
                         forward_header)
        newer_reverse = (5000.0, 1.0, 1.0, -1)
        self.assertEqual(select_packet_hit((reversed_packet, newer_reverse)),
                         newer_reverse)
        self.assertEqual(select_packet_hit((older_forward, newer_reverse)),
                         newer_reverse)
        self.assertIsNone(select_packet_hit(()))

    def test_tape_rocking_on_the_default_wire(self):
        # Forward 0-6, the tape reverses over 6-3, then plays forward 3-8,
        # through LiveInput with the receiver's wake-and-pick policy.
        profile = WireProfile('default')
        model = v7.load_model(TARGET, profile.encode_filter)
        with Image.open(v7.REFERENCE_FIXTURE) as image:
            values = v7.image_values(
                v7.prepare_image(image.convert('RGB'), profile.encode_filter),
                model.coder.grids, profile.encode_filter)
        packets = profile.encode(
            model, [values]*9, source_indices=range(9)).reshape(
                9, v7.PULSE_FRAME, 2)
        order = ([(index, 1) for index in range(7)] +
                 [(index, -1) for index in (6, 5, 4, 3)] +
                 [(index, 1) for index in range(3, 9)])
        capture = np.concatenate([
            packets[index] if way > 0 else packets[index][::-1]
            for index, way in order]).astype(np.float32)
        live = LiveInput(rate=v7.RATE)
        state = v7.PulseState()
        decoded_through = None
        shown = []
        with profile.receiving():
            for offset in range(0, len(capture), 1024):
                live.add(capture[offset:offset+1024].copy())
                audio = live.take(offset/v7.RATE)
                if audio is None:
                    continue
                hits = live.pulse_hits(audio)
                if not hits:
                    live.decoded()
                    continue
                audio_start = live.total-len(audio)
                start, scale, _, way = select_packet_hit(
                    hits, audio_start, decoded_through)
                decoded_through = int(round(audio_start+start))
                state.set_playback_direction(way)
                if way < 0:
                    results, _ = v7.decode_reverse_packet(
                        model, audio, start, scale, state=state,
                        sample_rate=v7.RATE, **profile.decode_options)
                else:
                    results, _ = v7.decode_pulse_stream(
                        model, audio, latest_only=True,
                        pulse_starts=live.pulse_starts(audio), state=state,
                        sample_rate=v7.RATE, **profile.decode_options)
                live.decoded()
                if results and (results[-1].status in ('received', 'verified')
                                or results[-1].diag.get('displayable')):
                    shown.append((results[-1].diag['source_index'], way))
        # The forward packet before the turn (source 6) is not decoded: the
        # receiver next wakes on the reversed preamble, one packet later, and
        # shows that newer packet. The last packet has no following header.
        self.assertEqual(shown, [
            (0, 1), (1, 1), (2, 1), (3, 1), (4, 1), (5, 1),
            (6, -1), (5, -1), (4, -1), (3, -1),
            (3, 1), (4, 1), (5, 1), (6, 1), (7, 1)])

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
