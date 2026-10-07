"""Emitted levels: the header's pulses are the loudest part of every packet,
a few dB under full scale, and the picture body stays below them."""
import unittest

import numpy as np

from animation_modem import v7
from animation_modem.v7_coded_pilot import add_fold500_coded_pilot

TARGET = .1521/np.sqrt(1 + 10**(v7.CLOCK_REL_DB/10))
BODY = slice(v7.PULSE.SYNC_LEN, v7.PULSE.SYNC_LEN+v7.FRAME)


def _db(value):
    return 20*np.log10(value)


class EmittedLevelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = v7.load_model(TARGET, 'box')
        rng = np.random.default_rng(4)
        deviation = np.sqrt(cls.model.lam)
        # A quiet picture, an ordinary one and one far louder than the model.
        cls.packets = [
            v7.encode_pulse_frame_coeffs(
                cls.model, cls.model.mu+scale*deviation *
                rng.standard_normal(len(deviation)), counter,
                pilot_tones=True)
            for counter, scale in enumerate((.3, 1.0, 2.0, 4.0), 1)]

    def test_header_is_at_one_level_near_full_scale(self):
        peaks = [float(np.max(np.abs(packet[:v7.PULSE.SYNC_LEN])))
                 for packet in self.packets]
        for peak in peaks:
            self.assertLess(_db(peak), -2.0)
            self.assertGreater(_db(peak), -3.2)
        self.assertLess(_db(max(peaks))-_db(min(peaks)), .5)

    def test_body_stays_below_the_header_and_nothing_passes_full_scale(self):
        for packet in self.packets:
            header = float(np.max(np.abs(packet[:v7.PULSE.SYNC_LEN])))
            body = float(np.max(np.abs(packet[BODY])))
            self.assertGreater(_db(header)-_db(body), 1.0)
            self.assertLess(float(np.max(np.abs(packet))), 1.0)

    def test_only_a_loud_body_is_scaled_down(self):
        def body_peak(scale):
            rng = np.random.default_rng(4)
            coefficients = self.model.mu+scale*np.sqrt(self.model.lam) * \
                rng.standard_normal(len(self.model.mu))
            packet = v7.encode_pulse_frame_coeffs(self.model, coefficients, 1)
            framing = min(float(np.max(np.abs(packet[:v7.PULSE.SYNC_LEN]))),
                          float(np.max(np.abs(packet[v7.EOF_MARKER_OFFSET:]))))
            return (float(np.max(np.abs(packet[BODY]))),
                    framing*10**(-v7.BODY_BELOW_EOF_DB/20))
        # A quiet picture's body goes out as coded, under the ceiling the
        # packet's own header and end marker set; loud ones are held at it
        # however loud they are.
        quiet, ceiling = body_peak(.3)
        self.assertLess(quiet, .8*ceiling)
        for scale in (4.0, 8.0):
            peak, ceiling = body_peak(scale)
            self.assertAlmostEqual(peak, ceiling, delta=.01)

    def test_end_marker_is_at_the_header_pulse_level(self):
        level = v7.emitted_pulse_level()
        quiet = v7.encode_pulse_frame_coeffs(
            self.model, self.model.mu, 1)
        marker = quiet[-v7.EOF_MARKER_LENGTH:, 0]
        np.testing.assert_allclose(np.abs(marker), level, rtol=.02)

    def test_final_packet_keeps_header_eof_body_metadata_level_order(self):
        body_region = slice(v7.PULSE.SYNC_LEN,
                            v7.PULSE.SYNC_LEN+v7.FRAME)
        metadata_start = v7.PULSE.SYNC_LEN+v7.FRAME
        metadata_region = slice(metadata_start,
                                metadata_start+v7.META_SYMBOL)
        header_region = slice(0, v7.PULSE.SYNC_LEN)
        eof_region = slice(v7.EOF_MARKER_OFFSET, v7.PULSE_FRAME)
        deviation = np.sqrt(self.model.lam)

        def peak(packet, region):
            return float(np.max(np.abs(packet[region])))

        def delta_db(high, low):
            return 20*np.log10(high/low)

        # Cover every pulse word and all eight pilot-tone phase origins. The
        # loud coefficient cases engage the body limiter; the quiet case checks
        # that it does not raise a naturally lower body to the target.
        for profile in range(6):
            for counter in range(1, 9):
                for level in (.3, 4.0):
                    rng = np.random.default_rng(
                        10_000+profile*100+counter*10+int(level*10))
                    coeffs = (self.model.mu+level*deviation*
                              rng.standard_normal(len(self.model.mu)))
                    packet = v7.encode_pulse_frame_coeffs(
                        self.model, coeffs, counter, source_index=counter-1,
                        extra_tone_mixer=lambda audio: add_fold500_coded_pilot(
                            audio, counter),
                        pulse_profile_code=profile)
                    header_peak = peak(packet, header_region)
                    eof_peak = peak(packet, eof_region)
                    body_peak = peak(packet, body_region)
                    metadata_peak = peak(packet, metadata_region)

                    context = (profile, counter, level)
                    # Tone mixing changes the measured region peaks slightly;
                    # the untoned pulse-level relationship is 1.5 dB.
                    self.assertAlmostEqual(
                        delta_db(header_peak, eof_peak), 1.5, delta=.5,
                        msg=f'header/EOF level gap {context}')
                    eof_body_gap = delta_db(eof_peak, body_peak)
                    self.assertGreaterEqual(
                        eof_body_gap, v7.BODY_BELOW_EOF_DB-.02,
                        f'EOF/body level gap {context}')
                    if level >= 4.0:
                        self.assertLessEqual(
                            eof_body_gap, 2.0+.02,
                            f'body was over-limited {context}')
                    self.assertGreaterEqual(
                        delta_db(body_peak, metadata_peak),
                        v7.METADATA_BELOW_BODY_DB-.02,
                        f'body/metadata level gap {context}')
                    packet_peak = float(np.max(np.abs(packet)))
                    self.assertAlmostEqual(packet_peak, header_peak,
                                           delta=1e-6,
                                           msg=f'header is not packet peak {context}')
                    self.assertLess(packet_peak, 1.0,
                                    f'packet clipped {context}')

                    metadata = v7.decode_metadata(
                        self.model, packet, metadata_start, 1.0, None)
                    self.assertIsNotNone(metadata, context)
                    self.assertEqual(metadata.source_index, counter-1,
                                     context)


if __name__ == '__main__':
    unittest.main()
