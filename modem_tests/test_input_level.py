"""Normalise what arrives, so the decoder's absolute thresholds still apply.

Acquisition triggers a Schmitt at a FIXED +/-0.066 -- 12% of the preamble's
nominal 0.55. Measured, a signal at 0.12 of nominal decodes every frame and one
at 0.06 finds zero edges and acquires nothing at all. Not a degraded picture:
silence. A deck a few dB down, or a line input padded for a microphone, lands
there.
"""
import unittest

import numpy as np

from animation_modem import transport3 as v3
from animation_modem.audio_common import InputLevel
from animation_modem.imaging import plane_shapes

LAYOUT = v3.ALL_PRESETS['lean-v3']


def transmission(frames=6):
    coder = v3.SourceCoder(plane_shapes('color-lean'))
    values = np.random.default_rng(3).uniform(-.2, .2, coder.count)
    audio = np.concatenate([
        v3.encode(values, LAYOUT, coder, n, n, frames,
                  profile=v3.profile_code('color-lean'))
        for n in range(1, frames+1)])
    return audio, values, coder


def receive(signal, coder, values, level=None):
    rx = v3.Receiver(LAYOUT, coder)
    out = []
    for i in range(0, len(signal), 256):
        block = np.asarray(signal[i:i+256], np.float32)
        out += rx.feed(level.process(block) if level else block)
    out += rx.flush()
    good = [r for r in out if r.identity == 'verified_header']
    error = (np.median([np.sqrt(np.mean((r.values-values)**2)) for r in good])
             if good else float('nan'))
    return len(good), error


class QuietInputTests(unittest.TestCase):
    def test_a_quiet_input_acquires_nothing_without_levelling(self):
        """The premise, measured rather than assumed."""
        audio, values, coder = transmission()
        self.assertEqual(receive(audio*.06, coder, values)[0], 0)
        self.assertEqual(len(v3.edge_intervals(np.asarray(audio*.06, np.float32)[:400, 0])), 0)

    def test_levelling_recovers_it(self):
        audio, values, coder = transmission()
        for gain in (.12, .06, .02, .005):
            with self.subTest(gain=gain):
                frames, error = receive(audio*gain, coder, values, InputLevel())
                self.assertEqual(frames, 6)
                self.assertLess(error, 1e-4)

    def test_a_healthy_input_is_left_exactly_alone(self):
        """`encode` normalises a packet to 0.95, so a good signal is already
        where it should be and any gain would only add drift."""
        audio, values, coder = transmission()
        level = InputLevel()
        for gain in (1.0, .5):
            frames, error = receive(audio*gain, coder, values, InputLevel())
            self.assertEqual(frames, 6)
            self.assertLess(error, 1e-6)      # float32 round trip only
        block = np.asarray(audio[:256]*.95, np.float32)
        np.testing.assert_array_equal(level.process(block), block)
        self.assertEqual(level.gain, 1.0)

    def test_silence_does_not_run_the_gain_away(self):
        level = InputLevel()
        quiet = np.zeros((256, 2), np.float32)
        for _ in range(500):
            level.process(quiet)
        self.assertEqual(level.gain, 1.0)


class BothChannelsTests(unittest.TestCase):
    def test_one_gain_for_the_pair_not_two(self):
        """Levelling the channels separately would flatten the amplitude
        difference between them, which is part of what the 2x2 channel estimate
        solves for and what skew and crosstalk report."""
        level = InputLevel()
        block = np.zeros((256, 2), np.float32)
        block[:, 0] = .05*np.sin(np.linspace(0, 40, 256))
        block[:, 1] = .01*np.sin(np.linspace(0, 40, 256))
        before = np.max(np.abs(block[:, 0]))/np.max(np.abs(block[:, 1]))
        for _ in range(200):
            out = level.process(block)
        after = np.max(np.abs(out[:, 0]))/np.max(np.abs(out[:, 1]))
        self.assertAlmostEqual(after, before, places=4)
        self.assertGreater(level.gain, 2)        # it did boost the pair


class LimiterTests(unittest.TestCase):
    def test_the_output_never_exceeds_the_ceiling(self):
        level = InputLevel()
        rng = np.random.default_rng(1)
        worst = 0.0
        for scale in (.5, 3.0, 12.0, .2, 30.0):
            for _ in range(20):
                block = (rng.normal(0, scale, (256, 2))).astype(np.float32)
                worst = max(worst, float(np.max(np.abs(level.process(block)))))
        self.assertLessEqual(worst, level.ceiling+1e-6)

    def test_the_limiter_catches_what_the_gain_path_will_not(self):
        """A level change big enough to jump to is handled by the gain. The
        limiter is for the band in between, where the step rate deliberately
        refuses to move fast and the block would otherwise come out over 1.0.
        """
        level = InputLevel()
        steady = np.zeros((256, 2), np.float32)
        steady[:, 0] = .5*np.sin(np.linspace(0, 40, 256))
        for _ in range(50):
            level.process(steady)            # settle
        # +6.8 dB: past the do-nothing window so the gain engages, but not far
        # enough to trigger the jump, so the step rate holds it back and the
        # block would come out at 1.10 without the limiter.
        louder = steady*2.2
        out = level.process(louder)
        self.assertLessEqual(float(np.max(np.abs(out))), level.ceiling+1e-6)
        self.assertGreater(level.limited, 0)

    def test_it_cannot_undo_clipping_that_already_happened(self):
        """Worth stating: if the interface clipped, the damage is upstream of
        anything here. Levelling neither helps nor hurts that signal."""
        audio, values, coder = transmission()
        clipped = np.clip(audio*2.5, -1, 1)
        plain = receive(clipped, coder, values)
        levelled = receive(clipped, coder, values, InputLevel())
        self.assertEqual(plain[0], levelled[0])
        self.assertAlmostEqual(plain[1], levelled[1], places=3)


class StabilityTests(unittest.TestCase):
    def test_the_gain_barely_moves_across_one_packet(self):
        """A packet's channel estimate comes from its training symbols, so a
        gain that drifts within one scales the coefficients against an estimate
        taken at a different level."""
        level = InputLevel()
        block = np.asarray(np.random.default_rng(2).normal(0, .01, (256, 2)),
                           np.float32)
        for _ in range(400):                      # converge first
            level.process(block)
        settled = level.gain
        for _ in range(LAYOUT.frame//256 + 1):    # one packet's worth
            level.process(block)
        drift = 20*np.log10(max(level.gain, 1e-12)/max(settled, 1e-12))
        self.assertLess(abs(drift), .3)


class OnLossDefaultTests(unittest.TestCase):
    def test_damaged_is_the_default(self):
        """A picture whose header did not verify has still been reconstructed.
        Holding the last good frame hides it, which is what a deck rolling off
        looks like: a black screen over a working decoder."""
        from utilities import modem_v3_check as check
        from unittest.mock import patch
        with patch.object(check, 'do_live_receive') as receive_stub:
            check.main(['live-receive', '--device', '0', '--headless'])
        self.assertEqual(receive_stub.call_args.args[0].on_loss, 'damaged')


if __name__ == '__main__':
    unittest.main()
