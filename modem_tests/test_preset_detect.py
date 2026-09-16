"""Work the preset out of the signal instead of being told it.

Profile could go in the header. Preset cannot: the header carries top_bin, but
you need the layout to know where the header IS, so reading it first is
circular. The preamble is the one part that does not depend on the layout, so
acquisition still works without knowing it -- and from there the receiver
decodes against each candidate and lets the header CRC decide.

A wrong layout reshapes the body wrongly, so the 32-bit CRC fails on top of a
magic and a top_bin that also have to agree. One attempt costs about 1.4 ms and
the search is paid once per lock, not per frame.
"""
import unittest

import numpy as np

from animation_modem import transport3 as v3
from animation_modem.imaging import fit_shapes, plane_shapes

PROGRESSIVE = [n for n, l in v3.ALL_PRESETS.items() if l.progressive]
LEAN = v3.ALL_PRESETS['lean-v3']


def coder_for(profile, layout):
    shapes = plane_shapes(profile)
    if sum(int(np.prod(s)) for s in shapes) > layout.capacity:
        shapes = fit_shapes(shapes, layout.capacity)
    return v3.SourceCoder(shapes)


def candidates():
    out = []
    for name in PROGRESSIVE:
        layout = v3.ALL_PRESETS[name]
        coders = {v3.profile_code(p): coder_for(p, layout)
                  for p in v3.PROFILE_CODES}
        out.append((layout, coders[v3.profile_code('lean-dct')], coders))
    return out


def send(preset, profile, frames=3):
    layout = v3.ALL_PRESETS[preset]
    coder = coder_for(profile, layout)
    values = np.random.default_rng(5).uniform(-.2, .2, coder.count)
    audio = np.concatenate([
        v3.encode(values, layout, coder, n, n, frames,
                  profile=v3.profile_code(profile))
        for n in range(1, frames+1)])
    return np.asarray(audio, np.float32), values


def receiver(**kw):
    """Always built on lean-v3/lean-dct, so it is wrong for most senders."""
    return v3.Receiver(LEAN, coder_for('lean-dct', LEAN), **kw)


class PresetDetectionTests(unittest.TestCase):
    def test_every_progressive_preset_identifies_itself(self):
        for preset in PROGRESSIVE:
            for profile in v3.PROFILE_CODES:
                with self.subTest(preset=preset, profile=profile):
                    audio, values = send(preset, profile)
                    rx = receiver(candidates=candidates())
                    out = rx.feed(audio)+rx.flush()
                    good = [r for r in out if r.identity == 'verified_header']
                    self.assertEqual(len(good), 3)
                    self.assertEqual(rx.detected, preset)
                    for r in good:
                        self.assertEqual(r.extra['preset'], preset)
                        self.assertEqual(r.extra['profile'], profile)
                        self.assertLess(
                            np.sqrt(np.mean((r.values-values)**2)), 1e-4)

    def test_the_search_is_paid_once_not_per_frame(self):
        """Adopting the layout is the point; re-searching every packet would
        cost 1.4 ms x candidates x frame rate forever."""
        audio, _ = send('mid-v3', 'lean-dct', frames=6)
        rx = receiver(candidates=candidates())
        calls = []
        real = rx._demodulate

        def counted(begin, scale, layout, coder, coders):
            calls.append(layout.name)
            return real(begin, scale, layout, coder, coders)
        rx._demodulate = counted
        out = rx.feed(audio)+rx.flush()
        self.assertEqual(len([r for r in out if r.identity == 'verified_header']), 6)
        # Five more packets after the one that identified the preset, each
        # decoded exactly once, against the layout that was adopted.
        self.assertEqual(calls[-5:], ['mid-v3']*5)
        self.assertLessEqual(len(calls), len(PROGRESSIVE)+5)

    def test_a_short_recording_still_decodes(self):
        """Regression. Waiting for the longest candidate before trying any of
        them meant tape-v3's 6192-sample packet set a floor on the whole
        receiver: a one- or two-frame lean-v3 file (2768 and 5536 samples)
        decoded NOTHING, silently, while three frames worked. The shortest
        candidate is what decides when decoding can start."""
        longest = max(l.packet for l, _, _ in candidates())
        shortest = min(l.packet for l, _, _ in candidates())
        self.assertLess(shortest*2, longest, 'the gap this test exists for')
        for frames in (1, 2, 3):
            with self.subTest(frames=frames):
                audio, values = send('lean-v3', 'lean-dct', frames=frames)
                self.assertLess(len(audio), longest if frames < 3 else 1 << 30)
                rx = receiver(candidates=candidates())
                out = rx.feed(audio)+rx.flush()
                good = [r for r in out if r.identity == 'verified_header']
                self.assertEqual(len(good), frames)
                self.assertEqual(rx.detected, 'lean-v3')
                for r in good:
                    self.assertLess(
                        np.sqrt(np.mean((r.values-values)**2)), 1e-4)

    def test_a_candidate_is_not_retried_while_waiting(self):
        """Feeding in small blocks must not re-run the candidates already
        ruled out at the same position."""
        audio, _ = send('lean-v3', 'lean-dct', frames=2)
        rx = receiver(candidates=candidates())
        calls = []
        real = rx._demodulate

        def counted(begin, scale, layout, coder, coders):
            calls.append(layout.name)
            return real(begin, scale, layout, coder, coders)
        rx._demodulate = counted
        out = []
        for i in range(0, len(audio), 256):       # small blocks, many retries
            out += rx.feed(audio[i:i+256])
        out += rx.flush()
        self.assertEqual(len([r for r in out if r.identity == 'verified_header']), 2)
        self.assertLessEqual(calls.count('lean-v3'), 2 + 1)

    def test_without_candidates_nothing_changes(self):
        """The search is opt-in. A receiver given none behaves as before and
        simply fails to read a preset it was not built for."""
        audio, _ = send('mid-v3', 'lean-dct')
        rx = receiver()
        out = rx.feed(audio)+rx.flush()
        self.assertIsNone(rx.detected)
        self.assertTrue(all(r.identity != 'verified_header' for r in out))

    def test_noise_identifies_nothing(self):
        rx = receiver(candidates=candidates())
        noise = np.random.default_rng(3).normal(0, .05, (40000, 2)).astype(np.float32)
        out = rx.feed(noise)+rx.flush()
        self.assertIsNone(rx.detected)
        self.assertTrue(all(r.identity != 'verified_header' for r in out))


class DocumentationTests(unittest.TestCase):
    """Catch prose that still asserts a contract the code has dropped.

    Behaviour changed three times here -- rate, profile, preset -- and each
    time the docstrings and --help text went on promising that both ends had
    to be told. Wrong documentation is worse than none: it sends someone
    looking for an argument that no longer exists.
    """

    RETIRED = [
        ('must name the same', 'presets are now identified from the signal'),
        ('not interchangeable', 'presets are now identified from the signal'),
        ('must still match at both endpoints', 'only --allocation still must'),
        ('no longer accepts preset/profile', 'it never needed them; say why'),
        ('Live transport uses lean-v3 / color-lean', 'any progressive preset is live'),
        ('lean-v3 / color-lean', 'not a requirement any more, only a default'),
    ]

    def files(self):
        import pathlib
        root = pathlib.Path(__file__).resolve().parent.parent
        for name in ('animation_modem/core.py', 'animation_modem/transport3.py',
                     'utilities/modem_v3_check.py', 'modem_screen.py',
                     'modem_display.py', 'docs/LIVE_MODEM.md'):
            yield root/name

    def test_no_file_still_promises_matched_arguments(self):
        for path in self.files():
            text = path.read_text()
            for phrase, why in self.RETIRED:
                with self.subTest(file=path.name, phrase=phrase):
                    self.assertNotIn(phrase, text, f'{path.name}: {why}')

    def test_the_help_text_says_the_receiver_reads_it(self):
        """The two receiving arguments must describe themselves as fallbacks,
        or someone will keep setting them and wonder why they do nothing."""
        import io
        import contextlib
        from utilities import modem_v3_check as check
        out = io.StringIO()
        with contextlib.redirect_stdout(out), self.assertRaises(SystemExit):
            check.main(['read', '--help'])
        text = ' '.join(out.getvalue().split())
        self.assertIn('Fallback only', text)
        self.assertIn('identified from the signal', text)
        self.assertIn('read from the header', text)


if __name__ == '__main__':
    unittest.main()
