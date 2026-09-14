"""Picture geometry declared on the wire, in two bits that were already there.

top_bin is validated to 10..63, so bits 6 and 7 of its header byte have always
been zero. They now carry a profile code, which lets a receiver reconstruct
whatever a transmitter sends instead of both ends needing the same launch
argument. The header does not grow and the CRC already covers it.
"""
import unittest

import numpy as np

from animation_modem import core, transport3 as v3
from animation_modem.imaging import PROFILES, fit_shapes, plane_shapes

WIDE = v3.ALL_PRESETS['wide-v3']        # 2880 slots: every profile fits whole
LEAN = v3.ALL_PRESETS['lean-v3']


def coder_for(name, layout):
    shapes = plane_shapes(name)
    if sum(int(np.prod(s)) for s in shapes) > layout.capacity:
        shapes = fit_shapes(shapes, layout.capacity)
    return v3.SourceCoder(shapes)


def coders_for(layout):
    return {v3.profile_code(n): coder_for(n, layout) for n in v3.PROFILE_CODES}


class ProfileCodeTests(unittest.TestCase):
    def test_the_code_table_matches_the_profiles_that_exist(self):
        """Wire order against the real table. These drifting apart is the one
        way this silently sends the wrong geometry."""
        self.assertEqual(set(v3.PROFILE_CODES), set(PROFILES))
        self.assertLessEqual(len(v3.PROFILE_CODES), 4, 'two bits hold four')

    def test_names_and_codes_round_trip(self):
        for name in v3.PROFILE_CODES:
            self.assertEqual(v3.profile_name(v3.profile_code(name)), name)
        with self.assertRaises(ValueError):
            v3.profile_code('nonexistent-profile')

    def test_the_header_byte_splits_at_its_widest(self):
        """top_bin 63 and profile 3 fill the byte: 0b11111111.

        Packed wrong, the profile's low bit reads as part of the band and the
        header verifies against the wrong layout.
        """
        raw = core.pack_header(0, 63, 1, 1, 1, 0, magic=b'V3', profile=3)
        packed = raw[3]
        self.assertEqual(packed, 0xff)
        self.assertEqual(packed & core.TOP_BIN_MASK, 63)
        self.assertEqual(packed >> 6, 3)

    def test_a_band_too_wide_for_six_bits_is_refused(self):
        """Masking to 0xff would have handed bit 6 to the profile silently."""
        with self.assertRaises(ValueError):
            core.pack_header(0, 64, 1, 1, 1, 0)
        with self.assertRaises(ValueError):
            core.pack_header(0, 54, 1, 1, 1, 0, profile=4)


class ProfileNegotiationTests(unittest.TestCase):
    """The receiver's own default is deliberately wrong throughout."""

    def decode(self, audio, layout, default, coders=None):
        rx = v3.Receiver(layout, default, coders=coders)
        return rx.feed(np.asarray(audio, np.float32))+rx.flush()

    def send(self, name, layout, frames=3):
        coder = coder_for(name, layout)
        values = np.random.default_rng(9).uniform(-.2, .2, coder.count)
        audio = np.concatenate([
            v3.encode(values, layout, coder, n, n, frames,
                      profile=v3.profile_code(name))
            for n in range(1, frames+1)])
        return audio, coder, values

    def test_every_profile_survives_a_receiver_that_assumed_another(self):
        for name in v3.PROFILE_CODES:
            with self.subTest(profile=name):
                audio, coder, values = self.send(name, WIDE)
                wrong = coder_for('detail' if name != 'detail' else 'mono', WIDE)
                out = self.decode(audio, WIDE, wrong, coders_for(WIDE))
                self.assertEqual([r.absolute for r in out], [1, 2, 3])
                for r in out:
                    self.assertEqual(r.extra['profile'], name)
                    self.assertEqual(tuple(r.extra['shapes']), tuple(coder.shapes))
                    self.assertLess(np.sqrt(np.mean((r.values-values)**2)), 1e-4)

    def test_it_works_on_the_live_layout_where_profiles_get_shrunk(self):
        """lean-v3 holds 2200 slots, so 'color' is fit_shapes-shrunk. Both ends
        run the same deterministic shrink, so the code still names it."""
        audio, coder, values = self.send('color', LEAN)
        out = self.decode(audio, LEAN, coder_for('color-lean', LEAN),
                          coders_for(LEAN))
        self.assertEqual(out[0].extra['profile'], 'color')
        self.assertEqual(tuple(out[0].extra['shapes']), tuple(coder.shapes))
        self.assertLess(np.sqrt(np.mean((out[0].values-values)**2)), 1e-4)

    def test_without_the_mapping_nothing_changes(self):
        """Opting in is what makes the receiver follow the header. A caller
        that does not pass coders keeps its own geometry, as before."""
        audio, coder, values = self.send('color-lean', WIDE)
        out = self.decode(audio, WIDE, coder)
        self.assertEqual([r.absolute for r in out], [1, 2, 3])
        self.assertLess(np.sqrt(np.mean((out[0].values-values)**2)), 1e-4)
        # The declaration is still reported, even when it is not acted on.
        self.assertEqual(out[0].extra['profile'], 'color-lean')

    def test_an_undeclared_profile_reads_as_code_zero(self):
        """Guards the compatibility hazard rather than hiding it.

        A transmitter from before this change leaves the bits zero, which is
        indistinguishable from declaring PROFILE_CODES[0]. A receiver holding
        the mapping will believe it. Both ends have to move together; there is
        no spare bit left to express "undeclared".
        """
        coder = coder_for('color-lean', WIDE)
        values = np.random.default_rng(2).uniform(-.2, .2, coder.count)
        # profile=0 is exactly what an old encoder puts on the wire.
        audio = v3.encode(values, WIDE, coder, 1, 1, 1, profile=0)
        out = self.decode(audio, WIDE, coder, coders_for(WIDE))
        self.assertEqual(out[0].extra['profile'], v3.PROFILE_CODES[0])

    def test_a_declared_profile_does_not_loosen_the_band_check(self):
        """The other six bits still have to match the layout."""
        audio, coder, _ = self.send('color', WIDE)
        out = self.decode(audio, v3.ALL_PRESETS['mid-v3'],
                          coder_for('color', v3.ALL_PRESETS['mid-v3']))
        self.assertTrue(all(r.identity != 'verified_header' for r in out))


class LiveGuardTests(unittest.TestCase):
    """Live pins the preset, not the profile.

    Preset is wire format the receiver cannot negotiate -- training layout,
    header placement, band -- and live-receive is fixed on lean-v3, so a
    mismatch decodes nothing. Profile is declared in the header now, so
    refusing one at the transmitter stopped protecting anything.
    """

    def setUp(self):
        import contextlib
        import io
        import os
        import sys
        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        self.quiet = contextlib.redirect_stdout(io.StringIO())
        self.quiet.__enter__()
        self.addCleanup(lambda: self.quiet.__exit__(None, None, None))

    def test_a_v2_preset_is_still_refused_live(self):
        """Generation is the one thing left that cannot be worked out: a v2
        layout puts a different magic on the wire and no v3 candidate is
        looking for it."""
        import modem_screen
        with self.assertRaisesRegex(SystemExit, 'v2 wire format'):
            modem_screen.main(['--source', 'test', '--preset', 'tape'])

    def test_a_progressive_preset_is_allowed_live(self):
        """This is what the old guard blocked: it named lean-v3 alone."""
        import modem_screen
        for name, layout in v3.ALL_PRESETS.items():
            if not layout.progressive:
                continue
            with self.subTest(preset=name):
                try:
                    modem_screen.main(['--source', 'test', '--preset', name,
                                       '--frames', '0'])
                except SystemExit as exc:
                    self.assertNotIn('preset', str(exc))
                except Exception:
                    pass          # anything past the guard is not our concern

    def test_every_profile_is_allowed_live(self):
        """The message named color-lean and this is what it blocked."""
        import modem_screen
        import tempfile
        from pathlib import Path
        for name in v3.PROFILE_CODES:
            with self.subTest(profile=name), tempfile.TemporaryDirectory() as d:
                out = str(Path(d)/'x.wav')
                # --write takes the same guard and needs no audio device.
                modem_screen.main(['--source', 'test', '--profile', name,
                                   '--frames', '1', '--write', out])
                self.assertTrue(Path(out).exists())

    def test_what_is_written_declares_itself(self):
        """A profile too big for lean-v3 is shrunk, and still names itself."""
        import modem_screen
        import tempfile
        import wave
        from pathlib import Path
        with tempfile.TemporaryDirectory() as d:
            out = str(Path(d)/'mono.wav')
            modem_screen.main(['--source', 'test', '--profile', 'mono',
                               '--frames', '2', '--write', out])
            with wave.open(out) as w:
                raw = np.frombuffer(w.readframes(w.getnframes()), '<i2')
            audio = raw.reshape(-1, 2).astype(np.float32)/32768
            rx = v3.Receiver(LEAN, coder_for('color-lean', LEAN),
                             coders=coders_for(LEAN))
            got = rx.feed(audio)+rx.flush()
            self.assertTrue(got)
            for r in got:
                self.assertEqual(r.identity, 'verified_header')
                self.assertEqual(r.extra['profile'], 'mono')
                self.assertEqual(tuple(r.extra['shapes']),
                                 tuple(coder_for('mono', LEAN).shapes))


if __name__ == '__main__':
    unittest.main()
