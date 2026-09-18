"""Picture geometry declared on the wire, in two bits that were already there.

top_bin is validated to 10..63, so bits 6 and 7 of its header byte have always
been zero. They now carry a profile code, which lets a receiver reconstruct
whatever a transmitter sends instead of both ends needing the same launch
argument. The header does not grow and the CRC already covers it.
"""
import unittest
from dataclasses import replace

import numpy as np

from animation_modem import core, transport3 as v3
from animation_modem.imaging import (PROFILE_GRIDS, PROFILES, fit_shapes,
                                     plane_grids, plane_shapes, wire_profiles)

WIDE = v3.WIRE                          # 3280 slots: every profile fits whole
# A deliberately narrow wire, only to keep the fit_shapes-shrink path under
# test: color-dct (2880) does not fit, so both ends must run the same
# deterministic shrink and still agree on the declared geometry.
SMALL = replace(v3.WIRE, top_bin=32, name='small')


def coder_for(name, layout):
    shapes = plane_shapes(name)
    grids = plane_grids(name)
    if sum(int(np.prod(s)) for s in shapes) > layout.capacity:
        shapes = fit_shapes(shapes, layout.capacity)
        grids = shapes
    return v3.SourceCoder(shapes, grids=grids)


def coders_for(layout):
    return {v3.profile_code(n): coder_for(n, layout) for n in v3.PROFILE_CODES}


class ProfileCodeTests(unittest.TestCase):
    def test_the_code_table_matches_the_profiles_that_exist(self):
        """Wire order against the real table. These drifting apart is the one
        way this silently sends the wrong geometry."""
        # Every profile holds its own code now -- no aliases, nothing to opt
        # into, nothing for the two ends to disagree about.
        self.assertEqual(set(v3.PROFILE_CODES), set(wire_profiles()))
        self.assertLessEqual(len(v3.PROFILE_CODES), 4, 'two bits hold four')
        self.assertTrue(set(PROFILE_GRIDS) <= set(PROFILES))

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
        values = np.random.default_rng(9).uniform(-.2, .2, coder.source_count)
        audio = np.concatenate([
            v3.encode(values, layout, coder, n, n, frames,
                      profile=v3.profile_code(name))
            for n in range(1, frames+1)])
        return audio, coder, values

    def test_every_profile_survives_a_receiver_that_assumed_another(self):
        for name in v3.PROFILE_CODES:
            with self.subTest(profile=name):
                audio, coder, values = self.send(name, WIDE)
                other = 'color-dct' if name == 'color-dct' else 'color-dct'
                wrong = coder_for(other, WIDE)
                out = self.decode(audio, WIDE, wrong, coders_for(WIDE))
                self.assertEqual([r.absolute for r in out], [1, 2, 3])
                for r in out:
                    self.assertEqual(r.extra['profile'], name)
                    self.assertEqual(tuple(r.extra['shapes']), tuple(coder.grids))
                    # A truncating profile is LOSSY by construction -- it sends
                    # a corner of the spectrum and zero-fills the rest -- and
                    # these values are uniform noise, which is the worst case
                    # for that: its energy is spread over every frequency, all
                    # of it above the corner discarded. What the wire has to
                    # preserve is the coefficients it carried, not the ones it
                    # deliberately dropped.
                    limit = .25 if coder.truncated else 1e-4
                    self.assertLess(np.sqrt(np.mean((r.values-values)**2)), limit)

    def test_it_works_on_a_narrow_wire_where_profiles_get_shrunk(self):
        """SMALL holds 1696 slots, so 'color-dct' is fit_shapes-shrunk. Both
        ends run the same deterministic shrink, so the code still names it."""
        audio, coder, values = self.send('color-dct', SMALL)
        out = self.decode(audio, SMALL, coder_for('color-dct', SMALL),
                          coders_for(SMALL))
        self.assertEqual(out[0].extra['profile'], 'color-dct')
        self.assertEqual(tuple(out[0].extra['shapes']), tuple(coder.grids))
        self.assertLess(np.sqrt(np.mean((out[0].values-values)**2)), 1e-4)

    def test_without_the_mapping_nothing_changes(self):
        """Opting in is what makes the receiver follow the header. A caller
        that does not pass coders keeps its own geometry, as before."""
        coder = coder_for('color-dct', WIDE)
        # Low-frequency content: everything survives the truncating coder's
        # corner, so the round trip is lossless and the comparison is exact.
        layers = []
        for grid in coder.grids:
            h, w = grid
            yy, xx = np.mgrid[0:h, 0:w]
            layers.append((.15+.1*np.sin(xx/20)+.08*np.cos(yy/24)).ravel())
        values = np.concatenate(layers)
        audio = np.concatenate([
            v3.encode(values, WIDE, coder, n, n, 3,
                      profile=v3.profile_code('color-dct'))
            for n in range(1, 4)])
        out = self.decode(audio, WIDE, coder)
        self.assertEqual([r.absolute for r in out], [1, 2, 3])
        # float32 wire round-trip floor, not truncation: the source is far
        # inside the corner so it survives the DCT losslessly.
        self.assertLess(np.sqrt(np.mean((out[0].values-values)**2)), 1e-3)
        # The declaration is still reported, even when it is not acted on.
        self.assertEqual(out[0].extra['profile'], 'color-dct')

    def test_an_undeclared_profile_reads_as_code_zero(self):
        """Guards the compatibility hazard rather than hiding it.

        A transmitter that leaves the bits zero is indistinguishable from
        declaring PROFILE_CODES[0]. A receiver holding the mapping will believe
        it. Both ends have to move together; there is no spare bit left to
        express "undeclared".
        """
        coder = coder_for('color-dct', WIDE)
        values = np.random.default_rng(2).uniform(-.2, .2, coder.source_count)
        # profile=0 is exactly what an encoder without a declaration puts on
        # the wire.
        audio = v3.encode(values, WIDE, coder, 1, 1, 1, profile=0)
        out = self.decode(audio, WIDE, coder, coders_for(WIDE))
        self.assertEqual(out[0].extra['profile'], v3.PROFILE_CODES[0])

    def test_a_declared_profile_does_not_loosen_the_band_check(self):
        """The other six bits still have to match the layout."""
        audio, coder, _ = self.send('color-dct', WIDE)
        out = self.decode(audio, SMALL, coder_for('color-dct', SMALL))
        self.assertTrue(all(r.identity != 'verified_header' for r in out))


class LiveGuardTests(unittest.TestCase):
    """Live pins the wire, not the profile.

    The wire is fixed -- training layout, header placement, band -- so there
    is nothing left to negotiate and no generation guard. Profile is declared
    in the header, so refusing one at the transmitter would protect nothing.
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

    def test_the_wire_renders_live(self):
        """No generation guard exists any more: the wire is fixed."""
        import modem_screen
        import tempfile
        from pathlib import Path
        with tempfile.TemporaryDirectory() as d:
            # --write takes the same guard path and needs no audio device.
            out = str(Path(d)/'x.wav')
            modem_screen.main(['--source', 'test', '--frames', '1',
                               '--write', out])
            self.assertTrue(Path(out).exists())

    def test_every_profile_is_allowed_live(self):
        """The old refusal named a profile and this is what it blocked."""
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
        """A rendered packet names its profile in the header."""
        import modem_screen
        import tempfile
        import wave
        from pathlib import Path
        with tempfile.TemporaryDirectory() as d:
            out = str(Path(d)/'declared.wav')
            modem_screen.main(['--source', 'test', '--profile', 'color-dct',
                               '--frames', '2', '--write', out])
            with wave.open(out) as w:
                raw = np.frombuffer(w.readframes(w.getnframes()), '<i2')
            audio = raw.reshape(-1, 2).astype(np.float32)/32768
            rx = v3.Receiver(WIDE, coder_for('color-dct', WIDE),
                             coders=coders_for(WIDE))
            got = rx.feed(audio)+rx.flush()
            self.assertTrue(got)
            for r in got:
                self.assertEqual(r.identity, 'verified_header')
                self.assertEqual(r.extra['profile'], 'color-dct')
                self.assertEqual(tuple(r.extra['shapes']),
                                 tuple(coder_for('color-dct', WIDE).grids))


if __name__ == '__main__':
    unittest.main()
