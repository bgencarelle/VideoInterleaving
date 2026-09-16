"""PROFILES and PROFILE_CODES have to agree, and the senders have to import.

modem_screen.py imported `image_values_dct` from animation_modem.imaging,
which did not define it. That is an ImportError at module load -- the sender
could not start at all, on any profile, and no test noticed because nothing
imported modem_screen. There is now one that does, in a subprocess.

The profile table is the other half. Two header bits hold four codes; two are
spent -- color-dct, lean-dct -- and two are reserved. Reassigning one is a WIRE
BREAK, because a recording carries the number and not its meaning and the CRC
covers the number -- so it is done deliberately or not at all.
"""
import subprocess
import sys
import unittest
from pathlib import Path

import numpy as np

from animation_modem import transport3 as v3
from animation_modem.imaging import (PROFILES, fit_shapes, plane_grids,
                                     plane_shapes)

ROOT = Path(__file__).resolve().parent.parent


class SenderImportTests(unittest.TestCase):
    """Import in a subprocess: a bare `import` here would be cached by whatever
    ran first, and modem_screen pulls in optional capture dependencies."""

    def test_modem_screen_imports(self):
        out = subprocess.run(
            [sys.executable, '-c', 'import modem_screen; print(modem_screen.__name__)'],
            cwd=ROOT, capture_output=True, text=True)
        self.assertEqual(out.returncode, 0, out.stderr)

    def test_every_name_the_senders_import_from_imaging_exists(self):
        import ast
        missing = []
        for path in (ROOT/'modem_screen.py', ROOT/'utilities'/'modem_v3_check.py',
                     ROOT/'modem_display.py'):
            if not path.exists():
                continue
            tree = ast.parse(path.read_text())
            for node in ast.walk(tree):
                if (isinstance(node, ast.ImportFrom)
                        and node.module == 'animation_modem.imaging'):
                    import animation_modem.imaging as imaging
                    for alias in node.names:
                        if not hasattr(imaging, alias.name):
                            missing.append(f'{path.name}: {alias.name}')
        self.assertEqual(missing, [], f'imported but not defined: {missing}')


class ProfileCodeTests(unittest.TestCase):
    def test_every_profile_holds_its_own_code(self):
        """Not an alias. Two header bits hold four codes and two are spent, so
        each profile is named on the wire and the receiver reads it rather than
        being told out of band."""
        self.assertEqual(v3.PROFILE_CODES[0], 'color-dct')
        self.assertEqual(v3.profile_name(0), 'color-dct')
        self.assertEqual(v3.profile_code('color-dct'), 0)
        self.assertEqual(v3.profile_code('lean-dct'), 1)

    def test_it_is_the_default(self):
        import modem_screen
        from animation_modem.imaging import DEFAULT_PROFILE
        self.assertEqual(DEFAULT_PROFILE, 'color-dct')
        self.assertEqual(modem_screen.parser().parse_args([]).profile, 'color-dct')

    def test_build_accepts_it_and_gives_the_coder_the_finer_grid(self):
        import argparse
        import modem_screen
        args = argparse.Namespace(profile='color-dct', preset='wide-v3')
        layout, coder, grids = modem_screen.build(args)
        self.assertTrue(coder.truncated)
        self.assertEqual(coder.count, 2880)
        self.assertEqual(coder.source_count, 11520)
        self.assertEqual(tuple(grids[0]), (96, 80))

    def test_every_transmittable_profile_round_trips(self):
        for name in v3.PROFILE_CODES:
            with self.subTest(profile=name):
                self.assertIn(name, PROFILES)
                code = v3.profile_code(name)
                self.assertEqual(v3.profile_name(code), name)

    def test_the_header_field_cannot_name_a_third_profile(self):
        """Two codes spent, two reserved. The field is two bits and a code above
        1 must read back as None rather than wrapping to a real profile, which
        is what a recording carrying the number would decode to."""
        self.assertEqual(len(v3.PROFILE_CODES), 2)
        self.assertIsNone(v3.profile_name(2))
        self.assertIsNone(v3.profile_name(3))


class TruncationTests(unittest.TestCase):
    def test_truncating_does_not_change_the_slot_count(self):
        """Truncation changes the SOURCE, not the wire. Same slot count, same
        plane shapes, four times the pixels behind them -- which is also why it
        is not a bandwidth saving on its own."""
        wire = v3.SourceCoder(plane_shapes('color-dct'),
                              grids=plane_grids('color-dct'))
        plain = v3.SourceCoder(plane_shapes('color-dct'))
        self.assertEqual(wire.count, plain.count)
        self.assertEqual(wire.source_count, 4*plain.source_count)

    def test_the_transform_happens_exactly_once(self):
        """Pre-transforming outside the coder and passing coefficients in
        transforms them twice. forward() of an image is SPARSE; forward() of
        something already transformed is dense, because the energy gets smeared
        back across every slot -- which is what destroys the match between the
        allocation table and the data."""
        coder = v3.SourceCoder(plane_shapes('color-dct'),
                               grids=plane_grids('color-dct'))
        rng = np.random.default_rng(4)
        flat = np.full(coder.source_count, .25)
        sparse = coder.forward(flat)
        self.assertLess(float(np.mean(np.abs(sparse) > 1e-9)), .01)

    def test_a_plain_profile_is_count_preserving(self):
        """grids defaults to shapes, so a coder built without grids transmits
        the wire shapes directly."""
        shapes = plane_shapes('lean-dct')
        coder = v3.SourceCoder(shapes)
        self.assertFalse(coder.truncated)
        self.assertEqual(coder.count, sum(int(np.prod(s)) for s in shapes))
        self.assertEqual(coder.source_count, coder.count)
        values = np.random.default_rng(4).uniform(-.5, .5, coder.count)
        back = coder.inverse(coder.forward(values))
        self.assertLess(float(np.max(np.abs(back-values))), 1e-9)

    def test_a_truncated_packet_round_trips_through_its_own_grid(self):
        coder = v3.SourceCoder(plane_shapes('color-dct'),
                               grids=plane_grids('color-dct'))
        rng = np.random.default_rng(5)
        # a smooth field, so it survives having its high frequencies dropped
        y, x = np.mgrid[0:96, 0:80]/96
        plane = (.4*np.sin(3*x)+.3*np.cos(2*y)).ravel()
        chroma = np.zeros(48*40)
        values = np.concatenate([plane, chroma, chroma])
        back = coder.inverse(coder.forward(values))
        self.assertLess(float(np.sqrt(np.mean((back-values)**2))), .02)


if __name__ == '__main__':
    unittest.main()


class WireDeclarationTests(unittest.TestCase):
    """A truncating profile on the wire, read back off the header.

    The declaration travels with the packet, so the receiver is TOLD which
    profile was sent and reconstructs on the matching grid. Encoding with a
    grid and decoding without one is the one mismatched pair the receiver
    cannot catch, because the allocation table matched the grid.
    """

    def send(self, profile='color-dct'):
        from animation_modem.imaging import image_values
        layout = v3.ALL_PRESETS['wide-v3']
        coder = v3.SourceCoder(plane_shapes(profile),
                               grids=plane_grids(profile))
        y, x = np.mgrid[0:coder.grids[0][0], 0:coder.grids[0][1]]
        y, x = y/coder.grids[0][0], x/coder.grids[0][1]
        luma = (.5*np.sin(4*x)+.3*np.cos(3*y)).ravel()
        chroma = np.zeros(int(np.prod(coder.grids[1])))
        values = np.concatenate([luma, chroma, chroma])
        audio = v3.encode(values, layout, coder, 1, 1, 1,
                          profile=v3.profile_code(profile))
        return layout, audio, values

    def decode(self, layout, audio, coder):
        rx = v3.Receiver(layout, coder)
        out = rx.feed(np.asarray(audio, np.float32))+rx.flush()
        return [r for r in out if r.values is not None]

    def test_a_plain_receiver_decodes_but_gets_it_WRONG(self):
        """The hazard, pinned so nobody documents this as compatibility.

        Nothing malfunctions -- same slots, same shapes, header verifies, a
        picture comes out. But the allocation was built against the sampling
        grid, so the picture is mis-exposed and frequency-distorted. That is
        why a truncating profile is shared state like --allocation.
        """
        layout, audio, values = self.send()
        plain = v3.SourceCoder(plane_shapes('color-dct'))
        got = self.decode(layout, audio, plain)
        self.assertEqual(len(got), 1)
        self.assertEqual(len(got[0].values), plain.count)
        self.assertEqual(tuple(got[0].extra['shapes'][0]), (48, 40))
        wide = v3.SourceCoder(plane_shapes('color-dct'),
                              grids=plane_grids('color-dct'))
        right = self.decode(layout, audio, wide)[0].values
        small = got[0].values[:48*40].reshape(48, 40)
        proper = right[:96*80].reshape(96, 80).reshape(48, 2, 40, 2).mean((1, 3))
        self.assertGreater(abs(float(np.std(small))/float(np.std(proper))-1), .1)

    def test_an_opted_in_receiver_gets_the_finer_grid(self):
        layout, audio, values = self.send()
        wide = v3.SourceCoder(plane_shapes('color-dct'),
                              grids=plane_grids('color-dct'))
        got = self.decode(layout, audio, wide)
        self.assertEqual(len(got), 1)
        self.assertEqual(len(got[0].values), wide.source_count)
        self.assertEqual(tuple(got[0].extra['shapes'][0]), (96, 80))
        self.assertLess(float(np.sqrt(np.mean((got[0].values-values)**2))), .02)

    def test_a_whole_packet_scale_cannot_be_signalled(self):
        """Why the mismatch above cannot simply be corrected away. encode
        normalises the packet and the receiver's channel estimate divides that
        back out, so any uniform gain applied to every coefficient is absorbed
        end to end and arrives as exactly nothing."""
        from animation_modem.imaging import image_values
        layout = v3.ALL_PRESETS['wide-v3']
        coder = v3.SourceCoder(plane_shapes('color-dct'))
        values = np.random.default_rng(8).uniform(-.3, .3, coder.source_count)
        plain = self.decode(layout, v3.encode(values, layout, coder, 1, 1, 1), coder)
        halved = self.decode(
            layout, v3.encode(values*.5, layout, coder, 1, 1, 1), coder)
        a, b = plain[0].values, halved[0].values
        self.assertLess(float(np.max(np.abs(a/np.std(a)-b/np.std(b)))), .05)
