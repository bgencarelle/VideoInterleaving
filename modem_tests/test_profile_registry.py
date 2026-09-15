"""PROFILES and PROFILE_CODES have to agree, and the senders have to import.

Two bugs this pins down, both live in 627bfbc6:

modem_screen.py imported `image_values_dct` from animation_modem.imaging,
which does not define it. That is an ImportError at module load -- the sender
could not start at all, on any profile, and no test noticed because nothing
imported modem_screen.

`color-dct` sits in PROFILES with no entry in PROFILE_CODES. argparse accepts
it from --profile, build() gets a coder for it, and the crash arrives later at
V3.encode with the audio device already open.
"""
import subprocess
import sys
import unittest
from pathlib import Path

import numpy as np

from animation_modem import transport3 as v3
from animation_modem.imaging import PROFILES, fit_shapes, plane_shapes

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
    def test_the_parser_will_not_accept_a_bake_only_profile(self):
        import modem_screen
        self.assertNotIn('color-dct', modem_screen.parser()
                         .parse_args([]).__dict__.get('profile', ''))
        with self.assertRaises(SystemExit):
            modem_screen.parser().parse_args(['--profile', 'color-dct'])

    def test_build_refuses_it_too_for_a_programmatic_caller(self):
        """Belt and braces: argparse is not the only way into build(), and the
        failure must not wait for the first encode with the device open."""
        import argparse
        import modem_screen
        args = argparse.Namespace(profile='color-dct', preset='lean-v3')
        with self.assertRaises(SystemExit) as caught:
            modem_screen.build(args)
        self.assertIn('color-dct', str(caught.exception))

    def test_every_transmittable_profile_round_trips(self):
        for name in v3.PROFILE_CODES:
            with self.subTest(profile=name):
                self.assertIn(name, PROFILES)
                code = v3.profile_code(name)
                self.assertEqual(v3.profile_name(code), name)

    def test_the_header_field_cannot_name_a_fifth_profile(self):
        """Why color-dct has no code, stated so a future reader does not have
        to rediscover it: the field is two bits, and both the flags byte and
        the rest of the top_bin byte are spoken for."""
        self.assertEqual(len(v3.PROFILE_CODES), 4)
        for code in range(4, 8):
            self.assertEqual(v3.profile_name(code), v3.PROFILE_CODES[code & 3])


class ShrinkTests(unittest.TestCase):
    def test_color_dct_would_decode_smaller_than_color_lean(self):
        """The trap, as a number. A profile named for more resolution that
        delivers less, because fit_shapes has to reach the layout's budget."""
        layout = v3.ALL_PRESETS['lean-v3']
        fitted = fit_shapes(plane_shapes('color-dct'), layout.capacity)
        lean = plane_shapes('color-lean')
        self.assertLess(int(np.prod(fitted[0])), int(np.prod(lean[0])))
        self.assertEqual((fitted[0][1], fitted[0][0]), (34, 42))

    def test_the_dct_transmits_every_coefficient(self):
        """SourceCoder decorrelates, it does not compress. One wire slot per
        source coefficient, which is why a bigger bake buys nothing on its own.
        """
        shapes = plane_shapes('color-lean')
        coder = v3.SourceCoder(shapes)
        self.assertEqual(coder.count, sum(int(np.prod(s)) for s in shapes))
        values = np.random.default_rng(4).uniform(-.5, .5, coder.count)
        self.assertEqual(coder.forward(values).shape, (coder.count,))
        back = coder.inverse(coder.forward(values))
        self.assertLess(float(np.max(np.abs(back-values))), 1e-9)


if __name__ == '__main__':
    unittest.main()
