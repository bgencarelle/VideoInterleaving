"""Aspect-matched mono colour Fold 500 (experimental aspect-mono-500 profile)."""
import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
for extra in (ROOT/'test_modem_v7', ROOT/'tools'):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))

from animation_modem import v7                                           # noqa: E402
from common import TARGET                                                # noqa: E402
from aspect_fold import (LAYOUT_NAMES, assemble_model, base_tables,     # noqa: E402
                         layout_positions)
from aspect_mono import PROFILE, STATUS_MODE, AspectMonoWire              # noqa: E402
from mono_video import (FOLD_SLOTS, FRESH_SLOTS, MONO_COLOUR_FOLD_D,     # noqa: E402
                        MONO_COLOUR_FOLD_TABLE_SHA256, MonoColourFoldWire)
import tone_code                                                         # noqa: E402
from tools import v7_live                                                # noqa: E402
from modem_tests.test_v7_aspect_fold import _text_frame                 # noqa: E402

PACKETS = 5


def _decode(wire, model, audio):
    with wire.receiving():
        results, info = v7.decode_pulse_stream(
            model, audio, sample_rate=v7.RATE, pilot_timing='tone-seeded', state=v7.PulseState(tail_memory=False))
    return [result for result in results if result.status != 'lost'], info


class AspectMonoTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.base = v7.load_model(TARGET, 'box')
        cls.frame = _text_frame()
        cls.aspect = v7.aspect_wire_code(cls.frame.size)
        cls.values = v7_live._values(cls.base, cls.frame, 'box',
                                     brightness=1.0)[0]
        cls.wire = AspectMonoWire(cls.base)

    def _luma_error(self, values):
        return float(np.mean((values[:96*80]-self.values[:96*80])**2))

    def test_signalled_as_mono_off_with_the_mono_colour_wire(self):
        self.assertEqual(STATUS_MODE, tone_code.MONO_OFF)
        self.assertEqual(self.wire.status_mode, tone_code.MONO_OFF)
        self.assertEqual(self.wire.wire_profile, PROFILE)
        self.assertTrue(self.wire.disable_tail_memory)

    def test_every_layout_sends_one_fixed_fresh_set_and_folds_luma(self):
        identities = set()
        for layout in LAYOUT_NAMES:
            model = self.wire.model_for(self.base, layout)
            codec = self.wire._codec(model)
            with self.subTest(layout=layout):
                np.testing.assert_array_equal(model.coder.positions,
                                              layout_positions(layout)[0])
                tables = model.rank_tables
                for ranks in tables[1:]:
                    np.testing.assert_array_equal(ranks, tables[0])
                sent = tables[0][tables[0] >= 0]
                self.assertEqual(len(sent), FRESH_SLOTS)
                plane = np.asarray(model.plane)
                self.assertEqual(len(codec.hosts), FOLD_SLOTS)
                self.assertTrue(np.all(plane[codec.hosts] == 0))
                self.assertTrue(np.all(plane[codec.guest_model_indices] == 0))
                self.assertTrue(np.isin(codec.hosts, sent).all())
                self.assertFalse(np.isin(codec.guest_model_indices, sent).any())
                self.assertEqual(codec.D, MONO_COLOUR_FOLD_D)
                np.testing.assert_allclose(model.scale, assemble_model(
                    self.base, base_tables(layout)).scale*np.sqrt(2))
                identities.add(codec.identity)
        self.assertEqual(len(identities), len(LAYOUT_NAMES))
        self.assertNotIn(MONO_COLOUR_FOLD_TABLE_SHA256, identities)

    def test_packets_follow_the_aspect_code_and_improve_widescreen_luma(self):
        audio = self.wire.encode(self.base, [self.values]*PACKETS,
                                 aspect_codes=[self.aspect]*PACKETS)
        model = self.wire._packet_model(self.base, self.aspect)
        self.assertIs(model, self.wire.model_for(self.base, '16:9'))
        good, info = _decode(self.wire, model, audio)
        self.assertEqual(len(good), PACKETS, info)
        self.assertTrue(all(
            result.diag['pilot_timing']['coded_status_mode'] == STATUS_MODE
            for result in good))
        aspect_values = self.wire.values(model, good[-1])
        codec = self.wire._codec(model)
        self.assertGreater(codec.last_score, .8)
        self.assertEqual(codec.last_unfolded_slots, FOLD_SLOTS)

        colour = MonoColourFoldWire(self.base)
        colour_model = colour.model_for(self.base)
        colour_good, _ = _decode(colour, colour_model, colour.encode(
            self.base, [self.values]*PACKETS))
        colour_values = colour.values(colour_model, colour_good[-1])
        self.assertLess(self._luma_error(aspect_values),
                        .95*self._luma_error(colour_values))

    def test_forced_layout_ignores_the_aspect_code(self):
        wire = AspectMonoWire(self.base, side='right', layout='4:3')
        self.assertEqual(wire.layout_for(self.aspect), '4:3')
        self.assertEqual(wire.layout_for(None), '4:3')
        self.assertIsNone(self.wire.layout_for(None))
        audio = wire.encode(self.base, [self.values]*2,
                            aspect_codes=[self.aspect]*2)
        self.assertTrue(np.all(audio[:, 0] == 0.0))
        self.assertGreater(float(np.max(np.abs(audio[:, 1]))), 0.0)

    def test_sender_cli_maps_to_the_mono_video_path(self):
        args = v7_live.parser().parse_args([
            'send', '--device', 'null', '--source', 'test',
            '--profile', 'aspect-mono-500', '--aspect-layout', '16:9',
            '--dct-encode'])
        v7_live._apply_profile_option(args)
        self.assertTrue(args.experimental_mono_fold)
        self.assertTrue(args.aspect_mono)
        self.assertEqual(v7_live._fold_slots(args), 0)
        self.assertEqual(v7_live._send_profile(args, 500)[0], 'box')

    def test_adaptive_receiver_decodes_the_packet_aspect_layout(self):
        from tone_code import coded_pilot_timing

        fold = v7_live._experimental_fold(500)
        profile = v7_live._AdaptiveProfileDecoder(fold, self.base)
        self.assertIn(STATUS_MODE, profile.mono_status_modes)
        self.assertEqual(profile._mode_names[STATUS_MODE], PROFILE)
        sender = AspectMonoWire(self.base, side='right')
        audio = sender.encode(self.base, [self.values]*4,
                              aspect_codes=[self.aspect]*4)[:, 1:2]
        profile.install()
        try:
            with coded_pilot_timing():
                profile.active_mode = profile.dispatch_mode = STATUS_MODE
                results, _ = v7.decode_pulse_stream(
                    self.base, audio, latest_only=True,
                    state=v7.PulseState(tail_memory=False),
                    sample_rate=v7.RATE, pilot_timing='tone-seeded')
        finally:
            profile.uninstall()
        result = results[-1]
        self.assertIn(result.status, ('received', 'verified'))
        self.assertEqual(result.diag['wire_profile'], PROFILE)
        self.assertEqual(result.diag['aspect_layout'], '16:9')
        self.assertNotIn('aspect_tail', result.diag)
        values = profile.values(self.base, result)
        self.assertLess(self._luma_error(values),
                        .5*float(np.mean(self.values[:96*80]**2)))


if __name__ == '__main__':
    unittest.main()
