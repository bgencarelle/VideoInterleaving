"""Regression checks for the V7 sender's legacy clock selection edges."""
import contextlib
import io
import sys
import types
import unittest
from unittest.mock import patch

import index_calculator
from modem_v7_display import _open_midi_input


class ModemClockTests(unittest.TestCase):
    def test_mtc_zero_is_an_explicit_clock_selection(self):
        fake_midi = types.ModuleType('midi_control')
        previous = (index_calculator.clock_mode, index_calculator.midi_mode,
                    index_calculator.midi_control)
        try:
            with patch.dict(sys.modules, {'midi_control': fake_midi}), \
                    patch('builtins.input', side_effect=AssertionError(
                        'explicit clock selection must not prompt')), \
                    contextlib.redirect_stdout(io.StringIO()):
                index_calculator.set_clock_mode(0)
            self.assertEqual(index_calculator.clock_mode, 0)
            self.assertTrue(index_calculator.midi_mode)
            self.assertIs(index_calculator.midi_control, fake_midi)
        finally:
            (index_calculator.clock_mode, index_calculator.midi_mode,
             index_calculator.midi_control) = previous

    def test_live_midi_clock_opens_an_input_and_rejects_none(self):
        port = types.SimpleNamespace(close=lambda: None)
        calls = []
        fake_midi = types.SimpleNamespace(
            mido=types.SimpleNamespace(get_input_names=lambda: ['input']),
            input_port=port,
            midi_control_stuff_main=lambda: calls.append('open'))
        clock = types.SimpleNamespace(midi_mode=True, midi_control=fake_midi)
        self.assertIs(_open_midi_input(clock), port)
        self.assertEqual(calls, ['open'])

        fake_midi.mido.get_input_names = lambda: []
        with self.assertRaisesRegex(RuntimeError, 'no MIDI input'):
            _open_midi_input(clock)


if __name__ == '__main__':
    unittest.main()
