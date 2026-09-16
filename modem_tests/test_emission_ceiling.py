"""Getting the whole signal under a ceiling, not just the carriers.

top_bin decides where the information sits. It does not decide what leaves the
DAC, because the preamble is biphase-mark -- square edges, harmonics to
Nyquist. Every preset emits essentially the same tail regardless of its band,
which is the thing that makes "lower the band" insufficient on its own for a
channel with a hard ceiling.
"""
import unittest

import numpy as np

from animation_modem import transport3 as v3
from animation_modem.core import (N, REFERENCE_RATE as RATE, bound_emission)
from animation_modem.imaging import fit_shapes, plane_shapes

BIN_HZ = RATE/N


def occupied(signal, fraction=.9999, rate=RATE):
    """Frequency below which `fraction` of the energy sits."""
    x = np.asarray(signal, float)
    if x.ndim > 1:
        x = x[:, 0]
    power = np.abs(np.fft.rfft(x*np.hanning(len(x))))**2
    freqs = np.fft.rfftfreq(len(x), 1/rate)
    return freqs[np.searchsorted(np.cumsum(power)/np.sum(power), fraction)]


def coder_for(profile, layout):
    shapes = plane_shapes(profile)
    if sum(int(np.prod(s)) for s in shapes) > layout.capacity:
        shapes = fit_shapes(shapes, layout.capacity)
    return v3.SourceCoder(shapes), shapes


def transmission(layout, profile='color-dct', frames=4):
    coder, shapes = coder_for(profile, layout)
    values = np.random.default_rng(6).uniform(-.2, .2, coder.count)
    audio = np.concatenate([
        v3.encode(values, layout, coder, n, n, frames,
                  profile=v3.profile_code(profile))
        for n in range(1, frames+1)])
    return coder, values, audio


def decode(layout, coder, signal, values):
    rx = v3.Receiver(layout, coder)
    out = []
    signal = np.asarray(signal, np.float32)
    for i in range(0, len(signal), 256):
        out += rx.feed(signal[i:i+256])
    out += rx.flush()
    good = [r for r in out if r.identity == 'verified_header']
    error = (float(np.median([np.sqrt(np.mean((r.values-values)**2))
                              for r in good])) if good else float('nan'))
    return len(good), sum(1 for r in out if r.values is not None), error


class PreambleTailTests(unittest.TestCase):
    def test_a_low_band_preset_still_emits_past_20_kHz(self):
        """The premise. tape-v3's carriers stop at 10125 Hz and it still puts
        99.99% of its energy out past 20 kHz, because the preamble does not
        care what top_bin is."""
        layout = v3.ALL_PRESETS['tape-v3']
        self.assertLess(layout.band_at(RATE)[1], 10500)
        coder, values, audio = transmission(layout, 'lean-dct')
        self.assertGreater(occupied(audio), 20000)

    def test_the_tail_is_the_preamble_not_the_carriers(self):
        """Same measurement on the preamble alone, so the attribution is not
        an inference."""
        self.assertGreater(occupied(np.column_stack([v3.PREAMBLE]*2), .99), 20000)


class CeilingTests(unittest.TestCase):
    def test_mid_14k_under_a_14_kHz_ceiling(self):
        layout = v3.ALL_PRESETS['mid-14k']
        coder, values, audio = transmission(layout)
        self.assertGreater(occupied(audio), 20000)       # before
        bounded = bound_emission(audio, 14000, RATE)
        self.assertLess(occupied(bounded), 14000)        # after
        headers, pictures, error = decode(layout, coder, bounded, values)
        self.assertEqual((headers, pictures), (4, 4))
        self.assertLess(error, .01)

    def test_lean_14k_keeps_lean_v3s_frame_rate(self):
        lean = v3.ALL_PRESETS['lean-v3']
        layout = v3.ALL_PRESETS['lean-14k']
        self.assertAlmostEqual(layout.fps, lean.fps, places=6)
        self.assertLess(layout.band_at(RATE)[1], 14000)
        coder, values, audio = transmission(layout, 'lean-dct')
        bounded = bound_emission(audio, 14000, RATE)
        self.assertLess(occupied(bounded), 14000)
        self.assertEqual(decode(layout, coder, bounded, values)[:2], (4, 4))

    def test_both_new_presets_leave_guard_under_the_cut(self):
        """Why top_bin 34 and not 37. Crowding the cut costs the top carriers,
        measured at 0.036 error against 0.0062 with the guard."""
        for name in ('mid-14k', 'lean-14k'):
            with self.subTest(preset=name):
                top = v3.ALL_PRESETS[name].band_at(RATE)[1]
                self.assertLess(top, 13000)
                self.assertGreater(14000-top, 1000)

    def test_a_ceiling_above_nyquist_is_a_no_op(self):
        layout = v3.ALL_PRESETS['mid-14k']
        _, _, audio = transmission(layout)
        np.testing.assert_array_equal(bound_emission(audio, None, RATE), audio)
        np.testing.assert_array_equal(bound_emission(audio, 30000, RATE), audio)

    def test_the_level_survives_filtering(self):
        """encode normalises to 0.95 and the receiver's channel estimate
        divides a whole-packet gain back out, but the packet must not come
        back louder than it went in and clip the DAC."""
        layout = v3.ALL_PRESETS['mid-14k']
        _, _, audio = transmission(layout)
        bounded = bound_emission(audio, 14000, RATE)
        self.assertAlmostEqual(float(np.max(np.abs(bounded))),
                               float(np.max(np.abs(audio))), places=4)


class SharpnessTests(unittest.TestCase):
    def test_a_sharper_filter_is_worse_not_better(self):
        """The counter-intuitive one, and the reason EMIT_TAPS is 63 rather
        than as high as it will go. CP is 16 samples; a filter that rings
        longer than that smears across the symbol boundary and breaks
        orthogonality, so error rises with tap count."""
        layout = v3.ALL_PRESETS['mid-14k']
        coder, values, audio = transmission(layout)
        errors = [decode(layout, coder,
                         bound_emission(audio, 14000, RATE, taps=t), values)[2]
                  for t in (63, 511, 1023)]
        self.assertLess(errors[0], errors[1])
        self.assertLess(errors[1], errors[2])


class RollOffTests(unittest.TestCase):
    def test_a_channel_that_rolls_off_needs_no_sender_filter(self):
        """Worth stating so nobody reaches for --emit-ceiling by default. A
        receive-side low-pass at the same frequency costs nothing: the tail
        carries no information, only energy."""
        from animation_modem import impairments as IMP
        layout = v3.ALL_PRESETS['mid-14k']
        coder, values, audio = transmission(layout)
        emulator = IMP.Emulator(IMP.Settings(lowpass_hz=14000, noise_dbfs=-42))
        signal = np.concatenate([np.zeros((300, 2)), audio])
        rx = v3.Receiver(layout, coder)
        out = []
        for i in range(0, len(signal), 256):
            out += rx.feed(emulator.process(signal[i:i+256]))
        out += rx.flush()
        self.assertEqual(sum(1 for r in out if r.identity == 'verified_header'), 4)


class DeviceRateTests(unittest.TestCase):
    """The ceiling is in hertz, and the packet is on the reference grid.

    bound_emission runs BEFORE band_limited resamples to the device, so the
    filter must be designed against REFERENCE_RATE whatever the device is set
    to. Designing it against the device rate scales the real cutoff by
    REFERENCE_RATE/device: on a 96 kHz device a 14000 Hz ceiling actually cut
    at 7008 Hz, which removes most of mid-14k's 375-12750 Hz band, and the
    live receiver decoded nothing at all.
    """

    def test_the_cutoff_does_not_follow_the_device_rate(self):
        layout = v3.ALL_PRESETS['mid-14k']
        coder, values, audio = transmission(layout, 'lean-dct')
        bounded = bound_emission(audio, 14000, RATE)
        top = layout.band_at(RATE)[1]
        kept = np.abs(np.fft.rfft(bounded[:, 0]))
        freqs = np.fft.rfftfreq(len(bounded), 1/RATE)
        # energy must survive right up to the carriers, not be cut at half
        band = kept[(freqs > top-1500) & (freqs < top)]
        self.assertGreater(float(np.max(band))/float(np.max(kept)), 1e-3)
        self.assertEqual(decode(layout, coder, bounded, values)[:2], (4, 4))

    def test_a_ceiling_designed_against_the_wrong_rate_breaks_it(self):
        """The bug, stated as the measurement that would have caught it."""
        layout = v3.ALL_PRESETS['mid-14k']
        coder, values, audio = transmission(layout, 'lean-dct')
        wrong = bound_emission(audio, 14000, 96000)      # what the live path did
        self.assertLess(occupied(wrong), 9000)
        headers, _, error = decode(layout, coder, wrong, values)
        self.assertTrue(headers == 0 or error > .05,
                        'cutting at 7 kHz must visibly break mid-14k')

    def test_the_live_sender_designs_against_the_reference_rate(self):
        """Pins the call site itself, since the failure is invisible at 48 kHz
        -- REFERENCE_RATE and the device rate are the same number there."""
        import ast
        from pathlib import Path
        source = (Path(__file__).resolve().parent.parent/'modem_screen.py').read_text()
        calls = [n for n in ast.walk(ast.parse(source))
                 if isinstance(n, ast.Call)
                 and getattr(n.func, 'id', None) == 'bound_emission']
        self.assertTrue(calls, 'modem_screen must apply the ceiling')
        for call in calls:
            self.assertEqual(getattr(call.args[2], 'id', None), 'RATE',
                             'bound_emission takes the rate the packet is AT')


if __name__ == '__main__':
    unittest.main()
