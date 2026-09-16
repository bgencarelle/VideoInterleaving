"""Where the header sits decides whether a tape machine can carry the picture.

`spread_carriers` moves the header from the bottom of the band to the middle,
to survive a channel that rolls off at its own bottom edge. A cassette deck has
the opposite problem: it rolls off at the top, and the middle is exactly where
it stops.

The failure is worth naming precisely, because it does not look like a decode
failure. The pictures still arrive -- every one of them -- but without a
verified header, and the live receiver only holds a verified frame for display.
So a deck that rolls off at 8 kHz gives a black screen while the decoder is
reconstructing all four frames.
"""
import unittest

import numpy as np

from animation_modem import impairments as IMP, transport3 as v3
from animation_modem.imaging import fit_shapes, plane_shapes

BIN_HZ = 48000/128
LEAN = v3.ALL_PRESETS['lean-v3']
TAPE = v3.ALL_PRESETS['lean-v3-tape']


def coder_for(profile, layout):
    shapes = plane_shapes(profile)
    if sum(int(np.prod(s)) for s in shapes) > layout.capacity:
        shapes = fit_shapes(shapes, layout.capacity)
    return v3.SourceCoder(shapes)


def through(layout, frames=4, **impairment):
    coder = coder_for('lean-dct', layout)
    values = np.random.default_rng(6).uniform(-.2, .2, coder.count)
    clean = np.concatenate([v3.encode(values, layout, coder, n, n, frames,
                                      profile=v3.profile_code('lean-dct'))
                            for n in range(1, frames+1)])
    emulator = IMP.Emulator(IMP.Settings(**impairment))
    signal = np.concatenate([np.zeros((300, 2)), clean])
    rx = v3.Receiver(layout, coder)
    out = []
    for i in range(0, len(signal), 256):
        out += rx.feed(emulator.process(signal[i:i+256]))
    out += rx.flush()
    return (sum(1 for r in out if r.identity == 'verified_header'),
            sum(1 for r in out if r.values is not None))


class HeaderPlacementTests(unittest.TestCase):
    def test_the_two_layouts_differ_only_in_where_the_header_sits(self):
        self.assertEqual(TAPE.frame, LEAN.frame)
        self.assertEqual(TAPE.capacity, LEAN.capacity)
        self.assertEqual(TAPE.fps, LEAN.fps)
        self.assertEqual(TAPE.top_bin, LEAN.top_bin)
        self.assertLess(TAPE.header_bins.max()*BIN_HZ, 9000)
        self.assertGreater(LEAN.header_bins.max()*BIN_HZ, 13000)

    def test_a_cassette_rolloff_takes_the_mid_band_header_out(self):
        """The regression, stated as a measurement. Note the second number:
        the pictures are all there. Only identity is lost, and the live
        receiver shows nothing without it."""
        headers, pictures = through(LEAN, lowpass_hz=8000, noise_dbfs=-42)
        self.assertEqual(headers, 0)
        self.assertEqual(pictures, 4)

    def test_the_bottom_header_survives_the_same_deck(self):
        self.assertEqual(through(TAPE, lowpass_hz=8000, noise_dbfs=-42), (4, 4))

    def test_it_survives_flutter_on_top_of_the_rolloff(self):
        """Tape wobbles as well as rolling off; timing comes from the preamble,
        which neither layout puts in the affected band."""
        coder = coder_for('lean-dct', TAPE)
        values = np.random.default_rng(6).uniform(-.2, .2, coder.count)
        clean = np.concatenate([v3.encode(values, TAPE, coder, n, n, 4,
                                          profile=v3.profile_code('lean-dct'))
                                for n in range(1, 5)])
        signal = np.concatenate([np.zeros((300, 2)), clean])
        n = np.arange(len(signal))
        warp = n + .004*len(signal)/(2*np.pi*6)*np.sin(2*np.pi*6*n/48000)
        signal = np.stack([np.interp(warp, n, signal[:, c]) for c in range(2)],
                          axis=1)
        emulator = IMP.Emulator(IMP.Settings(lowpass_hz=8000, noise_dbfs=-42))
        rx = v3.Receiver(TAPE, coder)
        out = []
        for i in range(0, len(signal), 256):
            out += rx.feed(emulator.process(signal[i:i+256]))
        out += rx.flush()
        self.assertEqual(sum(1 for r in out if r.identity == 'verified_header'), 4)

    def test_direct_audio_is_unaffected_for_both(self):
        """The tape layout must not cost anything on the path that works."""
        self.assertEqual(through(LEAN), (4, 4))
        self.assertEqual(through(TAPE), (4, 4))

    def test_what_the_mid_band_header_buys_in_return(self):
        """Honest about the trade. spread_carriers exists for bottom-edge
        rolloff, and it does help -- over a narrow range. Both placements lose
        every header by a 375 Hz high-pass, the case its own comment cites."""
        self.assertEqual(through(LEAN, highpass_hz=300, noise_dbfs=-45)[0], 4)
        self.assertEqual(through(TAPE, highpass_hz=300, noise_dbfs=-45)[0], 0)
        self.assertEqual(through(LEAN, highpass_hz=375, noise_dbfs=-45)[0], 0)
        self.assertEqual(through(TAPE, highpass_hz=375, noise_dbfs=-45)[0], 0)

    def test_the_receiver_tells_them_apart(self):
        """Same packet length, same top_bin, overlapping header carriers -- but
        the header is read off different ones, so the CRC separates them and
        neither needs naming at the receiver."""
        self.assertEqual(TAPE.packet, LEAN.packet)
        cands = sorted(
            [(l, coder_for('lean-dct', l),
              {v3.profile_code(p): coder_for(p, l) for p in v3.PROFILE_CODES})
             for l in (LEAN, TAPE)], key=lambda c: c[0].packet)
        for layout in (LEAN, TAPE):
            with self.subTest(sent=layout.name):
                coder = coder_for('lean-dct', layout)
                values = np.random.default_rng(2).uniform(-.2, .2, coder.count)
                audio = v3.encode(values, layout, coder, 1, 1, 1,
                                  profile=v3.profile_code('lean-dct'))
                rx = v3.Receiver(LEAN, coder_for('lean-dct', LEAN),
                                 candidates=cands)
                out = rx.feed(np.asarray(audio, np.float32))+rx.flush()
                self.assertEqual(rx.detected, layout.name)
                self.assertLess(
                    np.sqrt(np.mean((out[0].values-values)**2)), 1e-5)


if __name__ == '__main__':
    unittest.main()
