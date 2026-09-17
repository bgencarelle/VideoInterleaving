"""Header CRC tolerance: soft-reliability bit correction on the V3 header.

The header CRC is all-or-nothing: any flipped bit in the 20-byte word (16 data
+ stored CRC32) fails the check wholesale. On a band-limited wire the header
can sit a handful of bits off -- on the wire an 8.8 kHz brick-wall clips the
top header carriers (which reach 8.6 kHz) and flips some headers past the
straight CRC -- while the magnitude is untouched. These tests pin the tolerance
fallback that turns such verdicts from ``lost`` into verified identities by
borrowing soft reliability from the QPSK symbols (see core._correct_header).

Clean wires never enter the search (the straight CRC passes); its only cost is
paid where the hard decision was marginal.
"""
import unittest

import numpy as np

import animation_modem.core as core
from animation_modem import transport3 as v3
from animation_modem.core import _correct_header, _resolve_header, pack_header
from animation_modem.imaging import plane_shapes, plane_grids


LAYOUT = v3.WIRE
HEADER = (16, LAYOUT.top_bin, 7, 3, 24, 42)  # (flags, top, absolute, index, count, stamp)
PROFILE = 2


def word_with_flips(word, positions):
    """Corrupt `positions` (bit indices into the 160-bit stream) in `word`."""
    word = bytearray(word)
    for p in positions:
        word[p >> 3] ^= 1 << (7 - (p & 7))
    return bytes(word)


def flat_from_word(word):
    """The 80 QPSK symbols a receiver would see for a hard-decoded `word`.

    Signs match `word`; every component is strong (|v| = 1.0) so the caller can
    mark specific positions marginal afterwards.
    """
    bits = np.unpackbits(np.frombuffer(word, np.uint8))
    flat = np.empty(80, complex)
    for j in range(80):
        flat[j] = (1.0 if bits[2*j] else -1.0) + 1j * (1.0 if bits[2*j+1] else -1.0)
    return flat


def degrade(flat, positions):
    """Mark the given bit positions as low-reliability in `flat`."""
    flat = flat.copy()
    for p in positions:
        axis = p % 2
        j = p >> 1
        if axis == 0:
            flat[j] = 0.02 * np.sign(flat[j].real) + 1j * flat[j].imag
        else:
            flat[j] = flat[j].real + 1j * 0.02 * np.sign(flat[j].imag)
    return flat


class HeaderCorrectionTest(unittest.TestCase):
    def setUp(self):
        self.word = pack_header(*HEADER, magic=LAYOUT.wire_magic,
                                profile=PROFILE)
        self.expected = (HEADER[0], HEADER[2], HEADER[3], HEADER[4], HEADER[5],
                         PROFILE)

    def test_clean_word_resolves(self):
        self.assertEqual(_resolve_header(self.word, LAYOUT), self.expected)

    def test_single_bit_correction(self):
        for pos in (0, 61, 128):
            bad = word_with_flips(self.word, (pos,))
            flat = degrade(flat_from_word(bad), (pos,))
            self.assertIsNone(_resolve_header(bad, LAYOUT))
            self.assertEqual(_correct_header(bad, flat, 1, LAYOUT), self.expected)

    def test_double_bit_correction(self):
        positions = (4, 133)
        bad = word_with_flips(self.word, positions)
        flat = degrade(flat_from_word(bad), positions)
        self.assertEqual(_correct_header(bad, flat, 2, LAYOUT), self.expected)
        # One short of the tolerance misses it.
        self.assertIsNone(_correct_header(bad, flat, 1, LAYOUT))

    def test_triple_bit_correction(self):
        positions = (11, 44, 150)
        bad = word_with_flips(self.word, positions)
        flat = degrade(flat_from_word(bad), positions)
        self.assertEqual(_correct_header(bad, flat, 3, LAYOUT), self.expected)

    def test_correction_undershoots_and_overshoot(self):
        # Two errors, tolerance says one -> nothing. Tolerance beyond the
        # truth still lands on the exact flip set, never an arbitrary CRC hit.
        positions = (7, 99)
        bad = word_with_flips(self.word, positions)
        flat = degrade(flat_from_word(bad), positions)
        self.assertIsNone(_correct_header(bad, flat, 1, LAYOUT))
        self.assertEqual(_correct_header(bad, flat, 4, LAYOUT), self.expected)


def make_stream(layout, coder, frames, seed=7):
    rng = np.random.default_rng(seed)
    wire = []
    for n in range(frames):
        values = rng.uniform(-1, 1, coder.source_count)
        wire.append(np.asarray(v3.encode(values, layout, coder, n+1, n+1,
                                         frames, profile=0), np.float32))
    return np.concatenate(wire)


def band_limit(wire, cutoff, rate=v3.REFERENCE_RATE):
    win = np.fft.rfft(wire, axis=0)
    win[np.fft.rfftfreq(len(wire), 1/rate) > cutoff] = 0
    return np.fft.irfft(win, n=len(wire), axis=0).astype(np.float32)


def decode_stream(wire, tolerance, layout, coder, chunk=8192):
    receiver = v3.Receiver(layout, coder, pulse_only=True, fast=True,
                           header_tolerance=tolerance)
    records = []
    for start in range(0, len(wire), chunk):
        records.extend(receiver.feed(wire[start:start+chunk]))
    records.extend(receiver.flush())
    return records


class BandLimitRecoveryTest(unittest.TestCase):
    """Baseline engine vs the tolerance engine on the same wire.

    Two engines, same exact wire, side by side:

      * baseline  -- the pre-tolerance decoder (header_tolerance=0). Its CRC is
        all-or-nothing: an all-else-well decoded image with a few flipped
        header bits is discarded as `'lost'`, never identified.
      * engine -- the soft-reliability correction added on top. Identical on
        clean wires (the straight CRC passes first), strictly better where the
        hard header decision was marginal.

    The tests assert the equality on clean wire (no behaviour drift from the
    decode optimizations) and the delta on a band-limited wire (0 verified
    frames becomes a fully-verified stream).
    """
    FRAMES = 24
    CUTOFF = 8800

    @classmethod
    def setUpClass(cls):
        shapes = plane_shapes('color-dct')
        cls.coder = v3.SourceCoder(shapes, grids=plane_grids('color-dct'))
        cls.wire = make_stream(LAYOUT, cls.coder, cls.FRAMES)
        cls.clipped = band_limit(cls.wire, cls.CUTOFF)
        cls.wire_records = decode_stream(cls.wire, 0, LAYOUT, cls.coder)
        cls.clipped_records = {tol: decode_stream(cls.clipped, tol, LAYOUT,
                                                   cls.coder)
                               for tol in (0, 2, 3, 4)}

    def verified(self, tolerance):
        records = self.clipped_records[tolerance]
        return sum(1 for r in records if r.identity == 'verified_header')

    def by_index(self, records):
        return {r.index: r for r in records if r.identity == 'verified_header'}

    def test_engine_is_baseline_on_clean_wire(self):
        # tolerance engages nothing when the straight CRC passes: zero drift.
        fresh = decode_stream(self.wire, 2, LAYOUT, self.coder)
        self.assertEqual(
            [(r.status, r.identity, r.index) for r in fresh],
            [(r.status, r.identity, r.index) for r in self.wire_records])

    def test_rolloff_engine_never_worse_than_baseline(self):
        baseline = self.verified(0)
        for tol in (2, 3, 4):
            self.assertGreaterEqual(self.verified(tol), baseline)

    def test_rolloff_engine_recovers_a_baseline_lost_frame(self):
        # On this synthetic wire the baseline verifies 16 of 24 packets; the
        # tolerance engine brings back the other 8 (observed tol2 -> 24/24
        # for this seed).
        self.assertGreater(self.verified(3), self.verified(0))

    def test_recovered_payload_unchanged(self):
        # Where both engines verified the same frame, the values must be
        # byte-identical: tolerance relabels lost frames, never reshapes the
        # payload it already agreed on.
        baseline = self.by_index(self.clipped_records[0])
        for index, record in self.by_index(self.clipped_records[3]).items():
            if index in baseline:
                self.assertIsNotNone(record.values)
                np.testing.assert_array_equal(record.values,
                                              baseline[index].values)

    def test_recovered_frames_still_say_the_told_story(self):
        # A corrected header is not a hallucinated one: index/count/absolute
        # all match what was transmitted.
        for r in self.clipped_records[3]:
            if r.identity == 'verified_header':
                self.assertEqual(r.count, self.FRAMES)
                self.assertIn(r.index, range(1, self.FRAMES+1))
                self.assertEqual(r.absolute, r.index)

    def test_tolerance_does_not_double_flip(self):
        # The search demands the CRC verify a corrected word end to end, so a
        # tolerance larger than the true error count must land on the exact
        # flip set -- the recovered headers agree with tolerance 3 baseline.
        recovered = [(r.index, r.absolute, r.stamp_ms)
                     for r in self.clipped_records[3]
                     if r.identity == 'verified_header']
        overkill = [(r.index, r.absolute, r.stamp_ms)
                    for r in self.clipped_records[4]
                    if r.identity == 'verified_header']
        self.assertEqual(overkill, recovered)


if __name__ == '__main__':
    unittest.main()