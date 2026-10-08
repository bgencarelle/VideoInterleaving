"""Tape-like recording chains for the torture tests (tools/v7_torture_matrix).

A deck is modelled the way it bends a signal, in the order it happens:

record   the record level (``drive_db``), then record equalisation: the highs
         are boosted before they reach the tape (a shelf from the deck's
         time constant, ``eq_us``, rising ``shelf_db`` at most);
tape     magnetic saturation: nothing happens below a knee ``knee_db`` under
         the tape's maximum output level (``mol``), and above it the level
         bends smoothly toward that maximum. Because the highs were boosted
         first, they saturate first and hardest, as on real tape;
hiss     the tape's own hiss (``hiss_dbfs``), at a fixed level whatever the
         record level, so a quiet recording has it close under the signal;
playback playback equalisation undoes the record boost exactly (so anything
         added on the tape, hiss included, comes off with its highs cut), and
         the playback head's gap and spacing losses roll the top off
         (``gap_hz``).

Not modelled: magnetic hysteresis itself (the loop's memory); its audible
part, the frequency-dependent compression, is what the equalisation around
the saturation gives. Bias, noise reduction and the cassette's 3180 us bass
time constant (record cut and playback boost cancel) are left out.

Levels are digital: ``mol`` is the tape's maximum output level in full-scale
units, as if the deck's playback were captured with its 0 VU a few dB under
the converter's full scale. The numbers are typical of each kind of deck,
not of one machine.
"""
from dataclasses import dataclass

import numpy as np
from scipy.signal import bilinear, butter, lfilter, sosfilt


@dataclass(frozen=True)
class Deck:
    name: str
    eq_us: float          # record/playback equalisation time constant
    shelf_db: float       # most the record EQ boosts the highs
    knee_db: float        # saturation starts this far under the maximum
    mol: float = .95      # maximum output level, full-scale units
    gap_hz: float = 0     # playback head loss corner (0: none)
    drive_db: float = 0   # record level against the wire (cheap decks: < 0)
    hiss_dbfs: float | None = None   # the tape's own hiss, before playback EQ


DECKS = {
    # Cheap decks go in too quiet: the record level is set low, so the tape
    # barely saturates and its hiss sits close under the signal.
    # Ferric cassette (IEC type I, 120 us) on a cheap deck, 8 dB under.
    'cassette-i': Deck('cassette-i', eq_us=120, shelf_db=12, knee_db=6,
                       gap_hz=12000, drive_db=-8, hiss_dbfs=-48),
    # Chrome/high-bias cassette (IEC type II, 70 us), 4 dB under.
    'cassette-ii': Deck('cassette-ii', eq_us=70, shelf_db=10, knee_db=5,
                        gap_hz=15000, drive_db=-4, hiss_dbfs=-52),
    # Ferric cassette recorded 4 dB hot, as an over-eager record level.
    'cassette-i-hot': Deck('cassette-i-hot', eq_us=120, shelf_db=12,
                           knee_db=6, gap_hz=12000, drive_db=4,
                           hiss_dbfs=-48),
    # Reel-to-reel at 15 ips (50 us), properly aligned: much more headroom,
    # little head loss, little hiss.
    'reel-15ips': Deck('reel-15ips', eq_us=50, shelf_db=6, knee_db=4,
                       gap_hz=20000, hiss_dbfs=-62),
}


def _shelf(eq_us, shelf_db, rate, inverse=False):
    """The record EQ (or its exact inverse): (1+s*t1)/(1+s*t2)."""
    t1 = eq_us*1e-6
    t2 = t1/10**(shelf_db/20)
    num, den = [t1, 1.0], [t2, 1.0]
    if inverse:
        num, den = den, num
    return bilinear(num, den, fs=rate)


def saturate(x, knee_db, mol):
    """Unity below the knee; above it, bend smoothly toward ``mol`` (the
    slope is continuous at the knee)."""
    knee = mol*10**(-knee_db/20)
    magnitude = np.abs(x)
    over = magnitude > knee
    out = np.array(x, float, copy=True)
    room = mol-knee
    out[over] = np.sign(x[over])*(
        knee+room*np.tanh((magnitude[over]-knee)/room))
    return out


def record(x, deck, rate):
    """Record level, record EQ and the tape's saturation."""
    x = np.asarray(x, float)*10**(deck.drive_db/20)
    b, a = _shelf(deck.eq_us, deck.shelf_db, rate)
    return saturate(lfilter(b, a, x, axis=0), deck.knee_db, deck.mol)


def playback(x, deck, rate):
    """Playback EQ (the record boost undone) and the head's losses. The
    level is not set back: a recording made quiet plays back quiet, with the
    tape's hiss where it was."""
    b, a = _shelf(deck.eq_us, deck.shelf_db, rate, inverse=True)
    x = lfilter(b, a, x, axis=0)
    if deck.gap_hz and deck.gap_hz < rate/2:
        x = sosfilt(butter(2, deck.gap_hz, fs=rate, output='sos'), x, axis=0)
    return x


def digital_clip(x, overdrive_db):
    """A converter driven past full scale: a hard clip, nothing soft."""
    return np.clip(np.asarray(x, float)*10**(overdrive_db/20), -1, 1)
