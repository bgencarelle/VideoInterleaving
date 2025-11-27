# main.py

import argparse
import random
from typing import Optional

import harmonizer
from harmonizer import HarmonizerConfig, chord_midi_from_pair
import audio_engine

try:
    import settings
except ImportError:
    class _FallbackSettings:
        DEFAULT_A_PC = 0
        DEFAULT_B_PC = 7
    settings = _FallbackSettings()


def generate_chord_for_index(
    index: int,
    max_index: int,
    a_pc: Optional[int] = None,
    b_pc: Optional[int] = None,
    config: Optional[HarmonizerConfig] = None,
):
    """
    Library-style API: from index + optional pair → MIDI notes.
    """
    if a_pc is None:
        a_pc = getattr(settings, "DEFAULT_A_PC", 0)
    if b_pc is None:
        b_pc = getattr(settings, "DEFAULT_B_PC", 7)

    if config is None:
        config = HarmonizerConfig()

    midi_notes = chord_midi_from_pair(a_pc, b_pc, index, max_index, config)
    return midi_notes


def main():
    parser = argparse.ArgumentParser(
        description="Index-driven chord generator (standalone demo)."
    )
    parser.add_argument("--index", type=int, default=0, help="Current index (N >= 0)")
    parser.add_argument("--max-index", type=int, default=100, help="Largest index value")
    parser.add_argument("--a-pc", type=int, default=None, help="Pitch class of first note (0–11)")
    parser.add_argument("--b-pc", type=int, default=None, help="Pitch class of second note (0–11)")
    parser.add_argument("--random-pair", action="store_true", help="Ignore a_pc/b_pc and choose random pair")
    parser.add_argument("--out", type=str, default="chord.wav", help="Output WAV filename")

    args = parser.parse_args()

    if args.random_pair:
        a_pc = random.randint(0, 11)
        b_pc = random.randint(0, 11)
    else:
        a_pc = args.a_pc if args.a_pc is not None else getattr(settings, "DEFAULT_A_PC", 0)
        b_pc = args.b_pc if args.b_pc is not None else getattr(settings, "DEFAULT_B_PC", 7)

    config = HarmonizerConfig()
    midi_notes = chord_midi_from_pair(a_pc, b_pc, args.index, args.max_index, config)

    print(f"Index: {args.index}/{args.max_index}")
    print(f"Input PCs: a={a_pc}, b={b_pc}")
    print(f"MIDI notes: {midi_notes}")

    audio_engine.render_sine_chord(midi_notes, args.out)
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
