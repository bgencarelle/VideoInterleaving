# audio_harmonizer.py
#
# Pure “index → chord” engine.
# - Can be imported from your main app
# - Or run as a tiny CLI to inspect chords for a given index
#
# It does NOT play sound – it just returns MIDI note numbers.
# You hook those into whatever synth / MIDI system you like.

from dataclasses import dataclass
from typing import List, Tuple, Optional

try:
    import settings  # your project-wide settings.py
except ImportError:
    settings = None


# ------------------------------
# Config
# ------------------------------

@dataclass
class HarmonizerConfig:
    """
    Runtime config. Normally loaded from settings.py, but you can
    also construct this manually for API control.
    """
    mode: str = "preset"              # "preset" or "api"
    preset: str = "hybrid4"           # default harmonizer preset
    base_octave: int = 3              # rough register: 3 = around C3
    octave_span: int = 3              # how many octaves index can walk through

def load_config_from_settings() -> HarmonizerConfig:
    """
    Read audio-related values from settings.py if present.
    Falls back to defaults otherwise.
    """
    cfg = HarmonizerConfig()
    if settings is not None:
        cfg.mode        = getattr(settings, "AUDIO_MODE", cfg.mode)
        cfg.preset      = getattr(settings, "AUDIO_PRESET", cfg.preset)
        cfg.base_octave = getattr(settings, "AUDIO_BASE_OCTAVE", cfg.base_octave)
        cfg.octave_span = getattr(settings, "AUDIO_OCTAVE_SPAN", cfg.octave_span)
    return cfg


# ------------------------------
# Helpers: index → note pair
# ------------------------------

def derive_note_pair_from_folders(
    index: int,
    main_count: int,
    float_count: int
) -> Tuple[int, int]:
    """
    Derive two pitch classes (0–11) from:
      - the global index
      - main folder count
      - float folder count

    This is intentionally simple and deterministic.
    Adjust if you have a different mapping in mind.
    """
    mc = max(1, main_count)
    fc = max(1, float_count)

    # Base: simple mod of each folder
    a_raw = index % mc
    # Second note: scrambled a bit to avoid trivial lockstep
    b_raw = (index * 7) % fc   # 7 is just a primitive-ish multiplier here

    # Map to pitch classes
    a_pc = a_raw % 12
    b_pc = b_raw % 12

    return a_pc, b_pc


# ------------------------------
# Harmonizer “presets”
# ------------------------------

def preset_product_mod_triad(a: int, b: int, index: int) -> List[int]:
    """
    2-note input; third note = (a * b) mod 12.
    Returns a 3-note chord as pitch classes.
    """
    third = (a * b) % 12
    return [a, b, third]


def preset_midpoint(a: int, b: int, index: int) -> List[int]:
    """
    2-note input; third note is the rounded midpoint.
    Returns a 3-note chord as pitch classes.
    """
    mid = round((a + b) / 2) % 12
    return [a, b, mid]


def preset_hybrid4(a: int, b: int, index: int) -> List[int]:
    """
    Hybrid: {a, b, (a*b mod 12), (a+b mod 12)}.
    Returns a 4-note set as sorted, deduplicated pitch classes.
    """
    p = (a * b) % 12
    s = (a + b) % 12
    pcs = {a, b, p, s}
    return sorted(pcs)


PRESETS = {
    "product_mod_triad": preset_product_mod_triad,
    "midpoint":          preset_midpoint,
    "hybrid4":           preset_hybrid4,
}


# ------------------------------
# Pitch class → MIDI, with index-driven octave
# ------------------------------

def pcs_to_midi(
    pcs: List[int],
    index: int,
    cfg: HarmonizerConfig
) -> List[int]:
    """
    Convert pitch classes (0–11) to MIDI notes, using:
      - cfg.base_octave
      - cfg.octave_span
      - index to walk through the span

    MIDI convention:  C4 = 60, so:
      midi = 12 * (octave + 1) + pitch_class
    """
    # Walk index through the available octaves.
    # This is a simple saw; if you want ping-pong, mirror it yourself.
    octave_offset = 0
    if cfg.octave_span > 0:
        octave_offset = index % cfg.octave_span

    octave = cfg.base_octave + octave_offset

    midi_notes: List[int] = []
    for p in pcs:
        midi = 12 * (octave + 1) + (p % 12)
        midi_notes.append(midi)

    return midi_notes


# ------------------------------
# Public API: index → chord
# ------------------------------

def chord_pcs_from_index(
    index: int,
    total: int,
    main_count: int,
    float_count: int,
    preset: str = "hybrid4",
    *,
    explicit_notes: Optional[Tuple[int, int]] = None
) -> List[int]:
    """
    Core pure function:
      (index, total, folder sizes, preset[, explicit note pair]) → pitch classes

    - If explicit_notes is given, it's (a,b) pitch classes and we skip folder logic.
    - Otherwise we derive the pair via modulo of main/float folder sizes.
    """
    if explicit_notes is not None:
        a_pc, b_pc = (explicit_notes[0] % 12, explicit_notes[1] % 12)
    else:
        a_pc, b_pc = derive_note_pair_from_folders(index, main_count, float_count)

    fn = PRESETS.get(preset, preset_hybrid4)
    pcs = fn(a_pc, b_pc, index)
    return pcs


def get_midi_chord_for_index(
    index: int,
    total: int,
    main_count: int,
    float_count: int,
    cfg: Optional[HarmonizerConfig] = None,
    preset: Optional[str] = None,
    *,
    explicit_notes: Optional[Tuple[int, int]] = None
) -> List[int]:
    """
    High-level entry point:
      (index, total, folder sizes, [cfg], [preset], [explicit_notes]) → MIDI notes

    - If cfg is None, we auto-load from settings.py.
    - If preset is None, we use cfg.preset.
    - explicit_notes lets you bypass folder/mod logic and drive it directly.
    """
    if cfg is None:
        cfg = load_config_from_settings()

    effective_preset = preset or cfg.preset
    pcs = chord_pcs_from_index(
        index=index,
        total=total,
        main_count=main_count,
        float_count=float_count,
        preset=effective_preset,
        explicit_notes=explicit_notes,
    )
    midi_notes = pcs_to_midi(pcs, index=index, cfg=cfg)
    return midi_notes


# ------------------------------
# Standalone CLI mode (debug / testing)
# ------------------------------

def _main_cli() -> None:
    """
    Minimal CLI to test mappings:

      python audio_harmonizer.py --index 42 --total 1000 --main-count 37 --float-count 29 --preset hybrid4

    Prints the pitch classes and MIDI notes for inspection.
    """
    import argparse

    parser = argparse.ArgumentParser(description="Index → chord debug tool")
    parser.add_argument("--index",       type=int, required=True, help="Current index (0-based)")
    parser.add_argument("--total",       type=int, required=True, help="Maximum index / total images")
    parser.add_argument("--main-count",  type=int, required=True, help="Main folder image count")
    parser.add_argument("--float-count", type=int, required=True, help="Float folder image count")
    parser.add_argument("--preset",      type=str, default=None,
                        help=f"Preset name: {', '.join(PRESETS.keys())}")

    args = parser.parse_args()

    cfg = load_config_from_settings()
    if args.preset is not None:
        preset_name = args.preset
    else:
        preset_name = cfg.preset

    pcs = chord_pcs_from_index(
        index=args.index,
        total=args.total,
        main_count=args.main_count,
        float_count=args.float_count,
        preset=preset_name,
    )
    midi_notes = pcs_to_midi(pcs, index=args.index, cfg=cfg)

    print(f"Config: {cfg}")
    print(f"Preset: {preset_name}")
    print(f"Index {args.index}/{args.total}")
    print(f"Pitch classes: {pcs}")
    print(f"MIDI notes:    {midi_notes}")


if __name__ == "__main__":
    _main_cli()
