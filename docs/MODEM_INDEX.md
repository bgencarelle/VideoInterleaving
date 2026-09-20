# Modem documentation index

This is the entry point for the modem subsystem. The active implementation is
on the `standalone-modem` branch. Read the specification first, then use the
testing and source links below.

## Start here

1. **[Transport timing and encoding specification](transport_timing_and_encoding.md)**
   — normative/reimplementation-oriented wire description, provenance,
   timing, headers, CRC, transforms, placement, receiver behavior, vectors,
   and discrepancy audit.
2. **[Transport state machines](transport_state_machines.md)** — encode,
   decode, acquisition, recovery, and placement state machines for V1, V2,
   V3, V5, tape wires, V6, V6-repeat, and tape-ordered V6.
3. **[V6/tape testing guide](TAPE_WIRE_TESTING.md)** — synthetic controls,
   live commands, tape-band limits, baseline V6, full-repeat, and the new
   bench-only tape-ordered V6 candidate.

## Operational guides

- [Modem mode](MODEM_MODE.md) — sender/display behavior, aspect handling,
  source preparation, filters, and decoded display scaling.
- [Self-describing live modem](LIVE_MODEM.md) — receiver discovery, sample
  rate independence, wire/profile selection, and live operation.
- [Modem pitch probe](MODEM_PITCH_PROBE.md) — diagnostic pitch/band inspection.
- [Project modem roadmap](../MODEM_TODO.md) — authoritative experimental
  direction, acceptance limits, and remaining work.

## Reproducibility and diagnostics

- [Deterministic vector generator](../tools/spec_vectors.py) — emits the ten
  documented one-packet WAV vectors under `scratch/spec-vectors/` by default.
  V1/V2 vectors extract their pinned historical source with `git show`.
- [V6 tape-lift benchmark](../tools/bench_v6_tape_lift.py) — compares
  baseline, tape-ordered, full-repeat, and no-copy controls under a synthetic
  shared spacing-loss diagnostic. It is not a real-deck model.
- [V6 tests](../modem_tests/test_v6.py), [V6-repeat tests](../modem_tests/test_v6_repeat.py),
  and [tape-placement tests](../modem_tests/test_v6_tape.py) — placement,
  protection, recovery, and clean synthetic-wire invariants.
- [Tape-wire tests](../modem_tests/test_tape_wire.py) — tape layouts,
  placement diversity, emission ceiling, rate changes, and round trips.
- [Current transport tests](../modem_tests/test_transport3.py) — pulse
  acquisition, framing, headers, training, rate/speed handling, and decode.

## Source map

| Area | Source |
|---|---|
| Shared headers, CRC, mapping, equalizer, and decoder | [`animation_modem/core.py`](../animation_modem/core.py) |
| Current pulse-counted transport and layouts | [`animation_modem/transport3.py`](../animation_modem/transport3.py) |
| V6 protection and placement | [`animation_modem/v6.py`](../animation_modem/v6.py) |
| Haar/CDF 9/7 and repeat coders | [`animation_modem/wavelet.py`](../animation_modem/wavelet.py) |
| Image preparation and color planes | [`animation_modem/imaging.py`](../animation_modem/imaging.py) |
| Active engine/profile dispatch | [`animation_modem/engines.py`](../animation_modem/engines.py) |
| Audio PCM and per-channel level handling | [`animation_modem/audio_common.py`](../animation_modem/audio_common.py) |
| Historical V1 | pinned `experiment` history, [`animation_modem/transport.py`](https://github.com/bgencarelle/VideoInterleaving/blob/99d60665688ef6af197be37ebcded57bc36aaf25/animation_modem/transport.py) |
| Historical V2 | pinned `experiment` history, [`animation_modem/transport2.py`](https://github.com/bgencarelle/VideoInterleaving/blob/99d60665688ef6af197be37ebcded57bc36aaf25/animation_modem/transport2.py) |

## Scope and status

- The wire specification is the source of truth for reimplementation claims;
  items marked **UNVERIFIED** require a conformance vector or measurement.
- V1/V2 are historical references. The current V3-family transport owns the
  active `wire`, `wire-hd`, tape, V6, and V6-repeat paths.
- V6 tape-ordered placement is currently a bench-only coder on `WIRE_V6`; it
  is not a separate wire or automatic live receiver candidate.
- Synthetic clean and impairment results do not establish real-tape
  acceptance. Keep generated WAVs, images, and logs in `scratch/`; do not
  commit them.

## Validation commands

Run from the repository root with the project environment:

```bash
.venv/bin/python tools/spec_vectors.py
.venv/bin/python -m unittest modem_tests.test_v6 modem_tests.test_v6_repeat \
  modem_tests.test_v6_tape modem_tests.test_tape_wire -v
git diff --check
```
