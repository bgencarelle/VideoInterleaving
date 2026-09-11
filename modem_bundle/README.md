# Modem extraction bundle

Frozen from `bgencarelle/VideoInterleaving`, branch `modem`, commit
`8e80b8c8b4d9fef8aec7f7482f5ffa02ba68b0db`.

This folder collects the existing code for a later standalone port. Original
project files remain in place. Copied source files are byte-for-byte snapshots;
no decoder, encoding, device-selection or clock behavior was changed here.

## Preserved versions

The entire `source/` tree is the **unpatched baseline**, before the proposed
reference speed correction. This preserves both the version being tested and
the older utility paths that worked when the reference receiver froze.

| Implementation | Location under `source/` | Baseline behavior |
| --- | --- | --- |
| Original image modem | `animation_modem/transport.py` | Original v1 wire format and receiver |
| Current packet transport | `animation_modem/transport2.py` | Used by utility WAV `read`; search-based acquisition |
| Progressive receiver | `animation_modem/progressive.py` | Default live receiver, single-pass payload with inherited acquisition |
| Reference-event receiver | `animation_modem/reference_receiver.py` | Unpatched first-pass implementation; known 1.12x rejection remains |
| Independent pilot prototype | `animation_modem/pilot_clock_prototype.py` | Reference-only experiment; does not carry images; timing bias unresolved |

`patches/modem-reference-speed-fix.patch` is the separately delivered correction.
It is **not applied** to the snapshot. From the VideoInterleaving repository root,
you can check it against the bundled copy without altering the live project:

```bash
git apply --check --directory=modem_bundle/source modem_bundle/patches/modem-reference-speed-fix.patch
```

Keep this baseline intact when starting comparative work. Make a working copy
before applying the optional patch there.

## Contents

- `source/animation_modem/`: all current modem package code, including capture,
  playback, image coding, references, partial display, deadlines and impairments.
- `source/modem_receive.py`, `modem_display.py`, `modem_bake.py`: receiving,
  transmission integration and baked layer access.
- `source/utilities/`: modem checks/probes, converter and shared bake helpers.
- `source/modem_tests/`, `test_modem_integration.py`: preserved focused and
  application-integration tests, with no new claims that they all pass.
- `source/MODEM*.md`: original documentation, including historical limitations.
- Shared support modules: settings/constants, frame selection, shared clock,
  MIDI support, file ordering, and scope helpers imported by the existing baker.
- `source/main.py` and `server_config.py`: preserved application CLI containing
  the modem entry path. Other application modes are not part of this bundle.
- `integration_reference/`: full-application installer, runner and requirements
  for porting reference. These still assume the full project and are not a
  standalone modem installer. The macOS installer is still Homebrew-only;
  MacPorts/Sierra support has not been implemented.
- `SNAPSHOT.json`: provenance, file roles and SHA-256 checksums for every copied
  file, including the separately retained speed patch.

No baked image libraries, WAV recordings, virtual environments or generated
logs are included. Those are external input assets, not modem source code.
No duplicate LTC project was available in this checkout.

## Starting the later port

Use `source/` as the candidate project root. The copied receivers/utilities keep
normal import paths; they do not need wrappers that silently import the parent
checkout. A minimal dependency list is preserved as `requirements-modem.txt`.
Run import checks from that root using your selected Python environment:

```bash
python utilities/check_modem_setup.py
python modem_receive.py --help
python utilities/modem_v2_check.py --help
```

The standalone work still needs to:

1. Extract a modem-only transmitter CLI from `main.py`, retaining the existing
   shared-time scheduler and folder-selection semantics.
2. Separate shared RGBA loading/slab writing from scope-specific helpers.
3. Define standalone settings and package metadata. Live MIDI integration needs
   its optional dependencies; scope-only functions carry additional imports.
4. Provide a dedicated installer, including an explicit Python choice and
   MacPorts support for older Macs. Do not run the archived application setup
   script as a standalone installer.
5. Decide which receiver becomes the default after real-world comparison.

The package retains existing code and historical documentation for comparison;
it does not establish new robustness, performance or Sierra compatibility.
Only copy/checksum integrity, Python syntax and patch applicability were checked
while assembling this bundle. No audio streams, device scans or modem test suite
were run.
