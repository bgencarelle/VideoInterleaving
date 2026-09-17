# AGENTS.md

Flat, single-package Python app (no `pyproject.toml`/setup.py). Entry point is
`main.py`. Active modem work runs on branch `experiment`; the mode docs
(`docs/MODEM_MODE.md`, `docs/SCOPE_MODE.md`) are authoritative for their
subsystems.

## Layout

- Root holds the entry point, the shared core (`settings.py`, `shared_state.py`,
  `server_config.py`, `index_calculator.py`, `folder_selector.py`), and the mode
  engines (`image_display.py`, `web_service.py`, `ascii_*`, `scope_display.py`,
  `modem_display.py`, ...). Imports are flat and absolute; CWD is assumed to be
  the repo root everywhere (systemd units, scripts, and tests all rely on it).
- `docs/` — documentation except this file and `README.md`.
- `scripts/` — shell entry points (`setup_app.sh`, `run_app.sh`, kiosk setups).
  They re-anchor to the repo root internally; run them from anywhere.
- `tools/` — standalone diagnostic/inspection tools. Nothing imports them; the
  ones that need app modules insert the repo root into `sys.path` themselves.
- `tests/` — root-level test suites; `modem_tests/` is the modem package suite.

## Modes

`main.py --mode` accepts `web | ascii | asciiweb | local | scope | modem`. Each
mode is **standalone and owns the run**; modes never attach to each other.
Scope needs no GL/TurboJPEG/image-loader; modem needs no ports/GL; web/ascii
need no audio. CLI args that only mean something in another mode are reported
as ignored (with a `⚠️`), not silently dropped.

**Touch only the mode you are working on.** Do not modify any other
VideoInterleaving mode (`web`, `ascii`, `asciiweb`, `local`, `scope`) unless the
task is explicitly about it. Current active work is the **modem**; keep changes
inside the modem subsystem (`animation_modem/`, `modem_*.py`,
`utilities/modem_*`, `modem_tests/`) and leave everything else alone.

## Long-term direction (modem)

- The modem transmits to ANALOG TAPE-compatible media: noisy, warbly,
  band-limited. Long-term goal: a tape-tuned wire at one tape-compatible
  bandwidth, preserving the baked 80x96 resolution, with classical-computing
  (cheap, deterministic) decode and stable recovery under noise + wow/flutter.
  Recovery priority on bad tape: the image must still DECODE with acceptable
  color; losing detail/softness is fine. The V4 goal is a wire whose whole
  band fits inside the tape bandwidth, so the picture rides fully within the
  medium instead of relying on graceful degradation when the band exceeds it.
- The picture path MUST NEVER default to, emit, or display black. A damaged or
  partially decoded frame is shown as-is; a frame that cannot be decoded at all
  holds the last good frame. There is no black fallback anywhere in the modem
  video path, so black-frame guards are not a requirement.
- Timing sync is edge/pulse-counted, LTC-style — NOT FFT-correlated.
  `transport3.measure_pulses` is the designated acquisition path
  (`Receiver(pulse_only=True)` is the default). Keep it so. The waveform/
  correlation paths (`_acquire`, `_correlate`, `measure_speed`, receiver-side
  `_fit_preamble`) are fallbacks to be benchmarked and likely retired. The only
  FFTs that should remain are unavoidable, fixed-size ones: the per-symbol
  `rfft(n=128)` OFDM demod and the inverse DCT in `SourceCoder.inverse`.
- The `V3_PRESETS` table is INTERIM. Plan: find the most tape-compatible
  (preset, profile) within the existing setup, set it as the default, then
  refactor the preset sprawl away. Design and decode validation runs as
  SYNTHETIC IDEAL tests in `tools/bench_modem.py` (round-trip decode cost and
  fidelity on a clean wire). Tape emulation is kept to a minimum — it cannot
  be made realistic, and real tape measurements are the user's job.

## Environment

- `.venv` is created with `--system-site-packages` (needed for Tk). Always use
  `.venv/bin/python`; on macOS the Homebrew `python-tk` must match the Python
  version used to create the venv. `scripts/setup_app.sh` (sudo) does setup + systemd.
- `utilities/check_modem_setup.py` verifies modem imports/bindings without
  opening audio devices.
- No linter, typechecker, or formatter is configured. Inline prose comments
  explaining invariants are the norm — keep that style.

## main.py quirks

- **CLI is parsed at import time**: `cli_args, log_filename = configure_runtime()`
  runs at module scope (main.py:786). Importing `main` parses `sys.argv`,
  patches `settings` globals, and can `sys.exit()` (e.g. busy ports). Tests that
  import `main` must drive it via args in-process or as a subprocess.
- **Lazy per-mode imports are an invariant** (`tests/test_lazy_imports.py` pins it):
  a mode must only import its own dependencies, at the point of use. Do not add
  module-level imports of heavy/optional libs to `main.py`.
- Config flow is CLI → writes `settings.<GLOBAL>` → modules read `settings.*`
  at runtime. No config framework. `settings.py` re-exports
  `constantStorage/*`.

## Ports and caches

- `server_config.py` owns port assignments: web monitor 1978 / stream 8080;
  ascii 2323 (monitor 2324); asciiweb 2423/2424; scope 8890. `require_ports()`
  exits on a busy port, but scope mode deliberately skips it (it binds nothing).
  Ports 2423/2424 are reserved for asciiweb.
- Image scan caches live in `_cache/generated_lists_<src>_<mode>_<port>/` and
are wiped at startup; `--rebuild` forces regeneration. `logs/` holds per-run
logs (stdout and stderr are teed there).

## Bakes (all gitignored, never commit)

- `*_xy/` — scope geometry, baked once by `utilities/convert_to_xy.py`. The
  compact bake stores 128px luminance+alpha and stipple coords, not full-res
  frames; scope runtime never opens a photo.
- `*_modem/` — RGBA DCT slabs from `utilities/convert_to_modem_dct.py`;
  must contain `modem.json`, else `main.py --mode modem` errors.
- `images_sbs/` — SBS JPEGs; `*_modem/`/`*_xy/` derive from `--dir`.

## Tests

`unittest` style, run from the repo root (tests import `animation_modem.*`,
`utilities.*`, and the app modules directly):

```bash
.venv/bin/python -m unittest discover -s modem_tests -v
.venv/bin/python -m unittest tests.test_modem_integration tests.test_lazy_imports -v
.venv/bin/python -m unittest tests.test_ascii_converter_adjustments tests.test_ascii_scaling -v
```

- `test_ascii_converter_adjustments` / `test_ascii_scaling` live in `tests/`,
  outside `modem_tests/`: they pin the ascii grading knobs
  (`--ascii-contrast`/`--ascii-brightness`/`--ascii-gamma`) and the
  neutral-at-1.0 contrast that keeps the shipped picture byte-identical.
- Failing on the current checkout (pre-existing, matches `.pytest_cache`):
  `test_modem_screen` ffmpeg-failure (needs ffmpeg/live capture) — **ignore
  this one**, the video path never uses black (it shows the damaged frame or
  holds the last good one), so the black-frame guard it checks is moot;
  `test_pilot_continuity` smooth-drift steps (-0.25, 0.18),
  `test_tape_band` mid-band-header trade.
- Some scope tests/tools block on an audio-device prompt when run without a
  tty or configured `--device` (e.g. `tests/test_scope_pair.py` — that one is
  an interactive inspection tool, not a unit test). Never blanket
  `discover -s tests` — it would collect it.
- `tests/test_scope_stochastic.py` and `tests/test_scope_pair.py` each
  construct a bare `Scope()` (default device = the speakers). Run them only
  on a box where the default output is safe (e.g. BlackHole 2ch).
- `tests/test_scope_pair.py`, `tools/spec.py`, `tools/bake_advisor.py`,
  `tools/verify_scope_files.py` are inspection/diagnostic tools, not part of
  the suite.

## Independence constraints

- `animation_modem/` is a self-contained transport package: keep it free of
  app imports (renderer, settings, status).
- Audio output uses `sounddevice`; give a device by name (`--device BlackHole`)
  or index, never assume the default. `--device null` runs scope headless
  (browser renders the samples). Anything that opens a REAL device must use
  `BlackHole 2ch`. BlackHole is not stock macOS: if this is a Mac and
  BlackHole is missing, STOP and ask the user to install it
  (`brew install blackhole-2ch`) — never let tests use the built-in speakers.