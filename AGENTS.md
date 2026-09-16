# AGENTS.md

Flat, single-package Python app (no `pyproject.toml`/setup.py). Entry point is
`main.py`. The current branch is `main` (modem — and scope inside it — was
merged back); the mode docs (`MODEM_MODE.md`, `SCOPE_MODE.md`) are
authoritative for their subsystems.

## Modes

`main.py --mode` accepts `web | ascii | asciiweb | local | scope | modem`. Each
mode is **standalone and owns the run**; modes never attach to each other.
Scope needs no GL/TurboJPEG/image-loader; modem needs no ports/GL; web/ascii
need no audio. CLI args that only mean something in another mode are reported
as ignored (with a `⚠️`), not silently dropped.

## Environment

- `.venv` is created with `--system-site-packages` (needed for Tk). Always use
  `.venv/bin/python`; on macOS the Homebrew `python-tk` must match the Python
  version used to create the venv. `setup_app.sh` (sudo) does setup + systemd.
- `utilities/check_modem_setup.py` verifies modem imports/bindings without
  opening audio devices.
- No linter, typechecker, or formatter is configured. Inline prose comments
  explaining invariants are the norm — keep that style.

## main.py quirks

- **CLI is parsed at import time**: `cli_args, log_filename = configure_runtime()`
  runs at module scope (main.py:786). Importing `main` parses `sys.argv`,
  patches `settings` globals, and can `sys.exit()` (e.g. busy ports). Tests that
  import `main` must drive it via args in-process or as a subprocess.
- **Lazy per-mode imports are an invariant** (`test_lazy_imports.py` pins it):
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
.venv/bin/python -m unittest test_modem_integration test_lazy_imports -v
.venv/bin/python -m unittest test_ascii_converter_adjustments test_ascii_scaling -v
```

- `test_ascii_converter_adjustments` / `test_ascii_scaling` live at the repo
  root, outside `modem_tests/`: they pin the ascii grading knobs
  (`--ascii-contrast`/`--ascii-brightness`/`--ascii-gamma`) and the
  neutral-at-1.0 contrast that keeps the shipped picture byte-identical.
- Failing on the current checkout (pre-existing, matches `.pytest_cache`):
  `test_modem_screen` ffmpeg-failure (needs ffmpeg/live capture),
  `test_pilot_continuity` smooth-drift steps (-0.25, 0.18),
  `test_tape_band` mid-band-header trade.
- Some scope tests/tools block on an audio-device prompt when run without a
  tty or configured `--device` (e.g. `test_scope_pair.py` — that one is an
  interactive inspection tool, not a unit test).
- `test_scope_pair.py`, `spec.py`, `bake_advisor.py`, `verify_scope_files.py`
  are inspection/diagnostic tools, not part of the suite.

## Independence constraints

- `animation_modem/` is a self-contained transport package: keep it free of
  app imports (renderer, settings, status).
- Audio output uses `sounddevice`; give a device by name (`--device BlackHole`)
  or index, never assume the default. `--device null` runs scope headless
  (browser renders the samples).