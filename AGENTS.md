# AGENTS.md

## Project shape

- This is a flat Python application rooted at `main.py`; run commands from the
  repository root because imports, tests, and services assume that CWD.
- `main.py --mode` runs one standalone mode (`web`, `ascii`, `asciiweb`, `local`,
  `scope`, or `modem`). Keep mode-specific changes in that mode unless the task
  explicitly crosses boundaries.
- The baked application modem (`main.py --mode modem`) and standalone V7 live
  sender/receiver (`tools/v7_live.py`, `vi.modem-*`) are different pipelines.
  Read `docs/MODEM_MODE.md` for the baked mode and
  `docs/transport_v7_spec.md` for V7 wire/live behavior.
- `animation_modem/` is a transport package: do not add imports from app UI,
  settings, or status modules.
- `main.py` parses CLI arguments at module import and may exit. Tests importing
  it must control `sys.argv` or use a subprocess. Lazy per-mode imports are
  intentional and pinned by `tests/test_lazy_imports.py`.
- Check the current branch and worktree before editing; preserve existing
  unrelated changes.
- Scope defaults and CLI parameters live in `settings.py` and `main.py`; the GUI
  launcher assembles them in `tools/scope_launcher_gui.py`. Before scope changes,
  read `docs/SCOPE_UPGRADE.md` and start from its latest Handoff. Treat current
  defaults as D0; keep its charter/stages frozen and append evidence/check-ins,
  following its Numba, realtime/headless, baseline and stage requirements.

## Environment and artifacts

- Use the repository `.venv/bin/python` (Python 3.11+); it uses
  `--system-site-packages`. Device-backed audio uses `sounddevice` over
  PortAudio; `requirements.txt` includes modem and scope Python requirements.
  Native libraries such as PortAudio are OS packages.
- `xvfb-run -a` is available for GUI/GL checks that need an X display. For
  software-rendered GL, use `xvfb-run -a env LIBGL_ALWAYS_SOFTWARE=1 ...`.
- `./scripts/setup_app.sh` performs system setup and may configure systemd.
  `utilities/check_modem_setup.py` verifies modem imports without opening audio.
- No `pyproject.toml`, formatter, linter, or type checker is configured.
- Put generated measurements, screenshots, and temporary scripts in repo-local
  `tmp/`; it is gitignored. Never commit generated bakes (`*_xy/`, `*_modem/`).
- Baked-source `main.py --mode scope` requires an XY bake; `--scope-source images`
  uses runtime thumbnails instead. The baked modem mode requires a modem bake
  with `modem.json`. Use `utilities/convert_to_xy.py` and
  `utilities/convert_to_modem_dct.py` for the respective bakes.

## Existing audio validation tooling

- User-directed validation scope: use the existing ALSA/Pulse software loopback
  and built-in browser/native/CPU scope visualizers. Physical/offline tests are
  the user's separate work (see `docs/SCOPE_UPGRADE.md` A043). Do not make
  physical hardware acquisition or scope hookup the next agent task or a
  prerequisite for this software delivery. Use the available visual inspection
  tools and renderer artifacts; report a disconnected browser tool precisely
  rather than assuming the application or audio tooling is unavailable.
- Verified in this workspace on 2026-10-07: `.venv/bin/python` imports
  `sounddevice` 0.5.6 and loads PortAudio V19.6. The ALSA host API exposes
  `pulse` (then index 0) and `default` (then index 1). PortAudio is available;
  hardware scope/DAC access is a separate question. An underflow or stream
  failure is not evidence that PortAudio is missing.
- Reuse the existing route: PortAudio `pulse` -> PulseAudio null sink `loop`
  -> `loop.monitor`. The verified default sink/source are `loop` and
  `loop.monitor`; the sole sink is `module-null-sink`, stereo, 44.1 kHz.
  Refresh these observations with `pactl info`, `pactl list short sinks`, and
  `pactl list short sources` before output. Prefer the device name `pulse`
  over a remembered numeric index. If routing differs, investigate the
  existing route before proceeding.
- Opening/reopening a test stream on this existing route is normal. Do not
  create additional sinks/devices, load loopback modules, change defaults, or
  reinstall audio tooling merely to run the next validation. Device-change
  experiments need their own explicit task scope; routine tests reuse `loop`.
- `pactl` inspects routing and `parec` captures `loop.monitor`; both are
  installed. Reuse `parec` monitor capture rather than adding a concurrent
  PortAudio input stream by default. The existing harness documents scheduling
  problems observed with a second PortAudio stream.
- Use channels 1/2 for this stereo monitor. PortAudio advertising 32 channels
  does not establish a 24/25 monitor path. Explicit 48/96-kHz output streams
  have run through this route, while monitor capture is 44.1 kHz. NullStream
  (`--device null`), PortAudio-to-null-sink, and physical output are distinct
  validation paths; label evidence accordingly.
- Existing local harnesses: `tmp/scope_s0_harness_v3.py` (callback observation
  and `parec` capture), `tmp/scope_s7_pa_route_case.py` (rate/block screening),
  `tmp/scope_s7_pa_video_case.py` with `tmp/scope_s7_pa_matched_run.py`
  (D0/integrated video through the PortAudio null-sink route),
  `tmp/scope_s7_final_matched.py` (NullStream D0/integrated moving-video
  comparison), and `tmp/scope_s7_software_matrix.py` (mode/settings checks).
  Inspect and reuse them when present; they are gitignored local artifacts, not
  guaranteed in a fresh checkout. Results and limitations are in
  `docs/SCOPE_UPGRADE.md`, especially Appendices A059/A060. The rate-case runner
  reuses output filenames, so preserve prior evidence or assign unique filenames
  before repeat runs.
- `xvfb-run`, `ffmpeg`, and `ffprobe` are also installed. Check tooling with
  imports, enumeration, and `command -v` before declaring it unavailable.

## Modem invariants

- V7 timing acquisition is edge/pulse-counted through
  `animation_modem/transport3.py::measure_pulses`; preserve it rather than
  replacing it with FFT correlation. V7 packets use EOF framing by default.
- Fold-500 carries stereo M/S data. A single raw M/S leg is not ordinary mono;
  preserve the stereo interpretation during dropout or recovery handling.
- On damage, display a partially decoded frame as-is; if a frame cannot be
  decoded, hold the last good frame. Do not add a black-frame fallback.
- Analog tape is the target medium, but synthetic channels are not tape
  validation. Keep wire/decode design deterministic and validate real tape on
  hardware; see `docs/transport_v7_spec.md`.

## Tests and audio safety

Run from the repository root:

```bash
.venv/bin/python -m unittest discover -s modem_tests -v
.venv/bin/python -m unittest tests.test_modem_integration tests.test_lazy_imports -v
.venv/bin/python -m unittest tests.test_ascii_converter_adjustments tests.test_ascii_scaling -v
```

Run a focused modem test, for example:

```bash
.venv/bin/python -m unittest modem_tests.test_v7_live_input -v
```

Do not blanket-discover `tests/`: it also contains interactive scope inspection
and audio-device-dependent tests. In particular, `tests/test_scope_pair.py` is
an inspection tool. Use `--device null` for headless scope checks; it does not
verify PortAudio/device behavior. Run device-backed audio tests only through an
explicitly verified virtual/loopback route, never physical speakers.
`server_config.py` is the source of truth for application port assignments.
