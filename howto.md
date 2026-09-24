# How to test the modes

Practical run + test notes for every `main.py --mode` in VideoInterleaving.
Each mode is **standalone and owns the run** — modes never attach to each other.

## Ground rules (all modes)

```bash
cd /Users/ben/PycharmProjects/VideoInterleaving   # CWD must be the repo root
.venv/bin/python ...                              # always this interpreter
```

- Run everything from the repo root (systemd units, scripts, and tests assume it).
- CLI args that belong to another mode are reported as ignored (`⚠️`), not dropped.
- `logs/` holds per-run logs (stdout/stderr teed there). Generated image lists
  are refreshed at startup; scope can use the manifest in its XY bake instead.
- **Audio:** pass a device by name or index (`--device BlackHole`), never assume
  the default. `--device null` runs headless. Anything opening a REAL device
  must use **BlackHole 2ch**; if it's missing on macOS, `brew install blackhole-2ch`
  and stop — never use built-in speakers for tests.
- **ffmpeg** must be installed for video/clip paths and some screen tests.

## Mode map

| mode | what it does | check it at | tests |
|---|---|---|---|
| `web` | MJPEG stream + monitor | `http://<IP>:8080`, monitor `:1978` | lazy-imports only |
| `ascii` | telnet ASCII video | `telnet <IP> 2323` / `nc <IP> 2323` | `test_ascii_converter_adjustments` |
| `asciiweb` | ASCII over telnet + web | ports `2423`/`2424` (reserved) | lazy-imports only |
| `local` | windowed GL playback | a window opens | lazy-imports only |
| `scope` | XY audio oscilloscope art | scope / `http://127.0.0.1:8890/scope` | scope suite (audio rules!) |
| `modem` | stereo audio image modem | WAV file or live window + JSON | `modem_tests/` + integration |

## web

```bash
.venv/bin/python main.py --mode web --dir ./images_sbs
```

- Stream: `http://<IP>:8080`, monitor: `http://<IP>:1978`. Binds both or exits
  on a busy port (`require_ports()`).
- Needs SBS JPEGs in `--dir` (see Bakes).

## ascii

```bash
.venv/bin/python main.py --mode ascii --dir ./images_tiny
# connect:  telnet <IP> 2323   (monitor on 2324)
```

- Use small images (~150px wide) to save CPU.
- Grading knobs (applied in this order — saturation, contrast, brightness, gamma):

```bash
.venv/bin/python main.py --mode ascii --dir ./images_tiny \
  --ascii-contrast 1.4 --ascii-brightness 1.2 --ascii-gamma 0.9
```

- `1.0` is neutral for all three; the shipped picture is byte-identical at defaults.
- Tests pin the knobs — don't "fix" grading without updating them:

```bash
.venv/bin/python -m unittest tests.test_ascii_converter_adjustments -v
```

## asciiweb

```bash
.venv/bin/python main.py --mode asciiweb --dir ./images_tiny
```

- Telnet + web on the reserved ports **2423/2424** — nothing else may bind those
  (`main.py` errors out if you try). Same grading knobs as `ascii`.

## local

```bash
.venv/bin/python main.py --mode local --dir ./images_sbs
# handy flags: --rotation 90 --mirror
```

- Windowed GL playback; needs a display. No dedicated test module beyond
  `tests.test_lazy_imports` (each mode must import only its own deps).

## scope

```bash
.venv/bin/python utilities/convert_to_xy.py -i ./images -o ./images_xy   # bake once
.venv/bin/python main.py --mode scope --xy-dir ./images_xy --scope-mode stochastic
.venv/bin/python main.py --mode scope --xy-dir ./images_xy --device BlackHole   # real audio
# web preview: http://127.0.0.1:8890/scope
```

- Scope binds nothing (skips `require_ports()` deliberately).
- **Audio safety is the whole game here:** scope tests/tools can block on an
  audio-device prompt without a tty, and bare `Scope()` objects default to the
  speakers. Rules:
  - Never run blanket `discover -s tests` — it collects `test_scope_pair.py`,
    an interactive inspection tool, not a unit test.
  - `test_scope_stochastic.py` / `test_scope_pair.py` only on a box where the
    default output is safe (e.g. BlackHole 2ch).
  - `tools/spec.py`, `tools/bake_advisor.py`, `tools/verify_scope_files.py`
    are diagnostics, not suite members.

## modem (active work)

One wire, one format (`WIRE`: 375–20250 Hz, 15.00 fps, full 2880-slot
`color-dct` picture). No presets; the receiver reads the profile from the header.

```bash
# fastest confidence check — no audio device needed:
.venv/bin/python tools/bench_modem.py
# fixed-pair WAV round trip:
.venv/bin/python main.py --mode modem --modem-dir images_modem \
  --modem-pair 1,0 --modem-wav modem_test.wav --modem-frames 100
.venv/bin/python utilities/modem_v3_check.py read --wav modem_test.wav
# live loop over BlackHole:
.venv/bin/python utilities/modem_v3_check.py live-receive --device BlackHole --channels 1,2 --emulate clean
.venv/bin/python main.py --mode modem --modem-dir images_modem \
  --device BlackHole --modem-channels 1,2
# sender without a bake:
.venv/bin/python modem_screen.py --source test --frames 5 --write out.wav
```

- Needs `images_modem/` containing `modem.json` (see Bakes), else modem mode errors.
- Picture policy: never black — damaged frames show as-is, undecodable frames
  hold the last good one.
- Test suites (run from root):

```bash
.venv/bin/python -m unittest discover -s modem_tests -v
.venv/bin/python -m unittest tests.test_modem_integration tests.test_lazy_imports -v
.venv/bin/python -m unittest tests.test_eq_dispersion -v
```

- Known failures, do not chase: `test_modem_screen` ffmpeg case (needs
  ffmpeg/live capture; moot under the never-black policy), two
  `test_pilot_continuity` drift steps (open wow/flutter work), one
  `test_fit_allocation` expected-failure (`--allocation` vs spread wire).
- `utilities/check_modem_setup.py` verifies modem imports/bindings without
  opening audio devices.

## Bakes (all gitignored, never commit)

| bake | make it with | consumed by |
|---|---|---|
| `*_xy/` scope geometry | `utilities/convert_to_xy.py -i ./images -o ./images_xy` | `--mode scope` (runtime never opens a photo) |
| `*_modem/` DCT slabs + `modem.json` | `utilities/convert_to_modem_dct.py` | `--mode modem`, `modem_screen.py`, bench |
| `images_sbs/` SBS JPEGs | (your SBS pipeline) | `web`, `local` |

## Troubleshooting

| symptom | likely cause |
|---|---|
| `require_ports()` exits / busy port | another mode (or run) holds it; scope is the only mode that binds nothing |
| audio prompt hangs a test | scope test without tty/device — give `--device`, use BlackHole, or skip it |
| sound from laptop speakers | something opened the default device — stop, switch to BlackHole 2ch |
| modem mode errors on start | `images_modem/modem.json` missing — bake first |
| `No module named animation_modem` | wrong CWD or wrong interpreter — root + `.venv/bin/python` |
