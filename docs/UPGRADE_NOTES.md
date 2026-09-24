# Upgrade Notes

Audit notes for the current `modem-v7-integration` checkout, compared with
`main` at `0e30eb25` on 2026-09-23. This records observed behavior and agreed
cleanup direction; it is not a promise that every listed refactor is complete.

## Mode flow

`main.py` owns startup configuration and dispatch. Each mode then runs on its
own rather than attaching to another mode.

| Mode | Runtime path |
| --- | --- |
| `web` | Monitor and stream services, then `image_display` |
| `local` | Monitor service, then `image_display` |
| `ascii` | Telnet and stats services, then `image_display` |
| `asciiweb` | WebSocket and monitor services, then `image_display` |
| `scope` | `scope_display`, with an optional web monitor; scope does not bind the monitor port itself |
| `modem` | `modem_v7_display`; bypasses common image-list, port, and display setup |

## Clock modes and MIDI history

The current clock constants are in `constantStorage/midi_constants.py`:

| Value | Name | Current state |
| --- | --- | --- |
| `0` | `MTC_CLOCK` | Legacy MIDI Time Code path; no longer initialized by normal app startup |
| `1` | `MIDI_CLOCK` | Legacy MIDI pulse path; no longer initialized by normal app startup |
| `2` | `MIXED_CLOCK` | Legacy combined MIDI/MTC path; no longer initialized by normal app startup |
| `3` | `CLIENT_MODE` | Client-index path was removed; the current index calculator treats this as free clock |
| `255` | `FREE_CLOCK` | Active wall-clock path and current default |

**Intended restoration:** live MIDI controls where notes select folders and
pitch bend continuously controls the float/overlay layer's opacity over the
main layer, with external clock choices of MTC or LTC. Do not mistake the old
`MIDI_CLOCK`/`MIXED_CLOCK` beat-pulse modes or the MIDI-file-duration preset
for that target. Control messages are a separate logical path from clock
selection (MTC can share the same physical MIDI input). The current free clock
remains the working default unless deliberately changed.

MIDI/MTC was real, previously wired functionality—not a feature invented by
the modem work. The 2024-era `image_display` path selected a clock with
`set_clock_mode()`, opened the MIDI port with
`midi_control.midi_control_stuff_main()`, and polled MIDI messages during
display updates. For example, that wiring is present in commit `5c8ca9de`
(2024-11-13).

The connection appears to have been broken during the display refactor in
`8071dd46` (2025-03-16). That change removed the display path's MIDI import,
clock-mode setup, MIDI-port startup, and client startup. The newer
`index_calculator` retained MIDI calculation code and `set_clock_mode()`, but
normal `main.py` startup never calls the setter. Consequently
`index_calculator.midi_mode` remains false and normal local/web/ASCII/scope
playback uses the free clock. Setting `settings.CLOCK_MODE` to a MIDI value can
still send image-list setup through the old preset path, but does not activate
MIDI playback. `settings.MIDI_MODE` is also unused.

The modem clock option is a separate, much newer addition. `--modem-clock` and
`--modem-frame-duration` were introduced with the modem sender in
`364c2928` (2026-09-09), and are present on `main` because the modem work was
merged there. Checking `main` does not reveal an alternate working MIDI startup
path: its sender calls `set_clock_mode()`, but does not open a MIDI input port.
The modem-only option had two concrete failures: an explicit MTC value of `0`
used to enter the interactive prompt, and live MIDI selection did not open
`midi_control.input_port`. `set_clock_mode()` now accepts zero as an explicit
value, and the live V7 sender opens and closes the selected input. WAV export
rejects non-free clocks rather than silently computing its indices from the
free clock. `CLIENT_MODE` is rejected because its client-index path was removed.
The intended live MIDI controls and MTC/LTC clock choices remain separate
behavior from these legacy clock flags.

The repository has MTC quarter-frame parsing in `midi_control.py`, but no
standard SMPTE LTC decoder. The modem's `animation_modem/transport3.py`
edge-counts its own V7 biphase preamble; despite the LTC-style analogy in modem
notes, that is not an LTC timecode input path. LTC clock support would require
an audio-input decoder, separate from modem receive framing.

The old `README.md` advertised MIDI/MTC and client/server index synchronization,
while `docs/SCOPE_MODE.md` said scope stayed in sync with MIDI/MTC. Those claims
did not describe current runtime behavior and have been corrected to distinguish
the free-clock runtime from the intended MTC/LTC restoration.

### MIDI performance and folder controls

The old MIDI path also controlled which folders were composited. In the
2024-era display code, after reading MIDI messages:

- Note-on pitch selected the main/face folder using
  `(note % 12) % main_folder_count`.
- CC1/mod-wheel value selected the float folder using
  `int(value / 127 * float_folder_count) % float_folder_count`.
- This folder-control branch was selected whenever the clock was not
  `FREE_CLOCK`; it was not an independent MIDI-control enable switch. Note
  velocity and channel were not used by this mapping, and the top-level note
  range was limited to 12 pitch classes.

The current `folder_selector.py` still contains this MIDI-driven mapping, and
`midi_control.py` still records note-on, note-off, modulation, and BPM data.
However, the normal app no longer starts the MIDI reader or copies its events
into the shared control state. With a non-free `CLOCK_MODE`, the folder
selector can enter the MIDI branch but only sees its default values. Thus the
folder hook is retained code, not currently live behavior.

The desired live control is narrower and differs from that old mapping: notes
on MIDI channel 1 select face/main folders; notes on channel 2 select float
folders. These are the user-facing channel numbers; Mido reports them as
zero-based channels 0 and 1. Pitch bend controls the float layer's opacity
rather than selecting a float folder with CC1. The pitchwheel handler is
currently a stub. Existing compositors already honor per-pixel float alpha,
but there is no shared live opacity control: the image renderer has GPU and
CPU compositors, scope combines layer alpha into its luminance composite, and
modem builds a composite frame before encoding it. A restored control must
apply the pitch-bend opacity consistently at the applicable composition
point. The exact note-to-folder map and pitch-bend range/neutral behavior are
still to be specified.

Other legacy MIDI handlers have narrower effects: Start, Continue, Stop, Song
Position, and Reset reset parts of the clock counters; clock messages calculate
BPM and advance MIDI-clock indexing; MTC quarter frames update timecode and
index state. Note-off is stored but not consumed by the folder selector;
program-change and pitchwheel handlers are stubs, while active sensing only
prints. A replacement should decide deliberately which of these controls to
preserve rather than treating every handler as an implemented feature.

### Legacy sorting and frame-duration calculator

The remembered “calculator” is likely the old `calculators.py`, which combined
image-list selection/loading with frame-duration presets and index calculations.
It was deleted in `be447977` (2025-12-19). Its responsibilities are now split:

- `make_file_lists.py` scans and naturally sorts image paths, creates the
  interleaved CSV lists, selects/loads those lists, and retains the old
  video-length preset flow (`get_midi_length()`, `setup_video_length()`, and
  `initialize_image_lists()`).
- `index_calculator.py` holds the free-clock and MIDI-clock index calculations
  and the `frame_duration` value used by the old MIDI mapping.

The preset flow's “MIDI-derived length” means reading a `.mid` file's duration
and multiplying it by an assumed 30 fps to estimate a video length. It is not
the live MIDI clock or note/mod-wheel folder-control path. This belongs to the
earlier video-synth sequencing workflow: MIDI supplied a composition duration,
which was converted into a playback-length/frame-scaling preset for the sorted
image sequence. That legacy prompt and preset path still runs for non-
`FREE_CLOCK` settings even though the normal app no longer initializes live
MIDI clock input. Natural sorting itself is still active: the XY and modem
bakers use helpers from `make_file_lists.py`, so that module cannot be
discarded as an obsolete calculator without moving those shared helpers first.

Other surviving video-synth lineage:

- `folder_selector.py` still performs the randomized face/float changes and
  rest periods during free-clock playback. This behavior is used by the current
  image, scope, and modem-source paths; it is active composition behavior, not
  dead sorting/calculator code.
- The old client-index implementation is gone (`index_client.py` was removed
  in December 2025), but `CLIENT_MODE` remains in the clock constants and the
  selector still treats non-free clock values as the MIDI-control branch. In
  current playback, `CLIENT_MODE` falls through to the free clock and has no
  client synchronization.
- `globals.py` still defines a second `midi_data_dictionary`; current code has
  no readers of that copy. The live MIDI module has its own dictionary, which
  is the one `index_calculator` would consume if the MIDI path were restored.

## Confirmed cleanup direction (applied)

- **Keep startup list generation fresh.** Generated-list caches are cleared at
  startup and the list-reuse branch and `--rebuild` switches have been removed.
  Scope can bypass image scanning when its XY bake supplies the folder
  manifest; the direct `scope_display.py` bootstrap follows the same policy.
- **Remove no-op modem flags.** `--modem-profile` and `--modem-numbered` (`-f`)
  are no longer registered, and the modem examples no longer pass them. The
  current V7 sender did not consume these options.
- **Remove unused settings.** `HTTP_MONITOR` and `SCOPE_MODE` are no longer
  defined or set; neither had runtime readers.
- **Keep scope independent of graphics dependencies.** `main.py` now skips
  `display_manager` import and resolution cleanup for scope. A lazy-import test
  blocks the display/GL modules while exercising the scope startup path.
- **Retain, but document, import-time startup side effects for now.** Importing
  `main.py` runs `configure_runtime()`, parses `sys.argv`, mutates settings,
  creates/logs to a runtime file, and later redirects standard streams. Tests
  that need configuration currently extract the function rather than importing
  `main`.

## Other cleanup and compatibility notes

- `SERVER_MODE` and `ASCII_MODE` still have runtime readers in the image and
  display code; do not remove them until those consumers are migrated.
- Scope compatibility settings such as `SCOPE_RASTER`, `SCOPE_YT`, and
  `SCOPE_YT_TRIGGER_US` still have readers. They are not dead merely because
  their preferred CLI spelling changed.
- Normal-mode engine exceptions are logged, then `main.py` exits with status 1
  after cleanup so service managers can detect and restart failed runs.
- `main.py` and `scope_display.py` each have a startup path for scope. Direct
  `python scope_display.py` is documented, so it is not dead code, but the two
  bootstraps can drift.

## Utilities and stale references

Keep the utilities that produce or validate assets required by a mode:

- `utilities/convert_to_xy.py` builds the scope libraries.
- `utilities/convert_to_modem_dct.py` builds modem assets.
- `utilities/bake_assets.py` contains the shared `write_slab()` used by the
  modem baker. Its standalone RGBA-slab CLI may be old, but the module itself
  is still required.
- `utilities/check_modem_setup.py` is called by the setup script.
- `utilities/audio_levels.py` and the scope verifiers are useful diagnostics.

The scripts in `utilities/archive/` are already explicitly archived. The other
image-processing scripts are standalone operator tools, not runtime
dependencies; lack of imports alone is not enough reason to remove them.

The utility documentation is stale. `utilities/README.md` omits the current
mode-critical bakers and describes the old slab workflow as an active asset
pipeline. Other docs refer to paths missing from this checkout, including
`tools/convert_to_sbs_fixed.py`, `utilities/modem_v3_check.py`,
`utilities/modem_pitch_probe.py`, and `tools/bench_modem.py`. There are also two
scope-verifier scripts: `tools/verify_scope_files.py` is the newer, broader
copy; `utilities/verify_scope_files.py` has older expectations.

## Test notes

Run the modem unit suite and the targeted integration/lazy-import suites using
`.venv/bin/python`. `modem_tests/test_modem_clock.py` covers explicit MTC value
zero and V7 MIDI-input initialization; normal-mode MIDI startup remains
inactive. Do not run interactive scope inspection tools as part of a blanket
test discovery; see `AGENTS.md` for device and test constraints.
