# V7 sender/receiver GUI profile audit

## Change ledger

Baseline: pushed `e2b412429c993f133a8bf0ff14f58d514a6b5834`.

| File | Before | After | Reason / dependency | Coverage |
|---|---|---|---|---|
| `tools/v7_live.py` | Historical shared-ID-2 dispatch labelled both adaptive profiles through an inherited wrapper and required metadata subtype inference. | Superseded by distinct identities: ID 2 adaptive mono, ID 4 adaptive stereo; sender readiness reports selected profile and receiver dispatches current validated wire identity. No subtype wait. | See the newer identity/route ledger row below; signature validation remains required. | Device-free sender identity and receiver transition tests; historical live captures are explicitly marked. |
| `modem_tests/test_v7_adaptive_profile_dispatch.py` | No assertions for unknown subtype, deferred switch, adaptive stereo other-leg dispatch, or GUI command's nested adaptive flags. | Adds those assertions, including preserving the existing legacy-profile switch notification. | Locks down the dispatch and labeling changes above. | Passed (5 tests in the focused module). |
| `tools/v7_send_gui.py` | Kernel chooser listed registry kernels without filtering against the selected profile's parameter defaults. Profile changes silently replaced a valid-but-incompatible saved kernel. Start validation could fall through to runtime rejection after GUI selection. | Chooser filters each offered profile by actual `PROFILE_DEFAULTS`, preserving `reference`; retained incompatible selections show a notice and are excluded from choices. Removed kernels migrate to a valid profile default. Starting with an incompatible retained value now raises the kernel's profile-specific error before launch. A/B descriptions use the same filtered set. | Runtime probe showed `antiring` has mono Aspect settings but no adaptive-stereo settings; registry eligibility alone had been insufficient. Keep a user's explicit saved kernel when switching away and back, while preventing it from being started on a profile for which it has no settings. | Sender-kernel GUI module, 41 tests in final targeted run. |
| `tools/v7_send_gui.py` / `modem_tests/test_v7_send_gui_kernels.py` | When `reference` was explicitly selected for a profile with a non-reference default, command construction omitted `--dct-kernel reference`; runtime then silently reapplied the profile default. | Send `--dct-kernel reference` only for an explicit reference selection that overrides a profile-specific non-reference default. General V7 still sends no redundant flag. | Ensures GUI selection reaches the sender process; live evidence labelled with a requested `reference` but no CLI flag actually used the profile default and is not treated as a reference-kernel comparison. | `test_reference_overrides_a_nonreference_profile_default`; existing general reference no-flag test remains. |
| `modem_tests/test_v7_send_gui_kernels.py` | No adaptive-profile chooser filter, retained-choice migration, or fail-closed profile/kernel pairing assertion. | Adds chooser assertions for adaptive mono/stereo, explicit preservation/notice behavior, and start rejection for an unsupported retained kernel. | Regression coverage for the GUI/runtime mismatch above. | 41 tests passed. |
| `dct_kernels/gaussian.py` | Lowest mean natural-image score in the matched shipped-kernel sweep (`-43.77` ideal vs `-26.71` reference; `-55.69` bilinear vs `-45.28` reference); profile-specific Aspect mono/stereo tuning was effectively reference/no-op. | Removed from the registered kernels and its runtime/GUI tests. Existing GUI preferences migrate to that profile's existing default with a visible “no longer installed” notice. | Narrow removal of a measured low-quality, redundant choice. No profile default or transform was retuned. Historical scores and rationale are in `docs/V7_KERNEL_DEFAULTS_TUNING.md`. | Kernel test suite and retired-preference migration test. |
| `tools/v7_nested_experiment.py` / `modem_tests/test_v7_nested_gui_patch.py` | The `set_default=False` GUI install log printed the process-wide stock `nested_fold.TABLES` path as though it were the table selected for the adaptive decoder. Stale saved preference values for an old stock table path were not explicitly pinned as irrelevant. | Diagnostic separately prints resolved `selected_tables` path and SHA-256, stock `default_tables` path, and selected geometry. Test confirms the receiver GUI continues to use the bundled table even if a stale preferences file contains `nested_tables.npz`. | Avoids mistaking an unchanged legacy cache default for the chosen GUI table; the table field is initialized from the bundled candidate and is not restored from preferences. No receiver-GUI file-picker code was changed. | Tests assert selected path/hash, unchanged default path, and stale-preference immunity. |
| `animation_modem/transport3.py`, `test_modem_v7/tone_code.py`, `tools/v7_profile_registry.py`, `tools/v7_live.py`, `tools/v7_receiver_audio.py` | Adaptive mono and stereo both used ID 2; receiver inferred subtype from metadata and sticky profile state could select a decoder/side before current-stream identity was established. Route evidence and paired-leg state could survive a profile switch or input restart. | Central registry assigns ID 2 to adaptive mono and reuses retired ID 4 (`MONO_500`) for adaptive stereo. Receiver dispatches by validated ID; adaptive stereo owns both legs and never exposes one as passthrough. Saved profile is only a startup hint. Switches and capture restarts clear stale identity, side, layout, route and paired-tail state; adaptive table signature validation remains required. | Separate wire identities remove the adaptive subtype wait; OS audio rates and Aspect Fold 500/kernel defaults are preserved. Historical ID-4 Mono-500 meaning is retired; old recordings are not a compatibility target. | Device-free identity/transition tests cover mono → stereo → Aspect Fold 500 → mono, sticky preferences, restart reset and router passthrough. Existing stereo-slice tests cover partial-leg and hold-last behavior. |
| `tools/v7_receiver_gui.py`, `tools/v7_native_file_dialog.py`, `modem_tests/test_v7_native_file_dialog.py` | Save-folder selection initialized Tk in the GLFW process. The owner's macOS traceback aborts in Tk 9's Cocoa initialization (`NSApplication macOSVersion`). | macOS uses a native `osascript` folder picker; other platforms isolate Tk in a child interpreter. Cancellation preserves the current selection, and helper failures become ordinary GUI errors. | Avoid loading competing GUI toolkits into one Cocoa application. No decode, audio or image-processing changes. | Five dialog tests plus the receiver-GUI suite pass. Native macOS execution is not available on this Linux host; Linux also lacks Tk, so an actual Tk dialog smoke check was unavailable. |

The status showed these pre-existing dirty files before this work and they were not edited here: `AGENTS.md`, `docs/V7_FIXES.md`, `tools/v7_nested_build.py`, `tools/v7_nested_corpus.py`, `tools/v7_sk_study.py`, `generated_img_lists.source-manifest.json.gz`, and `modem_tests/test_v7_curvature.py`. No commit or push was made.

## Live profile evidence

The historical live captures in this section predate distinct adaptive IDs;
their ID-2 stereo labels are not evidence for the current mapping. The current
ID-2 mono / ID-4 stereo identity map and retired ID-4 meaning are recorded in
the follow-up ledger row above. The live sequence and saved-profile restart
below now provide end-to-end evidence for that mapping.

The distinct-ID GUI sequence was run through the real sender GUI, 96-kHz
PulseAudio sender output on channels 24/25, receiver AUTO, and viewer using the
repository-picture source. The receiver confirmed adaptive mono ID 2 on the
right video leg, adaptive stereo ID 4 with both channel modes active and both
halves decoded, Aspect Fold 500 ID 0, then adaptive mono ID 2 again. Route
traces show mono-right video with left-leg passthrough, stereo with no audio
leg, and restored mono-right/left-passthrough after the final switch. The
receiver restarted with saved `adaptive-mono-nested`; its first live adaptive
stereo packets selected ID 4 and stereo decoding, then adaptive mono ID 2
overrode the preference and restored the mono route. Captured receiver RGB is
under `tmp/v7-gui-live-pair/results-distinct-ids/`; the saved preference was
`adaptive-mono-nested`. PortAudio traces verified 96-kHz/25-channel input from
the temporary sink monitor and 44.1-kHz stereo output on the separate `loop`
sink. The temporary sink was unloaded, and defaults returned to `loop` /
`loop.monitor` at 44.1 kHz.

The owner-requested mono comparison used six matched live runs: portrait/text
and coloured text, each sent as `aspect-mono-500`, the saved-profile alias
`aspect-mono-nested`, and `adaptive-mono-nested`. Each source's three runs used
the same video,
`viewer_solve`, box encoding, 160-px capture, 96-kHz output and no luma-adjust;
receiver AUTO decoded 71/69/17 frames for portrait/text and 70/69/14 for
coloured text, respectively. Aspect mono and its legacy nested alias both
confirm wire ID 3 and use 80x96 luma plus two 40x48 chroma planes. The legacy
alias uses `test_modem_v7/nested_tables.npz` (SHA-256
`f7b94d5c2adc1b023ac675a184fb3cbba6c784ba79812187ceedd0f4e25d43f1`); plain
Aspect Mono 500 has no nested table. Adaptive mono confirms ID 2 and uses
100x120 luma plus two 50x60 chroma planes with `chroma50.npz` (SHA-256
`bc02a63d491fe1c6a9aef2a5f61541048967a57db6d79de9f38c9426ddf1dd3e`).
Sender input-grid values, verified receiver RGB, source RGB, commands and route
traces are under `tmp/v7-mono-version-comparison/`; inline sheets are
`portrait-text-mono-comparison-sheet.png` and
`text-colour-mono-comparison-sheet.png`. Existing 17-tap filter captures were
not mixed into these pairs because they do not share this exact source/settings
matrix.

The real sender GUI command builder and live receiver GUI were exercised over PortAudio/PulseAudio using the repository-picture video, 96 kHz, a temporary 25-channel null sink/monitor, and channel pair 24/25. `pactl` stream inspection showed actual `float32le 25ch 96000Hz` streams; the harness routed sender output and receiver input on the same pair. The temporary sink was unloaded and defaults returned to `loop` / `loop.monitor` at 44.1 kHz stereo.

Both selected GUI profiles constructed adaptive nested commands and sent status ID 2. Sender ready output identifies `adaptive-mono-nested` or `adaptive-stereo-nested` rather than the inherited base wire name. An initial run established dispatch (120 mono and 107 stereo records; traffic counts only). The repaired capture hook now stores the RGB reconstructed from the exact decoder values passed to the viewer, plus verified coded status, metadata, `source_index`, slice kinds and half quality. Captures pair to sender command intervals, whose sender grids are saved alongside the actual ffmpeg-decoded source RGB. The active test route is verified 96-kHz PulseAudio, 25-channel float32, with channel pair 24/25; the temporary sink was unloaded after each run and defaults returned to `loop` / `loop.monitor`.

Current matched live runs cover the portrait/text and colour/text sources. Inspection of the actual sender logs caught that requested `reference` was omitted from the command whenever a profile had a non-reference default; those named `reference` runs actually used `viewer_solve` on Aspect/adaptive mono and `upscale_precomp` on adaptive stereo. Thus this capture set exercises the defaults on those profiles, plus actual `antiring` adaptive mono; it does not compare Reference against the defaults. The builder fix now sends `--dct-kernel reference` when that selection overrides a non-reference profile default. No additional audio run was made after discovering this; this screenshot matrix is not retroactively described as a per-kernel A/B. Static portrait source is a lossless FFV1 RGB round-trip of `tmp/v7-under3-block12-multi/sources/benFaceSource0000.png`. At the verified decode handoff, adaptive mono records a single non-slice image; adaptive stereo records two valid halves (`left+right`) on the sampled frames and reports `stereo-slices`, `metadata_valid=true`, coded/profile ID 2. Captures and sender grids are under `results-matched-audit/current-matched/`.

Visual inspection of actual receiver screenshots on the lossless portrait source shows Aspect Fold 500 substantially more legible than the adaptive captures. Clean-HEAD/current Aspect Fold 500 screenshots are near identical on both lossless portrait and colour/text sources, with the same `viewer_solve` parameters and sender-ready settings; the comparison sheet and screenshot-crop deltas are in `results-matched-audit/controlled-aspect500-head-current.png`. This does not explain the owner's older installation comparison. Adaptive captures show artifacts; the two stereo labels actually applied the same `upscale_precomp` default, so they do not establish a Reference-vs-default comparison. Gaussian is the only kernel purged here; its evidence comes from the prior comprehensive matched benchmark, not the live screenshot matrix. No profile default was retuned.

The owner also reports that Aspect Fold 500 looked better than before. It is included as a distinct control because it is the existing non-serpentine Aspect Fold profile, not the new adaptive mode. Controlled clean-HEAD Aspect Fold 500 runs used an isolated receiver config with a matching sticky profile, 12-second sender windows, the same lossless sources, supported sender commands, 96-kHz route, and identical `viewer_solve` parameters as current. Both sources produced over 130 verified frames per sender interval. Screenshots and grids are in `results-matched-audit/clean-head-controlled/`; the current counterparts are in `current-matched/`. Picture-crop MAE across the intended reference/default comparisons ranges 0.04–0.17 RGB value per channel; both revisions actually applied `viewer_solve`. This establishes visual preservation against pushed HEAD under matched settings, not why the owner's older installation looked different. Do not attribute the reported improvement to adaptive stereo.

The receiver's table selector is initialized from `tools.v7_nested_gui_patch.DEFAULT_TABLE_PATH` (`test_modem_v7/experimental_tables/chroma50.npz`); it is not included in `_persist_preferences` / `_restore_preferences`, so an old saved GUI preference cannot restore `nested_tables.npz`. `_build_arguments` resolves the actual field value and passes it as `adaptive_nested_table`; `ReceiverGui.receive` passes that same path to `install_receiver`, and `_AdaptiveProfileDecoder` passes it to both adaptive mono and adaptive slice wire constructors. The configured table SHA-256 is `bc02a63d491fe1c6a9aef2a5f61541048967a57db6d79de9f38c9426ddf1dd3e`; its grids are luma 100x120 and chroma 50x60, matching captured receiver grid metadata. Stock `nested_fold.TABLES` is intentionally left pointing to `test_modem_v7/nested_tables.npz` for legacy codecs. The log now distinguishes the selected experimental file from that stock default. Folder-picker changes are separate from table/profile selection and are documented in the ledger above.

## Kernel registry and quality audit status

The six sender profiles are in `tools/v7_profile_registry.py`. The chooser now filters the offered kernel list against runtime profile settings: adaptive stereo has `reference` and `upscale_precomp`; adaptive mono has the mono-compatible kernels including `antiring`. The test/GUI preserves an unsupported saved value with an explicit notice so user preference is not silently lost, omits it from choices for that profile, and refuses start with the same profile-specific error the runtime kernel produces. The active defaults remain `viewer_solve` for adaptive mono, `upscale_precomp` for adaptive stereo, and `viewer_solve` for Aspect Fold 500.

Runtime-compatible chooser inventory (compatibility is not a quality result):

| Profile | Offered kernels |
|---|---|
| Aspect Fold 500 stereo | `reference`, `antiring`, `band_taper`, `csf_diamond`, `csf_peak`, `lanczos`, `magic_kernel_sharp`, `mitchell`, `slepian`, `tv_cartoon`, `upscale_precomp`, `viewer_solve` |
| Aspect Fold 500 mono | the stereo list plus `shock_edge` |
| Aspect mono nested | same as Aspect mono |
| Legacy stereo nested | `reference`, `upscale_precomp` |
| Adaptive mono nested | same as Aspect mono |
| Adaptive stereo nested | `reference`, `upscale_precomp` |

Source-matched live captures exercise `viewer_solve` on Aspect Fold 500 and adaptive mono, actual `antiring` on adaptive mono, and `upscale_precomp` on adaptive stereo. Requested Reference runs applied profile defaults and are not counted as a separate candidate. Gaussian's comprehensive sweep evidence and retirement rationale are in `docs/V7_KERNEL_DEFAULTS_TUNING.md`; it was not in the live picture matrix. Other listed kernels have not undergone source-matched decoded-image quality testing in this audit; do not interpret the inventory as a ranked or fully quality-vetted menu.

Other offered kernels, variable/animated sources, induced metadata-corruption behavior and an explicit live Reference-vs-default comparison were not exhausted by this live quality audit. Recovery regression coverage includes partial stereo-leg decode and loss of one live input leg (`modem_tests.test_v7_adaptive_serpentine`, `modem_tests.test_v7_slices`). Only Gaussian was removed on the comprehensive matched sweep evidence; defaults remain unchanged.

## Checks and boundaries

Passed:

```text
.venv/bin/python -c "import sys; sys.argv=['unittest']; import unittest; suite=unittest.defaultTestLoader.loadTestsFromNames([...]); result=unittest.TextTestRunner(verbosity=1).run(suite); sys.exit(not result.wasSuccessful())"  # 313 tests, focused profile/recovery/GUI/lazy-import integration set
.venv/bin/python -m unittest modem_tests.test_v7_kernels -v  # 64 tests
.venv/bin/python -m unittest modem_tests.test_v7_send_gui_kernels -q  # 41 tests
.venv/bin/python -m unittest modem_tests.test_v7_adaptive_profile_dispatch modem_tests.test_v7_receiver_gui modem_tests.test_v7_nested_gui_patch -v
.venv/bin/python -m unittest tests.test_modem_integration tests.test_lazy_imports -v
.venv/bin/python -m unittest modem_tests.test_v7_send_gui_kernels -v
.venv/bin/python -m unittest modem_tests.test_v7_adaptive_serpentine modem_tests.test_v7_slices -v  # 23 tests; includes partial-leg and receiver leg-loss recovery
.venv/bin/python -m unittest modem_tests.test_v7_kernels -v  # includes retired kernel registry and behavior coverage
git diff --check
.venv/bin/python -m py_compile tools/v7_live.py tools/v7_send_gui.py modem_tests/test_v7_adaptive_profile_dispatch.py modem_tests/test_v7_send_gui_kernels.py
```

The receiver-GUI suite exercises settings/CLI argument construction, setup controls, saved preference compatibility, and the graphics/page paths, but it does not replace end-to-end testing of all legacy profiles, all metadata-corruption/recovery paths, all variable geometries, or device-specific routes. No claim is made that this focused change establishes those broad regressions. No filters, default kernels, decoder tuners, or per-frame numerical kernels were changed.
