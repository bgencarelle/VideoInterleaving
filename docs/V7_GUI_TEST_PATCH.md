# Matched nested-codec GUI test patch

> Historical instructions and GUI/Pulse results below predate distinct
> adaptive identities. Current wire IDs are ID 2 for adaptive mono and ID 4
> for adaptive stereo; the former ID-4 Mono-500 meaning is retired. See
> `docs/transport_v7_spec.md` §14.1 and `docs/V7_GUI_PROFILE_AUDIT.md` for the
> current registry and device-free transition checks.

The patch targets clean pushed baseline
`04d1fb2e7365c4f385b34218675b12af06ed4838` and includes the GUI changes plus
the runtime dependencies and focused tests needed by this table. No dirty
worktree files are prerequisites.

The committed checkout includes the runtime support and validated table.
Open `./vi.modem-send-gui` and `./vi.modem-receive-gui` normally; no archive
installation or table build is required. For the earlier standalone overlay
bundle, save any local versions of files being overlaid, then install with:

```sh
unzip tmp/v7-gui-test-patch.zip -d .
cp -a tmp/v7-gui-test-patch/overlay/. .
```

The validated `100×120` luma / `50×60` chroma table is included at
`test_modem_v7/experimental_tables/chroma50.npz` in the committed checkout.
The earlier overlay archive uses `tmp/v7-gui-test-patch/tables/chroma50.npz`.
No corpus build or retraining is
needed.

1. In the sender GUI choose **Adaptive stereo nested** or **Adaptive mono
   nested**. The bundled table path is prefilled.
2. The receiver detects adaptive mono ID 2 or adaptive stereo ID 4 from the
   current validated wire status and uses the same bundled table; the saved
   receiver profile is only a startup hint.
3. The adaptive profiles use serpentine mapping, guest factors 1/2/4/8/16,
   fixed host level −1, step factor 0.25 and exact inverse. Stereo additionally
   enables chroma detail. Sender channel count and sample rate continue to
   follow the selected device and OS configuration.
4. **Legacy nested** mono/stereo choices retain the previous mapping and
   profile identities. Saved settings migrate to adaptive profiles only when
   the old matched-codec opt-in was enabled.

## Live GUI/Pulse smoke (2026-10-11)

The final live smoke reused `tmp/v7-gui-live-pair/run.py` with its existing
sender → PortAudio → PulseAudio → PortAudio receiver → viewer route. Four RGB
sources (`text-colour`, `repository-picture`, `tight-line`, `portrait-text`)
were sent with both adaptive profiles. For each source/profile run, the sender
sequence was legacy → adaptive → legacy → adaptive, while the receiver GUI was
configured only with the matched table path and detected the wire profile from
the stream. All eight runs produced saved received RGB frames (1,238 total),
and the receiver logged both legacy and adaptive profile transitions. The
stereo runs also reported `adaptive-stereo-nested on right`; at that time both
adaptive profiles shared ID 2 and metadata selected mono versus stereo-slices.
That implementation is superseded by the distinct IDs above.

Both directions were verified on the temporary `v7_gui_test_96` PulseAudio
null sink at 96 kHz, float32, 25 channels, selecting channels 24/25. The
receiver route trace records the monitor input and the sender trace records
the matching output. The GUI process logged the adaptive signature/table
configuration under the then-shared ID-2 design. Each run saved a viewer
screenshot and received RGB PNGs under
`tmp/v7-gui-live-pair/results-adaptive-id2-auto/`; `smoke-summary.json` records
source, profile, detected hints/transitions, table, routes, frame counts and
artifact paths. For direct picture inspection, each run also has
`received-rgb.png`, an unmodified copy of its final received frame, and the
complete sequence under `received/`. A desktop `viewer.png` captured after the
receiver stops can show a blank window, so `received-rgb.png` is the direct
decoded-picture evidence. After the harness unloaded its temporary sink, the
defaults were `loop` and `loop.monitor` and only the existing `loop` null sink
remained.

This smoke is a live software loopback verification of the then-shared ID-2
design, not a test of current ID-2 mono / ID-4 stereo transitions or a
physical-medium test. Current receiver dispatch uses the validated wire ID;
adaptive signature verification remains required for adaptive values.

Run focused, device-free checks from the repository root:

```sh
.venv/bin/python -m unittest modem_tests.test_v7_nested_gui_patch \
  modem_tests.test_v7_adaptive_serpentine modem_tests.test_v7_serpentine -v
.venv/bin/python -m py_compile tools/v7_send_gui.py tools/v7_receiver_gui.py \
  tools/v7_nested_gui_patch.py tools/v7_nested_experiment.py
```

Table SHA-256: `bc02a63d491fe1c6a9aef2a5f61541048967a57db6d79de9f38c9426ddf1dd3e`.
The archive includes the original table validation manifest, provenance, and
the separate live GUI trial summary. The receiver GUI's **Colour detail**
display control is a separate reconstruction option; keep it matched during
comparisons.

The existing `tools/v7_nested_build.py` workflow can still be used to create
other research tables, but those are separate candidates and do not replace
the bundled validated table.
