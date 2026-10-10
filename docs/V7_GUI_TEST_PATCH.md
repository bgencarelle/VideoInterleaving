# Matched nested-codec GUI test patch

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

1. In the sender GUI choose **Stereo nested** or **Aspect mono nested** and
   enable **Matched nested test patch · opt-in**. The bundled table path is
   prefilled.
2. In the receiver GUI enable the same opt-in, select the same nested profile,
   and use the same prefilled table path.
3. Start both endpoints. The test patch selects serpentine mapping, adaptive
   guest factors 1/2/4/8/16, fixed host level −1, step factor 0.25 and exact
   inverse. Stereo nested additionally enables chroma detail; aspect mono
   nested omits it. Enabling the patch turns off sender **Luma adjustment**
   to match the validated chroma50 source configuration. The receiver reports
   these settings in its information log.
4. Disable the opt-in at both ends for the ordinary baseline. Keep profile,
   source and display settings matched.

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
