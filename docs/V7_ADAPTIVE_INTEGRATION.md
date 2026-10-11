# Adaptive experimental pipeline integration

Exact implemented equations and legacy differences: [V7_SERPENTINE_MATH.md](V7_SERPENTINE_MATH.md).

## Working sample feed

The committed launcher runs the sample configuration through the real public
sender and receiver. Start the receiver first, using your existing routed audio
devices (substitute their names for the placeholders below):

```bash
.venv/bin/python tools/v7_sample_feed.py receive --device INPUT_DEVICE
.venv/bin/python tools/v7_sample_feed.py send --mode mono --device OUTPUT_DEVICE --video-source picture.mkv
.venv/bin/python tools/v7_sample_feed.py send --mode stereo --device OUTPUT_DEVICE --video-source picture.mkv
```

The receiver opens the live viewer and detects both new modes automatically.
Only run one sender at a time; Ctrl-C stops it. The launcher does not configure
audio routing or override the OS sampling rate. Existing public CLI options
can follow the launcher arguments, including a deliberate `--rate 96000` for
the authorised software test route. `--print-command` prints the exact command
without opening a stream.

The sample uses Reference, neutral brightness/gamma, 768-pixel capture, no luma
adjustment, and no sharpening/clarity. `sample_sender_settings()` in
`tools/v7_adaptive_pipeline.py` is also directly consumable by the sender GUI's
`build_command()`. It uses the existing adaptive mono/stereo modes rather than
introducing another wire identity. Legal-range reconstruction remains the
explicit option described below; it is not hidden inside this preset.

Selecting either adaptive profile in the GUI applies that same sample preset.
Saved adaptive preferences predating sample revision 1 migrate once to it;
subsequent explicitly saved adjustments are retained. Reference is now the
adaptive kernel default in both CLI and GUI. Legacy profile defaults are
unchanged. A direct CLI invocation can still explicitly select another kernel
or capture width; use the sample launcher for the complete matched preset.

GUI-selection verification: selecting each adaptive profile through
`SenderGui._assign` and launching its generated command produced live received
RGB at 96 kHz on channels 24/25 (2 mono frames, 3 paired stereo frames in the
short coloured-text run). The first sender-grid NPZ was byte-identical to the
sample-preset run for both modes, SHA-256
`3eafaa3735050539f6cb646dd3234fd4e1716bb5b5ccd9a4766c99d7de771085`.
This verifies source-value/configuration parity, not bit-identical audio-route
decodes or elimination of the remaining mono artifacts. 199 focused tests
passed, including profile selection and one-time preference migration.

The public adaptive mono (ID 2) and stereo (ID 4) paths share the table-scoped
serpentine codec: exact inverse, host level -1, stair factor 0.25, guest range
factors 1/2/4/8/16, and bundled chroma50 geometry. Stereo enables the additional
chroma-detail allocation and paired reconstruction. Legacy profiles retain
their existing codec and body filter.

Direct CLI sends now install the same codec/table as GUI-wrapped sends.
Adaptive emission uses the experimental 17-tap, 14-kHz FIR and 0.8-dB body/EOF
target. Final transmitted peaks must be measured after resampling and mixing;
the target is not the measured header/body separation.

Reference source conversion uses compiled area integration onto the exact
100x120 Y and 50x60 Cb/Cr grids. Enhanced kernels retain their own full source
conversion, prefilter, coefficient gains and post-processing; area conversion
must not overwrite them. Selecting an enhanced kernel therefore deliberately
differs from the area/Reference experiment.

Legal-range reconstruction remains available through the public experimental
launcher with `--luma-range-passes 16 --range-host-cells`. It is not enabled by
default: lettering improved but texture and isolated darkening had tradeoffs.
Filter bypass, white filling, source-informed concealment and learned priors
are not included.

This is code integration, not completed fidelity acceptance. The previously
captured portrait candidate is not a valid before/after comparison: the
same-source pushed baseline receiver exited without a received frame. Both
mono and stereo still require matched live RGB acceptance on the exact final
code and settings. Unit tests do not substitute for that evidence.

## Direct integration check

197 focused adaptive/profile/GUI/mono-video/audio tests passed. The compiled
17-tap filter measured 0.329 ms/packet versus 0.385 ms for the existing FIR
implementation (200 warmed calls, 3920x2 float32, this workspace).

Live checks on the temporary Pulse 96-kHz, 25-channel route (24/25) produced
14 mono viewer_solve frames and 32 stereo Reference frames, including 31 paired
stereo frames. A separate mono Reference run also produced received RGB.
Visual inspection found a substantially cleaner stereo portrait but severe
crosshatching in mono viewer_solve and residual grid artifacts in mono
Reference. These are functional integration checks, not a matched baseline
quality pass. Artifacts: tmp/v7-self-integration/ and tmp/v7-self-reference.log.

The final sample preset (Reference, neutral tone, 768 capture) subsequently
produced 29 live mono frames and 26 live stereo frames, all 26 stereo frames
paired. Received RGB still shows mono grid artifacts. The sample feed works;
the integration is not presented as a completed mono fidelity correction.
198 focused tests pass, including public sample command parsing. Sample feed
artifacts: tmp/v7-self-integration/integrated-sample/.
