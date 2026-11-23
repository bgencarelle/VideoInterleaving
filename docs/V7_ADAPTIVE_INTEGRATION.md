# Adaptive nested profiles

How the two adaptive profiles work today. The maths of the serpentine slot
mapping is in [V7_SERPENTINE_MATH.md](V7_SERPENTINE_MATH.md).

## What they are

- `adaptive-mono-nested` (wire profile ID 2): picture on one channel, sound on
  the other.
- `adaptive-stereo-nested` (wire profile ID 4): picture on both channels.

They send the picture on a larger grid than the other profiles: 120×100 for
brightness and 60×50 for colour, with the coefficients placed by the table
`test_modem_v7/experimental_tables/chroma50.npz` (SHA-256 starting `2d4d7f09`).
Both ends must have the same table. The receiver recognises both profiles on
its own; there is nothing to set.

## How they are sent

- **Fold.** They run on the stock nested fold with the serpentine slot
  mapping. The level of each picture is chosen per picture, by trying the
  levels and keeping the one that decodes best (the stock nested rule). An
  older fixed-level path (host level −1, guest factors 1 to 16) is still in
  the code behind `adaptive_guest_range`; the sender does not use it.
- **Detail guests (stereo).** On folded slots the first channel's guest
  carries the detail itself, and the receiver blends it with the two-channel
  reading. `NESTED_FOLD_DETAIL_GUESTS=0` at both ends turns this off.
- **Table shape.** The table's brightness guests fill outward from the hosts
  by radius (`tools/v7_reshape_guests.py`), which removed the cross through a
  Siemens star and the streaks in a flat sky. The colour guests are not
  reshaped yet; the faint streaks left in a sky are colour.
- **From the source picture.** With the Reference kernel (no kernel), the
  sender averages the source's pixels straight onto the exact 120×100 and
  60×50 grids. Any other kernel keeps its own path. Pixel encode skips this.
- **Sound.** The picture part of each packet is shaped with a 17-tap filter
  and set 0.8 dB under the end marker (the other profiles use 1.5 dB):
  `BODY_FILTER_TAPS` and `BODY_LEVEL_BELOW_EOF_DB` in
  `tools/v7_adaptive_pipeline.py`.

## Settings

Choosing either adaptive profile in the sender window applies one fixed
preset, `sample_sender_settings()` in `tools/v7_adaptive_pipeline.py`:
Reference kernel, neutral brightness and gamma, 768-pixel capture, no luma
adjustment, no sharpening. The shared sender defaults
(`tools/v7_send_defaults.py`) also give both profiles the Reference kernel, so
the command line and `main.py --mode modem --modem-profile ...` match the
window. You can still pick another kernel by hand.

## Sample feed

A launcher runs the preset through the real sender and receiver. Start the
receiver first:

```bash
.venv/bin/python tools/v7_sample_feed.py receive --device INPUT_DEVICE
.venv/bin/python tools/v7_sample_feed.py send --mode mono --device OUTPUT_DEVICE --video-source picture.mkv
.venv/bin/python tools/v7_sample_feed.py send --mode stereo --device OUTPUT_DEVICE --video-source picture.mkv
```

Run one sender at a time. `--print-command` shows the command without opening
any audio. Other sender options can follow, for example `--rate 96000`.

## Optional: legal-range reconstruction

`--luma-range-passes 16 --range-host-cells` on the experimental launcher
(`tools/v7_nested_experiment.py`) keeps the decoded brightness inside the legal
range. It helped lettering but darkened some texture, so it is off by default.
