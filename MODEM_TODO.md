# Modem TODO (standalone-modem)

## Verify (after c903a1d2..0005 patches)
- [ ] `modem_tests`: 258 run; only the 3 known failures
      (test_rgb_fidelity, test_rolloff_engine…, test_unspread_layout…)
- [ ] `tools/compare_codecs.py --modem-dir images_modem`: v5 − v3 positive
      on every channel except clean
- [ ] Live over BlackHole: hd-dwt fills the window, 0 deadline misses,
      receiver auto-detects v3/v5 without --codec
- [ ] Real cassette: v5 vs v3, same deck and tape

## Open bugs
- [ ] One channel stays undecoded after damage ends (heavy compression or
      distortion on L). Not reproducible offline; capture ~20 s of
      clean → damage → clean from BlackHole as stuck.wav and decode with
      `modem_v3_check.py read --wav stuck.wav`.

## Next (cheap)
- [ ] Pitch-shift correction: edge ratio from the preamble (measure_pulses)
      vs the nominal cadence; undo per symbol (resample by 1/r, or sample
      the spectrum at r·k).
- [ ] Pitch → hue control: intentional pitch = edge ratio ÷ cadence ratio
      (tape speed cancels). Deadband 0.3–0.5 semitone, smooth over 3–5
      frames, 1 octave = 360°, rotate Cb/Cr at display time. Both channels
      shifted only.
- [ ] Per-channel robustness: hold a damaged channel's InputLevel gain,
      re-admit after ~3 good packets with a one-step relevel from its
      preamble.
- [ ] Constant frequency offset correction per channel (estimate from
      the training symbols).
- [ ] Separate timing per channel when `skew_samples` is large (azimuth).

## Notes
- v5 wire format changed in patch 4: update sender and receiver together;
  old v5 recordings will not decode.
- Mono is possible, but v3 quality in mono means ~7 fps.