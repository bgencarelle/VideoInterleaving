# First-pass reference-event decoder

Patch base: `e095fcd83fe937ac14de56fb3578334db124982a` (`modem`).

Select the new decoder by adding `--receiver reference` to your existing
`modem_receive.py` command, retaining your working device and channel arguments.
The existing progressive decoder remains the default for comparison.

For a recording:

```bash
python modem_receive.py --receiver reference --wav recording.wav --fast --headless
```

The live inspection utility also accepts the selector:

```bash
python utilities/modem_v2_check.py live-receive --receiver reference --device 1 --channels 1,2
```

Use your own working device/channel arguments. No device scanning, sample-rate
probing, encoding changes, or shared-time scheduling changes are introduced.
Live capture still opens the selected device's reported rate through the existing
capture path.

## What it does

1. Consume stereo energy observations once with a fixed eight-sample envelope.
   Detect the L-only preamble followed by the R-only second training symbol.
   Their nominal onset separation is `SYNC_LEN + SYMBOL - 16` = 416 samples.
   Packet-to-packet spacing is never used to infer playback speed.
2. Reject leading low-level ringing using a relative onset threshold. Use the
   two events for a coarse duration estimate, then make **one** linear reference
   measurement of gain, delay and dilation against the known preamble. Its
   derivative and regression projection are cached. There is no iterative fit,
   correlation scan, alternate scale, candidate replay or old-search fallback.
   A poor reference residual rejects this event pair.
3. Use the measured affine clock and the existing sinc LUT to read only new
   FFT windows. Reuse the progressive receiver's once-per-packet stereo training
   calibration, per-symbol pilot phase continuity and fresh coefficient buffer.
4. Publish the existing early preview and paced refinements. Missing image slots
   stay zero, never copied from an older packet. Rejected input emits no blank
   image; the existing display retains its last usable picture. Each snapshot
   still uses an inverse DCT of the image planes; reconstruction is paced, not
   an incremental inverse-DCT implementation.

`input_path` is `reference_events`. `reference_fits` and `reference_rejections`
are cumulative event counters for this receiver object. `fft_symbols` counts
training, header and image symbols within the packet. Existing CPU timing fields
are per emitted event, not a complete per-frame total when previews are emitted.
No target-hardware throughput claim is made.

## Limits that remain

This is an experimental decoder, not a completed general-audio robustness fix.
The event detector needs distinguishable channel-exclusive reference sections.
Strong crosstalk, mono collapse, roll-off, noise and smeared transitions can hide
those events or bias their timing. The one-step preamble correction also assumes
that the received reference is reasonably close to a delayed/dilated, scaled
copy. Independent pitch processing can violate that assumption.

The independent pilot prototype's inaccurate timing estimator is not used.
Its cached projection idea is reused for a fixed reference regression; its
continuous-tone frequency estimator is not transplanted into symbol-dependent
production pilots. Timing and independent pitch are **not yet jointly estimated**.
The existing pilot phase correction does not drive the sampling clock. Clock
speed stays constant inside each packet; nonlinear flutter remains unresolved.

The detector's provisional scale range is 0.5 to 4 (roughly 2x to quarter-speed).
This is a detection bound, not a tested robustness guarantee or a way to recover
carriers already lost to aliasing. No search fallback is hidden behind failures.

## Focused checks

```bash
python -W error::RuntimeWarning -m unittest modem_tests.test_reference_receiver modem_tests.test_progressive_receive -v
```

Twelve focused tests passed during development: forbidden search/bank/replay
calls, one-pass symbol counts, clean fidelity, half/quarter-speed references,
feed boundaries, gaps, changed gain, fresh frames, erased headers, noise and reset,
plus the existing progressive checks. Short 0.8x inspection also recovered a
verified frame. No full suite, hardware benchmark, broad impairment sweep,
independent-pitch validation or tape experiment was run.
