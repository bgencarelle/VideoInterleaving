# V2 recovery update (base: 7a5a79ac)

V2 receive now follows a SEARCH -> BODY -> emit state machine. It uses only
incoming stereo samples and a static format configuration. It does not read a
transmit clock, assume frame spacing, consult Chrony, or schedule display from
recorded timestamps. Every packet carries its own rate reference. Previous rate
estimates only accelerate acquisition; abrupt rate changes trigger a full search.

Like the supplied LTC loop, acquisition is incremental and output is emitted
when the word/frame is complete. OFDM image resampling, FFTs and reconstruction
still cost more CPU than an 80-bit timecode decoder. Image airtime is separate
from decode CPU time; slower tape necessarily takes longer to deliver a frame.

## Changes

- Bounded, decimated coarse acquisition; preamble-only position/rate refinement.
  No five-fold full image decode during rate estimation. Cached layout arrays
  remove repeated carrier/pilot construction from the hot path.
- Early packets are consumed in order, regardless of WAV/callback block size.
  Complete EOF packets need no artificial padding or following packet. `flush()`
  permits only two samples of fractional endpoint rounding, not truncated images.
- Weak coefficients use a Wiener estimate with channel reliability/noise. CRC
  still gates identity, but coherent training can salvage the current picture.
- Packet-wide peak scaling avoids allocation-induced PCM clipping. Training and
  body scale together, so the channel estimator recovers the original amplitudes.
  User `--gain` above unity can still clip; it remains an explicit level control.
- Small presets now have matching source coders, allocation generation and image
  reconstruction. Read streams the WAV instead of materializing the whole file.
- Playback speed reporting now means actual playback speed, not duration ratio.
  Audio input overflow resets receive/filter state. Filtering permits slow signals.

## Try it

Existing image bakes are reusable. Use the same preset, profile and allocation
table on both ends. Old allocation files work only where their shapes match;
regenerate for `tape-fast`, `narrow` or `lofi`. Do not reuse a `color` table for
another source profile. This patch does not change v1 or wire v2 into `main.py`;
the independent v2 tool remains its entry point.

```bash
python -m unittest discover -s modem_tests -v
python -m unittest test_modem_integration test_lazy_imports -v

# Fast first check, no bake or allocation file required:
python utilities/modem_v2_check.py write --preset tape-fast --frames 12 --out v2_check.wav
python utilities/modem_v2_check.py read --preset tape-fast --wav v2_check.wav

# Your own library; build a matching table on the sending machine:
python utilities/modem_v2_check.py allocate --modem-dir images_modem \
  --preset tape-fast --frames 200 --out allocation_tape_fast.npy

# Receiver first (copy allocation_tape_fast.npy to it):
python utilities/modem_v2_check.py live-receive --device "BlackHole 2ch" \
  --preset tape-fast --allocation allocation_tape_fast.npy

# Transmitter; no receiver scheduling or clock coupling:
python utilities/modem_v2_check.py live-send --device "BlackHole 2ch" \
  --modem-dir images_modem --preset tape-fast --allocation allocation_tape_fast.npy
```

The live-send tool prebakes its selected packets and loops them. This remains a
link test, not the main project's clock/folder-selection driver. Use `--quiet`
on live receive to eliminate per-frame terminal output after testing. Saving PNGs
also adds I/O latency. Live output displays the latest complete recovered image;
it holds the last image through total signal loss, without fabricating a new one.

## Tape presets

Default color shapes (width x height), at nominal playback speed:

| Preset | Image carrier ceiling | Luma size | Nominal fps |
|---|---:|---:|---:|
| tape | 10.125 kHz | 40 x 48 | 7.71 |
| tape-fast | 10.125 kHz | 26 x 32 | 14.35 |
| lofi | 3.75 kHz | 14 x 18 | 6.91 |

For very limited-bandwidth tape, use `--preset lofi` on allocate/write/read or
both live commands. The preamble remains the existing wideband waveform; the
lofi receiver matches its surviving low-band portion. Narrowing image carriers
does not restore information lost by the tape or by ADC aliasing.

Default acquisition covers 0.5x..2x playback. `--min-speed .25` enables quarter
speed, with more expensive cold acquisition. The speed window is a search bound,
not an expected FPS. Both read and live-receive accept these bounds. At 0.5x,
nominal frame durations double; this is airtime, not a decoder wait policy.

## Evidence and limitations

Regression tests cover cold start at .5, .67, .8, .95, 1.05, 1.2, 1.5 and 2x;
optional quarter speed; abrupt rate changes with arbitrary gaps; 0.3% sinusoidal
wow/flutter at 4/20/60 Hz; half-speed bandlimit/hiss/polarity; header erasure;
noise rejection; PCM peaks; EOF; one-sample through whole-file chunk sizes; and
WAV write/read including small presets.

On this development machine, nominal clean receive was approximately 1–2 ms per
frame; non-nominal acquisition/resampling costs more. `decode_ms` excludes search;
`acquire_ms` includes search since the previous recovery, and `receive_cpu_ms`
is their sum. They exclude sound-device buffering, filter/UI/terminal/PNG work
and screen refresh. A long no-signal interval can increase accumulated search CPU.

The synthetic three-frame tape benchmark measured about 42.4 dB clean for both
versions, 25.6 dB v2 versus 17.5 dB v1 on its 10 kHz/hiss model. Lofi also
recovered a coarse image and CRC identity on a separate 3 kHz/0.003 RMS hiss
simulation. These are software measurements, not tape-deck guarantees or Pi
timings. Real cassette equalization, saturation, azimuth, dropouts and complex
flutter must still be tested. Speed recovery cannot reconstruct aliased carriers.

V2's existing header does not identify an allocation table or every source-shape
choice: endpoints must agree on static configuration. A good CRC authenticates
neither the sender nor a matching external allocation file; it detects bit errors.
