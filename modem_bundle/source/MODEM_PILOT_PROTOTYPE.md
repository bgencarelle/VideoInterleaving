Independent delivery-speed / pitch prototype

This is a reference-only experiment, not a replacement for image transport.
It occupies both audio channels: L carries amplitude timing marks on a carrier;
R carries continuous 1300 Hz and 3300 Hz pilots. No image frames or image header
are present. Do not feed this signal to the production modem receiver.

The timing reference repeats 15 times per nominal second. Envelope edge spacing
measures delivery speed independently of pilot frequency. The two observed
pilot frequencies estimate a proportional frequency scale and a common additive
Hz offset. The proportional scale divided by delivery speed gives extra pitch.
Both measurements are averaged over the same marker interval to reduce confusion
between ordinary flutter and independent pitch change.

The receiver works incrementally in 256-sample hops. A 1024-sample FFT finds the
pilots initially; while locked, two projections against cached oscillator tables
and phase differences track them. It reacquires when pilot amplitude disappears
or the frequency estimate leaves the tracking window. Frequency history and
pending marker intervals have fixed bounds. No image decode retries are used.

Install the existing project dependencies. Write a four-second clean reference:

```bash
python utilities/modem_pilot_prototype.py write --out pilot_clean.wav
python utilities/modem_pilot_prototype.py read --wav pilot_clean.wav
```

The expected `delivery_speed` and `frequency_scale` are approximately 1, with
`extra_pitch_cents` and `frequency_offset_hz` approximately 0. Measurements have
finite precision; do not expect exact zeros.

Create ideal independent pitch/timing controls:

```bash
python utilities/modem_pilot_prototype.py write --out pilot_half_plus5.wav --speed 0.5 --pitch-cents 5
python utilities/modem_pilot_prototype.py read --wav pilot_half_plus5.wav

python utilities/modem_pilot_prototype.py write --out pilot_quarter.wav --speed 0.25
python utilities/modem_pilot_prototype.py read --wav pilot_quarter.wav

python utilities/modem_pilot_prototype.py write --out pilot_wow.wav --wow-percent 0.3 --wow-hz 4
python utilities/modem_pilot_prototype.py read --wav pilot_wow.wav --verbose
```

Half speed plus five cents should report about 0.5 delivery speed, 0.501446
frequency scale, +5 cents extra pitch, and zero additive Hz offset. Quarter
speed should report about 0.25 speed and frequency scale with zero extra pitch.
Wow varies instantaneous speed and frequency together. Inspect time-series output
for false extra-pitch readings; this is not yet calibrated against real tape.
`--offset-hz 5` is an optional synthetic common frequency-offset control.

The generator models ideal continuous-phase changes, NOT a particular digital
pitch-shifting effect. Apply effects to the rendered clean WAV separately to
measure a real processor. Preserve stereo and export 16-bit PCM for `read`.

For tape or loopback, play/record `pilot_clean.wav` using your existing audio
routing, and run this separate receiver on the returned stereo pair:

```bash
python utilities/modem_pilot_prototype.py live-receive --device 1 --channels 1,2
```

Replace the device number and channels with your working input. The receiver
opens the device's reported default sample rate once. Output is limited to one
record per second after references are detected. Stop with Ctrl-C. There is no
channel scan. A recording of the same reference can be read using `read`.

Initial acquisition needs at least two usable timing edges and a pilot history
covering their interval. At slow speed this can take several hundred milliseconds.
This is measurement lock time, not the processing time per block. A report waits
for pilot estimates covering the marker interval (about 13 ms extra at 48 kHz).

Only one short synthetic sanity check was run: two seconds at 0.5x with +5 cents
produced 13 reports, median delivery 0.499922x and extra pitch +5.2704 cents.
Generation plus analysis took about 46 ms on this development machine. No broad
speed/EQ/dropout sweep, live-device test or tape test was performed.

Before integration: measure pitch bias under flutter/EQ, test mono/stereo routing
and dropouts, and determine a reference allocation that can coexist with images.
The present amplitude thresholds, 1024-sample tracking window and coarse-search
limits are prototype choices. They are not an established robustness guarantee.
The prototype measures a mismatch; it does not yet correct image carrier extraction
or undo phase-vocoder damage.
