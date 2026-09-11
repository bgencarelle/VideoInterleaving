This patch is based on modem commit c9c21a0a. It fixes one source of image
corruption during smooth within-frame timing drift. It does not make arbitrary
digital pitch shifting transparent to the modem.

Pilot phase now follows successive symbols, instead of independently unwrapping
widely spaced frequency pilots on every symbol. Missing pilots retain the
independent-symbol fallback. Live cold acquisition uses the existing sinc LUT,
as locked acquisition already did. There is no additional image decode pass,
no encoder-format change, and no warning suppression.

Apply from the repository directory:

```bash
git apply --check /path/to/modem-pilot-continuity.patch
git apply /path/to/modem-pilot-continuity.patch
python -W error::RuntimeWarning -m unittest discover -s modem_tests -v
```

To regenerate the experiment, supply a folder containing the source images.
FFmpeg with the rubberband audio filter is required for `make` only. Each image
is encoded into one normal fixed-format packet. All effects are applied to
rendered audio, not to the runtime encoder.

```bash
python utilities/modem_pitch_probe.py make --folder /path/to/images --frames 20 --cents 5 --out pitch_probe
python utilities/modem_pitch_probe.py check --wav pitch_probe/clean.wav
python utilities/modem_pitch_probe.py check --wav pitch_probe/processor_zero.wav
python utilities/modem_pitch_probe.py check --wav pitch_probe/pitch_up.wav
python utilities/modem_pitch_probe.py check --wav pitch_probe/pitch_down.wav
python utilities/modem_pitch_probe.py check --wav pitch_probe/speed_up.wav
python utilities/modem_pitch_probe.py check --wav pitch_probe/warble.wav
```

Use `--pattern 'benFaceSource00*.webp'` if the folder contains other sequences.
Keep values.npy with these WAVs: it supplies the known image values for error
measurement. Pitch processing uses linked stereo and unchanged tempo. The
zero-cent processor control deliberately goes through the same effect.
`speed_up` changes duration and pitch together by the equivalent five cents.
`warble` models 0.3% peak speed variation at 4 Hz.

Results from the 20 supplied face images, on this development machine:

| Case | Verified before / after | Value RMSE before / after | Median receiver ms before / after |
| --- | --- | --- | --- |
| Clean | 20 / 20 | 0.000119 / 0.000119 | 62.80 / 63.86 |
| +5 cents, unchanged tempo | 12 / 12 | 0.335240 / 0.332626 | 97.00 / 95.13 |
| -5 cents, unchanged tempo | 13 / 13 | 0.334984 / 0.333326 | 153.66 / 144.55 |
| Warble | 20 / 20 | 0.188772 / 0.038763 | 84.51 / 81.12 |

Timing is the median of three runs of each 1.393-second WAV; it excludes audio
capture, GUI rendering and file loading. RMSE is measured in the source-value
range [-1,1], for frames with verified matching identities. It does not score
unidentified frames; read verified counts alongside it. A verified header and
full carrier coverage do not guarantee accurate image values.

The initial controls also recovered all 20 frames from the equivalent speed
change (RMSE about 0.00249). The zero-cent effect recovered all 20 headers but
changed the image values (RMSE about 0.141). These findings apply to this FFmpeg
Rubber Band experiment; they do not establish the behavior of every processor.
Supplying the known unchanged symbol duration in a diagnostic did not repair
the +5-cent copy. The recorded two-second processing stalls have not been
reproduced by this short experiment, and this patch is not a hard CPU budget.

For a physical input, use the usual command and the known working device:

```bash
python utilities/modem_v2_check.py live-receive --device 1 --channels 1,2
```
