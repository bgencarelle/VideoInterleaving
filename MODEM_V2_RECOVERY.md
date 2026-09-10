# Fixed-format v2 operation

Normal encoding uses one clean signal: 48 kHz stereo, the wide layout,
color 40x48 luma and 20x24 chroma, 2880 image values and deterministic
built-in allocation. No allocation file, preset or profile is needed on
either endpoint. Source bakes are converted to this fixed color format.
Source selection defaults to consecutive frames (stride 1).

```bash
python utilities/modem_v2_check.py write --modem-dir images_modem --frames 200 --out clean.wav
python utilities/modem_v2_check.py read --wav clean.wav

# Receiver first, then sender in another terminal:
python utilities/modem_v2_check.py live-receive --device "BlackHole 2ch" --channels 1,2
python utilities/modem_v2_check.py live-send --device "BlackHole 2ch" --channels 1,2 --modem-dir images_modem --frames 200
```

Omit --modem-dir for synthetic images. Read reports results; use --save-frames
to export decoded images. Live receive displays the latest recovered image.
Apply distortion to a copy of clean.wav afterward. Normal commands reject
--preset, --profile, --allocation and --gain; experiments remain under bench.
Old recordings made with alternate layouts or custom allocation tables need
the earlier explicitly configured tool. They are not automatically compatible.

Receiver acquisition remains independent of transmit scheduling. It searches
incoming samples, estimates playback rate and channel response, and emits
when a packet is complete. A corrupted header can still yield picture_only
when training supports recovery; identity still requires a valid CRC.
The receiver holds the last displayed image during complete signal loss.

This change simplifies operation; it does not establish new tape performance.
The fixed wide signal can lose high-frequency information on limited-bandwidth
tape. Prior lofi/tape-fast results used different transmit formats and do not
prove equivalent recovery of this signal. Test the fixed signal through your
actual channel. Decode CPU time also excludes airtime and device/UI buffering.

The standalone live sender loops selected composites from folder pair 0,0.
It is not yet integrated with main.py's Chrony/locksync scheduler or normal
folder selection. Its timestamps do not schedule receiver presentation.
