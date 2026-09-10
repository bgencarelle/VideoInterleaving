# Fixed-format v2 operation

## Automatic EQ recovery

The 16-sample cyclic prefix lasts only 0.333 ms. Low-frequency EQ can have a
longer impulse response, contaminating adjacent symbols and the two short
training symbols. Static EQ therefore is not always corrected by the original
per-carrier equalizer. In synthetic tests, bass EQ caused large image errors
even while playback-speed estimates remained close to correct.

When normal recovery has a bad header or pilot error above 0.15, the receiver
now also fits a 128-tap stereo FIR response using only the known preamble and
training prefix. It tries regularized frequency-domain inversion of that same
packet and keeps it only when the existing recovery-quality comparison improves.
The training matrix is cached. This costs additional computation on poor input,
but does not wait for another frame or change the encoder, image format or WAV.
The chosen path appears as input_path=eq_corrected in JSON output.

Short synthetic tests used consecutive frames through Q=1 bell EQ of +/-6 dB
at 375 and 1000 Hz, with output scaled by 0.25 to avoid clipping. Image-value
RMS errors dropped from approximately 0.26-1.15 to 0.04-0.08 (values nominally
span -1 to 1). This is not a guarantee for Audacity effects, arbitrary EQ,
hard spectral nulls, long ringing, time-varying filters or nonlinear distortion.
Deep loss still cannot be inverted safely. Clean gray/slow-playback tests remain
on the raw path. Hardware testing and processing-time measurements remain yours.

## Automatic receiver update

Live receive and WAV read now use the same automatic receiver. No input-filter
or speed-range switches are required or exposed in the normal CLI. Acquisition
searches 0.25x through 2x by default; this is a supported range, not a promise
to recover arbitrary speed or lost/aliased information.

Raw input is preserved. Acquisition tries full-band timing first and falls back
to low-band fitting for roll-off. If search fails, a zero-phase high-pass search
window rejects low-frequency interference. At packet completion, verified
low-error packets take the raw fast path. Otherwise the decoder compares raw
and conditioned versions of that same packet, preferring usable recovery,
then valid CRC identity, then lower pilot error. There is no added frame wait.
Logs include acquisition_path and input_path so the automatic choice is visible.

Bounded synthetic checks preserved neutral gray at 1x, 0.8x, 0.5x and 0.25x
with no user configuration. Added 50 Hz / 0.3 amplitude hum triggered automatic
conditioning. At half/quarter speed that hum example recovered picture_only;
severe-channel color and timing accuracy remain imperfect. Pilot error and CRC
are evidence for choosing a path, not proof of exact reconstructed pixels.
No encoder or wire change is included in this receiver update; V3 WAVs remain
usable. Tests against physical tape are still required.

## Progressive low-band update (wire V3)

The normal wide layout now occupies 375 Hz through 20.25 kHz. Bins at
375 and 750 Hz carry image data. Coefficients from all three image planes
are ordered by spatial frequency, with all DC terms first. Coarse information
occupies the lowest audio carriers over the packet; high carriers add detail.
The earlier always-on high-pass has been replaced by automatic conditioning.
Two pilots at 1125 and 1875 Hz support timing correction after roll-off.

Install transport2.py and modem_v2_check.py together on both endpoints and
regenerate WAVs. Normal packets now identify as V3; prior V2 recordings need
the previous decoder. The picture-only fallback does not prove wire identity.
The fixed resolution and packet duration are unchanged. This adds two carriers
without adding image samples or increasing frame airtime.

A bounded synthetic check recovered an image through a sixth-order 6 kHz
low-pass plus 0.0003 RMS hiss. At 3 kHz it acquired but reconstructed poorly;
at 1.5 kHz it did not acquire. Severe roll-off is still an open receiver issue:
the preamble remains wideband and channel delay biases timing estimation.
These checks are not physical tape measurements.

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
