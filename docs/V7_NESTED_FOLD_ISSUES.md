# V7 nested fold: open issues and filter handoff

Scope: the two new sender options only, `aspect-mono-nested` and
`stereo-nested`. Written after first viewing reported four things: reverse
loses colour and resolution, more grain, banding, and desaturated colour.
Background and headline results are in `V7_SK_VERDICT.md`.

All numbers here are synthetic-channel measurements on Kodak stills
(`tmp/sk/corpus`) through the live sender wires and live receiver code. None
of this is tape validation.

## Status at a glance

| # | Issue | Status |
|---|---|---|
| 1 | Reverse loses resolution (stereo) | **Fixed** |
| 2 | Reverse / forward colour loss | Not reproduced against the base wire; two real causes found (3, 7) |
| 3 | Band-limited nested packets pass the pilot gate and are shown damaged | **Partly fixed**; gate policy still open |
| 4 | Grain | **Reduced**: decoding inside the bounds (every packet) and dither (held pictures). Parts A and B still open |
| 5 | Banding | **Fixed** by dither (no dead zone, no fixed contours) |
| 6 | Stereo nested behind stock slices under a 4 kHz low-pass | Open |
| 7 | Mono and slices wires carry less chroma than `aspect-fold-500` | Open, by design of the base wires |
| 8 | Cold start when the signature is unreadable | Open, minor |
| 9 | GUI kernel defaults ignored for nested profiles | **Fixed** |
| 10 | Body sent 1.5 dB lower since `b147922` (all profiles) | By design: the auto-leveller keeps header and end marker loudest |
| 12 | Through the real display the folds do not beat stock | **Open, the main finding** |
| 13 | False colour in every mode | Diagnosed: too little chroma is sent |
| 11 | Sender GUI could start the sender on the wrong output device | **Fixed** |

## 1. Reverse resolution (fixed)

Reverse needs nothing special from the fold. `decode_reverse_packet` already
reverses the samples in memory, compensates header and EOF, and hands the
result to the forward decoder; the nested decoder sits behind that and never
knows the direction.

The fault was in the live receive loop (`tools/v7_live.py`): for
`stereo-slices` packets the second channel was only decoded when
`packet_direction >= 0`. In reverse the picture was rebuilt from one channel,
so it lost the second channel's guests and the stereo averaging. The same
shortcut was copied into `tools/v7_nested_eval.py`, which is why the earlier
reverse numbers looked acceptable: they were all one-channel numbers.

Now the loop takes the nearest reverse hit in the other channel's probe and
decodes it with `decode_reverse_packet` too.

| Stereo nested, live receive loop | Luma error |
|---|---|
| Forward | 0.0592 |
| Reverse, before | 0.0763 |
| Reverse, after | 0.0592 |

Holds at 0.5× and 1.5×. Stock `stereo-slices` gains the same fix. Covered by
`LiveReceiverReverseTests` in `modem_tests/test_v7_nested_fold.py`, which runs
the real receive loop on a reversed waveform and requires it to equal forward
and beat one channel.

## 2. Colour: what was and was not found

Chroma was compared nested against stock **on the same base wire** in five
places: the evaluation rig, the real sender CLI with kernels and luma adjust,
the live receive loop, under hiss, and through the display path. In every
case nested chroma equals the base wire's chroma (saturation ratio about
0.99). The fold does not touch chroma slots, and reverse does not either.

So the fold itself is not desaturating. Two things can still make these
modes look less saturated than what you are used to:

- Issue 7: you are probably comparing against `aspect-fold-500`, which has
  more chroma slots than either base wire.
- Issue 3: damaged packets shown instead of held.

If reverse still shows a colour shift after fix 1, it is one of those two, or
something in the capture chain; a capture file would settle it.

## 3. Band-limited packets shown damaged (partly fixed)

Reproduction: 2× playback captured at 48 kHz, which removes the top half of
the band.

- Stock packet: pilot noise 2.98, over the 2.0 gate, so the receiver holds the
  last good frame.
- Nested packet: pilot noise 0.28, because auto-levelled bodies make the
  pilots loud relative to the residual. It passes the gate and is displayed.
  Luma error rises and chroma falls (Cb transfer 0.45 to 0.30).

That is a real desaturation path, and it only affects nested packets.

Done so far, in `test_modem_v7/nested_fold.py`:

- `slot_noise` now floors each slot's noise at a share (`SLOT_NOISE_SHARE`,
  0.4) of the equaliser's own per-slot estimate, so slots in a dead part of
  the band fade instead of being read as confident stairs.
- `packet_noise` subtracts what the equaliser already attributes to the
  signature slots.
- `Recognition` keeps the nested/stock decision and level for 36 packets when
  the signature cannot be read.

Effect: stereo nested at 2× on 48 kHz, 975 to about 2,035 effective luma
coefficients; mono nested under a 4 kHz low-pass, 693 to 959 (stock mono 649).

Still open, and a policy decision rather than a bug: should a nested packet
with half its band missing be shown or held? The repo rule is "display a
partially decoded frame as-is", which argues for showing it, but then chroma
should fade toward the last good frame rather than toward grey. Options:
gate on the equaliser's per-slot noise instead of pilot noise, or hold chroma
only.

## 4 and 5. Grain and banding (open)

Measured through the display path, three Kodak pictures, error energy by
radial band:

| | Band (cycles/picture height) | Nested | Stock, same wire |
|---|---|---|---|
| Stereo, clean | 12–24 | 6.4 | 0.83 |
| Mono, clean | 0–12 | 2.7 | 1.2 |

Flat-area grain is 4–11% higher than stock when clean, and up to 45% higher
for stereo at hiss −40 dB.

So the fold wins on total error and on fine detail, and pays for it in the
low and middle bands, which is where the eye is most sensitive. That is the
grain. Cause:

1. **Stair quantisation of hosts.** A host is sent as
   `step * round(a / step)`. The rounding error is deterministic and signal
   dependent, not noise-like. Small host values round to zero (a dead zone),
   which on a smooth gradient gives contours: the banding. It is also
   identical frame to frame on a still scene, so the eye cannot average it
   out the way it averages hiss.
2. **Stereo puts the error where it is amplified.** In `stereo-nested` the
   slots chosen to fold are the weakest-base ones, and those carry the
   strongest riding detail (slot = base ± 0.35·detail). Stair error on the
   slot enters the detail estimate multiplied by 1/0.35.
3. **The design objective was unweighted MSE.** Table design
   (`tools/v7_nested_build.py`) minimises total squared error, so it happily
   moves error from 40 c/ph, where nobody sees it, into 15 c/ph.

### The filter to build

Yes, this needs a proper filter, in three parts. In order of expected payoff:

**A. Perceptual weighting in table design (sender tables only).**
Weight each coefficient's error by a contrast-sensitivity curve before
choosing hosts, guests and `kappa`. Practically: hosts below roughly
20 c/ph stay linear (no stair, no guest), and folding is confined to slots
whose error lands above that. Costs some effective coefficients on paper;
the target is band error at or below stock in 0–24 c/ph while keeping the
fine-band gain. Receiver unchanged: tables already declare which slots fold.
Acceptance: the two rows in the table above at or under stock, and
flat-area grain within 2% of stock when clean.

**B. Stereo: stop folding under riding detail.**
On folded slots set the ride to 0 and carry that slot's detail coefficient as
a guest instead (it is a coefficient like any other). Removes the 1/0.35
amplification. Needs a table field and a rebuild; the paired decoder
(`_decode_paired`) already handles per-slot roles.

**C. Subtractive dither on the stairs.**
Add a pseudo-random offset `d` in ±step/2 before rounding, and subtract it
after decoding. Seed from slot index and a packet counter that both ends
already share. This turns the dead zone and contouring into noise-like error
of the same power, which removes the banding outright, and because the
offset changes every packet the error averages down over successive packets
of a still scene. In stereo, use opposite-sign dither on the two channels so
their stair errors partly cancel in the average. Soft decoding carries over:
the posterior mean is computed on the shifted lattice. Cheap, table-free,
and the one I would do first if only one is done.

**D. Receiver display filter (optional, after A–C).**
The soft decoder already computes a posterior variance per coefficient and
throws it away. Keep it and use it as a per-coefficient Wiener weight at
display time, so coefficients the decoder is unsure of are shrunk rather
than shown at face value. Also the right hook for fading chroma toward the
last good frame in issue 3.

A and B change tables, so they require a retune (`tools/v7_nested_build.py`)
and re-running `tools/v7_nested_eval.py` torture and playback. C changes
sender and receiver together and is not readable by a receiver without it,
so it needs its own level-pattern rows or a new signature.

### Part C as built (subtractive dither)

Code: `test_modem_v7/nested_fold.py` (`DITHER`, `FrozenFold.offset`,
`Held`), kernels `*_dithered` in `tools/v7_sk_fold.py`. No table change.

- Every folded slot's staircase is shifted by a known offset within half a
  step. There are seven offset sets; the packet's tail slice (the sender's
  counter modulo 7, which the metadata already carries, forwards and in
  reverse) picks one. The second stereo channel uses the first's offsets
  negated, so a mono sum is read exactly as before.
- A dithered packet carries its level pattern negated. The receiver reads
  the sign per packet, so plain-stair packets still decode. A receiver from
  before this change does not read dithered packets (it sees no nested
  signature); `NESTED_FOLD_DITHER=0` on the sender sends plain stairs.
- `Held`: the receiver shows the mean of the last seven decoded pictures
  while the picture is not moving. Motion is read on the packet-to-packet
  change of the hosts, against what noise and stair error explain; a moved
  picture, a level change, a packet out of sequence or plain stairs start
  again from the packet in hand, shown as decoded. `NESTED_FOLD_HOLD=0`
  turns the average off.
- Dithered packets are not passed through `--temporal-fusion held`, which
  averages slot values before the fold is undone.

Measured through the live sender and receiver, clean, on the repo's 3:4
motion fixture frame and the reference face (luma error energy by band,
cycles per picture height; "held" is the picture shown after one cycle):

| | Band | Stock, same wire | Plain stairs | Dithered, first packet | Dithered, held |
|---|---|---|---|---|---|
| Mono | 0–12 | 0.10 | 13.8 | 12.7 | 2.0 |
| Mono | 12–24 | 112 | 44.0 | 47.2 | 34.6 |
| Stereo | 12–24 | 0.55 | 43.8 | 45.5 | 9.6 |
| Stereo, hiss −45 | 12–24 | | 84.1 | 87.6 | 18.1 |

On a moving picture (the fixture movie, a new frame every packet) the
average never engages and the error is that of a single packet: mono about
1% above plain stairs, stereo equal. So dither fixes the fixed pattern
(banding, static grain on stills and slow scenes); it does not lower the
error of one packet. That needs parts A and B, which need a table rebuild.

### Decoding inside the bounds (receiver only)

Code: `smooth_within_bounds` in `tools/v7_sk_fold.py` (numba);
`FrozenFold.room` and `FrozenFold.clean` in `test_modem_v7/nested_fold.py`.
No sender or wire change; it works on a single packet, so also in motion.

What arrives is a set of bounds, not values: a folded host lies within half
a stair of what was decoded, a plain host within its noise, a guest is taken
as read, and a coefficient that was not sent is unknown. The receiver used
to show the middle of every bound and zero for everything unsent. It now
shows the picture of least total variation that fits the bounds (projected
gradient descent with momentum, 16 passes, about 2 ms). Flat areas come out flat and the
ringing of the cut-off spectrum goes, because the unsent coefficients are
free to fill in.

- `--nested-smooth PASSES` on the receiver, or `NESTED_FOLD_SMOOTH`: the
  strength; 0 shows the decoded values as they are.
- `NESTED_FOLD_SMOOTH_ROOM` (default 0.5): the share of the half stair a
  folded host may move. 1 is flattest and starts to look painted; 0 leaves
  every sent value alone and only fills in what was not sent.
- One channel or a mono sum of the stereo table is shown as decoded: there
  only base plus or minus detail is bounded, not each coefficient.

Luma error through the live sender and receiver, clean, one packet:

| Picture | Mono nested, as decoded | Mono, shown | Stereo nested, as decoded | Stereo, shown |
|---|---|---|---|---|
| Reference face | 0.0755 | 0.0678 | 0.0681 | 0.0604 |
| Robot movie frame | 0.2584 | 0.2487 | 0.2004 | 0.1918 |
| 1/f noise (test texture) | 0.0942 | 0.0994 | 0.0819 | 0.0891 |

It costs dense texture with no flat areas about 5 to 9%. Judged by eye on
the fixture movies: the mottle on flat dark areas is gone and text edges are
cleaner; fine vertical stripes in `stereo-nested` on the line chart are
reduced, not removed.

Seen in the same pictures and not addressed: false colour in every mode,
stock included (green and purple on the black-and-white checkerboard, pink
on white bars, colour bars smearing upward). That is the chroma of the base
wires (issue 7), not the fold.

### Cost per packet and the downscale kernel

Per packet through the live receiver on the sandbox's two cores, clean
link: mono nested 9.7 ms to 5.0 ms, stereo nested 14.5 ms to 6.3 ms.

- The stair decoder skips terms more than seven noise deviations away
  (2.7 ms to 0.2 ms clean, about 2 ms at heavy noise; results change by
  under 1e-7).
- The stereo decode reuses the stair decode for the averaged picture.
- The smoothing carries momentum: 16 passes reach what 40 plain ones did
  (5.3 ms to 2.2 ms).
- Sender: a held picture's level is chosen once (0.5 to 0.8 ms a packet).
  `dct_kernels/viewer_solve.py` now solves on the coefficient grid with a
  compiled conjugate gradient (`animation_modem/v7_kernel_solve.py`);
  output unchanged to 1e-15, about 4.8 ms to 1.5 ms a frame.

Downscale kernels were scored through the live nested sender and receiver
at a bilinear display (SSIMULACRA2, higher is better; mean of astronaut,
chelsea, coffee, rocket and the reference face):

| Kernel | Mono nested | Stereo nested | Stock stereo-slices |
|---|---|---|---|
| none (reference) | -52.09 | -45.25 | -43.00 |
| viewer_solve | **-50.40** | -42.84 | -43.71 |
| upscale_precomp | -50.79 | **-41.91** | -42.55 |
| csf_peak | -50.44 | -44.71 | |

`aspect-mono-nested` keeps `viewer_solve`. `stereo-nested` used to inherit
"no kernel" from `stereo-slices`; its default is now `upscale_precomp`, in
the GUI and the CLI. Kernel parameters were left at their defaults; none
was tuned for the folds.

## 12. Through the real display the folds do not beat stock (open, the main finding)

Everything above was judged on the decoded 96x80 grid or on coefficient
error. The receiver does not show that grid: `tools/v7_gl_viewer.py` puts it
through 4x DCT reconstruction, edge reconstruction at 75% and luma-guided
colour. Edge reconstruction already does what "decoding inside the bounds"
does for unsent coefficients (it keeps every received coefficient exactly
and fills the rest by total variation), and it treats every non-zero
coefficient as received. So it rebuilds detail from stock's clean
coefficients, and it keeps the folds' stair-noisy ones exactly.

`tools/v7_nested_display_eval.py` scores through that path (SSIMULACRA2,
higher is better; astronaut, chelsea, coffee and the reference face, none of
them used to build tables; 3:4 layout):

| Mode, tables | Clean | Hiss -45 | Hiss -35 |
|---|---|---|---|
| aspect-mono-500 (stock) | -38.8 | -47.5 | -54.9 |
| mono nested, shipped tables | -38.2 | -53.1 | -60.4 |
| mono nested, shipped, smoothing off | -39.1 | -51.8 | -59.9 |
| mono nested, rebuilt on photo samples | -39.0 | -52.3 | -60.2 |
| stereo-slices (stock) | -19.9 | -26.2 | -42.7 |
| stereo nested, shipped tables | -26.3 | -38.3 | -53.4 |
| stereo nested, rebuilt on photo samples | -21.3 | -36.8 | -50.3 |
| stereo nested, rebuilt, 900 of 1,020 slots plain | -20.7 | -29.7 | -46.2 |

- Mono nested equals stock when clean and is 5 to 6 points worse under hiss.
- Stereo nested is worse than stock in every condition.
- A sweep of the step size (kappa 6 to 28) and of the number of plain slots
  went one way only: coarser steps and fewer folded slots score better, that
  is, the nearer a table is to stock the better it looks. Finer steps score
  worse, because the guest gets less room. The table rebuild that parts A
  and B call for does not fix this.
- By eye the folds are sharper than stock and noisier; the metric prefers
  clean and soft. That judgement needs a person.
- The training pictures matter: the photo samples
  (`tools/v7_nested_corpus.py`) give better stereo tables than the shipped
  Kodak ones here; adding textures and the test charts made them worse.
  The shipped tables are unchanged: with the rebuilt set one reverse test
  fails, and no table beats stock anyway.

## 13. False colour is the chroma budget (diagnosed, open)

Green and purple on black-and-white detail, pink on white, colour bars
smearing upward, in every mode. On the mono wire 76 Cb and 136 Cr
coefficients are sent of 1,920 each. Swapped in through the display: the
sent chroma arriving exactly looks the same as the decode; the full chroma
is clean. So it is neither coding error nor the display's colour step, it
is how little chroma is sent, and anti-ringing cannot repair that. A
receiver-side rebuild that lets colour change only at luma edges confines
the bars a little on the chart (CIEDE2000 18.4 to 17.5) and makes the
astronaut blotchy (SSIMULACRA2 -53.6 to -57.3): not adopted. The fix is
more chroma on the wire.

## 10. Body level since `b147922` (by design)

`encode_pulse_frame_coeffs` now fits the body 1.5 dB under the lower of the
header and end-marker peaks, measured after the timing tones are mixed. The
end marker sits about 1.5 dB under the header, so every EOF packet's body,
stock profiles included, goes out about 1.5 dB lower than at `d658975`
(body RMS 0.147 to 0.124 on `aspect-fold-500`, 0.112 to 0.094 on mono
nested). Clean scores are identical. At hiss −45 the effective luma
coefficients fall 3 to 6% (mono nested 1,324 to 1,247, stereo nested 2,146
to 2,078); under soft saturation they rise 3 to 12%. Whether the marker
needs that margin is a tape question.

## 11. Sender GUI output device (fixed)

`tools/v7_send_gui.py` passed the sender a bare PortAudio index. The GUI
reads its device list once, in its own process; the sender is a new process
with a fresh list. If a device was plugged in or removed after the GUI
started, the same index named a different device and the sender opened it.
The GUI now also passes `--device-name` and `--device-hostapi`, and the
sender (`_resolve_named_send_device` in `tools/v7_live.py`) checks the index
against the name, corrects it, and refuses to start if the device is gone.
The GUI's own list is still only as fresh as its PortAudio session.

## 6. Stereo nested under a 4 kHz low-pass (open)

Effective luma coefficients 1,126–1,145 against 1,516 for stock slices. The
loss is in the detail zone: guests in the surviving band are being read
while their partner slots above 4 kHz are gone. Fix B should help because it
removes detail from the folded slots. Mono-sum is the other stereo torture
case behind stock, by 3%.

## 7. Chroma budget of the base wires (open, by design)

`aspect-mono-500` and `stereo-slices` carry fewer chroma slots than
`aspect-fold-500`: chroma transfer about 0.90 against 0.94, chroma error
about 25% higher. Stock mono and stock slices show the same. If these modes
are to match the stereo M/S profile's colour, the options are to give back
some of the folded luma gain as chroma slots, or to fold chroma guests the
same way luma guests are folded.

## 8. Cold start (open, minor)

`Recognition` is sticky once it has seen a readable signature. If the very
first packets of a stream have unreadable signatures it falls back to a
score comparison (nested score above stock and above 0.25), which can pick
wrong for those packets. Self-corrects at the first readable signature.

## 9. GUI kernel defaults (fixed)

`tools/v7_send_gui.py` looked up kernel defaults by profile name and found
nothing for the two nested names, so they started with no kernel values.
All four lookups now go through `NESTED_BASE_PROFILES`.

## Test notes

Failing in this sandbox before and after this work, unrelated to the fold:
the GUI picker click test
(`test_video_audio_source_and_input_pickers_are_clickable`), the test that
needs PortAudio, and a wall-clock timing test in the mono-video suite.

## Not yet validated

Real tape, faces, and moving pictures at length. Dither's temporal averaging
in particular is a claim about motion that stills cannot confirm.
