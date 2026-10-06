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
| 4 | Grain | **Reduced on held pictures** by dither (part C, done). Single-packet error unchanged; parts A and B still open |
| 5 | Banding | **Fixed** by dither (no dead zone, no fixed contours) |
| 6 | Stereo nested behind stock slices under a 4 kHz low-pass | Open |
| 7 | Mono and slices wires carry less chroma than `aspect-fold-500` | Open, by design of the base wires |
| 8 | Cold start when the signature is unreadable | Open, minor |
| 9 | GUI kernel defaults ignored for nested profiles | **Fixed** |
| 10 | Body sent 1.5 dB lower since `b147922` (all profiles) | Open, a level-budget decision |
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

## 10. Body level since `b147922` (open)

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
