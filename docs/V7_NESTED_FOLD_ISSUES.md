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
| 4 | Grain | Open. Cause known; needs the filter below |
| 5 | Banding | Open. Same cause; needs dither |
| 6 | Stereo nested behind stock slices under a 4 kHz low-pass | Open |
| 7 | Mono and slices wires carry less chroma than `aspect-fold-500` | Open, by design of the base wires |
| 8 | Cold start when the signature is unreadable | Open, minor |
| 9 | GUI kernel defaults ignored for nested profiles | **Fixed** |

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
