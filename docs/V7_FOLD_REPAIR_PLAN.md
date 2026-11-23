# V7 nested fold: repair plan

Written 2026-10-07. Replaces the "next work" list in the handoff. Background
and every measurement quoted here: `V7_NESTED_FOLD_ISSUES.md`.

## Where we are

The fold works as a way of carrying more: it sends about 2,000 luma
components where `aspect-mono-500` sends 1,550, and reaches the edge of the
96x80 grid where stock reaches about 51x38. On the project's test screens
that is 44 to 50% more resolved detail. Resolution is not the problem.

The problem is noise handling. Three things are wanted:

1. keep the higher resolution;
2. no stair-stepping;
3. black shown as black, not purple.

## Standing rules

- **This is a video project.** Tables and defaults are built for ordinary
  moving pictures. The line, moire and colour screens are for testing
  resolution and for calibration. They are never training material and never
  get a table of their own.
- **The receiver is handed an arbitrary signal.** Nothing may depend on
  choosing a setting to suit the line. The torture test is the judge.
- **Only four modes are compared:** `aspect-mono-nested` against
  `aspect-mono-500`, and `stereo-nested` against `aspect-fold-500`. The other
  profiles are deprecated.
- **Assume a slow machine.** Per-packet and per-frame work is compiled
  (numba) and measured; no BLAS in anything that runs per packet.
- **Per-packet choices ride on what is already transmitted** (the marker
  slots and the packet metadata). No new side channel.
- **Judge pictures, not only numbers:** every stage ends with side-by-side
  renders through the receiver's normal display, on video.

## Order of work

### Stage 0. One test everybody trusts

Before changing anything else, so each later stage has a before and after.

- One command that runs the whole torture matrix for the four modes and
  writes one table: the original score (effective components) and the
  display score, brightness and colour reported separately.
- Pictures: frames of real video first. The fixture screens and moire
  patterns are reported in their own rows, as resolution and calibration
  checks.
- **Needs from Ben:** a few minutes of ordinary footage (or a few hundred
  frames). Until then the stand-in is nine bundled photographs, which is
  thin.
- Done when: the table reproduces the known results (mono nested ahead of
  stock on the screens; identical decode to `d658975` with the work-arounds
  off).

### Stage 1. Levels (first fix)

Intended: header and end marker 1 to 1.5 dB under clipping; the picture body
1 to 1.5 dB under those. Measured today:

| | Now | Target |
|---|---|---|
| Header peak | -2.7 dB | -1 to -1.5 dB |
| End marker peak | -4.4 dB | -1 to -1.5 dB |
| Picture body peak | -5.5 dB | 1 to 1.5 dB under the two above |

- Raise the end marker to the header's peak, and both to the target.
- Fit the body 1.25 dB under the lower of the two after the timing tones are
  mixed (the fit that exists, with new numbers). Metadata stays just under
  the body.
- Worth about 3 dB of signal-to-noise for every mode.
- Checks: nothing reaches full scale on any fixture; header and end marker
  still clip first under the soft-saturation torture case; the pulse edge
  detector and end-marker detection still pass at a quarter and four times
  the level; MP3 at 192 kbit/s and below (a code comment says a header nearer
  full scale costs frames there: re-measure, do not assume).
- Wire levels are pinned by hash in the tests; those hashes change and are
  updated in the same commit.

### Stage 2. Black

Colour is accurate apart from being a little unsaturated and wrong in black
(and pink on white). Cause, measured: the wire carries 76 and 136 colour
components per plane of 1,920, and the leftover ripple is shown as colour
where no colour can exist.

- Move sender and receiver to the usual broadcast ranges: luma 16 to 235,
  colour 16 to 240, with the footroom and headroom that implies, in place of
  today's full-range mapping.
- Measure, on the black-and-white checkerboard beside the colour bars and on
  video, whether that alone removes the purple. If ripple at black still
  shows as a tint, add the other half of broadcast practice at the receiver:
  bring colour that is outside the legal range for its brightness back
  inside it (black and white then carry no colour by construction).
- Cost to watch: footroom and headroom use about 1.3 dB of the body's
  amplitude. Report it against the Stage 1 gain.
- Done when: black areas beside saturated colour are neutral on the chart
  and on video, and the colour score on video does not fall.

### Stage 3. Dither

Dithering the stairs on the wire was the wrong place. It turned error that
plain rounding concentrates in a few components into noise across all of
them, flat areas included; single frames scored worse (stereo -35.8 against
-31.0).

- The sender never dithers. The receiver keeps reading dithered packets for
  one release so existing recordings play, then that path is deleted.
- The held-picture average and the decoder-side smoothing stay off.
- If masking is still wanted after Stage 4, use the display's own dither and
  grain stage, which already exists, tuned on video.

### Stage 4. The fold itself

One rebuild, judged by Stage 0. In this order, each measured before the next
is added:

- **4a. Put the rounding on the fine component.** Today the coarser
  component of each folded slot is rounded to a stair (about 18 dB of
  precision in mono, 11 dB in stereo) and the finer one rides inside
  accurately (about 26 dB). Swap them: fine detail takes the rounding,
  where it is hard to see, and the coarser component rides inside and
  degrades smoothly with noise. This is also the answer to "the step is
  fixed for a line the sender cannot see": the step then only governs fine
  detail, which is what should go first on a bad line. Estimate, not yet
  measured: about 25 dB for the coarser component on a clean line, falling
  with noise. Prototype at table level, then through the wire.
- **4b. Stair size by frequency.** One step for every slot is a design flaw.
  Each slot already has its own step in the table file; the builder sets
  them all from one number. Grade them so error goes where it is least
  visible.
- **4c. Stereo.** Stop folding into slots where a detail component is
  riding, or carry that detail as the tucked-in component, so stair error is
  never divided by 0.35. (Proposed; not yet discussed.)
- **4d. Which components are folded.** Choose them as a compact block in
  frequency order, so each fine component arrives with its neighbours,
  instead of by energy alone. One set, for video.
- **4e. Tables.** Built from video only. Screens and patterns are scored,
  never trained on.

Done when, across the torture matrix on video: nested is at or above its
aspect mode in every condition where it is today, and no worse than 1 point
behind anywhere else; stair error in flat areas is not visible in the
renders; the screens still show the resolution gain.

### Stage 5. Receiver noise reading

On a clean mono link the receiver reads the noise at 0.8 to 0.9 of the
design level when the true figure is about 0.46, so it decodes too
cautiously. Fix the estimate and re-run Stage 0. Independent of Stage 4 and
can be done alongside it.

### Stage 6. Speed on slow machines

- A fast compiled transform to replace matrix multiplication in the
  per-packet paths, and BLAS out of everything that remains.
- Budget set on a slow machine, not the development sandbox;
  `tools/v7_cpu_per_call.py` is the measure.

## Not in this plan

- A larger picture grid. Mono nested already reaches the edge of 96x80;
  going further is a bigger change than this repair.
- More colour on the wire. Revisit after Stage 2 if colour is still short.

## State of the code when this was written

- Unmerged, in the patch that carries this file: work-arounds off by
  default; deprecated profiles removed as choices; the display scoring tool
  (`tools/v7_nested_display_eval.py`); the slow-kernel check no longer times
  a kernel's first call.
- Shipped tables are the original Kodak ones. A trial build that included
  the test screens and patterns was measured and not adopted.
