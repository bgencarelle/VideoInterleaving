# V7 handoff — nested folds, adaptive tables, hot recordings (2026-10-12)

Start here, then `docs/V7_FIXES.md` (the working list; items 14, 28, 35 carry
this session's reports). Not reviewed by the owner unless marked.

## How the owner works (keep to this)

- Nothing is coded until the owner says go. Say back what you understood
  first; one short question at a time. Owner uses speech-to-text: numbers and
  words can come through wrong, so confirm item numbers.
- Testing means going through the modem and looking at what comes out, at
  96 kHz both ends. One picture set, judged by eye. No torture matrices, no
  endless rounds, no sleeping between steps: make, run, look, report.
- No charts or tables in explanations. Pictures of modem output are welcome.
- Short answers. Goal for the codec: show as much as possible above a
  threshold. No legacy compatibility to protect.
- The owner applies patches himself (`git format-patch` against
  `origin/scope-mode-upgrade`, applied with `git am`); pushing from the
  session was refused (403).

## State of the code

Upstream `scope-mode-upgrade` at 4260206 has the first batch: false-header
and overlapping-preamble fixes, the stereo chroma double-scaling fix,
luma-adjust masks on the enlarged grid, the posterior-mean serpentine
decoder, serpentine mapping on the stock nested tables with centroid stairs,
and adaptive tables running on the stock serpentine nested fold with the
level chosen per picture.

Waiting in the patch (6 commits on top of 4260206):

1. Nested stereo: on folded slots the riding detail also travels as the
   first channel's guest; the receiver blends guest, two-channel reading and
   prior by variance (`sk.blend_detail`), and the level trial does not score
   those guests. Removes the mesh and beaded lines in stereo nested and
   adaptive stereo; better at every noise level. Switch:
   `NESTED_FOLD_DETAIL_GUESTS=0` (both ends).
2. Tape model: `cassette-i-hot` records 4 dB over the normal cassette case
   (it was 4 dB over the wire, which with the record boost hard-clipped the
   tape 14 dB over its maximum).
3. Adaptive table `chroma50.npz` reshaped: luma guests fill outward from the
   hosts by radius, no thin strips along the first rows and columns (they
   drew a cross through a Siemens star and streaks in a flat sky). Tool:
   `tools/v7_reshape_guests.py ORIGINAL.npz OUT.npz`. SHA updated in the
   docs. The GUI patch test now matches the adaptive tables running on the
   stock fold.
4. Docs: item 14 second cause, item 28 next steps, new item 35.

Tests: `modem_tests.test_v7_nested_fold`, `test_v7_adaptive_pipeline`,
`test_v7_nested_gui_patch` pass except one stale upstream test
(`test_the_sender_gui_offers_both_options` expects `viewer_solve` for
adaptive mono; upstream 42a2163 changed the default to `reference`).

## What was learned

- The adaptive tables send the same 2,880 luma coefficients at nearly the
  same picture frequencies as the stock tables; their 120x100 grid cannot be
  sharper as built. More detail needs more coefficients (fold the plain
  hosts, nest deeper, or interleave fine coefficients across packets on
  still pictures; all discussed, none started).
- Grain on flat areas (item 14) is mostly the coefficients that are not
  sent: edges elsewhere ripple into flat areas. The receiver's smoothing
  (`NESTED_FOLD_SMOOTH=20`) flattens it and scores +1 to +2 dB on some
  pictures, but the owner rejected the look (it wipes real texture on coins,
  grass, skin). It stays off.
- Hot recordings (item 28): the modem holds at 4 dB over a normal cassette
  recording, degrades at 8, and the test frame fails at 12 (the face
  survives). Tilting the top carriers and peak clipping do not help.
  Relaxing the gates alone shows nothing more.
- Distortion cancelling (prototype only, item 28 next steps): rebuild the
  packet from what the receiver trusts, run it through the chain's bend,
  subtract the predicted distortion, decode again, widening the trusted set.
  It must work on the raw equalised carriers (before the Hadamard spread is
  undone) with the known pilots on every carrier; rebuilding from decoded
  picture values does nothing because the spread mixes bad carriers into
  every value. Result: 8 dB over 11.3 -> 12.5 dB and visibly sharper; 12 dB
  over 0 -> 52 of 56 packets shown. The prototype assumed the chain known;
  the real version fits it from headers, pilot tones, EOF and metadata.
  General transmission problems, not tape alone. For lossy codecs the plan
  is a codec map learned offline from noise pictures (receiver weights,
  sender shaping); the pilot tones are the calibration signal.
- Header vs body: body peaks about 2 dB under the header/EOF (target 1 to
  1.5): item 35. On average level the gap is about 11 dB, which is the body's
  crest factor, not a level setting.

## Open next (owner's choice)

- Item 28: build the receiver-side cancelling (bend fitted from the known
  parts, trust by measured carrier noise). Codec map after.
- Adaptive tables: remaining sky streaks on the rocket look like chroma
  (only luma guests were reshaped). Old stereo's tables have a milder strip
  of the same kind.
- `docs/V7_ADAPTIVE_INTEGRATION.md` still describes the separate serpentine
  codec (host level -1, step 0.25); the adaptive path no longer uses it, and
  the constants in `tools/v7_adaptive_pipeline.py` are now unused there.
- Items 13, 20 (owner testing via `main.py --mode modem`), 22, 32 open; 27
  long-term; 19 and 33 solved, awaiting a second pass.

## Test rig (scratch, rebuild as needed)

The session's scripts lived in a scratch folder and are not in the repo.
The method:

- Audio: a PulseAudio null sink at 96 kHz (`XDG_RUNTIME_DIR=/run/user/0`),
  the real sender (`tools/v7_live.py send --device pulse --rate 96000
  --profile ...`) with `sounddevice.OutputStream.write` wrapped to keep what
  it writes; impair that recording in a file with
  `tools/v7_torture_matrix.impair` (or ffmpeg for codecs); play it into the
  real receiver (`receive --device pulse --headless --save-dir ...`) with
  `capture_rate_for` pinned to 96000. Frames are saved as PNGs; compare with
  the source at the decoded size.
- Fold only (no audio): `nested_fold.slice_wire(layout, send=True, path=...)`,
  `channel_coefficients` per side with optional noise, `half` and `values`
  on the receiving end. Fast for checking where error comes from: split the
  luma error by coefficient class (hosts plain/folded, details, guests per
  channel, not sent) in the DCT domain.
- Useful hooks (monkeypatched in the scratch scripts): receiver gates
  (`v7.LIVE_MAX_PILOT_NOISE`, `DISPLAY_MAX_PILOT_NOISE`,
  `LIVE_VALID_HEAD_CONFIDENCE`), `v7.DEBUG['Z']`/`['H']` after
  `decode_frame` for raw carriers, and `v7_live._encode_pulse_frame_coeffs`
  to make the real sender rebuild packets from given values (it matches the
  original to 40 dB).
- Pick the kill target by process id: a `pkill` pattern that matches your
  own shell kills it.
