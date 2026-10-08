# V7 fixes

A working list of faults found in V7 and how each one is being dealt with.
`docs/transport_v7_spec.md` is considered dirty while this list is open.

## How this list works

1. **One item per iteration.** Take the top open item, work on it alone,
   report back, then stop.
2. **Every item carries a status:** `open`, `solved`, `not fixed yet` or
   `unsolvable`. A report says which, and how it was checked.
3. **New findings are added** at the bottom as they turn up, with the file
   and, where known, the lines.
4. **Anything that is breaking other work moves to the top.**
5. **When the list has been worked through, go round again** and test every
   `solved` item on the live path. If the fault is really gone, delete the
   item and any code left behind for it (switches, work-arounds, dead
   branches, tests that only pinned the fault).
6. **Nothing is changed without a go** from the owner.

## How anything on this list is tested

1. A test counts only if it runs the live sender into the live receiver and
   judges what the live viewer draws.
2. Audio always goes through PortAudio and the picture through the live
   viewer, on a virtual device and a virtual screen if need be. If either is
   missing, install it or ask. Never substitute a stand-in silently.
3. Before a test tool is trusted, show it is bit-identical to the live path
   on one picture, at the input to the modem and at its output.
4. The first run of a session is the current build on a clean signal. Every
   other result is shown beside it.
5. Judge by the repo's own charts: Siemens star, zone plate, wedges, line
   widths, the video test frames and the reference face. Pictures first,
   numbers second.
6. Same for same: same picture, same settings, same packet position, one
   change at a time.
7. Both ends use the GUI defaults unless the test says otherwise, and the
   settings used are stated.
8. Clean signal first. Noise and damaged-line tests come after the clean
   result is agreed, and use only cases from the torture list
   (`tools/v7_torture_matrix.py`).
9. Every result says how it was obtained: live run, read in the code, or
   tool only.
10. Sender and receiver both run at 96 kHz unless the test is about a rate
    difference. Confirm it from the receiver's own start-up line before
    trusting a result. (MP3 cases go through the encoder at 48 kHz, since
    MP3 has no 96 kHz mode.)
11. Nothing may lie above half the sample rate when samples are written. A
    filter used for that must have finished cutting by that frequency, not
    merely have its corner there.
12. Use test pictures that pass cleanly at normal speed: the reference face,
    the slanted edge and the video test frame. Charts finer than the wire can
    carry are for judging resolution only.

## How to add an item

Copy this and fill it in. Leave out what is not known; do not guess lines.

```
### N. Short name
- Status: open
- What is wrong: one or two plain sentences.
- Where: path/to/file.py, lines A-B (or "not located yet").
- How it was found: live run | read in code | tool only.
- What "fixed" looks like: the check that will show it is gone.
- Notes: decisions, dependencies, things tried.
```

## The list

### 28. Levels are too low: the picture data sits far under the ceiling
- Status: not fixed yet (levels, EOF and leveler done; peak reduction built,
  left off; one regression open, item 30)
- What is wrong: the picture data's average level, which is what noise
  competes with, was held far under the ceiling: the EOF was 1.3 dB under
  the header, header and EOF peaked 3 dB under clipping, most packets' bodies
  were not raised to the ceiling, and each body's rare spikes set the level
  of the whole packet.
- Where: `animation_modem/v7.py`: `HEADER_PEAK_DB`, `emitted_pulse_level`,
  `body_auto_level`, `encode_pulse_frame_coeffs`, `PEAK_CLIP_DB` and
  `_reduce_peaks`.
- How it was found: measured on the sender's own output.
- What "fixed" looks like: header and EOF peak at the same level, 1 to 1.5 dB
  below clipping; the data's average several dB higher; live pictures clean
  and with hiss better than before, and no worse through the MP3 cases.
- Report: done: the EOF is sent at the header's peak; header and EOF now
  peak at -1.0 to -1.4 dBFS (were -2.7 and -4.0); every packet's body is
  levelled to 1.5 dB under its framing (was: only loud ones). The data's
  average went from -17.5 to -14.5 dBFS on the face, -18.2 to -15.1 on the
  test frame. Live, PortAudio loopback, 48 kHz both ends, the sender's own
  audio impaired then played into the live receiver, pictures scored against
  the source: clean and light hiss unchanged; heavy hiss clearly better
  (face, hiss 25 dB down: 25.6 to 27.2 dB PSNR; test frame, hiss 30 dB down:
  11.8 to 13.1; hiss 25 dB down: no pictures before, 32 of 57 now).
  Peak reduction (clip and filter) is built and switched off: clipping 9 dB
  over the average raised the data a further 2.1 dB and helped only under
  heavy hiss (+0.5 to +0.9 dB), cost 0.4 dB on a clean line, and on the test
  frame dropped two packets in every eight live, cause unknown; clipping
  7 dB over was worse everywhere.
- Report, second round (all at 96 kHz): the live leveler now sets its gain
  from each complete packet's own header and EOF, putting their peak at
  -1 dBFS (it used to put the loudest 0.5% of all samples at the old header
  level, so data and saturation steered it, and the two channels of one
  signal could sit at different gains). The header detector's switching
  level follows the level the leveler restores the header to. The EOF is now
  band-limited like the header: square, it rang 1.5 dB over at 96 kHz and
  the sender's resampler turned the whole packet down to hold it. Header and
  EOF now both peak at -1.2 to -1.3 dBFS at 96 kHz; the data's average is
  3.1 dB above the original build. Live, 96 kHz both ends, original build
  against this one, test frame: clean 57 to 66 pictures, hiss 30 dB down
  none to 58 (11.4 dB PSNR), type I tape 12.8 to 13.5 dB; face: hiss 25 dB
  down 22.2 to 24.8 dB, hiss 30 dB down 26.0 to 27.5 dB. Clean and MP3 192
  unchanged.
- Report, third round (realistic tape, 96 kHz): cheap decks go in too
  quiet, so the cassette cases now record 8 dB (type I) and 4 dB (type II)
  under the wire with the tape's own hiss; reel-to-reel is aligned. On those,
  this build is as good as the original or better: face, worn type I 28.3 to
  28.5 dB; test frame, worn type I 13.7 to 14.1 dB and 60 to 66 pictures.
  Only a cassette recorded 4 dB hot is worse (face 26.4 to 23.6 dB; test
  frame 22 pictures to none): the extra level drives the top carriers into
  the tape's saturation.
- Left over: a cassette recorded 4 dB hot is still worse than the original
  (test frame: no pictures; face 49 to 43 pictures since item 31). Peak
  reduction stays off.
- Second pass: header and EOF peak at -1.2 to -1.6 and -1.3 to -1.9 dBFS at 96 kHz (the sender's resampler holds back the inter-sample peaks, so some packets land a little under the 1 to 1.5 dB aim); data average -15.0 dBFS (face). Hiss 30 dB down and type I cassette: every packet shown. The hot cassette is still the one bad case (face 58 of 70 at 23.6 dB, test frame none).


### 8. The default kernel corrects for a viewer that is not the one in use
- Status: not fixed yet (the means is in; the default is the owner's call)
- What is wrong: `viewer_solve` models the receiver as shrinking to half the
  grid and enlarging bilinearly by 4. The receiver enlarges in frequency
  space, runs the edge rebuild, then draws bicubic.
- Where: `animation_modem/v7_viewer_model.py` (the one description of how
  the receiver draws); `tools/v7_gl_viewer.py` (display defaults now read
  from it); `dct_kernels/viewer_solve.py` (`viewer` setting 3 and
  `viewer_from`).
- How it was found: read in the code. Effect on the picture not measured.
- What "fixed" looks like: the sender's model of the viewer is taken from
  the same description the receiver draws with.
- Report: the receiver's display defaults now sit in one description that
  the viewer reads, and the kernel has a setting, `viewer=3`, that models
  that display. Against it there is nothing to correct: the receiver's
  frequency-space enlargement is the very picture the kernel aims for, so
  the solve hands back the plain coefficients. Live run, PortAudio
  loopback, 48 kHz both ends, real receiver GUI with untouched settings on
  a virtual screen, reference face, slanted edge and video test frame:
  `viewer=3` and kernel off put sample-identical audio on the device and
  the same picture on screen. Today's default (bilinear model at 20
  percent on stereo) shows slightly crisper than that; the same correction
  at full strength puts a bright rim round the slanted edge.
- Left over: the default has not been changed. Today's 20 and 25 percent
  were tuned by eye and act as a mild sharpening, not as a correction for
  the viewer in use. Matching the default to the receiver means the kernel
  does nothing; keeping it means the setting should be named for what it
  is. The edge rebuild and the final bicubic draw are in the description
  but not in the kernel's model (the rebuild is not a fixed filter). A fine
  grid texture shows in the flat areas of the slanted-edge picture in all
  three cases (see item 14).
- Notes: possible link to the squares and stairs seen earlier; to be settled
  on the charts by switching the kernel and the edge rebuild off in turn.

### 11. The test tool does not decode the way the live receiver does
- Status: open
- What is wrong: the tool hands the decoder a whole recording, which takes a
  different branch from the live rolling window, and it reads mono video
  from the left channel only while the live sender puts it on the right.
- Where: `tools/v7_nested_eval.py`, lines 98-115 (`_results`) and line 144.
- How it was found: live run (not bit-identical at the output of the modem).
- What "fixed" looks like: the tool's decoded picture is bit-identical to
  the live receiver's for one picture in every mode.

### 12. The test tool's drawing step has not been checked against the live viewer
- Status: open
- What is wrong: the tool draws with its own function. It has never been
  compared with what the live viewer puts on screen, including the final
  draw to the window.
- Where: `tools/v7_nested_display_eval.py`, line 52 (`display`).
- How it was found: noticed during testing; no comparison was ever run.
- What "fixed" looks like: a capture of the live viewer on a virtual screen
  matches the tool's picture for one picture in every mode.
- Notes: the live viewer now runs here. The real receiver GUI opens on a
  virtual screen (Xvfb with software OpenGL), is started on the loopback
  input with its settings untouched, and the screen is grabbed. Used for
  item 8. The comparison against the tool's own drawing is still to do.

### 13. The photo score marks a sharper picture as worse
- Status: open
- What is wrong: the display score used to steer earlier work rewards smooth
  pictures. It rated the build that resolves more on the sharpness charts
  below a smoother, lower-resolution one.
- Where: `tools/v7_nested_display_eval.py`, line 171.
- How it was found: live run on the repo's sharpness charts.
- What "fixed" looks like: no decision rests on that score alone; the charts
  are the judge (see the testing rules above).

### 14. Source of the grain in the nested modes is not known
- Status: open
- What is wrong: the nested modes show a fine grain over flat areas. Dither,
  held-picture averaging and the decoder's smoothing are all off by default,
  so it is not those.
- Where: not located yet.
- How it was found: live run on the sharpness charts.
- What "fixed" looks like: the cause is named and shown by switching it off.

### 15. The nested sender drops audio about five seconds in
- Status: open
- What is wrong: with `aspect-mono-nested`, the sender reports one output
  underrun about 65 packets after it starts, in every run. About 200 samples
  go missing from the stream there, one header is never found, and the
  receiver loses one or two pictures. Stock mono did not do it in six runs.
- Where: not located yet (sender side, `tools/v7_live.py`).
- How it was found: live run over PortAudio, ten runs, with both the
  untouched and the fixed receiver.
- What "fixed" looks like: a 12 second nested send with no underrun and no
  missing picture.
- Notes: seen in this test environment only; needs confirming on real
  hardware.
- Looked into: the sender on its own does not do it (four live runs, two of
  them nested: no underrun, no late frame, no late write). Choosing the
  step size for a new picture takes about 1 ms, so it is not that. It
  happens only with a receiver running beside the sender, on a two-core
  machine through a software sound device, and always at the same point.
  Cause not found. May belong to the test rig, not the modem.


### 16. The end of a stream is reported as one lost packet
- Status: not fixed yet
- What is wrong: when the sender stops, the receiver logs one more packet as
  lost, with a playback speed near 3x and a timing figure thousands of parts
  per million off. Nothing is shown for it.
- Where: not located yet.
- How it was found: live run over PortAudio, stock stereo.
- What "fixed" looks like: a stream that ends cleanly leaves no lost-packet
  report.
- Looked into: the report is marked `recovered`. The live decode treats the
  audio after a packet's end marker as the start of a packet whose header
  was lost (`animation_modem/v7.py`, `recovered_starts`, lines 4695-4772),
  so after the last packet it tries to decode whatever follows. Read in the
  code; not yet confirmed by a test.
- Notes: under the rule that a packet is header to EOF and nothing else,
  the audio of a decoded packet should be dropped once it is decoded, and
  nothing after an end marker should be taken for a packet without a header
  of its own.
- Report: two changes made. (1) The live decode no longer takes the audio
  after an end marker for a packet whose header was lost, and the offline
  decode no longer does either (`animation_modem/v7.py`: `recovered_starts`
  and `header_recovered` are gone). (2) `LiveInput.decoded()` drops a
  decoded packet's audio: what is kept starts where that packet ended.
  Each decode is now handed about 4,450 samples and 1.5 headers on average
  where it was handed 8,819 samples and 2.3 headers.
- Why "not fixed yet": the lost-packet report at the end of a stream could
  not be reproduced with sender and receiver at the same rate, on either
  build (ten packets then silence; ten packets then a cut-off eleventh).
  It was seen once, on the mismatched test rig. So the cause it was
  attributed to is removed, but there is no before and after to show.
- Tests changed: the two tests that expected a packet with a cut header to
  be recovered now expect it not to be shown and the others to be
  (`modem_tests/test_v7_splice.py`).
- Left over: what should carry across packets (input level, playback
  speed) still does; tail memory and the last verified metadata are
  untouched.


### 17. Double speed at a 48 kHz output
- Status: open (test rig suspected)
- Owner's note: double speed works fine on real hardware. Treat everything
  below as a finding about the test rig until it is reproduced on hardware.
  Not to be worked on ahead of other items.
- What is wrong: three separate things, found on stock mono.
  1. A busy picture (the video test frame) is not shown at all at double
     speed: every packet is marked lost. The face and the slanted edge are
     shown, softer than at normal speed.
  2. The sender lets through a speed the output rate cannot carry whole. The
     packet is limited to about 14 kHz, so at double speed the part from 12
     to 14 kHz lands above 24 kHz. The speed conversion's filter cuts right
     at 24 kHz with a gentle slope (`animation_modem/v7_core.py`,
     `_speed_filter`, line 426), so some of that folds back.
  3. The receiver's status does not follow the picture: packets marked
     received, degraded and lost look alike.
- Where: `animation_modem/v7_core.py`, `speed_resample` and `_speed_filter`
  (lines 426-465); `animation_modem/v7.py`, `max_wire_speed` (line 1029).
  The lost video frame: not located yet.
- How it was found: live run over PortAudio, sender and receiver both at
  48 kHz, three pictures.
- What "fixed" looks like: at any allowed speed the sender writes nothing
  above half the output rate, by leaving out the picture frequencies that do
  not fit instead of filtering through them; and every test picture is shown.
- Notes: earlier runs of this item had the receiver capturing at 44.1 kHz
  while the sender wrote 48 kHz. The blob patterns seen then came from that
  mismatch in the test rig and are not a modem fault. A 16th-order
  Butterworth placed with its corner at the limit made things worse: it had
  not finished cutting when the samples were dropped, and it cut into wanted
  picture frequencies.

### 19. The rotating tail modes are still in the codec
- Status: open
- What is wrong: since item 9 the stereo profile always uses the fixed tail,
  but the codec still holds the rotating tails (chroma, split, luma), their
  tail memory and the seven per-slice tail layouts, unreachable from the
  sender and receiver.
- Where: `test_modem_v7/aspect_fold.py` (`TAIL_MODES`, `FROZEN_TAIL_MODES`
  and their pinned tables), the tail memory in `animation_modem/v7.py`, and
  the tests that exercise every tail.
- How it was found: item 9.
- What "fixed" looks like: only the fixed tail remains in the codec; nothing
  else changes on the wire.
- Notes: the slice number itself stays on the wire. It was listed here as
  dead, but it is in use on purpose: the nested fold's dither picks its
  offset set by it (commit "Dither: folded slots' stairs..."), and slices 5
  and 6 mark the packets that carry the loop fields (item 20).

### 20. Packets on slices 5 and 6 pass a damaged metadata check
- Status: open
- What is wrong: on those two slices the check value is sent mixed with a
  loop field, and the receiver accepts whatever difference it sees
  consistently. When the same metadata bits are damaged in every packet, the
  damage is taken for a loop field and the packet is reported as received.
- Where: `animation_modem/v7.py`, `_tail_slice_mask` (line 308),
  `metadata_word` (line 347), `LoopLock` (around lines 505-525) and the
  check around line 631.
- How it was found: live run at double speed (only slices 5 and 6 were
  accepted, and their pictures were as bad as the rejected ones); mechanism
  read in the code.
- What "fixed" looks like: a packet with damaged metadata is never reported
  as received, on any slice.

### 21. The metadata sits where a band limit removes it
- Status: open
- What is wrong: the metadata symbol's 20 data cells and 11 reference tones
  are spread over the whole band, 1.5 to 12.75 kHz. Anything that cuts the
  top of the band (a speed-up, a codec, a band-limited line) damages it even
  when most of the picture arrives.
- Where: `animation_modem/v7.py`, lines 189-196 (`META_PILOTS`,
  `META_DATA_BINS`).
- How it was found: read in the code; double-speed runs.
- What "fixed" looks like: metadata still checks out on a line that keeps
  only the lower part of the band.

### 22. The receiver cannot be told its capture rate
- Status: open
- What is wrong: the receiver opens the input at the rate the device reports
  as its default and has no option to choose another. In the test rig that
  gave a 44.1 kHz capture of a 48 kHz stream without any notice.
- Where: `tools/v7_live.py`, `capture_rate_for` (line 2195).
- How it was found: live run (the receiver's start-up line).
- What "fixed" looks like: the capture rate can be set, and the start-up
  line makes a mismatch with the device obvious.


### 24. The diagnostic tone speed reading is wrong on coded-pilot wires
- Status: open
- What is wrong: `pilot_tone_speed` assumes a plain tone. The aspect
  profiles flip the tone's sign symbol by symbol, and the reading comes out
  up to 1.5 percent off (60 samples per packet).
- Where: `animation_modem/v7.py`, `pilot_tone_speed`.
- How it was found: live run for item 4; the first tone backup was built on
  it and ended packets 30 to 60 samples off.
- What "fixed" looks like: it agrees with `_packet_tone_scale`, or is
  replaced by it.
- Notes: used only for diagnostics, so nothing shown on screen depends on
  it today.

### 25. Receiver showed nothing more after an input gap
- Status: open
- What is wrong: twice the live receiver reported `input_gap_reacquire`
  (171 blocks dropped) right at the start of a stock mono run, showed one
  picture and then nothing for the rest of the run, although the sender
  kept sending.
- Where: not located. The message comes from the live receive loop in
  `tools/v7_live.py`.
- How it was found: live run, twice, both times the run straight after
  another one; three repeats of the same run on their own were clean.
- What "fixed" looks like: after an input gap the receiver shows pictures
  again within a few packets.
- Notes: may be the test rig (the loopback daemon is unreliable here), like
  items 15 and 17.

### 27. Time-stretched audio: damaged pictures pass, and the bar is in the wrong place
- Status: open (long-term goal, not for now)
- What is wrong: the receiver has no measure of how much of a packet's
  picture is real, so it cannot hold a quality bar. A packet is shown if
  its metadata decodes, however damaged the picture. The design goal is to
  show as much as possible above a threshold, not to show only perfect
  packets.
- Where: not located yet. The time-stretcher search is
  `_spliced_eof_marker` and the splice mapping around it in
  `animation_modem/v7.py`.
- How it was found: live run. The sender's own audio for the reference
  face (58 packets, stock stereo) was time-stretched with ffmpeg `atempo`
  (pitch kept) and played through PortAudio into the live receiver at
  48 kHz, with the search on and off. Packets accepted, search on / off:
  10% slower 20 / 15, 5% slower 42 / 33, 5% faster 37 / 26, 10% faster
  15 / 6, 25% faster 11 / 0 (55 unstretched). Slowed, every extra packet the
  search finds is clean. Sped up, many accepted pictures are mostly noise:
  about half at 10% faster and all at 25% faster with the search on; some
  with it off too.
- What "fixed" looks like: time-stretched audio shows every packet whose
  picture is above an agreed quality bar and none below it.
- Notes: the search stays in: by the design goal it adds real pictures.
  The receiver may already compute a usable damage measure internally; not
  checked.

### 29. The optional pulse-warp timing reads a filtered EOF mark's spacing as speed
- Status: open
- What is wrong: pulse-warp bends the packet's time map using the EOF mark's
  three-edge spacing as the speed at the packet's end. A 300 Hz high-pass
  shifts that spacing (0.5% on the square mark, 1.1% on the shaped one), and
  the warp then reads every packet wrong. Its test now fails and is marked
  as a known failure.
- Where: `animation_modem/v7.py`, `_pulse_warp_anchor_conflict` and the
  `following_scale` taken from the mark; `modem_tests/test_v7_pulse_warp.py`.
- How it was found: test after the EOF was band-limited (item 28).
- What "fixed" looks like: pulse-warp decodes the high-passed stream, or is
  removed (it is not the default timing).

### 32. Lossy codecs: packets found whole but not shown
- Status: open
- What is wrong: through lossy codecs most packets are found and ended
  correctly, but are then not shown. Some fail the metadata check; many pass
  it and are dropped by the receiver's noise limit. Face: MP3 192 41 of 53
  shown, MP3 128 1, AAC 256 37, AAC 128 5, Opus 128 10, Opus 96 4. Test
  frame: only MP3 320 (65 of 69) and AAC 256 (30) show anything.
- Where: the noise limit and the decision to drop a packet are in the
  decoder (`animation_modem/v7.py`, the `noise` diagnostics and `lost`
  status); the metadata layout is item 21.
- How it was found: live run for item 10.
- What "fixed" looks like: to be agreed. By the design goal, a packet whose
  picture is mostly there should be shown; where the bar sits is item 27.

### 33. The first two packets of every run are not shown
- Status: solved (awaiting second pass)
- What is wrong: the receiver waits for three packets in a row to agree on
  the profile before it shows anything, so the first two packets of every
  start (about 160 ms) are never shown, even when the profile is the same as
  last time.
- Where: `tools/v7_live.py`, `_AdaptiveProfileDecoder._observe`
  (`REQUIRED_STREAK`).
- How it was found: second pass over this list; every run showed 68 of 70.
- What "fixed" looks like: with the same sender and mode as last time, the
  first packet is shown.
- Report: the receiver keeps the last profile it confirmed (and the mono
  video side) in `~/.config/modemTest/v7_receiver_profile.json`, one for the
  receiver, and starts on it; its first packet is shown at once. Another
  profile still needs three agreeing packets, and then replaces the saved
  one. Live, 96 kHz, reference face: first run 68 of 70 (nothing saved yet);
  second run 70 of 70 from the first packet; switching the sender to mono,
  the next run confirmed mono after three packets as before and saved it;
  the run after that showed every packet from the first.

### 34. Header speed and header-to-EOF length disagree by up to 90 ppm
- Status: open
- What is wrong: on a clean line the speed read from a packet's header and
  the length measured from header to EOF disagree: median 9-12 ppm, largest
  about 90 ppm at 96 kHz (about 55 at 48 kHz). The receiver reports it as its
  timing error. Left over from item 1.
- Where: the header's scale fit (`animation_modem/transport3.py`,
  `measure_pulses`) against the EOF end (`_measure_eof_marker`).
- How it was found: item 1 and its second pass.
- What "fixed" looks like: on a clean line every packet within about 10 ppm.

## Deleted items

Gone on the second pass and taken off the list. Each line says what fixed it
and how it was checked.

- **1. Live packet start is rounded to a whole sample** — gone (second pass, live, 96 kHz both ends): the fraction of the packet start is kept; clean timing error median 9-12 ppm. What remains (up to about 90 ppm at 96 kHz) is item 34.
- **2. A finished packet is not decoded until the next header arrives** — gone (second pass, live, 96 kHz both ends): a packet is decoded when its own EOF is in; one decode per picture, the last packet of a run is shown.
- **3. Each wake first tries a header that has no EOF yet** — gone (second pass, live, 96 kHz both ends): no wake on a packet that is not complete; decode calls equal pictures shown.
- **4. The next header is used as a packet's end when the EOF is missing** — gone (second pass, live, 96 kHz both ends): a packet with no EOF ends by its own timing tone; every packet shown with every mark blanked. The time-stretcher look-ahead stays for item 27.
- **5. Stock stereo shows no picture on one test chart** — gone (second pass, live, 96 kHz both ends): not reproduced at matched rates; the chart shows in all four modes.
- **6. Command-line defaults do not match the GUI defaults** — gone (second pass, live, 96 kHz both ends): one set of sender defaults; a bare sender and an untouched GUI send sample-identical audio.
- **7. Kernel strength depends on a profile's name** — gone (second pass, live, 96 kHz both ends; now merged): a kernel without settings for a mode is refused; stereo-nested's settings are written down; every mode sends exactly as before.
- **9. The tail mode is set by hand at both ends** — gone (second pass, live, 96 kHz both ends; now merged): aspect-fold-500's tail is part of the profile; no tail setting anywhere; every picture decodes with nothing to match. The rotating tails left in the codec are item 19.
- **10. The end marker is too short to survive lossy audio codecs** — gone (second pass, live, 96 kHz both ends): no change needed; through MP3 and AAC every packet's end is found by its mark. Codec losses are picture, not end: item 32.
- **18. Code for a wire without an end marker is still there** — gone (second pass, live, 96 kHz both ends): the no-end-marker wire is removed; nothing left in the code.
- **23. Something else is taken for the EOF mark when the real one is gone** — gone (second pass, live, 96 kHz both ends): the mark's thresholds follow the header's level; no false marks with every mark blanked.
- **26. The command-line receiver does not draw the way the receiver GUI does** — gone (second pass, live, 96 kHz both ends): the command-line viewer draws as the receiver GUI does (0.37 apart on a 0-255 scale).
- **30. Under soft saturation the left channel's header is read as the wrong profile** — gone (second pass, live, 96 kHz both ends): no wrong-profile reads on the realistic tape cases; the false headers were item 31.
- **31. Things in the picture data were taken for headers** — gone (second pass, live, 96 kHz both ends): a header counts only if its packet ends where its scale says; no packets lost to false headers.
