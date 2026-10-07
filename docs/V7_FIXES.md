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
10. Sender and receiver run at the same sample rate unless the test is about
    a rate difference. Confirm it from the receiver's own start-up line
    before trusting a result.
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

### 1. Live packet start is rounded to a whole sample
- Status: solved (awaiting second pass)
- What is wrong: the live decode restarts from the whole-number part of the
  measured packet start, so each packet reads up to one sample too long and
  its picture data is stretched to fit. Offline decode keeps the fraction.
  It gets worse with playback speed (one sample is about 255 parts per
  million at 1x and about 1,020 at 4x).
- Where: `animation_modem/v7.py`, lines 4777-4778 (`cursor = int(fs)`,
  `measured = (16*sc, sc, conf)`).
- How it was found: live run. The receiver's reported timing error matched
  the dropped fraction exactly; on a digital copy it showed as an 8-packet
  cycle.
- What "fixed" looks like: on a clean digital copy every live packet reports
  under about 10 parts per million, and successive pictures of a still agree.
- Notes: a wider audit of whole-number conversions of positions found only
  search-window edges (harmless) and the live input's arrival bookkeeping
  (harmless, re-measured). Not every line of the receiver has been read.
- Report: the fraction is kept (`animation_modem/v7.py`, the live restart
  in `_decode_pulse_stream`). Live run over a PortAudio loopback, sender at
  48 kHz, receiver at the device's 44.1 kHz, stock mono, one chart: before,
  69 of 137 pictures read more than 100 parts per million off (up to 232);
  after, none of 137 (largest 54). Same result on stock stereo, both nested
  modes and at half speed. Pictures of a held still differ less from one
  another (stock mono 4.5 to 2.6 and 3.5 to 1.8 on a 0-255 scale; mono
  nested 1.65 to 0.99).
- Left over: the 35 to 55 parts per million that remain are the difference
  between the header's own scale reading and the header-to-EOF length, not
  an error in the length. Not looked at further.


### 2. A finished packet is not decoded until the next header arrives
- Status: solved (awaiting second pass)
- What is wrong: the live input hands audio to the decoder only when a new
  header has arrived. A packet that is complete, header to EOF, waits for
  the following packet, and the last packet of a finite input is never
  decoded. The rule should be: header and EOF both in, decode, at any
  playback speed, without looking at the next packet.
- Where: `animation_modem/v7_live_input.py`, line 210 onward (the wake
  rule); `animation_modem/v7.py`, lines 4705-4776 (live candidate
  selection); `tools/v7_live.py`, line 4064 onward (the live decode call).
- How it was found: read in the code and the spec; seen in the live log.
- What "fixed" looks like: a single packet followed by silence is decoded
  and shown; a stream's last packet is shown.
- Notes: highest priority after anything that is breaking work.
- Report: `LiveInput` now hands a packet over when its own end marker is
  found and its last samples are in (`animation_modem/v7_live_input.py`,
  `_count_complete` and `_marker_in`); `tools/v7_live.py` passes the frame
  boundary through. Live run: ten packets followed by silence, played into
  the loopback. Before, the receiver showed packets 2 to 8; after, 2 to 9
  (the first two go to profile detection in both). Six live runs of 12
  seconds: one decode per picture, none missing, none repeated, last packet
  shown. `modem_tests`: 954 tests, the same two failures as the untouched
  build (a missing package and one test that fails now and then on both).
- Tests changed: three tests pinned the old rule ("every frame but the
  last") and now expect every packet, the last included
  (`modem_tests/test_v7_live_input.py`, `modem_tests/test_v7_reverse.py`).
- Left over: a wire with no end marker (frame boundary `baseline`) still
  ends a packet at the next header, since it has nothing else. If that wire
  is retired, the `frame_boundary` argument of `LiveInput` and its branch go
  with it. A packet whose marker never appears is handed over at the far
  edge of the marker search; the decoder then falls back to the next-header
  endpoint (item 4).


### 3. Each wake first tries a header that has no EOF yet
- Status: solved (awaiting second pass)
- What is wrong: on waking, the decoder checks candidates newest first, so
  it always tries the packet that has only just started, fails, and falls
  back to the previous one.
- Where: `animation_modem/v7.py`, lines 4751-4767.
- How it was found: live log (two marker searches per wake, the first always
  empty).
- What "fixed" looks like: one marker search per decoded packet.
- Notes: expected to disappear with item 2.
- Report: gone with item 2. In six live runs every wake decoded the newest
  header's packet at the first attempt (one decode call per picture, no
  empty calls).


### 4. The next header is used as a packet's end when the EOF is missing
- Status: open
- What is wrong: the backup for a lost or badly fitting EOF reads past the
  packet into the next one. The backup should come from inside the packet:
  the pitch of the timing tones gives the playback speed, the speed gives
  the packet length, and header position plus length gives the end.
- Where: `animation_modem/v7.py`, `_spliced_eof_marker` at line 4243, its
  calls at 4756 and 4813, and the alternates at 4966 and 5055;
  `pilot_tone_speed` at line 1864 already reads speed from the tones and is
  used only for diagnostics.
- How it was found: read in the code and the spec. Not tested.
- What "fixed" looks like: a packet with its EOF removed is still decoded,
  from its own header and tones, with nothing after it in the input.

### 5. Stock stereo shows no picture on one test chart
- Status: open
- What is wrong: with the vertical line-widths chart, the live receiver
  decoded nothing for `aspect-fold-500` in two attempts. Its profile
  detector reported a different profile.
- Where: not located yet. The profile detector is `_ProfileStatusProbe`,
  `tools/v7_live.py`, line 2308.
- How it was found: live run, one picture only.
- What "fixed" looks like: every chart in the test set is shown in every
  mode on a clean signal.

### 6. Command-line defaults do not match the GUI defaults
- Status: open
- What is wrong: the GUI turns on Direct DCT encode, luma adjustment and the
  tuned kernel; a bare command line turns on none of them and falls back to
  a different profile. The GUI's defaults should be the defaults everywhere,
  with arguments as overrides. Direct DCT should be used for everything
  except pixel-exact mode.
- Where: `tools/v7_send_gui.py`, lines 1788-1790 (GUI defaults) and line 60
  (GUI default profile); `tools/v7_live.py`, lines 4512, 4552 and 4618
  (on-only switches and the profile argument).
- How it was found: live run and read in the code.
- What "fixed" looks like: the sender started with no arguments and the
  sender started from an untouched GUI hand bit-identical packets to the
  modem.
- Notes: needs "off" switches, since today's switches can only turn things
  on. One definition of the defaults, used by both.

### 7. Kernel strength depends on a profile's name
- Status: open
- What is wrong: each kernel holds tuned settings by profile name. A profile
  that is not listed silently gets the kernel's raw, full-strength setting.
  Two builds of the same mode ended up at different strengths that way.
- Where: `dct_kernels/viewer_solve.py`, line 27; `dct_kernels/
  upscale_precomp.py`, line 18; `tools/v7_send_gui.py`, lines 58 and
  322-337; `tools/v7_live.py`, lines 1123 and 1174.
- How it was found: read in the code; confirmed by the live sender's own
  start-up report of the kernel in use.
- What "fixed" looks like: every selectable mode states its kernel and
  strength explicitly, and a mode without one refuses to start.
- Notes: the existing profile settings were tuned by hand and stay. For
  testing, the `reference` kernel (plain projection, no shaping) is the
  common baseline; each mode's tuned setting is a second run.

### 8. The default kernel corrects for a viewer that is not the one in use
- Status: open
- What is wrong: `viewer_solve` models the receiver as shrinking to half the
  grid and enlarging bilinearly by 4. The receiver enlarges in frequency
  space, runs the edge rebuild, then draws bicubic.
- Where: `dct_kernels/viewer_solve.py`, lines 21 and 49;
  `tools/v7_gl_viewer.py`, lines 103-123 (display defaults).
- How it was found: read in the code. Effect on the picture not measured.
- What "fixed" looks like: the sender's model of the viewer is taken from
  the same description the receiver draws with.
- Notes: lowest priority for now. Possible link to the squares and stairs
  seen earlier; to be settled on the charts by switching the kernel and the
  edge rebuild off in turn.

### 9. The tail mode is set by hand at both ends
- Status: open
- What is wrong: what the 96 tail slots carry is not signalled and is a
  separate setting on sender and receiver that must be matched by hand. It
  should be part of the profile.
- Where: `test_modem_v7/aspect_fold.py`, lines 27-28 and 72-76;
  `tools/v7_live.py`, lines 4463-4467; the matching settings in
  `tools/v7_send_gui.py` and `tools/v7_receiver_gui.py`.
- How it was found: read in the code and the spec.
- What "fixed" looks like: choosing a profile fixes the tail; there is no
  separate tail setting to match.
- Notes: the fixed colour tail stays as the default. Only the mid/side
  stereo wire has tail slots; mono wires and stereo modes built from two
  mono packets have none.

### 10. The end marker is too short to survive lossy audio codecs
- Status: open
- What is wrong: the EOF marker is 24 samples, half a millisecond, a single
  short burst. Lossy codecs smear and reshape a burst that short.
- Where: `animation_modem/v7.py`, lines 49-51 (`PULSE_GUARD_BASE`,
  `EOF_MARKER_LENGTH`) and the marker search at `_eof_marker_kernel`.
- How it was found: tool only (frames lost through MP3 and AAC). The length
  needed, about 4 milliseconds of an in-band tone, is an estimate, not a
  measurement.
- What "fixed" looks like: all packets shown after a live run through MP3
  and AAC at agreed bit rates.
- Notes: lengthening to about 200 samples costs about 4.5% of picture rate.
  The header is 288 samples; whether its shape suits a codec has not been
  looked at. Changing the packet length touches sender, receiver, reverse
  playback and tests.

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

### 18. Code for a wire without an end marker is still there
- Status: solved (awaiting second pass)
- What is wrong: the end marker is a core part of V7 and nothing is used
  without it. The code still carries the other way of ending a packet (the
  next header, frame boundary `baseline`): in the decoder, as an argument of
  `LiveInput` kept when item 2 was fixed, as a receiver option, and in tests
  that send a wire with no marker.
- Where: `animation_modem/v7.py` (`frame_boundary`, the `baseline` branches
  around lines 4713-4722 and 4835-4860); `animation_modem/v7_live_input.py`
  (`frame_boundary`); `tools/v7_live.py` (the `frame_boundary` option);
  `modem_tests/test_v7_live_input.py` and others that use the wire without
  a marker.
- How it was found: owner's decision.
- What "fixed" looks like: one way to end a packet. No `baseline` option,
  branch or test.
- Report: there is one way to end a packet. Removed: the `eof_marker`
  argument of the packet encoders and of every wrapper around them (they
  always write the marker); the `frame_boundary` argument of the decoders
  and its next-header branches; the `frame_boundary` argument of
  `LiveInput`; the receiver's `--frame-boundary` option and its GUI entry;
  the application sender's `--modem-eof-marker` option. 43 files had the
  now-redundant arguments stripped from calls.
- Checked: `modem_tests`, 953 tests, the only failure is the missing
  package the untouched build also fails on. Live run at 48 kHz both ends,
  all four modes: every packet shown, none repeated.
- Tests changed: tests that sent a wire with no marker now send the only
  wire there is; the ones that expected "every frame but the last" expect
  every frame; the two pilot-tone tests take the tone overlay from the
  function that makes it, because a packet's final level is set after the
  tones are mixed in; two level tests measure the body against the ceiling
  the packet's own header and end marker set; one test that only refused a
  missing marker was deleted.
- Two tests pinned behaviour that existed only on the old wire and now
  assert less: `test_timing_aided_retry_recovers_metadata_in_previous_
  lowpass_case` (the plain decode already reads 39 of 40 there, so the
  retry has nothing to add) and `test_static_filter_bias_falls_back_to_
  packet_average_scale` (with the end pinned by the marker the fall-back
  no longer happens). Worth a look in the second pass.
- Left over: `docs/LIVE_MODEM.md` and the dirty spec still describe the
  removed options.


### 19. The tail slice number is still counted, sent and used
- Status: open
- What is wrong: with the fixed tail nothing rotates, but every packet still
  carries a slice number from 0 to 6 in three metadata bits, on every wire,
  mono included. The decoder picks a slot layout by it, tail memory is
  updated by it, the nested fold's optional dither uses it as its counter,
  and predicted metadata advances it.
- Where: `animation_modem/v7.py`, `metadata_word` (line 318), the layout
  lookup at line 569, lines 5154-5177 and 5257-5260;
  `test_modem_v7/nested_fold.py`, line 502 onward;
  `test_modem_v7/aspect_fold.py`, lines 665-668.
- How it was found: read in the code; seen in the double-speed runs.
- What "fixed" looks like: no slice number on the wire; the rotating tail
  modes, tail memory and the seven layouts are gone; the bits are free.
- Notes: two slice values (5 and 6) also mark the packets that carry the
  loop fields (item 20), so the loop fields need another way to be marked or
  carried first. Changes the metadata layout, so it changes the wire. Ties
  into item 9.

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
