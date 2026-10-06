# V7 dimension folding: verdict

## Current state: two sender options, one receiver that reads everything

The fold is now wired into the live tools as two experimental sender options.
Requirements they were built to: one table for every link with a gentle
fallback, stereo better than mono on the same link and more redundant, the
stock speed range, and reverse playback. Everything in this section was
measured through the live sender wires and the live receiver's dispatcher
(`tools/v7_nested_eval.py`), not the earlier research read-out. Sections
below this one are the history that led here. Known problems and the
filter still to build are in `V7_NESTED_FOLD_ISSUES.md`; reverse numbers below
that predate it were one-channel readings.

| Sender option (GUI and `--profile`) | Wire it rides on | What changes |
|---|---|---|
| Mono video · nested fold (`aspect-mono-nested`) | `aspect-mono-500` | 972 more luma coefficients folded into the 1,036 luma slots (3:4) |
| Stereo redundant · nested fold (`stereo-nested`) | `stereo-slices` | each channel keeps the stock slot (base ± detail) and folds 420 further coefficients, different on each channel |

The receiver needs no setting: it tells nested packets from stock ones per
packet by the signature slots, and stock packets decode exactly as before.
Layouts without a table stay on the stock mapping. Code:
`test_modem_v7/nested_fold.py`; tables `test_modem_v7/nested_tables.npz`
(14 tables: both wires, seven layouts), rebuilt by `tools/v7_nested_build.py`.

**How "decode what it can" works.** There is one table per layout and no noise
setting. The receiver measures each packet's noise on the 16 signature slots
and decodes every slot by posterior mean at that noise
(`soft_decode_frame`): on a quiet link stairs and guests are read exactly; as
noise rises the guests fade and the hosts slide to a plain shrunk reading.
Nothing is thresholded. The picture's level is one of 15 orthogonal sign
patterns on the signature slots (the stock signature is the sixteenth), a
16-slot correlation that stays readable far past the point where the picture
has faded. This replaced the amplitude-coded level that failed at hiss −40.

**Stereo.** The first design (same hosts on both channels, different guests)
lost to stock slices under noise. The shipped design keeps the stock slices
slot as each channel's host, so when the guests fade what remains *is* the
stock slices picture; crosstalk is undone from the markers at any measurable
share, not only up to the stock bound.

### Update: per-packet auto-level (supersedes the two result tables below)

A level check found header protection intact but nested bodies 4 to 7 dB
quieter than stock on line patterns and charts: the sender scales such
pictures down to fit their large fine-detail values, leaving the body well
under its ceiling. Since line hiss is fixed, that was signal-to-noise thrown
away.

Fix: `v7.body_auto_level()`. The stock encoder already scales a loud body
*down* to a ceiling 1.5 dB under the header; inside this context a quiet body
is also scaled *up* to that ceiling (by at most 12 dB). The pilots are in the
body and rise with it, so the receiver needs nothing. It applies per emitted
packet, and only to nested packets; the stock wire is byte-for-byte unchanged.

Levels after the fix (six pictures, quiet photograph to loud chart): header
−2.6 to −2.7 dB below full scale in every packet, body peak at least 1.26 dB
under it (stock: 1.30), nothing reaches full scale, body RMS −15 to −17 dB
against stock's −15 to −19.

Held-out photographs, auto-levelled (three packets per picture):

| Case | Stock mono | Mono nested | Stock stereo slices | Stereo nested | Stock stereo fold |
|---|---:|---:|---:|---:|---:|
| clean-96k | 1,340 | 1,655 | 1,878 | 2,278 | 2,173 |
| hiss-50 | 1,287 | 1,547 | 1,832 | 2,197 | 2,085 |
| hiss-45 | 1,194 | 1,383 | 1,766 | 2,045 | 1,874 |
| hiss-40 | 1,021 | 1,125 | 1,628 | 1,745 | 1,707 |
| hiss-35 | 886 | 884 | 1,380 | 1,384 | 1,598 |
| one-leg-only | 1,340 | 1,655 | 940 | 1,044 | 537 |
| mono-sum | 1,340 | 1,655 | 974 | 927 | 1,112 |

Torture matrix, auto-levelled (four packets per picture, 36 per cell):

| Case | aspect-fold-500 | aspect-mono-500 | mono nested | stereo-slices | stereo nested |
|---|---:|---:|---:|---:|---:|
| clean-96k | 1,877 (36/36) | 1,136 (36/36) | 1,819 (36/36) | 1,889 (36/36) | 2,629 (36/36) |
| lowpass-18k | 1,878 (36/36) | 1,148 (36/36) | 1,839 (36/36) | 1,892 (36/36) | 2,629 (36/36) |
| lowpass-15k | 1,881 (36/36) | 1,154 (36/36) | 1,839 (36/36) | 1,896 (36/36) | 2,631 (36/36) |
| lowpass-12k | 1,893 (36/36) | 1,163 (36/36) | 1,859 (36/36) | 1,911 (36/36) | 2,624 (36/36) |
| lowpass-10k | 1,920 (36/36) | 1,191 (36/36) | 1,847 (36/36) | 1,918 (36/36) | 2,546 (36/36) |
| hiss-45 | 1,769 (36/36) | 1,032 (36/36) | 1,550 (36/36) | 1,771 (36/36) | 2,497 (36/36) |
| hiss-40 | 1,714 (36/36) | 983 (36/36) | 1,313 (36/36) | 1,610 (36/36) | 2,306 (36/36) |
| hiss-35 | 1,342 (36/36) | 931 (36/36) | 1,002 (36/36) | 1,408 (36/36) | 1,889 (36/36) |
| wow-flutter | 1,776 (36/36) | 978 (36/36) | 1,326 (36/36) | 1,767 (36/36) | 2,497 (36/36) |
| azimuth-12us | 1,876 (36/36) | 1,136 (36/36) | 1,819 (36/36) | 1,889 (36/36) | 2,629 (36/36) |
| crosstalk-10pct | 1,872 (36/36) | 1,136 (36/36) | 1,819 (36/36) | 1,889 (36/36) | 2,629 (36/36) |
| right-minus-4db | 1,877 (36/36) | 1,136 (36/36) | 1,819 (36/36) | 1,889 (36/36) | 2,629 (36/36) |
| dc-hum | 1,835 (36/36) | 1,116 (36/36) | 1,791 (36/36) | 1,862 (36/36) | 2,624 (36/36) |
| bias-leak-30k | 1,878 (36/36) | 1,135 (36/36) | 1,815 (36/36) | 1,889 (36/36) | 2,629 (36/36) |
| soft-saturation | 1,698 (36/36) | 941 (36/36) | 1,303 (36/36) | 1,565 (36/36) | 2,352 (36/36) |
| dropouts | 1,877 (36/36) | 1,136 (36/36) | 1,819 (36/36) | 1,889 (36/36) | 2,629 (36/36) |
| nr-pumping | 1,716 (36/36) | 970 (36/36) | 1,089 (36/36) | 1,490 (36/36) | 2,293 (36/36) |
| type-i | 1,437 (36/36) | 932 (36/36) | 1,011 (36/36) | 1,418 (36/36) | 1,912 (36/36) |
| type-ii | 1,733 (36/36) | 975 (36/36) | 1,279 (36/36) | 1,657 (36/36) | 2,347 (36/36) |
| fast-flutter | 1,724 (36/36) | 959 (36/36) | 1,037 (36/36) | 1,565 (36/36) | 2,266 (36/36) |
| mains-buzz | 1,155 (36/36) | 932 (36/36) | 1,251 (36/36) | 1,357 (36/36) | 2,344 (36/36) |
| highpass-300 | 1,779 (36/36) | 1,044 (36/36) | 1,574 (36/36) | 1,780 (36/36) | 2,591 (36/36) |
| lowpass-4k | 1,170 (34/36) | 649 (35/36) | 693 (34/36) | 1,516 (36/36) | 1,133 (36/36) |
| mono-sum | 1,088 (36/36) | 1,136 (36/36) | 1,819 (36/36) | 944 (36/36) | 919 (36/36) |
| one-leg-only | 256 (36/36) | 1,136 (36/36) | 1,819 (36/36) | 899 (36/36) | 1,242 (36/36) |
| hiss-50 | 1,804 (36/36) | 1,087 (36/36) | 1,717 (36/36) | 1,839 (36/36) | 2,587 (36/36) |
| hiss-30 | 557 (36/36) | 686 (36/36) | 759 (36/36) | 1,145 (36/36) | 1,486 (36/36) |
| azimuth-50us | 2,057 (26/36) | 1,136 (36/36) | 1,819 (36/36) | 1,889 (36/36) | 2,629 (36/36) |
| right-minus-12db | 1,877 (36/36) | 1,136 (36/36) | 1,819 (36/36) | 1,889 (36/36) | 2,629 (36/36) |
| crosstalk-25pct | 1,832 (36/36) | 1,136 (36/36) | 1,819 (36/36) | 1,519 (36/36) | 2,629 (36/36) |
| left-leg-only | 256 (36/36) | 1,136 (36/36) | 1,819 (36/36) | 899 (36/36) | 1,242 (36/36) |

- Mono nested is now at or ahead of stock mono in **all 31 cases** (it was
  behind at hiss −35, hiss −30 and type I).
- Stereo nested is the best of the five modes in 27 of 31 and ahead of stock
  slices in 29. Behind: 4 kHz low-pass (−25%, fewer packets shown there) and
  mono sum (−3%).
- Heavy hiss gained most: stereo nested +20% to +24% at hiss −40 to −30 and
  +22% on type I over the un-levelled result.
- Cost: under soft saturation a hotter body distorts more. Nested lost 6%
  (stereo) and 13% (mono) there against un-levelled, though both remain well
  ahead of stock (+50% and +38%).

### Held-out photographs (12 Kodak; tables fitted on the other 12)

Effective luma coefficients, four packets per picture, every packet shown.

| Case | Stock mono | Mono nested | Stock stereo slices | Stereo nested | Stock stereo fold |
|---|---:|---:|---:|---:|---:|
| clean-96k | 1,340 | 1,655 | 1,878 | 2,278 | 2,173 |
| hiss-50 | 1,287 | 1,500 | 1,832 | 2,176 | 2,082 |
| hiss-45 | 1,194 | 1,302 | 1,767 | 1,992 | 1,866 |
| hiss-40 | 1,023 | 1,023 | 1,630 | 1,653 | 1,707 |
| hiss-35 | 886 | 816 | 1,382 | 1,298 | 1,594 |
| one-leg-only | 1,340 | 1,655 | 940 | 1,044 | 537 |
| mono-sum | 1,340 | 1,655 | 974 | 927 | 1,112 |

- Mono nested against stock mono: +24% clean, +17% hiss −50, +9% hiss −45,
  level at hiss −40, −8% at hiss −35.
- Stereo nested against stock slices: +21% clean, +19% hiss −50, +13% hiss −45,
  +1% hiss −40, −6% hiss −35; one channel alone +11%; mono sum −5%.
- Stereo nested against mono nested on the same link: +38% clean, +45% hiss
  −50, +53% hiss −45, +62% hiss −40, +59% hiss −35.
- These clean-link gains are smaller than the +50% to +59% of the earlier
  clean-tuned tables. That is the price of one table for every link.

### Torture matrix, live path (nine pictures: three photographs, three movie frames, three line patterns)

Six packets per picture; cells are effective luma coefficients and packets
shown out of 54. The last six rows are cases added to the stock 25.

| Case | aspect-fold-500 | aspect-mono-500 | mono nested | stereo-slices | stereo nested |
|---|---:|---:|---:|---:|---:|
| clean-96k | 1,878 (54/54) | 1,136 (54/54) | 1,818 (54/54) | 1,889 (54/54) | 2,629 (54/54) |
| lowpass-18k | 1,878 (54/54) | 1,148 (54/54) | 1,839 (54/54) | 1,892 (54/54) | 2,629 (54/54) |
| lowpass-15k | 1,881 (54/54) | 1,154 (54/54) | 1,841 (54/54) | 1,896 (54/54) | 2,631 (54/54) |
| lowpass-12k | 1,893 (54/54) | 1,163 (54/54) | 1,860 (54/54) | 1,911 (54/54) | 2,624 (54/54) |
| lowpass-10k | 1,920 (54/54) | 1,191 (54/54) | 1,846 (54/54) | 1,918 (54/54) | 2,546 (54/54) |
| hiss-45 | 1,768 (54/54) | 1,030 (54/54) | 1,378 (54/54) | 1,771 (54/54) | 2,330 (54/54) |
| hiss-40 | 1,714 (54/54) | 980 (54/54) | 1,074 (54/54) | 1,612 (54/54) | 1,918 (54/54) |
| hiss-35 | 1,342 (54/54) | 931 (54/54) | 856 (54/54) | 1,410 (54/54) | 1,520 (54/54) |
| wow-flutter | 1,775 (54/54) | 977 (54/54) | 1,330 (54/54) | 1,749 (54/54) | 2,479 (54/54) |
| azimuth-12us | 1,876 (54/54) | 1,136 (54/54) | 1,818 (54/54) | 1,889 (54/54) | 2,629 (54/54) |
| crosstalk-10pct | 1,872 (54/54) | 1,136 (54/54) | 1,818 (54/54) | 1,889 (54/54) | 2,629 (54/54) |
| right-minus-4db | 1,877 (54/54) | 1,136 (54/54) | 1,818 (54/54) | 1,889 (54/54) | 2,629 (54/54) |
| dc-hum | 1,838 (54/54) | 1,116 (54/54) | 1,753 (54/54) | 1,862 (54/54) | 2,621 (54/54) |
| bias-leak-30k | 1,878 (54/54) | 1,134 (54/54) | 1,815 (54/54) | 1,889 (54/54) | 2,629 (54/54) |
| soft-saturation | 1,698 (54/54) | 941 (54/54) | 1,491 (54/54) | 1,568 (54/54) | 2,510 (54/54) |
| dropouts | 1,878 (54/54) | 1,136 (54/54) | 1,818 (54/54) | 1,889 (54/54) | 2,629 (54/54) |
| nr-pumping | 1,716 (54/54) | 969 (54/54) | 1,030 (54/54) | 1,490 (54/54) | 2,259 (54/54) |
| type-i | 1,432 (54/54) | 932 (54/54) | 921 (54/54) | 1,417 (54/54) | 1,583 (54/54) |
| type-ii | 1,729 (54/54) | 975 (54/54) | 1,137 (54/54) | 1,650 (54/54) | 2,060 (54/54) |
| fast-flutter | 1,719 (54/54) | 959 (54/54) | 1,018 (54/54) | 1,556 (54/54) | 2,244 (54/54) |
| mains-buzz | 1,189 (54/54) | 931 (54/54) | 1,020 (54/54) | 1,359 (54/54) | 2,067 (54/54) |
| highpass-300 | 1,774 (54/54) | 1,044 (54/54) | 1,554 (54/54) | 1,785 (54/54) | 2,555 (54/54) |
| lowpass-4k | 1,201 (43/54) | 592 (46/54) | 927 (45/54) | 1,515 (46/54) | 1,396 (45/54) |
| mono-sum | 1,088 (54/54) | 1,136 (54/54) | 1,818 (54/54) | 944 (54/54) | 919 (54/54) |
| one-leg-only | 255 (54/54) | 1,136 (54/54) | 1,818 (54/54) | 899 (54/54) | 1,242 (54/54) |
| hiss-50 | 1,804 (54/54) | 1,083 (54/54) | 1,591 (54/54) | 1,839 (54/54) | 2,515 (54/54) |
| hiss-30 | 557 (54/54) | 686 (54/54) | 671 (54/54) | 1,139 (54/54) | 1,237 (54/54) |
| azimuth-50us | 1,971 (44/54) | 1,136 (54/54) | 1,818 (54/54) | 1,889 (54/54) | 2,629 (54/54) |
| right-minus-12db | 1,877 (54/54) | 1,136 (54/54) | 1,818 (54/54) | 1,889 (54/54) | 2,629 (54/54) |
| crosstalk-25pct | 1,832 (54/54) | 1,136 (54/54) | 1,818 (54/54) | 1,519 (54/54) | 2,629 (re-run after fix, 36/36) |
| left-leg-only | 255 (54/54) | 1,136 (54/54) | 1,818 (54/54) | 899 (54/54) | 1,242 (54/54) |

- **Stereo nested is the best of the five modes in 27 of 31 cases** (and ahead
  of stock slices in 29), including
  hiss down to −30, both tape types, wow and flutter, saturation, NR pumping
  and mains buzz. Behind: 4 kHz low-pass (−8% against stock slices), mono sum
  (−3%), and the one-channel cases, where the mono modes lead by design.
- **Mono nested is ahead of stock mono in 28 of 31**; behind at hiss −35
  (−8%), hiss −30 (−2%) and type I (−1%).
- Nothing collapses: the worst nested cell against its stock wire is −8%.

### Playback speed and reverse

A waveform is generated, resampled or reversed, and decoded as usual (three
pictures, nine packets; stereo modes read one channel here, as reverse does).

| Playback | aspect-fold-500 | aspect-mono-500 | mono nested | stereo-slices | stereo nested |
|---|---:|---:|---:|---:|---:|
| 1x forward | 1,816 (27/27) | 1,011 (27/27) | 2,372 (27/27) | 914 (27/27) | 1,405 (27/27) |
| 1x reverse | 1,780 (18/27) | 1,007 (21/27) | 2,380 (21/27) | 914 (21/27) | 1,405 (21/27) |
| 0.5x forward | 1,816 (27/27) | 1,011 (27/27) | 2,372 (27/27) | 914 (27/27) | 1,405 (27/27) |
| 0.5x reverse | 1,780 (18/27) | 1,011 (18/27) | 2,386 (18/27) | 914 (18/27) | 1,405 (18/27) |
| 1.5x forward | 1,815 (27/27) | 1,009 (27/27) | 2,370 (27/27) | 914 (27/27) | 1,405 (27/27) |
| 1.5x reverse | 1,771 (21/27) | 1,006 (21/27) | 2,393 (21/27) | 914 (21/27) | 1,407 (21/27) |
| 2.0x forward | 1,815 (27/27) | 1,011 (27/27) | 2,377 (27/27) | 914 (27/27) | 1,405 (27/27) |
| 2.0x reverse | 1,780 (21/27) | 1,006 (21/27) | 2,369 (21/27) | 914 (21/27) | 1,405 (21/27) |

Both nested modes hold their forward quality at 0.5×, 1.5× and 2× and when the
waveform is reversed. Reverse shows 18 to 21 of 27 packets for nested and stock
alike: a fresh reverse receiver holds its first packets. Reverse reads one
slices channel; that is the stock wire's limit and applies to stereo nested.

### Cost and limits

- Per packet the nested modes add about 4 ms to encoding and 4 ms to decoding
  (packet interval about 82 ms).
- Tables are fitted on 24 general photographs, not on any one kind of content.
- Synthetic channels only; no tape.
- The stock stereo fold (`aspect-fold-500`) has no nested option: its picture
  is split across both channels, which is the opposite of the redundancy asked
  of stereo.
- In the sandbox two existing tests fail before and after this work:
  a wall-clock sender timing test and one sender-GUI picker test.


## Answer

**Folding more picture coordinates into the unchanged V7 packet does carry more
detail. It does not, and cannot, deliver the requested gain.**

| Claim | Verdict | Evidence |
|---|---|---|
| Folding carries more detail than the current packet | **True** | Adaptive fold, real packets, held-out pictures: 3,574 effective luma coefficients against 2,261 (+58%), a 1.26× linear gain |
| 1.5× linear resolution (2.25× detail) | **Not reached; allowed by Shannon only on a quiet link** | Ceiling 1.74× clean, 1.52× at hiss −45, 1.39× at hiss −40. Best mapping built: 1.26× clean, 1.14× at −45, 1.04× at −40 |
| 3× linear resolution (9× detail) | **Impossible at V7 precision** | Needs about 20,000 effective coefficients; the clean-wire ceiling is 6,814 |
| The earlier negative trials show folding does not work | **False** | They measured a defective mapping (below). The same slots with a correct mapping gain on every condition tested |

"Effective coefficients" is the number of radially ordered luma coefficients a
noise-free truncation would need to reach the same error; linear resolution is
its square root. Baseline is the best existing mapping in each condition.

Nothing here is tape validation: hiss is synthetic white noise before the
production receiver.

## Update: 50% more detail per packet, confirmed on the real wire

Three changes to the repaired fold (`tools/v7_sk_adaptive.py`), tables chosen
on training pictures only (48 layouts per condition), then run once on the
held-out pictures through real packets:

1. **Backward-adaptive guest scale.** Pictures differ widely in fine detail. The
   guest scale now follows the picture: both ends compute it from the decoded
   host stairs (outer half of the hosts, three orientation sectors), so nothing
   extra is sent. The fitted law is close to proportional (slopes 0.86 to 1.05).
2. **Guests chosen by training variance** instead of by radius.
3. **Wider mid-layer thresholds** for the doubly folded slots.

Effective luma coefficients, real wire, held-out (gain over the best existing
mapping in that condition):

| Condition | Best existing | Repaired fold | Adaptive fold | Detail gain | Linear |
|---|---:|---:|---:|---:|---:|
| clean | 2,261 | 3,350 | **3,574** | **+58%** | 1.26× |
| hiss −60 | 2,238 | 3,136 | **3,396** | **+52%** | 1.23× |
| hiss −50 | 2,080 | 2,667 | 2,831 | +36% | 1.17× |
| hiss −45 | 1,860 | 2,336 | 2,423 | +30% | 1.14× |
| hiss −40 | 1,807 | 1,946 | 1,950 | +8% | 1.04× |

Zero packets lost; packet length, RMS and peak match production. Clean, the
packet carries 4,508 luma coefficients against 2,388 (+89% raw), worth 58% more
effective detail after their noise is counted. The 11-of-12 fine-band limit is
38 cycles against 32 for production as shipped (hiss −50: 34 against 28). At
hiss −60 the chosen table drops that limit to 26 although its error is lower.

This is a 50% gain in detail per packet, which is 1.26× per axis. It is not the
1.5× per axis asked for originally; that needs 2.25× the detail and remains
above anything built here. Gratings were not re-run with the adaptive table.

## Repository test videos: a failure found and fixed

Scored on 25 frames from the five fixture movies (`modem_tests/fixtures/*.mp4`,
five frames each, centre-cropped to 3:4), with tables trained on Kodak only.

First result: the adaptive fold was **worse** than production (1,765 against
2,235 effective coefficients, clean). Cause: the sender holds every packet at
one audio level. These charts have slot values about 3.5× the design level, so
the packet is turned down and its noise, counted in slot units, is about 4× the
profile the stairs were built for. Measured on the wire: noise ≈ 1.2 × slot
RMS, never below the measured floor. The slot-level study now models this and
reproduces the failure (1,677 predicted).

Fix (`Leveled` in `tools/v7_sk_adaptive.py`): the sender divides picture detail
by a factor from a fixed ladder (steps of 1.25) so slot values sit at the design
level, and writes the ladder index into the amplitude of the 16 reserved
signature slots (3% to 48% lower, still above the production detection
threshold). The receiver reads it there and multiplies back. This uses existing
reserved slots; it adds no packet resource.

Real wire, levelled adaptive fold against the best existing mapping:

| Condition | Test videos | Gain | Kodak held-out | Gain |
|---|---:|---:|---:|---:|
| clean | 3,253 vs 2,235 | **+46%** | 3,552 vs 2,261 | **+57%** |
| hiss −60 | 3,108 vs 2,201 | +41% | 3,289 vs 2,238 | +47% |
| hiss −50 | 2,620 vs 2,008 | +30% | 2,716 vs 2,080 | +31% |
| hiss −45 | 2,300 vs 1,831 | +26% | 2,371 vs 1,860 | +27% |
| hiss −40 | 1,808 vs 1,814 | 0% | 1,774 vs 1,807 | −2% |

Zero packets lost. On Kodak the levelled fold gives up one to nine points
against the unlevelled one, because the ladder is conservative for pictures
that were already near the design level. The unlevelled fold must not be used:
it fails on loud pictures.

## Fine-line, aliasing and moire pattern suite

`tools/v7_test_patterns.py` writes 27 analytically defined patterns (zone
plates, sine and square sweeps, line-pair blocks, wedges, Siemens stars, rings,
slanted gratings, checkerboards, two-grating beats, hairline hatch, isolated
lines and dots, near-axis stair lines, a slanted edge), each box-filtered from
16 samples per pixel so the files carry no rendering aliases. Frequencies are in
cycles per picture height; `MANIFEST.json` says what each one shows. Layouts
3:4, 4:3, 16:9 and 5:6.

Whole suite through the real wire with the Kodak-trained tables, one noise draw:

| Condition | Best existing | Levelled adaptive fold | Gain |
|---|---:|---:|---:|
| clean | 2,079 | 2,472 | +19% |
| hiss −50 | 1,998 | 2,266 | +13% |

A gain, but much smaller than on photographs or the fixture movies. These
patterns put their energy in a few very large coefficients, which the
natural-picture compander clips. The 11-of-12 band limit is not defined here
because most patterns leave most bands empty.

## Lines and luma: level chosen by trial decode (current best)

The first levelled fold gained only 19% on the line-pattern suite. The
per-pattern breakdown showed why: broadband patterns (zone plates, sweeps, line
blocks, stars) were recovered well, but narrow-band ones (rings, slanted
gratings, beats, wedges) have almost no energy in the hosts, so a level set from
host loudness amplified them and the guests clipped.

Change (`Leveled.level`): the sender tries every ladder level, adds noise at the
level that packet will meet, decodes as the receiver will, and keeps the level
with the least error. The ladder is now 17 levels a factor 1.5 apart, written
as a signature amplitude from 1.6 down to 0.4 of nominal. Tables are unchanged
and still trained on Kodak only; no pattern or movie frame was used for fitting.

Real wire, levelled fold against the best existing mapping in each cell:

| Condition | Line patterns (27) | Fixture movies (25) | Kodak held-out (12) |
|---|---:|---:|---:|
| clean | 3,222 vs 2,079, **+55%** | 3,299 vs 2,235, **+48%** | 3,601 vs 2,261, **+59%** |
| hiss −60 | not run | not run | 3,476 vs 2,238, +55% |
| hiss −50 | 2,636 vs 1,998, +32% | 2,802 vs 2,008, +40% | 2,923 vs 2,080, +41% |
| hiss −45 | not run | not run | 2,467 vs 1,859, +33% |
| hiss −40 | 2,108 vs 1,752, +20% | 2,025 vs 1,815, +12% | **1,535 vs 1,806, −15%** |

Patterns and movies: one noise draw per picture; Kodak: two. Zero packets lost.

Known defect: at hiss −40 on Kodak the levelled fold is worse than plain linear
(the unlevelled fold scored 1,950 there). Not yet diagnosed; the likely causes
are a misread level or a level chosen too low for that noise. Do not use this
version at that noise level.

Tried and rejected for lines: training statistics on a mix that included
patterns (worse on every set), and switching guests to axis-hugging supports
per picture (+0.3%).

## Torture matrix against stock

`tools/v7_sk_torture.py` runs the repository's own 25 impairment cases
(`tools/v7_torture_matrix.py`) on streams of 12 real packets per picture, nine
pictures (three Kodak held-out, three fixture-movie frames, three line
patterns), production receiver. Cells are effective luma coefficients over the
packets that decoded, with packets decoded out of 108. Two frozen fold tables
are shown because a table is tuned to one noise level; nothing selects between
them yet.

| Case | stock as shipped | stock mapping, retrained | fold, clean table | fold, hiss-45 table |
|---|---:|---:|---:|---:|
| clean-96k | 1,878 (108/108) | 2,174 (108/108) | 3,847 (108/108) | 3,127 (108/108) |
| lowpass-18k | 1,879 (108/108) | 2,176 (108/108) | 3,853 (108/108) | 3,127 (108/108) |
| lowpass-15k | 1,881 (108/108) | 2,177 (108/108) | 3,862 (108/108) | 3,128 (108/108) |
| lowpass-12k | 1,892 (108/108) | 2,161 (108/108) | 3,872 (108/108) | 3,132 (108/108) |
| lowpass-10k | 1,919 (108/108) | 2,141 (108/108) | 3,842 (108/108) | 3,127 (108/108) |
| hiss-45 | 1,776 (108/108) | 1,718 (108/108) | 318 (108/108) | 2,896 (108/108) |
| hiss-40 | 1,702 (108/108) | 1,418 (108/108) | 1 (108/108) | 2,103 (108/108) |
| hiss-35 | 1,229 (108/108) | 1,166 (108/108) | 0 (108/108) | 657 (108/108) |
| wow-flutter | 1,779 (108/108) | 1,746 (108/108) | 2,435 (108/108) | 2,891 (108/108) |
| azimuth-12us | 1,871 (108/108) | 2,172 (108/108) | 3,842 (108/108) | 3,127 (108/108) |
| crosstalk-10pct | 1,849 (108/108) | 2,169 (108/108) | 3,851 (108/108) | 3,127 (108/108) |
| right-minus-4db | 1,878 (108/108) | 2,174 (108/108) | 3,847 (108/108) | 3,127 (108/108) |
| dc-hum | 1,832 (108/108) | 2,089 (108/108) | 2,995 (108/108) | 3,119 (108/108) |
| bias-leak-30k | 1,878 (108/108) | 2,177 (108/108) | 3,847 (108/108) | 3,127 (108/108) |
| soft-saturation | 1,718 (108/108) | 1,465 (108/108) | 2,093 (108/108) | 2,954 (108/108) |
| dropouts | 1,908 (97/108) | 2,212 (88/108) | 3,783 (90/108) | 3,127 (90/108) |
| nr-pumping | 1,746 (108/108) | 1,454 (108/108) | 1,184 (108/108) | 2,621 (108/108) |
| type-i | 1,371 (106/108) | 1,274 (100/108) | 1 (96/108) | 1,514 (96/108) |
| type-ii | 1,734 (108/108) | 1,465 (108/108) | 27 (108/108) | 2,498 (108/108) |
| fast-flutter | 1,742 (108/108) | 1,433 (108/108) | 1,126 (108/108) | 2,510 (108/108) |
| mains-buzz | 1,125 (108/108) | 1,359 (108/108) | 15 (108/108) | 1,866 (108/108) |
| highpass-300 | 1,772 (108/108) | 1,764 (108/108) | 2,017 (108/108) | 2,985 (108/108) |
| lowpass-4k | 1,311 (108/108) | 1,370 (108/108) | 818 (107/108) | 1,601 (106/108) |
| mono-sum | 1,084 (108/108) | 0 (108/108) | 0 (108/108) | 0 (108/108) |
| one-leg-only | 186 (108/108) | 86 (108/108) | 0 (108/108) | 0 (108/108) |

Reading it:

- **Hiss −45 table: ahead of both stock columns in 22 of 25 cases**, typically
  +40% to +70%, including wow and flutter, saturation, NR pumping, mains buzz,
  both tape types and the 4 kHz low-pass.
- **Clean table: ahead in 14 of 25**, by about +75% where the link is benign
  (filters to 10 kHz, azimuth, crosstalk, level imbalance, bias leak, dropouts),
  and it collapses wherever noise or distortion rises (hiss, both tape types,
  mains buzz, NR pumping, fast flutter). It must not be used without a
  receiver-side noise check.
- **Losses for both tables:** hiss −35 (657 against 1,229), **mono sum** and
  **one leg only** (nothing usable). In the mono cases the retrained stock
  mapping also reads nothing, so the cause is this tool's read-out at the
  equaliser, which does not implement the production M/S fallback; the fold has
  no mono path at all yet.
- **Delivery:** with dropouts and type I tape the fold read 90 and 96 packets
  where stock showed 97 and 106. Stock displays partly damaged packets; the fold
  read-out does not yet.

One seed per case; the matrix is synthetic and is not tape validation.

## Stereo phase and level differences; mono estimate

Stereo fold with larger channel differences than the stock matrix (six-packet
streams, same nine pictures, real wire; packets decoded out of 54):

| Case | stock as shipped | stock mapping, retrained | fold, clean table | fold, hiss-45 table |
|---|---:|---:|---:|---:|
| clean | 1,878 (54/54) | 2,175 (54/54) | 3,847 (54/54) | 3,127 (54/54) |
| azimuth-12us | 1,872 (54/54) | 2,172 (54/54) | 3,842 (54/54) | 3,127 (54/54) |
| azimuth-25us | 1,868 (54/54) | 2,152 (54/54) | 3,816 (54/54) | 3,127 (54/54) |
| azimuth-50us | 1,846 (45/54) | 2,050 (45/54) | 3,559 (46/54) | 3,018 (46/54) |
| azimuth-100us | 1,719 (45/54) | 1,849 (45/54) | 2,497 (45/54) | 3,072 (45/54) |
| right-minus-4db | 1,878 (54/54) | 2,175 (54/54) | 3,847 (54/54) | 3,127 (54/54) |
| right-minus-8db | 1,878 (54/54) | 2,175 (54/54) | 3,847 (54/54) | 3,127 (54/54) |
| right-minus-12db | 1,878 (54/54) | 2,175 (54/54) | 3,847 (54/54) | 3,127 (54/54) |
| crosstalk-10pct | 1,851 (54/54) | 2,169 (54/54) | 3,851 (54/54) | 3,127 (54/54) |
| crosstalk-25pct | 1,834 (54/54) | 2,114 (54/54) | 3,851 (54/54) | 3,127 (54/54) |
| combined-mild | 1,869 (54/54) | 2,170 (54/54) | 3,845 (54/54) | 3,127 (54/54) |
| combined-hard | 1,832 (54/54) | 2,090 (54/54) | 3,806 (54/54) | 3,127 (54/54) |
| combined-hard+hiss-45 | 1,411 (54/54) | 1,154 (54/54) | 0 (54/54) | 1,529 (54/54) |

The fold keeps its lead through 25 us of azimuth, 12 dB of level imbalance and
25% crosstalk, and through 40 us + 8 dB + 15% together: the production equaliser
absorbs them before the fold is read. At 50 and 100 us nine packets are lost for
stock and fold alike. With hiss -45 added to the hard combination the hiss-45
table is 8% ahead of stock and the clean table reads nothing.

Mono (`aspect-mono-500`: 1,052 luma slots against 1,920) has **not** been run on
the wire. Slot-level estimate only, Kodak held-out, using the stereo noise
profile of the same slots as a stand-in:

| | Stock mono mapping | Mono fold | Stock stereo |
|---|---:|---:|---:|
| clean | 1,368 to 1,384 | 2,106 to 2,246 | 2,248 |
| hiss -45 | 962 to 1,125 | 1,402 to 1,518 | 1,851 |

The range is the same noise as stereo to 3 dB quieter. A mono fold would roughly
reach stock stereo on a clean link (94% to 100%) and about 80% of it at hiss -45.

## Mono profile fold (`aspect-mono-500`), built and run on the real wire

The fold now runs on the mono profile: production mono packets (one leg, mono
pilots, coded status, 1,264 slots), with the 1,052 luma slots carrying the
levelled adaptive fold and the receiver reading the equaliser output the
production mono fold reads. Mono wire noise was measured (median slot SNR
43.5 dB clean) and tables were chosen on training pictures only
(`--profile aspect-mono-500` on `v7_sk_wire.py`, `v7_sk_adaptive.py`,
`v7_sk_torture.py`).

Kodak held-out, effective luma coefficients, zero packets lost:

| Condition | Best stock mono | Mono fold | Gain | Best stock stereo | Mono fold as share of stereo |
|---|---:|---:|---:|---:|---:|
| clean | 1,361 | 2,100 | +54% | 2,261 | 93% |
| hiss −60 | 1,344 | 2,002 | +49% | 2,238 | 89% |
| hiss −50 | 1,255 | 1,647 | +31% | 2,080 | 79% |
| hiss −45 | 1,137 | 1,433 | +26% | 1,859 | 77% |
| hiss −40 | 1,002 | 1,086 | +8% | 1,806 | 60% |

The goal of mono matching current stereo is **not met**: 93% on a clean link,
falling with noise. The slot-level estimate made beforehand (94% to 100% clean)
was slightly optimistic. On the strict 11-of-12 band test the mono fold reads
20 cycles against 24 for stock mono on the clean wire, so its gain is in
overall error, not in that limit.

### Hiss −40 defect: found and fixed

The level was written as an absolute signature amplitude. Under heavy hiss the
receiver's estimates carry a small common gain error, enough to shift the read
level by one step and rescale the whole picture. The level is now written as
the balance between the two halves of the signature slots, which a common gain
error cannot change. Stereo Kodak at hiss −40 went from 1,535 (−15%) to 1,994
(+10% over stock); clean and hiss −45 are unchanged (3,602 and 2,468).

### Mono torture matrix

Same 25 cases and nine pictures as the stereo run. Cells are computed over the
packets that decoded, so a cell with few packets is not comparable with a full
one.

| Case | stock as shipped | stock mapping, retrained | fold, clean table | fold, hiss-45 table |
|---|---:|---:|---:|---:|
| clean-96k | 1,116 (108/108) | 1,363 (108/108) | 2,397 (108/108) | 1,798 (108/108) |
| lowpass-18k | 1,122 (108/108) | 1,377 (108/108) | 2,410 (108/108) | 1,798 (108/108) |
| lowpass-15k | 1,125 (108/108) | 1,373 (108/108) | 2,422 (108/108) | 1,799 (108/108) |
| lowpass-12k | 1,127 (108/108) | 1,335 (108/108) | 2,433 (108/108) | 1,799 (108/108) |
| lowpass-10k | 1,155 (108/108) | 1,350 (108/108) | 2,397 (108/108) | 1,797 (108/108) |
| hiss-45 | 1,673 (82/108) | 1,062 (107/108) | 1 (108/108) | 1,721 (108/108) |
| hiss-40 | 1,630 (59/108) | 1,027 (68/108) | 0 (105/108) | 1,428 (104/108) |
| hiss-35 | 23,117 (4/108) | 1,501 (2/108) | 0 (11/108) | 0 (10/108) |
| wow-flutter | 1,459 (90/108) | 954 (104/108) | 1,407 (108/108) | 1,762 (108/108) |
| azimuth-12us | 1,116 (108/108) | 1,363 (108/108) | 2,397 (108/108) | 1,798 (108/108) |
| crosstalk-10pct | 1,115 (108/108) | 1,362 (108/108) | 2,397 (108/108) | 1,798 (108/108) |
| right-minus-4db | 1,116 (108/108) | 1,363 (108/108) | 2,397 (108/108) | 1,798 (108/108) |
| dc-hum | 1,078 (108/108) | 1,329 (108/108) | 1,964 (108/108) | 1,796 (108/108) |
| bias-leak-30k | 1,114 (108/108) | 1,366 (108/108) | 2,396 (108/108) | 1,798 (108/108) |
| soft-saturation | 771 (65/108) | 168 (73/108) | 978 (108/108) | 1,541 (107/108) |
| dropouts | 1,102 (99/108) | 1,323 (98/108) | 2,333 (99/108) | 1,759 (99/108) |
| nr-pumping | none decoded | 3,862 (3/108) | none decoded | 1,572 (1/108) |
| type-i | 5,361 (27/108) | 655 (29/108) | 0 (63/108) | 50 (51/108) |
| type-ii | 1,633 (59/108) | 743 (77/108) | 0 (106/108) | 1,807 (105/108) |
| fast-flutter | 2,261 (62/108) | 748 (81/108) | 363 (106/108) | 1,319 (106/108) |
| mains-buzz | 161 (4/108) | 1 (4/108) | none decoded | none decoded |
| highpass-300 | 533 (79/108) | 445 (86/108) | 892 (106/108) | 1,652 (106/108) |
| lowpass-4k | 684 (108/108) | 110 (108/108) | 0 (108/108) | 271 (107/108) |
| mono-sum | 1,115 (108/108) | 1,362 (108/108) | 2,397 (108/108) | 1,798 (108/108) |
| one-leg-only | 1,116 (108/108) | 1,363 (108/108) | 2,397 (108/108) | 1,798 (108/108) |

- Benign cases, mono sum and one leg only: the fold is ahead with every packet
  decoded (clean table about +75%, hiss −45 table about +30%).
- Wow and flutter, soft saturation, type II, 300 Hz high-pass, hiss −45: the
  hiss −45 table is ahead and decodes more packets than stock.
- Behind or unusable: type I, 4 kHz low-pass, hiss −35, mains buzz and NR
  pumping (the last two decode almost nothing for stock either). At hiss −40
  and fast flutter stock scores higher on the fewer packets it decodes.

## Why 4× detail is out of reach in the unchanged packet

Asked for 4× detail (9,044 effective coefficients). Checked before building:

| Limit, clean wire | Effective coefficients | Gain |
|---|---:|---:|
| Adaptive fold, measured | 3,574 | 1.58× |
| Every coefficient of the 96×80 lattice delivered perfectly | about 7,000 | 3.1× |
| Perfect code, fixed table | 6,814 | 3.0× |
| Perfect code that also knows each picture's own spectrum, slot power re-divided | 7,850 to 8,700 | 3.5 to 3.85× |
| Requested | 9,044 | 4.0× |

4× needs 32% more channel capacity than the clean packet has even with perfect
coding (17.4 kbit against 13.2 kbit). The adaptive fold delivers what a perfect
code would with 6.8 kbit, so at its efficiency 4× needs roughly 2.5× the slots.

A slot-menu estimate (stairs holding up to four sign- or three-level guests
plus an analogue guest, each slot free to choose) predicts about 1.7× for the
nested-fold family. The loss is not per-slot waste: a doubly folded slot already
uses about 6.2 of its 6.9 bits. It is allocation. The ideal code spreads those
bits over some 15,000 coefficients at 0 to 6 dB each; a scalar stair can hold
only a few guests before its guards consume the range.

Routes that can reach 4×: more slots per picture, building held pictures up over
about three packets, or a receiver noise floor 13 dB or more below today's
41 dB on links quieter than tape.

## What V7 precision actually is

Measured on real stereo `aspect-fold-500` packets through the production
receiver, at the equaliser output the production fold reads
(`tools/v7_sk_wire.py measure`, twelve training pictures):

| Condition | Wire noise (signal ≈ 1) | Median slot SNR | Slot range | Luma capacity |
|---|---:|---:|---:|---:|
| clean | 0.0081 | 41.2 dB | 30.7 to 54.1 dB | 13.2 kbit/packet |
| hiss −60 dBFS | 0.0109 | 37.1 dB | 30.0 to 49.7 dB | 12.0 kbit |
| hiss −50 dBFS | 0.0244 | 29.2 dB | 26.6 to 41.7 dB | 9.7 kbit |
| hiss −45 dBFS | 0.0417 | 24.6 dB | 22.1 to 37.0 dB | 8.2 kbit |
| hiss −40 dBFS | 0.0735 | 19.7 dB | 17.2 to 32.1 dB | 6.6 kbit |

The clean wire is not infinitely precise. The receiver's own timing,
band-limiting and equalisation leave a floor near 41 dB, with a stable per-slot
profile (split-half correlation 0.99). 16-bit quantisation adds nothing
measurable. Every design below uses this measured per-slot profile.

## The ceiling

A packet's luma slots are 1,920 parallel channels of known SNR, so their
capacity is fixed. Reverse water-filling over the scored pictures' own spectrum
(288×240 reference lattice) gives the least error any mapping could reach with
that many bits, and therefore the most effective coefficients:

| Condition | Best existing | Ceiling | Ceiling as linear gain |
|---|---:|---:|---:|
| clean | 2,261 | 6,814 | 1.74× |
| hiss −60 | 2,238 | 6,220 | 1.67× |
| hiss −50 | 2,080 | 5,022 | 1.55× |
| hiss −45 | 1,860 | 4,299 | 1.52× |
| hiss −40 | 1,807 | 3,469 | 1.39× |

Re-dividing slot power optimally moves the clean ceiling to 1.79×. Perfect use
of the coefficients' measured non-Gaussian marginals (entropy power 0.95 of
Gaussian) moves it to 1.77×.

For 3×, a weaker test also fails: spending the whole clean packet to bring
every coefficient only to the pass threshold (correlation 0.8, 0.74 bit each)
reaches at most 17,895 coefficients; 3× needs 21,492.

Limits of this bound: it treats coefficients as independent with fixed
variances, which is what every fixed-table mapping here assumes. A coder that
exploited dependence between coefficients inside one picture, with no side
information, is not covered. The 3× conclusion has enough margin that this does
not rescue it; the 1.5× conclusion is "permitted on a quiet link, far from any
mapping we can build", not "disproved".

## What was wrong with the earlier surface

`tools/v7_fold_surface.py`, measured on the same slots and pictures:

| Defect | Effect |
|---|---|
| The new coordinate was the coarse strip index; the existing guest was squeezed into the strip | Each layer costs already-carried detail 20·log10(strips) dB |
| Strips decode to compressed-domain midpoints | A 3-strip coordinate decodes to 0 below 2.1σ and to 5.8σ above it |
| Strip width unrelated to slot noise, no guard band | Layers are too coarse when clean and slip under hiss |
| Nested inside the 500 production guests, which are already the fine half of a fold | The 1,420 unfolded luma slots, where the capacity is, were never used |

Real-wire result: 3→2 with three strips is 0.99× clean and 0.21× at hiss −40.
Both defects are pinned by `EarlierSurfaceDiagnosisTests`.

The production fold itself is tuned for a noisy link (step of one host standard
deviation). On a clean wire that step is far coarser than the noise requires.

## The repaired mapping

`tools/v7_sk_fold.py`. Each luma slot carries its host coefficient on a uniform
staircase whose step is `kappa` times that slot's measured noise, with one or
two further coefficients inside the stair:

```
s = step*round(a/step) + alpha*step*c(b1)                      one guest
s = step*round(a/step) + alpha*step*(m*delta + alpha2*delta*c(b2))   two guests
```

`alpha = 1/2 - guard/kappa` keeps a guard band; `c` is a compander fitted to
training guests; every reconstruction point is a conditional mean fitted on
training pictures. One frozen table at both ends, no per-frame side
information, frame-independent. Header, pilots, metadata, EOF, chroma slots,
slot gains, packet length and the 16 signature slots are unchanged; candidates
are turned down to the production packet's RMS and peak (gain 0.99 to 1.00).

Tables come from one training-only search of 4,203 layouts per condition
(`tools/v7_sk_study.py`), frozen before the held-out wire run.

## Real-wire confirmation

`tools/v7_sk_wire.py confirm`: twelve held-out pictures, three noise draws,
fresh receiver per packet, zero packets lost in any cell. Slot-level
predictions and wire results agree within 1% (clean: 3,341 predicted, 3,350
measured).

Effective luma coefficients (linear gain over the best existing mapping):

| Condition | Linear 1,920 | Production fold, retrained | Production as shipped | Earlier 3→2 surface | Repaired fold |
|---|---:|---:|---:|---:|---:|
| clean | 1,887 | 2,261 | 2,166 | 2,212 | **3,350 (1.22×)** |
| hiss −60 | 1,887 | 2,238 | 2,154 | 2,103 | **3,136 (1.18×)** |
| hiss −50 | 1,879 | 2,080 | 2,051 | 1,090 | **2,667 (1.13×)** |
| hiss −45 | 1,860 | 1,828 | 1,854 | 293 | **2,336 (1.12×)** |
| hiss −40 | 1,807 | 1,596 | 1,700 | 80 | **1,946 (1.04×)** |

Winning tables: clean, every slot folded, 700 of them twice (4,508 luma
coefficients carried); hiss −45, 208 slots left linear and the rest folded once;
hiss −40, 1,000 left linear.

Highest frequency at which 11 of 12 pictures pass the frozen fine-band test
(correlation ≥ 0.8, transfer 0.5 to 1.5, residual ≤ 0.5), cycles per picture
height:

| Condition | Linear | Production as shipped | Repaired, least error | Repaired, resolution pick |
|---|---:|---:|---:|---:|
| clean | 28 | 32 | 36 | 40 |
| hiss −60 | 28 | 32 | 36 | 40 |
| hiss −50 | 28 | 28 | 32 | 20 |
| hiss −45 | 28 | 28 | 28 | 18 |
| hiss −40 | 28 | 28 | 20 | 20 |

Clean, that is 1.25× over production as shipped. The resolution pick does not
generalise once hiss is added, and at hiss −40 folding every table lowers this
limit below plain linear: guests are being bought with host precision the link
no longer has.

## Where it does not help

`tools/v7_sk_wire.py gratings`: single sinusoids, 8 to 46 cycles, four phases,
both axes, contrasts 0.05 and 0.3, real wire.

| Clean wire limit (cycles) | Contrast 0.05 | Contrast 0.3 |
|---|---:|---:|
| Linear 1,920 | 26 | 26 |
| Production fold, retrained | 30 | 28 |
| Repaired fold, least error | 26 | 26 |
| Repaired fold, resolution pick | 28 | 26 |

No gain, and slightly worse than production. Strong gratings beyond the linear
zone exceed the compander's range, which is fitted to natural pictures; weak
ones sit under the noise that thousands of folded guests add. At hiss −50 every
folded mapping, production included, fails the 0.05-contrast probe outright.
The gain is real for natural-picture detail and absent for isolated periodic
patterns.

## Open

- Faces. The repository's face corpus (`images_sbs/`) is not in the checkout;
  statistics and scores use the 24 Kodak pictures (odd numbers train, even
  score). Re-run all three commands on faces before any production decision.
- Mono (`aspect-mono-500`) and layouts other than 3:4 were not run.
- Chroma slots were left alone.
- A fold table is tuned to one noise level. A deployable version needs a
  receiver-measured choice among a few frozen tables, as the production fold's
  noise gates do today.
- Tape.

## Reproduce

```bash
.venv/bin/python -m unittest modem_tests.test_v7_sk_fold -v
.venv/bin/python tools/v7_sk_wire.py measure --corpus tmp/sk/corpus --out tmp/sk/wire
.venv/bin/python tools/v7_sk_study.py --corpus tmp/sk/corpus --noise tmp/sk/wire/NOISE.npz --out tmp/sk/study
.venv/bin/python tools/v7_sk_adaptive.py --noise tmp/sk/wire/NOISE.npz --out tmp/sk/study/ADAPTIVE.json
.venv/bin/python tools/v7_sk_wire.py confirm --study tmp/sk/study/STUDY.json --out tmp/sk/wire
.venv/bin/python tools/v7_sk_wire.py gratings --conditions clean hiss-50 --out tmp/sk/wire
.venv/bin/python tools/v7_sk_figures.py --out tmp/sk/figures
```

The corpus folder holds the pictures (PNG or JPG); alternate files train and
score. Numerical kernels are Numba-compiled. The earlier surface is scored by
calling its own NumPy `surface_encode`/`surface_decode` unchanged.
