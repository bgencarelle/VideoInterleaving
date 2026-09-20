# Modem transport state machines

This attachment expands the stateful behavior referenced by
[`transport_timing_and_encoding.md`](transport_timing_and_encoding.md). Arrows
are transitions; bracketed text is a guard or side effect. `UNVERIFIED` means
the cited implementation does not expose a stronger guarantee.

The current V3-family encoder is a synchronous packet builder. The current
receiver is incremental, chunk-independent, and pulse-acquisition driven.
`[C:animation_modem/transport3.py:482-550]`, `[C:animation_modem/transport3.py:583-975]`

## V1 — encode

```text
SOURCE
  -> VALIDATE_FIELDS [absolute/index/count/profile valid]
  -> PREPARE_IMAGE [RGB aspect-pad, Lanczos]
  -> YCBCR_SAMPLE [luma + optional BOX chroma, values in [-1,1]]
  -> PACK_HEADER [SI01/SI02 or ST; append CRC32]
  -> FILL_TRAINING_HEADER_IMAGE [19 OFDM symbols, pilots]
  -> IFFT_CP [128 useful + 16 CP per symbol]
  -> FRAME [sync in channel 0, body at sample 288, zero tail to 3200]
  -> PCM_OUTPUT
```

`VALIDATE_FIELDS` through `FRAME` are `[H364:animation_modem/transport.py:66-136]`.
The historical V1 packet/frame constants are `[H364:animation_modem/transport.py:12-29]`.

## V1 — decode

```text
INPUT_BUFFER
  -> CORRELATE_SYNC [threshold .45]
  -> LOCK_PACKET [fixed 3200-frame spacing]
  -> TRACK_RATE [after 3 plausible packet intervals]
  -> RESAMPLE_IF_NEEDED [8-tap windowed sinc]
  -> FFT_AND_PROFILE_FIT [color, detail, mono insertion order]
  -> EQUALIZE_AND_PILOT_FIT
  -> HEADER_CANDIDATES [stereo mean, channel 0, channel 1]
      -> CRC_AND_STRUCTURE_OK -> RECONSTRUCT -> VERIFIED_HEADER
      -> CRC_FAIL + salvage thresholds -> RECONSTRUCT -> PARTIAL_HEADER_UNKNOWN
      -> otherwise -> CORRUPT_HEADER
  -> OUTPUT / REACQUIRE
```

The acquisition loop, spacing tracker, missing-frame estimates, and rate
resampling are `[H99:animation_modem/transport.py:352-469]`; profile fitting,
header candidates, CRC, and salvage guards are `[H99:animation_modem/transport.py:199-290]`.

## V2 — encode (`99d60665`)

```text
SOURCE_VALUES
  -> VALIDATE_COUNT
  -> DCT_FORWARD_AND_GAIN [allocation table -> normalized gains]
  -> COEFFICIENT_SLOT_MAP [progressive layouts only]
  -> PACK_V2_HEADER [V2 or progressive V3 magic; CRC32]
  -> FILL_TRAINING_HEADER_IMAGE [layout-specific preset]
  -> IFFT_CP
  -> FRAME [sync/body/32-sample guard]
  -> PCM_OUTPUT
```

The encoder transitions are `[H99:animation_modem/transport2.py:167-229]` and
`[H99:animation_modem/transport2.py:286-342]`.

## V2 — decode (`99d60665`)

```text
INPUT_BUFFER
  -> COARSE_SYNC_BANK [29 scaled templates, .5x..2x]
  -> TIMING_FIT [threshold .4; coarse candidates require max(.4,.55)]
  -> WAIT_FOR_COMPLETE_PACKET
  -> SINC_RESAMPLE_IF_OFF_SPEED
  -> FFT_AND_TRAINING_EQUALIZER
  -> PER-SYMBOL_PILOT_PHASE_FIT
  -> HEADER_STEREO_MEAN
      -> CRC/structure pass -> PAYLOAD_WIENER -> VERIFIED_HEADER
      -> mean fail -> CHANNEL_0_HEADER
          -> pass -> PAYLOAD_WIENER -> VERIFIED_HEADER
      -> channel 0 fail -> CHANNEL_1_HEADER
          -> pass -> PAYLOAD_WIENER -> VERIFIED_HEADER
      -> all fail + coherence/coverage guard -> PICTURE_ONLY
      -> otherwise -> LOST
  -> DROP_PACKET / SEARCH_NEXT
```

The historical V2 acquisition and packet loop are
`[H99:animation_modem/transport2.py:612-796]`; equalization, header fallback,
CRC, and picture-only guards are `[H99:animation_modem/transport2.py:388-450]`.

## V3 `wire` — encode/decode

### Encode

```text
SOURCE_IMAGE
  -> PREPARE_80x96_AND_ASPECT
  -> YCBCR_SAMPLE
  -> PROFILE_TRANSFORM [DCT code 0 / Haar code 1]
  -> RANK_AND_SLOT [dense-header + spread-carrier order]
  -> PACK_HEADER [V4 magic, top-bin bits, profile bits, aspect bits, CRC]
  -> TRAINING/PILOTS/OFDM
  -> OPTIONAL_RATE_ADAPT
  -> OUTPUT_FRAME [3168 packet + 32 guard = 3200]
```

### Decode

```text
CAPTURE_BUFFER
  -> PULSE_ACQUIRE
      -> COAST_IF_LOCKED
      -> EDGE_SEARCH_IF_COLD
      -> CORRELATION_FALLBACK
  -> SIZE_PACKET
  -> FFT_AND_2x2_MMSE_EQUALIZE
  -> PILOT_PHASE_TRACK
  -> OPTIONAL_DRIFT_REFIT
  -> HEADER_CRC_AND_PROFILE_SELECT
      -> verified -> PAYLOAD_INVERSE -> RECEIVED/DEGRADED
      -> unverified but usable -> PAYLOAD_INVERSE -> PICTURE_ONLY
      -> unusable -> SINGLE_INPUT_RETRIES or LOST
  -> PREDICT_NEXT / RELOCK_AFTER_4_FAILURES
```

Encode is `[C:animation_modem/transport3.py:482-550]`; acquisition and packet
states are `[C:animation_modem/transport3.py:664-975]`; equalization, drift,
header, retries, and fallback are `[C:animation_modem/core.py:883-1207]`,
`[C:animation_modem/core.py:1218-1554]`.

## `wire-hd` — encode/decode

`wire-hd` uses the same state graph as V3 `wire`, with these named states:

```text
ENCODE: SOURCE_80x96 -> CDF97_PYRAMID -> COPY_800_LL/DETAIL
       -> WIRE_HD_SLOTS -> V3_OFDM_FRAME -> OUTPUT

DECODE: PULSE_ACQUIRE -> WIRE_HD_LENGTH(3488)
       -> V3_EQUALIZE -> PROFILE_2(hd-dwt)
       -> CDF97_SOFT_INVERSE_AND_RECOVERY_GATE -> OUTPUT
```

The layout is `[C:animation_modem/transport3.py:70-76]`; the coder and recovery
states are `[C:animation_modem/wavelet.py:705-833]`; the engine selection is
`[C:animation_modem/engines.py:34-45]`.

## V5 active `hd-dwt` — encode/decode

```text
ENCODE: V5_ENGINE_SELECT
       -> WIRE_HD + PROFILE_2
       -> STANDARD_V3_ENCODE
       -> OUTPUT

DECODE: UTILITY_CANDIDATES
       -> WIRE_HD_CANDIDATE
       -> STANDARD_V3_RECEIVER
       -> PROFILE_2_CDF97_INVERSE
       -> OUTPUT
```

V5 is deliberately an alias of the active `wire-hd` state machine at the
transport level. `[C:animation_modem/engines.py:129-170]`

The separate parked path is not this state machine:

```text
PARKED_V5_ALT: MONO_FRAME -> PULSE_CHECK -> MONO_OFDM
              -> LDPC_HEADER -> LDPC_PAYLOAD -> CDF97_INVERSE
```

Its implementation is `[C:animation_modem/transport3_v5.py:46-141]` and it is
not selected by `V5Engine`. `[C:animation_modem/engines.py:129-167]`

## `wire-tape` — encode/decode

```text
ENCODE: SOURCE_80x96 -> DCT_800_VALUES
       -> STEREO_REPEAT_800 -> OPPOSITE_TRACK/FREQUENCY SLOTS
       -> V3_OFDM -> 14kHz_BOUND -> OUTPUT

DECODE: PULSE_ACQUIRE -> WIRE_TAPE_LENGTH(2912)
       -> V3_EQUALIZE -> PROFILE_2_LAYOUT-SPECIFIC DCT CODER
       -> STEREO_REPEAT_SOFT_COMBINE -> DCT_INVERSE -> OUTPUT
```

The layout/coder dispatch is `[C:animation_modem/transport3.py:78-84]`,
`[C:animation_modem/engines.py:34-49]`; repeat placement and inverse are
`[C:animation_modem/wavelet.py:885-956]`.

## `wire-tape-25` — encode/decode

```text
ENCODE: SOURCE_80x60 -> DCT_360_VALUES
       -> STEREO_REPEAT_360 -> DIVERSE SLOTS
       -> V3_OFDM -> 14kHz_BOUND -> OUTPUT

DECODE: PULSE_ACQUIRE -> WIRE_TAPE_25_LENGTH(1904)
       -> V3_EQUALIZE -> PROFILE_3
       -> STEREO_REPEAT_SOFT_COMBINE -> DCT_INVERSE_80x60 -> OUTPUT
```

The layout is `[C:animation_modem/transport3.py:86-92]`; the coder geometry is
`[C:animation_modem/wavelet.py:959-962]`.

## V6 — encode/decode

```text
ENCODE: SOURCE_80x96 -> DCT_OR_CDF97_2880
       -> SELECT_720_FOUNDATION
       -> ANALOG_GAIN_AND_APPEND_COPIES
       -> GLOBAL_RANK_SLOT_ASSIGNMENT
       -> V6_HEADER/OFDM -> 14kHz_BOUND -> OUTPUT

DECODE: PULSE_ACQUIRE -> WIRE_V6_LENGTH(5360)
       -> V3_EQUALIZE/PILOT_TRACK
       -> PROFILE_0_OR_1
       -> RELIABILITY_FUSION + CONFIDENCE_FLOORS
       -> DCT_OR_CDF97_INVERSE -> VERIFIED/DAMAGED OUTPUT
```

The implementation is `[C:animation_modem/v6.py:77-224]` and the V6 layout is
`[C:animation_modem/transport3.py:94-100]`.

## V6-repeat — encode/decode

```text
ENCODE: SOURCE_80x96 -> DCT_OR_CDF97_2880
       -> COPY_ALL_2880
       -> GLOBAL_OPPOSITE_TRACK/FREQUENCY/SYMBOL ASSIGNMENT
       -> VR_HEADER/OFDM -> 14kHz_BOUND -> OUTPUT

DECODE: PULSE_ACQUIRE -> WIRE_V6_REPEAT_LENGTH(7952)
       -> V3_EQUALIZE/PILOT_TRACK
       -> PROFILE_0_OR_1
       -> RELIABILITY_FUSION_OVER_ALL_COPIES
       -> INVERSE_TRANSFORM -> VERIFIED/DAMAGED OUTPUT
```

The repeat layout and coder are `[C:animation_modem/transport3.py:102-108]`,
`[C:animation_modem/v6.py:204-216]`; receiver construction precomputes the
global placement before packet one at `[C:animation_modem/engines.py:68-75]`.

## V6 tape-ordered placement — placement state machine

```text
CODER_CREATED
  -> COMPUTE_RANKS [DCT normalized rank or CDF97 packed rank]
  -> BUILD_TAPE_SLOT_ORDER
       [bin 1 health=28; other bins by frequency; coprime symbol walk]
  -> SELECT_FOUNDATION [720 lowest-rank values]
  -> BUILD_LOW_CARRIER_ZONE [homes low half, copies upper half]
  -> PAIR_COPIES
       [opposite track, >=7-bin spread, half-packet symbol shift,
        same I/Q part]
  -> PLACE_REMAINING_DETAIL [rank order in unused slots]
  -> CACHE_MAPPING_PER_LAYOUT
  -> ENCODE/DECODE_USE_SAME_MAPPING
```

This is a bench-only coder on `WIRE_V6`; it does not add a new packet length or
magic and is not selected by the current live engine. The placement state is
`[C:animation_modem/v6.py:92-119]`, `[C:animation_modem/v6.py:285-369]`; test
invariants are `[C:modem_tests/test_v6_tape.py:33-66]`.
