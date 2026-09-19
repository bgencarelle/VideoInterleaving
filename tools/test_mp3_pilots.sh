#!/usr/bin/env bash
# Test whether known low-level tones make MP3 preserve important modem bands.
# Usage:
#   tools/test_mp3_pilots.sh scratch/audio-tests/ceiling-14000.wav
#
# Results are written beside the input under mp3-pilot-tests/.
set -euo pipefail

INPUT=${1:?usage: $0 INPUT_WAV [BITRATE]}
BITRATE=${2:-320k}
FFMPEG=${FFMPEG:-ffmpeg}
FFPROBE=${FFPROBE:-ffprobe}
PYTHON=${PYTHON:-.venv/bin/python}
ROOT=$(cd "$(dirname "$INPUT")" && pwd)
NAME=$(basename "${INPUT%.*}")
OUT="$ROOT/mp3-pilot-tests/$NAME-$BITRATE"
mkdir -p "$OUT"

if ! command -v "$FFMPEG" >/dev/null; then
  echo "ffmpeg not found: $FFMPEG" >&2
  exit 1
fi
if [[ ! -x "$PYTHON" ]]; then
  echo "Python not found: $PYTHON" >&2
  exit 1
fi

DURATION=$($FFPROBE -hide_banner -v error -i "$INPUT" \
  -show_entries format=duration -of default=nw=1:nk=1 2>/dev/null || true)
if [[ -z "$DURATION" ]]; then
  echo "Could not determine duration of $INPUT" >&2
  exit 1
fi

# Frequencies are deliberately below the 14 kHz ceiling that was most usable.
# Levels are modest: enough to test codec preservation without dominating the
# modem waveform. The baseline is essential for a direct comparison.
FREQUENCIES=(none 375 3750 7500 11250)
LEVELS=(0.005 0.01 0.02 0.04)

run_case() {
  local label=$1
  local tone=$2
  local level=$3
  local mixed="$OUT/$label-input.wav"
  local mp3="$OUT/$label.mp3"
  local decoded="$OUT/$label-decoded.wav"
  local frames="$OUT/$label-frames"

  if [[ "$tone" == none ]]; then
    cp "$INPUT" "$mixed"
  else
    # Explicit stereo tone: both modem legs receive the same known pilot.
    $FFMPEG -y -hide_banner -loglevel error \
      -i "$INPUT" \
      -f lavfi -i "aevalsrc=${level}*sin(2*PI*${tone}*t)|${level}*sin(2*PI*${tone}*t):s=48000:d=${DURATION}" \
      -filter_complex "[0:a][1:a]amix=inputs=2:duration=first:normalize=0" \
      -ar 48000 -ac 2 -c:a pcm_s16le "$mixed"
  fi

  $FFMPEG -y -hide_banner -loglevel error \
    -i "$mixed" -ar 48000 -ac 2 \
    -c:a libmp3lame -b:a "$BITRATE" "$mp3"

  $FFMPEG -y -hide_banner -loglevel error \
    -i "$mp3" -ar 48000 -ac 2 \
    -c:a pcm_s16le "$decoded"

  "$PYTHON" tools/decode_wav.py "$decoded" \
    --out "$frames" --frames 60 -v \
    > "$OUT/$label-decode.log" 2>&1 || true

  echo "completed $label"
}

run_case baseline none 0
for tone in "${FREQUENCIES[@]:1}"; do
  for level in "${LEVELS[@]}"; do
    run_case "pilot-${tone}hz-${level}" "$tone" "$level"
  done
done

echo
echo "Pilot tests written to: $OUT"
echo "Review decode logs with:"
echo "  grep -E 'verified|picture|pilot|coverage|frames|lost' $OUT/*-decode.log"
