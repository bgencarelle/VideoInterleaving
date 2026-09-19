#!/usr/bin/env bash
# Run the practical lossy-media acceptance matrix.
#
# Usage:
#   tools/test_lossy_media.sh
#   ATRAC_WAV=scratch/atrac-roundtrip.wav tools/test_lossy_media.sh
#
# The ATRAC path is supplied by a real MiniDisc/ATRAC round trip when available;
# ffmpeg generally cannot create a representative ATRAC recording here.
set -euo pipefail

ROOT=$(cd "$(dirname "$0")/.." && pwd)
cd "$ROOT"
FFMPEG=${FFMPEG:-ffmpeg}
PYTHON=${PYTHON:-.venv/bin/python}
OUT=${LOSSY_OUT:-scratch/lossy-media-tests}
FRAMES=${LOSSY_FRAMES:-60}
BITRATES_MP3=(192k 256k 320k)
BITRATES_AAC=(128k 256k 384k)

command -v "$FFMPEG" >/dev/null || { echo "ffmpeg not found: $FFMPEG" >&2; exit 1; }
[[ -x "$PYTHON" ]] || { echo "Python not found: $PYTHON" >&2; exit 1; }
mkdir -p "$OUT"

make_reference() {
  "$PYTHON" modem_screen.py --source test --profile hd-dwt \
    --frames "$FRAMES" --write "$OUT/reference.wav"
  "$PYTHON" tools/decode_wav.py "$OUT/reference.wav" \
    --out "$OUT/reference-frames" --frames "$FRAMES" --scale 1 -v \
    > "$OUT/reference.log" 2>&1
}

decode_case() {
  local label=$1
  local wav=$2
  "$PYTHON" tools/decode_wav.py "$wav" \
    --out "$OUT/$label-frames" --frames "$FRAMES" --scale 1 -v \
    > "$OUT/$label.log" 2>&1 || true
}

encode_mp3() {
  local rate=$1
  local label="mp3-$rate"
  "$FFMPEG" -y -hide_banner -loglevel error -i "$OUT/reference.wav" \
    -ar 48000 -ac 2 -c:a libmp3lame -b:a "$rate" -joint_stereo 1 \
    "$OUT/$label.mp3"
  "$FFMPEG" -y -hide_banner -loglevel error -i "$OUT/$label.mp3" \
    -ar 48000 -ac 2 -c:a pcm_s16le "$OUT/$label.wav"
  decode_case "$label" "$OUT/$label.wav"
}

encode_aac() {
  local rate=$1
  local label="aac-$rate"
  "$FFMPEG" -y -hide_banner -loglevel error -i "$OUT/reference.wav" \
    -ar 48000 -ac 2 -c:a aac -b:a "$rate" "$OUT/$label.m4a"
  "$FFMPEG" -y -hide_banner -loglevel error -i "$OUT/$label.m4a" \
    -ar 48000 -ac 2 -c:a pcm_s16le "$OUT/$label.wav"
  decode_case "$label" "$OUT/$label.wav"
}

make_reference
for rate in "${BITRATES_MP3[@]}"; do encode_mp3 "$rate"; done
for rate in "${BITRATES_AAC[@]}"; do encode_aac "$rate"; done

if [[ -n "${ATRAC_WAV:-}" ]]; then
  if [[ -f "$ATRAC_WAV" ]]; then
    decode_case atrac "$ATRAC_WAV"
  else
    echo "ATRAC_WAV does not exist: $ATRAC_WAV" >&2
  fi
else
  echo "ATRAC: skipped; set ATRAC_WAV to a real MiniDisc/ATRAC round-trip WAV"
fi

"$PYTHON" tools/summarize_lossy_media.py "$OUT" \
  --frames "$FRAMES" | tee "$OUT/summary.txt"
echo "PNG frames and logs: $ROOT/$OUT"
