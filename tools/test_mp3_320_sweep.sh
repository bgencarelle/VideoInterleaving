#!/usr/bin/env bash
# MP3 320k modem sweep: emission ceiling x encoder preparation filter.
set -euo pipefail

ROOT=$(cd "$(dirname "$0")/.." && pwd)
cd "$ROOT"
PYTHON=${PYTHON:-.venv/bin/python}
FFMPEG=${FFMPEG:-ffmpeg}
OUT=${MP3_SWEEP_OUT:-scratch/mp3-320-sweep}
FRAMES=${MP3_SWEEP_FRAMES:-60}
mkdir -p "$OUT"

command -v "$FFMPEG" >/dev/null || { echo "ffmpeg not found" >&2; exit 1; }
[[ -x "$PYTHON" ]] || { echo "Python not found: $PYTHON" >&2; exit 1; }

for ceiling in 14000 16000 18000 20250; do
  for filter in box nearest lanczos bicubic; do
    label="ceiling-${ceiling}-${filter}"
    echo "=== $label ==="
    "$PYTHON" modem_screen.py --source test --profile hd-dwt \
      --emit-ceiling "$ceiling" --encode-filter "$filter" \
      --frames "$FRAMES" --write "$OUT/$label-reference.wav"
    "$FFMPEG" -y -hide_banner -loglevel error \
      -i "$OUT/$label-reference.wav" -ar 48000 -ac 2 \
      -c:a libmp3lame -b:a 320k -joint_stereo 1 \
      "$OUT/$label.mp3"
    "$FFMPEG" -y -hide_banner -loglevel error \
      -i "$OUT/$label.mp3" -ar 48000 -ac 2 -c:a pcm_s16le \
      "$OUT/$label.wav"
    "$PYTHON" tools/decode_wav.py "$OUT/$label.wav" \
      --out "$OUT/$label-frames" --frames "$FRAMES" --scale 1 -v \
      > "$OUT/$label.log" 2>&1 || true
  done
done

"$PYTHON" tools/summarize_lossy_media.py "$OUT" \
  --frames "$FRAMES" | tee "$OUT/summary.txt"
