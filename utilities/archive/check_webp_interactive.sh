#!/usr/bin/env bash

# Interactive, parallel WebP validator with real-time progress
# Works on macOS (old Bash) and Linux

set -u  # no -e so one failure doesn't kill the loop

echo "========================================"
echo "  Interactive WebP Integrity Scanner"
echo "     (parallel + live progress)"
echo "========================================"
echo

# --- Dependency Check ---
if ! command -v webpinfo >/dev/null 2>&1; then
  echo "ERROR: webpinfo not found."
  echo "Install with:"
  echo "  Linux : sudo apt install webp"
  echo "  macOS : brew install webp"
  exit 1
fi

if ! command -v dwebp >/dev/null 2>&1; then
  echo "ERROR: dwebp not found."
  echo "Install with:"
  echo "  Linux : sudo apt install webp"
  echo "  macOS : brew install webp"
  exit 1
fi

# --- Directory Prompt ---
read -rp "Enter directory to scan [default: current directory]: " DIR
DIR="${DIR:-.}"

if [[ ! -d "$DIR" ]]; then
  echo "ERROR: '$DIR' is not a valid directory."
  exit 1
fi

DIR=$(cd "$DIR" && pwd)

echo
echo "Scanning directory:"
echo "  $DIR"
echo

# --- Mode Selection ---
echo "Validation mode:"
echo "  1) Fast header check only   (webpinfo)"
echo "  2) Full decode validation   (dwebp)  ✅ safest"
read -rp "Choose mode [2]: " MODE
MODE="${MODE:-2}"

case "$MODE" in
  1) MODE_NAME="webpinfo" ;;
  2) MODE_NAME="dwebp" ;;
  *)
    echo "Invalid selection."
    exit 1
    ;;
esac

# --- Parallelism ---
# Try to guess a sensible default: number of CPUs
JOBS_DEFAULT="$(getconf _NPROCESSORS_ONLN 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4)"
read -rp "How many parallel workers? [default: $JOBS_DEFAULT]: " JOBS
JOBS="${JOBS:-$JOBS_DEFAULT}"

if ! [[ "$JOBS" =~ ^[0-9]+$ ]] || [ "$JOBS" -lt 1 ]; then
  echo "Invalid worker count."
  exit 1
fi

# --- Optional Log File ---
read -rp "Write bad files to a persistent log file? [y/N]: " LOG_CHOICE

LOG_FILE=""
KEEP_LOG=false
case "$LOG_CHOICE" in
  y|Y|yes|YES)
    read -rp "Enter log file path [default: bad_webp_files.txt]: " LOG_FILE
    LOG_FILE="${LOG_FILE:-bad_webp_files.txt}"
    : > "$LOG_FILE"
    echo "Logging to: $LOG_FILE"
    KEEP_LOG=true
    ;;
  *)
    # Temporary log only for counting
    LOG_FILE="$(mktemp -t bad_webp_XXXXXX)"
    KEEP_LOG=false
    ;;
esac

# --- Precompute file list size ---
echo
echo "----------------------------------------"
echo "Analyzing file list..."
echo "----------------------------------------"

TOTAL=$(find "$DIR" -type f -iname '*.webp' | wc -l | tr -d '[:space:]')

if [ "$TOTAL" -eq 0 ]; then
  echo "No .webp files found under: $DIR"
  $KEEP_LOG || rm -f "$LOG_FILE"
  exit 0
fi

echo "Total WebP files found: $TOTAL"
echo "Mode: $MODE_NAME"
echo "Workers: $JOBS"
echo

# --- Progress tracking ---
PROGRESS_FILE="$(mktemp -t webp_progress_XXXXXX)"

# Progress watcher
show_progress() {
  local done bad
  while kill -0 "$1" 2>/dev/null; do
    if [ -f "$PROGRESS_FILE" ]; then
      done=$(wc -l < "$PROGRESS_FILE" 2>/dev/null || echo 0)
    else
      done=0
    fi

    if [ -f "$LOG_FILE" ]; then
      bad=$(wc -l < "$LOG_FILE" 2>/dev/null || echo 0)
    else
      bad=0
    fi

    printf "\rChecked: %d / %d | Invalid: %d" "$done" "$TOTAL" "$bad"
    sleep 0.3
  done

  # Final update after workers finish
  if [ -f "$PROGRESS_FILE" ]; then
    done=$(wc -l < "$PROGRESS_FILE" 2>/dev/null || echo 0)
  else
    done=0
  fi
  if [ -f "$LOG_FILE" ]; then
    bad=$(wc -l < "$LOG_FILE" 2>/dev/null || echo 0)
  else
    bad=0
  fi
  printf "\rChecked: %d / %d | Invalid: %d\n" "$done" "$TOTAL" "$bad"
}

echo "----------------------------------------"
echo "Starting parallel scan..."
echo "----------------------------------------"
echo

export MODE
export LOG_FILE
export PROGRESS_FILE

# Run find+xargs in a subshell in the background
(
  find "$DIR" -type f -iname '*.webp' -print0 |
    xargs -0 -n 1 -P "$JOBS" bash -c '
      FILE="$1"
      # Validate file
      if [ "$MODE" = "1" ]; then
        if ! webpinfo "$FILE" >/dev/null 2>&1; then
          echo "BAD: $FILE"
          echo "$FILE" >> "$LOG_FILE"
        fi
      else
        if ! dwebp "$FILE" -ppm -o /dev/null >/dev/null 2>&1; then
          echo "BAD (decode failed): $FILE"
          echo "$FILE" >> "$LOG_FILE"
        fi
      fi
      # Record progress (one line per processed file)
      echo "." >> "$PROGRESS_FILE"
    ' _
) &
XARGS_PID=$!

# Start progress watcher
show_progress "$XARGS_PID" &
PROG_PID=$!

# Wait for workers to finish
wait "$XARGS_PID" 2>/dev/null || true
# Stop progress watcher
wait "$PROG_PID" 2>/dev/null || true

echo
echo "========================================"
echo "Scan Complete"
echo "========================================"

BAD=0
if [ -f "$LOG_FILE" ]; then
  BAD=$(wc -l < "$LOG_FILE" 2>/dev/null || echo 0)
fi

echo "Total WebP files checked : $TOTAL"
echo "Invalid / Corrupt files  : $BAD"

if $KEEP_LOG; then
  if [ "$BAD" -gt 0 ]; then
    echo "Bad files logged in: $LOG_FILE"
  else
    echo "No bad files; log file is empty: $LOG_FILE"
  fi
else
  # Temporary log; discard it
  rm -f "$LOG_FILE"
fi

# Clean up progress file
rm -f "$PROGRESS_FILE"

if [ "$BAD" -gt 0 ]; then
  echo "STATUS: ❌ Corruption detected"
  exit 1
else
  echo "STATUS: ✅ All WebP files are valid"
  exit 0
fi
