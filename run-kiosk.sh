#!/usr/bin/env bash
set -euo pipefail

# Kiosk launcher script for Weston autolaunch
# This script is designed to be launched by Weston's autolaunch feature

# Ensure HOME is set - use fallback if not available
if [ -z "${HOME:-}" ]; then
  export HOME="$(getent passwd "$(id -u)" | cut -d: -f6)"
fi

# Verify HOME is valid
if [ -z "$HOME" ] || [ ! -d "$HOME" ]; then
  echo "ERROR: Cannot determine home directory" >&2
  exit 1
fi

APP_DIR="$HOME/VideoInterleaving"
VENV_PY="$APP_DIR/.venv/bin/python"
LOG="/var/log/videointerleaving.log"
RESTART_DELAY=3

# Logging function - always append to log file
log_msg() {
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" >>"$LOG" 2>&1
}

# Change to app directory
cd "$APP_DIR" || {
  log_msg "ERROR: Cannot cd to $APP_DIR"
  exit 1
}

# Verify Python executable exists
if [ ! -x "$VENV_PY" ]; then
  log_msg "ERROR: Python not found at $VENV_PY"
  exit 1
fi

# Main restart loop
restart_count=0
while true; do
  if [ $restart_count -gt 0 ]; then
    log_msg "Restarting application (attempt #$restart_count)..."
    sleep "$RESTART_DELAY"
  else
    log_msg "Starting VideoInterleaving application..."
  fi
  
  # Run Python with unbuffered output, redirecting BOTH stdout and stderr to log
  set +e  # Don't exit on error, we want to check exit code
  "$VENV_PY" -u main.py --mode local >>"$LOG" 2>&1
  exit_code=$?
  set -e
  
  if [ $exit_code -eq 0 ]; then
    log_msg "Application exited normally (code: $exit_code)"
    # Exit on clean shutdown, don't restart
    break
  else
    log_msg "Application crashed or was killed (exit code: $exit_code) - will restart"
    restart_count=$((restart_count + 1))
    # Continue loop to restart
  fi
done
