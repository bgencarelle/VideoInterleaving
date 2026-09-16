#!/usr/bin/env bash
set -euo pipefail

die() { echo "ERROR: $*" >&2; exit 1; }
info() { echo "==> $*"; }

need_root() {
  if [[ "${EUID:-$(id -u)}" -ne 0 ]]; then
    die "Run with sudo: sudo $0"
  fi
}

detect_user() {
  # Prefer the user who invoked sudo; otherwise fall back to logname/id.
  if [[ -n "${SUDO_USER:-}" && "${SUDO_USER}" != "root" ]]; then
    echo "$SUDO_USER"
    return
  fi
  if command -v logname >/dev/null 2>&1; then
    local ln
    ln="$(logname 2>/dev/null || true)"
    if [[ -n "$ln" && "$ln" != "root" ]]; then
      echo "$ln"
      return
    fi
  fi
  echo "$(id -un)"
}

user_home() {
  local u="$1"
  getent passwd "$u" | awk -F: '{print $6}'
}

detect_keyboard_layout() {
  # Try to detect keyboard layout from system
  if command -v localectl >/dev/null 2>&1; then
    localectl status 2>/dev/null | grep -i "X11 Layout:" | awk '{print $3}' | tr '[:upper:]' '[:lower:]' || echo "us"
  elif [ -f /etc/default/keyboard ]; then
    grep "^XKBLAYOUT=" /etc/default/keyboard 2>/dev/null | cut -d= -f2 | tr -d '"' | tr '[:upper:]' '[:lower:]' || echo "us"
  else
    echo "us"
  fi
}

need_root

TARGET_USER="$(detect_user)"
TARGET_HOME="$(user_home "$TARGET_USER")"
[[ -n "$TARGET_HOME" ]] || die "Could not determine home directory for user '$TARGET_USER'."

# App is always assumed to be at ~/VideoInterleaving for the user
APP_DIR_CHECK="$TARGET_HOME/VideoInterleaving"
VENV_PY_CHECK="$APP_DIR_CHECK/.venv/bin/python"
MAIN_PY_CHECK="$APP_DIR_CHECK/main.py"
RUN_KIOSK_SCRIPT="$APP_DIR_CHECK/run-kiosk.sh"
LOG_PATH="/var/log/videointerleaving.log"

# Weston config paths
WESTON_CONFIG_DIR="$TARGET_HOME/.config"
WESTON_CONFIG_FILE="$WESTON_CONFIG_DIR/weston.ini"

info "Target user: $TARGET_USER"
info "Target home: $TARGET_HOME"
info "Expecting app: $APP_DIR_CHECK"
info "Weston config: $WESTON_CONFIG_FILE"
info "Run script: $RUN_KIOSK_SCRIPT"
info "Log: $LOG_PATH"
info ""
info "NOTE: Weston autolaunch feature requires Weston 15.0-alpha or later"
info "      (includes Weston 14.0.91 alpha / Weston 15.0-alpha+)"
info "      Older versions do not support the [autolaunch] section"

# Sanity checks (do not create venv automatically)
[[ -d "$APP_DIR_CHECK" ]] || die "Missing app directory: $APP_DIR_CHECK"
[[ -f "$MAIN_PY_CHECK" ]] || die "Missing: $MAIN_PY_CHECK"
[[ -x "$VENV_PY_CHECK" ]] || die "Missing venv python: $VENV_PY_CHECK"

# 1) Create run-kiosk.sh script in app directory
info "Creating run-kiosk.sh launcher script..."
cat >"$RUN_KIOSK_SCRIPT" <<'EOF'
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
EOF

chmod 0755 "$RUN_KIOSK_SCRIPT"
chown "$TARGET_USER":"$TARGET_USER" "$RUN_KIOSK_SCRIPT"
info "Created run-kiosk.sh: $RUN_KIOSK_SCRIPT"

# 2) Ensure log exists and is writable by the target user
info "Configuring log file permissions..."
touch "$LOG_PATH"
chown "$TARGET_USER":"$TARGET_USER" "$LOG_PATH"
chmod 0644 "$LOG_PATH"

# 3) Detect keyboard layout
KEYBOARD_LAYOUT="$(detect_keyboard_layout)"
info "Detected keyboard layout: $KEYBOARD_LAYOUT"

# 4) Create Weston configuration file
info "Creating Weston configuration..."
mkdir -p "$WESTON_CONFIG_DIR"
chown -R "$TARGET_USER":"$TARGET_USER" "$WESTON_CONFIG_DIR"

cat >"$WESTON_CONFIG_FILE" <<WESTONEOF
[core]
shell=fullscreen
idle-time=0
# Keep Xwayland only if you need X11 apps:
xwayland=true

# If colors are odd, these two are the first sane DRM knobs to try:
# (leave them commented unless you need them)
# gbm-format=xrgb8888

[keyboard]
keymap_layout=$KEYBOARD_LAYOUT
# keymap_variant=nodeadkeys
# keymap_options=compose:ralt

[output]
mode=preferred
# If colors look wrong / banded / odd depth:
# max-bpc=8
# gbm-format=xrgb8888

# Autolaunch section - requires Weston 15.0-alpha or later
# For older Weston versions, this section will be ignored
[autolaunch]
path=$RUN_KIOSK_SCRIPT
watch=true
WESTONEOF

chown "$TARGET_USER":"$TARGET_USER" "$WESTON_CONFIG_FILE"
chmod 0644 "$WESTON_CONFIG_FILE"
info "Created Weston config: $WESTON_CONFIG_FILE"

info "Done."
echo
echo "================================================================"
echo "WESTON KIOSK SETUP COMPLETE"
echo "================================================================"
echo
echo "Configuration files created:"
echo "  1. Weston config: $WESTON_CONFIG_FILE"
echo "  2. Kiosk launcher: $RUN_KIOSK_SCRIPT"
echo "  3. Log file: $LOG_PATH"
echo
echo "To start Weston with this configuration:"
echo "  weston --config=$WESTON_CONFIG_FILE"
echo
echo "Or if Weston is configured to use ~/.config/weston.ini automatically:"
echo "  weston"
echo
echo "The application will auto-launch when Weston starts."
echo
echo "Check logs:"
echo "  tail -f $LOG_PATH"
echo
echo "To modify the configuration:"
echo "  Edit: $WESTON_CONFIG_FILE"
echo
