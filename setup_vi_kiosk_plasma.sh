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

need_root

TARGET_USER="$(detect_user)"
TARGET_HOME="$(user_home "$TARGET_USER")"
[[ -n "$TARGET_HOME" ]] || die "Could not determine home directory for user '$TARGET_USER'."

# App is always assumed to be at ~/VideoInterleaving for the user
APP_DIR_CHECK="$TARGET_HOME/VideoInterleaving"
VENV_PY_CHECK="$APP_DIR_CHECK/.venv/bin/python"
MAIN_PY_CHECK="$APP_DIR_CHECK/main.py"

KIOSK_DIR="/opt/kiosk"
WRAPPER_PATH="$KIOSK_DIR/run_videointerleaving.sh"
LOG_PATH="/var/log/videointerleaving.log"

# Plasma-specific paths
PLASMA_AUTOSTART_DIR="$TARGET_HOME/.config/autostart"
DESKTOP_FILE="$PLASMA_AUTOSTART_DIR/videointerleaving.desktop"

info "Target user: $TARGET_USER"
info "Target home: $TARGET_HOME"
info "Expecting app: $APP_DIR_CHECK"
info "Wrapper: $WRAPPER_PATH"
info "Autostart: $DESKTOP_FILE"
info "Log: $LOG_PATH"

# Sanity checks (do not create venv automatically)
[[ -d "$APP_DIR_CHECK" ]] || die "Missing app directory: $APP_DIR_CHECK"
[[ -f "$MAIN_PY_CHECK" ]] || die "Missing: $MAIN_PY_CHECK"
[[ -x "$VENV_PY_CHECK" ]] || die "Missing venv python: $VENV_PY_CHECK"

# Verify Plasma/Wayland environment
if [[ -z "${WAYLAND_DISPLAY:-}" ]] && [[ "${XDG_SESSION_TYPE:-}" != "wayland" ]]; then
  info "Warning: WAYLAND_DISPLAY not set and XDG_SESSION_TYPE is not 'wayland'."
  info "This script is for Plasma Wayland, but continuing anyway..."
fi

# 1) Create wrapper script (IMPORTANT: uses $HOME at runtime)
info "Creating wrapper script..."
mkdir -p "$KIOSK_DIR"

cat >"$WRAPPER_PATH" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail

APP_DIR="$HOME/VideoInterleaving"
VENV_PY="$APP_DIR/.venv/bin/python"
LOG="/var/log/videointerleaving.log"
RESTART_DELAY=3

# Logging function - always append to log file
log_msg() {
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" >>"$LOG" 2>&1
}

# Set Wayland environment variables explicitly
export XDG_SESSION_TYPE=wayland
export WAYLAND_DISPLAY="${WAYLAND_DISPLAY:-wayland-0}"
export XDG_RUNTIME_DIR="${XDG_RUNTIME_DIR:-/run/user/$(id -u)}"

# Wait for Plasma session to be ready (max 30 seconds)
MAX_WAIT=30
WAITED=0
while [ $WAITED -lt $MAX_WAIT ]; do
  if [ -S "${XDG_RUNTIME_DIR}/wayland-${WAYLAND_DISPLAY#wayland-}" ] 2>/dev/null || \
     [ -n "${WAYLAND_DISPLAY:-}" ] && command -v kwin_wayland >/dev/null 2>&1; then
    break
  fi
  sleep 1
  WAITED=$((WAITED + 1))
done

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

# Small delay to ensure Plasma compositor is fully ready
sleep 2

# Main restart loop (no systemd needed)
restart_count=0
while true; do
  if [ $restart_count -gt 0 ]; then
    log_msg "Restarting application (attempt #$restart_count)..."
    sleep "$RESTART_DELAY"
  else
    log_msg "Starting VideoInterleaving application..."
  fi
  
  # Run Python with unbuffered output, redirecting BOTH stdout and stderr to log
  # Use exec to replace shell process, but we can't do that in a loop, so we run it normally
  # This ensures all output (including from Tee class) goes to the log file
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

chmod 0755 "$WRAPPER_PATH"

# 2) Ensure log exists and is writable by the target user
info "Configuring log file permissions..."
touch "$LOG_PATH"
chown "$TARGET_USER":"$TARGET_USER" "$LOG_PATH"
chmod 0644 "$LOG_PATH"

# 3) Create Plasma autostart .desktop file
info "Configuring Plasma autostart..."
mkdir -p "$PLASMA_AUTOSTART_DIR"
chown -R "$TARGET_USER":"$TARGET_USER" "$PLASMA_AUTOSTART_DIR"

cat >"$DESKTOP_FILE" <<EOF
[Desktop Entry]
Type=Application
Name=VideoInterleaving
Comment=VideoInterleaving Kiosk Mode
Exec=$WRAPPER_PATH
Icon=video-display
Terminal=false
NoDisplay=false
Hidden=false
X-KDE-Autostart-enabled=true
X-KDE-StartupNotify=false
X-KDE-Wayland-Enable=true
EOF

chown "$TARGET_USER":"$TARGET_USER" "$DESKTOP_FILE"
chmod 0644 "$DESKTOP_FILE"

# 4) Create systemd user service (optional but recommended for better reliability)
info "Creating systemd user service..."
SYSTEMD_USER_DIR="$TARGET_HOME/.config/systemd/user"
SERVICE_FILE="$SYSTEMD_USER_DIR/videointerleaving.service"

mkdir -p "$SYSTEMD_USER_DIR"
chown -R "$TARGET_USER":"$TARGET_USER" "$SYSTEMD_USER_DIR"

cat >"$SERVICE_FILE" <<EOF
[Unit]
Description=VideoInterleaving Kiosk Mode
After=plasma-workspace.target graphical.target
Wants=plasma-workspace.target

[Service]
Type=simple
ExecStart=$WRAPPER_PATH
Restart=always
RestartSec=5
# Use 'process' mode to track the Python process directly (after exec in wrapper)
KillMode=process
# Ensure systemd detects when the main process exits
RemainAfterExit=no
StandardOutput=append:$LOG_PATH
StandardError=append:$LOG_PATH
Environment="XDG_SESSION_TYPE=wayland"
Environment="WAYLAND_DISPLAY=wayland-0"

[Install]
WantedBy=default.target
EOF

chown "$TARGET_USER":"$TARGET_USER" "$SERVICE_FILE"
chmod 0644 "$SERVICE_FILE"

# Try to enable and start the service if user session is available
info "Systemd user service created at: $SERVICE_FILE"
if command -v systemctl >/dev/null 2>&1; then
    # Check if we can access user systemd (requires user session)
    if sudo -u "$TARGET_USER" systemctl --user daemon-reload 2>/dev/null; then
        info "Enabling and starting systemd user service..."
        if sudo -u "$TARGET_USER" systemctl --user enable videointerleaving.service 2>/dev/null; then
            info "Service enabled"
            if sudo -u "$TARGET_USER" systemctl --user start videointerleaving.service 2>/dev/null; then
                info "Service started"
            else
                info "Service created but could not start (may need user to log in first)"
            fi
        else
            info "Service created but could not enable (may need user to log in first)"
        fi
    else
        info "User systemd not available (user may need to log in first)"
        info "To enable the service later, run as $TARGET_USER:"
        info "  systemctl --user enable videointerleaving.service"
        info "  systemctl --user start videointerleaving.service"
    fi
else
    info "systemctl not available - service file created but not enabled"
fi

info "Done."
echo
echo "Setup complete! Two autostart methods have been configured:"
echo "  1. Plasma .desktop autostart: $DESKTOP_FILE"
echo "  2. Systemd user service: $SERVICE_FILE"
echo
echo "To apply changes:"
echo "  1. Log out and log back in (recommended), OR"
echo "  2. Reboot: sudo reboot"
echo
echo "To enable systemd service (optional, provides auto-restart on crash):"
echo "  sudo -u $TARGET_USER systemctl --user enable videointerleaving.service"
echo "  sudo -u $TARGET_USER systemctl --user start videointerleaving.service"
echo
echo "Check logs:"
echo "  tail -f $LOG_PATH"
echo
echo "To disable autostart:"
echo "  rm $DESKTOP_FILE"
echo "  sudo -u $TARGET_USER systemctl --user disable videointerleaving.service"
echo
echo "To test manually:"
echo "  $WRAPPER_PATH"
