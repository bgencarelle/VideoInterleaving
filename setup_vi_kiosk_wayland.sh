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

# Fixed, per your requirement:
# App is always assumed to be at ~/VideoInterleaving for the user running labwc.
APP_DIR_CHECK="$TARGET_HOME/VideoInterleaving"
VENV_PY_CHECK="$APP_DIR_CHECK/.venv/bin/python"
MAIN_PY_CHECK="$APP_DIR_CHECK/main.py"

KIOSK_DIR="/opt/kiosk"
WRAPPER_PATH="$KIOSK_DIR/run_videointerleaving.sh"
LOG_PATH="/var/log/videointerleaving.log"

LABWC_DIR="$TARGET_HOME/.config/labwc"
AUTOSTART_FILE="$LABWC_DIR/autostart"
AUTOSTART_LINE="/usr/bin/lwrespawn $WRAPPER_PATH"

info "Target user: $TARGET_USER"
info "Target home: $TARGET_HOME"
info "Expecting app: $APP_DIR_CHECK"
info "Wrapper: $WRAPPER_PATH"
info "Autostart: $AUTOSTART_FILE"
info "Log: $LOG_PATH"

# Sanity checks (do not create venv automatically)
[[ -d "$APP_DIR_CHECK" ]] || die "Missing app directory: $APP_DIR_CHECK"
[[ -f "$MAIN_PY_CHECK" ]] || die "Missing: $MAIN_PY_CHECK"
[[ -x "$VENV_PY_CHECK" ]] || die "Missing venv python: $VENV_PY_CHECK"

# 1) Create wrapper script (IMPORTANT: uses $HOME at runtime)
info "Creating wrapper script..."
mkdir -p "$KIOSK_DIR"

cat >"$WRAPPER_PATH" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail

APP_DIR="$HOME/VideoInterleaving"
VENV_PY="$APP_DIR/.venv/bin/python"
LOG="/var/log/videointerleaving.log"

cd "$APP_DIR"

# Unbuffered so logs are real-time and crash context is preserved
exec "$VENV_PY" -u main.py --mode local --dir images_sbs >>"$LOG" 2>&1
EOF

chmod 0755 "$WRAPPER_PATH"

# 2) Ensure log exists and is writable by the target user
info "Configuring log file permissions..."
touch "$LOG_PATH"
chown "$TARGET_USER":"$TARGET_USER" "$LOG_PATH"
chmod 0644 "$LOG_PATH"

# 3) Add labwc autostart line (idempotent)
info "Configuring labwc autostart..."
mkdir -p "$LABWC_DIR"
chown -R "$TARGET_USER":"$TARGET_USER" "$TARGET_HOME/.config"

if [[ ! -f "$AUTOSTART_FILE" ]]; then
  touch "$AUTOSTART_FILE"
  chown "$TARGET_USER":"$TARGET_USER" "$AUTOSTART_FILE"
  chmod 0644 "$AUTOSTART_FILE"
fi

if grep -qxF "$AUTOSTART_LINE" "$AUTOSTART_FILE"; then
  info "Autostart line already present."
else
  echo "$AUTOSTART_LINE" >>"$AUTOSTART_FILE"
  info "Autostart line appended."
fi

info "Done."
echo
echo "Reboot or log out/in to apply:"
echo "  sudo reboot"
echo
echo "Check logs:"
echo "  tail -n 200 $LOG_PATH"
echo
echo "If it launches twice on Raspberry Pi OS, also inspect:"
echo "  /etc/xdg/labwc/autostart"
