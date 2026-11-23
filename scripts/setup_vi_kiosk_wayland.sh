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

# lwrespawn is a Raspberry Pi OS addition; stock Debian's labwc doesn't ship it.
# On the Pi keep the exact autostart line that is known to work. Elsewhere the
# wrapper restarts the app itself (VI_RESPAWN=1) and is backgrounded so it
# doesn't hold up the rest of labwc's autostart.
if [[ -x /usr/bin/lwrespawn ]]; then
  AUTOSTART_LINE="/usr/bin/lwrespawn $WRAPPER_PATH"
else
  AUTOSTART_LINE="VI_RESPAWN=1 $WRAPPER_PATH &"
fi

# Plasma's own settings file is /etc/sddm.conf.d/kde_settings.conf; SDDM reads
# conf.d in alphabetical order with later files winning, hence the zz- prefix.
SDDM_AUTOLOGIN_CONF="/etc/sddm.conf.d/zz-videointerleaving-autologin.conf"
LABWC_SESSION_FILE="/usr/share/wayland-sessions/labwc.desktop"

info "Target user: $TARGET_USER"
info "Target home: $TARGET_HOME"
info "Expecting app: $APP_DIR_CHECK"
info "Wrapper: $WRAPPER_PATH"
info "Autostart: $AUTOSTART_FILE"
info "Autostart line: $AUTOSTART_LINE"
info "Log: $LOG_PATH"

# Sanity checks (do not create venv automatically)
command -v labwc >/dev/null 2>&1 || die "labwc is not installed: sudo apt install labwc"
[[ -d "$APP_DIR_CHECK" ]] || die "Missing app directory: $APP_DIR_CHECK"
[[ -f "$MAIN_PY_CHECK" ]] || die "Missing: $MAIN_PY_CHECK"
[[ -x "$VENV_PY_CHECK" ]] || die "Missing venv python: $VENV_PY_CHECK"

# 1) Create wrapper script (IMPORTANT: uses $HOME at runtime)
info "Creating wrapper script..."
mkdir -p "$KIOSK_DIR"

cat >"$WRAPPER_PATH" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail

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

cd "$APP_DIR"

# Without lwrespawn (stock Debian labwc) the autostart line sets VI_RESPAWN=1
# and this wrapper restarts the app itself, mirroring what lwrespawn does on
# the Pi: relaunch after any exit, with a short delay so a crash loop can't
# spin the CPU.
#
# The loop is tied to the labwc instance that launched it. Autostart runs in
# a shell that exits right away, so this wrapper outlives its parent; without
# the check, a labwc restart (SDDM Relogin) would leave the old loop running
# alongside the new session's one and the app would start twice. labwc
# exports LABWC_PID to autostart; older versions don't, so fall back to the
# newest labwc owned by this user.
if [ "${VI_RESPAWN:-0}" = "1" ]; then
  COMPOSITOR_PID="${LABWC_PID:-$(pgrep -u "$(id -u)" -n -x labwc || true)}"
  if [ -z "$COMPOSITOR_PID" ]; then
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] ERROR: labwc not running; not starting" >>"$LOG"
    exit 1
  fi
  while kill -0 "$COMPOSITOR_PID" 2>/dev/null; do
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting VideoInterleaving (local)" >>"$LOG"
    "$VENV_PY" -u main.py --mode local --dir images_sbs >>"$LOG" 2>&1 || true
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] App exited; restarting in 3s" >>"$LOG"
    sleep 3
  done
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] labwc ($COMPOSITOR_PID) is gone; wrapper exiting" >>"$LOG"
  exit 0
fi

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
  # Drop any other variant of our line (e.g. after lwrespawn appeared or went
  # away) so the app never gets launched twice.
  sed -i "\#$WRAPPER_PATH#d" "$AUTOSTART_FILE"
  echo "$AUTOSTART_LINE" >>"$AUTOSTART_FILE"
  info "Autostart line written."
fi

# 4) Log straight into labwc where SDDM is the display manager (e.g. a Debian
# machine that came with Plasma). Raspberry Pi OS uses lightdm and sets up
# autologin through raspi-config, so it never reaches this branch.
if systemctl is-enabled --quiet sddm 2>/dev/null; then
  if [[ -f "$LABWC_SESSION_FILE" ]]; then
    info "Configuring SDDM autologin into labwc..."
    mkdir -p "$(dirname "$SDDM_AUTOLOGIN_CONF")"
    cat >"$SDDM_AUTOLOGIN_CONF" <<EOF
[Autologin]
User=$TARGET_USER
Session=$(basename "$LABWC_SESSION_FILE")
Relogin=true
EOF
    info "Wrote $SDDM_AUTOLOGIN_CONF (delete it to get the normal login screen back)."
    # /etc/sddm.conf is read after conf.d, so an [Autologin] section there
    # would silently override the file above.
    if [[ -f /etc/sddm.conf ]] && grep -q '^\[Autologin\]' /etc/sddm.conf; then
      info "WARNING: /etc/sddm.conf has its own [Autologin] section, which takes"
      info "         priority. Remove it or set Session=labwc.desktop there."
    fi
  else
    info "WARNING: SDDM is enabled but $LABWC_SESSION_FILE is missing;"
    info "         autologin not configured. Pick labwc at the login screen."
  fi
fi

# vi-local.service starts on its own schedule and can't see the compositor;
# if it is enabled it will fight this autostart for the display.
if systemctl is-enabled --quiet vi-local.service 2>/dev/null; then
  info "WARNING: vi-local.service is enabled and will compete with this setup."
  info "         Disable it with: sudo systemctl disable --now vi-local.service"
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
