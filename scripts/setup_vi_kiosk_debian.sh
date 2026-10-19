#!/usr/bin/env bash
set -euo pipefail

# Kiosk setup for plain Debian machines (e.g. a tablet that came with Plasma):
# the Debian counterpart of the Raspberry Pi setup in setup_vi_kiosk_wayland.sh.
#
# Default (no display manager): boot to the console, log the user in on tty1
# automatically, start labwc from that login, and let labwc's autostart run the
# app. Nothing else is started, which is as light as it gets.
#
#   sudo scripts/setup_vi_kiosk_debian.sh              # set up (no SDDM)
#   sudo scripts/setup_vi_kiosk_debian.sh --use-sddm   # keep SDDM, autologin into labwc
#   sudo scripts/setup_vi_kiosk_debian.sh --with-deps  # also run setup_app.sh first
#   sudo scripts/setup_vi_kiosk_debian.sh --revert     # undo, back to the login screen
#
# The app itself is launched exactly as on the Pi: setup_vi_kiosk_wayland.sh
# writes /opt/kiosk/run_videointerleaving.sh and the labwc autostart line, and
# this script calls it rather than duplicating it.
#
# vi-local.service is deliberately not used: a systemd unit starts on its own
# schedule and can't see the compositor. The headless services (vi-web,
# vi-ascii, vi-asciiweb) are left alone.

die() { echo "ERROR: $*" >&2; exit 1; }
info() { echo "==> $*"; }
warn() { echo "WARNING: $*" >&2; }

usage() {
  sed -n '4,15p' "$0" | sed 's/^# \{0,1\}//'
  exit "${1:-0}"
}

USE_SDDM=false
WITH_DEPS=false
REVERT=false
FORCE=false
for arg in "$@"; do
  case "$arg" in
    --use-sddm) USE_SDDM=true ;;
    --with-deps) WITH_DEPS=true ;;
    --revert) REVERT=true ;;
    --force) FORCE=true ;;
    -h|--help) usage 0 ;;
    *) echo "Unknown option: $arg" >&2; usage 1 ;;
  esac
done

[[ "${EUID:-$(id -u)}" -eq 0 ]] || die "Run with sudo: sudo $0 $*"

# Same user detection as the other kiosk scripts: the user who ran sudo.
detect_user() {
  if [[ -n "${SUDO_USER:-}" && "${SUDO_USER}" != "root" ]]; then
    echo "$SUDO_USER"; return
  fi
  local ln
  ln="$(logname 2>/dev/null || true)"
  if [[ -n "$ln" && "$ln" != "root" ]]; then
    echo "$ln"; return
  fi
  id -un
}

TARGET_USER="$(detect_user)"
[[ "$TARGET_USER" != "root" ]] || die "Run this with sudo from the kiosk user's account, not as root directly."
TARGET_HOME="$(getent passwd "$TARGET_USER" | cut -d: -f6)"
TARGET_SHELL="$(getent passwd "$TARGET_USER" | cut -d: -f7)"
[[ -d "$TARGET_HOME" ]] || die "Could not determine home directory for '$TARGET_USER'."

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PROJECT_DIR=$(cd "$SCRIPT_DIR/.." && pwd)
WAYLAND_SETUP="$SCRIPT_DIR/setup_vi_kiosk_wayland.sh"
WRAPPER_PATH="/opt/kiosk/run_videointerleaving.sh"

LABWC_DIR="$TARGET_HOME/.config/labwc"
LABWC_ENV_FILE="$LABWC_DIR/environment"
LABWC_AUTOSTART="$LABWC_DIR/autostart"
GETTY_OVERRIDE="/etc/systemd/system/getty@tty1.service.d/zz-videointerleaving-autologin.conf"
SDDM_AUTOLOGIN_CONF="/etc/sddm.conf.d/zz-videointerleaving-autologin.conf"
# Remembers which display manager we switched off so --revert can restore it.
STATE_FILE="/etc/videointerleaving-kiosk.state"

BLOCK_BEGIN="# >>> VideoInterleaving kiosk >>>"
BLOCK_END="# <<< VideoInterleaving kiosk <<<"

as_user() { sudo -u "$TARGET_USER" -H "$@"; }

# The display manager behind display-manager.service (sddm, lightdm, gdm3...),
# or nothing when the machine already boots to a console.
current_dm() {
  local link
  link="$(readlink -f /etc/systemd/system/display-manager.service 2>/dev/null || true)"
  [[ -n "$link" && -e "$link" ]] && basename "$link" .service
  return 0
}

# Login shell profile that starts labwc. bash reads ~/.bash_profile and only
# falls back to ~/.profile when it doesn't exist, so use whichever bash will
# actually read rather than creating a new file that would shadow ~/.profile.
profile_file() {
  case "$(basename "$TARGET_SHELL")" in
    zsh) echo "$TARGET_HOME/.zprofile" ;;
    bash)
      if [[ -f "$TARGET_HOME/.bash_profile" ]]; then echo "$TARGET_HOME/.bash_profile"
      elif [[ -f "$TARGET_HOME/.bash_login" ]]; then echo "$TARGET_HOME/.bash_login"
      else echo "$TARGET_HOME/.profile"; fi ;;
    sh|dash) echo "$TARGET_HOME/.profile" ;;
    *) echo "" ;;
  esac
}

remove_block() {
  local f="$1"
  [[ -f "$f" ]] || return 0
  sed -i "/^$BLOCK_BEGIN\$/,/^$BLOCK_END\$/d" "$f"
}

# Every path into the user's systemd instance; works whether or not they are
# logged in right now (falls back to removing the enablement links by hand).
disable_user_unit() {
  local unit="$1" dir="$TARGET_HOME/.config/systemd/user"
  [[ -f "$dir/$unit" ]] || return 0
  if ! systemctl --user -M "$TARGET_USER@" disable --now "$unit" >/dev/null 2>&1; then
    find "$dir" -path "*.wants/$unit" -delete 2>/dev/null || true
  fi
  info "Disabled user service $unit"
}

# --------------------------------------------------------------------------
# Revert
# --------------------------------------------------------------------------
if $REVERT; then
  info "Reverting kiosk setup for $TARGET_USER..."
  rm -f "$GETTY_OVERRIDE" "$SDDM_AUTOLOGIN_CONF"
  rmdir --ignore-fail-on-non-empty "$(dirname "$GETTY_OVERRIDE")" 2>/dev/null || true
  for f in "$TARGET_HOME/.bash_profile" "$TARGET_HOME/.bash_login" \
           "$TARGET_HOME/.profile" "$TARGET_HOME/.zprofile" "$LABWC_ENV_FILE"; do
    remove_block "$f"
  done
  [[ -f "$LABWC_AUTOSTART" ]] && sed -i "\#$WRAPPER_PATH#d" "$LABWC_AUTOSTART"

  # Put back the Plasma autostart entry if we parked it.
  parked="$TARGET_HOME/.config/autostart/videointerleaving.desktop.disabled-by-kiosk"
  [[ -f "$parked" ]] && info "Plasma autostart entry left parked at $parked (rename it back to re-enable)."

  prev_dm=""
  [[ -f "$STATE_FILE" ]] && prev_dm="$(sed -n 's/^previous_dm=//p' "$STATE_FILE")"
  if [[ -n "$prev_dm" ]]; then
    info "Re-enabling $prev_dm..."
    systemctl enable "$prev_dm.service" >/dev/null 2>&1 || warn "Could not enable $prev_dm.service"
    systemctl set-default graphical.target >/dev/null
  fi
  rm -f "$STATE_FILE"
  systemctl daemon-reload
  info "Reverted. Reboot to get the normal login screen back: sudo reboot"
  exit 0
fi

# --------------------------------------------------------------------------
# Setup
# --------------------------------------------------------------------------

# On the Pi, setup_vi_kiosk_wayland.sh alone is the tested path, and this
# script's console autologin would fight Raspberry Pi OS's own (raspi-config).
if [[ -x /usr/bin/lwrespawn || -f /etc/rpi-issue ]] && ! $FORCE; then
  die "This looks like Raspberry Pi OS. Use: sudo $WAYLAND_SETUP  (or pass --force)"
fi

DM="$(current_dm)"
if $USE_SDDM && [[ "$DM" != "sddm" ]]; then
  if [[ -f "$STATE_FILE" ]]; then
    die "SDDM was switched off by an earlier run. Run with --revert first, then --use-sddm."
  fi
  die "--use-sddm needs SDDM as the display manager (found: ${DM:-none})."
fi

PROFILE=""
if ! $USE_SDDM; then
  PROFILE="$(profile_file)"
  [[ -n "$PROFILE" ]] || die "Unsupported login shell for $TARGET_USER: $TARGET_SHELL (use bash, zsh or sh, or --use-sddm)."
fi

info "Kiosk user: $TARGET_USER ($TARGET_HOME)"
info "Project:    $PROJECT_DIR"
info "Display manager now: ${DM:-none}"
if $USE_SDDM; then
  info "Mode: SDDM autologin into labwc"
else
  info "Mode: no display manager (tty1 autologin -> labwc, via $PROFILE)"
fi
echo

# The wayland setup script and the wrapper both assume ~/VideoInterleaving.
[[ "$(readlink -f "$PROJECT_DIR")" == "$(readlink -f "$TARGET_HOME/VideoInterleaving")" ]] || \
  die "The kiosk wrapper runs $TARGET_HOME/VideoInterleaving, but this checkout is $PROJECT_DIR."

# 1) Packages
info "Installing labwc..."
if ! apt-get install -y -q labwc >/dev/null 2>&1; then
  apt-get update -q >/dev/null
  apt-get install -y -q labwc >/dev/null
fi

if $WITH_DEPS; then
  info "Running setup_app.sh for Python and system dependencies..."
  "$SCRIPT_DIR/setup_app.sh"
fi
[[ -x "$PROJECT_DIR/.venv/bin/python" ]] || \
  die "No virtualenv at $PROJECT_DIR/.venv. Run again with --with-deps, or run scripts/setup_app.sh first."

# 2) Switch off every other way the local app may have been started, so only
#    the labwc autostart launches it.
info "Disabling other local-mode launchers..."
if systemctl is-enabled --quiet vi-local.service 2>/dev/null; then
  systemctl disable --now vi-local.service >/dev/null 2>&1 || true
  info "Disabled vi-local.service"
fi
disable_user_unit vi-local.service
disable_user_unit videointerleaving.service      # from setup_vi_kiosk_plasma.sh
plasma_desktop="$TARGET_HOME/.config/autostart/videointerleaving.desktop"
if [[ -f "$plasma_desktop" ]]; then
  mv "$plasma_desktop" "$plasma_desktop.disabled-by-kiosk"
  info "Parked Plasma autostart entry ($plasma_desktop)"
fi

# 3) Login path. Done before calling the wayland setup script, because that
#    script writes SDDM autologin whenever SDDM is still enabled.
if ! $USE_SDDM; then
  if [[ -n "$DM" ]]; then
    info "Disabling display manager $DM (boot to console instead)..."
    if [[ ! -f "$STATE_FILE" ]]; then
      echo "previous_dm=$DM" >"$STATE_FILE"
    fi
    systemctl disable "$DM.service" >/dev/null 2>&1 || true
  fi
  rm -f "$SDDM_AUTOLOGIN_CONF"     # left over from an earlier --use-sddm run
  systemctl set-default multi-user.target >/dev/null

  info "Configuring autologin on tty1..."
  mkdir -p "$(dirname "$GETTY_OVERRIDE")"
  cat >"$GETTY_OVERRIDE" <<EOF
[Service]
ExecStart=
ExecStart=-/sbin/agetty --autologin $TARGET_USER --noclear %I \$TERM
EOF

  info "Starting labwc from the tty1 login ($PROFILE)..."
  [[ -f "$PROFILE" ]] || { touch "$PROFILE"; chown "$TARGET_USER":"$TARGET_USER" "$PROFILE"; }
  remove_block "$PROFILE"
  # Only on tty1 and only outside a graphical session, so SSH logins and other
  # consoles stay ordinary shells. If labwc exits, the getty restarts the login
  # and labwc comes back.
  cat >>"$PROFILE" <<EOF
$BLOCK_BEGIN
if [ -z "\${WAYLAND_DISPLAY:-}" ] && [ "\$(tty)" = "/dev/tty1" ]; then
    export XDG_SESSION_TYPE=wayland
    exec labwc
fi
$BLOCK_END
EOF
fi

# 4) The app launch itself: wrapper, labwc autostart line and (with
#    --use-sddm) SDDM autologin, exactly as on the Pi.
info "Running setup_vi_kiosk_wayland.sh..."
echo "----"
SUDO_USER="$TARGET_USER" "$WAYLAND_SETUP"
echo "----"

# 5) Make sure the app uses GLFW's Wayland build. The pip glfw package ships
#    both and only picks Wayland when PYGLFW_LIBRARY_VARIANT=wayland or
#    XDG_SESSION_TYPE=wayland; a console login sets XDG_SESSION_TYPE=tty, so
#    without this the app would silently go through Xwayland. labwc passes
#    this file's variables to everything it starts.
info "Setting PYGLFW_LIBRARY_VARIANT=wayland in $LABWC_ENV_FILE..."
as_user mkdir -p "$LABWC_DIR"
[[ -f "$LABWC_ENV_FILE" ]] || as_user touch "$LABWC_ENV_FILE"
remove_block "$LABWC_ENV_FILE"
cat >>"$LABWC_ENV_FILE" <<EOF
$BLOCK_BEGIN
PYGLFW_LIBRARY_VARIANT=wayland
$BLOCK_END
EOF

chown -R "$TARGET_USER":"$TARGET_USER" "$LABWC_DIR"

# 6) Hand back anything root owns in the checkout or the user's config (from
#    this run, or from an earlier `sudo` run of the setup scripts): a
#    root-owned .venv, .git object or __pycache__ breaks pip, git and the app.
info "Fixing ownership of root-owned files in $PROJECT_DIR and ~/.config..."
find "$PROJECT_DIR" "$TARGET_HOME/.config" -xdev -user 0 \
  -exec chown -h "$TARGET_USER:$(id -gn "$TARGET_USER")" {} + 2>/dev/null || true

systemctl daemon-reload

echo
info "Done. Reboot to start the kiosk:"
echo "    sudo reboot"
echo
echo "Then, over SSH:"
echo "    tail -f /var/log/videointerleaving.log"
echo
echo "Ctrl+Alt+F2 on the device gives a normal console. To undo everything:"
echo "    sudo $0 --revert"
