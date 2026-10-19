#!/usr/bin/env bash
set -euo pipefail

# Kiosk setup for plain Debian machines (e.g. a tablet that came with Plasma):
# the Debian counterpart of the Raspberry Pi setup in setup_vi_kiosk_wayland.sh.
#
# Default (no display manager): boot to the console, log the user in on tty1
# automatically, start labwc from that login, and let labwc's autostart run the
# app. Nothing else is started, which is as light as it gets.
#
# Power: performance profile, never sleeps (also on battery), and on battery
# warns at 10% and shuts down cleanly at 5%.
#
#   sudo scripts/setup_vi_kiosk_debian.sh              # set up (no SDDM)
#   sudo scripts/setup_vi_kiosk_debian.sh --use-sddm   # keep SDDM, autologin into labwc
#   sudo scripts/setup_vi_kiosk_debian.sh --with-deps  # also run setup_app.sh first
#   sudo scripts/setup_vi_kiosk_debian.sh --keep-power # leave power settings alone
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
  sed -n '4,19p' "$0" | sed 's/^# \{0,1\}//'
  exit "${1:-0}"
}

USE_SDDM=false
WITH_DEPS=false
REVERT=false
FORCE=false
KEEP_POWER=false
for arg in "$@"; do
  case "$arg" in
    --use-sddm) USE_SDDM=true ;;
    --with-deps) WITH_DEPS=true ;;
    --revert) REVERT=true ;;
    --force) FORCE=true ;;
    --keep-power) KEEP_POWER=true ;;
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
LABWC_RC="$LABWC_DIR/rc.xml"
GETTY_OVERRIDE="/etc/systemd/system/getty@tty1.service.d/zz-videointerleaving-autologin.conf"
SDDM_AUTOLOGIN_CONF="/etc/sddm.conf.d/zz-videointerleaving-autologin.conf"
# Remembers which display manager we switched off so --revert can restore it.
STATE_FILE="/etc/videointerleaving-kiosk.state"

# Power settings (step 7). Battery thresholds are read by the guard script and
# can be changed in it after install.
LOGIND_CONF="/etc/systemd/logind.conf.d/zz-videointerleaving-kiosk.conf"
SLEEP_TARGETS=(sleep.target suspend.target hibernate.target hybrid-sleep.target suspend-then-hibernate.target)
GUARD_SCRIPT="/usr/local/sbin/vi-battery-guard"
GUARD_UNIT="vi-battery-guard"
PERF_UNIT="vi-cpu-performance"
BATTERY_WARN_PERCENT=10
BATTERY_SHUTDOWN_PERCENT=5

BLOCK_BEGIN="# >>> VideoInterleaving kiosk >>>"
BLOCK_END="# <<< VideoInterleaving kiosk <<<"

as_user() { sudo -u "$TARGET_USER" -H "$@"; }

# key=value memory of what was there before, for --revert. A key is only
# written once, so a rerun never overwrites the original value with ours.
state_get() { [[ -f "$STATE_FILE" ]] && sed -n "s/^$1=//p" "$STATE_FILE" | head -n1; return 0; }
state_set_once() {
  grep -q "^$1=" "$STATE_FILE" 2>/dev/null || echo "$1=$2" >>"$STATE_FILE"
}

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

# On Wayland the app opens a borderless, screen-sized window titled
# "Fullscreen" rather than asking for fullscreen (display_manager.py), and
# labwc defaults to server-side decorations, so it would draw a title bar
# saying "Fullscreen". This rule removes the decoration and makes the window
# truly fullscreen. The block is fenced with comments so reruns replace it and
# --revert removes it; an existing rc.xml is merged into, never replaced.
# Usage: labwc_rule add|remove
labwc_rule() {
  python3 - "$1" "$LABWC_RC" <<'PY'
import os, re, sys
mode, path = sys.argv[1], sys.argv[2]
BEGIN, END = "<!-- >>> VideoInterleaving kiosk >>> -->", "<!-- <<< VideoInterleaving kiosk <<< -->"
RULE = ('    <windowRule title="Fullscreen" serverDecoration="no">\n'
        '      <action name="ToggleFullscreen"/>\n'
        '    </windowRule>\n')
fence = re.compile(r"[ \t]*" + re.escape(BEGIN) + r".*?" + re.escape(END) + r"[ \t]*\n?", re.S)
text = open(path).read() if os.path.exists(path) else None
if mode == "remove":
    if text is not None and BEGIN in text:
        open(path, "w").write(fence.sub("", text))
    sys.exit(0)
if text is None:
    open(path, "w").write('<?xml version="1.0"?>\n<labwc_config>\n  <windowRules>\n'
                          f"  {BEGIN}\n{RULE}  {END}\n  </windowRules>\n</labwc_config>\n")
    print("created"); sys.exit(0)
text = fence.sub("", text)
# A hand-added copy of the rule must not be doubled: two ToggleFullscreen
# actions would switch fullscreen on and straight back off.
if re.search(r'<windowRule\b[^>]*title="Fullscreen"', text, re.I):
    print("present"); sys.exit(0)
if re.search(r"<windowRules\s*>", text):
    text = re.sub(r"(<windowRules\s*>[ \t]*\n?)", lambda m: m.group(1) + f"  {BEGIN}\n{RULE}  {END}\n", text, count=1)
elif re.search(r"</(labwc|openbox)_config\s*>", text):
    block = f"  {BEGIN}\n  <windowRules>\n{RULE}  </windowRules>\n  {END}\n"
    text = re.sub(r"(</(?:labwc|openbox)_config\s*>)", lambda m: block + m.group(1), text, count=1)
else:
    print("unknown"); sys.exit(0)
open(path, "w").write(text)
print("merged")
PY
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
  labwc_rule remove

  # Put back the Plasma autostart entry if we parked it.
  parked="$TARGET_HOME/.config/autostart/videointerleaving.desktop.disabled-by-kiosk"
  [[ -f "$parked" ]] && info "Plasma autostart entry left parked at $parked (rename it back to re-enable)."

  # Power: sleep allowed again, guard and CPU units gone, profile restored.
  rm -f "$LOGIND_CONF"
  systemctl unmask "${SLEEP_TARGETS[@]}" >/dev/null 2>&1 || true
  for u in "$GUARD_UNIT.timer" "$GUARD_UNIT.service" "$PERF_UNIT.service"; do
    systemctl disable --now "$u" >/dev/null 2>&1 || true
    rm -f "/etc/systemd/system/$u"
  done
  rm -f "$GUARD_SCRIPT"
  if [[ "$(state_get added_ppd_wants)" == yes ]]; then
    rm -f /etc/systemd/system/multi-user.target.wants/power-profiles-daemon.service
  fi
  prev_profile="$(state_get previous_power_profile)"
  if [[ -n "$prev_profile" ]] && command -v powerprofilesctl >/dev/null 2>&1; then
    powerprofilesctl set "$prev_profile" >/dev/null 2>&1 && info "Power profile restored to $prev_profile."
  fi

  prev_dm="$(state_get previous_dm)"
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
  if [[ -n "$(state_get previous_dm)" ]]; then
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
    state_set_once previous_dm "$DM"
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

# 5b) Make the app's "Fullscreen" window really fullscreen, without a title
#     bar (see labwc_rule above).
case "$(labwc_rule add)" in
  created) info "Created $LABWC_RC with the fullscreen window rule." ;;
  merged)  info "Added the fullscreen window rule to $LABWC_RC." ;;
  present) info "$LABWC_RC already has a rule for the \"Fullscreen\" window; left as is." ;;
  *)       warn "Could not place the fullscreen window rule in $LABWC_RC; add it by hand:"
           warn '  <windowRule title="Fullscreen" serverDecoration="no"><action name="ToggleFullscreen"/></windowRule>' ;;
esac

chown -R "$TARGET_USER":"$TARGET_USER" "$LABWC_DIR"

# 6) Power: full speed, never sleep (on battery too), clean shutdown before
#    the battery runs flat. Without Plasma nothing blanks or dims the screen,
#    so only sleep itself needs blocking.
if $KEEP_POWER; then
  info "Leaving power settings alone (--keep-power)."
else
  info "Configuring power: performance, no sleep, low-battery shutdown..."

  # Sleep is blocked at the source: with the targets masked nothing (lid,
  # idle, keys, a stray `systemctl suspend`) can suspend or hibernate. The
  # logind drop-in keeps the lid and keys from even trying; it takes effect at
  # the next boot.
  systemctl mask "${SLEEP_TARGETS[@]}" >/dev/null
  mkdir -p "$(dirname "$LOGIND_CONF")"
  cat >"$LOGIND_CONF" <<'EOF'
# VideoInterleaving kiosk: never suspend, on battery or not.
[Login]
HandleSuspendKey=ignore
HandleHibernateKey=ignore
HandleLidSwitch=ignore
HandleLidSwitchExternalPower=ignore
HandleLidSwitchDocked=ignore
IdleAction=ignore
EOF

  # Performance profile through power-profiles-daemon when present (what
  # Plasma's power mode slider uses); balanced where the hardware has no
  # performance profile. ppd keeps the choice across reboots, but its unit is
  # normally pulled in by graphical.target, which a console boot never
  # reaches, so it is also made a dependency of multi-user.target.
  if command -v powerprofilesctl >/dev/null 2>&1 && powerprofilesctl get >/dev/null 2>&1; then
    state_set_once previous_power_profile "$(powerprofilesctl get)"
    if powerprofilesctl list 2>/dev/null | grep -q 'performance:'; then
      powerprofilesctl set performance && info "Power profile: performance"
    else
      powerprofilesctl set balanced && info "Power profile: balanced (no performance profile on this hardware)"
    fi
    if ! $USE_SDDM && [[ ! -e /etc/systemd/system/multi-user.target.wants/power-profiles-daemon.service ]]; then
      if systemctl add-wants multi-user.target power-profiles-daemon.service >/dev/null 2>&1; then
        state_set_once added_ppd_wants yes
      fi
    fi
  else
    # No power-profiles-daemon: set the CPU governor directly at every boot.
    cat >"/etc/systemd/system/$PERF_UNIT.service" <<'EOF'
[Unit]
Description=VideoInterleaving kiosk: CPU performance governor

[Service]
Type=oneshot
RemainAfterExit=yes
ExecStart=/bin/sh -c 'for g in /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor; do grep -qw performance "$${g%%/*}/scaling_available_governors" 2>/dev/null && echo performance >"$$g"; done; for e in /sys/devices/system/cpu/cpu*/cpufreq/energy_performance_preference; do echo performance >"$$e" 2>/dev/null; done; true'

[Install]
WantedBy=multi-user.target
EOF
    systemctl daemon-reload
    systemctl enable --now "$PERF_UNIT.service" >/dev/null 2>&1 || true
    info "power-profiles-daemon not found; CPU governor set to performance at boot instead."
  fi

  # Battery guard: every minute, and only while discharging, warn at
  # BATTERY_WARN_PERCENT and power off cleanly at BATTERY_SHUTDOWN_PERCENT.
  # First check 2 minutes after boot, so a nearly flat tablet can still be
  # reached over SSH (and plugged in) before it shuts itself down.
  cat >"$GUARD_SCRIPT" <<EOF
#!/bin/sh
# VideoInterleaving kiosk battery guard (installed by setup_vi_kiosk_debian.sh).
# Runs every minute from $GUARD_UNIT.timer. Edit the thresholds here.
WARN_PERCENT=$BATTERY_WARN_PERCENT
SHUTDOWN_PERCENT=$BATTERY_SHUTDOWN_PERCENT
EOF
  cat >>"$GUARD_SCRIPT" <<'EOF'
LOG=/var/log/videointerleaving.log
WARNED=/run/vi-battery-guard.warned

say() {
  logger -t vi-battery-guard "$1"
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] BATTERY: $1" >>"$LOG" 2>/dev/null
  wall "VideoInterleaving kiosk: $1" 2>/dev/null
}

# Lowest charge among the device's own batteries that are discharging
# (scope=Device marks a stylus, mouse or other peripheral battery).
lowest=""
for b in /sys/class/power_supply/*; do
  [ "$(cat "$b/type" 2>/dev/null)" = Battery ] || continue
  [ "$(cat "$b/scope" 2>/dev/null)" = Device ] && continue
  [ "$(cat "$b/status" 2>/dev/null)" = Discharging ] || continue
  c="$(cat "$b/capacity" 2>/dev/null)" || continue
  case "$c" in ''|*[!0-9]*) continue ;; esac
  if [ -z "$lowest" ] || [ "$c" -lt "$lowest" ]; then lowest="$c"; fi
done

# On mains power, or no battery: nothing to do; re-arm the warning.
if [ -z "$lowest" ]; then
  rm -f "$WARNED"
  exit 0
fi

if [ "$lowest" -le "$SHUTDOWN_PERCENT" ]; then
  say "battery at ${lowest}% - shutting down now to protect it."
  sleep 5
  systemctl poweroff
elif [ "$lowest" -le "$WARN_PERCENT" ]; then
  if [ ! -e "$WARNED" ]; then
    say "battery at ${lowest}% - plug in power. Shutting down at ${SHUTDOWN_PERCENT}%."
    touch "$WARNED"
  fi
else
  rm -f "$WARNED"
fi
EOF
  chmod 0755 "$GUARD_SCRIPT"

  cat >"/etc/systemd/system/$GUARD_UNIT.service" <<EOF
[Unit]
Description=VideoInterleaving kiosk: warn on low battery, shut down when critical

[Service]
Type=oneshot
ExecStart=$GUARD_SCRIPT
EOF
  cat >"/etc/systemd/system/$GUARD_UNIT.timer" <<EOF
[Unit]
Description=VideoInterleaving kiosk: check the battery every minute

[Timer]
OnBootSec=2min
OnUnitActiveSec=1min
AccuracySec=10s

[Install]
WantedBy=timers.target
EOF
  systemctl daemon-reload
  systemctl enable --now "$GUARD_UNIT.timer" >/dev/null
  info "Battery guard: warning at ${BATTERY_WARN_PERCENT}%, clean shutdown at ${BATTERY_SHUTDOWN_PERCENT}% (on battery only)."
fi

# 7) Hand back anything root owns in the checkout or the user's config (from
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
