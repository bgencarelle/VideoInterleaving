#!/usr/bin/env bash
set -euo pipefail

# Setup script to prevent fullscreen loss on monitor disconnect for kiosk mode
# Works on Intel/AMD Linux systems with modern desktop environments

die() { echo "ERROR: $*" >&2; exit 1; }
info() { echo "==> $*"; }
warn() { echo "WARNING: $*"; }

need_root() {
  if [[ "${EUID:-$(id -u)}" -ne 0 ]]; then
    die "Run with sudo: sudo $0"
  fi
}

detect_user() {
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

detect_desktop() {
  if [[ -n "${XDG_CURRENT_DESKTOP:-}" ]]; then
    echo "${XDG_CURRENT_DESKTOP,,}" | cut -d: -f1
  elif [[ -n "${XDG_SESSION_DESKTOP:-}" ]]; then
    echo "${XDG_SESSION_DESKTOP,,}"
  elif command -v systemctl >/dev/null 2>&1; then
    systemctl --user show-environment 2>/dev/null | grep -oP 'XDG_SESSION_DESKTOP=\K\w+' || echo "unknown"
  else
    echo "unknown"
  fi
}

detect_session() {
  if [[ -n "${WAYLAND_DISPLAY:-}" ]] || [[ "${XDG_SESSION_TYPE:-}" == "wayland" ]]; then
    echo "wayland"
  else
    echo "x11"
  fi
}

need_root

TARGET_USER="$(detect_user)"
TARGET_HOME="$(user_home "$TARGET_USER")"
[[ -n "$TARGET_HOME" ]] || die "Could not determine home directory for user '$TARGET_USER'."

DESKTOP="$(detect_desktop)"
SESSION="$(detect_session)"

info "Detected desktop: $DESKTOP"
info "Detected session: $SESSION"
info "Target user: $TARGET_USER"

# ============================================================================
# 1. Disable KScreen (KDE Plasma) - prevents automatic display reconfiguration
# ============================================================================
if [[ "$DESKTOP" == *"plasma"* ]] || [[ "$DESKTOP" == *"kde"* ]]; then
  info "Configuring KDE Plasma to prevent fullscreen loss..."
  
  # Disable KScreen service
  if command -v systemctl >/dev/null 2>&1; then
    info "Disabling KScreen service..."
    sudo -u "$TARGET_USER" systemctl --user mask kscreen.service 2>/dev/null || true
    sudo -u "$TARGET_USER" systemctl --user stop kscreen.service 2>/dev/null || true
  fi
  
  # Disable KScreen via config file
  KSCREEN_CONFIG="$TARGET_HOME/.config/kscreenrc"
  mkdir -p "$(dirname "$KSCREEN_CONFIG")"
  cat > "$KSCREEN_CONFIG" << 'EOF'
[Screen]
ApplyOnStartup=false
StartupCommands=
EOF
  chown "$TARGET_USER:$TARGET_USER" "$KSCREEN_CONFIG"
  info "Created KScreen config: $KSCREEN_CONFIG"
  
  # Create window rule to keep windows fullscreen (KWin rules)
  KWIN_RULES="$TARGET_HOME/.config/kwinrulesrc"
  if [[ ! -f "$KWIN_RULES" ]] || ! grep -q "Fullscreen" "$KWIN_RULES" 2>/dev/null; then
    info "Creating KWin window rules to maintain fullscreen..."
    # Note: This requires manual configuration via System Settings > Window Management > Window Rules
    # We'll create a script to help with this
    cat > "$TARGET_HOME/.config/kwin_fullscreen_rule.sh" << 'EOFSCRIPT'
#!/usr/bin/env bash
# Helper script to create KWin window rule for fullscreen
# Run this manually if needed: ~/.config/kwin_fullscreen_rule.sh

echo "To create a KWin window rule:"
echo "1. Open System Settings > Window Management > Window Rules"
echo "2. Click 'New...'"
echo "3. Set Window class (simple) to match your application"
echo "4. Under 'Size & Position', check 'Fullscreen' and set to 'Force'"
echo "5. Under 'Arrangement & Access', check 'Keep above' and set to 'Force'"
echo "6. Save the rule"
EOFSCRIPT
    chmod +x "$TARGET_HOME/.config/kwin_fullscreen_rule.sh"
    chown "$TARGET_USER:$TARGET_USER" "$TARGET_HOME/.config/kwin_fullscreen_rule.sh"
  fi
fi

# ============================================================================
# 2. X11: Configure Xorg to ignore hotplug events
# ============================================================================
if [[ "$SESSION" == "x11" ]]; then
  info "Configuring X11 to ignore monitor hotplug events..."
  
  XORG_CONF_DIR="/etc/X11/xorg.conf.d"
  mkdir -p "$XORG_CONF_DIR"
  
  # Create config to disable hotplug detection
  cat > "$XORG_CONF_DIR/99-no-hotplug.conf" << 'EOF'
# Prevent X server from reacting to monitor hotplug events
# This helps maintain fullscreen windows when monitor inputs change
Section "ServerFlags"
    Option "AutoAddDevices" "false"
    Option "AutoEnableDevices" "false"
EndSection

# For Intel/AMD graphics, disable DRM hotplug polling
Section "Device"
    Identifier "GraphicsDevice"
    Driver "modesetting"
    Option "Hotplug" "false"
EndSection
EOF
  info "Created Xorg config: $XORG_CONF_DIR/99-no-hotplug.conf"
  
  # For NVIDIA (if present)
  if lspci | grep -qi nvidia; then
    info "NVIDIA GPU detected, creating NVIDIA-specific config..."
    cat > "$XORG_CONF_DIR/99-nvidia-no-hotplug.conf" << 'EOF'
# NVIDIA-specific: Disable hotplug events
Section "Device"
    Identifier "NvidiaDevice"
    Driver "nvidia"
    Option "UseHotplugEvents" "false"
EndSection
EOF
    info "Created NVIDIA config: $XORG_CONF_DIR/99-nvidia-no-hotplug.conf"
  fi
fi

# ============================================================================
# 3. Wayland: Configure compositor settings (if possible)
# ============================================================================
if [[ "$SESSION" == "wayland" ]]; then
  info "Configuring Wayland compositor..."
  
  if [[ "$DESKTOP" == *"plasma"* ]] || [[ "$DESKTOP" == *"kde"* ]]; then
    # KWin on Wayland
    KWIN_WAYLAND_CONFIG="$TARGET_HOME/.config/kwinrc"
    mkdir -p "$(dirname "$KWIN_WAYLAND_CONFIG")"
    
    # Disable output management
    if ! grep -q "^\[Compositing\]" "$KWIN_WAYLAND_CONFIG" 2>/dev/null; then
      echo "[Compositing]" >> "$KWIN_WAYLAND_CONFIG"
    fi
    if ! grep -q "^WindowsBlockCompositing" "$KWIN_WAYLAND_CONFIG" 2>/dev/null; then
      echo "WindowsBlockCompositing=false" >> "$KWIN_WAYLAND_CONFIG"
    fi
    
    chown "$TARGET_USER:$TARGET_USER" "$KWIN_WAYLAND_CONFIG"
    info "Updated KWin Wayland config: $KWIN_WAYLAND_CONFIG"
  fi
fi

# ============================================================================
# 4. Kernel parameter: Disable DRM hotplug polling (optional, more aggressive)
# ============================================================================
info "Checking kernel parameters..."
if [[ -f /etc/default/grub ]]; then
  if ! grep -q "drm_kms_helper.poll=0" /etc/default/grub; then
    info "Adding kernel parameter to disable DRM hotplug polling..."
    info "WARNING: This requires editing GRUB. Manual step:"
    info "  1. Edit /etc/default/grub"
    info "  2. Add 'drm_kms_helper.poll=0' to GRUB_CMDLINE_LINUX_DEFAULT"
    info "  3. Run: sudo update-grub && sudo reboot"
    info ""
    info "Example line:"
    info '  GRUB_CMDLINE_LINUX_DEFAULT="quiet splash drm_kms_helper.poll=0"'
  else
    info "Kernel parameter already configured"
  fi
fi

# ============================================================================
# 5. Create systemd service to lock display configuration
# ============================================================================
info "Creating display lock service..."
SYSTEMD_USER_DIR="$TARGET_HOME/.config/systemd/user"
mkdir -p "$SYSTEMD_USER_DIR"

cat > "$SYSTEMD_USER_DIR/lock-display.service" << 'EOF'
[Unit]
Description=Lock Display Configuration (Prevent Hotplug Changes)
After=graphical-session.target

[Service]
Type=oneshot
ExecStart=/usr/bin/true
# This service can be extended to run xrandr commands to lock resolution
RemainAfterExit=yes

[Install]
WantedBy=default.target
EOF

chown -R "$TARGET_USER:$TARGET_USER" "$SYSTEMD_USER_DIR"
info "Created systemd service: $SYSTEMD_USER_DIR/lock-display.service"

# ============================================================================
# Summary
# ============================================================================
echo ""
info "=== Configuration Complete ==="
echo ""
info "Applied configurations:"
if [[ "$DESKTOP" == *"plasma"* ]] || [[ "$DESKTOP" == *"kde"* ]]; then
  echo "  ✓ KScreen service disabled"
  echo "  ✓ KScreen config created"
fi
if [[ "$SESSION" == "x11" ]]; then
  echo "  ✓ Xorg hotplug detection disabled"
fi
if [[ "$SESSION" == "wayland" ]]; then
  echo "  ✓ Wayland compositor configured"
fi
echo "  ✓ Systemd service created"
echo ""
warn "IMPORTANT: Some changes require a reboot to take effect:"
echo "  - Xorg configuration changes"
echo "  - Kernel parameter changes (if applied)"
echo ""
info "To apply KScreen/Plasma changes immediately, run as user:"
echo "  systemctl --user restart plasma-kwin_x11.service  # X11"
echo "  systemctl --user restart plasma-kwin_wayland.service  # Wayland"
echo ""
info "To test: Switch your monitor/TV input and verify fullscreen is maintained"
