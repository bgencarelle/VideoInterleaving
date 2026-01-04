#!/usr/bin/env bash
set -e

# --- Configuration ---
CONFIG_FILE="/boot/firmware/config.txt"
CMDLINE_FILE="/boot/firmware/cmdline.txt"

# --- Mode Selection ---
# Default to -hdmi if no argument is provided
INPUT="${1:--hdmi}"

if [[ "$INPUT" == "-hdmi" || "$INPUT" == "--hdmi" ]]; then
    MODE="hdmi"
    echo ">> Mode selected: HDMI (Safe Mode: 640x480 @ 60Hz)"
elif [[ "$INPUT" == "-analog" || "$INPUT" == "--analog" ]]; then
    MODE="analog"
    echo ">> Mode selected: ANALOG (Composite PAL 720x576)"
else
    echo "Usage: sudo ./set-video-out.sh [-hdmi | -analog]"
    exit 1
fi

if [[ $EUID -ne 0 ]]; then
   echo "!! Must be run as root. Try: sudo $0"
   exit 1
fi

# --- 1. CONFIG.TXT ---
BEGIN_MARK="# >>> halcyon-video BEGIN"
END_MARK="# <<< halcyon-video END"

tmp_cfg="$(mktemp)"

# Copy config.txt excluding our old block
awk -v b="$BEGIN_MARK" -v e="$END_MARK" '
  $0==b {skip=1; next}
  $0==e {skip=0; next}
  skip!=1 {print}
' "$CONFIG_FILE" > "$tmp_cfg"

# Append new block
{
  echo "$BEGIN_MARK"
  if [[ "$MODE" == "hdmi" ]]; then
    echo "enable_tvout=0"
    echo "hdmi_force_hotplug=1"
  else
    echo "enable_tvout=1"
    echo "hdmi_ignore_hotplug=1"
    echo "dtoverlay=vc4-kms-v3d,composite"
  fi
  echo "$END_MARK"
} >> "$tmp_cfg"

mv "$tmp_cfg" "$CONFIG_FILE"
echo ">> Updated config.txt"

# --- 2. CMDLINE.TXT ---
# Clean up old video flags
sed -i 's/ video=HDMI-A-1:[^ ]*//g' "$CMDLINE_FILE"
sed -i 's/ video=HDMI-A-2:[^ ]*//g' "$CMDLINE_FILE"
sed -i 's/ video=Composite-1:[^ ]*//g' "$CMDLINE_FILE"
sed -i 's/ vc4.tv_norm=[^ ]*//g' "$CMDLINE_FILE"

# Read line stripping newline
CURRENT_LINE=$(tr -d '\n' < "$CMDLINE_FILE")

# Append new flags
if [[ "$MODE" == "hdmi" ]]; then
    # FOR HDMI: Force 640x480 @ 60Hz.
    # 'D' forces the digital output on even if headless.
    echo "${CURRENT_LINE} video=HDMI-A-1:640x480@60D video=HDMI-A-2:640x480@60D" > "$CMDLINE_FILE"
else
    # FOR ANALOG: Force PAL 576i (Interlaced)
    echo "${CURRENT_LINE} vc4.tv_norm=PAL video=Composite-1:720x576@50ie" > "$CMDLINE_FILE"
fi

echo ">> Updated cmdline.txt"

# --- 3. REBOOT ---
echo ">> Configuration applied. Rebooting in 3 seconds..."
sleep 3
sudo reboot