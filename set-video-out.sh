#!/usr/bin/env bash
set -euo pipefail

# --- Configuration ---
# In 2026 (Debian Trixie), the boot partition is strictly mounted at /boot/firmware
CONFIG_FILE="/boot/firmware/config.txt"
CMDLINE_FILE="/boot/firmware/cmdline.txt"
BACKUP_DIR="/boot/firmware"

# KMS Connector Names (Standard DRM naming for Pi 4/5/6)
HDMI_CONN_1="HDMI-A-1"
HDMI_CONN_2="HDMI-A-2"
COMP_CONN="Composite-1"

usage() {
  cat <<'EOF'
Usage:
  sudo ./set-video-out.sh -hdmi
  sudo ./set-video-out.sh -analog

Target: Debian Trixie / Raspberry Pi OS (Bookworm+)
System: KMS/DRM Graphics Stack (Pi 4, 5, 6)

Actions:
  - Backs up config.txt and cmdline.txt
  - HDMI: Forces 720x576@50Hz (576p) on HDMI-A-1 via Kernel arguments.
  - ANALOG: Enables 'enable_tvout', forces PAL standard, and sets Composite-1 mode.
EOF
}

MODE="${1:-}"
case "$MODE" in
  -hdmi|--hdmi)           MODE="hdmi" ;;
  -analog|--analog|-composite|--composite) MODE="analog" ;;
  *) usage; exit 2 ;;
esac

# Root check
if [[ ${EUID:-$(id -u)} -ne 0 ]]; then
  echo "Escalating to root..."
  exec sudo -E "$0" "$@"
fi

# Sanity check paths
if [[ ! -f "$CONFIG_FILE" || ! -f "$CMDLINE_FILE" ]]; then
  echo "ERROR: Could not find boot configuration at /boot/firmware/." >&2
  echo "Ensure you are running this on a Raspberry Pi with modern partition layout." >&2
  exit 1
fi

# --- Backups ---
ts="$(date +%Y%m%d_%H%M%S)"
cp -a "$CONFIG_FILE"  "${CONFIG_FILE}.bak.${ts}"
cp -a "$CMDLINE_FILE" "${CMDLINE_FILE}.bak.${ts}"
echo "Backups created with timestamp: ${ts}"

# --- Step 1: Modify config.txt ---
# We use a marker block to manage firmware flags.
# Note: In KMS mode, we DO NOT set hdmi_group/mode here. We only manage electrical enable/disable.

BEGIN_MARK="# >>> halcyon-video BEGIN"
END_MARK="# <<< halcyon-video END"

tmp_cfg="$(mktemp)"

# Remove old block if exists
awk -v b="$BEGIN_MARK" -v e="$END_MARK" '
  $0==b {skip=1; next}
  $0==e {skip=0; next}
  skip!=1 {print}
' "$CONFIG_FILE" > "$tmp_cfg"

# Append new block
{
  echo "$BEGIN_MARK"
  echo "# Managed by set-video-out.sh (Trixie/KMS)"
  echo "[all]"
  if [[ "$MODE" == "hdmi" ]]; then
    cat <<EOF
# Disable analog PHY to save power/clocks
enable_tvout=0
# Ensure HDMI hotplug is respected or forced if headless
hdmi_force_hotplug=1
EOF
  else
    cat <<EOF
# Enable analog PHY (required for Pi 4/5 composite)
enable_tvout=1
# Ignore HDMI hotplug to prevent bootloader confusion
hdmi_ignore_hotplug=1
EOF
  fi
  echo "$END_MARK"
} >> "$tmp_cfg"

# --- Step 2: Handle dtoverlay=vc4-kms-v3d ---
# For Composite on Pi 4/5, we need the ",composite" parameter on the overlay.
# For HDMI, we ideally remove it to free up resources, though harmless if present.

tmp_cfg2="$(mktemp)"
awk -v mode="$MODE" '
  function rtrim(s) { sub(/[[:space:]]+$/, "", s); return s }

  {
    line=$0
    # Look for the active KMS overlay line (ignoring comments)
    if (line ~ /^[[:space:]]*dtoverlay=vc4-kms-v3d/) {

      # Clean existing composite param if present
      sub(/,composite(=[^,[:space:]]*)?/, "", line)
      sub(/,composite/, "", line) # catch leftovers

      if (mode == "analog") {
        # Append composite enable
        # Note: We append to the end of the params
        if (line ~ /#/) {
            sub(/([[:space:]]*#.*)/, ",composite&", line)
        } else {
            line = line ",composite"
        }
      }
      print line
      next
    }
    print line
  }
' "$tmp_cfg" > "$tmp_cfg2"

mv "$tmp_cfg2" "$CONFIG_FILE"
rm -f "$tmp_cfg"
echo "Updated: $CONFIG_FILE"

# --- Step 3: Modify cmdline.txt ---
# This is where the heavy lifting happens for KMS resolutions.

read -r cmdline_line < "$CMDLINE_FILE" || cmdline_line=""

# Split into array
IFS=' ' read -r -a toks <<< "$cmdline_line"

new_toks=()
for t in "${toks[@]}"; do
  case "$t" in
    # Remove any existing video= or vc4 specific mode arguments
    video=*|vc4.tv_norm=*) ;;
    *) new_toks+=("$t") ;;
  esac
done

if [[ "$MODE" == "hdmi" ]]; then
    # Force HDMI-A-1 to 720x576 @ 50Hz.
    # D = Digital/Enable (Force output even if no cable detected)
    # We apply it to HDMI-A-2 as well just in case cable is swapped.
    new_toks+=("video=${HDMI_CONN_1}:720x576@50D" "video=${HDMI_CONN_2}:720x576@50D")
else
    # PAL Standard
    new_toks+=("vc4.tv_norm=PAL")
    # Force Composite-1 to 720x576 @ 50Hz Interlaced.
    # e = Enable
    # i = Interlaced (Critical for analog TV)
    new_toks+=("video=${COMP_CONN}:720x576@50ie")
fi

# Write back as single line
printf '%s\n' "${new_toks[*]}" > "$CMDLINE_FILE"
echo "Updated: $CMDLINE_FILE"

# --- Conclusion ---
echo "----------------------------------------------------"
echo "Configuration applied for: $MODE (576i/50Hz)"
echo "Target Architecture: KMS/DRM (Pi 4/5/6)"
echo "Please reboot to apply: sudo reboot"
echo "----------------------------------------------------"