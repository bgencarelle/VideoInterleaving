# Kiosk Fullscreen Lock Setup

This guide helps prevent fullscreen windows from losing fullscreen mode when monitor inputs change (e.g., TV input switch) on Linux systems.

## Quick Start

Run the setup script as root:

```bash
sudo ./setup_kiosk_fullscreen_lock.sh
```

This script automatically detects your desktop environment and session type, then applies the appropriate configurations.

## What It Does

### 1. KDE Plasma / KWin
- Disables KScreen service (prevents automatic display reconfiguration)
- Creates KScreen config to prevent auto-adjustment
- Provides instructions for creating KWin window rules

### 2. X11 (Xorg)
- Creates Xorg config to disable hotplug detection
- Prevents X server from reacting to monitor connect/disconnect events
- Supports both Intel/AMD (modesetting) and NVIDIA drivers

### 3. Wayland
- Configures KWin on Wayland to maintain window states
- Disables output management that might reset windows

### 4. Application-Level
- Added GLFW window hints: `RESIZABLE=FALSE` and `FLOATING=FALSE`
- These hints help prevent window managers from resizing fullscreen windows

## Manual Configuration Steps

### KDE Plasma: Create Window Rule

1. Open **System Settings** > **Window Management** > **Window Rules**
2. Click **New...**
3. Set **Window class (simple)** to match your application (e.g., `python3` or your app name)
4. Under **Size & Position**:
   - Check **Fullscreen** → Set to **Force**
5. Under **Arrangement & Access**:
   - Check **Keep above** → Set to **Force**
6. Save the rule

### Kernel Parameter (Optional, More Aggressive)

To completely disable DRM hotplug polling:

1. Edit `/etc/default/grub`
2. Add `drm_kms_helper.poll=0` to `GRUB_CMDLINE_LINUX_DEFAULT`:
   ```bash
   GRUB_CMDLINE_LINUX_DEFAULT="quiet splash drm_kms_helper.poll=0"
   ```
3. Update GRUB and reboot:
   ```bash
   sudo update-grub
   sudo reboot
   ```

**Warning**: This will prevent the system from detecting new monitors until reboot.

## Testing

After applying configurations:

1. **Reboot** (required for Xorg changes)
2. Start your application in fullscreen
3. Switch your monitor/TV input
4. Verify the window remains fullscreen

## Troubleshooting

### Fullscreen Still Lost

1. **Check KScreen is disabled**:
   ```bash
   systemctl --user status kscreen.service
   ```
   Should show "masked" or "inactive"

2. **Verify Xorg config**:
   ```bash
   cat /etc/X11/xorg.conf.d/99-no-hotplug.conf
   ```

3. **Check window manager rules** (KDE):
   - System Settings > Window Management > Window Rules
   - Ensure your app has a rule forcing fullscreen

4. **Try kernel parameter** (more aggressive):
   - See "Kernel Parameter" section above

### Application Not Starting

If the application fails to start after changes:

1. Check Xorg logs: `/var/log/Xorg.0.log`
2. Temporarily rename Xorg config files to test
3. Revert kernel parameters if applied

## Desktop Environment Support

- ✅ **KDE Plasma** (X11 and Wayland)
- ✅ **GNOME** (X11 and Wayland) - partial support
- ✅ **Xfce** (X11)
- ✅ **Generic X11** window managers
- ⚠️ **Wayland compositors** - varies by compositor

## Files Created

- `/etc/X11/xorg.conf.d/99-no-hotplug.conf` - X11 hotplug disable
- `/etc/X11/xorg.conf.d/99-nvidia-no-hotplug.conf` - NVIDIA-specific (if NVIDIA GPU)
- `~/.config/kscreenrc` - KScreen configuration
- `~/.config/systemd/user/lock-display.service` - Systemd service
- `~/.config/kwin_fullscreen_rule.sh` - Helper script for window rules

## Reverting Changes

To revert:

```bash
# Remove Xorg configs
sudo rm /etc/X11/xorg.conf.d/99-*-no-hotplug.conf

# Re-enable KScreen (KDE)
systemctl --user unmask kscreen.service
systemctl --user start kscreen.service

# Remove KScreen config
rm ~/.config/kscreenrc

# Revert kernel parameter (if applied)
# Edit /etc/default/grub and remove drm_kms_helper.poll=0
sudo update-grub
sudo reboot
```

## Additional Notes

- **Hardware Solution**: For maximum reliability, consider using a DisplayPort/HDMI hotplug maintainer device
- **Window Manager Specific**: Some window managers may need additional configuration
- **Testing Required**: Always test in your specific environment before deploying to production
