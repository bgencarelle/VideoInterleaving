#!/bin/bash
set -e

# --- CONFIGURATION ---
PROJECT_DIR=$(pwd)
VENV_DIR="$PROJECT_DIR/.venv"

# Detect real user if running via sudo, otherwise default to current user
if [ "$SUDO_USER" ]; then
    USERNAME="$SUDO_USER"
else
    USERNAME=$(whoami)
fi

echo ">>> 🛰️  Starting Application Setup..."
echo "    Running as User: $USERNAME"
echo "    Project Dir:     $PROJECT_DIR"

# --------------------------------------------
# 1. Hardware Detection
# --------------------------------------------
IS_PI=false
HAS_GPU=false
if [ -d "/dev/dri" ]; then HAS_GPU=true; fi
if grep -q "Raspberry Pi" /proc/device-tree/model 2>/dev/null; then IS_PI=true; HAS_GPU=true; fi

echo "    Hardware: GPU=$HAS_GPU | Pi=$IS_PI"

# --------------------------------------------
# 2. Dependencies
# --------------------------------------------
echo ">>> 📦 Installing system libraries..."

# Check if system-requirements.txt exists
if [ ! -f "$PROJECT_DIR/system-requirements.txt" ]; then
    echo "❌ ERROR: system-requirements.txt not found in $PROJECT_DIR"
    exit 1
fi

# Read system packages (skip comments and empty lines)
SYSTEM_PACKAGES=$(grep -v '^#' "$PROJECT_DIR/system-requirements.txt" | grep -v '^$' | tr '\n' ' ')

if [ -z "$SYSTEM_PACKAGES" ]; then
    echo "❌ ERROR: No packages found in system-requirements.txt"
    exit 1
fi

sudo apt update -qq
sudo apt install -y $SYSTEM_PACKAGES

if [ "$HAS_GPU" = true ] && [ "$USERNAME" != "root" ]; then
    sudo usermod -aG video,render "$USERNAME" || true
fi

# --------------------------------------------
# 3. Python Environment
# --------------------------------------------
echo ">>> 🐍 Setting up Python venv..."

# Check if requirements.txt exists
if [ ! -f "$PROJECT_DIR/requirements.txt" ]; then
    echo "❌ ERROR: requirements.txt not found in $PROJECT_DIR"
    exit 1
fi

# Create venv with system-site-packages enabled (allows access to system-installed Python packages)
# Create venv with ownership of the real user, not root
if [ -d "$VENV_DIR" ]; then rm -rf "$VENV_DIR"; fi
sudo -u "$USERNAME" python3 -m venv --system-site-packages "$VENV_DIR"

# Install requirements inside the venv
# We use full path to pip to ensure we use the venv's pip
sudo -u "$USERNAME" "$VENV_DIR/bin/pip" install --upgrade pip wheel
sudo -u "$USERNAME" "$VENV_DIR/bin/pip" install -r "$PROJECT_DIR/requirements.txt"

# --------------------------------------------
# 4. Logrotate
# --------------------------------------------
echo ">>> 📜 Configuring Logrotate..."
sudo cat <<EOF > "/etc/logrotate.d/videointerleaving"
$PROJECT_DIR/*.log {
    daily
    rotate 5
    compress
    missingok
    copytruncate
    size 10M
}
EOF

# --------------------------------------------
# 5. Display Detection Functions
# --------------------------------------------

detect_display() {
    # Check Wayland
    if [ -n "$WAYLAND_DISPLAY" ] || [ "$XDG_SESSION_TYPE" = "wayland" ]; then
        echo "wayland"
        return
    fi
    
    # Check X11
    if [ -n "$DISPLAY" ]; then
        echo "x11"
        return
    fi
    
    # Check for X11 socket
    if [ -S "/tmp/.X11-unix/X0" ]; then
        echo "x11"
        return
    fi
    
    # Check for other X11 sockets
    for socket in /tmp/.X11-unix/X*; do
        if [ -S "$socket" ]; then
            echo "x11"
            return
        fi
    done
    
    # Check framebuffer (Pi)
    if [ -c "/dev/fb0" ]; then
        echo "framebuffer"
        return
    fi
    
    # Default fallback
    echo "x11"
}

get_user_display() {
    local user=$1
    local display=""
    
    # Try to get DISPLAY from systemd user session
    if command -v loginctl >/dev/null 2>&1; then
        local session=$(loginctl list-sessions --user="$user" --no-legend 2>/dev/null | head -n1 | awk '{print $1}')
        if [ -n "$session" ]; then
            display=$(loginctl show-session "$session" -p Display --value 2>/dev/null || echo "")
        fi
    fi
    
    # Fallback: check for X11 sockets
    if [ -z "$display" ]; then
        if [ -S "/tmp/.X11-unix/X0" ]; then
            display=":0"
        else
            # Find first available X socket
            for socket in /tmp/.X11-unix/X*; do
                if [ -S "$socket" ]; then
                    local num=$(basename "$socket" | sed 's/X//')
                    display=":$num"
                    break
                fi
            done
        fi
    fi
    
    # Final fallback
    if [ -z "$display" ]; then
        display=":0"
    fi
    
    echo "$display"
}

get_xauthority() {
    local user=$1
    local xauth_path=""
    
    # Try user's home directory
    local home_dir=$(eval echo ~$user)
    if [ -f "$home_dir/.Xauthority" ]; then
        xauth_path="$home_dir/.Xauthority"
    fi
    
    # Try systemd session
    if [ -z "$xauth_path" ] && command -v loginctl >/dev/null 2>&1; then
        local session=$(loginctl list-sessions --user="$user" --no-legend 2>/dev/null | head -n1 | awk '{print $1}')
        if [ -n "$session" ]; then
            xauth_path=$(loginctl show-session "$session" -p XAuthority --value 2>/dev/null || echo "")
        fi
    fi
    
    echo "$xauth_path"
}

# --------------------------------------------
# 6. Systemd Services
# --------------------------------------------
echo ">>> ⚙️  Creating Systemd Services..."

if [ "$IS_PI" = true ]; then
    ENV_BLOCK="Environment=MESA_GL_VERSION_OVERRIDE=3.3\nEnvironment=MESA_GLSL_VERSION_OVERRIDE=330"
elif [ "$HAS_GPU" = true ]; then
    ENV_BLOCK="# Generic GPU Auto-detect"
else
    ENV_BLOCK="Environment=GALLIUM_DRIVER=llvmpipe"
fi

# Detect display environment
DISPLAY_TYPE=$(detect_display)
DISPLAY_VAR=$(get_user_display "$USERNAME")
XAUTH_PATH=$(get_xauthority "$USERNAME")

# Build display environment block based on detected type
if [ "$DISPLAY_TYPE" = "wayland" ]; then
    WAYLAND_DISPLAY_VAR=${WAYLAND_DISPLAY:-wayland-0}
    USER_UID=$(id -u "$USERNAME")
    ENV_DISPLAY="Environment=WAYLAND_DISPLAY=$WAYLAND_DISPLAY_VAR
Environment=XDG_RUNTIME_DIR=/run/user/$USER_UID"
elif [ "$DISPLAY_TYPE" = "x11" ]; then
    ENV_DISPLAY="Environment=DISPLAY=$DISPLAY_VAR"
    if [ -n "$XAUTH_PATH" ]; then
        ENV_DISPLAY="$ENV_DISPLAY
Environment=XAUTHORITY=$XAUTH_PATH"
    fi
else
    ENV_DISPLAY="# Framebuffer mode - no DISPLAY needed"
fi

# --- SERVICE 1: WEB MODE (vi-web) ---
sudo cat <<EOF > "/etc/systemd/system/vi-web.service"
[Unit]
Description=VideoInterleaving (Web Stream)
After=network.target

[Service]
User=$USERNAME
WorkingDirectory=$PROJECT_DIR
Environment=PYTHONUNBUFFERED=1
$ENV_BLOCK
ExecStart=$VENV_DIR/bin/python -O main.py --mode web
Restart=always
RestartSec=3
StandardOutput=append:$PROJECT_DIR/vi-web.log
StandardError=append:$PROJECT_DIR/vi-web.log

[Install]
WantedBy=multi-user.target
EOF

# --- SERVICE 2: ASCII MODE (vi-ascii) ---
sudo cat <<EOF > "/etc/systemd/system/vi-ascii.service"
[Unit]
Description=VideoInterleaving (ASCII Telnet)
After=network.target

[Service]
User=$USERNAME
WorkingDirectory=$PROJECT_DIR
Environment=PYTHONUNBUFFERED=1
$ENV_BLOCK
ExecStart=$VENV_DIR/bin/python -O main.py --mode ascii
Restart=always
RestartSec=3
StandardOutput=append:$PROJECT_DIR/vi-ascii.log
StandardError=append:$PROJECT_DIR/vi-ascii.log

[Install]
WantedBy=multi-user.target
EOF

# --- SERVICE 3: LOCAL MODE (vi-local) ---
sudo cat <<EOF > "/etc/systemd/system/vi-local.service"
[Unit]
Description=VideoInterleaving (Local GUI)
After=network.target graphical.target

[Service]
User=$USERNAME
WorkingDirectory=$PROJECT_DIR
Environment=PYTHONUNBUFFERED=1
$ENV_DISPLAY
$ENV_BLOCK
ExecStart=$VENV_DIR/bin/python -O main.py --mode local
Restart=always
RestartSec=3
StandardOutput=append:$PROJECT_DIR/vi-local.log
StandardError=append:$PROJECT_DIR/vi-local.log

[Install]
WantedBy=graphical.target
EOF

sudo systemctl daemon-reload

# --------------------------------------------
# 7. Firewall
# --------------------------------------------
if command -v ufw >/dev/null; then
    sudo ufw allow 2323/tcp >/dev/null 2>&1
fi

echo "----------------------------------------------------"
echo "✅ App Setup Complete."
echo ""
echo "👉 Web Mode:   systemctl enable --now vi-web"
echo "👉 ASCII Mode: systemctl enable --now vi-ascii"
echo "👉 Local Mode: systemctl enable --now vi-local"
echo "----------------------------------------------------"