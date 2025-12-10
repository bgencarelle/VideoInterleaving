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
sudo apt update -qq
sudo apt install -y python3-venv python3-dev python3-pip build-essential cmake pkg-config \
    libwebp-dev libsdl2-dev libasound2-dev libgl1-mesa-dev libglu1-mesa-dev \
    libegl1-mesa-dev mesa-utils chrony ninja-build python-is-python3 \
    libjpeg-dev

if [ "$HAS_GPU" = true ] && [ "$USERNAME" != "root" ]; then
    sudo usermod -aG video,render "$USERNAME" || true
fi

# --------------------------------------------
# 3. Python Environment
# --------------------------------------------
echo ">>> 🐍 Setting up Python venv..."
# Create venv with ownership of the real user, not root
if [ -d "$VENV_DIR" ]; then rm -rf "$VENV_DIR"; fi
sudo -u "$USERNAME" python3 -m venv "$VENV_DIR"

# Install requirements inside the venv
# We use full path to pip to ensure we use the venv's pip
sudo -u "$USERNAME" "$VENV_DIR/bin/pip" install --upgrade pip wheel
sudo -u "$USERNAME" "$VENV_DIR/bin/pip" install -r requirements.txt

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
# 5. Systemd Services
# --------------------------------------------
echo ">>> ⚙️  Creating Systemd Services..."

if [ "$IS_PI" = true ]; then
    ENV_BLOCK="Environment=MESA_GL_VERSION_OVERRIDE=3.3\nEnvironment=MESA_GLSL_VERSION_OVERRIDE=330"
elif [ "$HAS_GPU" = true ]; then
    ENV_BLOCK="# Generic GPU Auto-detect"
else
    ENV_BLOCK="Environment=GALLIUM_DRIVER=llvmpipe"
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
Environment=DISPLAY=:0
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
# 6. Firewall
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