#!/bin/bash
set -e

# --- CONFIGURATION ---
PROJECT_DIR=$(pwd)
VENV_DIR="$PROJECT_DIR/.venv"
USERNAME="root" # Change this if you aren't running as root user

echo ">>> 🛰️  Starting Application Setup (Hardware & Services)..."

# --------------------------------------------
# 1. Hardware Detection
# --------------------------------------------
IS_PI=false
HAS_GPU=false
if [ -d "/dev/dri" ]; then HAS_GPU=true; fi
if grep -q "Raspberry Pi" /proc/device-tree/model 2>/dev/null; then IS_PI=true; HAS_GPU=true; fi

echo "    Hardware Detected: GPU=$HAS_GPU | Pi=$IS_PI"

# --------------------------------------------
# 2. System Dependencies
# --------------------------------------------
echo ">>> 📦 Installing system libraries..."
apt-get update -qq
apt-get install -y \
    python3-venv python3-dev python3-pip build-essential \
    libwebp-dev libgl1-mesa-dev libglu1-mesa-dev libegl1-mesa-dev \
    libgbm-dev mesa-utils \
    chrony ninja-build python-is-python3 libjpeg-dev ufw

if [ "$HAS_GPU" = true ] && [ "$USERNAME" != "root" ]; then
    usermod -aG video,render "$USERNAME" || true
fi

# --------------------------------------------
# 3. Python Environment
# --------------------------------------------
echo ">>> 🐍 Setting up Python venv..."
if [ -d "$VENV_DIR" ]; then rm -rf "$VENV_DIR"; fi
python3 -m venv "$VENV_DIR"
source "$VENV_DIR/bin/activate"
pip install --upgrade pip wheel
pip install -r requirements.txt

# --------------------------------------------
# 4. Generate Lists & Logrotate
# --------------------------------------------
echo ">>> 📂 Generating file lists..."
"$VENV_DIR/bin/python" make_file_lists.py

echo ">>> 📜 Configuring Logrotate..."
cat <<EOF > "/etc/logrotate.d/videointerleaving"
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

# Determine Environment Variables based on Hardware
if [ "$IS_PI" = true ]; then
    ENV_BLOCK="Environment=MESA_GL_VERSION_OVERRIDE=3.3\nEnvironment=MESA_GLSL_VERSION_OVERRIDE=330"
elif [ "$HAS_GPU" = true ]; then
    ENV_BLOCK="# Generic GPU Auto-detect"
else
    ENV_BLOCK="Environment=GALLIUM_DRIVER=llvmpipe"
fi

# --- SERVICE 1: WEB MODE (vi-web) ---
cat <<EOF > "/etc/systemd/system/vi-web.service"
[Unit]
Description=VideoInterleaving (Web Stream)
After=network.target

[Service]
User=$USERNAME
WorkingDirectory=$PROJECT_DIR
Environment=PYTHONUNBUFFERED=1
$ENV_BLOCK
# Web Mode (Port 8080 Stream, 1978 Monitor)
ExecStart=$VENV_DIR/bin/python -O main.py --mode web
Restart=always
RestartSec=3
StandardOutput=append:$PROJECT_DIR/vi-web.log
StandardError=append:$PROJECT_DIR/vi-web.log

[Install]
WantedBy=multi-user.target
EOF

# --- SERVICE 2: ASCII MODE (vi-ascii) ---
cat <<EOF > "/etc/systemd/system/vi-ascii.service"
[Unit]
Description=VideoInterleaving (ASCII Telnet)
After=network.target

[Service]
User=$USERNAME
WorkingDirectory=$PROJECT_DIR
Environment=PYTHONUNBUFFERED=1
$ENV_BLOCK
# ASCII Mode (Port 2323 Telnet, 1980 Monitor)
ExecStart=$VENV_DIR/bin/python -O main.py --mode ascii
Restart=always
RestartSec=3
StandardOutput=append:$PROJECT_DIR/vi-ascii.log
StandardError=append:$PROJECT_DIR/vi-ascii.log

[Install]
WantedBy=multi-user.target
EOF

systemctl daemon-reload

# --------------------------------------------
# 6. Basic Firewall (App Ports Only)
# --------------------------------------------
# We open 2323 here because it is a direct TCP port, not HTTP/Nginx
if command -v ufw >/dev/null; then
    ufw allow 2323/tcp >/dev/null 2>&1
fi

echo "----------------------------------------------------"
echo "✅ App Setup Complete."
echo "   Run './setup_nginx.sh' next to configure web access."
echo "----------------------------------------------------"