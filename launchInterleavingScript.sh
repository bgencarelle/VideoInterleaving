#!/bin/bash
set -e

# --- CONFIGURATION ---
PROJECT_DIR=$(pwd)
VENV_DIR="$PROJECT_DIR/.venv"
SERVICE_NAME="videointerleaving"
USERNAME="root"

echo ">>> 🛰️  Starting Universal Headless Setup..."

# --------------------------------------------
# 0. SELECT INSTANCE MODE
# --------------------------------------------
echo "Select Instance Mode:"
echo "  1) Web Stream (MJPEG)"
echo "  2) ASCII Stream (Telnet)"
echo "  3) Local / Default"
read -p "Enter choice [1-3]: " MODE_CHOICE

# Base Port from typical settings (change if your settings.py differs)
BASE_PORT=1978

case $MODE_CHOICE in
    1)
        INSTANCE_MODE="web"
        PY_ARGS="--mode web"
        # Web stays on Default Base Port
        MONITOR_PORT=$BASE_PORT
        ;;
    2)
        INSTANCE_MODE="ascii"
        PY_ARGS="--mode ascii"
        # ASCII gets +2
        MONITOR_PORT=$((BASE_PORT + 2))
        ;;
    *)
        INSTANCE_MODE="local"
        PY_ARGS="--mode local"
        # Local gets +1
        MONITOR_PORT=$((BASE_PORT + 1))
        ;;
esac

echo ">>> Selected Mode: $INSTANCE_MODE"
echo ">>> Monitor Port:  $MONITOR_PORT"

# --------------------------------------------
# 1. Hardware Detection
# --------------------------------------------
IS_PI=false
HAS_GPU=false
if [ -d "/dev/dri" ]; then HAS_GPU=true; fi
if grep -q "Raspberry Pi" /proc/device-tree/model 2>/dev/null; then IS_PI=true; HAS_GPU=true; fi

# --------------------------------------------
# 2. Dependencies
# --------------------------------------------
echo ">>> 📦 Installing system dependencies..."
apt-get update
apt-get install -y python3-venv python3-dev python3-pip build-essential \
    libwebp-dev libgl1-mesa-dev libglu1-mesa-dev libegl1-mesa-dev mesa-utils \
    chrony ninja-build python-is-python3 nginx ufw libjpeg-dev

if [ "$HAS_GPU" = true ] && [ "$USERNAME" != "root" ]; then
    usermod -aG video,render "$USERNAME" || true
fi

# --------------------------------------------
# 3. Python Env
# --------------------------------------------
echo ">>> 🐍 Setting up Python environment..."
if [ -d "$VENV_DIR" ]; then rm -rf "$VENV_DIR"; fi
python3 -m venv "$VENV_DIR"
source "$VENV_DIR/bin/activate"
pip install --upgrade pip wheel
pip install -r requirements.txt

# --------------------------------------------
# 4. Generate Lists
# --------------------------------------------
echo ">>> 📂 Generating file lists..."
"$VENV_DIR/bin/python" make_file_lists.py

# --------------------------------------------
# 5. Systemd Service
# --------------------------------------------
echo ">>> ⚙️  Creating Systemd Service ($SERVICE_NAME)..."
SERVICE_FILE="/etc/systemd/system/${SERVICE_NAME}.service"

if [ "$IS_PI" = true ]; then
    ENV_BLOCK="Environment=MESA_GL_VERSION_OVERRIDE=3.3\nEnvironment=MESA_GLSL_VERSION_OVERRIDE=330"
elif [ "$HAS_GPU" = true ]; then
    ENV_BLOCK="# Generic GPU Auto-detect"
else
    ENV_BLOCK="Environment=GALLIUM_DRIVER=llvmpipe"
fi

cat <<EOF > "$SERVICE_FILE"
[Unit]
Description=VideoInterleaving ($INSTANCE_MODE)
After=network.target

[Service]
User=$USERNAME
WorkingDirectory=$PROJECT_DIR
Environment=PYTHONUNBUFFERED=1
$ENV_BLOCK
ExecStart=$VENV_DIR/bin/python -O main.py $PY_ARGS
Restart=always
RestartSec=3
StandardOutput=append:$PROJECT_DIR/systemd_out.log
StandardError=append:$PROJECT_DIR/systemd_err.log

[Install]
WantedBy=multi-user.target
EOF

systemctl daemon-reload
systemctl enable "$SERVICE_NAME"

# --------------------------------------------
# 6. Logrotate
# --------------------------------------------
LOGROTATE_FILE="/etc/logrotate.d/$SERVICE_NAME"
cat <<EOF > "$LOGROTATE_FILE"
$PROJECT_DIR/runtime.log {
    daily
    rotate 7
    compress
    missingok
    notifempty
    copytruncate
    size 10M
}
EOF

# --------------------------------------------
# 7. Nginx Proxy (Dynamic Port)
# --------------------------------------------
echo ">>> 🌐 Configuring Nginx..."
NGINX_CONF="/etc/nginx/sites-available/$SERVICE_NAME"

# Note: We configure the video stream logic conditionally.
# If in ASCII mode, proxying / to 8080 is pointless, but we leave it
# standard for simplicity (it just won't connect).
# The important part is the MONITOR port matching the python script.

cat <<EOF > "$NGINX_CONF"
server {
    listen 80 default_server;
    listen [::]:80 default_server;
    server_name _;

    location = /monitor { return 301 /monitor/; }

    # Main Stream (Only active if Web Mode)
    location / {
        proxy_pass http://127.0.0.1:8080;
        proxy_http_version 1.1;
        proxy_set_header Connection "";
        proxy_buffering off;
        proxy_cache off;
        proxy_read_timeout 7d;
    }

    # Monitor Dashboard (Dynamically Assigned Port)
    location /monitor/ {
        proxy_pass http://127.0.0.1:$MONITOR_PORT/;
        proxy_http_version 1.1;
        proxy_set_header Connection "";
        proxy_buffering off;
        proxy_cache off;
    }
}
EOF

rm -f /etc/nginx/sites-enabled/default
ln -sf "$NGINX_CONF" "/etc/nginx/sites-enabled/$SERVICE_NAME"
systemctl reload nginx || echo "Warning: Nginx reload failed."

# --------------------------------------------
# 8. Firewall
# --------------------------------------------
if command -v ufw >/dev/null; then
    echo ">>> 🛡️  Updating Firewall..."
    ufw allow 80/tcp >/dev/null 2>&1
    # If ASCII mode, open Telnet port
    if [ "$INSTANCE_MODE" == "ascii" ]; then
        echo "    Opening Port 2323 for Telnet..."
        ufw allow 2323/tcp >/dev/null 2>&1
    fi
fi

echo "----------------------------------------------------"
echo "✅ Setup Complete for [$INSTANCE_MODE] mode!"
echo "   Monitor Port: $MONITOR_PORT"
echo "   Start: systemctl start $SERVICE_NAME"
echo "----------------------------------------------------"