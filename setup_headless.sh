#!/bin/bash
set -e

# --- CONFIGURATION ---
PROJECT_DIR=$(pwd)
VENV_DIR="$PROJECT_DIR/.venv"
USERNAME="root"

echo ">>> 🛰️  Starting Non-Interactive Universal Setup..."

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
echo ">>> 📦 Installing system dependencies..."
apt-get update
# Added 'libgbm-dev' (Critical for Headless EGL on Pi)
apt-get install -y \
    python3-venv python3-dev python3-pip build-essential \
    libwebp-dev libgl1-mesa-dev libglu1-mesa-dev libegl1-mesa-dev \
    libgbm-dev mesa-utils \
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
echo ">>> 📂 Generating file lists (Default/Local)..."
"$VENV_DIR/bin/python" make_file_lists.py

# --------------------------------------------
# 5. Create Systemd Services (WEB & ASCII)
# --------------------------------------------
echo ">>> ⚙️  Creating Services..."

# Env block logic
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
# 6. Logrotate
# --------------------------------------------
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
# 7. Nginx Proxy (Universal)
# --------------------------------------------
echo ">>> 🌐 Configuring Nginx..."
NGINX_CONF="/etc/nginx/sites-available/videointerleaving"

# We configure Nginx to support BOTH.
# Web Stream -> root
# Web Monitor -> /monitor/
# ASCII Monitor -> /monitor_ascii/

cat <<EOF > "$NGINX_CONF"
server {
    listen 80 default_server;
    listen [::]:80 default_server;
    server_name _;

    # -- WEB MODE (Service: vi-web) --
    location / {
        proxy_pass http://127.0.0.1:8080;
        proxy_http_version 1.1;
        proxy_set_header Connection "";
        proxy_buffering off;
        proxy_cache off;
        proxy_read_timeout 7d;
    }
    location /monitor/ {
        proxy_pass http://127.0.0.1:1978/;
        proxy_http_version 1.1;
        proxy_set_header Connection "";
        proxy_buffering off;
    }

    # -- ASCII MODE (Service: vi-ascii) --
    # Telnet is direct (Port 2323), but we proxy the monitor here
    location /monitor_ascii/ {
        proxy_pass http://127.0.0.1:1980/;
        proxy_http_version 1.1;
        proxy_set_header Connection "";
        proxy_buffering off;
    }
}
EOF

rm -f /etc/nginx/sites-enabled/default
ln -sf "$NGINX_CONF" "/etc/nginx/sites-enabled/videointerleaving"
systemctl reload nginx || echo "Warning: Nginx reload failed (check logs)."

# --------------------------------------------
# 8. Firewall
# --------------------------------------------
if command -v ufw >/dev/null; then
    echo ">>> 🛡️  Updating Firewall..."
    ufw allow 80/tcp >/dev/null 2>&1    # Web Stream + Monitors
    ufw allow 2323/tcp >/dev/null 2>&1  # ASCII Telnet
fi

echo "----------------------------------------------------"
echo "✅ Setup Complete. Tools are ready."
echo ""
echo "👉 To run WEB Mode:"
echo "   systemctl enable --now vi-web"
echo ""
echo "👉 To run ASCII Mode:"
echo "   systemctl enable --now vi-ascii"
echo ""
echo "You can view the monitors at:"
echo "   http://<IP>/monitor/       (Web Mode)"
echo "   http://<IP>/monitor_ascii/ (ASCII Mode)"
echo "----------------------------------------------------"