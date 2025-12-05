#!/bin/bash
set -e  # Exit immediately if a command exits with a non-zero status

# --- CONFIGURATION ---
PROJECT_DIR=$(pwd)
VENV_DIR="$PROJECT_DIR/.venv"
SERVICE_NAME="videointerleaving"
USERNAME="root"  # Change to 'ubuntu' or 'pi' if not running as root

echo ">>> 🛰️  Starting Universal Headless Setup for $SERVICE_NAME..."
echo ">>> Project Directory: $PROJECT_DIR"

# --------------------------------------------
# 1. Hardware & GPU Detection
# --------------------------------------------
echo ">>> 🔍 Scanning Hardware..."

IS_PI=false
HAS_GPU=false

# Check for Direct Rendering Infrastructure (DRI)
if [ -d "/dev/dri" ]; then
    HAS_GPU=true
    echo "    ✅ GPU/DRI detected (Hardware Acceleration Available)."
else
    echo "    ⚠️  No GPU detected. Will use CPU Software Rendering."
fi

# Check specifically for Raspberry Pi (needs extra driver hints)
if grep -q "Raspberry Pi" /proc/device-tree/model 2>/dev/null; then
    IS_PI=true
    HAS_GPU=true # Pi implies GPU
    echo "    🍓 Raspberry Pi Model detected."
fi

# --------------------------------------------
# 2. Install System Dependencies
# --------------------------------------------
echo ">>> 📦 Installing system dependencies..."
apt-get update
# Dependencies for Python, GL, WebP, Nginx, and Firewall management
apt-get install -y \
    python3-venv python3-dev python3-pip build-essential \
    libwebp-dev libgl1-mesa-dev libglu1-mesa-dev libegl1-mesa-dev mesa-utils \
    chrony ninja-build python-is-python3 nginx ufw

# --------------------------------------------
# 3. Permissions & Groups (Critical for GPU access)
# --------------------------------------------
if [ "$HAS_GPU" = true ] && [ "$USERNAME" != "root" ]; then
    echo ">>> 🔑 Adding user $USERNAME to video/render groups..."
    usermod -aG video,render "$USERNAME" || true
fi

# --------------------------------------------
# 4. Configure Chrony
# --------------------------------------------
echo ">>> ⏱️  Configuring Chrony..."
systemctl enable --now chrony

# --------------------------------------------
# 5. Python Virtual Environment
# --------------------------------------------
echo ">>> 🐍 Setting up Python environment..."

# Clean up old venv if it exists in the project folder
if [ -d "$VENV_DIR" ]; then
    rm -rf "$VENV_DIR"
fi

# Clean up legacy global path
if [ -d "/root/PyIntervenv" ]; then
    rm -rf "/root/PyIntervenv"
fi

python3 -m venv "$VENV_DIR"
source "$VENV_DIR/bin/activate"

echo "    Installing requirements..."
pip install --upgrade pip
pip install wheel
pip install -r requirements.txt

# --------------------------------------------
# 6. Generate Folder Lists
# --------------------------------------------
echo ">>> 📂 Generating file lists..."
python make_file_lists.py

# --------------------------------------------
# 7. Setup Systemd Service (Universal Adaptive)
# --------------------------------------------
echo ">>> ⚙️  Creating Systemd Service ($SERVICE_NAME)..."

SERVICE_FILE="/etc/systemd/system/${SERVICE_NAME}.service"

# -- Build Environment Variables Block --
if [ "$IS_PI" = true ]; then
    # RASPBERRY PI: Force Desktop GL 3.3 context on VideoCore drivers
    ENV_BLOCK=$(cat <<EOF
# Pi Hardware Acceleration
Environment=MESA_GL_VERSION_OVERRIDE=3.3
Environment=MESA_GLSL_VERSION_OVERRIDE=330
EOF
)
elif [ "$HAS_GPU" = true ]; then
    # GENERIC GPU (Intel/Nvidia/AMD): Let the system pick the driver.
    # We do NOT force llvmpipe.
    ENV_BLOCK=$(cat <<EOF
# Generic Hardware Acceleration (Auto-detected)
# No overrides needed; system uses /dev/dri
EOF
)
else
    # VPS / NO GPU: Force Software Rasterizer
    ENV_BLOCK=$(cat <<EOF
# CPU Software Rendering (llvmpipe)
Environment=GALLIUM_DRIVER=llvmpipe
# Environment=LP_NUM_THREADS=2
EOF
)
fi

# -- Write Service File --
cat <<EOF > "$SERVICE_FILE"
[Unit]
Description=VideoInterleaving Headless Stream
After=network.target

[Service]
User=$USERNAME
WorkingDirectory=$PROJECT_DIR

# Common Env Tuning
Environment=PYTHONUNBUFFERED=1
# Environment=MALLOC_TRIM_THRESHOLD_=100000

# Hardware Specific Env
$ENV_BLOCK

# Executing Python directly from the local project .venv
ExecStart=$VENV_DIR/bin/python -O main.py

# Auto-restart config
Restart=always
RestartSec=3

# Logging
StandardOutput=append:$PROJECT_DIR/systemd_out.log
StandardError=append:$PROJECT_DIR/systemd_err.log

[Install]
WantedBy=multi-user.target
EOF

systemctl daemon-reload
systemctl enable "$SERVICE_NAME"

# --------------------------------------------
# 8. Setup Logrotate
# --------------------------------------------
echo ">>> 📝 Configuring Logrotate..."

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

# Verify config
logrotate -d "$LOGROTATE_FILE" > /dev/null 2>&1

# --------------------------------------------
# 9. Setup Nginx Reverse Proxy
# --------------------------------------------
echo ">>> 🌐 Configuring Nginx..."

NGINX_CONF="/etc/nginx/sites-available/$SERVICE_NAME"

cat <<EOF > "$NGINX_CONF"
server {
    listen 80 default_server;
    listen [::]:80 default_server;
    server_name _;

    # Redirect /monitor -> /monitor/
    location = /monitor { return 301 /monitor/; }

    # Main Stream (Port 8080)
    location / {
        proxy_pass http://127.0.0.1:8080;
        proxy_http_version 1.1;
        proxy_set_header Connection "";
        proxy_set_header Host              \$host;
        proxy_set_header X-Real-IP         \$remote_addr;
        proxy_set_header X-Forwarded-For   \$proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto \$scheme;

        # Streaming Optimizations
        proxy_buffering off;
        proxy_cache off;
        proxy_request_buffering off;
        proxy_read_timeout 3600s;
        sendfile off;
        tcp_nodelay on;
        gzip off;
    }

    # Monitor Dashboard (Port 1978)
    location /monitor/ {
        proxy_pass http://127.0.0.1:1978/;
        proxy_http_version 1.1;
        proxy_set_header Connection "";
        proxy_set_header Host              \$host;
        proxy_set_header X-Real-IP         \$remote_addr;
        proxy_set_header X-Forwarded-For   \$proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto \$scheme;

        proxy_buffering off;
        proxy_cache off;
        proxy_request_buffering off;
        proxy_read_timeout 3600s;
    }

    client_max_body_size 20M;
    add_header X-Content-Type-Options "nosniff" always;
    add_header X-Frame-Options "SAMEORIGIN" always;
    add_header Referrer-Policy "no-referrer-when-downgrade" always;
}
EOF

# Enable Site & Disable Default
rm -f /etc/nginx/sites-enabled/default
ln -sf "$NGINX_CONF" "/etc/nginx/sites-enabled/$SERVICE_NAME"

# Test Nginx Config
nginx -t > /dev/null 2>&1
if [ $? -eq 0 ]; then
    echo "    Nginx config valid. Reloading..."
    systemctl reload nginx
else
    echo "    WARNING: Nginx config failed. Check $NGINX_CONF manually."
fi

# --------------------------------------------
# 10. Firewall Configuration (UFW)
# --------------------------------------------
echo ">>> 🛡️  Checking Firewall..."
if command -v ufw >/dev/null; then
    # Only enable rule if UFW is active or installed
    echo "    Allowing Nginx HTTP traffic..."
    ufw allow 'Nginx Full' >/dev/null 2>&1 || ufw allow 80 >/dev/null 2>&1
fi

# --------------------------------------------
# 11. Final Instructions
# --------------------------------------------
echo "----------------------------------------------------"
echo "✅ Setup Complete!"
echo ""
echo "Detected Mode: $(if [ "$IS_PI" = true ]; then echo "Raspberry Pi (Overrides Active)"; elif [ "$HAS_GPU" = true ]; then echo "Hardware Accelerated (Standard GPU)"; else echo "Software Rendering (VPS)"; fi)"
echo ""
echo "To start the service:"
echo "  systemctl start $SERVICE_NAME"
echo ""
echo "To view status:"
echo "  systemctl status $SERVICE_NAME"
echo ""
echo "To follow logs:"
echo "  journalctl -u $SERVICE_NAME -f"
echo "----------------------------------------------------"