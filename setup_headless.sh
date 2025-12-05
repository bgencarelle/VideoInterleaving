#!/bin/bash
set -e  # Exit immediately if a command exits with a non-zero status

# --- CONFIGURATION ---
PROJECT_DIR=$(pwd)
# CHANGED: Venv is now a hidden folder inside the project directory
VENV_DIR="$PROJECT_DIR/.venv"
SERVICE_NAME="videointerleaving"
USERNAME="root"

echo ">>> 🛰️  Starting Headless Server Setup for $SERVICE_NAME..."
echo ">>> Project Directory: $PROJECT_DIR"
echo ">>> Virtual Env Location: $VENV_DIR"

# --------------------------------------------
# 1. Install System Dependencies
# --------------------------------------------
echo ">>> 📦 Installing system dependencies..."
apt-get update
apt-get install -y \
    python3-venv python3-dev python3-pip build-essential \
    libwebp-dev libgl1-mesa-dev libglu1-mesa-dev libegl1-mesa-dev mesa-utils \
    chrony ninja-build python-is-python3 nginx

# --------------------------------------------
# 2. Configure Chrony
# --------------------------------------------
echo ">>> ⏱️  Configuring Chrony..."
systemctl enable --now chrony

# --------------------------------------------
# 3. Python Virtual Environment
# --------------------------------------------
echo ">>> 🐍 Setting up Python environment..."

# Clean up old venv if it exists in the project folder
if [ -d "$VENV_DIR" ]; then
    echo "    Removing existing .venv..."
    rm -rf "$VENV_DIR"
fi

# Also clean up the OLD global venv if it exists (cleanup)
if [ -d "/root/PyIntervenv" ]; then
    echo "    Removing old global /root/PyIntervenv..."
    rm -rf "/root/PyIntervenv"
fi

python3 -m venv "$VENV_DIR"
source "$VENV_DIR/bin/activate"

echo "    Installing requirements..."
pip install --upgrade pip
pip install wheel
pip install -r requirements.txt

# --------------------------------------------
# 4. Generate Folder Lists
# --------------------------------------------
echo ">>> 📂 Generating file lists..."
python make_file_lists.py

# --------------------------------------------
# 5. Setup Systemd Service
# --------------------------------------------
echo ">>> ⚙️  Creating Systemd Service ($SERVICE_NAME)..."

SERVICE_FILE="/etc/systemd/system/${SERVICE_NAME}.service"

cat <<EOF > "$SERVICE_FILE"
[Unit]
Description=VideoInterleaving Headless Stream
After=network.target

[Service]
User=$USERNAME
WorkingDirectory=$PROJECT_DIR

# Environment Tuning
Environment=GALLIUM_DRIVER=llvmpipe
#Environment=LP_NUM_THREADS=2
#Environment=PYTHONUNBUFFERED=1
#Environment=MALLOC_TRIM_THRESHOLD_=100000

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
# 6. Setup Logrotate
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
if [ $? -eq 0 ]; then
    echo "    Logrotate config is valid."
else
    echo "    WARNING: Logrotate config check failed."
fi

# --------------------------------------------
# 7. Final Instructions
# --------------------------------------------
echo "----------------------------------------------------"
echo "✅ Setup Complete!"
echo ""
echo "To start the service:"
echo "  systemctl start $SERVICE_NAME"
echo ""
echo "To view status: "
echo "  systemctl status $SERVICE_NAME"
echo ""
echo "To follow logs:"
echo "  journalctl -u $SERVICE_NAME -f"
echo "----------------------------------------------------"