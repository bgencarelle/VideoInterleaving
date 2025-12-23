#!/bin/bash
set -euo pipefail  # Better error handling: exit on error, undefined vars, pipe failures

# --- CONFIGURATION ---
PROJECT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
VENV_DIR="$PROJECT_DIR/.venv"

# Parse command line arguments
DRY_RUN=false
for arg in "$@"; do
    case $arg in
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        *)
            # Unknown option
            ;;
    esac
done

# Detect real user if running via sudo, otherwise default to current user
if [ -n "${SUDO_USER:-}" ]; then
    USERNAME="$SUDO_USER"
else
    USERNAME=$(whoami)
fi

echo ">>> 🛰️  Starting Application Setup..."
if [ "$DRY_RUN" = true ]; then
    echo "    ⚠️  DRY RUN MODE - No changes will be made"
fi
echo "    Running as User: $USERNAME"
echo "    Project Dir:     $PROJECT_DIR"

# --------------------------------------------
# 0. OS Detection & Package Manager Setup
# --------------------------------------------
detect_os() {
    local os=""
    local pkg_manager=""
    local pkg_update_cmd=""
    local pkg_install_cmd=""
    local needs_sudo=true
    
    if [[ "$OSTYPE" == "darwin"* ]]; then
        # macOS
        os="macos"
        if command -v brew >/dev/null 2>&1; then
            pkg_manager="brew"
            pkg_update_cmd="brew update"
            pkg_install_cmd="brew install"
            needs_sudo=false
        else
            echo "❌ ERROR: Homebrew not found. Install from https://brew.sh"
            exit 1
        fi
    elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
        # Linux - detect distribution
        if [ -f /etc/os-release ]; then
            . /etc/os-release
            case "$ID" in
                debian|ubuntu|raspbian)
                    os="debian"
                    pkg_manager="apt"
                    pkg_update_cmd="apt update -qq"
                    pkg_install_cmd="apt install -y"
                    ;;
                fedora|rhel|centos|rocky|almalinux)
                    os="rhel"
                    if command -v dnf >/dev/null 2>&1; then
                        pkg_manager="dnf"
                        pkg_update_cmd="dnf check-update -q || true"
                        pkg_install_cmd="dnf install -y"
                    else
                        pkg_manager="yum"
                        pkg_update_cmd="yum check-update -q || true"
                        pkg_install_cmd="yum install -y"
                    fi
                    ;;
                arch|manjaro)
                    os="arch"
                    pkg_manager="pacman"
                    pkg_update_cmd="pacman -Sy"
                    pkg_install_cmd="pacman -S --noconfirm"
                    ;;
                *)
                    echo "⚠️  WARNING: Unsupported Linux distribution: $ID"
                    echo "   Attempting Debian/Ubuntu package names..."
                    os="debian"
                    pkg_manager="apt"
                    pkg_update_cmd="apt update -qq"
                    pkg_install_cmd="apt install -y"
                    ;;
            esac
        else
            echo "⚠️  WARNING: Cannot detect Linux distribution. Assuming Debian/Ubuntu."
            os="debian"
            pkg_manager="apt"
            pkg_update_cmd="apt update -qq"
            pkg_install_cmd="apt install -y"
        fi
    else
        echo "❌ ERROR: Unsupported OS: $OSTYPE"
        exit 1
    fi
    
    echo "$os|$pkg_manager|$pkg_update_cmd|$pkg_install_cmd|$needs_sudo"
}

# Get platform-specific packages based on README.md instructions
get_packages_for_platform() {
    local os=$1
    
    case "$os" in
        debian)
            # From system-requirements.txt (Debian/Ubuntu/Raspbian)
            echo "python3-venv python3-dev python3-pip build-essential cmake pkg-config \
                  ninja-build python-is-python3 python3-pil python3-numpy python3-opencv \
                  python3-psutil python3-websockets python3-opengl python3-moderngl \
                  libwebp-dev libjpeg-dev libturbojpeg0 libgl1-mesa-dev libglu1-mesa-dev libegl1-mesa-dev \
                  libgles-dev libglvnd-dev libglfw3-dev mesa-utils mesa-utils-extra \
                  libsdl2-dev libasound2-dev chrony dnsutils"
            ;;
        rhel)
            # From README.md (Fedora/CentOS)
            echo "python3 python3-pip python3-devel gcc gcc-c++ make cmake pkgconfig \
                  libwebp-devel libjpeg-turbo-devel SDL2-devel alsa-lib-devel \
                  mesa-libGL-devel mesa-libGLU-devel mesa-libEGL-devel mesa-libGLES-devel \
                  libglvnd-devel glfw-devel mesa-utils \
                  chrony ninja-build bind-utils"
            ;;
        arch)
            # Arch Linux equivalents
            echo "python python-pip base-devel cmake pkg-config ninja \
                  libwebp libjpeg-turbo sdl2 alsa-lib mesa glu glfw \
                  chrony bind-tools"
            ;;
        macos)
            # From README.md (macOS/Homebrew) - minimal set
            echo "python webp pkg-config sdl2 chrony jpeg-turbo"
            ;;
        *)
            echo ""
            ;;
    esac
}

# Install system packages
install_system_packages() {
    local os=$1
    local pkg_manager=$2
    local update_cmd=$3
    local install_cmd=$4
    local needs_sudo=$5
    
    echo ">>> 📦 Installing system libraries ($pkg_manager)..."
    
    local packages=$(get_packages_for_platform "$os")
    
    if [ -z "$packages" ]; then
        echo "⚠️  WARNING: No packages defined for platform: $os"
        return
    fi
    
    if [ "$DRY_RUN" = true ]; then
        echo "[DRY-RUN] Would run: $update_cmd"
        if [ "$needs_sudo" = "true" ]; then
            echo "[DRY-RUN] Would run: sudo $install_cmd $packages"
        else
            echo "[DRY-RUN] Would run: $install_cmd $packages"
        fi
        echo "[DRY-RUN] Packages to install: $packages"
        return
    fi
    
    # Update package lists
    if [ "$needs_sudo" = "true" ]; then
        sudo $update_cmd || true
        sudo $install_cmd $packages
    else
        # Homebrew doesn't need sudo
        $update_cmd || true
        $install_cmd $packages
    fi
}

# --------------------------------------------
# 1. Hardware Detection (Linux only)
# --------------------------------------------
IS_PI=false
HAS_GPU=false
if [[ "$OSTYPE" == "linux-gnu"* ]]; then
    if [ -d "/dev/dri" ]; then HAS_GPU=true; fi
    if grep -q "Raspberry Pi" /proc/device-tree/model 2>/dev/null 2>&1; then 
        IS_PI=true
        HAS_GPU=true
    fi
fi

echo "    Hardware: GPU=$HAS_GPU | Pi=$IS_PI"

# Detect OS and package manager
OS_INFO=$(detect_os)
OS=$(echo "$OS_INFO" | cut -d'|' -f1)
PKG_MANAGER=$(echo "$OS_INFO" | cut -d'|' -f2)
PKG_UPDATE=$(echo "$OS_INFO" | cut -d'|' -f3)
PKG_INSTALL=$(echo "$OS_INFO" | cut -d'|' -f4)
NEEDS_SUDO=$(echo "$OS_INFO" | cut -d'|' -f5)

echo "    OS: $OS | Package Manager: $PKG_MANAGER"

# Install system packages
install_system_packages "$OS" "$PKG_MANAGER" "$PKG_UPDATE" "$PKG_INSTALL" "$NEEDS_SUDO"

# Add user to video/render groups (Linux only)
if [[ "$OSTYPE" == "linux-gnu"* ]] && [ "$HAS_GPU" = true ] && [ "$USERNAME" != "root" ]; then
    if [ "$DRY_RUN" = true ]; then
        echo "[DRY-RUN] Would add user $USERNAME to groups: video, render"
        echo "[DRY-RUN] Would run: sudo usermod -aG video,render $USERNAME"
    else
        sudo usermod -aG video,render "$USERNAME" 2>/dev/null || true
    fi
fi

# --------------------------------------------
# 2. Python Environment
# --------------------------------------------
echo ">>> 🐍 Setting up Python venv..."

if [ ! -f "$PROJECT_DIR/requirements.txt" ]; then
    echo "❌ ERROR: requirements.txt not found in $PROJECT_DIR"
    exit 1
fi

# Check Python version (3.11+ required)
if ! python3 -c "import sys; sys.exit(0 if sys.version_info >= (3, 11) else 1)" 2>/dev/null; then
    echo "❌ ERROR: Python 3.11+ required. Found: $(python3 --version 2>&1)"
    exit 1
fi

if [ "$DRY_RUN" = true ]; then
    echo "[DRY-RUN] Would create venv at: $VENV_DIR"
    if [ -d "$VENV_DIR" ]; then
        echo "[DRY-RUN] Would remove existing venv: $VENV_DIR"
    fi
    if [ "$OS" = "macos" ] || [ "$USERNAME" = "$(whoami)" ]; then
        echo "[DRY-RUN] Would run: python3 -m venv --system-site-packages $VENV_DIR"
    else
        echo "[DRY-RUN] Would run: sudo -u $USERNAME python3 -m venv --system-site-packages $VENV_DIR"
    fi
    echo "[DRY-RUN] Would install packages from: $PROJECT_DIR/requirements.txt"
    if [ "$OS" = "macos" ] || [ "$USERNAME" = "$(whoami)" ]; then
        echo "[DRY-RUN] Would run: $VENV_DIR/bin/pip install --upgrade pip wheel"
        echo "[DRY-RUN] Would run: $VENV_DIR/bin/pip install -r $PROJECT_DIR/requirements.txt"
    else
        echo "[DRY-RUN] Would run: sudo -u $USERNAME $VENV_DIR/bin/pip install --upgrade pip wheel"
        echo "[DRY-RUN] Would run: sudo -u $USERNAME $VENV_DIR/bin/pip install -r $PROJECT_DIR/requirements.txt"
    fi
else
    # Create venv
    if [ -d "$VENV_DIR" ]; then 
        echo "    Removing existing venv..."
        rm -rf "$VENV_DIR"
    fi

    # Create venv with appropriate ownership
    # On macOS or when running as the same user, no sudo needed
    if [ "$OS" = "macos" ] || [ "$USERNAME" = "$(whoami)" ]; then
        python3 -m venv --system-site-packages "$VENV_DIR"
    else
        sudo -u "$USERNAME" python3 -m venv --system-site-packages "$VENV_DIR"
    fi

    # Install Python packages
    VENV_PIP="$VENV_DIR/bin/pip"
    if [ "$OS" = "macos" ] || [ "$USERNAME" = "$(whoami)" ]; then
        "$VENV_PIP" install --upgrade pip wheel
        "$VENV_PIP" install -r "$PROJECT_DIR/requirements.txt"
    else
        sudo -u "$USERNAME" "$VENV_PIP" install --upgrade pip wheel
        sudo -u "$USERNAME" "$VENV_PIP" install -r "$PROJECT_DIR/requirements.txt"
    fi
fi

# --------------------------------------------
# 3. Logrotate (Linux only)
# --------------------------------------------
if [[ "$OSTYPE" == "linux-gnu"* ]] && [ -d "/etc/logrotate.d" ]; then
    echo ">>> 📜 Configuring Logrotate..."
    if [ "$DRY_RUN" = true ]; then
        echo "[DRY-RUN] Would create /etc/logrotate.d/videointerleaving with:"
        echo "[DRY-RUN] ---"
        cat <<EOF | sed 's/^/[DRY-RUN] /'
$PROJECT_DIR/*.log {
    daily
    rotate 5
    compress
    missingok
    copytruncate
    size 10M
}
EOF
        echo "[DRY-RUN] ---"
    else
        sudo tee "/etc/logrotate.d/videointerleaving" > /dev/null <<EOF
$PROJECT_DIR/*.log {
    daily
    rotate 5
    compress
    missingok
    copytruncate
    size 10M
}
EOF
    fi
fi

# --------------------------------------------
# 4. Display Detection Functions (Linux only)
# --------------------------------------------
if [[ "$OSTYPE" == "linux-gnu"* ]]; then
    detect_display() {
        if [ -n "${WAYLAND_DISPLAY:-}" ] || [ "${XDG_SESSION_TYPE:-}" = "wayland" ]; then
            echo "wayland"
            return
        fi
        
        if [ -n "${DISPLAY:-}" ]; then
            echo "x11"
            return
        fi
        
        if [ -S "/tmp/.X11-unix/X0" ] 2>/dev/null; then
            echo "x11"
            return
        fi
        
        for socket in /tmp/.X11-unix/X*; do
            if [ -S "$socket" ] 2>/dev/null; then
                echo "x11"
                return
            fi
        done
        
        if [ -c "/dev/fb0" ] 2>/dev/null; then
            echo "framebuffer"
            return
        fi
        
        echo "x11"
    }

    get_user_display() {
        local user=$1
        local display=""
        
        if command -v loginctl >/dev/null 2>&1; then
            local session=$(loginctl list-sessions --user="$user" --no-legend 2>/dev/null | head -n1 | awk '{print $1}')
            if [ -n "$session" ]; then
                display=$(loginctl show-session "$session" -p Display --value 2>/dev/null || echo "")
            fi
        fi
        
        if [ -z "$display" ]; then
            if [ -S "/tmp/.X11-unix/X0" ] 2>/dev/null; then
                display=":0"
            else
                for socket in /tmp/.X11-unix/X*; do
                    if [ -S "$socket" ] 2>/dev/null; then
                        local num=$(basename "$socket" | sed 's/X//')
                        display=":$num"
                        break
                    fi
                done
            fi
        fi
        
        echo "${display:-:0}"
    }

    get_xauthority() {
        local user=$1
        local xauth_path=""
        local home_dir=$(eval echo ~"$user")
        
        if [ -f "$home_dir/.Xauthority" ]; then
            xauth_path="$home_dir/.Xauthority"
        fi
        
        if [ -z "$xauth_path" ] && command -v loginctl >/dev/null 2>&1; then
            local session=$(loginctl list-sessions --user="$user" --no-legend 2>/dev/null | head -n1 | awk '{print $1}')
            if [ -n "$session" ]; then
                xauth_path=$(loginctl show-session "$session" -p XAuthority --value 2>/dev/null || echo "")
            fi
        fi
        
        echo "$xauth_path"
    }
fi

# --------------------------------------------
# 5. Systemd Services (Linux only)
# --------------------------------------------
if [[ "$OSTYPE" == "linux-gnu"* ]] && command -v systemctl >/dev/null 2>&1 && [ -d "/etc/systemd/system" ]; then
    echo ">>> ⚙️  Creating Systemd Services..."
    
    # Build GPU environment block
    if [ "$IS_PI" = true ]; then
        ENV_BLOCK="Environment=MESA_GL_VERSION_OVERRIDE=3.3
Environment=MESA_GLSL_VERSION_OVERRIDE=330"
    elif [ "$HAS_GPU" = true ]; then
        ENV_BLOCK="# Generic GPU Auto-detect"
    else
        ENV_BLOCK="Environment=GALLIUM_DRIVER=llvmpipe"
    fi
    
    # Detect display environment
    DISPLAY_TYPE=$(detect_display)
    DISPLAY_VAR=$(get_user_display "$USERNAME")
    XAUTH_PATH=$(get_xauthority "$USERNAME")
    
    # Build display environment block
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
    
    if [ "$DRY_RUN" = true ]; then
        echo "[DRY-RUN] Would create /etc/systemd/system/vi-web.service:"
        echo "[DRY-RUN] ---"
        cat <<EOF | sed 's/^/[DRY-RUN] /'
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
        echo "[DRY-RUN] ---"
        echo ""
        echo "[DRY-RUN] Would create /etc/systemd/system/vi-ascii.service:"
        echo "[DRY-RUN] ---"
        cat <<EOF | sed 's/^/[DRY-RUN] /'
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
        echo "[DRY-RUN] ---"
        echo ""
        echo "[DRY-RUN] Would create /etc/systemd/system/vi-local.service:"
        echo "[DRY-RUN] ---"
        cat <<EOF | sed 's/^/[DRY-RUN] /'
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
        echo "[DRY-RUN] ---"
        echo ""
        echo "[DRY-RUN] Would run: sudo systemctl daemon-reload"
    else
        # Create services
        sudo tee "/etc/systemd/system/vi-web.service" > /dev/null <<EOF
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

        sudo tee "/etc/systemd/system/vi-ascii.service" > /dev/null <<EOF
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

        sudo tee "/etc/systemd/system/vi-local.service" > /dev/null <<EOF
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
    fi
    
    echo ""
    echo "👉 Web Mode:   systemctl enable --now vi-web"
    echo "👉 ASCII Mode: systemctl enable --now vi-ascii"
    echo "👉 Local Mode: systemctl enable --now vi-local"
else
    echo ">>> ⚙️  Skipping Systemd Services (not available on this system)"
    if [ "$OS" = "macos" ]; then
        echo "    On macOS, run manually: $VENV_DIR/bin/python -O main.py --mode <web|ascii|local>"
    else
        echo "    Run manually: $VENV_DIR/bin/python -O main.py --mode <web|ascii|local>"
    fi
fi

# --------------------------------------------
# 6. Firewall (Linux only)
# --------------------------------------------
if command -v ufw >/dev/null 2>&1; then
    echo ">>> 🔥 Configuring Firewall..."
    if [ "$DRY_RUN" = true ]; then
        echo "[DRY-RUN] Would run: sudo ufw allow 2323/tcp"
        echo "[DRY-RUN] Would run: sudo ufw allow 2423/tcp  # ASCII-Web mode"
        echo "[DRY-RUN] Would run: sudo ufw allow 2424/tcp  # ASCII-Web mode"
        echo "[DRY-RUN] Would run: sudo ufw allow 8080/tcp  # Web stream"
        echo "[DRY-RUN] Would run: sudo ufw allow 1978/tcp  # Monitor"
    else
        sudo ufw allow 2323/tcp >/dev/null 2>&1 || true
        sudo ufw allow 2423/tcp >/dev/null 2>&1 || true  # ASCII-Web mode
        sudo ufw allow 2424/tcp >/dev/null 2>&1 || true  # ASCII-Web mode
        sudo ufw allow 8080/tcp >/dev/null 2>&1 || true  # Web stream
        sudo ufw allow 1978/tcp >/dev/null 2>&1 || true  # Monitor
    fi
fi

echo ""
echo "----------------------------------------------------"
if [ "$DRY_RUN" = true ]; then
    echo "✅ DRY RUN Complete - No changes were made."
    echo "   Run without --dry-run to apply these changes."
else
    echo "✅ App Setup Complete."
fi
echo "----------------------------------------------------"
