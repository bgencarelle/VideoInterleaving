#!/bin/bash
set -euo pipefail  # Better error handling: exit on error, undefined vars, pipe failures

# --- CONFIGURATION ---
PROJECT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
VENV_DIR="$PROJECT_DIR/.venv"

# Parse command line arguments
DRY_RUN=false
VERBOSE=false
for arg in "$@"; do
    case $arg in
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        --verbose|-v)
            VERBOSE=true
            shift
            ;;
        *)
            # Unknown option
            ;;
    esac
done

# --- LOGGING FUNCTIONS ---
log_info() {
    echo "ℹ️  $*"
}

log_success() {
    echo "✅ $*"
}

log_warning() {
    echo "⚠️  $*"
}

log_error() {
    echo "❌ $*" >&2
}

log_verbose() {
    if [ "$VERBOSE" = true ]; then
        echo "   [VERBOSE] $*"
    fi
}

log_step() {
    echo ""
    echo ">>> $*"
}

# Detect real user if running via sudo, otherwise default to current user
if [ -n "${SUDO_USER:-}" ]; then
    USERNAME="$SUDO_USER"
else
    USERNAME=$(whoami)
fi

# --- PORT DETECTION ---
# Default ports from server_config.py
DEFAULT_PORTS=(
    1978  # Monitor (WEB mode)
    1980  # Monitor (ASCIIWEB mode)
    2323  # ASCII Telnet
    2324  # ASCII Monitor (ASCII mode - primary_port+1)
    2424  # ASCII WebSocket (ASCIIWEB mode)
    8080  # Stream (WEB mode)
    8888  # Monitor (LOCAL mode)
)

check_port_available() {
    local port=$1
    if command -v netstat >/dev/null 2>&1; then
        if netstat -tuln 2>/dev/null | grep -q ":$port "; then
            return 1  # Port in use
        fi
    elif command -v ss >/dev/null 2>&1; then
        if ss -tuln 2>/dev/null | grep -q ":$port "; then
            return 1  # Port in use
        fi
    elif command -v lsof >/dev/null 2>&1; then
        if lsof -i ":$port" >/dev/null 2>&1; then
            return 1  # Port in use
        fi
    fi
    return 0  # Port available
}

detect_port_usage() {
    local ports_in_use=()
    local ports_available=()
    
    for port in "${DEFAULT_PORTS[@]}"; do
        if check_port_available "$port"; then
            ports_available+=("$port")
        else
            ports_in_use+=("$port")
            local process=$(lsof -i ":$port" 2>/dev/null | tail -n +2 | awk '{print $1}' | head -1 || echo "unknown")
            log_verbose "Port $port is in use by: $process"
        fi
    done
    
    if [ ${#ports_in_use[@]} -gt 0 ]; then
        log_warning "Some ports are already in use: ${ports_in_use[*]}"
        log_info "This is normal if services are already running"
    fi
}

# --- VALIDATION FUNCTIONS ---
preflight_checks() {
    local errors=0
    local warnings=0
    
    log_step "🔍 Running Pre-flight Checks..."
    
    # Check Python version
    if ! python3 -c "import sys; sys.exit(0 if sys.version_info >= (3, 11) else 1)" 2>/dev/null; then
        log_error "Python 3.11+ required. Found: $(python3 --version 2>&1)"
        errors=$((errors + 1))
    else
        log_success "Python version: $(python3 --version 2>&1)"
    fi
    
    # Check project directory structure
    if [ ! -f "$PROJECT_DIR/requirements.txt" ]; then
        log_error "requirements.txt not found in $PROJECT_DIR"
        errors=$((errors + 1))
    else
        log_success "requirements.txt found"
    fi
    
    if [ ! -f "$PROJECT_DIR/main.py" ]; then
        log_warning "main.py not found - project may be incomplete"
        warnings=$((warnings + 1))
    else
        log_success "main.py found"
    fi
    
    # Check disk space (at least 1GB free)
    if command -v df >/dev/null 2>&1; then
        local available_space=$(df "$PROJECT_DIR" | tail -1 | awk '{print $4}')
        if [ "$available_space" -lt 1048576 ]; then  # Less than 1GB in KB
            log_warning "Low disk space: $(df -h "$PROJECT_DIR" | tail -1 | awk '{print $4}') available"
            warnings=$((warnings + 1))
        else
            log_verbose "Disk space: $(df -h "$PROJECT_DIR" | tail -1 | awk '{print $4}') available"
        fi
    fi
    
    # Check if systemd is available (Linux)
    if [[ "$OSTYPE" == "linux-gnu"* ]]; then
        if ! command -v systemctl >/dev/null 2>&1; then
            log_warning "systemctl not found - systemd services will be skipped"
            warnings=$((warnings + 1))
        else
            log_success "systemd available"
        fi
    fi
    
    # Check port availability
    detect_port_usage
    
    # Summary
    if [ "$errors" -gt 0 ]; then
        log_error "Pre-flight checks failed with $errors error(s)"
        return 1
    elif [ "$warnings" -gt 0 ]; then
        log_warning "Pre-flight checks completed with $warnings warning(s)"
        return 0
    else
        log_success "Pre-flight checks passed"
        return 0
    fi
}

log_step "🛰️  Starting Application Setup..."
if [ "$DRY_RUN" = true ]; then
    log_warning "DRY RUN MODE - No changes will be made"
fi
log_info "Running as User: $USERNAME"
log_info "Project Dir:     $PROJECT_DIR"

# Run pre-flight checks
if ! preflight_checks; then
    log_error "Exiting due to validation errors"
    exit 1
fi

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

read_pkg_file() {
    # Reads a package list file, stripping comments/blank lines.
    local file="$1"
    if [ ! -f "$file" ]; then
        echo ""
        return
    fi
    sed 's/#.*$//' "$file" | sed '/^[[:space:]]*$/d'
}

# Get platform-specific packages based on README.md instructions
get_packages_for_platform() {
    local os=$1
    local pkg_list=""

    case "$os" in
        debian)
            # Preferred: read from system-requirements.txt (Debian/Ubuntu/Raspbian)
            pkg_list=$(read_pkg_file "$PROJECT_DIR/system-requirements.txt")
            # Add certbot for SSL certificate management
            if [ -n "$pkg_list" ]; then
                pkg_list="$pkg_list certbot python3-certbot-nginx"
            else
                pkg_list="certbot python3-certbot-nginx"
            fi
            ;;
        rhel)
            # From README.md (Fedora/CentOS)
            pkg_list="python3 python3-pip python3-devel gcc gcc-c++ make cmake pkgconfig \
libwebp-devel libjpeg-turbo-devel SDL2-devel alsa-lib-devel \
mesa-libGL-devel mesa-libGLU-devel mesa-libEGL-devel mesa-libGLES-devel \
libglvnd-devel glfw-devel mesa-utils \
chrony ninja-build bind-utils certbot python3-certbot-nginx"
            ;;
        arch)
            # Arch Linux equivalents
            pkg_list="python python-pip base-devel cmake pkg-config ninja \
libwebp libjpeg-turbo sdl2 alsa-lib mesa glu glfw \
chrony bind-tools certbot certbot-nginx"
            ;;
        macos)
            # From README.md (macOS/Homebrew) - minimal set
            pkg_list="python webp pkg-config sdl2 chrony jpeg-turbo"
            # Note: certbot on macOS is typically installed via pip or brew separately
            ;;
        *)
            pkg_list=""
            ;;
    esac

    echo "$pkg_list"
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
    # Check if user is already in groups
    local user_groups=$(groups "$USERNAME" 2>/dev/null || id -Gn "$USERNAME" 2>/dev/null || echo "")
    local needs_video=false
    local needs_render=false
    
    if ! echo "$user_groups" | grep -q "\bvideo\b"; then
        needs_video=true
    fi
    if ! echo "$user_groups" | grep -q "\brender\b"; then
        needs_render=true
    fi
    
    if [ "$needs_video" = true ] || [ "$needs_render" = true ]; then
        if [ "$DRY_RUN" = true ]; then
            log_info "[DRY-RUN] Would add user $USERNAME to groups: video, render"
            log_info "[DRY-RUN] Would run: sudo usermod -aG video,render $USERNAME"
        else
            sudo usermod -aG video,render "$USERNAME" 2>/dev/null || true
            log_success "Added user $USERNAME to groups: video, render"
            log_warning "User may need to log out and back in for group changes to take effect"
        fi
    else
        log_success "User $USERNAME already in required groups"
    fi
fi

# --- VENV DETECTION & VALIDATION ---
check_venv_valid() {
    if [ ! -d "$VENV_DIR" ]; then
        return 1  # Venv doesn't exist
    fi
    
    if [ ! -f "$VENV_DIR/bin/python" ]; then
        return 1  # Python executable missing
    fi
    
    # Check Python version matches
    local venv_python_version=$("$VENV_DIR/bin/python" --version 2>&1 | awk '{print $2}')
    local system_python_version=$(python3 --version 2>&1 | awk '{print $2}')
    
    if [ "$venv_python_version" != "$system_python_version" ]; then
        log_verbose "Venv Python version ($venv_python_version) differs from system ($system_python_version)"
        return 1
    fi
    
    # Check if key packages are installed
    if ! "$VENV_DIR/bin/python" -c "import moderngl" 2>/dev/null; then
        log_verbose "Key package 'moderngl' not found in venv"
        return 1
    fi
    
    return 0  # Venv is valid
}

# --------------------------------------------
# 2. Python Environment
# --------------------------------------------
log_step "🐍 Setting up Python venv..."

VENV_NEEDS_CREATE=true
VENV_NEEDS_UPDATE=false

if check_venv_valid; then
    log_success "Existing venv found and validated"
    VENV_NEEDS_CREATE=false
    
    # Check if requirements are up to date
    if [ -f "$PROJECT_DIR/requirements.txt" ]; then
        local req_hash=$(md5sum "$PROJECT_DIR/requirements.txt" 2>/dev/null | awk '{print $1}' || echo "")
        local venv_req_hash_file="$VENV_DIR/.requirements_hash"
        
        if [ -f "$venv_req_hash_file" ]; then
            local stored_hash=$(cat "$venv_req_hash_file" 2>/dev/null || echo "")
            if [ "$req_hash" != "$stored_hash" ]; then
                log_info "requirements.txt has changed - venv will be updated"
                VENV_NEEDS_UPDATE=true
            else
                log_success "Venv packages are up to date"
            fi
        else
            log_info "No requirements hash found - will update venv"
            VENV_NEEDS_UPDATE=true
        fi
    fi
else
    if [ -d "$VENV_DIR" ]; then
        log_warning "Existing venv is invalid - will recreate"
    else
        log_info "No existing venv found - will create"
    fi
    VENV_NEEDS_CREATE=true
fi

if [ "$DRY_RUN" = true ]; then
    if [ "$VENV_NEEDS_CREATE" = true ]; then
        log_info "[DRY-RUN] Would create venv at: $VENV_DIR"
        if [ -d "$VENV_DIR" ]; then
            log_info "[DRY-RUN] Would remove existing venv: $VENV_DIR"
        fi
    elif [ "$VENV_NEEDS_UPDATE" = true ]; then
        log_info "[DRY-RUN] Would update packages in existing venv"
    else
        log_info "[DRY-RUN] Venv is up to date - no changes needed"
    fi
    
    if [ "$VENV_NEEDS_CREATE" = true ] || [ "$VENV_NEEDS_UPDATE" = true ]; then
        if [ "$OS" = "macos" ] || [ "$USERNAME" = "$(whoami)" ]; then
            if [ "$VENV_NEEDS_CREATE" = true ]; then
                log_info "[DRY-RUN] Would run: python3 -m venv --system-site-packages $VENV_DIR"
            fi
            log_info "[DRY-RUN] Would run: $VENV_DIR/bin/pip install --upgrade pip wheel"
            log_info "[DRY-RUN] Would run: $VENV_DIR/bin/pip install -r $PROJECT_DIR/requirements.txt"
        else
            if [ "$VENV_NEEDS_CREATE" = true ]; then
                log_info "[DRY-RUN] Would run: sudo -u $USERNAME python3 -m venv --system-site-packages $VENV_DIR"
            fi
            log_info "[DRY-RUN] Would run: sudo -u $USERNAME $VENV_DIR/bin/pip install --upgrade pip wheel"
            log_info "[DRY-RUN] Would run: sudo -u $USERNAME $VENV_DIR/bin/pip install -r $PROJECT_DIR/requirements.txt"
        fi
    fi
else
    if [ "$VENV_NEEDS_CREATE" = true ]; then
        # Remove existing invalid venv
        if [ -d "$VENV_DIR" ]; then 
            log_info "Removing existing venv..."
            rm -rf "$VENV_DIR"
        fi

        # Create venv with appropriate ownership
        log_info "Creating venv..."
        if [ "$OS" = "macos" ] || [ "$USERNAME" = "$(whoami)" ]; then
            python3 -m venv --system-site-packages "$VENV_DIR"
        else
            sudo -u "$USERNAME" python3 -m venv --system-site-packages "$VENV_DIR"
        fi
        log_success "Venv created"
    fi

    # Install/update Python packages
    if [ "$VENV_NEEDS_CREATE" = true ] || [ "$VENV_NEEDS_UPDATE" = true ]; then
        VENV_PIP="$VENV_DIR/bin/pip"
        log_info "Installing/updating Python packages..."
        
        if [ "$OS" = "macos" ] || [ "$USERNAME" = "$(whoami)" ]; then
            "$VENV_PIP" install --upgrade pip wheel
            "$VENV_PIP" install -r "$PROJECT_DIR/requirements.txt"
        else
            sudo -u "$USERNAME" "$VENV_PIP" install --upgrade pip wheel
            sudo -u "$USERNAME" "$VENV_PIP" install -r "$PROJECT_DIR/requirements.txt"
        fi
        
        # Store requirements hash
        if [ -f "$PROJECT_DIR/requirements.txt" ]; then
            md5sum "$PROJECT_DIR/requirements.txt" 2>/dev/null | awk '{print $1}' > "$VENV_DIR/.requirements_hash" || true
        fi
        
        log_success "Python packages installed/updated"
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

# --- SYSTEMD SERVICE DETECTION ---
check_systemd_service_exists() {
    local service_name=$1
    if [ -f "/etc/systemd/system/$service_name" ]; then
        return 0  # Service exists
    fi
    return 1  # Service doesn't exist
}

backup_systemd_service() {
    local service_name=$1
    if [ -f "/etc/systemd/system/$service_name" ]; then
        local backup_dir="/etc/systemd/system/videointerleaving-backups"
        mkdir -p "$backup_dir"
        local timestamp=$(date +%Y%m%d_%H%M%S)
        cp "/etc/systemd/system/$service_name" "$backup_dir/$service_name.$timestamp" 2>/dev/null || true
        log_verbose "Backed up $service_name to $backup_dir/$service_name.$timestamp"
    fi
}

# --------------------------------------------
# 5. Systemd Services (Linux only)
# --------------------------------------------
if [[ "$OSTYPE" == "linux-gnu"* ]] && command -v systemctl >/dev/null 2>&1 && [ -d "/etc/systemd/system" ]; then
    log_step "⚙️  Creating/Updating Systemd Services..."

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
    
    # Check which services exist
    WEB_SERVICE_EXISTS=false
    ASCII_SERVICE_EXISTS=false
    LOCAL_SERVICE_EXISTS=false
    
    if check_systemd_service_exists "vi-web.service"; then
        WEB_SERVICE_EXISTS=true
        log_info "Existing vi-web.service found"
    fi
    if check_systemd_service_exists "vi-ascii.service"; then
        ASCII_SERVICE_EXISTS=true
        log_info "Existing vi-ascii.service found"
    fi
    if check_systemd_service_exists "vi-local.service"; then
        LOCAL_SERVICE_EXISTS=true
        log_info "Existing vi-local.service found"
    fi
    
    if [ "$DRY_RUN" = true ]; then
        if [ "$WEB_SERVICE_EXISTS" = true ]; then
            log_info "[DRY-RUN] Would update /etc/systemd/system/vi-web.service"
        else
            log_info "[DRY-RUN] Would create /etc/systemd/system/vi-web.service"
        fi
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
        if [ "$ASCII_SERVICE_EXISTS" = true ]; then
            log_info "[DRY-RUN] Would update /etc/systemd/system/vi-ascii.service"
        else
            log_info "[DRY-RUN] Would create /etc/systemd/system/vi-ascii.service"
        fi
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
        if [ "$LOCAL_SERVICE_EXISTS" = true ]; then
            log_info "[DRY-RUN] Would update /etc/systemd/system/vi-local.service"
        else
            log_info "[DRY-RUN] Would create /etc/systemd/system/vi-local.service"
        fi
        echo "[DRY-RUN] ---"
        log_info "[DRY-RUN] Would create user service at: /home/$USERNAME/.config/systemd/user/vi-local.service"
        log_info "[DRY-RUN] Would create system service at: /etc/systemd/system/vi-local.service"
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
ExecStart=$VENV_DIR/bin/python -O main.py --mode local --test
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
        # Backup existing services
        if [ "$WEB_SERVICE_EXISTS" = true ]; then
            backup_systemd_service "vi-web.service"
        fi
        if [ "$ASCII_SERVICE_EXISTS" = true ]; then
            backup_systemd_service "vi-ascii.service"
        fi
        if [ "$LOCAL_SERVICE_EXISTS" = true ]; then
            backup_systemd_service "vi-local.service"
        fi
        
        # Create/update services
        if [ "$WEB_SERVICE_EXISTS" = true ]; then
            log_info "Updating vi-web.service..."
        else
            log_info "Creating vi-web.service..."
        fi
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
        log_success "vi-web.service created/updated"

        if [ "$ASCII_SERVICE_EXISTS" = true ]; then
            log_info "Updating vi-ascii.service..."
        else
            log_info "Creating vi-ascii.service..."
        fi
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
        log_success "vi-ascii.service created/updated"

        if [ "$LOCAL_SERVICE_EXISTS" = true ]; then
            log_info "Updating vi-local.service..."
        else
            log_info "Creating vi-local.service..."
        fi
        # Local mode service - use user service for better GUI compatibility
        # Systemd user services work better for GUI applications
        USER_SERVICE_DIR="/home/$USERNAME/.config/systemd/user"
        if [ "$DRY_RUN" = false ] && [ -d "/home/$USERNAME" ]; then
            log_info "Creating user service for local mode (better GUI compatibility)..."
            mkdir -p "$USER_SERVICE_DIR"
            
            # Create user service
            cat <<EOF > "$USER_SERVICE_DIR/vi-local.service"
[Unit]
Description=VideoInterleaving (Local GUI)
After=graphical-session.target

[Service]
WorkingDirectory=$PROJECT_DIR
Environment=PYTHONUNBUFFERED=1
$ENV_DISPLAY
$ENV_BLOCK
ExecStart=$VENV_DIR/bin/python -O main.py --mode local --test
Restart=always
RestartSec=3
StandardOutput=append:$PROJECT_DIR/vi-local.log
StandardError=append:$PROJECT_DIR/vi-local.log

[Install]
WantedBy=default.target
EOF
            chown -R "$USERNAME:$USERNAME" "$USER_SERVICE_DIR/vi-local.service" 2>/dev/null || true
            log_success "User service created at $USER_SERVICE_DIR/vi-local.service"
            log_info "Enable with: systemctl --user enable --now vi-local.service"
        fi
        
        # Also create system service as fallback (but note it may not work for GUI)
        log_info "Creating system service for local mode (fallback)..."
        sudo tee "/etc/systemd/system/vi-local.service" > /dev/null <<EOF
[Unit]
Description=VideoInterleaving (Local GUI) - System Service
After=network.target graphical.target
Wants=graphical.target

[Service]
Type=simple
User=$USERNAME
WorkingDirectory=$PROJECT_DIR
Environment=PYTHONUNBUFFERED=1
$ENV_DISPLAY
$ENV_BLOCK
ExecStart=$VENV_DIR/bin/python -O main.py --mode local --test
Restart=on-failure
RestartSec=3
StandardOutput=append:$PROJECT_DIR/vi-local.log
StandardError=append:$PROJECT_DIR/vi-local.log

[Install]
WantedBy=graphical.target
EOF
        log_success "vi-local.service created/updated (system service)"
        log_warning "Note: For GUI applications, user service is recommended. Use: systemctl --user enable --now vi-local.service"

        log_info "Reloading systemd daemon..."
sudo systemctl daemon-reload
        log_success "Systemd daemon reloaded"
    fi
    
    echo ""
    log_info "To enable and start services, run:"
    echo "   👉 Web Mode:   systemctl enable --now vi-web"
    echo "   👉 ASCII Mode: systemctl enable --now vi-ascii"
    echo "   👉 Local Mode (recommended): systemctl --user enable --now vi-local"
    echo "   👉 Local Mode (fallback):    systemctl enable --now vi-local"
    echo ""
    log_info "For local mode, the user service is recommended for better GUI compatibility."
else
    log_step "⚙️  Skipping Systemd Services (not available on this system)"
    if [ "$OS" = "macos" ]; then
        log_info "On macOS, run manually: $VENV_DIR/bin/python -O main.py --mode <web|ascii|local>"
    else
        log_info "Run manually: $VENV_DIR/bin/python -O main.py --mode <web|ascii|local>"
    fi
fi

# --------------------------------------------
# 6. Firewall (Linux only)
# --------------------------------------------
if command -v ufw >/dev/null 2>&1; then
    log_step "🔥 Configuring Firewall..."
    
    # Check if firewall is active
    local firewall_status=$(ufw status 2>/dev/null | head -1 || echo "inactive")
    if echo "$firewall_status" | grep -q "inactive"; then
        log_warning "UFW firewall is inactive"
    else
        log_success "UFW firewall is active"
    fi
    
    local ports_to_add=(
        "1978/tcp:Monitor (WEB mode)"
        "1980/tcp:Monitor (ASCIIWEB mode)"
        "2323/tcp:ASCII Telnet"
        "2324/tcp:ASCII Monitor (ASCII mode)"
        "2423/tcp:ASCII-Web mode (primary port)"
        "2424/tcp:ASCII WebSocket (ASCIIWEB mode)"
        "8080/tcp:Web stream"
        "8888/tcp:Monitor (LOCAL mode)"
    )
    
    if [ "$DRY_RUN" = true ]; then
        for port_info in "${ports_to_add[@]}"; do
            local port=$(echo "$port_info" | cut -d: -f1)
            local desc=$(echo "$port_info" | cut -d: -f2)
            log_info "[DRY-RUN] Would run: sudo ufw allow $port  # $desc"
        done
    else
        local added_count=0
        for port_info in "${ports_to_add[@]}"; do
            local port=$(echo "$port_info" | cut -d: -f1)
            # Check if rule already exists
            if ufw status | grep -q "$port"; then
                log_verbose "Firewall rule for $port already exists"
            else
                sudo ufw allow "$port" >/dev/null 2>&1 && added_count=$((added_count + 1)) || true
            fi
        done
        if [ "$added_count" -gt 0 ]; then
            log_success "Added $added_count firewall rule(s)"
        else
            log_success "Firewall rules already configured"
        fi
    fi
fi

# --- FINAL SUMMARY ---
echo ""
echo "================================================================"
if [ "$DRY_RUN" = true ]; then
    log_success "DRY RUN Complete - No changes were made"
    log_info "Run without --dry-run to apply these changes"
else
    log_success "App Setup Complete"
    
    # Summary of what was done
    echo ""
    log_info "Summary:"
    if [ "$VENV_NEEDS_CREATE" = true ]; then
        echo "   ✅ Created Python virtual environment"
    elif [ "$VENV_NEEDS_UPDATE" = true ]; then
        echo "   ✅ Updated Python packages"
    else
        echo "   ⏭️  Python virtual environment (already up to date)"
    fi
    
    if [[ "$OSTYPE" == "linux-gnu"* ]] && command -v systemctl >/dev/null 2>&1; then
        if [ "$WEB_SERVICE_EXISTS" = true ] || [ "$ASCII_SERVICE_EXISTS" = true ] || [ "$LOCAL_SERVICE_EXISTS" = true ]; then
            echo "   ✅ Updated systemd services"
        else
            echo "   ✅ Created systemd services"
        fi
    fi
fi
echo "================================================================"
