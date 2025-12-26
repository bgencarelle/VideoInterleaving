#!/bin/bash

# --- CRITICAL: SUDO CHECK FIRST ---
# Check if running as root, if not, re-exec with sudo
if [ "$(id -u)" -ne 0 ]; then
    echo "⚠️  Not running as root. Attempting to elevate with sudo..."
    echo "   (You may be prompted for your password)"
    # Use absolute path to script to handle spaces and special characters
    if command -v readlink >/dev/null 2>&1; then
        SCRIPT_PATH="$(readlink -f "$0" 2>/dev/null || realpath "$0" 2>/dev/null)"
    fi
    if [ -z "$SCRIPT_PATH" ] || [ ! -f "$SCRIPT_PATH" ]; then
        # Fallback: construct absolute path manually
        SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
        SCRIPT_NAME="$(basename "$0")"
        SCRIPT_PATH="$SCRIPT_DIR/$SCRIPT_NAME"
    fi
    # Preserve all arguments and re-exec with sudo
    exec sudo "$SCRIPT_PATH" "$@"
    # Should never reach here, but just in case:
    exit $?
fi

set -euo pipefail

# --- CONFIGURATION ---
NGINX_AVAILABLE="/etc/nginx/sites-available/videointerleaving"
NGINX_ENABLED="/etc/nginx/sites-enabled/videointerleaving"
PROJECT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

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

log_step "🌐 Starting Robust Nginx Setup..."
if [ "$DRY_RUN" = true ]; then
    log_warning "DRY RUN MODE - No changes will be made"
fi

# --- PORT DETECTION ---
# Ports used by the application (from server_config.py)
APP_PORTS=(
    1978  # Monitor (WEB mode)
    1980  # Monitor (ASCIIWEB mode)
    2323  # ASCII Telnet
    2324  # ASCII Monitor (ASCII mode - primary_port+1)
    2424  # ASCII WebSocket (ASCIIWEB mode)
    8080  # Web stream
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

detect_app_ports() {
    local ports_in_use=()
    local ports_available=()
    
    for port in "${APP_PORTS[@]}"; do
        if check_port_available "$port"; then
            ports_available+=("$port")
            log_verbose "Port $port is available"
        else
            ports_in_use+=("$port")
            local process=$(lsof -i ":$port" 2>/dev/null | tail -n +2 | awk '{print $1}' | head -1 || echo "unknown")
            log_verbose "Port $port is in use by: $process"
        fi
    done
    
    if [ ${#ports_in_use[@]} -gt 0 ]; then
        log_warning "Some application ports are in use: ${ports_in_use[*]}"
        log_info "This is normal if the application is already running"
    fi
}

# --------------------------------------------
# Pre-flight Validation
# --------------------------------------------
preflight_validation() {
    local errors=0
    local warnings=0
    
    log_step "🔍 Running Pre-flight Validation..."
    
    # Check if nginx is installed
    if ! command -v nginx >/dev/null 2>&1; then
        log_warning "nginx not found - will attempt to install"
        warnings=$((warnings + 1))
    else
        log_success "nginx found: $(nginx -v 2>&1 | head -1)"
    fi
    
    # Validate project directory
    if [ ! -d "$PROJECT_DIR" ]; then
        log_error "Project directory not found: $PROJECT_DIR"
        errors=$((errors + 1))
    else
        log_success "Project directory: $PROJECT_DIR"
    fi
    
    if [ ! -f "$PROJECT_DIR/main.py" ]; then
        log_warning "main.py not found - project may be incomplete"
        warnings=$((warnings + 1))
    else
        log_success "main.py found"
    fi
    
    # Check if nginx directories exist
    if [ ! -d "/etc/nginx/sites-available" ]; then
        log_error "/etc/nginx/sites-available not found"
        errors=$((errors + 1))
    else
        log_success "nginx sites-available directory exists"
    fi
    
    if [ ! -d "/etc/nginx/sites-enabled" ]; then
        log_error "/etc/nginx/sites-enabled not found"
        errors=$((errors + 1))
    else
        log_success "nginx sites-enabled directory exists"
    fi
    
    # Check port 80 availability
    if command -v netstat >/dev/null 2>&1 || command -v ss >/dev/null 2>&1; then
        port80_in_use=false
        port80_process=""
        
        if command -v netstat >/dev/null 2>&1; then
            if netstat -tuln 2>/dev/null | grep -q ":80 "; then
                port80_in_use=true
                port80_process=$(netstat -tulpn 2>/dev/null | grep ":80 " | head -1 | awk '{print $7}' || echo "unknown")
            fi
        elif command -v ss >/dev/null 2>&1; then
            if ss -tuln 2>/dev/null | grep -q ":80 "; then
                port80_in_use=true
                port80_process=$(ss -tulpn 2>/dev/null | grep ":80 " | head -1 | awk '{print $6}' || echo "unknown")
            fi
        fi
        
        if [ "$port80_in_use" = true ]; then
            if echo "$port80_process" | grep -q nginx; then
                log_success "Port 80 is in use by nginx"
            else
                log_warning "Port 80 is in use by: $port80_process"
                warnings=$((warnings + 1))
            fi
        else
            log_verbose "Port 80 is available"
        fi
    fi
    
    # Check application ports
    detect_app_ports
    
    # Summary
    if [ "$errors" -gt 0 ]; then
        echo ""
        log_error "Pre-flight validation failed with $errors error(s)"
        return 1
    elif [ "$warnings" -gt 0 ]; then
        echo ""
        log_warning "Pre-flight validation completed with $warnings warning(s)"
        return 0
    else
        echo ""
        log_success "Pre-flight validation passed"
        return 0
    fi
}

# Run pre-flight validation
if ! preflight_validation; then
    log_error "Exiting due to validation errors"
    exit 1
fi

# --------------------------------------------
# Existing Config Detection
# --------------------------------------------
detect_existing_config() {
    if [ -f "$NGINX_AVAILABLE" ]; then
        # Extract domain from existing config
        local existing_domain=$(grep -E "^\s*server_name\s+" "$NGINX_AVAILABLE" 2>/dev/null | head -1 | sed 's/.*server_name\s*\([^;]*\);.*/\1/' | awk '{print $1}' | sed 's/www\.//')
        if [ -n "$existing_domain" ] && [ "$existing_domain" != "_" ]; then
            echo "$existing_domain"
            return 0
        fi
    fi
    return 1
}

# Check if SSL/certbot is configured
check_ssl_configured() {
    if [ -f "$NGINX_AVAILABLE" ]; then
        if grep -q "listen 443" "$NGINX_AVAILABLE" 2>/dev/null || grep -q "ssl_certificate" "$NGINX_AVAILABLE" 2>/dev/null; then
            return 0
        fi
    fi
    return 1
}

# Extract SSL block from first server block only (HTTPS block)
extract_ssl_from_first_server_block() {
    local config_file="$1"
    # Use awk to extract SSL lines only from first server block
    # Match lines that are SSL-related and within the first server block
    awk '
        /^server \{/ { 
            if (block_count == 0) {
                in_first_block=1
            }
            block_count++
        }
        /^\}/ && in_first_block { 
            in_first_block=0
        }
        in_first_block && /listen.*443|ssl_certificate|ssl_certificate_key|include.*letsencrypt|ssl_dhparam/ {
            print $0
        }
    ' "$config_file" 2>/dev/null | sed 's/^/    /' || echo ""
}

# Check if site is enabled
check_site_enabled() {
    if [ -L "$NGINX_ENABLED" ] && [ -f "$NGINX_ENABLED" ]; then
        return 0
    fi
    return 1
}

# 1. Install Nginx and Certbot if missing
if ! command -v nginx >/dev/null 2>&1; then
    log_step "📦 Installing Nginx..."
    if [ "$DRY_RUN" = true ]; then
        log_info "[DRY-RUN] Would install nginx and certbot"
    else
        # Detect package manager
        # Note: We should be root at this point, but add explicit check
        if [ "$(id -u)" -ne 0 ]; then
            log_error "Script must run as root for package installation"
            exit 1
        fi
        if command -v apt-get >/dev/null 2>&1; then
            apt-get update -qq
            apt-get install -y nginx python3-certbot-nginx
        elif command -v yum >/dev/null 2>&1; then
            yum install -y nginx python3-certbot-nginx
        elif command -v dnf >/dev/null 2>&1; then
            dnf install -y nginx python3-certbot-nginx
        else
            log_error "Cannot determine package manager for nginx installation"
            exit 1
        fi
        log_success "Nginx and certbot-nginx plugin installed"
    fi
else
    log_success "Nginx already installed: $(nginx -v 2>&1 | head -1)"
    
    # Check if certbot-nginx plugin is installed
    if ! python3 -c "import certbot_nginx" 2>/dev/null; then
        log_info "Installing certbot-nginx plugin..."
        if [ "$DRY_RUN" = true ]; then
            log_info "[DRY-RUN] Would install python3-certbot-nginx"
        else
            if [ "$(id -u)" -ne 0 ]; then
                log_error "Script must run as root for package installation"
                exit 1
            fi
            if command -v apt-get >/dev/null 2>&1; then
                apt-get install -y python3-certbot-nginx
            elif command -v yum >/dev/null 2>&1; then
                yum install -y python3-certbot-nginx
            elif command -v dnf >/dev/null 2>&1; then
                dnf install -y python3-certbot-nginx
            fi
            log_success "certbot-nginx plugin installed"
        fi
    else
        log_success "certbot-nginx plugin already installed"
    fi
fi

# 2. Detect existing configuration
EXISTING_DOMAIN=""
if detect_existing_config; then
    EXISTING_DOMAIN=$(detect_existing_config)
    log_step "🔍 Existing configuration detected"
    log_info "Domain: $EXISTING_DOMAIN"
    if check_ssl_configured; then
        log_success "SSL: Configured"
    fi
    if check_site_enabled; then
        log_success "Status: Enabled"
    else
        log_info "Status: Not enabled"
    fi
fi

# 3. Ask for Domain Name (skip if already configured and no override)
if [ -n "$EXISTING_DOMAIN" ] && [ -z "${FORCE_DOMAIN_UPDATE:-}" ] && [ "$DRY_RUN" = false ]; then
    echo ""
    log_info "Using existing domain: $EXISTING_DOMAIN"
    log_info "(Set FORCE_DOMAIN_UPDATE=1 to change)"
    DOMAIN_NAME="$EXISTING_DOMAIN"
else
    if [ "$DRY_RUN" = true ]; then
        DOMAIN_NAME="${EXISTING_DOMAIN:-example.com}"
        log_info "[DRY-RUN] Would prompt for domain name (using: $DOMAIN_NAME for preview)"
    else
        echo ""
        echo "----------------------------------------------------------------"
        if [ -n "$EXISTING_DOMAIN" ]; then
            read -p "Enter your domain name [current: $EXISTING_DOMAIN]: " DOMAIN_INPUT
        else
            read -p "Enter your domain name (e.g., mysite.com): " DOMAIN_INPUT
        fi
        DOMAIN_NAME=${DOMAIN_INPUT:-${EXISTING_DOMAIN:-_}}
        echo "----------------------------------------------------------------"
    fi
fi

# Validate domain name format
validate_domain() {
    local domain=$1
    if [ "$domain" = "_" ]; then
        return 0  # Default/placeholder is valid
    fi
    # Basic domain validation
    if echo "$domain" | grep -qE '^([a-zA-Z0-9]([a-zA-Z0-9\-]{0,61}[a-zA-Z0-9])?\.)+[a-zA-Z]{2,}$'; then
        return 0
    fi
    return 1
}

if ! validate_domain "$DOMAIN_NAME"; then
    log_warning "Domain name format may be invalid: $DOMAIN_NAME"
    log_info "Continuing anyway (use '_' for default server)"
fi

log_info "Using Server Name: $DOMAIN_NAME and www.$DOMAIN_NAME"

# 4. Remove old site configuration and backup existing config
SSL_BLOCK=""
SSL_ENABLED=false

# Remove old symlink in sites-enabled (if it exists)
if [ -L "$NGINX_ENABLED" ] || [ -f "$NGINX_ENABLED" ]; then
    log_step "🗑️  Removing old site configuration..."
    if [ "$DRY_RUN" = true ]; then
        log_info "[DRY-RUN] Would remove old symlink: $NGINX_ENABLED"
    else
        # Check if it's a symlink pointing to a different location
        if [ -L "$NGINX_ENABLED" ]; then
            current_target=$(readlink "$NGINX_ENABLED")
            if [ "$current_target" != "$NGINX_AVAILABLE" ]; then
                log_info "Removing old symlink pointing to: $current_target"
            fi
        fi
        rm -f "$NGINX_ENABLED"
        log_success "Old site configuration removed"
    fi
fi

# Backup existing config if it exists
if [ -f "$NGINX_AVAILABLE" ]; then
    log_step "💾 Backing up existing configuration..."
    if [ "$DRY_RUN" = true ]; then
        log_info "[DRY-RUN] Would backup $NGINX_AVAILABLE"
    else
        BACKUP_DIR="/etc/nginx/sites-available/videointerleaving-backups"
        mkdir -p "$BACKUP_DIR"
        timestamp=$(date +%Y%m%d_%H%M%S)
        cp "$NGINX_AVAILABLE" "$BACKUP_DIR/videointerleaving.conf.$timestamp"
        log_success "Backup saved to $BACKUP_DIR/videointerleaving.conf.$timestamp"
        # Verify backup was created
        if [ ! -f "$BACKUP_DIR/videointerleaving.conf.$timestamp" ]; then
            log_warning "Backup may have failed, but continuing..."
        fi
    fi
    
    # Extract SSL settings if present (Certbot managed)
    if check_ssl_configured; then
        log_info "Preserving SSL configuration (Certbot managed)..."
        # Extract SSL-related lines from FIRST server block only (HTTPS block)
        SSL_BLOCK=$(extract_ssl_from_first_server_block "$NGINX_AVAILABLE")
        if [ -n "$SSL_BLOCK" ]; then
            SSL_ENABLED=true
            log_verbose "Extracted SSL block from first server block"
        else
            SSL_ENABLED=false
            log_warning "SSL detected but extraction failed, will be recreated by Certbot"
        fi
    else
        SSL_ENABLED=false
    fi
fi

# 5. Create Nginx Configuration
log_step "📝 Creating Nginx Configuration..."
if [ "$DRY_RUN" = true ]; then
    log_info "[DRY-RUN] Would create/update $NGINX_AVAILABLE"
    echo "[DRY-RUN] Configuration preview:"
    echo "---"
fi

# Create config (or show in dry-run)
# Note: SSL configuration (listen 443, ssl_certificate, etc.) is managed by Certbot
# and will be added automatically when SSL is configured
if [ "$DRY_RUN" = true ]; then
    cat <<EOF | sed 's/^/[DRY-RUN] /'
server {
    # 1. FIX WWW ERROR: Listen for both bare domain and www subdomain
    server_name $DOMAIN_NAME www.$DOMAIN_NAME;
$([ "$SSL_ENABLED" != "true" ] && echo "    # Listen directives (HTTP only - removed when SSL is configured)")
$([ "$SSL_ENABLED" != "true" ] && echo "    listen 80 default_server;")
$([ "$SSL_ENABLED" != "true" ] && echo "    listen [::]:80 default_server;")

    # --- Global Settings ---
    client_max_body_size 20M;
    root $PROJECT_DIR;
    index index.html;

    # --- Redirects ---
    location = /monitor { return 301 /monitor/; }
    location = /monitor_ascii { return 301 /monitor_ascii/; }
    location = /ascii { return 301 /ascii/; }

    # --- 1. Main Stream (Web Mode) ---
    location / {
        proxy_pass http://127.0.0.1:8080;
        proxy_http_version 1.1;
        proxy_set_header Connection "";
        proxy_set_header Host \$host;
        proxy_set_header X-Real-IP \$remote_addr;
        proxy_set_header X-Forwarded-For \$proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto \$scheme;

        # 3. FIX BUFFERING: Kill all buffers for real-time streaming
        proxy_buffering off;
        proxy_cache off;
        proxy_request_buffering off;
        proxy_read_timeout 7d;
        sendfile off;
        tcp_nodelay on;
        gzip off;
    }

    # --- 2. Web Monitor Dashboard ---
    location /monitor/ {
        proxy_pass http://127.0.0.1:1978/;
        proxy_http_version 1.1;
        proxy_set_header Host \$host;
        proxy_set_header X-Real-IP \$remote_addr;
        proxy_set_header X-Forwarded-For \$proxy_add_x_forwarded_for;
        proxy_buffering off;
        proxy_read_timeout 3600s;
    }

    # --- 3. ASCII Monitor Dashboard ---
    location /monitor_ascii/ {
        proxy_pass http://127.0.0.1:1980/;
        proxy_http_version 1.1;
        proxy_set_header Host \$host;
        proxy_set_header X-Real-IP \$remote_addr;
        proxy_set_header X-Forwarded-For \$proxy_add_x_forwarded_for;
        proxy_buffering off;
        proxy_read_timeout 3600s;
    }

    # --- 4. ASCII Viewer Page (Proxied to Python) ---
    location /ascii/ {
        proxy_pass http://127.0.0.1:1980/ascii;
        proxy_http_version 1.1;
        proxy_set_header Host \$host;
        proxy_buffering off;
    }

    # --- 5. Static Assets ---
    # Proxy to Python because 'www-data' cannot read user home dir
    location /static/ {
        proxy_pass http://127.0.0.1:1978;
        proxy_set_header Host \$host;
        proxy_set_header X-Real-IP \$remote_addr;
        proxy_buffering off;
    }

    # --- 6. ASCII WebSocket Tunnel ---
    location /ascii_ws/ {
        proxy_pass http://127.0.0.1:2424/;
        proxy_http_version 1.1;

        proxy_set_header Upgrade \$http_upgrade;
        proxy_set_header Connection "Upgrade";
        proxy_set_header Host \$host;
        proxy_set_header X-Real-IP \$remote_addr;

        proxy_read_timeout 7d;
        proxy_buffering off;
    }
${SSL_BLOCK:-}
}
EOF
    echo "[DRY-RUN] ---"
else
    cat <<EOF > "$NGINX_AVAILABLE"
server {
    # 1. FIX WWW ERROR: Listen for both bare domain and www subdomain
    server_name $DOMAIN_NAME www.$DOMAIN_NAME;
$([ "$SSL_ENABLED" != "true" ] && echo "    # Listen directives (HTTP only - removed when SSL is configured)")
$([ "$SSL_ENABLED" != "true" ] && echo "    listen 80 default_server;")
$([ "$SSL_ENABLED" != "true" ] && echo "    listen [::]:80 default_server;")

    # --- Global Settings ---
    client_max_body_size 20M;
    root $PROJECT_DIR;
    index index.html;

    # --- Redirects ---
    location = /monitor { return 301 /monitor/; }
    location = /monitor_ascii { return 301 /monitor_ascii/; }
    location = /ascii { return 301 /ascii/; }

    # --- 1. Main Stream (Web Mode) ---
    location / {
        proxy_pass http://127.0.0.1:8080;
        proxy_http_version 1.1;
        proxy_set_header Connection "";
        proxy_set_header Host \$host;
        proxy_set_header X-Real-IP \$remote_addr;
        proxy_set_header X-Forwarded-For \$proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto \$scheme;

        # 3. FIX BUFFERING: Kill all buffers for real-time streaming
        proxy_buffering off;
        proxy_cache off;
        proxy_request_buffering off;
        proxy_read_timeout 7d;
        sendfile off;
        tcp_nodelay on;
        gzip off;
    }

    # --- 2. Web Monitor Dashboard ---
    location /monitor/ {
        proxy_pass http://127.0.0.1:1978/;
        proxy_http_version 1.1;
        proxy_set_header Host \$host;
        proxy_set_header X-Real-IP \$remote_addr;
        proxy_set_header X-Forwarded-For \$proxy_add_x_forwarded_for;
        proxy_buffering off;
        proxy_read_timeout 3600s;
    }

    # --- 3. ASCII Monitor Dashboard ---
    location /monitor_ascii/ {
        proxy_pass http://127.0.0.1:1980/;
        proxy_http_version 1.1;
        proxy_set_header Host \$host;
        proxy_set_header X-Real-IP \$remote_addr;
        proxy_set_header X-Forwarded-For \$proxy_add_x_forwarded_for;
        proxy_buffering off;
        proxy_read_timeout 3600s;
    }

    # --- 4. ASCII Viewer Page (Proxied to Python) ---
    location /ascii/ {
        proxy_pass http://127.0.0.1:1980/ascii;
        proxy_http_version 1.1;
        proxy_set_header Host \$host;
        proxy_buffering off;
    }

    # --- 5. Static Assets ---
    # Proxy to Python because 'www-data' cannot read user home dir
    location /static/ {
        proxy_pass http://127.0.0.1:1978;
        proxy_set_header Host \$host;
        proxy_set_header X-Real-IP \$remote_addr;
        proxy_buffering off;
    }

    # --- 6. ASCII WebSocket Tunnel ---
    location /ascii_ws/ {
        proxy_pass http://127.0.0.1:2424/;
        proxy_http_version 1.1;

        proxy_set_header Upgrade \$http_upgrade;
        proxy_set_header Connection "Upgrade";
        proxy_set_header Host \$host;
        proxy_set_header X-Real-IP \$remote_addr;

        proxy_read_timeout 7d;
        proxy_buffering off;
    }
${SSL_BLOCK:-}
}
EOF
    # Verify config was written successfully
    if [ -f "$NGINX_AVAILABLE" ] && [ -s "$NGINX_AVAILABLE" ]; then
        log_success "Nginx configuration created/updated"
        log_verbose "Config file size: $(wc -l < "$NGINX_AVAILABLE") lines"
    else
        log_error "Failed to write Nginx configuration file"
        exit 1
    fi
fi

# 6. Enable the Site
log_info "Linking configuration..."

# A. Remove the default site (it often conflicts/overrides)
if [ -f /etc/nginx/sites-enabled/default ]; then
    if [ "$DRY_RUN" = true ]; then
        log_info "[DRY-RUN] Would remove /etc/nginx/sites-enabled/default"
    else
        log_info "Removing default Nginx site to prevent conflicts..."
        rm -f /etc/nginx/sites-enabled/default
        log_success "Default site removed"
    fi
fi

# B. Create the symbolic link
#    Old symlink was already removed in step 4, so this creates a fresh one
if [ "$DRY_RUN" = true ]; then
    log_info "[DRY-RUN] Would create symlink: $NGINX_ENABLED -> $NGINX_AVAILABLE"
else
    ln -sf "$NGINX_AVAILABLE" "$NGINX_ENABLED"
    log_success "Configuration enabled"
fi

# 7. Test and Reload
if [ "$DRY_RUN" = true ]; then
    log_info "[DRY-RUN] Would test nginx configuration: nginx -t"
    log_info "[DRY-RUN] Would reload nginx: systemctl reload nginx"
else
    log_info "Testing Nginx syntax..."
    if nginx -t; then
        log_success "Nginx configuration is valid"
        
        log_info "Reloading Nginx..."
        if systemctl reload nginx; then
            log_success "Nginx reloaded successfully"
        else
            log_error "Failed to reload nginx"
            exit 1
        fi
    else
        log_error "Nginx configuration test failed"
        log_info "Configuration file: $NGINX_AVAILABLE"
        exit 1
    fi
fi

# 8. Firewall
if command -v ufw >/dev/null 2>&1; then
    log_step "🔥 Configuring Firewall..."
    
    local firewall_rules=(
        "Nginx Full:Nginx Full"
        "1978/tcp:Monitor (WEB mode)"
        "1980/tcp:Monitor (ASCIIWEB mode)"
        "2323/tcp:ASCII Telnet"
        "2324/tcp:ASCII Monitor (ASCII mode)"
        "2424/tcp:ASCII WebSocket (ASCIIWEB mode)"
        "8080/tcp:Web stream"
        "8888/tcp:Monitor (LOCAL mode)"
    )
    
    if [ "$DRY_RUN" = true ]; then
        for rule_info in "${firewall_rules[@]}"; do
            local rule=$(echo "$rule_info" | cut -d: -f1)
            local desc=$(echo "$rule_info" | cut -d: -f2)
            log_info "[DRY-RUN] Would run: ufw allow $rule  # $desc"
        done
    else
        local added_count=0
        for rule_info in "${firewall_rules[@]}"; do
            local rule=$(echo "$rule_info" | cut -d: -f1)
            # Check if rule already exists
            if ufw status | grep -q "$rule"; then
                log_verbose "Firewall rule for $rule already exists"
            else
                ufw allow "$rule" >/dev/null 2>&1 && added_count=$((added_count + 1)) || true
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
    log_success "Nginx Setup Complete!"
    
    echo ""
    log_info "Summary:"
    if [ -f "$NGINX_AVAILABLE" ] && [ ! -f "$NGINX_AVAILABLE.backup" ]; then
        echo "   ✅ Created Nginx configuration"
    elif [ -f "$NGINX_AVAILABLE" ]; then
        echo "   ✅ Updated Nginx configuration (backup created)"
    fi
    if check_site_enabled; then
        echo "   ✅ Site is enabled"
    fi
    if check_ssl_configured; then
        echo "   ✅ SSL configuration preserved"
    fi
fi

echo ""
log_info "Access URLs:"
echo "   - Main Site:     http://$DOMAIN_NAME/  (and www.$DOMAIN_NAME)"
echo "   - ASCII Viewer:  http://$DOMAIN_NAME/ascii/"

if [ "$DOMAIN_NAME" != "_" ] && [ "$DRY_RUN" = false ]; then
    echo ""
    log_warning "CRITICAL FINAL STEP FOR SSL:"
    log_info "Since we added 'www', you MUST run this command again:"
    echo "   sudo certbot --nginx -d $DOMAIN_NAME -d www.$DOMAIN_NAME"
fi
echo "================================================================"