#!/bin/bash
set -e

# --- CONFIGURATION ---
NGINX_AVAILABLE="/etc/nginx/sites-available/videointerleaving"
NGINX_ENABLED="/etc/nginx/sites-enabled/videointerleaving"
PROJECT_DIR=$(pwd)

echo ">>> 🌐 Starting Robust Nginx Setup..."

# 1. Install Nginx if missing
if ! command -v nginx >/dev/null; then
    echo "    Installing Nginx..."
    apt-get update -qq && apt-get install -y nginx
fi

# 2. Ask for Domain Name
echo ""
echo "----------------------------------------------------------------"
read -p "Enter your domain name (e.g., mysite.com): " DOMAIN_INPUT
DOMAIN_NAME=${DOMAIN_INPUT:-_}
echo "----------------------------------------------------------------"
echo "    Using Server Name: $DOMAIN_NAME and www.$DOMAIN_NAME"

# 3. Create Nginx Configuration
#    Changes:
#    - Added www.$DOMAIN_NAME to server_name
#    - Added explicit buffering disabling for all proxy blocks
cat <<EOF > "$NGINX_AVAILABLE"
server {
    listen 80 default_server;
    listen [::]:80 default_server;

    # 1. FIX WWW ERROR: Listen for both bare domain and www subdomain
    server_name $DOMAIN_NAME www.$DOMAIN_NAME;

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
    location /static/ {
        alias $PROJECT_DIR/static/;
        # Cache static files, but allow Nginx to bypass if changed
        expires 30d;
        try_files \$uri \$uri/ =404;
    }

    # --- 6. ASCII WebSocket Tunnel ---
    location /ascii_ws/ {
        proxy_pass http://127.0.0.1:2324/;
        proxy_http_version 1.1;

        proxy_set_header Upgrade \$http_upgrade;
        proxy_set_header Connection "Upgrade";
        proxy_set_header Host \$host;
        proxy_set_header X-Real-IP \$remote_addr;

        proxy_read_timeout 7d;
        proxy_buffering off;
    }
}
EOF

# 4. Enable the Site (Fixing Persistence)
echo "    Linking configuration..."

# A. Remove the default site (it often conflicts/overrides)
if [ -f /etc/nginx/sites-enabled/default ]; then
    echo "    Removing default Nginx site to prevent conflicts..."
    rm /etc/nginx/sites-enabled/default
fi

# B. Force create the symbolic link
#    'ln -sf' overwrites if it exists, ensuring it's always fresh
ln -sf "$NGINX_AVAILABLE" "$NGINX_ENABLED"

# 5. Test and Reload
echo "    Testing Nginx syntax..."
nginx -t

echo "    Reloading Nginx..."
systemctl reload nginx

# 6. Firewall
if command -v ufw >/dev/null; then
    echo "    Updating Firewall rules..."
    ufw allow 'Nginx Full' >/dev/null 2>&1
    ufw allow 2323/tcp >/dev/null 2>&1
    ufw allow 2324/tcp >/dev/null 2>&1
fi

echo ""
echo "================================================================"
echo "✅ Nginx Setup Complete!"
echo "================================================================"
echo "   - Main Site:     http://$DOMAIN_NAME/  (and www.$DOMAIN_NAME)"
echo "   - ASCII Viewer:  http://$DOMAIN_NAME/ascii/"
echo ""
if [ "$DOMAIN_NAME" != "_" ]; then
    echo "🔒 CRITICAL FINAL STEP FOR SSL:"
    echo "   Since we added 'www', you MUST run this command again:"
    echo "   sudo certbot --nginx -d $DOMAIN_NAME -d www.$DOMAIN_NAME"
fi
echo "================================================================"git gc --prune=now --aggressive