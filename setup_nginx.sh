#!/bin/bash
set -e

# --- CONFIGURATION ---
NGINX_CONF="/etc/nginx/sites-available/videointerleaving"
LINK_CONF="/etc/nginx/sites-enabled/videointerleaving"
PROJECT_DIR=$(pwd)

echo ">>> 🌐Starting Nginx Web Server Setup..."

# 1. Install Nginx
if ! command -v nginx >/dev/null; then
    echo "    Installing Nginx..."
    apt-get update -qq && apt-get install -y nginx
fi

# 2. Ask for Domain Name
echo ""
echo "----------------------------------------------------------------"
read -p "Enter your domain name (e.g., mysite.com) or press ENTER for default: " DOMAIN_INPUT
DOMAIN_NAME=${DOMAIN_INPUT:-_}
echo "----------------------------------------------------------------"
echo "    Using Server Name: $DOMAIN_NAME"

# 3. Create Nginx Configuration
cat <<EOF > "$NGINX_CONF"
server {
    listen 80 default_server;
    listen [::]:80 default_server;
    server_name $DOMAIN_NAME;

    # --- Global Settings ---
    client_max_body_size 20M;
    add_header X-Content-Type-Options "nosniff" always;
    add_header X-Frame-Options "SAMEORIGIN" always;
    add_header Referrer-Policy "no-referrer-when-downgrade" always;

    # --- Root for Static Files ---
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
        proxy_set_header X-Forwarded-Proto \$scheme;
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
        proxy_set_header X-Forwarded-Proto \$scheme;
        proxy_buffering off;
        proxy_read_timeout 3600s;
    }

    # --- 4. ASCII Viewer Page (Proxied to Python) ---
    # This ensures {{ASCII_WIDTH}} templates are rendered correctly
    location /ascii/ {
        proxy_pass http://127.0.0.1:1980/ascii;
        proxy_http_version 1.1;
        proxy_set_header Host \$host;
        proxy_set_header X-Real-IP \$remote_addr;
        proxy_buffering off;
    }

    # --- 5. Static Assets ---
    location /static/ {
        alias $PROJECT_DIR/static/;
        expires 30d;
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

# 4. Enable the Site
echo "    Linking configuration..."
if [ -f /etc/nginx/sites-enabled/default ]; then
    rm /etc/nginx/sites-enabled/default
fi
ln -sf "$NGINX_CONF" "$LINK_CONF"

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
echo "   - Web Stream:    http://$DOMAIN_NAME/"
echo "   - Web Monitor:   http://$DOMAIN_NAME/monitor/"
echo "   - ASCII Viewer:  http://$DOMAIN_NAME/ascii/"
echo "   - ASCII Monitor: http://$DOMAIN_NAME/monitor_ascii/"
echo ""
if [ "$DOMAIN_NAME" != "_" ]; then
    echo "🔒 TO ENABLE HTTPS (SSL):"
    echo "   sudo certbot --nginx -d $DOMAIN_NAME"
fi
echo "================================================================"