#!/usr/bin/env python3
"""
Independent Multi-Mode Monitor

Launches independently of all other servers and monitors:
- WEB mode (monitor + stream)
- LOCAL mode (monitor only)
- ASCII mode (telnet + stats)
- ASCIIWEB mode (websocket + monitor)

Checks server status via HTTP requests and port checks.
"""
import time
import json
import http.server
import socketserver
import socket
import urllib.request
import urllib.error
import threading
import sys
from server_config import ServerConfig, MODE_WEB, MODE_LOCAL, MODE_ASCII, MODE_ASCIIWEB

# Default ports for each mode (from server_config.py defaults)
MODE_PORTS = {
    MODE_WEB: {
        'monitor': 1978,
        'stream': 8080,
    },
    MODE_LOCAL: {
        'monitor': 8888,
    },
    MODE_ASCII: {
        'telnet': 2323,
        'monitor': 2324,  # primary_port + 1
    },
    MODE_ASCIIWEB: {
        'monitor': 1980,
        'websocket': 2424,
    }
}

# Status cache (updated by background thread)
_status_cache = {}
_cache_lock = threading.Lock()
_last_update = 0

HTML_TEMPLATE = """
<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>Multi-Mode System Monitor</title>
  <style>
    body { 
      background:#080808; 
      color:#ccc; 
      font-family:'Courier New', monospace; 
      margin:0; 
      padding:10px; 
    }
    h1 { 
      text-align:center; 
      color:#444; 
      margin: 5px 0; 
      font-size:1.2em; 
      text-transform:uppercase; 
      letter-spacing:2px;
    }

    /* Grid Layout - 4 panels */
    .grid { 
      display: grid; 
      grid-template-columns: 1fr 1fr; 
      grid-template-rows: 1fr 1fr;
      gap: 10px; 
      height: 90vh; 
    }

    .panel { 
      background: #111; 
      border: 2px solid #333; 
      padding: 10px; 
      display:flex; 
      flex-direction:column; 
      border-radius: 4px;
    }

    .panel.offline { 
      opacity: 0.4; 
      border-color:#500; 
    }
    .panel.active { 
      border-color: #0f0; 
      box-shadow: 0 0 15px #003300; 
    }
    .panel.partial { 
      border-color: #fa0; 
      box-shadow: 0 0 10px #333300; 
    }

    .header { 
      font-size: 1.3em; 
      font-weight:bold; 
      text-align:center; 
      border-bottom:1px solid #333; 
      padding-bottom:5px; 
      margin-bottom:10px; 
    }
    .web-head { color: #fa0; }
    .local-head { color: #0af; }
    .ascii-head { color: #0f0; }
    .asciiweb-head { color: #f0f; }

    .row { 
      display:flex; 
      justify-content:space-between; 
      margin-bottom:4px; 
      font-size:0.9em; 
      border-bottom:1px solid #1a1a1a;
      padding: 2px 0;
    }
    .label { color: #666; }
    .val { color: #eee; font-weight:bold; }
    .status { 
      display: inline-block;
      padding: 2px 8px;
      border-radius: 3px;
      font-size: 0.8em;
      font-weight: bold;
    }
    .status.online { background: #0f0; color: #000; }
    .status.offline { background: #500; color: #fff; }
    .status.partial { background: #fa0; color: #000; }

    .servers { 
      margin-top: 10px;
      padding-top: 10px;
      border-top: 1px solid #333;
    }
    .server-item {
      display: flex;
      justify-content: space-between;
      margin: 4px 0;
      font-size: 0.85em;
    }

    .error-box { 
      color: #f55; 
      margin-top:auto; 
      font-size:0.8em; 
      border-top:1px solid #500; 
      padding-top:5px;
      margin-top: 10px;
    }
  </style>
</head>
<body>
  <h1>Multi-Mode System Monitor</h1>
  <div style="text-align:center; color:#666; font-size:0.8em; margin-bottom:10px;">
    Independent monitor - checks all modes regardless of which is running
  </div>

  <div class="grid">
    <div class="panel" id="panel_web">
      <div class="header web-head">WEB (MJPEG)</div>
      <div class="content" id="data_web">Checking...</div>
    </div>

    <div class="panel" id="panel_local">
      <div class="header local-head">LOCAL</div>
      <div class="content" id="data_local">Checking...</div>
    </div>

    <div class="panel" id="panel_ascii">
      <div class="header ascii-head">ASCII (Telnet)</div>
      <div class="content" id="data_ascii">Checking...</div>
    </div>

    <div class="panel" id="panel_asciiweb">
      <div class="header asciiweb-head">ASCIIWEB (WebSocket)</div>
      <div class="content" id="data_asciiweb">Checking...</div>
    </div>
  </div>

  <script>
    const modes = ['web', 'local', 'ascii', 'asciiweb'];

    function renderPanel(mode, data) {
        const el = document.getElementById('data_' + mode);
        const panel = document.getElementById('panel_' + mode);

        if (!data || data.status === 'offline') {
            panel.classList.remove('active', 'partial');
            panel.classList.add('offline');
            el.innerHTML = "<br><br><center><span class='status offline'>OFFLINE</span></center>";
            return;
        }

        // Determine panel state
        const allOnline = data.servers && Object.values(data.servers).every(s => s.online);
        const someOnline = data.servers && Object.values(data.servers).some(s => s.online);

        if (allOnline) {
            panel.classList.remove('offline', 'partial');
            panel.classList.add('active');
        } else if (someOnline) {
            panel.classList.remove('offline', 'active');
            panel.classList.add('partial');
        } else {
            panel.classList.remove('active', 'partial');
            panel.classList.add('offline');
        }

        // Build HTML
        let html = '';
        
        // Status indicator
        if (allOnline) {
            html += `<div style="text-align:center; margin-bottom:10px;"><span class="status online">ALL ONLINE</span></div>`;
        } else if (someOnline) {
            html += `<div style="text-align:center; margin-bottom:10px;"><span class="status partial">PARTIAL</span></div>`;
        }

        // Server status
        if (data.servers) {
            html += '<div class="servers">';
            for (const [name, info] of Object.entries(data.servers)) {
                const statusClass = info.online ? 'online' : 'offline';
                const statusText = info.online ? 'ONLINE' : 'OFFLINE';
                html += `
                  <div class="server-item">
                    <span class="label">${name}:</span>
                    <span class="status ${statusClass}">${statusText}</span>
                  </div>
                `;
            }
            html += '</div>';
        }

        // Metrics (if available)
        if (data.metrics) {
            html += '<hr style="border:0; border-bottom:1px solid #333; margin:10px 0;">';
            const metrics = data.metrics;
            if (metrics.index !== undefined) {
                html += `<div class="row"><span class="label">INDEX</span> <span class="val">${metrics.index}</span></div>`;
            }
            if (metrics.fps !== undefined) {
                html += `<div class="row"><span class="label">FPS</span> <span class="val">${metrics.fps}</span></div>`;
            }
            if (metrics.cpu_percent !== undefined) {
                html += `<div class="row"><span class="label">CPU</span> <span class="val">${metrics.cpu_percent}%</span></div>`;
            }
            if (metrics.mem_used !== undefined) {
                html += `<div class="row"><span class="label">RAM</span> <span class="val">${metrics.mem_used}</span></div>`;
            }
        }

        if (data.error) {
            html += `<div class="error-box">ERROR: ${data.error}</div>`;
        }

        el.innerHTML = html;
    }

    function poll() {
        fetch('/all_data')
            .then(r => r.json())
            .then(combined => {
                modes.forEach(m => renderPanel(m, combined[m]));
            })
            .catch(e => {
                console.error('Poll error:', e);
            });
    }

    setInterval(poll, 2000);  // Poll every 2 seconds
    poll();
  </script>
</body>
</html>
"""


def check_port(host, port, timeout=0.5):
    """Check if a port is open (server is running)."""
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(timeout)
        result = sock.connect_ex((host, port))
        sock.close()
        return result == 0
    except:
        return False


def check_http_endpoint(host, port, path='/data', timeout=1.0):
    """Check if HTTP endpoint is accessible and return data if available."""
    try:
        url = f'http://{host}:{port}{path}'
        req = urllib.request.Request(url)
        with urllib.request.urlopen(req, timeout=timeout) as response:
            if response.status == 200:
                data = json.loads(response.read().decode('utf-8'))
                return True, data
            return False, None
    except (urllib.error.URLError, urllib.error.HTTPError, socket.timeout, json.JSONDecodeError):
        return False, None


def check_mode_status(mode):
    """Check status of all servers for a given mode."""
    ports = MODE_PORTS.get(mode, {})
    status = {
        'status': 'offline',
        'servers': {},
        'metrics': None,
        'error': None
    }

    try:
        if mode == MODE_WEB:
            # Check monitor server
            monitor_online, monitor_data = check_http_endpoint('127.0.0.1', ports['monitor'], '/data')
            status['servers']['Monitor'] = {'online': monitor_online, 'port': ports['monitor']}
            
            # Check stream server
            stream_online = check_port('127.0.0.1', ports['stream'])
            status['servers']['Stream'] = {'online': stream_online, 'port': ports['stream']}
            
            if monitor_online and monitor_data:
                status['metrics'] = monitor_data
                status['status'] = 'online' if (monitor_online and stream_online) else 'partial'
            elif stream_online:
                status['status'] = 'partial'
            else:
                status['status'] = 'offline'

        elif mode == MODE_LOCAL:
            # Check monitor server
            monitor_online, monitor_data = check_http_endpoint('127.0.0.1', ports['monitor'], '/data')
            status['servers']['Monitor'] = {'online': monitor_online, 'port': ports['monitor']}
            
            if monitor_online and monitor_data:
                status['metrics'] = monitor_data
                status['status'] = 'online'
            else:
                status['status'] = 'offline'

        elif mode == MODE_ASCII:
            # Check telnet server (port check only)
            telnet_online = check_port('127.0.0.1', ports['telnet'])
            status['servers']['Telnet'] = {'online': telnet_online, 'port': ports['telnet']}
            
            # Check stats monitor server
            stats_online, stats_data = check_http_endpoint('127.0.0.1', ports['monitor'], '/data')
            status['servers']['Stats'] = {'online': stats_online, 'port': ports['monitor']}
            
            if telnet_online or stats_online:
                status['status'] = 'partial' if not (telnet_online and stats_online) else 'online'
            else:
                status['status'] = 'offline'

        elif mode == MODE_ASCIIWEB:
            # Check monitor server
            monitor_online, monitor_data = check_http_endpoint('127.0.0.1', ports['monitor'], '/data')
            status['servers']['Monitor'] = {'online': monitor_online, 'port': ports['monitor']}
            
            # Check websocket server (port check only - can't easily test WS via HTTP)
            ws_online = check_port('127.0.0.1', ports['websocket'])
            status['servers']['WebSocket'] = {'online': ws_online, 'port': ports['websocket']}
            
            if monitor_online and monitor_data:
                status['metrics'] = monitor_data
                status['status'] = 'online' if (monitor_online and ws_online) else 'partial'
            elif ws_online:
                status['status'] = 'partial'
            else:
                status['status'] = 'offline'

    except Exception as e:
        status['error'] = str(e)
        status['status'] = 'error'

    return status


def update_status_cache():
    """Background thread that periodically updates status cache."""
    global _status_cache, _last_update
    while True:
        try:
            new_cache = {}
            for mode in [MODE_WEB, MODE_LOCAL, MODE_ASCII, MODE_ASCIIWEB]:
                new_cache[mode] = check_mode_status(mode)
            
            with _cache_lock:
                _status_cache = new_cache
                _last_update = time.time()
        except Exception as e:
            print(f"[Multimonitor] Update error: {e}", file=sys.stderr)
        
        time.sleep(2)  # Update every 2 seconds


class MultimonitorHandler(http.server.BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        return  # Suppress logging

    def do_GET(self):
        if self.path == "/" or self.path == "/index.html":
            self.send_response(200)
            self.send_header('Content-Type', 'text/html; charset=utf-8')
            self.end_headers()
            self.wfile.write(HTML_TEMPLATE.encode('utf-8'))

        elif self.path == "/all_data":
            with _cache_lock:
                # Convert mode constants to lowercase for frontend
                response = {
                    'web': _status_cache.get(MODE_WEB, {'status': 'offline'}),
                    'local': _status_cache.get(MODE_LOCAL, {'status': 'offline'}),
                    'ascii': _status_cache.get(MODE_ASCII, {'status': 'offline'}),
                    'asciiweb': _status_cache.get(MODE_ASCIIWEB, {'status': 'offline'}),
                }
            
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            self.wfile.write(json.dumps(response).encode('utf-8'))

        else:
            self.send_error(404)


class ThreadedTCPServer(socketserver.ThreadingTCPServer):
    allow_reuse_address = True
    daemon_threads = True


if __name__ == "__main__":
    # Start background status update thread
    update_thread = threading.Thread(target=update_status_cache, daemon=True, name="StatusUpdater")
    update_thread.start()
    
    # Default port for multimonitor (different from all other servers)
    MULTIMONITOR_PORT = 1999
    
    print(f"[Multimonitor] Starting independent monitor on http://localhost:{MULTIMONITOR_PORT}")
    print(f"[Multimonitor] Monitoring all modes: WEB, LOCAL, ASCII, ASCIIWEB")
    print(f"[Multimonitor] Access at: http://localhost:{MULTIMONITOR_PORT}")
    
    server = ThreadedTCPServer(('127.0.0.1', MULTIMONITOR_PORT), MultimonitorHandler)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[Multimonitor] Shutting down...")
        sys.exit(0)

