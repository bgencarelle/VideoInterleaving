#!/usr/bin/env python3
import time
import json
import http.server
import socketserver
import os
import glob
import sys

PORT = 1978

# --- THE UNIFIED DASHBOARD HTML ---
HTML_TEMPLATE = """
<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>Unified System Monitor</title>
  <style>
    body { background:#080808; color:#ccc; font-family:'Courier New', monospace; margin:0; padding:10px; }
    h1 { text-align:center; color:#444; margin: 5px 0; font-size:1.2em; text-transform:uppercase; letter-spacing:2px;}

    /* Grid Layout */
    .grid { display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 10px; height: 90vh; }

    .panel { 
      background: #111; border: 1px solid #333; padding: 10px; 
      display:flex; flex-direction:column; 
    }

    .panel.offline { opacity: 0.4; border-color:#300; }
    .panel.active { border-color: #0f0; box-shadow: 0 0 10px #003300; }

    .header { font-size: 1.5em; font-weight:bold; text-align:center; border-bottom:1px solid #333; padding-bottom:5px; margin-bottom:10px; }
    .local-head { color: #0af; }
    .web-head { color: #fa0; }
    .ascii-head { color: #0f0; }

    .row { display:flex; justify-content:space-between; margin-bottom:4px; font-size:0.9em; border-bottom:1px solid #1a1a1a;}
    .label { color: #666; }
    .val { color: #eee; font-weight:bold; }

    .cpu-bar { height:4px; background:#333; margin-top:2px; width:100%; }
    .cpu-fill { height:100%; background:#0f0; }

    .error-box { color: #f55; margin-top:auto; font-size:0.8em; border-top:1px solid #500; padding-top:5px;}
  </style>
</head>
<body>
  <h1>Video Interleaving Supervisor</h1>

  <div class="grid">
    <div class="panel" id="panel_local">
      <div class="header local-head">LOCAL</div>
      <div class="content" id="data_local">Waiting...</div>
    </div>

    <div class="panel" id="panel_web">
      <div class="header web-head">WEB (MJPEG)</div>
      <div class="content" id="data_web">Waiting...</div>
    </div>

    <div class="panel" id="panel_ascii">
      <div class="header ascii-head">ASCII</div>
      <div class="content" id="data_ascii">Waiting...</div>
    </div>
  </div>

  <script>
    const modes = ['local', 'web', 'ascii'];

    function renderPanel(mode, data) {
        const el = document.getElementById('data_' + mode);
        const panel = document.getElementById('panel_' + mode);

        if (!data || data.status === 'offline') {
            panel.classList.add('offline');
            panel.classList.remove('active');
            el.innerHTML = "<br><br><center>[ OFFLINE ]</center>";
            return;
        }

        panel.classList.remove('offline');
        panel.classList.add('active');

        // Key metrics to display top
        let html = `
          <div class="row"><span class="label">INDEX</span> <span class="val" style="font-size:1.4em">${data.index || 0}</span></div>
          <div class="row"><span class="label">FPS</span> <span class="val">${data.fps || 0}</span></div>
          <div class="row"><span class="label">UPTIME</span> <span class="val">${data.uptime || 0}</span></div>
          <div class="row"><span class="label">CPU</span> <span class="val">${data.cpu_percent}%</span></div>
          <div class="row"><span class="label">RAM</span> <span class="val">${data.mem_used}</span></div>
          <hr style="border:0; border-bottom:1px solid #333">
          <div class="row"><span class="label">Main Entropy</span> <span class="val">${data.main_entropy}</span></div>
          <div class="row"><span class="label">FIFO Depth</span> <span class="val">${data.fifo_depth}</span></div>
          <div class="row"><span class="label">Comp Slope</span> <span class="val">${(data.comp_slope || 0).toFixed(4)}</span></div>
        `;

        if (data.last_error) {
           html += `<div class="error-box">LAST ERROR: ${data.last_error}</div>`;
        }

        el.innerHTML = html;
    }

    function poll() {
        // We fetch a special endpoint that returns ALL data at once
        fetch('/all_data')
            .then(r => r.json())
            .then(combined => {
                modes.forEach(m => renderPanel(m, combined[m]));
            })
            .catch(e => console.log(e));
    }

    setInterval(poll, 1000);
    poll();
  </script>
</body>
</html>
"""


class MonitorHubHandler(http.server.BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        return

    def do_GET(self):
        if self.path == "/" or self.path == "/index.html":
            self.send_response(200)
            self.send_header('Content-Type', 'text/html; charset=utf-8')
            self.end_headers()
            self.wfile.write(HTML_TEMPLATE.encode('utf-8'))

        # NEW ENDPOINT: Returns { "local": {...}, "web": {...}, "ascii": {...} }
        elif self.path == "/all_data":
            response = {}
            for mode in ["local", "web", "ascii"]:
                fname = f"stats_{mode}.json"
                if os.path.exists(fname):
                    try:
                        # Check freshness (staleness check)
                        mtime = os.path.getmtime(fname)
                        if time.time() - mtime > 5:  # If file older than 5s, it's dead
                            response[mode] = {"status": "offline"}
                        else:
                            with open(fname, 'r') as f:
                                response[mode] = json.load(f)
                    except:
                        response[mode] = {"status": "error"}
                else:
                    response[mode] = {"status": "offline"}

            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps(response).encode('utf-8'))

        else:
            self.send_error(404)


class ThreadedTCPServer(socketserver.ThreadingTCPServer):
    allow_reuse_address = True
    daemon_threads = True


if __name__ == "__main__":
    print(f"[Hub] Unified Monitor running on http://localhost:{PORT}")
    server = ThreadedTCPServer(('127.0.0.1', PORT), MonitorHubHandler)
    server.serve_forever()