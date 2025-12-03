#!/usr/bin/env python3
import time
import logging
import threading
import uuid
from flask import Flask, Response, render_template, request, jsonify

import settings
from shared_state import exchange

# Suppress Flask logging
log = logging.getLogger('werkzeug')
log.setLevel(logging.ERROR)

app = Flask(__name__, static_folder="static", template_folder="templates")

# Per-client heartbeat: id -> last_yield_monotonic
_client_heartbeats: dict[str, float] = {}
_hb_lock = threading.Lock()

def _set_heartbeat(cid: str) -> None:
    with _hb_lock:
        _client_heartbeats[cid] = time.monotonic()

def _clear_heartbeat(cid: str) -> None:
    with _hb_lock:
        _client_heartbeats.pop(cid, None)

def generate_mjpeg_stream(client_id: str):
    """Yields the latest frame as multipart MJPEG; updates heartbeat per yield."""
    interval = 1.0 / getattr(settings, 'SERVER_CAPTURE_RATE', 24)
    print(f"[STREAM] MJPEG client connected id={client_id}")
    try:
        while True:
            #time.sleep(interval)
            frame_data = exchange.get_frame()
            if not frame_data:
                continue

            # Mark that THIS client received a frame just now
            _set_heartbeat(client_id)

            # Emit one MJPEG part
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_data + b'\r\n')
    except GeneratorExit:
        # normal close
        pass
    except Exception as e:
        print(f"[STREAM] client id={client_id} error: {e}")
    finally:
        _clear_heartbeat(client_id)
        print(f"[STREAM] MJPEG client disconnected id={client_id}")

@app.route('/')
def index():
    return render_template("index.html")

@app.route('/video_feed')
def video_feed():
    # Require a client id for precise heartbeat tracking
    cid = request.args.get('id')
    if not cid:
        # Fallback: generate one so direct hits still work,
        # but client won't poll /stream_alive unless it knows the id.
        cid = uuid.uuid4().hex

    # Prime heartbeat so a brand-new client isn't immediately considered stalled
    _set_heartbeat(cid)

    resp = Response(
        generate_mjpeg_stream(cid),
        mimetype='multipart/x-mixed-replace; boundary=frame',
        direct_passthrough=True
    )
    # streaming-friendly headers
    resp.headers['Cache-Control'] = 'no-store'
    resp.headers['Pragma'] = 'no-cache'
    resp.headers['X-Accel-Buffering'] = 'no'  # nginx: don't buffer this route
    return resp

@app.route('/stream_alive')
def stream_alive():
    """Return server-side heartbeat for this viewer's stream."""
    cid = request.args.get('id', '')
    with _hb_lock:
        ts = _client_heartbeats.get(cid, 0.0)
        now = time.monotonic()
    # no caching; tiny JSON
    resp = jsonify({"ok": bool(ts), "ts": ts, "now": now})
    resp.headers['Cache-Control'] = 'no-store'
    return resp

def start_server():
    port = getattr(settings, 'STREAM_PORT', 8080)
    host = getattr(settings, 'STREAM_HOST', '127.0.0.1')
    print(f"Video Stream active: http://{host}:{port}")
    app.run(host=host, port=port, threaded=True, debug=False, use_reloader=False)
