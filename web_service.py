# web_service.py
import time
import logging
from flask import Flask, Response, render_template

import settings
from shared_state import exchange

# Suppress Flask logging
log = logging.getLogger('werkzeug')
log.setLevel(logging.ERROR)

# templates/ and static/ are the defaults; this just makes it explicit
app = Flask(__name__, static_folder="static", template_folder="templates")


def generate_mjpeg_stream():
    """Yields the latest frame from the exchange as an MJPEG stream."""
    interval = 1.0 / getattr(settings, 'SERVER_CAPTURE_RATE', 24)
    print("[STREAM] MJPEG client connected")
    while True:
        time.sleep(interval)

        frame_data = exchange.get_frame()
        if frame_data:
            # You can uncomment this for debugging frame size:
            # print(f"[STREAM] yielding frame of size {len(frame_data)} bytes")
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_data + b'\r\n')
        else:
            # Optional: avoid spamming logs; only log occasionally if you want
            # print("[STREAM] no frame available")
            continue


@app.route('/')
def index():
    # Renders templates/index.html
    return render_template("index.html")


@app.route('/video_feed')
def video_feed():
    return Response(
        generate_mjpeg_stream(),
        mimetype='multipart/x-mixed-replace; boundary=frame'
    )


def start_server():
    port = getattr(settings, 'STREAM_PORT', 8080)
    host = getattr(settings, 'STREAM_HOST', '127.0.0.1')
    print(f"Video Stream active: http://{host}:{port}")
    app.run(host=host, port=port, threaded=True, debug=False, use_reloader=False)
g