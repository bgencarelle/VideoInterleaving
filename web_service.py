# web_service.py
import time
import logging
from flask import Flask, Response
import settings
from shared_state import exchange

log = logging.getLogger('werkzeug')
log.setLevel(logging.ERROR)

app = Flask(__name__)

def generate_mjpeg_stream():
    """Yields the latest frame from the exchange as an MJPEG stream."""
    interval = 1.0 / getattr(settings, 'SERVER_CAPTURE_RATE', 24)
    while True:
        time.sleep(interval)
        frame_data = exchange.get_frame()
        if frame_data:
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_data + b'\r\n')

@app.route('/')
def index():
    return ("<html><body style='background:black;text-align:center;color:#444;'>"
            "<img src='/video_feed' style='max-width:100%;height:auto;'>"
            "</body></html>")

@app.route('/video_feed')
def video_feed():
    return Response(generate_mjpeg_stream(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

def start_server():
    port = getattr(settings, 'STREAM_PORT', 8080)
    host = getattr(settings, 'STREAM_HOST', '0.0.0.0')
    print(f"Video Stream active: http://{host}:{port}")
    app.run(host=host, port=port, threaded=True, debug=False, use_reloader=False)


@app.route('/last_frame')
def last_frame():
    """Return a single JPEG of the latest frame for debugging."""
    frame_data = exchange.get_frame()
    if not frame_data:
        return "No frame available", 503
    return Response(frame_data, mimetype='image/jpeg')