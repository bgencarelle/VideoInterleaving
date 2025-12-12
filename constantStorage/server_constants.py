#server_constants
import sys
WEBSOCKET_PORT = 2324
STREAM_PORT = 8080           # Port for video stream (distinct from monitor port 1978)
STREAM_HOST = '0.0.0.0'      # Listen on all interfaces

if sys.platform == "darwin":
    HEADLESS_BACKEND = None
else:
    HEADLESS_BACKEND = "egl"
    
MAX_VIEWERS = 10 # Max simultaneous connections

