#server_constants
import sys

STREAM_PORT = 8080           # Port for video stream (distinct from monitor port 1978)
STREAM_HOST = '0.0.0.0'      # Listen on all interfaces
JPEG_QUALITY = 75# Image quality
HEADLESS_RES = (600, 800)   # Resolution for the virtual screen

if sys.platform == "darwin":
    HEADLESS_BACKEND = None
else:
    HEADLESS_BACKEND = "egl"
    
MAX_VIEWERS = 20 # Max simultaneous connections

