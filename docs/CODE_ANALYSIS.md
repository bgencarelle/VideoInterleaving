# Code Analysis: What This System Does

## High-Level Architecture

This is a **multi-protocol video/ASCII art streaming system** that takes rendered frames from a display loop and broadcasts them simultaneously via multiple server protocols.

```
┌─────────────────────────────────────────────────────────────┐
│                    Main Display Loop                        │
│              (image_display.run_display())                  │
│                                                              │
│  Renders frames → exchange.set_frame(frame_bytes)           │
└───────────────────────┬─────────────────────────────────────┘
                        │
                        ▼
            ┌───────────────────────┐
            │  FrameExchange        │
            │  (shared_state.py)   │
            │  Thread-safe queue   │
            └───────────┬───────────┘
                        │
        ┌───────────────┼───────────────┐
        │               │               │
        ▼               ▼               ▼
┌──────────────┐ ┌──────────────┐ ┌──────────────┐
│ HTTP Servers │ │ Telnet Server│ │ WebSocket    │
│ (web_service)│ │ (ascii_server)│ │ (ascii_web)  │
└──────────────┘ └──────────────┘ └──────────────┘
```

## Core Components

### 1. **Frame Exchange System** (`shared_state.py`)

**Purpose**: Thread-safe communication between producer (display loop) and consumers (servers)

**How it works**:
- `FrameExchange` uses a `threading.Condition` for synchronization
- Producer calls `set_frame()` → wakes all waiting threads
- Consumers call `get_frame()` → blocks until new frame arrives
- Supports timeout for stall detection

**Key Design**: Single producer, multiple consumers (broadcast pattern)

---

### 2. **HTTP Services** (`web_service.py`)

**Two separate HTTP servers** running on different ports:

#### A. **Monitor Server** (Port 1978/8888)
**Purpose**: Administrative dashboard and monitoring

**Endpoints**:
- `/` → HTML dashboard (from `lightweight_monitor`)
- `/data` → JSON metrics (FPS, CPU, memory, etc.)
- `/log` → Last 64KB of runtime log
- `/ascii` → ASCII viewer HTML page
- `/static/*` → Static files (CSS, JS, images)
- `/pagename` → Dynamic template serving (`templates/pagename.html`)

**Security**: 
- Defaults to `127.0.0.1` (localhost only)
- Path traversal protection
- Bot scan filtering (suppresses 400/403/404/408 logs)

#### B. **Stream Server** (Port 8080)
**Purpose**: MJPEG video streaming for public viewing

**Endpoints**:
- `/video_feed` → MJPEG stream (multipart/x-mixed-replace)
- `/stats` → JSON viewer count stats
- `/` → Main viewer page (`templates/index.html`)
- `/static/*` → Static assets

**Features**:
- Connection limiting (semaphore-based, max 20 viewers)
- Heartbeat tracking per client
- Stall detection (disconnects if no frames for 10s)
- Supports JPEG and WebP formats
- TCP_NODELAY for low latency
- Frame format: `[format_byte][frame_data]` where `'w'` = WebP, else JPEG

**Security**: Defaults to `0.0.0.0` (public access intended)

---

### 3. **ASCII Telnet Server** (`ascii_server.py`)

**Purpose**: Raw TCP/Telnet connection for ASCII art streaming

**Protocol**: Raw TCP with ANSI escape codes

**Features**:
- ANSI terminal control (clear screen, cursor positioning)
- Frame rate limiting (configurable FPS, skips frames if too fast)
- Socket optimization (TCP_NODELAY, keepalive)
- Connection limiting (semaphore)
- Handles both string and bytes frame data

**Use Case**: Terminal-based viewing via `telnet` or `nc`

---

### 4. **ASCII WebSocket Server** (`ascii_web_server.py`)

**Purpose**: WebSocket-based ASCII art streaming for browsers

**Architecture**:
- Single broadcast loop thread pushes to all connected clients
- Client connection management (add/remove from list)
- "Leaky bucket" fix: Skips frames for lagging clients (tab hidden)
- Connection limiting (semaphore)

**Features**:
- Small send buffer (4096 bytes) for low latency
- TCP_NODELAY enabled
- ANSI escape codes for terminal control

**Use Case**: Browser-based ASCII viewing (connects to `/ascii_ws/`)

---

### 5. **ASCII Stats Server** (`ascii_stats_server.py`)

**Purpose**: Telnet-based real-time monitoring dashboard

**Protocol**: Raw TCP with ANSI color codes

**Features**:
- Updates every 1 second
- Color-coded health status (green=synced, red=ahead/behind)
- Shows: FPS, index, delta, FIFO depth, staleness, folder coverage, entropy, CPU, memory, uptime, errors
- Full-screen refresh each update

**Use Case**: Terminal-based monitoring (`telnet localhost <port>`)

---

## Data Flow

### Frame Production
```
image_display.run_display()
  ↓
Renders frame (JPEG/WebP or ASCII string)
  ↓
exchange.set_frame(frame_bytes)
  ↓
Wakes all waiting server threads
```

### Frame Consumption (Multiple Servers)
```
Server Thread 1: exchange.get_frame() → HTTP MJPEG stream
Server Thread 2: exchange.get_frame() → Telnet ASCII stream  
Server Thread 3: exchange.get_frame() → WebSocket ASCII stream
Server Thread 4: exchange.get_frame() → (not used, stats is polling)
```

---

## Operating Modes

### Mode: `web`
- **Servers**: Monitor (1978) + Stream (8080)
- **Use**: Public MJPEG video streaming
- **Access**: HTTP/HTTPS via browser

### Mode: `local`
- **Servers**: Monitor only (8888)
- **Use**: Local development, windowed display
- **Access**: Localhost only

### Mode: `ascii`
- **Servers**: Telnet (2323) + Stats Monitor (2324)
- **Use**: Terminal-based ASCII art viewing
- **Access**: `telnet <host> 2323`

### Mode: `asciiweb`
- **Servers**: WebSocket (2424) + Monitor (1980)
- **Use**: Browser-based ASCII art viewing
- **Access**: WebSocket connection + HTTP viewer page

---

## Key Design Patterns

### 1. **Producer-Consumer with Broadcast**
- Single producer (display loop)
- Multiple consumers (different server protocols)
- Thread-safe via `threading.Condition`

### 2. **Connection Management**
- Semaphores for connection limiting
- Heartbeat tracking for client health
- Graceful disconnect handling

### 3. **Performance Optimizations**
- TCP_NODELAY (disable Nagle's algorithm) for low latency
- Pre-encoded headers (MJPEG boundary strings)
- Frame skipping for rate limiting
- Small buffers for WebSocket

### 4. **Security Hardening**
- Path traversal protection
- Static file serving restrictions
- Bot scan filtering
- Localhost defaults for sensitive endpoints

### 5. **Error Resilience**
- Silent handling of normal disconnects
- Stall detection (disconnect if producer dead)
- Graceful degradation (skip frames for lagging clients)

---

## What I Notice (Observations)

### ✅ **Strengths**
1. **Clean separation**: Each server handles one protocol
2. **Thread-safe**: Proper use of locks, conditions, semaphores
3. **Performance-focused**: TCP_NODELAY, pre-encoded headers, frame skipping
4. **Security-conscious**: Path traversal protection, localhost defaults
5. **Resilient**: Handles disconnects, stalls, errors gracefully

### ✅ **Clarifications (After Code Review)**

1. **Frame Format** - **RESOLVED**
   - From `image_display.py`: `exchange.set_frame(b'j' + enc)` for JPEG
   - Format: `[format_byte][frame_data]` where format byte is `'j'` (JPEG) or `'w'` (WebP)
   - ASCII mode: `exchange.set_frame(text_frame)` - raw string
   - **Design is correct**: Different formats for different modes

2. **Stats Server** - **RESOLVED**
   - `monitor_data` is a **separate metrics dictionary** (not frame data)
   - Updated by `lightweight_monitor.py` with system stats (FPS, CPU, memory, etc.)
   - **Correctly polls** `monitor_data` - it's not frame streaming data
   - This is intentional and correct design

### ⚠️ **Potential Issues**

1. **Frame Format Validation**
   - `_handle_mjpeg_stream()` checks `len(raw_payload) < 1` but doesn't validate format byte
   - Currently only handles `'w'` (WebP) vs default (JPEG)
   - **Question**: What if format byte is `'j'` explicitly? Currently treated as JPEG anyway
   - **Low risk**: Works but could be more explicit

2. **Heartbeat Tracking Not Used**
   - `_client_heartbeats` is tracked but never checked for dead clients
   - **Potential**: Memory leak if clients disconnect without cleanup
   - **Note**: `_clear_heartbeat()` is called in `finally` block, so cleanup should work
   - **Question**: Is heartbeat used for monitoring elsewhere, or just for future use?

3. **WebSocket Buffer Check**
   - Checks `client.sendq` (internal library attribute)
   - **Risk**: Library update could break this
   - **Question**: Is this documented behavior or implementation detail?
   - **Workaround**: Could use try/except or check library version

4. **No Authentication**
   - Monitor endpoints are localhost-only but no auth
   - If someone gets localhost access, full system visibility
   - **Acceptable risk**: Localhost-only is reasonable security boundary

5. **Exception Handling in Stream**
   - Line 307: `except:` (bare except) - catches all exceptions silently
   - Should probably log or handle more specifically

### 🤔 **Questions for You**

1. **Heartbeat Usage**: Is `_client_heartbeats` used elsewhere or just tracked for future use?

2. **WebSocket Library**: Is `SimpleWebSocketServer.sendq` a stable API, or should we add a fallback?

3. **Deployment**: In production, are all these servers always running, or mode-specific?

4. **Frame Format**: Should we explicitly handle `'j'` format byte, or is default JPEG handling sufficient?

---

## Summary

This is a **sophisticated multi-protocol streaming system** that:
- Takes rendered frames from a display loop
- Broadcasts them via HTTP (MJPEG), Telnet (ASCII), and WebSocket (ASCII)
- Includes monitoring, stats, and administrative dashboards
- Handles multiple concurrent clients with connection limiting
- Optimized for low latency and performance
- Security-hardened for production use

The architecture is **well-designed** with proper thread safety, error handling, and performance optimizations. The main areas for improvement are around frame format consistency, heartbeat cleanup, and potentially consolidating configuration.

