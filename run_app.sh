#!/bin/bash

# Robust application runner with crash detection and auto-restart
# Works on any platform/compositor without requiring systemd services

set -euo pipefail

# --- CONFIGURATION ---
RESTART_DELAY=${RESTART_DELAY:-3}  # Seconds to wait before restarting after crash
MAX_RESTARTS=${MAX_RESTARTS:-0}     # Max restart attempts (0 = unlimited)
LOG_FILE=${LOG_FILE:-""}            # Optional log file path (empty = stdout only)

# --- INITIALIZATION ---
# Get the directory where this script is located
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PROJECT_DIR="$SCRIPT_DIR"
VENV_DIR="$PROJECT_DIR/.venv"

# Change to project directory (fixes stalling if run from wrong location)
cd "$PROJECT_DIR"

# Verify project directory
if [ ! -f "$PROJECT_DIR/main.py" ]; then
    echo "❌ ERROR: main.py not found in $PROJECT_DIR" >&2
    echo "   Please ensure this script is in the VideoInterleaving project directory" >&2
    exit 1
fi

# --- LOGGING FUNCTIONS ---
log() {
    local message="$1"
    local timestamp=$(date '+%Y-%m-%d %H:%M:%S')
    local log_line="[$timestamp] $message"
    
    # Always print to stdout
    echo "$log_line"
    
    # Also write to log file if specified
    if [ -n "$LOG_FILE" ]; then
        echo "$log_line" >> "$LOG_FILE" 2>/dev/null || true
    fi
}

log_info() {
    log "ℹ️  $*"
}

log_success() {
    log "✅ $*"
}

log_warning() {
    log "⚠️  $*"
}

log_error() {
    log "❌ $*" >&2
}

# --- VIRTUAL ENVIRONMENT SETUP ---
setup_venv() {
    # Deactivate any existing virtual environment first
    if command -v deactivate &> /dev/null; then
        deactivate 2>/dev/null || true
    fi
    
    # Activate virtual environment if it exists
    if [ -f "$VENV_DIR/bin/activate" ]; then
        source "$VENV_DIR/bin/activate"
        log_success "Virtual environment activated"
        return 0
    else
        log_warning "Virtual environment not found at $VENV_DIR"
        log_info "Run setup_app.sh to create it, or continuing with system Python..."
        return 1
    fi
}

# --- ARGUMENT PARSING ---
parse_args() {
    local args=""
    local mode_specified=false
    
    # If no arguments provided, default to --local
    if [ $# -eq 0 ]; then
        args="--mode local"
        mode_specified=true
    else
        for arg in "$@"; do
            case "$arg" in
                --asciiweb)
                    args="$args --mode asciiweb"
                    mode_specified=true
                    ;;
                --ascii)
                    args="$args --mode ascii"
                    mode_specified=true
                    ;;
                --web)
                    args="$args --mode web"
                    mode_specified=true
                    ;;
                --local)
                    args="$args --mode local"
                    mode_specified=true
                    ;;
                --restart-delay=*)
                    RESTART_DELAY="${arg#*=}"
                    ;;
                --max-restarts=*)
                    MAX_RESTARTS="${arg#*=}"
                    ;;
                --log-file=*)
                    LOG_FILE="${arg#*=}"
                    ;;
                *)
                    args="$args $arg"
                    # Check if this argument specifies a mode
                    if [[ "$arg" == *"--mode"* ]]; then
                        mode_specified=true
                    fi
                    ;;
            esac
        done
    fi
    
    # Default to local mode if no mode was specified
    if [ "$mode_specified" = false ]; then
        args="$args --mode local"
    fi
    
    echo "$args"
}

# --- SIGNAL HANDLING ---
SHOULD_RESTART=true
GRACEFUL_EXIT=false
FIRST_SIGINT_TIME=0
SIGINT_TIMEOUT=3  # Seconds within which second Ctrl+C must be pressed

handle_sigint() {
    local current_time=$(date +%s)
    
    if [ "$FIRST_SIGINT_TIME" -eq 0 ]; then
        # First Ctrl+C - record time and warn user
        FIRST_SIGINT_TIME=$current_time
        log_warning "Ctrl+C pressed. Press Ctrl+C again within ${SIGINT_TIMEOUT} seconds to exit."
        log_info "Waiting for confirmation (or it will be ignored)..."
    else
        # Check if second Ctrl+C is within timeout
        local time_diff=$((current_time - FIRST_SIGINT_TIME))
        if [ $time_diff -le "$SIGINT_TIMEOUT" ]; then
            # Second Ctrl+C within timeout - confirm exit
            log_info "Shutdown confirmed - exiting gracefully..."
            SHOULD_RESTART=false
            GRACEFUL_EXIT=true
            # Kill the Python process if it's still running
            pkill -P $$ python3 2>/dev/null || true
            exit 0
        else
            # Too much time passed, treat as new first press
            FIRST_SIGINT_TIME=$current_time
            log_warning "Ctrl+C pressed. Press Ctrl+C again within ${SIGINT_TIMEOUT} seconds to exit."
        fi
    fi
}

handle_sigterm() {
    # SIGTERM is always treated as intentional shutdown (from system/service manager)
    log_info "Received SIGTERM - shutting down gracefully..."
    SHOULD_RESTART=false
    GRACEFUL_EXIT=true
    pkill -P $$ python3 2>/dev/null || true
    exit 0
}

trap handle_sigint SIGINT
trap handle_sigterm SIGTERM

# --- MAIN EXECUTION LOOP ---
run_app() {
    local app_args="$1"
    local restart_count=0
    
    log_info "Starting VideoInterleaving application runner"
    log_info "Project directory: $PROJECT_DIR"
    log_info "Restart delay: ${RESTART_DELAY}s"
    if [ "$MAX_RESTARTS" -gt 0 ]; then
        log_info "Max restarts: $MAX_RESTARTS"
    else
        log_info "Max restarts: unlimited"
    fi
    if [ -n "$LOG_FILE" ]; then
        log_info "Log file: $LOG_FILE"
    fi
    echo ""
    
    # Setup virtual environment
    if ! setup_venv; then
        log_warning "Continuing without virtual environment"
    fi
    
    # Set PYTHONPATH (handle case where it's not already set)
    export PYTHONPATH="${PYTHONPATH:-}:."
    
    # Main restart loop
    while true; do
        if [ "$GRACEFUL_EXIT" = true ]; then
            log_info "Graceful exit requested - not restarting"
            break
        fi
        
        # Check max restarts
        if [ "$MAX_RESTARTS" -gt 0 ] && [ "$restart_count" -ge "$MAX_RESTARTS" ]; then
            log_error "Maximum restart attempts ($MAX_RESTARTS) reached. Exiting."
            exit 1
        fi
        
        # Log restart attempt if not first run
        if [ "$restart_count" -gt 0 ]; then
            log_warning "Restart attempt #$restart_count"
            log_info "Waiting ${RESTART_DELAY} seconds before restart..."
            # Reset SIGINT tracking on restart
            FIRST_SIGINT_TIME=0
            sleep "$RESTART_DELAY"
        fi
        
        # Run the application
        log_info "Starting application: python3 main.py$app_args"
        log_info "---"
        
        # Run the application and capture exit code
        # Use eval to properly handle arguments with spaces
        set +e  # Don't exit on error, we want to check exit code
        eval "python3 main.py $app_args"
        local exit_code=$?
        set -e  # Re-enable exit on error
        
        log_info "---"
        
        # Check exit code
        if [ $exit_code -eq 0 ]; then
            log_success "Application exited normally (exit code: 0)"
            if [ "$SHOULD_RESTART" = false ] || [ "$GRACEFUL_EXIT" = true ]; then
                log_info "Not restarting (shutdown requested)"
                break
            fi
            # Exit code 0 could be KeyboardInterrupt (Python handles it gracefully)
            # To prevent accidental quits, we restart on exit code 0 unless explicitly stopped
            # This ensures the app keeps running even if Python exits cleanly
            log_warning "Application exited with code 0 - restarting to prevent accidental quit"
            log_info "If you want to stop, press Ctrl+C twice quickly"
            restart_count=$((restart_count + 1))
            continue
        else
            log_error "Application crashed with exit code: $exit_code"
            restart_count=$((restart_count + 1))
            
            if [ "$SHOULD_RESTART" = false ]; then
                log_info "Not restarting (shutdown requested)"
                break
            fi
            
            # Continue loop to restart
            continue
        fi
    done
    
    log_info "Application runner stopped"
}

# --- MAIN ---
APP_ARGS=$(parse_args "$@")
run_app "$APP_ARGS"
