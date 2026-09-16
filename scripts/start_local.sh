#!/bin/bash

# Get the repo root (this script lives in scripts/)
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PROJECT_DIR=$(cd "$SCRIPT_DIR/.." && pwd)
VENV_DIR="$PROJECT_DIR/.venv"

# Deactivate any existing virtual environment first
if command -v deactivate &> /dev/null; then
    deactivate 2>/dev/null || true
fi

# Activate virtual environment if it exists
if [ -f "$VENV_DIR/bin/activate" ]; then
    source "$VENV_DIR/bin/activate"
else
    echo "⚠️  Warning: Virtual environment not found at $VENV_DIR"
    echo "   Run setup_app.sh to create it, or continuing with system Python..."
fi

export PYTHONPATH=$PYTHONPATH:.

# Convert shorthand flags to proper arguments
ARGS=""
for arg in "$@"; do
    case "$arg" in
        --asciiweb)
            ARGS="$ARGS --mode asciiweb"
            ;;
        --ascii)
            ARGS="$ARGS --mode ascii"
            ;;
        --web)
            ARGS="$ARGS --mode web"
            ;;
        --local)
            ARGS="$ARGS --mode local"
            ;;
        *)
            ARGS="$ARGS $arg"
            ;;
    esac
done

if [[ "$ARGS" != *"--mode"* ]]; then
    ARGS="$ARGS --mode local"
fi

cd "$PROJECT_DIR"

python3 main.py $ARGS
