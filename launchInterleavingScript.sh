#!/bin/bash
set -e

# Change to the project directory (assumed to be "VideoInterleaving" in your home)
cd "$HOME/VideoInterleaving"

echo "Activating virtual environment..."
source .venv/bin/activate

echo "Running main.py..."
python main.py
