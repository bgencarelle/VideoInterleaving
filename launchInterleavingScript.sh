#!/bin/bash
set -e

# Change to the project directory (assumed to be "VideoInterleaving" in your home)
cd "$HOME/VideoInterleaving"

# Check for the virtual environment and activate it
if [ ! -d ".PyIntervenv" ]; then
    echo "Error: Virtual environment '.PyIntervenv' not found. Please run .bootstrap.sh first."
    exit 1
fi

echo "Activating virtual environment..."
source .PyIntervenv/bin/activate

echo "Running main.py..."
python main.py
