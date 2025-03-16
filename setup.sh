#!/bin/bash

set -e  # Exit on error

echo ">>> Resetting git state..."
git reset --hard
git pull

echo ">>> Removing old virtual environment..."
rm -rf .PyIntervenv

echo ">>> Creating new virtual environment..."
python3 -m venv .PyIntervenv

echo ">>> Activating virtual environment and installing dependencies..."
source .PyIntervenv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

echo ">>> Setup complete."
