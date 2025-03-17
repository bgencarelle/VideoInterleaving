#!/bin/bash
set -e  # Exit immediately if a command exits with a non-zero status

echo ">>> Bootstrapping pyInter project..."

# --------------------------------------------
# Check for Python 3.9 or higher
# --------------------------------------------
PYTHON_VERSION=$(python3 -c 'import sys; print(".".join(map(str, sys.version_info[:3])))')
REQUIRED_VERSION="3.9.0"

# Compare versions (using Python for reliable comparison)
python3 -c "import sys; sys.exit(0 if sys.version_info >= (3,9) else 1)" 2>/dev/null || {
    echo "Error: Python 3.9 or higher is required. You have $PYTHON_VERSION."
    exit 1
}
echo "Python version $PYTHON_VERSION OK."

# --------------------------------------------
# Check for key system packages (for OpenGL and SDL)
# --------------------------------------------
if ! command -v glxinfo &> /dev/null; then
    echo "Warning: glxinfo not found. You may need to install 'mesa-utils' and libgl1 for OpenGL support."
fi

if ! ldconfig -p | grep -q libsdl2; then
    echo "Warning: libsdl2 not found. You may need to install libsdl2 (e.g., via 'sudo apt install libsdl2-2.0-0')."
fi

# --------------------------------------------
# Reset Git repository and update code
# --------------------------------------------
echo ">>> Resetting git state..."
git reset --hard
git pull

# --------------------------------------------
# Remove old virtual environment (if any)
# --------------------------------------------
echo ">>> Removing old virtual environment (if any)..."
rm -rf .PyIntervenv

# --------------------------------------------
# Create and activate a new virtual environment
# --------------------------------------------
echo ">>> Creating virtual environment..."
python3 -m venv .PyIntervenv

echo ">>> Activating virtual environment and installing dependencies.. ."
source .PyIntervenv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

# --------------------------------------------
# Ensure the new runner script is executable
# --------------------------------------------
chmod +x launchInterleavingScript.sh

# --------------------------------------------
# Create a symbolic link in the home directory for easy access
# --------------------------------------------
ln -sf "$PWD/launchInterleavingScript.sh" "$HOME/launchInterleavingScript.sh"
echo "Created symbolic link: $HOME/launchInterleavingScript.sh -> $PWD/launchInterleavingScript.sh"

echo ">>> Bootstrap complete."
echo "You can now run the project with: launchInterleavingScript.sh"
