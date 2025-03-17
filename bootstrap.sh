#!/bin/bash
set -e  # Exit immediately if a command exits with a non-zero status

echo ">>> Bootstrapping pyInter project..."

# --------------------------------------------
# Check for Python 3.11 or higher
# --------------------------------------------
PYTHON_VERSION=$(python3 -c 'import sys; print(".".join(map(str, sys.version_info[:3])))')
REQUIRED_VERSION="3.11.0"

python3 -c "import sys; sys.exit(0 if sys.version_info >= (3,11) else 1)" 2>/dev/null || {
    echo "Error: Python 3.11 or higher is required. You have $PYTHON_VERSION."
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
    echo "Warning: libsdl2 not found. You may need to install libsdl2 (e.g., 'sudo apt install libsdl2-2.0-0')."
fi

# --------------------------------------------
# Record the project directory for runPortrait.sh
# --------------------------------------------
PROJECT_DIR="$(pwd)"
echo ">>> Project directory: $PROJECT_DIR"

# --------------------------------------------
# Reset Git repository and update code
# --------------------------------------------
echo ">>> Resetting Git state..."
git reset --hard
git pull

# --------------------------------------------
# Remove old virtual environment (if any)
# --------------------------------------------
VENV_DIR="$HOME/PyIntervenv"
echo ">>> Removing old virtual environment at $VENV_DIR (if any)..."
rm -rf "$VENV_DIR"

# --------------------------------------------
# Create and activate a new virtual environment
# --------------------------------------------
echo ">>> Creating virtual environment in $VENV_DIR..."
python3 -m venv "$VENV_DIR"

echo ">>> Activating virtual environment and installing dependencies..."
source "$VENV_DIR/bin/activate"
pip install --upgrade pip
pip install -r requirements.txt

# --------------------------------------------
# Create runPortrait.sh in the user's HOME directory
# --------------------------------------------
RUN_SCRIPT_PATH="$HOME/runPortrait.sh"
echo ">>> Creating $RUN_SCRIPT_PATH..."
cat <<EOF > "$RUN_SCRIPT_PATH"
#!/bin/bash
# Activate the virtual environment
source "$VENV_DIR/bin/activate"

# Change to the project directory
cd "$PROJECT_DIR"

# Run main.py
python main.py
EOF

chmod +x "$RUN_SCRIPT_PATH"

# --------------------------------------------
# Run the new script
# --------------------------------------------
echo ">>> Bootstrap complete. Running $RUN_SCRIPT_PATH..."
"$RUN_SCRIPT_PATH"
