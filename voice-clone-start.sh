#!/bin/bash
# Set terminal window title
echo -ne "\033]0;IndicF5 Neo Voice Clone Server\007"

# Navigate to script directory
cd "$(dirname "$0")"

echo "==================================================="
echo "  IndicF5 Neo Voice Clone Server Startup"
echo "==================================================="
echo

# 1. Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "[ERROR] Python3 is not installed or not in PATH."
    echo "Please install Python 3.10 or 3.11."
    read -p "Press enter to exit..."
    exit 1
fi

# 2. Check/create virtual environment
if [ -f ".venv/bin/activate" ]; then
    echo "[INFO] Activating virtual environment (.venv)..."
    source .venv/bin/activate
else
    echo "[WARNING] Virtual environment '.venv' not found."
    read -p "Would you like to automatically create '.venv' and install requirements? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "[INFO] Creating virtual environment..."
        python3 -m venv .venv
        if [ ! -f ".venv/bin/activate" ]; then
            echo "[ERROR] Failed to create virtual environment."
            read -p "Press enter to exit..."
            exit 1
        fi
        source .venv/bin/activate
        echo "[INFO] Upgrading pip..."
        python -m pip install --upgrade pip
        echo "[INFO] Installing dependencies..."
        pip install -r requirements.txt
        if [ $? -ne 0 ]; then
            echo "[ERROR] Dependency installation failed."
            read -p "Press enter to exit..."
            exit 1
        fi
        echo "[SUCCESS] Setup complete!"
        echo
    else
        echo "[INFO] Proceeding using default system Python..."
    fi
fi

# 3. Check Python version
python3 -c "import sys; sys.exit(0 if sys.version_info[:2] in [(3, 10), (3, 11)] else 1)" &> /dev/null
if [ $? -ne 0 ]; then
    echo "[WARNING] You are running an untested Python version."
    echo "Recommended version is 3.10 or 3.11."
    echo -n "Current version: "
    python3 --version
    echo
fi

# 4. Check critical dependencies
python3 -c "import gradio, torch, transformers, soundfile, pydub" &> /dev/null
if [ $? -ne 0 ]; then
    echo "[WARNING] Some required python packages are missing."
    echo "Attempting to install requirements automatically..."
    pip install -r requirements.txt
    if [ $? -ne 0 ]; then
        echo "[ERROR] Failed to install dependencies. Please run 'pip install -r requirements.txt' manually."
        read -p "Press enter to exit..."
        exit 1
    fi
fi

# 5. Launch the server
echo "[INFO] Starting IndicF5 Neo server..."
echo
python launch.py
if [ $? -ne 0 ]; then
    echo
    echo "[ERROR] Application crashed or exited with error code $?."
fi

read -p "Press enter to exit..."
