@echo off
title IndicF5 Neo Voice Clone Server
cd /d "%~dp0"

echo ===================================================
echo   IndicF5 Neo Voice Clone Server Startup
echo ===================================================
echo.

:: 1. Check if Python is installed
where python >nul 2>nul
if %errorlevel% neq 0 (
    echo [ERROR] Python is not installed or not added to your system PATH.
    echo Please download and install Python (3.10 or 3.11 recommended) from:
    echo https://www.python.org/downloads/
    echo.
    goto END
)

:: 2. Check/create virtual environment
if exist ".venv\Scripts\activate.bat" (
    echo [INFO] Activating virtual environment (.venv)...
    call ".venv\Scripts\activate.bat"
) else (
    echo [WARNING] Virtual environment '.venv' not found.
    echo.
    choice /M "Would you like to automatically create '.venv' and install requirements now"
    if errorlevel 2 (
        echo [INFO] Proceeding using default system Python...
    )
    if errorlevel 1 (
        echo.
        echo [INFO] Creating virtual environment (.venv)...
        python -m venv .venv
        if not exist ".venv\Scripts\activate.bat" (
            echo [ERROR] Failed to create virtual environment.
            goto END
        )
        call ".venv\Scripts\activate.bat"
        echo [INFO] Upgrading pip...
        python -m pip install --upgrade pip
        echo [INFO] Installing dependencies from requirements.txt...
        pip install -r requirements.txt
        if %errorlevel% neq 0 (
            echo [ERROR] Dependency installation failed.
            goto END
        )
        echo [SUCCESS] Setup complete!
        echo.
    )
)

:: 3. Check Python version (3.10 or 3.11 recommended)
python -c "import sys; sys.exit(0 if sys.version_info[:2] in [(3, 10), (3, 11)] else 1)" >nul 2>nul
if %errorlevel% neq 0 (
    echo [WARNING] You are running an untested Python version.
    echo Recommended version is 3.10 or 3.11.
    echo Current version:
    python --version
    echo.
)

:: 4. Check critical dependencies
python -c "import gradio, torch, transformers, soundfile, pydub" >nul 2>nul
if %errorlevel% neq 0 (
    echo [WARNING] Some required python packages are missing.
    echo Attempting to install requirements automatically...
    pip install -r requirements.txt
    if %errorlevel% neq 0 (
        echo [ERROR] Failed to install dependencies. Please run 'pip install -r requirements.txt' manually.
        goto END
    )
)

:: 5. Launch the server
echo [INFO] Starting IndicF5 Neo server...
echo.
python launch.py
if %errorlevel% neq 0 (
    echo.
    echo [ERROR] Application crashed or exited with error code %errorlevel%.
    echo Please review the error traceback above.
) else (
    echo.
    echo [INFO] Server stopped successfully.
)

:END
echo.
pause
