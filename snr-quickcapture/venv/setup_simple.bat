@echo off
echo Setting up SNR QuickCapture environment...

REM Set encoding to avoid character issues
chcp 65001 >nul

REM Check Python
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python not found. Please install Python 3.11+ first.
    pause
    exit /b 1
)

REM Remove existing venv if it exists
if exist "venv" (
    echo Removing existing virtual environment...
    rmdir /s /q venv
)

REM Create virtual environment
echo Creating virtual environment...
python -m venv venv
if errorlevel 1 (
    echo ERROR: Failed to create virtual environment.
    pause
    exit /b 1
)

REM Activate virtual environment
echo Activating virtual environment...
call venv\Scripts\activate.bat

REM Upgrade pip silently
echo Upgrading pip...
python -m pip install --upgrade pip --quiet

REM Install dependencies
echo Installing dependencies...
pip install -r requirements.txt --quiet

if errorlevel 1 (
    echo ERROR: Failed to install dependencies.
    echo Check the requirements.txt file and try again.
    pause
    exit /b 1
)

echo.
echo SUCCESS: Environment setup complete!
echo.
echo To activate the environment in the future:
echo   venv\Scripts\activate.bat
echo.
echo To run QuickCapture:
echo   python scripts\quick_add.py "your note here"
echo.
pause 