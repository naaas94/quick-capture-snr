@echo off
echo Starting QuickCapture Brain (Server)...
start "QuickCapture Brain" cmd /k "python scripts/server.py"

echo Waiting for server to initialize...
timeout /t 5

echo Starting QuickCapture Launcher...
start "QuickCapture Launcher" cmd /k "python scripts/launcher.py"

echo QuickCapture is running! Press Alt+Space to capture.
