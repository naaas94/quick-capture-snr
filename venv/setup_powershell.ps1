# PowerShell script for setting up SNR QuickCapture environment
# This version avoids hanging issues by using non-interactive commands

Write-Host "Setting up SNR QuickCapture environment..." -ForegroundColor Green

# Set encoding to avoid character issues
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
$OutputEncoding = [System.Text.Encoding]::UTF8

# Check if Python is available
try {
    $pythonVersion = & python --version 2>&1
    Write-Host "Python version: $pythonVersion" -ForegroundColor Cyan
} catch {
    Write-Host "Error: Python not found. Please install Python 3.11+ and try again." -ForegroundColor Red
    Read-Host "Press Enter to exit"
    exit 1
}

# Remove existing virtual environment if it exists
if (Test-Path "venv") {
    Write-Host "Removing existing virtual environment..." -ForegroundColor Yellow
    Remove-Item -Recurse -Force "venv"
}

# Create virtual environment
Write-Host "Creating virtual environment..." -ForegroundColor Yellow
& python -m venv venv
if ($LASTEXITCODE -ne 0) {
    Write-Host "Error: Failed to create virtual environment." -ForegroundColor Red
    Read-Host "Press Enter to exit"
    exit 1
}

# Activate virtual environment using batch file (more reliable)
Write-Host "Activating virtual environment..." -ForegroundColor Yellow
& ".\venv\Scripts\activate.bat"

# Upgrade pip silently
Write-Host "Upgrading pip..." -ForegroundColor Yellow
& python -m pip install --upgrade pip --quiet

# Install dependencies
Write-Host "Installing dependencies..." -ForegroundColor Yellow
& pip install -r requirements.txt --quiet

if ($LASTEXITCODE -ne 0) {
    Write-Host "Error: Failed to install dependencies." -ForegroundColor Red
    Write-Host "Check the requirements.txt file and try again." -ForegroundColor Yellow
    Read-Host "Press Enter to exit"
    exit 1
}

Write-Host ""
Write-Host "SUCCESS: Environment setup complete!" -ForegroundColor Green
Write-Host ""
Write-Host "To activate the environment in the future:" -ForegroundColor Cyan
Write-Host "  .\venv\Scripts\activate.bat" -ForegroundColor White
Write-Host ""
Write-Host "To run QuickCapture:" -ForegroundColor Cyan
Write-Host "  python scripts\quick_add.py `"your note here`"" -ForegroundColor White
Write-Host ""
Read-Host "Press Enter to exit" 