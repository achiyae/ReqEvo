# setup_env.ps1

Write-Host "Setting up Python Virtual Environment..." -ForegroundColor Cyan

# Create venv
python -m venv venv

# Activate venv
if (Test-Path ".\venv\Scripts\Activate.ps1") {
    . ".\venv\Scripts\Activate.ps1"
} else {
    Write-Error "Failed to create virtual environment."
    exit 1
}

Write-Host "Upgrading pip..." -ForegroundColor Yellow
python -m pip install --upgrade pip

Write-Host "Installing required AI libraries..." -ForegroundColor Yellow
pip install torch transformers peft trl datasets bitsandbytes accelerate

Write-Host "`n========================================================" -ForegroundColor Cyan
Write-Host "Environment setup complete!" -ForegroundColor Green
Write-Host "The virtual environment 'venv' has been created and populated." -ForegroundColor White
Write-Host "To activate it manually in the future, run:" -ForegroundColor White
Write-Host "  .\venv\Scripts\Activate.ps1" -ForegroundColor DarkGray
