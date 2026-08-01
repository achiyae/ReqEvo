# setup_env.ps1

$ErrorActionPreference = 'Stop'

Write-Host "Setting up Python virtual environment..." -ForegroundColor Cyan

$pythonCmd = $null
foreach ($candidate in @('python3', 'python')) {
    $command = Get-Command $candidate -ErrorAction SilentlyContinue
    if ($command) {
        $pythonCmd = $command.Source
        break
    }
}

if (-not $pythonCmd) {
    Write-Error "Could not find Python 3. Install Python 3 or add it to PATH."
    exit 1
}

Write-Host "Using Python interpreter: $pythonCmd" -ForegroundColor Yellow

# Create venv
& $pythonCmd -m venv venv

# Activate venv
$venvPath = (Resolve-Path ./venv).Path
$activateScript = if ($IsWindows) {
    Join-Path $venvPath 'Scripts/Activate.ps1'
} else {
    Join-Path $venvPath 'bin/Activate.ps1'
}
$venvPython = if ($IsWindows) {
    Join-Path $venvPath 'Scripts/python.exe'
} else {
    Join-Path $venvPath 'bin/python'
}

if (Test-Path $activateScript) {
    . $activateScript
} else {
    Write-Error "Failed to create virtual environment."
    exit 1
}

Write-Host "Upgrading pip..." -ForegroundColor Yellow
& $venvPython -m pip install --upgrade pip

Write-Host "Installing required AI libraries..." -ForegroundColor Yellow
& $venvPython -m pip install torch transformers peft trl datasets bitsandbytes accelerate

Write-Host "`n========================================================" -ForegroundColor Cyan
Write-Host "Environment setup complete!" -ForegroundColor Green
Write-Host "The virtual environment 'venv' has been created and populated." -ForegroundColor White
Write-Host "To activate it manually in the future, run:" -ForegroundColor White
if ($IsWindows) {
    Write-Host "  .\\venv\\Scripts\\Activate.ps1" -ForegroundColor DarkGray
} else {
    Write-Host "  . ./venv/bin/Activate.ps1" -ForegroundColor DarkGray
}
