param(
    [string]$PythonExe = "python",
    [switch]$NoLaunch
)

$ErrorActionPreference = "Stop"

Write-Host "==> Kronos one-click deploy (Windows PowerShell)" -ForegroundColor Cyan

# Resolve project root (script directory)
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $ScriptDir

# Paths
$VenvPath = Join-Path $ScriptDir ".venv"
$WebUiDir = Join-Path $ScriptDir "webui"

# 1) Create venv if missing or incomplete
$VenvPython = Join-Path $VenvPath "Scripts\python.exe"

if (!(Test-Path $VenvPath)) {
    Write-Host "==> Creating virtual environment" -ForegroundColor Green
    & $PythonExe -m venv $VenvPath
    if ($LASTEXITCODE -ne 0) {
        throw "Failed to create virtual environment"
    }
} elseif (!(Test-Path $VenvPython)) {
    Write-Host "==> Detected incomplete virtual environment, recreating..." -ForegroundColor Yellow
    Remove-Item -Path $VenvPath -Recurse -Force
    & $PythonExe -m venv $VenvPath
    if ($LASTEXITCODE -ne 0) {
        throw "Failed to create virtual environment"
    }
} else {
    Write-Host "==> Using existing virtual environment: $VenvPath" -ForegroundColor Green
}

# 3) Upgrade pip
Write-Host "==> Upgrading pip" -ForegroundColor Green
& $VenvPython -m pip install -U pip
if ($LASTEXITCODE -ne 0) {
    throw "Failed to upgrade pip"
}

# 4) Install core requirements
Write-Host "==> Installing core requirements" -ForegroundColor Green
& $VenvPython -m pip install -r requirements.txt
if ($LASTEXITCODE -ne 0) {
    throw "Failed to install core requirements"
}

# 5) Install Web UI requirements
Write-Host "==> Installing Web UI requirements" -ForegroundColor Green
& $VenvPython -m pip install -r (Join-Path $WebUiDir 'requirements.txt')
if ($LASTEXITCODE -ne 0) {
    throw "Failed to install Web UI requirements"
}

# 5.5) Install tqsdk on SYSTEM Python as well
# Flask runs in the venv, but TQSdk data fetching uses a subprocess with a
# separate Python interpreter because tqsdk's asyncio conflicts with Flask's
# threading model. _find_tqsdk_python() in app.py searches for a working
# interpreter at startup — it must find one outside the venv.
Write-Host "==> Installing tqsdk on system Python (needed for futures data subprocess)" -ForegroundColor Green
$SystemPythonCandidates = @(
    "C:\Users\Administrator\AppData\Local\Programs\Python\Python311\python.exe",
    "C:\Users\Administrator\AppData\Local\Programs\Python\Python312\python.exe",
    "C:\Users\Administrator\AppData\Local\Programs\Python\Python310\python.exe"
)
$SystemPython = $null
foreach ($candidate in $SystemPythonCandidates) {
    if (Test-Path $candidate) {
        $SystemPython = $candidate
        break
    }
}
# Also try PATH (but skip if it resolves back to the venv)
if (-not $SystemPython) {
    $pathPythonCmd = Get-Command python -ErrorAction SilentlyContinue
    $pathPython = if ($pathPythonCmd) { $pathPythonCmd.Source } else { $null }
    if ($pathPython -and ($pathPython -notlike "*$VenvPath*")) {
        $SystemPython = $pathPython
    }
}

if ($SystemPython) {
    Write-Host "==> Found system Python: $SystemPython" -ForegroundColor Green
    & $SystemPython -m pip install tqsdk pandas --quiet
    if ($LASTEXITCODE -eq 0) {
        Write-Host "==> tqsdk installed on system Python" -ForegroundColor Green
    } else {
        Write-Host "==> WARNING: tqsdk install on system Python failed, futures data may not work" -ForegroundColor Yellow
        Write-Host "==>          Run manually: $SystemPython -m pip install tqsdk" -ForegroundColor Yellow
    }
} else {
    Write-Host "==> WARNING: System Python not found. Futures data will be unavailable." -ForegroundColor Yellow
    Write-Host "==>          Install Python 3.11 from python.org, then run:" -ForegroundColor Yellow
    Write-Host "==>          python -m pip install tqsdk" -ForegroundColor Yellow
}

# 6) Optional: Pre-download models (skip by default)
Write-Host "==> Dependencies ready" -ForegroundColor Green

if ($NoLaunch) {
    Write-Host "==> Setup completed. Launch skipped (--NoLaunch)." -ForegroundColor Yellow
    exit 0
}

# 7) Launch Web UI
Write-Host "==> Starting Web UI on http://localhost:7070" -ForegroundColor Cyan
Write-Host "==> Using Hugging Face mirror: hf-mirror.com" -ForegroundColor Yellow
Set-Location $WebUiDir
$env:HF_ENDPOINT = "https://hf-mirror.com"
& $VenvPython run.py