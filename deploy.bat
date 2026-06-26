@echo off
setlocal enabledelayedexpansion

REM Kronos one-click deploy (Windows .bat)
REM This wrapper calls PowerShell to run deploy.ps1

set SCRIPT_DIR=%~dp0
cd /d "%SCRIPT_DIR%"

echo ================================================
echo Kronos Web UI - One-Click Deploy
echo ================================================
echo.

where powershell >nul 2>nul
if %errorlevel% neq 0 (
	echo [ERROR] PowerShell not found. Please run deploy.ps1 manually.
	echo.
	pause
	exit /b 1
)

REM Allow running in current process scope
powershell -NoProfile -ExecutionPolicy Bypass -File "%SCRIPT_DIR%deploy.ps1"

if %errorlevel% neq 0 (
	echo.
	echo ================================================
	echo [ERROR] Deployment failed. Check errors above.
	echo ================================================
	echo.
	pause
	exit /b 1
)

echo.
echo ================================================
echo [SUCCESS] Deployment completed!
echo ================================================
echo.

endlocal



