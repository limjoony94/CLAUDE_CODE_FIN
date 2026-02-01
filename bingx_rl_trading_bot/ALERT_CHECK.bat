@echo off
REM Alert Check for Pattern 5m Bot
REM Checks: stale state, consecutive losses, MDD, win rate

echo ================================================
echo Pattern 5m Bot - Alert Check
echo ================================================
echo.

python scripts\monitor\alert_check.py

echo.
if %ERRORLEVEL% NEQ 0 (
    echo.
    echo [ALERTS DETECTED] Exit code: %ERRORLEVEL%
    echo - Code 1 = Critical alerts found
    echo - Code 0 = All clear
) else (
    echo [ALL CLEAR] No alerts detected
)
echo.
pause
