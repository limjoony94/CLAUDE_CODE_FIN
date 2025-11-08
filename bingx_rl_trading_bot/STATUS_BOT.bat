@echo off
REM ====================================================================
REM STATUS_BOT.bat - Check Opportunity Gating Bot Status
REM ====================================================================

echo.
echo ========================================
echo  Bot Status Check
echo ========================================
echo.

REM Change to project directory
cd /d "%~dp0"

REM Check if bot is running
tasklist /FI "WINDOWTITLE eq Opportunity Gating Bot*" 2>NUL | find /I /N "python.exe">NUL
if "%ERRORLEVEL%"=="0" (
    echo [STATUS] Bot is RUNNING
    echo.
    echo Process details:
    tasklist /FI "WINDOWTITLE eq Opportunity Gating Bot*" /V
    echo.
) else (
    echo [STATUS] Bot is STOPPED
    echo.
)

REM Show state file
echo ========================================
echo  Current State
echo ========================================
if exist "results\opportunity_gating_bot_4x_state.json" (
    type results\opportunity_gating_bot_4x_state.json
    echo.
) else (
    echo State file not found.
    echo.
)

REM Show recent log entries
echo ========================================
echo  Recent Log Entries (Last 20 lines)
echo ========================================
echo.

REM Find most recent log file
for /f "delims=" %%i in ('dir /b /od logs\opportunity_gating_bot_4x_*.log 2^>nul') do set LATEST_LOG=%%i

if defined LATEST_LOG (
    echo Log file: logs\%LATEST_LOG%
    echo.
    powershell -Command "Get-Content 'logs\%LATEST_LOG%' -Tail 20"
) else (
    echo No log files found.
)

echo.
echo ========================================
echo  Status check complete
echo ========================================
echo.

pause
