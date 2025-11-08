@echo off
REM ML Exit Bot Monitoring Script
REM Shows real-time bot activity with ML Exit signals

color 0A
title ML Exit Bot Monitor - Real-time Logs

echo ================================================================================
echo ML EXIT BOT MONITOR - Real-time Activity
echo ================================================================================
echo.
echo Bot: Phase 4 Dual Entry + Dual Exit Model
echo Exit Strategy: ML-based (LONG/SHORT specialized)
echo Exit Threshold: 0.75 (75%% probability)
echo.
echo Expected Performance:
echo   - Returns: +2.85%% per 2 days (+39.2%% vs rule-based)
echo   - Win Rate: 94.7%% (+5.0%% vs rule-based)
echo   - Avg Holding: 2.36 hours (-41%% vs rule-based)
echo   - Exit Efficiency: 87.6%% ML Exit, 12.4%% Max Hold
echo.
echo ================================================================================
echo Press Ctrl+C to stop monitoring
echo ================================================================================
echo.

cd /d "C:\Users\J\OneDrive\CLAUDE_CODE_FIN\bingx_rl_trading_bot"

REM Find the most recent log file
for /f "delims=" %%i in ('dir /b /o-d logs\phase4_dynamic_testnet_trading_*.log 2^>nul') do (
    set logfile=logs\%%i
    goto :found
)

:notfound
echo [ERROR] No log files found matching: logs\phase4_dynamic_testnet_trading_*.log
echo.
echo Possible reasons:
echo   1. Bot is not running yet
echo   2. Bot just started and hasn't created log file
echo   3. Log file is in a different location
echo.
echo Checking if bot is running...
tasklist /FI "IMAGENAME eq python.exe" 2>NUL | find /I /N "python.exe">NUL
if "%ERRORLEVEL%"=="0" (
    echo Bot process found. Wait a few seconds for log file creation.
) else (
    echo Bot is not running. Please start the bot first.
)
echo.
pause
exit /b

:found
echo Monitoring: %logfile%
echo.
echo ================================================================================
echo LIVE LOGS (Most recent entries shown first, then real-time updates)
echo ================================================================================
echo.

REM Show last 20 lines first
powershell -Command "Get-Content '%logfile%' -Tail 20"

echo.
echo ===============================================================================
echo REAL-TIME UPDATES (New logs will appear below)
echo ================================================================================
echo.

REM Monitor log file in real-time
powershell -Command "Get-Content '%logfile%' -Wait -Tail 0"

pause
