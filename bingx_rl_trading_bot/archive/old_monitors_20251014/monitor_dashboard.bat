@echo off
REM ML Exit Bot Dashboard - Shows bot status and recent activity

color 0F
title ML Exit Bot Dashboard

:refresh
cls

echo ================================================================================
echo ML EXIT BOT DASHBOARD - Phase 4 Dual Entry + Dual Exit
echo ================================================================================
echo.
echo Updated: %date% %time%
echo.

cd /d "C:\Users\J\OneDrive\CLAUDE_CODE_FIN\bingx_rl_trading_bot"

REM Find the most recent log file
for /f "delims=" %%i in ('dir /b /o-d logs\phase4_dynamic_testnet_trading_*.log 2^>nul') do (
    set logfile=logs\%%i
    goto :logfound
)
set logfile=NOT_FOUND
:logfound

echo ===============================================================================
echo BOT STATUS
echo ===============================================================================

REM Check if bot is running
tasklist /FI "IMAGENAME eq python.exe" 2>NUL | find /I /N "python.exe">NUL
if "%ERRORLEVEL%"=="0" (
    echo Status: [32m RUNNING [0m
) else (
    echo Status: [31m STOPPED [0m
)

REM Check if lock file exists
if exist "results\bot_instance.lock" (
    echo Lock File: [32m EXISTS [0m ^(bot active^)
) else (
    echo Lock File: [31m MISSING [0m ^(bot not running^)
)

REM Check log file
if "%logfile%"=="NOT_FOUND" (
    echo Log File: [31m NOT FOUND [0m
    echo.
    echo [WARNING] No log files found. Bot may not have started yet.
) else (
    echo Log File: [32m FOUND [0m - %logfile%
    for %%A in ("%logfile%") do echo Log Size: %%~zA bytes
)

echo.
echo ===============================================================================
echo EXPECTED PERFORMANCE (ML Exit Model Backtest)
echo ===============================================================================
echo   Returns: +2.85%% per 2 days ^(+39.2%% vs rule-based^)
echo   Win Rate: 94.7%% ^(+5.0%% vs rule-based^)
echo   Avg Holding: 2.36 hours ^(-41%% vs rule-based^)
echo   Exit Efficiency: 87.6%% ML Exit, 12.4%% Max Hold
echo.

echo ===============================================================================
echo RECENT ACTIVITY (Last 15 log entries)
echo ===============================================================================
echo.

if "%logfile%"=="NOT_FOUND" (
    echo No log file available.
    echo.
    echo If bot is running, wait a few minutes for log file creation.
) else (
    powershell -Command "Get-Content '%logfile%' -Tail 15"
)

echo.
echo ===============================================================================
echo MONITORING OPTIONS - Enhanced Dashboard
echo ===============================================================================
echo.
echo === PERFORMANCE ===
echo [1] Performance Summary (Total trades, Win rate, P^&L, ROI)
echo [2] Trade History (Detailed trade records)
echo.
echo === SIGNALS ===
echo [3] Signal Strength (Entry/Exit probabilities)
echo [4] Exit Signals Only (ML Exit activity)
echo.
echo === SYSTEM ===
echo [5] Position Monitor (Real-time P^&L tracking)
echo [6] Full Log Monitor (All bot activity)
echo [7] Error Monitor (Errors and warnings only)
echo.
echo === CONTROL ===
echo [8] Refresh Dashboard
echo [9] Exit
echo.

set /p choice="Enter your choice (1-9): "

if "%choice%"=="1" (
    start monitor_performance.bat
    goto refresh
)
if "%choice%"=="2" (
    start monitor_trades.bat
    goto refresh
)
if "%choice%"=="3" (
    start monitor_signals.bat
    goto refresh
)
if "%choice%"=="4" (
    start monitor_ml_exit_signals.bat
    goto refresh
)
if "%choice%"=="5" (
    start monitor_positions.bat
    goto refresh
)
if "%choice%"=="6" (
    start monitor_ml_exit.bat
    goto refresh
)
if "%choice%"=="7" (
    start monitor_errors.bat
    goto refresh
)
if "%choice%"=="8" (
    goto refresh
)
if "%choice%"=="9" (
    exit
)

echo Invalid choice. Press any key to try again...
pause >nul
goto refresh
