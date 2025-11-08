@echo off
REM Signal Strength Monitor - Real-time Entry and Exit signal probabilities

color 0D
title Signal Strength Monitor

:refresh
cls

echo ================================================================================
echo SIGNAL STRENGTH MONITOR - Entry + Exit Probabilities
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

if "%logfile%"=="NOT_FOUND" (
    echo [ERROR] Log file not found. Bot may not be running.
    echo.
    pause
    exit /b
)

echo Log File: %logfile%
echo.

echo ================================================================================
echo ENTRY SIGNAL THRESHOLDS
echo ================================================================================
echo.
echo   LONG/SHORT Entry: >= 0.70 (70%% probability)
echo.

echo ================================================================================
echo RECENT ENTRY SIGNALS (Last 15)
echo ================================================================================
echo.

powershell -Command "Get-Content '%logfile%' | Select-String 'LONG signal|SHORT signal' | Select-Object -Last 15"

echo.
echo ================================================================================
echo EXIT SIGNAL THRESHOLDS
echo ================================================================================
echo.
echo   ML Exit Trigger: >= 0.75 (75%% probability)
echo.

echo ================================================================================
echo RECENT EXIT SIGNALS (Last 15)
echo ================================================================================
echo.

powershell -Command "Get-Content '%logfile%' | Select-String 'Exit.*prob=|EXIT.*probability' | Select-Object -Last 15"

echo.
echo ================================================================================
echo SIGNAL STRENGTH DISTRIBUTION (Last 30 entries)
echo ================================================================================
echo.

echo High Strength Entry Signals (>= 0.80):
powershell -Command "$entries = Get-Content '%logfile%' | Select-String 'LONG signal.*0\.[89]|SHORT signal.*0\.[89]' | Select-Object -Last 10; if ($entries) { $entries } else { Write-Host '  None in recent logs' }"

echo.
echo High Strength Exit Signals (>= 0.85):
powershell -Command "$exits = Get-Content '%logfile%' | Select-String 'Exit.*prob=0\.[89]|Exit.*prob=1\.' | Select-Object -Last 10; if ($exits) { $exits } else { Write-Host '  None in recent logs' }"

echo.
echo ================================================================================
echo SIGNAL TRIGGER EVENTS (Actual Entries/Exits)
echo ================================================================================
echo.

echo Recent Triggered Entry Signals:
powershell -Command "Get-Content '%logfile%' | Select-String 'Opening.*position|Enter LONG|Enter SHORT' | Select-Object -Last 5"

echo.
echo Recent Triggered Exit Signals:
powershell -Command "Get-Content '%logfile%' | Select-String 'ML Exit.*triggered|Closing position.*ML Exit' | Select-Object -Last 5"

echo.
echo ================================================================================
echo CURRENT SIGNAL STATUS
echo ================================================================================
echo.

echo Latest Entry Signals:
powershell -Command "Get-Content '%logfile%' | Select-String 'LONG signal|SHORT signal' | Select-Object -Last 2"

echo.
echo Latest Exit Signals:
powershell -Command "Get-Content '%logfile%' | Select-String 'Exit.*prob=' | Select-Object -Last 2"

echo.
echo ================================================================================
echo SIGNAL STATISTICS
echo ================================================================================
echo.

powershell -Command "$longSignals = (Get-Content '%logfile%' | Select-String 'LONG signal').Count; $shortSignals = (Get-Content '%logfile%' | Select-String 'SHORT signal').Count; $mlExits = (Get-Content '%logfile%' | Select-String 'ML Exit').Count; Write-Host \"Total LONG Signals: $longSignals\"; Write-Host \"Total SHORT Signals: $shortSignals\"; Write-Host \"Total ML Exit Signals: $mlExits\""

echo.
echo ================================================================================
echo AUTO-REFRESH OPTIONS
echo ================================================================================
echo.
echo [R] Refresh Now
echo [A] Auto-refresh every 30 seconds
echo [Q] Quit
echo.

set /p choice="Enter your choice: "

if /i "%choice%"=="R" goto refresh
if /i "%choice%"=="A" goto autorefresh
if /i "%choice%"=="Q" exit

goto refresh

:autorefresh
cls
echo Auto-refresh mode activated (30 second intervals)
echo Press Ctrl+C to stop
echo.
timeout /t 30 >nul
goto refresh
