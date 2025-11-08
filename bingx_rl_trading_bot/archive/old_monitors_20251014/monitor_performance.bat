@echo off
REM Performance Summary Monitor - Shows overall trading performance metrics

color 0B
title Performance Summary Monitor

:refresh
cls

echo ================================================================================
echo PERFORMANCE SUMMARY - ML Exit Bot
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
    echo Press any key to exit...
    pause >nul
    exit /b
)

echo Log File: %logfile%
echo.

echo ================================================================================
echo ACCOUNT SUMMARY
echo ================================================================================
echo.

REM Current Balance
powershell -Command "Get-Content '%logfile%' | Select-String 'Account Balance|Current Balance' | Select-Object -Last 1"

REM Initial Balance
powershell -Command "Get-Content '%logfile%' | Select-String 'Initial Balance' | Select-Object -Last 1"

REM Session P&L
powershell -Command "Get-Content '%logfile%' | Select-String 'Session P&L' | Select-Object -Last 1"

echo.
echo ================================================================================
echo TRADING STATISTICS
echo ================================================================================
echo.

REM Total Trades (POSITION CLOSED count)
echo Calculating total trades...
powershell -Command "$closed = (Get-Content '%logfile%' | Select-String 'POSITION CLOSED').Count; Write-Host \"Total Trades Closed: $closed\""

REM Open Positions
powershell -Command "Get-Content '%logfile%' | Select-String 'OPEN' | Select-String 'Position' | Select-Object -Last 1"

REM Win/Loss Analysis
echo.
echo Recent Trade Outcomes:
powershell -Command "Get-Content '%logfile%' | Select-String 'Return:|P&L:.*%%' | Select-Object -Last 10"

echo.
echo ================================================================================
echo EXIT MODEL PERFORMANCE
echo ================================================================================
echo.

REM ML Exit Statistics
powershell -NoProfile -ExecutionPolicy Bypass -Command "$mlExit = (Get-Content '%logfile%' | Select-String 'ML Exit').Count; $maxHold = (Get-Content '%logfile%' | Select-String 'Max Hold').Count; $total = $mlExit + $maxHold; if ($total -gt 0) { $mlPct = [math]::Round(($mlExit / $total) * 100, 1); Write-Host \"ML Exits: $mlExit ($mlPct%%) | Max Hold: $maxHold\" } else { Write-Host 'No exits recorded yet' }"

echo.
echo Recent Exit Reasons:
powershell -Command "Get-Content '%logfile%' | Select-String 'Exit Reason:|Closing position' | Select-Object -Last 5"

echo.
echo ================================================================================
echo SIGNAL ANALYSIS
echo ================================================================================
echo.

REM Entry Signals
echo Recent Entry Signals:
powershell -Command "Get-Content '%logfile%' | Select-String 'LONG signal|SHORT signal' | Select-Object -Last 5"

echo.
echo Recent Exit Signal Probabilities:
powershell -Command "Get-Content '%logfile%' | Select-String 'Exit.*prob=' | Select-Object -Last 5"

echo.
echo ================================================================================
echo PERFORMANCE vs EXPECTED
echo ================================================================================
echo.

echo Expected (from ML Exit backtest):
echo   - Returns: +2.85%% per trade
echo   - Win Rate: 94.7%%
echo   - Avg Holding: 2.36 hours
echo   - ML Exit Rate: 87.6%%
echo.
echo Actual: (see statistics above)
echo.

echo ================================================================================
echo OPTIONS
echo ================================================================================
echo.
echo [R] Refresh
echo [Q] Quit
echo.

set /p choice="Enter your choice: "

if /i "%choice%"=="R" goto refresh
if /i "%choice%"=="Q" exit

goto refresh
