@echo off
REM Unified Monitor - All-in-One Bot Monitoring Dashboard
REM Shows: Performance, Signals, Positions, and Latest Activity in ONE window

color 0A
title 🤖 ML Exit Bot - Unified Monitor

:refresh
cls

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

echo ================================================================================
echo                    ML EXIT BOT - UNIFIED MONITOR
echo ================================================================================
echo Updated: %date% %time%
echo Log: %logfile%
echo.

REM ============================================================================
REM SECTION 1: ACCOUNT & PERFORMANCE
REM ============================================================================
echo.
echo [1] ACCOUNT ^& PERFORMANCE
echo --------------------------------------------------------------------------------
powershell -Command "Get-Content '%logfile%' | Select-String 'Initial Balance' | Select-Object -Last 1"
powershell -Command "Get-Content '%logfile%' | Select-String 'Account Balance' | Select-Object -Last 1"
powershell -Command "Get-Content '%logfile%' | Select-String 'Session P&L' | Select-Object -Last 1"
echo.
echo Total Trades:
powershell -Command "$closed = (Get-Content '%logfile%' | Select-String 'POSITION CLOSED').Count; if ($closed -eq 0) { Write-Host '   No trades completed yet' } else { Write-Host \"   Closed: $closed trades\" }"

REM ============================================================================
REM SECTION 2: SIGNAL MONITORING (Latest)
REM ============================================================================
echo.
echo [2] SIGNAL MONITORING (Latest)
echo --------------------------------------------------------------------------------
powershell -Command "Get-Content '%logfile%' | Select-String 'LONG Model Prob' | Select-Object -Last 1"
powershell -Command "Get-Content '%logfile%' | Select-String 'SHORT Model Prob' | Select-Object -Last 1"
powershell -Command "Get-Content '%logfile%' | Select-String 'Threshold:' | Select-Object -Last 1"
powershell -Command "Get-Content '%logfile%' | Select-String 'Should Enter' | Select-Object -Last 1"

REM ============================================================================
REM SECTION 3: POSITION STATUS
REM ============================================================================
echo.
echo [3] POSITION STATUS
echo --------------------------------------------------------------------------------
powershell -Command "$hasPos = Get-Content '%logfile%' | Select-String 'Position:.*BTC @' | Select-Object -Last 1; if ($hasPos) { $hasPos } else { Write-Host '   No open position' }"
powershell -Command "Get-Content '%logfile%' | Select-String 'P&L:.*%%' | Select-Object -Last 1"

REM ============================================================================
REM SECTION 4: RECENT ACTIVITY (Last 10 lines)
REM ============================================================================
echo.
echo [4] RECENT ACTIVITY (Last 10 log entries)
echo --------------------------------------------------------------------------------
powershell -Command "Get-Content '%logfile%' -Tail 10 | ForEach-Object { if ($_ -match '\| (INFO|SUCCESS|WARNING|ERROR)') { Write-Host $_.Substring(0, [Math]::Min(100, $_.Length)) } }"

REM ============================================================================
REM SECTION 5: NEXT UPDATE
REM ============================================================================
echo.
echo [5] STATUS
echo --------------------------------------------------------------------------------
powershell -Command "Get-Content '%logfile%' | Select-String 'Next update' | Select-Object -Last 1"
powershell -Command "Get-Content '%logfile%' | Select-String 'Market Regime' | Select-Object -Last 1"
powershell -Command "Get-Content '%logfile%' | Select-String 'Current Price' | Select-Object -Last 1"

echo.
echo ================================================================================
echo [R] Refresh (manual)  [Q] Quit
echo.
echo Bot updates every 5 minutes. Press R to manually refresh this screen.
echo ================================================================================
echo.

set /p choice="Enter your choice: "

if /i "%choice%"=="R" goto refresh
if /i "%choice%"=="Q" exit

goto refresh
