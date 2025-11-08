@echo off
REM Trade History Viewer - Shows detailed trade history with entry/exit information

color 0E
title Trade History Viewer

:refresh
cls

echo ================================================================================
echo TRADE HISTORY VIEWER - Detailed Trade Information
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
echo RECENT TRADE ENTRIES (Last 10)
echo ================================================================================
echo.

powershell -Command "Get-Content '%logfile%' | Select-String 'Opening.*position|LONG signal.*Enter|SHORT signal.*Enter' | Select-Object -Last 10"

echo.
echo ================================================================================
echo RECENT TRADE EXITS (Last 10)
echo ================================================================================
echo.

powershell -Command "Get-Content '%logfile%' | Select-String 'POSITION CLOSED|Exit Reason:|Closing position' | Select-Object -Last 10"

echo.
echo ================================================================================
echo TRADE OUTCOMES (Last 10)
echo ================================================================================
echo.

powershell -Command "Get-Content '%logfile%' | Select-String 'Return:.*%%|P&L:.*USDT' | Select-Object -Last 10"

echo.
echo ================================================================================
echo CURRENT OPEN POSITIONS
echo ================================================================================
echo.

powershell -Command "$openPos = Get-Content '%logfile%' | Select-String 'Open Position:|OPEN.*held' | Select-Object -Last 5; if ($openPos) { $openPos } else { Write-Host 'No open positions' }"

echo.
echo ================================================================================
echo POSITION HOLDING TIME ANALYSIS
echo ================================================================================
echo.

powershell -Command "Get-Content '%logfile%' | Select-String 'held|Holding:' | Select-Object -Last 10"

echo.
echo ================================================================================
echo ENTRY/EXIT SIGNAL PAIRS
echo ================================================================================
echo.

echo Recent Entry Signals with Probabilities:
powershell -Command "Get-Content '%logfile%' | Select-String 'signal.*prob=|probability.*Enter' | Select-Object -Last 10"

echo.
echo Recent Exit Signals with Probabilities:
powershell -Command "Get-Content '%logfile%' | Select-String 'Exit.*prob=|ML Exit' | Select-Object -Last 10"

echo.
echo ================================================================================
echo TRADE SIZE ANALYSIS
echo ================================================================================
echo.

powershell -Command "Get-Content '%logfile%' | Select-String 'Position size:|Quantity:' | Select-Object -Last 10"

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
