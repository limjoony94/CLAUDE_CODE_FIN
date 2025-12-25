@echo off
REM ============================================
REM RSI Martingale Bot Starter (Background)
REM Strategy: RSI(7) 25/75 + Martingale 8x
REM ============================================

cd /d "%~dp0"

REM Check if already running
tasklist /FI "WINDOWTITLE eq RSI_MARTINGALE_BOT" 2>NUL | find /I "python.exe" >NUL
if not errorlevel 1 (
    echo [WARNING] RSI Martingale Bot is already running!
    echo Use STOP_RSI_MARTINGALE.bat to stop it first.
    pause
    exit /b 1
)

REM Create logs directory if not exists
if not exist "logs" mkdir logs

REM Get current date for log file
for /f "tokens=2 delims==" %%I in ('wmic os get localdatetime /value') do set datetime=%%I
set logdate=%datetime:~0,8%

echo ============================================
echo Starting RSI Martingale Bot...
echo Strategy: RSI(7) 25/75 + Martingale 8x
echo TP: 2.0%% | SL: 0.7%% | R:R = 2.86:1
echo ============================================
echo.
echo Log file: logs\rsi_martingale_bot_%logdate%.log
echo.

REM Start bot in background with unique window title
start "RSI_MARTINGALE_BOT" /MIN cmd /c "python scripts/production/rsi_martingale_bot.py >> logs\rsi_martingale_bot_%logdate%.log 2>&1"

echo [OK] Bot started in background!
echo.
echo Use MONITOR_RSI_MARTINGALE.bat to monitor
echo Use STOP_RSI_MARTINGALE.bat to stop
echo.
timeout /t 3
