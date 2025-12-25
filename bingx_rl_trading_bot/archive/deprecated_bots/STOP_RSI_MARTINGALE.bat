@echo off
REM ============================================
REM RSI Martingale Bot Stopper
REM ============================================

cd /d "%~dp0"

echo ============================================
echo Stopping RSI Martingale Bot...
echo ============================================
echo.

REM Find and kill the bot process by window title
taskkill /FI "WINDOWTITLE eq RSI_MARTINGALE_BOT" /F >NUL 2>&1

REM Also try to kill any python process running the bot script
wmic process where "commandline like '%%rsi_martingale_bot%%'" call terminate >NUL 2>&1

REM Remove lock file if exists
set LOCKFILE=results\rsi_martingale_bot.lock
if exist "%LOCKFILE%" (
    del /f "%LOCKFILE%" >NUL 2>&1
    echo [OK] Lock file removed
)

echo [OK] RSI Martingale Bot stopped!
echo.
timeout /t 2
