@echo off
REM ====================================================================
REM STOP_BOT.bat - Stop Opportunity Gating Bot
REM ====================================================================

echo.
echo ========================================
echo  Stopping Opportunity Gating Bot
echo ========================================
echo.

REM Check if bot is running
tasklist /FI "WINDOWTITLE eq Opportunity Gating Bot*" 2>NUL | find /I /N "python.exe">NUL
if "%ERRORLEVEL%"=="1" (
    echo.
    echo Bot is not running.
    echo.
    pause
    exit /b 0
)

REM Get process info
echo Current bot processes:
tasklist /FI "WINDOWTITLE eq Opportunity Gating Bot*" /V
echo.

REM Stop the bot
echo Stopping bot...
taskkill /FI "WINDOWTITLE eq Opportunity Gating Bot*" /T /F

REM Wait a moment
timeout /t 2 /nobreak >nul

REM Verify stopped
tasklist /FI "WINDOWTITLE eq Opportunity Gating Bot*" 2>NUL | find /I /N "python.exe">NUL
if "%ERRORLEVEL%"=="1" (
    echo.
    echo ========================================
    echo  Bot stopped successfully!
    echo ========================================
    echo.
) else (
    echo.
    echo WARNING: Bot may still be running
    echo Try manual termination if needed
    echo.
)

pause
