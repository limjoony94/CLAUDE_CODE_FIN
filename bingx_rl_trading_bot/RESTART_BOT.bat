@echo off
REM ====================================================================
REM RESTART_BOT.bat - Restart Opportunity Gating Bot
REM ====================================================================

echo.
echo ========================================
echo  Restarting Opportunity Gating Bot
echo ========================================
echo.

REM Change to script directory
cd /d "%~dp0"

REM Step 1: Stop bot
echo Step 1: Stopping bot...
echo.
call STOP_BOT.bat

REM Wait for clean shutdown
echo.
echo Waiting for clean shutdown...
timeout /t 3 /nobreak >nul

REM Step 2: Start bot
echo.
echo Step 2: Starting bot...
echo.
call START_BOT.bat

echo.
echo ========================================
echo  Restart complete!
echo ========================================
echo.

pause
