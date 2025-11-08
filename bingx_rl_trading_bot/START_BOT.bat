@echo off
REM ====================================================================
REM START_BOT.bat - Start Opportunity Gating Bot (4x Leverage)
REM ====================================================================

echo.
echo ========================================
echo  Starting Opportunity Gating Bot
echo ========================================
echo.

REM Change to project directory
cd /d "%~dp0"

REM Activate virtual environment (if exists)
if exist "venv\Scripts\activate.bat" (
    echo Activating virtual environment...
    call venv\Scripts\activate.bat
) else (
    echo WARNING: Virtual environment not found, using system Python
)

REM Check if bot is already running
tasklist /FI "WINDOWTITLE eq Opportunity Gating Bot*" 2>NUL | find /I /N "python.exe">NUL
if "%ERRORLEVEL%"=="0" (
    echo.
    echo ERROR: Bot is already running!
    echo Please stop the bot first using STOP_BOT.bat
    echo.
    pause
    exit /b 1
)

REM Start bot in new window
echo Starting bot...
start "Opportunity Gating Bot - 4x Leverage" python scripts\production\opportunity_gating_bot_4x.py

REM Wait a moment for startup
timeout /t 3 /nobreak >nul

REM Check if started successfully
tasklist /FI "WINDOWTITLE eq Opportunity Gating Bot*" 2>NUL | find /I /N "python.exe">NUL
if "%ERRORLEVEL%"=="0" (
    echo.
    echo ========================================
    echo  Bot started successfully!
    echo ========================================
    echo.
    echo Check logs at: logs\opportunity_gating_bot_4x_*.log
    echo.
) else (
    echo.
    echo ERROR: Failed to start bot!
    echo Check Python installation and dependencies
    echo.
)

pause
