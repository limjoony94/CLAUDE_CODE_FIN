@echo off
REM Production Bot Restart Script (Windows)
REM Upgrades from Phase 2 (33 features) to Phase 4 Base (37 features)
REM Expected improvement: +920%% (0.75%% → 7.68%%)

setlocal enabledelayedexpansion

set PROJECT_DIR=C:\Users\J\OneDrive\CLAUDE_CODE_FIN\bingx_rl_trading_bot
set LOG_DIR=%PROJECT_DIR%\logs
set SCRIPT=%PROJECT_DIR%\scripts\production\sweet2_paper_trading.py

echo ==========================================
echo Production Bot Restart
echo ==========================================
echo Upgrade: Phase 2 -^> Phase 4 Base
echo Expected: 0.75%% -^> 7.68%% (+920%%)
echo.

REM Step 1: Stop current bot
echo [1/5] Stopping current bot (Phase 2)...
taskkill /F /IM python.exe /FI "WINDOWTITLE eq sweet2*" 2>nul
timeout /t 2 /nobreak >nul
echo Done

REM Step 2: Verify Phase 4 Base model exists
echo.
echo [2/5] Verifying Phase 4 Base model...
set MODEL_FILE=%PROJECT_DIR%\models\xgboost_v4_phase4_advanced_lookahead3_thresh0.pkl
set FEATURES_FILE=%PROJECT_DIR%\models\xgboost_v4_phase4_advanced_lookahead3_thresh0_features.txt

if not exist "%MODEL_FILE%" (
    echo ERROR: Phase 4 Base model not found: %MODEL_FILE%
    exit /b 1
)

if not exist "%FEATURES_FILE%" (
    echo ERROR: Features file not found: %FEATURES_FILE%
    exit /b 1
)

echo Phase 4 Base model verified

REM Step 3: Backup old logs
echo.
echo [3/5] Backing up logs...
if not exist "%LOG_DIR%\backup" mkdir "%LOG_DIR%\backup"
copy "%LOG_DIR%\sweet2_paper_trading_*.log" "%LOG_DIR%\backup\" 2>nul
echo Logs backed up (if any existed)

REM Step 4: Start new bot with Phase 4 Base
echo.
echo [4/5] Starting bot with Phase 4 Base model...
cd /d "%PROJECT_DIR%"

REM Get timestamp
for /f "tokens=2-4 delims=/ " %%a in ('date /t') do (set mydate=%%c%%a%%b)
for /f "tokens=1-2 delims=/:" %%a in ('time /t') do (set mytime=%%a%%b)
set TIMESTAMP=%mydate%_%mytime%

REM Start bot in new window
start "Sweet2 Trading Bot - Phase 4 Base" /MIN python "%SCRIPT%"

echo Bot started in new window

REM Step 5: Verify startup
echo.
echo [5/5] Verifying startup...
timeout /t 5 /nobreak >nul

REM Check latest log
for /f "delims=" %%a in ('dir /b /od "%LOG_DIR%\sweet2_paper_trading_*.log" 2^>nul') do set LATEST_LOG=%%a

if defined LATEST_LOG (
    echo.
    echo === Latest Log (last 20 lines) ===
    powershell -Command "Get-Content '%LOG_DIR%\%LATEST_LOG%' -Tail 20"

    echo.
    echo === Checking for Phase 4 Base confirmation ===
    findstr /C:"Phase 4 Base" "%LOG_DIR%\%LATEST_LOG%" >nul 2>&1
    if !errorlevel! equ 0 (
        echo CONFIRMED: Phase 4 Base model loaded!
    ) else (
        findstr /C:"37 features" "%LOG_DIR%\%LATEST_LOG%" >nul 2>&1
        if !errorlevel! equ 0 (
            echo CONFIRMED: 37 features loaded (Phase 4 Base^)
        ) else (
            echo WARNING: Could not confirm Phase 4 Base in logs
            echo    Check manually: %LOG_DIR%\%LATEST_LOG%
        )
    )
)

echo.
echo ==========================================
echo RESTART COMPLETE
echo ==========================================
echo.
echo Next Steps:
echo 1. Monitor logs: tail -f %LOG_DIR%\%LATEST_LOG%
echo 2. Check first trades (expect 2-3 per day^)
echo 3. Verify XGBoost probabilities ^> 0.7 for entries
echo 4. Monitor returns vs 7.68%% expected
echo.
echo Expected Performance (Phase 4 Base^):
echo   - Returns: 7.68%% per 5 days (~1.5%% per day^)
echo   - Win Rate: 69.1%%
echo   - Trades: ~15 per 5 days (3 per day^)
echo   - Max Drawdown: ^<1%%
echo.
echo ==========================================

pause
