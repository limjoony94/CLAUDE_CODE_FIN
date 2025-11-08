@echo off
REM ML Exit Signals Monitor - Shows only Exit-related logs

color 0B
title ML Exit Signals Monitor - Exit Model Activity Only

echo ================================================================================
echo ML EXIT SIGNALS MONITOR
echo ================================================================================
echo.
echo Showing ONLY exit-related logs:
echo   - Exit Model Signal (LONG/SHORT)
echo   - ML Exit decisions
echo   - Exit reasons and probabilities
echo   - Position closing events
echo.
echo Expected ML Exit Behavior:
echo   - Threshold: 0.75 (75%% probability)
echo   - Expected ML Exit Rate: 87.6%%
echo   - Expected Max Hold: 12.4%%
echo.
echo ================================================================================
echo Press Ctrl+C to stop monitoring
echo ================================================================================
echo.

cd /d "C:\Users\J\OneDrive\CLAUDE_CODE_FIN\bingx_rl_trading_bot"

REM Find the most recent log file
for /f "delims=" %%i in ('dir /b /o-d logs\phase4_dynamic_testnet_trading_*.log 2^>nul') do (
    set logfile=logs\%%i
    goto :found
)

:notfound
echo [ERROR] No log files found matching: logs\phase4_dynamic_testnet_trading_*.log
echo.
echo Bot may not be running yet or log file hasn't been created.
echo.
pause
exit /b

:found
echo Monitoring: %logfile%
echo.
echo ===============================================================================
echo EXIT SIGNALS (Showing last 30 exit-related logs)
echo ===============================================================================
echo.

REM Show last 30 exit-related logs
powershell -Command "Get-Content '%logfile%' | Select-String -Pattern 'Exit|EXIT|exit' | Select-Object -Last 30"

echo.
echo ===============================================================================
echo REAL-TIME EXIT UPDATES (Only exit-related logs will appear)
echo ===============================================================================
echo.

REM Monitor exit-related logs in real-time
powershell -Command "Get-Content '%logfile%' -Wait -Tail 0 | Select-String -Pattern 'Exit|EXIT|exit'"

pause
