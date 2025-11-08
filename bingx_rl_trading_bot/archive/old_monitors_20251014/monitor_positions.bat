@echo off
REM Position Monitor - Shows position and P&L updates

color 0E
title Position Monitor - Live P&L Tracking

echo ================================================================================
echo POSITION MONITOR - Live P&L and Signal Tracking
echo ================================================================================
echo.
echo Showing position-related logs:
echo   - Position entries (LONG/SHORT)
echo   - Current P&L updates
echo   - Position holding time
echo   - Entry and current signals
echo   - Exit signals and decisions
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
echo POSITION UPDATES (Last 40 position-related logs)
echo ===============================================================================
echo.

REM Show last 40 position-related logs
powershell -Command "Get-Content '%logfile%' | Select-String -Pattern 'Position|P/L|Signal|EXECUTING|POSITION' | Select-Object -Last 40"

echo.
echo ===============================================================================
echo REAL-TIME POSITION UPDATES
echo ===============================================================================
echo.

REM Monitor position logs in real-time
powershell -Command "Get-Content '%logfile%' -Wait -Tail 0 | Select-String -Pattern 'Position|P/L|Signal|EXECUTING|POSITION|Exit Model Signal'"

pause
