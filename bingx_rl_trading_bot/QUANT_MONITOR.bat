@echo off
REM ============================================================================
REM PROFESSIONAL QUANTITATIVE TRADING MONITOR
REM ============================================================================
REM Institutional-grade real-time monitoring with advanced metrics
REM
REM Features:
REM - Risk-adjusted performance (Sharpe, Sortino, Calmar)
REM - Real-time risk analytics (VaR, CVaR, Drawdown)
REM - Signal quality tracking & model diagnostics
REM - Market regime analysis
REM - Automated alert system
REM - ASCII visualization
REM ============================================================================

title Professional Quant Trading Monitor
color 0A

cd /d "%~dp0"

REM Check if Python is available
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python not found in PATH
    echo Please install Python 3.7+ or add it to PATH
    pause
    exit /b 1
)

REM Check if NumPy is installed
python -c "import numpy" >nul 2>&1
if errorlevel 1 (
    echo WARNING: NumPy not found. Installing...
    pip install numpy
    if errorlevel 1 (
        echo ERROR: Failed to install NumPy
        pause
        exit /b 1
    )
)

echo.
echo ================================================================================
echo   PROFESSIONAL QUANTITATIVE TRADING MONITOR
echo ================================================================================
echo.
echo Starting advanced monitoring system...
echo.
echo Features enabled:
echo   - Real-time performance metrics (Sharpe, Sortino, Calmar)
echo   - Risk analytics (VaR, CVaR, Maximum Drawdown)
echo   - Signal quality tracking
echo   - Market regime analysis
echo   - Automated alerts
echo   - ASCII visualization
echo.
echo ================================================================================
echo.

python scripts\monitoring\quant_monitor.py

if errorlevel 1 (
    echo.
    echo ERROR: Monitor failed to start
    echo Check that:
    echo   1. Trading bot is running
    echo   2. State file exists: results\opportunity_gating_bot_4x_state.json
    echo   3. Log files exist in: logs\opportunity_gating_bot_4x_*.log
    echo.
    pause
)
