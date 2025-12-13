@echo off
title ADX Supertrend Trail Bot Monitor
echo ==========================================
echo  ADX Trend + Supertrend Trail Bot Monitor
echo ==========================================
echo.
echo Press Ctrl+C to exit
echo ==========================================
echo.

cd /d "%~dp0"
python scripts/monitoring/adx_supertrend_trail_monitor.py

pause
