@echo off
REM ============================================
REM RSI Martingale Bot Monitor
REM Strategy: RSI(7) 25/75 + Martingale 8x
REM ============================================

cd /d "%~dp0"

echo Starting RSI Martingale Monitor...
echo Press Ctrl+C to exit
echo.

python scripts/monitoring/rsi_martingale_monitor.py
