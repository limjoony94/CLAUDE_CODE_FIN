@echo off
title RSI Trend Filter Monitor
color 0B

cd /d "%~dp0"

python scripts/monitoring/rsi_trend_filter_monitor.py

pause
