@echo off
title RSI Trend Filter Bot v1.0
color 0A

echo ============================================================
echo   RSI Trend Filter Bot v1.0
echo   Strategy: RSI(14) 40/60 + EMA100 Trend Filter
echo   TP: 3.0%% / SL: 2.0%% / Leverage: 4x
echo ============================================================
echo.

cd /d "%~dp0"

echo Starting bot...
echo.

python scripts/production/rsi_trend_filter_bot.py

echo.
echo Bot stopped.
pause
