@echo off
title ADX Supertrend Trail Bot
echo ==========================================
echo  ADX Trend + Supertrend Trail Bot v1.0
echo ==========================================
echo.
echo Strategy: ADX +DI/-DI Crossover (ADX > 20)
echo Exit: TP 2.0%% / SL Supertrend Trailing
echo.
echo Research Results:
echo   - Return: +1276.6%% (314 days)
echo   - Win Rate: 70.3%%
echo   - Max Drawdown: 21.6%%
echo   - Trades: 1.98/day
echo.
echo ==========================================
echo.

cd /d "%~dp0"
python scripts/production/adx_supertrend_trail_bot.py

pause
