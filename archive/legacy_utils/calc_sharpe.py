#!/usr/bin/env python3
"""Calculate Sharpe ratio from backtest results"""
import pandas as pd
import numpy as np

# Read backtest results
df = pd.read_csv('results/full_backtest_OPTION_B_threshold_080_20251024_052634.csv')

# Calculate Sharpe ratio
returns = df['total_return_pct'] / 100  # Convert to decimal
mean_return = returns.mean()
std_return = returns.std()

# Sharpe ratio (per 5-day window)
sharpe_5d = mean_return / std_return if std_return > 0 else 0

# Annualize (assuming 73 periods per year: 365 days / 5 days)
periods_per_year = 365 / 5
annualized_sharpe = sharpe_5d * np.sqrt(periods_per_year)

print(f"Mean return per window: {mean_return*100:.2f}%")
print(f"Std of returns: {std_return*100:.2f}%")
print(f"Sharpe ratio (5-day window): {sharpe_5d:.3f}")
print(f"Annualized Sharpe ratio: {annualized_sharpe:.3f}")
print(f"\nTotal windows: {len(df)}")
print(f"Total LONG: {df['long_trades'].sum()}")
print(f"Total SHORT: {df['short_trades'].sum()}")
