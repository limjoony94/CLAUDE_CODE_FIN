"""
Calculate Buy-and-Hold Baseline
================================

Calculate simple buy-and-hold return for test period
to determine if period was profitable or market was bearish.

Created: 2025-10-30
"""

import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Data
DATA_DIR = PROJECT_ROOT / "data" / "features"
CSV_FILE = DATA_DIR / "BTCUSDT_5m_features_exit_ready.csv"

print("=" * 80)
print("BUY-AND-HOLD BASELINE CALCULATION")
print("=" * 80)
print()

# Load data
print("Loading data...")
df = pd.read_csv(CSV_FILE)
df['timestamp'] = pd.to_datetime(df['timestamp'])
df = df.sort_values('timestamp').reset_index(drop=True)
print(f"✅ Loaded {len(df):,} candles")
print()

# Period info
start_time = df['timestamp'].iloc[0]
end_time = df['timestamp'].iloc[-1]
duration_days = (end_time - start_time).days
duration_hours = (end_time - start_time).total_seconds() / 3600

print(f"Test Period:")
print(f"  Start: {start_time}")
print(f"  End: {end_time}")
print(f"  Duration: {duration_days} days ({duration_hours:.1f} hours)")
print()

# Price analysis
start_price = df['close'].iloc[0]
end_price = df['close'].iloc[-1]
price_change = end_price - start_price
price_change_pct = (end_price - start_price) / start_price

print(f"Price Movement:")
print(f"  Start Price: ${start_price:,.2f}")
print(f"  End Price: ${end_price:,.2f}")
print(f"  Change: ${price_change:+,.2f} ({price_change_pct*100:+.2f}%)")
print()

# Min/Max during period
min_price = df['close'].min()
max_price = df['close'].max()
max_drawdown_pct = (min_price - start_price) / start_price
max_gain_pct = (max_price - start_price) / start_price

print(f"Price Range:")
print(f"  Minimum: ${min_price:,.2f} ({max_drawdown_pct*100:+.2f}% from start)")
print(f"  Maximum: ${max_price:,.2f} ({max_gain_pct*100:+.2f}% from start)")
print()

# Buy-and-Hold scenarios
print("=" * 80)
print("BUY-AND-HOLD SCENARIOS")
print("=" * 80)
print()

# 1x leverage (spot)
spot_return = price_change_pct
print(f"1x Leverage (Spot):")
print(f"  Initial: $10,000")
print(f"  Final: ${10000 * (1 + spot_return):,.2f}")
print(f"  Return: {spot_return*100:+.2f}%")
print()

# 2x leverage
leverage_2x = spot_return * 2
print(f"2x Leverage:")
print(f"  Initial: $10,000")
print(f"  Final: ${10000 * (1 + leverage_2x):,.2f}")
print(f"  Return: {leverage_2x*100:+.2f}%")
print()

# 4x leverage
leverage_4x = spot_return * 4
print(f"4x Leverage:")
print(f"  Initial: $10,000")
print(f"  Final: ${10000 * (1 + leverage_4x):,.2f}")
print(f"  Return: {leverage_4x*100:+.2f}%")
print()

# Annualized returns
days_in_year = 365
annualized_return = (1 + spot_return) ** (days_in_year / duration_days) - 1

print(f"Annualized Return (Spot): {annualized_return*100:+.2f}%")
print()

# Volatility analysis
returns = df['close'].pct_change().dropna()
# Set timestamp as index for resampling
df_with_index = df.set_index('timestamp')
returns_with_index = df_with_index['close'].pct_change().dropna()
volatility_daily = returns_with_index.std()
volatility_annualized = volatility_daily * np.sqrt(252)  # Trading days

print(f"Volatility:")
print(f"  Daily Std Dev: {volatility_daily*100:.4f}%")
print(f"  Annualized: {volatility_annualized*100:.2f}%")
print()

# Sharpe Ratio (assuming 0% risk-free rate)
sharpe_ratio = (spot_return * (365 / duration_days)) / volatility_annualized if volatility_annualized > 0 else 0

print(f"Sharpe Ratio (Annualized): {sharpe_ratio:.3f}")
print()

# Strategy comparison
print("=" * 80)
print("STRATEGY PERFORMANCE vs BUY-AND-HOLD")
print("=" * 80)
print()

strategies = {
    'Buy-and-Hold 4x': leverage_4x * 100,
    'Strategy E (Technical)': -68.07,
    'Strategy A (Exit-Only)': -94.75,
    'Strategy F (Volatility)': -2.58,
    'ML Entry (Best)': -99.68,  # From full-data ensemble
}

print(f"{'Strategy':<25} {'Return':<15} {'vs B&H 4x':<15}")
print("-" * 55)
for name, ret in strategies.items():
    vs_bh = ret - (leverage_4x * 100)
    print(f"{name:<25} {ret:>+7.2f}%      {vs_bh:>+7.2f}%")

print()

# Market assessment
print("=" * 80)
print("MARKET ASSESSMENT")
print("=" * 80)
print()

if price_change_pct > 0.05:
    print("✅ BULLISH PERIOD: Price rose >5%")
    print("   → Strategies SHOULD have profited")
    print("   → Failure indicates strategy problems")
elif price_change_pct < -0.05:
    print("🔴 BEARISH PERIOD: Price fell >5%")
    print("   → Strategies correctly avoided losses")
    print("   → LONG-biased strategies expected to fail")
elif abs(price_change_pct) < 0.05:
    print("⚠️ RANGING MARKET: Price moved <5%")
    print("   → Choppy, trendless conditions")
    print("   → Trend-following strategies expected to struggle")

print()

# Visualize price movement
print("Generating price chart...")
plt.figure(figsize=(14, 8))

# Plot 1: Price over time
plt.subplot(2, 1, 1)
plt.plot(df['timestamp'], df['close'], linewidth=0.5, alpha=0.7)
plt.axhline(start_price, color='green', linestyle='--', label=f'Start: ${start_price:,.0f}', alpha=0.7)
plt.axhline(end_price, color='red', linestyle='--', label=f'End: ${end_price:,.0f}', alpha=0.7)
plt.title(f'BTC/USDT Price Movement\n{start_time.date()} to {end_time.date()}', fontsize=14, fontweight='bold')
plt.ylabel('Price (USDT)', fontsize=12)
plt.legend()
plt.grid(True, alpha=0.3)

# Plot 2: Cumulative returns
plt.subplot(2, 1, 2)
cumulative_returns = (1 + returns).cumprod() - 1
plt.plot(df['timestamp'].iloc[1:], cumulative_returns * 100, linewidth=1)
plt.axhline(0, color='black', linestyle='-', alpha=0.3)
plt.title('Cumulative Returns (%)', fontsize=14, fontweight='bold')
plt.ylabel('Return (%)', fontsize=12)
plt.xlabel('Date', fontsize=12)
plt.grid(True, alpha=0.3)

plt.tight_layout()

# Save chart
results_dir = PROJECT_ROOT / "results"
results_dir.mkdir(exist_ok=True)
chart_file = results_dir / "buyhold_baseline_chart.png"
plt.savefig(chart_file, dpi=150, bbox_inches='tight')
print(f"✅ Chart saved: {chart_file}")
print()

print("=" * 80)
print("BASELINE CALCULATION COMPLETE")
print("=" * 80)
print()

print("**CRITICAL INSIGHT**:")
if price_change_pct > 0:
    print(f"  Market was UP {price_change_pct*100:+.2f}% during test period.")
    print(f"  Buy-and-hold 4x would have returned {leverage_4x*100:+.2f}%.")
    print(f"  ALL strategies LOST money in a RISING market.")
    print(f"  → Strategies are FUNDAMENTALLY BROKEN, not market conditions.")
else:
    print(f"  Market was DOWN {price_change_pct*100:.2f}% during test period.")
    print(f"  Buy-and-hold would have LOST money.")
    print(f"  Strategies losing money is EXPECTED in bearish conditions.")
    print(f"  → Need to test on BULLISH period to validate strategies.")
