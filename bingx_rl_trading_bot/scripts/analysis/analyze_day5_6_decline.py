"""
Day 5-6 Decline Analysis
=========================

Deep dive into the sharp decline during Day 5-6 (Day 28-29 in 30-day test):
- 2025-10-15: -7.75% decline
- 2025-10-16: -5.68% decline

Analyzes:
1. All trades during this period
2. Losing trade patterns
3. Market conditions
4. ML model performance
5. Root causes and mitigation strategies
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta

PROJECT_ROOT = Path(__file__).parent.parent.parent
RESULTS_DIR = PROJECT_ROOT / "results"
DATA_DIR = PROJECT_ROOT / "data" / "historical"

print("="*80)
print("DAY 5-6 DECLINE ANALYSIS")
print("="*80)
print()

# =============================================================================
# STEP 1: Load Trade Data
# =============================================================================

print("STEP 1: Loading Trade Data")
print("-"*80)

# Find latest backtest CSV
csv_files = sorted(RESULTS_DIR.glob("backtest_30days_*.csv"), key=lambda x: x.stat().st_mtime, reverse=True)
if not csv_files:
    print("❌ No backtest CSV found!")
    exit(1)

csv_path = csv_files[0]
print(f"Loading: {csv_path.name}")

df_trades = pd.read_csv(csv_path)
print(f"✅ Loaded {len(df_trades)} trades\n")

# Convert timestamps
df_trades['entry_timestamp'] = pd.to_datetime(df_trades['entry_timestamp'])
df_trades['exit_timestamp'] = pd.to_datetime(df_trades['exit_timestamp'])

# Define Day 5-6 period (from 7-day test perspective)
# Day 1: 2025-10-11
# Day 5: 2025-10-15
# Day 6: 2025-10-16

day5_start = pd.to_datetime('2025-10-15 00:00:00')
day5_end = pd.to_datetime('2025-10-15 23:59:59')
day6_start = pd.to_datetime('2025-10-16 00:00:00')
day6_end = pd.to_datetime('2025-10-16 23:59:59')

# Filter trades that were active during Day 5-6
# Include trades that entered before and exited during, or entered during
df_day5 = df_trades[
    (df_trades['entry_timestamp'] <= day5_end) &
    (df_trades['exit_timestamp'] >= day5_start)
]

df_day6 = df_trades[
    (df_trades['entry_timestamp'] <= day6_end) &
    (df_trades['exit_timestamp'] >= day6_start)
]

print(f"Trades during Day 5 (2025-10-15): {len(df_day5)}")
print(f"Trades during Day 6 (2025-10-16): {len(df_day6)}\n")

# =============================================================================
# STEP 2: Analyze Day 5 Trades (-7.75% decline)
# =============================================================================

print("="*80)
print("STEP 2: Day 5 Analysis (2025-10-15) - Lost 7.75%")
print("="*80)
print()

if len(df_day5) > 0:
    total_pnl_day5 = df_day5['pnl_net'].sum()
    losers_day5 = df_day5[df_day5['pnl_net'] < 0]
    winners_day5 = df_day5[df_day5['pnl_net'] > 0]

    print(f"📊 Day 5 Summary:")
    print(f"   Total Trades: {len(df_day5)}")
    print(f"   Total P&L: ${total_pnl_day5:.2f}")
    print(f"   Winners: {len(winners_day5)} (${winners_day5['pnl_net'].sum():.2f})")
    print(f"   Losers: {len(losers_day5)} (${losers_day5['pnl_net'].sum():.2f})")
    print(f"   Win Rate: {len(winners_day5)/len(df_day5)*100:.1f}%")

    # Analyze losing trades
    if len(losers_day5) > 0:
        print(f"\n📉 Losing Trades Analysis:")
        print(f"   Count: {len(losers_day5)}")
        print(f"   Total Loss: ${losers_day5['pnl_net'].sum():.2f}")
        print(f"   Avg Loss: ${losers_day5['pnl_net'].mean():.2f}")
        print(f"   Largest Loss: ${losers_day5['pnl_net'].min():.2f}")

        # By side
        long_losers = losers_day5[losers_day5['side'] == 'LONG']
        short_losers = losers_day5[losers_day5['side'] == 'SHORT']
        print(f"\n   By Side:")
        print(f"   - LONG: {len(long_losers)} (${long_losers['pnl_net'].sum():.2f})")
        print(f"   - SHORT: {len(short_losers)} (${short_losers['pnl_net'].sum():.2f})")

        # By exit reason
        print(f"\n   By Exit Reason:")
        exit_reasons = losers_day5['exit_reason'].value_counts()
        for reason, count in exit_reasons.items():
            reason_trades = losers_day5[losers_day5['exit_reason'] == reason]
            print(f"   - {reason}: {count} (${reason_trades['pnl_net'].sum():.2f})")

        # Detail worst 5 trades
        print(f"\n   🔍 Worst 5 Trades:")
        worst_5 = losers_day5.nsmallest(5, 'pnl_net')
        for idx, trade in worst_5.iterrows():
            print(f"\n   Trade #{idx}:")
            print(f"      Side: {trade['side']}")
            print(f"      Entry: {trade['entry_timestamp']}")
            print(f"      Exit: {trade['exit_timestamp']}")
            print(f"      P&L: ${trade['pnl_net']:.2f} ({trade['pnl_pct']:.2f}%)")
            print(f"      Entry Price: ${trade['entry_price']:.2f}")
            print(f"      Exit Price: ${trade['exit_price']:.2f}")
            print(f"      Entry Prob: {trade['entry_prob']:.3f}")
            print(f"      Position Size: {trade['position_size_pct']*100:.1f}%")
            print(f"      Hold Time: {trade['hold_time']*5/60:.1f} hours")
            print(f"      Exit Reason: {trade['exit_reason']}")

else:
    print("No trades found for Day 5")

# =============================================================================
# STEP 3: Analyze Day 6 Trades (-5.68% decline)
# =============================================================================

print("\n" + "="*80)
print("STEP 3: Day 6 Analysis (2025-10-16) - Lost 5.68%")
print("="*80)
print()

if len(df_day6) > 0:
    total_pnl_day6 = df_day6['pnl_net'].sum()
    losers_day6 = df_day6[df_day6['pnl_net'] < 0]
    winners_day6 = df_day6[df_day6['pnl_net'] > 0]

    print(f"📊 Day 6 Summary:")
    print(f"   Total Trades: {len(df_day6)}")
    print(f"   Total P&L: ${total_pnl_day6:.2f}")
    print(f"   Winners: {len(winners_day6)} (${winners_day6['pnl_net'].sum():.2f})")
    print(f"   Losers: {len(losers_day6)} (${losers_day6['pnl_net'].sum():.2f})")
    print(f"   Win Rate: {len(winners_day6)/len(df_day6)*100:.1f}%")

    # Analyze losing trades
    if len(losers_day6) > 0:
        print(f"\n📉 Losing Trades Analysis:")
        print(f"   Count: {len(losers_day6)}")
        print(f"   Total Loss: ${losers_day6['pnl_net'].sum():.2f}")
        print(f"   Avg Loss: ${losers_day6['pnl_net'].mean():.2f}")
        print(f"   Largest Loss: ${losers_day6['pnl_net'].min():.2f}")

        # By side
        long_losers = losers_day6[losers_day6['side'] == 'LONG']
        short_losers = losers_day6[losers_day6['side'] == 'SHORT']
        print(f"\n   By Side:")
        print(f"   - LONG: {len(long_losers)} (${long_losers['pnl_net'].sum():.2f})")
        print(f"   - SHORT: {len(short_losers)} (${short_losers['pnl_net'].sum():.2f})")

        # By exit reason
        print(f"\n   By Exit Reason:")
        exit_reasons = losers_day6['exit_reason'].value_counts()
        for reason, count in exit_reasons.items():
            reason_trades = losers_day6[losers_day6['exit_reason'] == reason]
            print(f"   - {reason}: {count} (${reason_trades['pnl_net'].sum():.2f})")

        # Detail worst 5 trades
        print(f"\n   🔍 Worst 5 Trades:")
        worst_5 = losers_day6.nsmallest(5, 'pnl_net')
        for idx, trade in worst_5.iterrows():
            print(f"\n   Trade #{idx}:")
            print(f"      Side: {trade['side']}")
            print(f"      Entry: {trade['entry_timestamp']}")
            print(f"      Exit: {trade['exit_timestamp']}")
            print(f"      P&L: ${trade['pnl_net']:.2f} ({trade['pnl_pct']:.2f}%)")
            print(f"      Entry Price: ${trade['entry_price']:.2f}")
            print(f"      Exit Price: ${trade['exit_price']:.2f}")
            print(f"      Entry Prob: {trade['entry_prob']:.3f}")
            print(f"      Position Size: {trade['position_size_pct']*100:.1f}%")
            print(f"      Hold Time: {trade['hold_time']*5/60:.1f} hours")
            print(f"      Exit Reason: {trade['exit_reason']}")

else:
    print("No trades found for Day 6")

# =============================================================================
# STEP 4: Market Conditions Analysis
# =============================================================================

print("\n" + "="*80)
print("STEP 4: Market Conditions During Day 5-6")
print("="*80)
print()

# Load price data
data_file = DATA_DIR / "BTCUSDT_5m_max.csv"
df_price = pd.read_csv(data_file)
df_price['timestamp'] = pd.to_datetime(df_price['timestamp'])

# Filter to Day 5-6
df_day5_price = df_price[
    (df_price['timestamp'] >= day5_start) &
    (df_price['timestamp'] <= day5_end)
]

df_day6_price = df_price[
    (df_price['timestamp'] >= day6_start) &
    (df_price['timestamp'] <= day6_end)
]

if len(df_day5_price) > 0:
    print(f"📈 Day 5 Market Conditions (2025-10-15):")
    print(f"   Open: ${df_day5_price['open'].iloc[0]:,.2f}")
    print(f"   High: ${df_day5_price['high'].max():,.2f}")
    print(f"   Low: ${df_day5_price['low'].min():,.2f}")
    print(f"   Close: ${df_day5_price['close'].iloc[-1]:,.2f}")

    day5_range = (df_day5_price['high'].max() - df_day5_price['low'].min()) / df_day5_price['open'].iloc[0] * 100
    day5_change = (df_day5_price['close'].iloc[-1] - df_day5_price['open'].iloc[0]) / df_day5_price['open'].iloc[0] * 100

    print(f"   Daily Range: {day5_range:.2f}%")
    print(f"   Daily Change: {day5_change:+.2f}%")

    # Volatility
    df_day5_price['returns'] = df_day5_price['close'].pct_change()
    volatility_day5 = df_day5_price['returns'].std() * np.sqrt(288) * 100  # Annualized
    print(f"   Volatility (5min): {df_day5_price['returns'].std()*100:.3f}%")
    print(f"   Volatility (annualized): {volatility_day5:.1f}%")

if len(df_day6_price) > 0:
    print(f"\n📈 Day 6 Market Conditions (2025-10-16):")
    print(f"   Open: ${df_day6_price['open'].iloc[0]:,.2f}")
    print(f"   High: ${df_day6_price['high'].max():,.2f}")
    print(f"   Low: ${df_day6_price['low'].min():,.2f}")
    print(f"   Close: ${df_day6_price['close'].iloc[-1]:,.2f}")

    day6_range = (df_day6_price['high'].max() - df_day6_price['low'].min()) / df_day6_price['open'].iloc[0] * 100
    day6_change = (df_day6_price['close'].iloc[-1] - df_day6_price['open'].iloc[0]) / df_day6_price['open'].iloc[0] * 100

    print(f"   Daily Range: {day6_range:.2f}%")
    print(f"   Daily Change: {day6_change:+.2f}%")

    # Volatility
    df_day6_price['returns'] = df_day6_price['close'].pct_change()
    volatility_day6 = df_day6_price['returns'].std() * np.sqrt(288) * 100
    print(f"   Volatility (5min): {df_day6_price['returns'].std()*100:.3f}%")
    print(f"   Volatility (annualized): {volatility_day6:.1f}%")

# =============================================================================
# STEP 5: Root Cause Analysis
# =============================================================================

print("\n" + "="*80)
print("STEP 5: Root Cause Analysis")
print("="*80)
print()

# Combine Day 5-6 data
df_day5_6 = pd.concat([df_day5, df_day6]).drop_duplicates()
losers_total = df_day5_6[df_day5_6['pnl_net'] < 0]

print("🔍 Key Findings:\n")

# 1. Entry Signal Quality
print("1. Entry Signal Quality:")
if len(losers_total) > 0:
    avg_entry_prob_losers = losers_total['entry_prob'].mean()
    avg_entry_prob_winners = df_day5_6[df_day5_6['pnl_net'] > 0]['entry_prob'].mean()
    print(f"   Losers avg entry prob: {avg_entry_prob_losers:.3f}")
    print(f"   Winners avg entry prob: {avg_entry_prob_winners:.3f}")
    print(f"   → {'⚠️ Model overconfident on losing trades!' if avg_entry_prob_losers >= avg_entry_prob_winners else '✅ Model signals look good'}")

# 2. Position Sizing Impact
print(f"\n2. Position Sizing Impact:")
if len(losers_total) > 0:
    avg_size_losers = losers_total['position_size_pct'].mean() * 100
    avg_size_winners = df_day5_6[df_day5_6['pnl_net'] > 0]['position_size_pct'].mean() * 100
    print(f"   Losers avg position size: {avg_size_losers:.1f}%")
    print(f"   Winners avg position size: {avg_size_winners:.1f}%")
    print(f"   → {'⚠️ Too large positions on losing trades!' if avg_size_losers > avg_size_winners else '✅ Position sizing reasonable'}")

# 3. Exit Mechanism Performance
print(f"\n3. Exit Mechanism Performance:")
sl_trades = losers_total[losers_total['exit_reason'] == 'emergency_stop_loss']
ml_exit_trades = losers_total[losers_total['exit_reason'].str.contains('ml_exit', na=False)]
max_hold_trades = losers_total[losers_total['exit_reason'] == 'emergency_max_hold']

print(f"   Stop Loss exits: {len(sl_trades)} (${sl_trades['pnl_net'].sum():.2f})")
print(f"   ML Exit losses: {len(ml_exit_trades)} (${ml_exit_trades['pnl_net'].sum():.2f})")
print(f"   Max Hold exits: {len(max_hold_trades)} (${max_hold_trades['pnl_net'].sum():.2f})")

if len(sl_trades) > 0:
    print(f"   → ⚠️ Stop Loss triggered {len(sl_trades)} times - major damage source!")

# 4. Side Bias
print(f"\n4. Trading Direction:")
long_losers_total = losers_total[losers_total['side'] == 'LONG']
short_losers_total = losers_total[losers_total['side'] == 'SHORT']
print(f"   LONG losses: {len(long_losers_total)} (${long_losers_total['pnl_net'].sum():.2f})")
print(f"   SHORT losses: {len(short_losers_total)} (${short_losers_total['pnl_net'].sum():.2f})")

if abs(long_losers_total['pnl_net'].sum()) > abs(short_losers_total['pnl_net'].sum()) * 2:
    print(f"   → ⚠️ LONG trades performed poorly during this period!")
elif abs(short_losers_total['pnl_net'].sum()) > abs(long_losers_total['pnl_net'].sum()) * 2:
    print(f"   → ⚠️ SHORT trades performed poorly during this period!")
else:
    print(f"   → Both directions had losses")

# =============================================================================
# STEP 6: Mitigation Strategies
# =============================================================================

print("\n" + "="*80)
print("STEP 6: Mitigation Strategies")
print("="*80)
print()

print("💡 Recommended Actions:\n")

# Based on findings
if len(sl_trades) > 0:
    print("1. ✅ IMPLEMENT: Tighter Stop Loss (-3% instead of -6%)")
    print("   → Would have prevented large individual losses")
    print(f"   → Estimated savings: ~${abs(sl_trades['pnl_net'].sum()) * 0.5:.2f}")

print("\n2. 🔄 CONSIDER: Dynamic Position Sizing Based on Market Regime")
print("   → Reduce position size during high volatility")
print("   → Implement volatility-adjusted sizing")

print("\n3. 🎯 EVALUATE: Entry Signal Threshold Adjustment")
print("   → Potentially increase thresholds during uncertain markets")
print("   → Consider market regime filter")

print("\n4. ⏰ REVIEW: Max Hold Time Effectiveness")
if len(max_hold_trades) > 0:
    print(f"   → {len(max_hold_trades)} trades hit max hold with losses")
    print("   → Consider shorter hold time OR better exit signals")

print("\n5. 🛡️ ADD: Daily Loss Limit")
print("   → Implement -X% daily drawdown circuit breaker")
print("   → Pause trading if daily loss exceeds threshold")

print("\n" + "="*80)
print("ANALYSIS COMPLETE")
print("="*80)
print(f"\nGenerated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("="*80 + "\n")
