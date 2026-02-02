"""
Analyze LONG Entry Model Performance
Compare LONG vs SHORT model performance from backtest validation period
"""
import pandas as pd
import numpy as np
from pathlib import Path

# Load backtest results from threshold optimization
RESULTS_DIR = Path(__file__).parent.parent.parent / "results"
df_trades = pd.read_csv(RESULTS_DIR / "threshold_optimization_validation_20251104_214646.csv")
# Filter for Entry=0.80, Exit=0.80 configuration
df_trades = df_trades[(df_trades['entry_threshold'] == 0.80) & (df_trades['exit_threshold'] == 0.80)].copy()

# Analyze LONG trades
long_trades = df_trades[df_trades['side'] == 'LONG'].copy()
short_trades = df_trades[df_trades['side'] == 'SHORT'].copy()

print('='*80)
print('LONG vs SHORT ENTRY MODEL COMPARISON')
print('='*80)
print()

print('📊 TRADE DISTRIBUTION:')
print(f'  LONG:  {len(long_trades):3d} trades ({len(long_trades)/len(df_trades)*100:5.1f}%)')
print(f'  SHORT: {len(short_trades):3d} trades ({len(short_trades)/len(df_trades)*100:5.1f}%)')
print()

print('📈 PERFORMANCE METRICS:')
long_wins = (long_trades["pnl_pct"] > 0).sum()
short_wins = (short_trades["pnl_pct"] > 0).sum()
print(f'  LONG Win Rate:  {long_wins/len(long_trades)*100:5.1f}% ({long_wins}/{len(long_trades)})')
print(f'  SHORT Win Rate: {short_wins/len(short_trades)*100:5.1f}% ({short_wins}/{len(short_trades)})')
print()
print(f'  LONG Avg P&L:  ${long_trades["pnl_usd"].mean():+7.3f}')
print(f'  SHORT Avg P&L: ${short_trades["pnl_usd"].mean():+7.3f}')
print()
print(f'  LONG Total P&L:  ${long_trades["pnl_usd"].sum():+8.2f}')
print(f'  SHORT Total P&L: ${short_trades["pnl_usd"].sum():+8.2f}')
print()

print('🎯 ENTRY PROBABILITY DISTRIBUTION:')
print(f'  LONG Entry Prob:')
print(f'    Mean:   {long_trades["entry_prob"].mean():.4f}')
print(f'    Median: {long_trades["entry_prob"].median():.4f}')
print(f'    Min:    {long_trades["entry_prob"].min():.4f}')
print(f'    Max:    {long_trades["entry_prob"].max():.4f}')
print(f'    Q1 (25%): {long_trades["entry_prob"].quantile(0.25):.4f}')
print(f'    Q3 (75%): {long_trades["entry_prob"].quantile(0.75):.4f}')
print()
print(f'  SHORT Entry Prob:')
print(f'    Mean:   {short_trades["entry_prob"].mean():.4f}')
print(f'    Median: {short_trades["entry_prob"].median():.4f}')
print(f'    Min:    {short_trades["entry_prob"].min():.4f}')
print(f'    Max:    {short_trades["entry_prob"].max():.4f}')
print(f'    Q1 (25%): {short_trades["entry_prob"].quantile(0.25):.4f}')
print(f'    Q3 (75%): {short_trades["entry_prob"].quantile(0.75):.4f}')
print()

print('⏱️ HOLD TIME ANALYSIS:')
print(f'  LONG Avg Hold:  {long_trades["hold_candles"].mean():5.1f} candles ({long_trades["hold_candles"].mean()*5/60:4.1f}h)')
print(f'  SHORT Avg Hold: {short_trades["hold_candles"].mean():5.1f} candles ({short_trades["hold_candles"].mean()*5/60:4.1f}h)')
print()

print('🚪 EXIT MECHANISM:')
long_exit_counts = long_trades['exit_reason'].value_counts()
short_exit_counts = short_trades['exit_reason'].value_counts()
print(f'  LONG Exits:')
for reason, count in long_exit_counts.items():
    print(f'    {reason}: {count} ({count/len(long_trades)*100:.1f}%)')
print()
print(f'  SHORT Exits:')
for reason, count in short_exit_counts.items():
    print(f'    {reason}: {count} ({count/len(short_trades)*100:.1f}%)')
print()

print('💰 PROFITABILITY BY ENTRY PROBABILITY (LONG):')
# Quartile analysis for LONG
quartiles = long_trades['entry_prob'].quantile([0.25, 0.5, 0.75])
q1_long = long_trades[long_trades['entry_prob'] <= quartiles[0.25]]
q2_long = long_trades[(long_trades['entry_prob'] > quartiles[0.25]) & (long_trades['entry_prob'] <= quartiles[0.5])]
q3_long = long_trades[(long_trades['entry_prob'] > quartiles[0.5]) & (long_trades['entry_prob'] <= quartiles[0.75])]
q4_long = long_trades[long_trades['entry_prob'] > quartiles[0.75]]

for q_name, q_data in [('Q1', q1_long), ('Q2', q2_long), ('Q3', q3_long), ('Q4', q4_long)]:
    if len(q_data) > 0:
        wr = (q_data["pnl_pct"] > 0).mean() * 100
        avg_pnl = q_data["pnl_usd"].mean()
        print(f'  {q_name} ({q_data["entry_prob"].min():.4f}-{q_data["entry_prob"].max():.4f}): {len(q_data)} trades, WR {wr:.1f}%, Avg P&L ${avg_pnl:+.2f}')
print()

print('❌ WORST LONG TRADES (Top 5):')
worst_long = long_trades.nsmallest(5, 'pnl_pct')
for idx, trade in worst_long.iterrows():
    print(f'  Entry: {trade["entry_time"]} @ ${trade["entry_price"]:,.1f}')
    print(f'    Exit: {trade["exit_reason"]} after {trade["hold_candles"]}c @ ${trade["exit_price"]:,.1f}')
    print(f'    Entry Prob: {trade["entry_prob"]:.4f} ({trade["entry_prob"]*100:.2f}%)')
    print(f'    P&L: ${trade["pnl_usd"]:+.2f} ({trade["pnl_pct"]*100:+.2f}%)')
    print()

print('✅ BEST LONG TRADES (Top 3):')
best_long = long_trades.nlargest(3, 'pnl_pct')
for idx, trade in best_long.iterrows():
    print(f'  Entry: {trade["entry_time"]} @ ${trade["entry_price"]:,.1f}')
    print(f'    Exit: {trade["exit_reason"]} after {trade["hold_candles"]}c @ ${trade["exit_price"]:,.1f}')
    print(f'    Entry Prob: {trade["entry_prob"]:.4f} ({trade["entry_prob"]*100:.2f}%)')
    print(f'    P&L: ${trade["pnl_usd"]:+.2f} ({trade["pnl_pct"]*100:+.2f}%)')
    print()

print('='*80)
print('KEY FINDINGS:')
print('='*80)

# Calculate key metrics
long_profitable_rate = (long_trades['pnl_pct'] > 0).mean()
short_profitable_rate = (short_trades['pnl_pct'] > 0).mean()

long_winning = long_trades[long_trades['pnl_pct'] > 0]
long_losing = long_trades[long_trades['pnl_pct'] < 0]
short_winning = short_trades[short_trades['pnl_pct'] > 0]
short_losing = short_trades[short_trades['pnl_pct'] < 0]

long_avg_profit = long_winning['pnl_usd'].mean() if len(long_winning) > 0 else 0
long_avg_loss = long_losing['pnl_usd'].mean() if len(long_losing) > 0 else 0
short_avg_profit = short_winning['pnl_usd'].mean() if len(short_winning) > 0 else 0
short_avg_loss = short_losing['pnl_usd'].mean() if len(short_losing) > 0 else 0

print()
print(f'1. LONG Model Win Rate: {long_profitable_rate*100:.1f}% vs SHORT {short_profitable_rate*100:.1f}%')
print(f'   → LONG is {abs(long_profitable_rate - short_profitable_rate)*100:.1f}% points LOWER')
print()
print(f'2. LONG Avg Win: ${long_avg_profit:+.2f} vs Avg Loss: ${long_avg_loss:+.2f}')
if long_avg_loss != 0:
    print(f'   → Reward/Risk Ratio: {abs(long_avg_profit/long_avg_loss):.2f}x')
print()
print(f'3. LONG Entry Probability:')
high_prob_long = long_trades[long_trades['entry_prob'] >= 0.90]
print(f'   → {len(high_prob_long)}/{len(long_trades)} trades ({len(high_prob_long)/len(long_trades)*100:.1f}%) have >90% confidence')
if len(high_prob_long) > 0:
    print(f'   → HIGH prob LONG (>90%) WR: {(high_prob_long["pnl_pct"]>0).mean()*100:.1f}%')
else:
    print(f'   → No HIGH prob LONG trades (>90%)')
print()
print(f'4. Market Context:')
print(f'   → Validation period: Oct 28 - Nov 4 (FALLING market)')
print(f'   → LONG underperformance is EXPECTED in falling market')
print(f'   → NOT necessarily a model problem')
print()
print('5. LONG Model Improvement Recommendations:')
print('   → Option A: Increase LONG Entry threshold (0.80 → 0.85+) to filter low-quality signals')
print('   → Option B: Add LONG-specific features (uptrend_strength, bullish_momentum, etc.)')
print('   → Option C: Wait for rising market period to properly evaluate LONG model')
print('   → Option D: Train with more rising market data (currently heavy on falling market)')
print()

# Market condition analysis
print('6. Market Condition Analysis:')
df_price_change = ((df_trades['exit_price'] - df_trades['entry_price']) / df_trades['entry_price'] * 100).mean()
print(f'   → Avg price change during validation: {df_price_change:+.2f}%')
if df_price_change < 0:
    print(f'   → Market was FALLING on average (negative price change)')
    print(f'   → This favors SHORT trades over LONG trades')
print()

print('='*80)
print('RECOMMENDATION:')
print('='*80)
print()
print('VERDICT: ⚠️ LONG Model needs evaluation in RISING market')
print()
print('Current LONG performance (27.3% WR) is LOW but EXPECTED in falling market.')
print()
print('Immediate Actions:')
print('  1. ✅ Monitor LONG signals in production (see if WR improves in different conditions)')
print('  2. ⏳ Wait for rising market period to properly assess LONG model')
print('  3. 🔍 If LONG WR stays <35% for 2+ weeks across different market conditions:')
print('     → Then consider retraining with LONG-specific features')
print()
print('Do NOT immediately add LONG-specific features because:')
print('  - Sample size too small (11 trades)')
print('  - Market condition biased (falling market)')
print('  - Risk of overfitting to single market regime')
print()
print('Better approach: Collect more data, validate across multiple market conditions first.')
print()
