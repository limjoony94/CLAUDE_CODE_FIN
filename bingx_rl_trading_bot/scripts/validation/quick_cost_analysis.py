"""
Quick Transaction Cost Sensitivity Analysis
기존 백테스트 결과를 재활용하여 빠르게 비용 민감도 분석
"""

import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent
RESULTS_DIR = PROJECT_ROOT / "results"

# Load backtest results
trades_file = RESULTS_DIR / "backtest_90_10_trades.csv"
df = pd.read_csv(trades_file)

print("="*100)
print("TRANSACTION COST SENSITIVITY ANALYSIS")
print("="*100)

# Calculate average position size
long_trades = df[df['strategy'] == 'LONG']
short_trades = df[df['strategy'] == 'SHORT']

avg_long_price = long_trades['entry_price'].mean()
avg_short_price = short_trades['entry_price'].mean()

# Estimate average position values
INITIAL_CAPITAL = 10000
LONG_CAPITAL = INITIAL_CAPITAL * 0.90
SHORT_CAPITAL = INITIAL_CAPITAL * 0.10

avg_long_position_value = LONG_CAPITAL * 0.95  # 95% position size
avg_short_position_value = SHORT_CAPITAL * 0.95

print(f"\n📊 Backtest Summary:")
print(f"   Total Trades: {len(df)}")
print(f"   LONG Trades: {len(long_trades)}")
print(f"   SHORT Trades: {len(short_trades)}")
print(f"   Average LONG position: ${avg_long_position_value:,.2f}")
print(f"   Average SHORT position: ${avg_short_position_value:,.2f}")

# Cost scenarios
scenarios = [
    ('Optimistic', 0.0001, "Best case (maker only, no slippage)"),
    ('Current Assumption', 0.0002, "Current backtest assumption"),
    ('Maker Fee', 0.0004, "BingX maker fee (0.04%)"),
    ('Taker Fee', 0.0005, "BingX taker fee (0.05%)"),
    ('Low Slippage', 0.0008, "Taker + minor slippage"),
    ('Medium Slippage', 0.0010, "Taker + moderate slippage"),
    ('High Slippage', 0.0015, "Taker + high slippage"),
    ('Worst Case', 0.0020, "Extreme slippage scenario"),
]

results = []

for scenario_name, cost_pct, description in scenarios:
    # Calculate costs for each trade
    costs = []

    for _, trade in df.iterrows():
        entry_price = trade['entry_price']
        exit_price = trade['exit_price']
        strategy = trade['strategy']

        if strategy == 'LONG':
            position_value = avg_long_position_value
        else:
            position_value = avg_short_position_value

        quantity = position_value / entry_price

        # Transaction costs (entry + exit)
        entry_cost = entry_price * quantity * cost_pct
        exit_cost = exit_price * quantity * cost_pct
        total_cost = entry_cost + exit_cost

        costs.append(total_cost)

    df['cost'] = costs

    # Calculate net P&L
    long_net_pnl = (long_trades['pnl_usd'] - long_trades.index.map(lambda i: df.loc[i, 'cost'])).sum()
    short_net_pnl = (short_trades['pnl_usd'] - short_trades.index.map(lambda i: df.loc[i, 'cost'])).sum()

    total_net_pnl = long_net_pnl + short_net_pnl
    total_costs = df['cost'].sum()

    # Calculate returns
    total_capital = INITIAL_CAPITAL + total_net_pnl
    total_return_pct = (total_net_pnl / INITIAL_CAPITAL) * 100

    # Estimate monthly return (59.8 days)
    days = 59.8
    monthly_return_pct = (total_return_pct / days) * 30.42

    results.append({
        'scenario': scenario_name,
        'cost_pct': cost_pct * 100,
        'description': description,
        'total_costs': total_costs,
        'total_net_pnl': total_net_pnl,
        'total_return_pct': total_return_pct,
        'monthly_return_pct': monthly_return_pct,
        'final_capital': total_capital
    })

results_df = pd.DataFrame(results)

# Print results
print(f"\n{'='*100}")
print(f"{'Scenario':<25} {'Cost%':<10} {'Monthly':<12} {'Total Costs':<15} {'Final Capital':<18}")
print(f"{'-'*100}")

baseline = results_df[results_df['scenario'] == 'Current Assumption'].iloc[0]

for _, row in results_df.iterrows():
    scenario = row['scenario']
    cost = row['cost_pct']
    monthly = row['monthly_return_pct']
    costs = row['total_costs']
    capital = row['final_capital']

    deg = monthly - baseline['monthly_return_pct']

    if monthly >= 15:
        status = "✅"
    elif monthly >= 10:
        status = "⚠️"
    else:
        status = "🚨"

    print(f"{scenario:<25} {cost:>6.3f}%   {monthly:>+7.2f}% ({deg:+6.2f}pp)  ${costs:>9.2f}      ${capital:>12.2f}    {status}")

print(f"{'='*100}")

# Analysis
print(f"\n📊 CRITICAL FINDINGS:\n")

taker = results_df[results_df['scenario'] == 'Taker Fee'].iloc[0]
degradation = taker['monthly_return_pct'] - baseline['monthly_return_pct']

print(f"1. Taker Fee Impact:")
print(f"   Current (0.02%): {baseline['monthly_return_pct']:+.2f}% monthly")
print(f"   Taker (0.05%): {taker['monthly_return_pct']:+.2f}% monthly")
print(f"   Degradation: {degradation:.2f}pp ({(degradation/baseline['monthly_return_pct'])*100:+.1f}%)")

if degradation < -5:
    print(f"   🚨 CRITICAL: Taker fees cause significant degradation!")
    print(f"   🚨 RECOMMENDATION: Use maker orders whenever possible")

high_slip = results_df[results_df['scenario'] == 'High Slippage'].iloc[0]
print(f"\n2. Slippage Impact:")
print(f"   High Slippage (0.15%): {high_slip['monthly_return_pct']:+.2f}% monthly")
print(f"   vs Baseline: {high_slip['monthly_return_pct'] - baseline['monthly_return_pct']:.2f}pp")

if high_slip['monthly_return_pct'] < 15:
    print(f"   ⚠️ WARNING: High slippage reduces returns below target!")
    print(f"   ⚠️ MONITOR: Actual slippage closely in testnet")

# Break-even
print(f"\n3. Break-even Analysis:")
gross_pnl = df['pnl_usd'].sum()
total_trades = len(df)

# Break-even cost = gross PnL / (2 * total trades * avg position value)
avg_total_position = (len(long_trades) * avg_long_position_value + len(short_trades) * avg_short_position_value) / total_trades
breakeven_cost = gross_pnl / (2 * total_trades * avg_total_position)

print(f"   Gross P&L: ${gross_pnl:,.2f}")
print(f"   Break-even Cost: {breakeven_cost*100:.3f}%")
print(f"   Safety Margin: {(breakeven_cost / 0.0002):.1f}× current assumption")

if breakeven_cost < 0.001:  # 0.1%
    print(f"   ⚠️ WARNING: Break-even cost is relatively low!")
    print(f"   ⚠️ Strategy is sensitive to transaction costs")

# Recommendations
print(f"\n🎯 RECOMMENDATIONS:\n")

print(f"1. Target Order Type:")
if degradation < -3:
    print(f"   ✅ Use MAKER orders whenever possible (limit orders)")
    print(f"   ⚠️ Avoid TAKER orders (market orders) - significant cost impact")
else:
    print(f"   ✅ TAKER orders acceptable (impact manageable)")

print(f"\n2. Expected Real-World Performance:")
realistic_cost = results_df[results_df['scenario'] == 'Low Slippage'].iloc[0]
print(f"   Realistic scenario (0.08%): {realistic_cost['monthly_return_pct']:+.2f}% monthly")
print(f"   Conservative estimate: {realistic_cost['monthly_return_pct'] * 0.8:+.2f}% monthly (80% of realistic)")

print(f"\n3. Risk Management:")
if breakeven_cost < 0.0015:
    print(f"   ⚠️ Monitor actual costs closely in testnet")
    print(f"   ⚠️ If costs > 0.10%, consider reducing trade frequency")
    print(f"   ⚠️ Strategy profitability sensitive to execution quality")

print(f"\n{'='*100}")

# Save results
output_file = RESULTS_DIR / "cost_sensitivity_quick_analysis.csv"
results_df.to_csv(output_file, index=False)
print(f"\n✅ Results saved: {output_file}")
print(f"{'='*100}")
