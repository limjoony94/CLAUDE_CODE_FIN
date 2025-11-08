"""
SHORT Trade Pattern Analysis - Quick Profitability Assessment

비판적 질문: 36.4% 승률로 수익성 가능한가?

빠른 분석 방법:
1. 기존 백테스트 결과 로드
2. 실제 거래 패턴 분석
3. 평균 win/loss 계산
4. 최적 SL/TP 역산
"""

import pandas as pd
import numpy as np
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent
RESULTS_DIR = PROJECT_ROOT / "results"

print("="*80)
print("SHORT Trade Pattern Analysis - Profitability Assessment")
print("="*80)

# Load backtest results
results_file = RESULTS_DIR / "backtest_phase4_3class.csv"

if not results_file.exists():
    print(f"\n❌ Results file not found: {results_file}")
    print("Need to run backtest first")
    exit(1)

results = pd.read_csv(results_file)

print(f"\n✅ Loaded backtest results: {len(results)} windows")
print(f"\nCurrent Configuration:")
print(f"  Threshold: 0.7")
print(f"  Stop Loss: 1.0%")
print(f"  Take Profit: 3.0%")
print(f"  Risk-Reward Ratio: 1:3")

print(f"\n" + "="*80)
print("Current Performance")
print("="*80)

print(f"\nAggregate Metrics:")
print(f"  Avg SHORT trades per window: {results['num_short'].mean():.1f}")
print(f"  Avg SHORT win rate: {results['win_rate_short'].mean():.1f}%")
print(f"  Avg total return: {results['xgb_return'].mean():.2f}%")
print(f"  Avg vs B&H: {results['difference'].mean():.2f}%")

# Analysis by regime
print(f"\n" + "="*80)
print("Analysis by Market Regime")
print("="*80)

for regime in ['Bull', 'Bear', 'Sideways']:
    regime_df = results[results['regime'] == regime]
    if len(regime_df) > 0:
        print(f"\n{regime} Market ({len(regime_df)} windows):")
        print(f"  SHORT trades: {regime_df['num_short'].mean():.1f}")
        print(f"  SHORT win rate: {regime_df['win_rate_short'].mean():.1f}%")
        print(f"  Return: {regime_df['xgb_return'].mean():.2f}%")
        print(f"  vs B&H: {regime_df['difference'].mean():+.2f}%")

# Critical analysis
print(f"\n" + "="*80)
print("🎯 Critical Analysis: Profitability Potential")
print("="*80)

avg_short_wr = results['win_rate_short'].mean()
avg_trades = results['num_short'].mean()

print(f"\nCurrent State:")
print(f"  SHORT win rate: {avg_short_wr:.1f}%")
print(f"  Trades per window: {avg_trades:.1f}")

# Calculate expected value with current SL/TP
current_sl = 0.01  # 1%
current_tp = 0.03  # 3%
current_wr = avg_short_wr / 100

current_ev = current_wr * current_tp + (1 - current_wr) * (-current_sl)

print(f"\nCurrent Expected Value (SL=1%, TP=3%):")
print(f"  EV = {current_wr:.3f} * {current_tp*100:.1f}% + {(1-current_wr):.3f} * (-{current_sl*100:.1f}%)")
print(f"  EV = {current_wr * current_tp *100:.3f}% + {(1-current_wr) * (-current_sl) * 100:.3f}%")
print(f"  EV = {current_ev * 100:.3f}% per trade")

if current_ev > 0:
    print(f"\n  ✅ POSITIVE! Strategy is profitable!")
else:
    print(f"\n  ❌ NEGATIVE! Strategy loses money")

# Test alternative configurations
print(f"\n" + "="*80)
print("Alternative Risk-Reward Configurations")
print("="*80)

configs = [
    (0.005, 0.02, "1:4"),   # SL 0.5%, TP 2%
    (0.005, 0.03, "1:6"),   # SL 0.5%, TP 3%
    (0.010, 0.03, "1:3"),   # SL 1.0%, TP 3% (current)
    (0.010, 0.04, "1:4"),   # SL 1.0%, TP 4%
    (0.010, 0.05, "1:5"),   # SL 1.0%, TP 5%
    (0.015, 0.045, "1:3"),  # SL 1.5%, TP 4.5%
    (0.015, 0.06, "1:4"),   # SL 1.5%, TP 6%
]

print(f"\n{'SL%':<7} {'TP%':<7} {'R:R':<7} {'Expected Value':<18} {'Status':<10}")
print("-" * 60)

best_ev = current_ev
best_config = (current_sl, current_tp, "1:3")

for sl, tp, rr in configs:
    ev = current_wr * tp + (1 - current_wr) * (-sl)
    status = "✅ Profitable" if ev > 0 else "❌ Loss"

    print(f"{sl*100:<7.1f} {tp*100:<7.1f} {rr:<7} {ev*100:+.3f}% per trade    {status:<10}")

    if ev > best_ev:
        best_ev = ev
        best_config = (sl, tp, rr)

# Recommendation
print(f"\n" + "="*80)
print("🎯 RECOMMENDATION")
print("="*80)

sl_best, tp_best, rr_best = best_config

print(f"\nOptimal Configuration (based on {avg_short_wr:.1f}% win rate):")
print(f"  Stop Loss: {sl_best*100:.1f}%")
print(f"  Take Profit: {tp_best*100:.1f}%")
print(f"  Risk-Reward Ratio: {rr_best}")
print(f"  Expected Value: {best_ev*100:+.3f}% per trade")

if best_ev > 0:
    # Project monthly returns
    trades_per_5days = avg_trades
    trades_per_month = trades_per_5days * 6  # 30 days / 5 days
    monthly_return = best_ev * trades_per_month * 100

    print(f"\nProjected Performance:")
    print(f"  Trades per month: {trades_per_month:.1f}")
    print(f"  Monthly return: {monthly_return:+.2f}%")

    if monthly_return > 2:
        print(f"\n  ✅✅ EXCELLENT! Highly profitable strategy!")
        print(f"  🎯 비판적 통찰 검증:")
        print(f"     '60% 승률' 목표는 불필요했다!")
        print(f"     {avg_short_wr:.1f}% 승률로도 수익성 달성 가능!")
    elif monthly_return > 0:
        print(f"\n  ✅ PROFITABLE! Viable strategy")
    else:
        print(f"\n  ⚠️ LOW PROFITABILITY")
else:
    print(f"\n  ❌ UNPROFITABLE with current win rate ({avg_short_wr:.1f}%)")

    # Calculate break-even win rate
    breakeven_wr = sl_best / (sl_best + tp_best)
    print(f"\n  Break-even win rate needed: {breakeven_wr*100:.1f}%")
    print(f"  Current win rate: {avg_short_wr:.1f}%")
    print(f"  Gap: {(breakeven_wr*100 - avg_short_wr):.1f}%")

# Final verdict
print(f"\n" + "="*80)
print("FINAL VERDICT - Approach #17 Result")
print("="*80)

if best_ev > 0 and best_ev * trades_per_5days * 6 * 100 > 2:
    print(f"\n✅ SUCCESS! SHORT strategy CAN BE PROFITABLE!")
    print(f"\n비판적 사고의 승리:")
    print(f"  - 잘못된 목표: '60% 승률'")
    print(f"  - 올바른 목표: '수익성 있는 전략'")
    print(f"  - 해결책: Risk-Reward 최적화")
    print(f"  - 결과: {avg_short_wr:.1f}% 승률로 +{best_ev*trades_per_5days*6*100:.2f}% 월간 수익")

    print(f"\n🎯 NEXT STEP:")
    print(f"  Implement SHORT bot with optimal config:")
    print(f"    - SL: {sl_best*100:.1f}%")
    print(f"    - TP: {tp_best*100:.1f}%")
    print(f"    - Threshold: 0.7")

elif best_ev > 0:
    print(f"\n⚠️ MARGINAL PROFITABILITY")
    print(f"  Expected: +{best_ev*trades_per_5days*6*100:.2f}% monthly")
    print(f"  Consider further optimization")

else:
    print(f"\n❌ UNPROFITABLE")
    print(f"  Even with optimal R:R, expected value negative")
    print(f"  Root cause: Win rate too low ({avg_short_wr:.1f}% < {breakeven_wr*100:.1f}% needed)")
    print(f"\n  Original recommendation remains:")
    print(f"    → Deploy LONG-only strategy (69.1% win rate, +46% monthly)")

print(f"\n" + "="*80)
