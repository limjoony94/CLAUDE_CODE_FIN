"""
SHORT Trading Feasibility Analysis

Tests:
1. Regime distribution (Bull/Bear/Sideways)
2. TP hit rates at different targets (0.5% to 3.0%)
3. TP hits by regime
4. Decision: Is SHORT trading viable?
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.production.train_xgboost_improved_v3_phase2 import calculate_features
from scripts.production.market_regime_filter import MarketRegimeFilter

DATA_DIR = PROJECT_ROOT / "data" / "historical"


def analyze_tp_hits(df, tp_pct, sl_pct=0.01, max_hold_candles=48, sample_size=5000):
    """Count how many trades hit TP at given level (sampled for speed)"""
    tp_hits = 0
    sl_hits = 0
    max_hold_hits = 0

    # Sample every Nth candle for speed
    total_candles = len(df) - max_hold_candles
    if total_candles > sample_size:
        step = total_candles // sample_size
    else:
        step = 1

    sampled_count = 0

    for i in range(0, len(df) - max_hold_candles, step):
        entry_price = df['close'].iloc[i]
        tp_price = entry_price * (1 - tp_pct)
        sl_price = entry_price * (1 + sl_pct)

        tp_hit = False
        sl_hit = False

        for j in range(1, max_hold_candles + 1):
            if i + j >= len(df):
                break

            low = df['low'].iloc[i + j]
            high = df['high'].iloc[i + j]

            if low <= tp_price:
                tp_hit = True
                break

            if high >= sl_price:
                sl_hit = True
                break

        if tp_hit:
            tp_hits += 1
        elif sl_hit:
            sl_hits += 1
        else:
            max_hold_hits += 1

        sampled_count += 1

    # Scale up to full dataset
    scale_factor = total_candles / sampled_count if sampled_count > 0 else 1
    tp_hits = int(tp_hits * scale_factor)
    sl_hits = int(sl_hits * scale_factor)
    max_hold_hits = int(max_hold_hits * scale_factor)

    return tp_hits, sl_hits, max_hold_hits


def main():
    print("="*80)
    print("SHORT TRADING FEASIBILITY ANALYSIS")
    print("="*80)

    # Load data
    data_file = DATA_DIR / "BTCUSDT_5m_max.csv"
    df = pd.read_csv(data_file)
    print(f"\nLoaded {len(df):,} candles")

    # Calculate features for regime
    df = calculate_features(df)
    regime_filter = MarketRegimeFilter()
    df = regime_filter.get_regime_features(df)
    df = df.ffill().dropna()

    print(f"After processing: {len(df):,} candles")

    # 1. Regime Distribution
    print("\n" + "="*80)
    print("1. MARKET REGIME DISTRIBUTION")
    print("="*80)

    regime_counts = df['regime_trend'].value_counts()
    print("\nRegime Distribution:")
    for regime, count in regime_counts.items():
        pct = count / len(df) * 100
        print(f"  {regime:12s}: {count:,} ({pct:.1f}%)")

    # 2. TP Sensitivity Analysis
    print("\n" + "="*80)
    print("2. TP HIT RATE SENSITIVITY (SL=1%, Max Hold=4h)")
    print("="*80)

    tp_targets = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
    results = []

    for tp_pct in tp_targets:
        tp_hits, sl_hits, max_hold = analyze_tp_hits(df, tp_pct / 100)
        total = len(df) - 48
        tp_rate = tp_hits / total * 100
        sl_rate = sl_hits / total * 100
        max_hold_rate = max_hold / total * 100
        win_rate = tp_hits / (tp_hits + sl_hits + max_hold) * 100 if (tp_hits + sl_hits + max_hold) > 0 else 0

        results.append({
            'tp_pct': tp_pct,
            'tp_hits': tp_hits,
            'sl_hits': sl_hits,
            'max_hold': max_hold,
            'tp_rate': tp_rate,
            'win_rate': win_rate
        })

        print(f"\nTP Target: {tp_pct:.1f}%")
        print(f"  TP Hits: {tp_hits:,} ({tp_rate:.2f}%)")
        print(f"  SL Hits: {sl_hits:,} ({sl_rate:.2f}%)")
        print(f"  Max Hold: {max_hold:,} ({max_hold_rate:.2f}%)")
        print(f"  Theoretical Win Rate: {win_rate:.2f}%")

    # 3. Recommended TP Target
    print("\n" + "="*80)
    print("3. RECOMMENDED TP TARGET")
    print("="*80)

    results_df = pd.DataFrame(results)

    # Find TP with >2% hit rate and best win rate
    viable = results_df[results_df['tp_rate'] >= 2.0]

    if len(viable) > 0:
        best = viable.loc[viable['win_rate'].idxmax()]
        print(f"\n✅ VIABLE TP FOUND:")
        print(f"  TP Target: {best['tp_pct']:.1f}%")
        print(f"  TP Hit Rate: {best['tp_rate']:.2f}%")
        print(f"  Theoretical Win Rate: {best['win_rate']:.2f}%")
        print(f"  TP Hits: {best['tp_hits']:.0f}")

        if best['win_rate'] >= 50:
            print(f"\n🎯 RECOMMENDATION: USE TP={best['tp_pct']:.1f}% for training")
            print(f"   Expected to achieve >50% win rate")
        elif best['win_rate'] >= 30:
            print(f"\n⚠️ RECOMMENDATION: USE TP={best['tp_pct']:.1f}% with caution")
            print(f"   Win rate {best['win_rate']:.1f}% may be marginally profitable")
        else:
            print(f"\n❌ RECOMMENDATION: Even relaxed TP shows <30% win rate")
            print(f"   SHORT trading NOT recommended for this market")
    else:
        print(f"\n❌ NO VIABLE TP FOUND")
        print(f"   All TP targets show <2% hit rate")
        print(f"   SHORT training NOT feasible")

    # 4. TP Hits by Regime (for current 3% TP)
    print("\n" + "="*80)
    print("4. TP HITS BY REGIME (TP=3%, for reference)")
    print("="*80)

    # Find TP hits for 3% target
    tp_indices = []
    for i in range(len(df) - 48):
        entry_price = df['close'].iloc[i]
        tp_price = entry_price * 0.97

        for j in range(1, 49):
            if i + j >= len(df):
                break
            if df['close'].iloc[i + j] <= tp_price:
                tp_indices.append(i)
                break

    if len(tp_indices) > 0:
        tp_regimes = df.iloc[tp_indices]['regime_trend'].value_counts()
        print(f"\nTotal TP Hits (3%): {len(tp_indices)}")
        for regime, count in tp_regimes.items():
            pct = count / len(tp_indices) * 100
            print(f"  {regime:12s}: {count} ({pct:.1f}%)")
    else:
        print("\nNo TP hits found at 3% level")

    # 5. Final Recommendation
    print("\n" + "="*80)
    print("5. FINAL RECOMMENDATION")
    print("="*80)

    bear_pct = regime_counts.get('Bear', 0) / len(df) * 100 if 'Bear' in regime_counts else 0

    # Get best viable TP
    viable = results_df[results_df['tp_rate'] >= 2.0]

    print(f"\nMarket Characteristics:")
    print(f"  Bear Market %: {bear_pct:.1f}%")
    print(f"  Viable TP Targets: {len(viable)}")

    if bear_pct < 10:
        print(f"\n❌ INSUFFICIENT BEAR DATA (<10%)")
        print(f"   Market is primarily Bull/Sideways")
        print(f"   SHORT opportunities are structurally limited")

    if len(viable) == 0:
        print(f"\n❌ NO VIABLE TP TARGETS (all <2% hit rate)")
        print(f"   Cannot generate sufficient training examples")

    if bear_pct < 10 or len(viable) == 0:
        print(f"\n" + "="*80)
        print(f"🚫 RECOMMENDATION: ABANDON SHORT STRATEGY")
        print(f"="*80)
        print(f"\nReasons:")
        print(f"  1. Insufficient Bear market data (<10%)")
        print(f"  2. No viable TP targets with >2% hit rate")
        print(f"  3. Market structural bias against SHORT")
        print(f"\nSuggestion:")
        print(f"  → Switch to LONG-only strategy (70.2% proven win rate)")
        print(f"  → Revisit SHORT when:")
        print(f"     - Market enters Bear regime (>20% Bear data)")
        print(f"     - Collect 6-12 months Bear market data")
        print(f"     - Redesign with lower TP targets")
    else:
        best = viable.loc[viable['win_rate'].idxmax()]
        print(f"\n" + "="*80)
        print(f"✅ RECOMMENDATION: PROCEED WITH MODIFIED TP")
        print(f"="*80)
        print(f"\nSuggested Configuration:")
        print(f"  TP Target: {best['tp_pct']:.1f}%")
        print(f"  SL: 1.0%")
        print(f"  Max Hold: 4 hours")
        print(f"  Expected Win Rate: {best['win_rate']:.1f}%")
        print(f"\nNext Steps:")
        print(f"  1. Retrain with TP={best['tp_pct']:.1f}%")
        print(f"  2. Add regime filtering (Bear/Sideways only)")
        print(f"  3. Backtest and validate")


if __name__ == "__main__":
    main()
