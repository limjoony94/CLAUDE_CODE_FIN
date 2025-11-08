"""
SHORT Model - Rule-Based Expert System (Approach #15)

Final Critical Insight:
- 14 ML approaches all failed (best: 46%, most: 10-27%)
- Why? ML tries to learn from noisy 5-min data
- Problem: SHORT patterns too random for ML

New Paradigm: Rule-Based System
- Don't learn from data
- Code expert technical analysis knowledge directly
- Use proven trading rules

Rules for SHORT Entry:
1. OVERBOUGHT: RSI > 70 AND price near BB upper
2. RESISTANCE: Price rejected at recent high
3. BEARISH DIVERGENCE: Price up, RSI/MACD down
4. DISTRIBUTION: High selling volume + price weakness
5. MOMENTUM EXHAUSTION: Strong rally → reversal signals

Expected: 50-60% win rate (expert rules beat ML on noisy data)
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from sklearn.model_selection import TimeSeriesSplit

import sys
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.production.train_xgboost_improved_v3_phase2 import calculate_features
from scripts.production.short_specific_features import ShortSpecificFeatures
from scripts.production.market_regime_filter import MarketRegimeFilter

# Directories
DATA_DIR = PROJECT_ROOT / "data" / "historical"
MODELS_DIR = PROJECT_ROOT / "models"


def create_short_labels(df, lookahead=3, threshold=0.002):
    """Create SHORT validation labels"""
    labels = []
    for i in range(len(df)):
        if i >= len(df) - lookahead:
            labels.append(0)
            continue

        current_price = df['close'].iloc[i]
        future_prices = df['close'].iloc[i+1:i+1+lookahead]
        min_future = future_prices.min()
        decrease_pct = (current_price - min_future) / current_price

        if decrease_pct >= threshold:
            labels.append(1)
        else:
            labels.append(0)

    return np.array(labels)


def apply_short_rules(df):
    """
    Apply rule-based SHORT entry logic

    Rules (ALL must be satisfied for SHORT):
    1. Overbought: RSI > 65
    2. Resistance: Price > 98% of BB upper
    3. Bearish signal: Bearish divergence OR bearish pattern
    4. Regime: Short allowed (not strong uptrend)
    5. Volume confirmation: Volume above average

    Score system: Each rule = points, enter if score >= threshold
    """
    print("\n" + "="*80)
    print("Applying Rule-Based SHORT Logic")
    print("="*80)

    # Rule 1: Overbought (RSI > 65)
    rule_overbought = (df['rsi'] > 65).astype(int)

    # Rule 2: At resistance (price > 98% of BB upper)
    rule_resistance = (df['close'] >= df['bb_upper'] * 0.98).astype(int)

    # Rule 3: Bearish signals
    bearish_signals = 0
    if 'bearish_div_rsi' in df.columns:
        bearish_signals += df['bearish_div_rsi']
    if 'bearish_div_macd' in df.columns:
        bearish_signals += df['bearish_div_macd']
    if 'shooting_star' in df.columns:
        bearish_signals += df['shooting_star']
    if 'bearish_engulfing' in df.columns:
        bearish_signals += df['bearish_engulfing']

    rule_bearish = (bearish_signals >= 1).astype(int)

    # Rule 4: Regime filter
    rule_regime = df['short_allowed'].astype(int)

    # Rule 5: Volume confirmation (above average)
    rule_volume = (df['volume_ratio'] > 1.0).astype(int)

    # Rule 6: Momentum exhaustion (RSI declining)
    df['rsi_change'] = df['rsi'].diff()
    rule_momentum_exhaustion = (df['rsi_change'] < -2).astype(int)

    # Scoring system
    df['short_score'] = (
        rule_overbought * 2 +       # Weight: 2 (critical)
        rule_resistance * 2 +        # Weight: 2 (critical)
        rule_bearish * 2 +           # Weight: 2 (important)
        rule_regime * 1 +            # Weight: 1 (filter)
        rule_volume * 1 +            # Weight: 1 (confirmation)
        rule_momentum_exhaustion * 1  # Weight: 1 (confirmation)
    )

    # Entry threshold: score >= 5 (out of 9 possible)
    df['short_signal'] = (df['short_score'] >= 5).astype(int)

    print(f"\nRule Statistics:")
    print(f"  Overbought (RSI>65): {rule_overbought.sum()} / {len(df)} ({rule_overbought.sum()/len(df)*100:.1f}%)")
    print(f"  At Resistance: {rule_resistance.sum()} / {len(df)} ({rule_resistance.sum()/len(df)*100:.1f}%)")
    print(f"  Bearish Signals: {rule_bearish.sum()} / {len(df)} ({rule_bearish.sum()/len(df)*100:.1f}%)")
    print(f"  Regime Allowed: {rule_regime.sum()} / {len(df)} ({rule_regime.sum()/len(df)*100:.1f}%)")
    print(f"  Volume Confirmation: {rule_volume.sum()} / {len(df)} ({rule_volume.sum()/len(df)*100:.1f}%)")
    print(f"  Momentum Exhaustion: {rule_momentum_exhaustion.sum()} / {len(df)} ({rule_momentum_exhaustion.sum()/len(df)*100:.1f}%)")
    print(f"\n  SHORT Signals (score>=5): {df['short_signal'].sum()} / {len(df)} ({df['short_signal'].sum()/len(df)*100:.1f}%)")

    return df


def backtest_rules(df, y_short):
    """Backtest rule-based SHORT system"""
    print("\n" + "="*80)
    print("Backtesting Rule-Based SHORT System")
    print("="*80)

    tscv = TimeSeriesSplit(n_splits=5)

    all_trades = []

    for fold, (train_idx, val_idx) in enumerate(tscv.split(df)):
        df_val = df.iloc[val_idx]
        y_val = y_short[val_idx]

        for i, idx in enumerate(val_idx):
            if df['short_signal'].iloc[idx] == 1:
                actual = y_val[i]

                trade = {
                    'fold': fold,
                    'index': idx,
                    'score': df['short_score'].iloc[idx],
                    'predicted': 1,
                    'actual': actual,
                    'correct': (actual == 1)
                }
                all_trades.append(trade)

        # Fold results
        fold_trades = [t for t in all_trades if t['fold'] == fold]
        if len(fold_trades) > 0:
            fold_correct = sum(t['correct'] for t in fold_trades)
            fold_total = len(fold_trades)
            fold_wr = fold_correct / fold_total
            print(f"Fold {fold}: {fold_total} trades, {fold_correct} correct ({fold_wr*100:.1f}%)")

    # Overall results
    print(f"\n{'='*80}")
    print("Rule-Based SHORT Results")
    print(f"{'='*80}")

    if len(all_trades) == 0:
        print("\n❌ No trades generated")
        return 0.0, all_trades

    total_trades = len(all_trades)
    correct_trades = sum(t['correct'] for t in all_trades)
    win_rate = correct_trades / total_trades

    print(f"\nOverall Results:")
    print(f"  Total Trades: {total_trades}")
    print(f"  Correct: {correct_trades}")
    print(f"  Incorrect: {total_trades - correct_trades}")
    print(f"  Win Rate: {win_rate*100:.1f}%")

    # Score distribution
    trades_df = pd.DataFrame(all_trades)
    print(f"\nScore Distribution:")
    for score in sorted(trades_df['score'].unique()):
        score_trades = trades_df[trades_df['score'] == score]
        score_correct = score_trades['correct'].sum()
        score_total = len(score_trades)
        score_wr = score_correct / score_total * 100 if score_total > 0 else 0
        print(f"  Score {score}: {score_total} trades, {score_correct} correct ({score_wr:.1f}%)")

    return win_rate, all_trades


def main():
    """Main pipeline"""
    print("="*80)
    print("SHORT Model - Rule-Based Expert System (Approach #15)")
    print("="*80)
    print("Final Critical Insight:")
    print("  - 14 ML approaches failed (best: 46%, most: 10-27%)")
    print("  - Why? ML can't learn from noisy 5-min SHORT patterns")
    print("  - Solution: Code expert technical analysis rules directly")
    print("")
    print("Rule-Based Strategy:")
    print("  1. Overbought (RSI > 65)")
    print("  2. Resistance (price near BB upper)")
    print("  3. Bearish signals (divergence, patterns)")
    print("  4. Regime filter (short allowed)")
    print("  5. Volume confirmation")
    print("  6. Momentum exhaustion")
    print("")
    print("  Enter SHORT if score >= 5/9")
    print("="*80)

    # Load data
    print("\n" + "="*80)
    print("Loading Data")
    print("="*80)

    df = pd.read_csv(DATA_DIR / "BTCUSDT_5m_max.csv")
    print(f"Loaded {len(df)} candles")

    # Calculate features
    df = calculate_features(df)

    short_features = ShortSpecificFeatures(lookback=20)
    df = short_features.calculate_all_features(df)

    regime_filter = MarketRegimeFilter()
    df = regime_filter.get_regime_features(df)

    df = df.ffill().dropna()
    print(f"After processing: {len(df)} candles")

    # Create validation labels
    y_short = create_short_labels(df, lookahead=3, threshold=0.002)

    unique, counts = np.unique(y_short, return_counts=True)
    print(f"\nSHORT Validation Labels:")
    print(f"  NO SHORT (0): {counts[0]} ({counts[0]/len(y_short)*100:.1f}%)")
    print(f"  SHORT (1): {counts[1]} ({counts[1]/len(y_short)*100:.1f}%)")

    # Apply rules
    df = apply_short_rules(df)

    # Backtest
    win_rate, trades = backtest_rules(df, y_short)

    # Final decision
    print("\n" + "="*80)
    print("Final Decision - Rule-Based System")
    print("="*80)

    if win_rate >= 0.60:
        print(f"✅ SUCCESS! SHORT win rate {win_rate*100:.1f}% >= 60%")
        print(f"✅ Rule-based system WORKS!")

    elif win_rate >= 0.50:
        print(f"🔄 SIGNIFICANT: SHORT win rate {win_rate*100:.1f}% (50-60%)")
        print(f"🔄 Rules show promise, tune thresholds!")

    elif win_rate >= 0.45:
        print(f"⚠️ MODERATE: SHORT win rate {win_rate*100:.1f}% (45-50%)")
        print(f"⚠️ Near Approach #1, consider refinement")

    else:
        print(f"❌ INSUFFICIENT: SHORT win rate {win_rate*100:.1f}%")

    print(f"\nComplete Progress Summary (All 15 Approaches):")
    print(f"  ML-based approaches (#2-14): 9-46% range")
    print(f"  #1  2-Class Inverse (ML): 46.0% ✅ Best ML")
    print(f"  #15 Rule-Based System: {win_rate*100:.1f}%")

    if win_rate >= 0.50:
        print(f"\n🎉 Rule-based beats ML! Expert knowledge > data mining")
    elif win_rate < 0.30:
        print(f"\n💭 Even rules fail → fundamental market structure limits")

    return win_rate, trades


if __name__ == "__main__":
    win_rate, trades = main()
