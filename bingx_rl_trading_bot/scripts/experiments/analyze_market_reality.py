"""
Market Reality Check

1. Visualize actual price movements
2. Check if market is truly bullish or mixed
3. Identify real SHORT opportunities
4. Find root cause of model failure
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

DATA_DIR = PROJECT_ROOT / "data" / "historical"


def analyze_price_trends(df):
    """Analyze actual price trends"""
    print("\n" + "="*80)
    print("PRICE MOVEMENT ANALYSIS")
    print("="*80)

    # Overall price change
    start_price = df['close'].iloc[0]
    end_price = df['close'].iloc[-1]
    total_change = (end_price - start_price) / start_price * 100

    print(f"\nOverall Period:")
    print(f"  Start Price: ${start_price:,.2f}")
    print(f"  End Price: ${end_price:,.2f}")
    print(f"  Total Change: {total_change:+.2f}%")

    # Rolling windows analysis
    window_days = [1, 7, 14, 30]

    for days in window_days:
        window_candles = days * 288  # 5min candles

        if window_candles > len(df):
            continue

        returns = df['close'].pct_change(window_candles) * 100

        positive_periods = (returns > 0).sum()
        negative_periods = (returns < 0).sum()
        total_periods = len(returns.dropna())

        pos_pct = positive_periods / total_periods * 100
        neg_pct = negative_periods / total_periods * 100

        print(f"\n{days}-Day Rolling Windows:")
        print(f"  Positive: {positive_periods:,} ({pos_pct:.1f}%)")
        print(f"  Negative: {negative_periods:,} ({neg_pct:.1f}%)")
        print(f"  Average |Return|: {abs(returns).mean():.2f}%")


def analyze_drawdowns(df):
    """Find significant drawdown periods (SHORT opportunities)"""
    print("\n" + "="*80)
    print("DRAWDOWN ANALYSIS (SHORT Opportunities)")
    print("="*80)

    # Calculate drawdown from rolling max
    rolling_max = df['close'].rolling(window=288*7, min_periods=1).max()  # 7-day max
    drawdown = (df['close'] - rolling_max) / rolling_max * 100

    # Find significant drawdowns
    thresholds = [-1, -2, -3, -5, -10]

    print(f"\nDrawdown Frequency (from 7-day high):")
    for threshold in thresholds:
        count = (drawdown <= threshold).sum()
        pct = count / len(df) * 100
        print(f"  >= {abs(threshold)}% drop: {count:,} candles ({pct:.2f}%)")

    # Identify distinct drawdown periods
    in_drawdown = drawdown <= -3  # 3%+ drawdown

    # Count consecutive drawdown periods
    drawdown_periods = []
    current_period_start = None

    for i in range(len(df)):
        if in_drawdown.iloc[i] and current_period_start is None:
            current_period_start = i
        elif not in_drawdown.iloc[i] and current_period_start is not None:
            period_length = i - current_period_start
            max_drawdown = drawdown.iloc[current_period_start:i].min()
            drawdown_periods.append({
                'start': current_period_start,
                'length': period_length,
                'max_dd': max_drawdown
            })
            current_period_start = None

    if len(drawdown_periods) > 0:
        print(f"\nDistinct 3%+ Drawdown Periods: {len(drawdown_periods)}")
        print(f"  Average Duration: {np.mean([p['length'] for p in drawdown_periods]):.0f} candles")
        print(f"  Average Max DD: {np.mean([p['max_dd'] for p in drawdown_periods]):.2f}%")

        # Show top 5 largest drawdowns
        drawdown_periods.sort(key=lambda x: x['max_dd'])
        print(f"\nTop 5 Largest Drawdowns:")
        for i, period in enumerate(drawdown_periods[:5], 1):
            print(f"  {i}. {period['max_dd']:.2f}% over {period['length']} candles ({period['length']*5/60:.1f} hours)")
    else:
        print(f"\n⚠️ No 3%+ drawdown periods found!")


def analyze_short_trades_realistic(df):
    """Simulate SHORT trades with various TP targets"""
    print("\n" + "="*80)
    print("REALISTIC SHORT TRADE SIMULATION")
    print("="*80)

    configs = [
        {'tp': 0.005, 'sl': 0.01, 'max_hold': 12, 'name': '0.5% TP, 1h hold'},
        {'tp': 0.01, 'sl': 0.01, 'max_hold': 24, 'name': '1.0% TP, 2h hold'},
        {'tp': 0.015, 'sl': 0.01, 'max_hold': 48, 'name': '1.5% TP, 4h hold'},
        {'tp': 0.01, 'sl': 0.005, 'max_hold': 24, 'name': '1.0% TP, 0.5% SL, 2h'},
    ]

    for config in configs:
        tp_pct = config['tp']
        sl_pct = config['sl']
        max_hold = config['max_hold']

        tp_count = 0
        sl_count = 0
        max_hold_count = 0

        # Sample every 10th candle for speed
        for i in range(0, len(df) - max_hold, 10):
            entry_price = df['close'].iloc[i]
            tp_price = entry_price * (1 - tp_pct)
            sl_price = entry_price * (1 + sl_pct)

            tp_hit = False
            sl_hit = False

            for j in range(1, max_hold + 1):
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
                tp_count += 1
            elif sl_hit:
                sl_count += 1
            else:
                max_hold_count += 1

        total = tp_count + sl_count + max_hold_count
        win_rate = tp_count / total * 100 if total > 0 else 0

        # Calculate expected value
        expected_value = (tp_count * tp_pct * 100 - sl_count * sl_pct * 100) / total if total > 0 else 0

        print(f"\n{config['name']}:")
        print(f"  Win Rate: {win_rate:.1f}%")
        print(f"  TP: {tp_count}, SL: {sl_count}, Max Hold: {max_hold_count}")
        print(f"  Expected Value: {expected_value:+.2f}%")


def check_feature_availability(df):
    """Check which features are actually available"""
    print("\n" + "="*80)
    print("FEATURE AVAILABILITY CHECK")
    print("="*80)

    expected_features = [
        'rsi', 'macd', 'macd_signal', 'bb_upper', 'bb_lower',
        'ema_21', 'ema_50', 'atr', 'volume_ratio', 'price_distance_ema',
        'atr_ratio', 'bb_width', 'true_range', 'high_low_range',
        'stochrsi', 'willr', 'cci', 'cmo', 'uo', 'roc', 'mfi', 'tsi', 'kst',
        'adx', 'di_plus', 'di_minus', 'aroon_up', 'aroon_down', 'vi',
        'obv', 'cmf',
        'macd_histogram', 'bb_position', 'price_momentum'
    ]

    available = []
    missing = []

    for feature in expected_features:
        if feature in df.columns:
            available.append(feature)
        else:
            missing.append(feature)

    print(f"\nExpected Features: {len(expected_features)}")
    print(f"Available: {len(available)}")
    print(f"Missing: {len(missing)}")

    if missing:
        print(f"\n⚠️ Missing Features ({len(missing)}):")
        for feature in missing:
            print(f"  - {feature}")

    # Check what features ARE available
    all_columns = df.columns.tolist()
    feature_like = [c for c in all_columns if c not in ['timestamp', 'open', 'high', 'low', 'close', 'volume']]

    print(f"\nActually Available Features: {len(feature_like)}")
    if len(feature_like) < 40:
        print("Available features:")
        for f in sorted(feature_like):
            print(f"  - {f}")


def main():
    print("="*80)
    print("MARKET REALITY CHECK")
    print("="*80)
    print("\n🎯 Goal: Understand why SHORT model fails")
    print("       Is it really a bull market? Or something else?")

    # Load data
    data_file = DATA_DIR / "BTCUSDT_5m_max.csv"
    df = pd.read_csv(data_file)

    print(f"\nData: {len(df):,} candles")
    print(f"Period: {df['timestamp'].iloc[0]} to {df['timestamp'].iloc[-1]}")

    # Add date for better analysis
    df['date'] = pd.to_datetime(df['timestamp'])

    # 1. Price trends
    analyze_price_trends(df)

    # 2. Drawdowns (SHORT opportunities)
    analyze_drawdowns(df)

    # 3. Realistic SHORT simulations
    analyze_short_trades_realistic(df)

    # 4. Load processed data to check features
    from scripts.production.train_xgboost_improved_v3_phase2 import calculate_features
    from scripts.production.advanced_technical_features import AdvancedTechnicalFeatures

    df_features = calculate_features(df.copy())
    adv = AdvancedTechnicalFeatures()
    df_features = adv.calculate_all_features(df_features)

    check_feature_availability(df_features)

    print("\n" + "="*80)
    print("CONCLUSIONS")
    print("="*80)
    print("\nBased on this analysis:")
    print("1. Is this a bull market or mixed market?")
    print("2. Are there enough SHORT opportunities?")
    print("3. What's the real issue with SHORT model?")
    print("4. What should we fix?")


if __name__ == "__main__":
    main()
