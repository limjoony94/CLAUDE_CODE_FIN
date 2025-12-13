"""
Dynamic Stop-Loss Research for ADX Trend Strategy
Compare fixed % SL vs ATR-based, swing-based, and indicator-based stops
"""

import pandas as pd
import numpy as np
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

DATA_PATH = "data/btc_15m_full_2025.csv"

def load_data():
    df = pd.read_csv(DATA_PATH)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values('timestamp').reset_index(drop=True)
    return df

def add_indicators(df):
    """Add all required indicators"""
    # ATR
    high_low = df['high'] - df['low']
    high_close = abs(df['high'] - df['close'].shift(1))
    low_close = abs(df['low'] - df['close'].shift(1))
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df['atr'] = tr.ewm(span=14, adjust=False).mean()

    # ADX, +DI, -DI
    plus_dm = df['high'].diff()
    minus_dm = -df['low'].diff()
    plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0)
    minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0)

    atr_14 = tr.ewm(span=14, adjust=False).mean()
    plus_di = 100 * (plus_dm.ewm(span=14, adjust=False).mean() / atr_14)
    minus_di = 100 * (minus_dm.ewm(span=14, adjust=False).mean() / atr_14)
    dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
    df['adx'] = dx.ewm(span=14, adjust=False).mean()
    df['plus_di'] = plus_di
    df['minus_di'] = minus_di

    # Swing High/Low (lookback 10 candles)
    df['swing_low'] = df['low'].rolling(10).min()
    df['swing_high'] = df['high'].rolling(10).max()

    # Supertrend
    hl2 = (df['high'] + df['low']) / 2
    mult = 3.0
    upper = hl2 + mult * df['atr']
    lower = hl2 - mult * df['atr']

    supertrend = np.zeros(len(df))
    direction = np.zeros(len(df))

    for i in range(1, len(df)):
        if df['close'].iloc[i] > upper.iloc[i-1]:
            direction[i] = 1
        elif df['close'].iloc[i] < lower.iloc[i-1]:
            direction[i] = -1
        else:
            direction[i] = direction[i-1]

        if direction[i] == 1:
            supertrend[i] = max(lower.iloc[i], supertrend[i-1] if direction[i-1] == 1 else lower.iloc[i])
        else:
            supertrend[i] = min(upper.iloc[i], supertrend[i-1] if direction[i-1] == -1 else upper.iloc[i])

    df['supertrend'] = supertrend
    df['st_direction'] = direction

    return df

def generate_signals(df, adx_threshold=20):
    """Generate ADX trend signals (+DI/-DI crossover when ADX > threshold)"""
    signals = np.zeros(len(df))
    adx = df['adx'].values
    plus_di = df['plus_di'].values
    minus_di = df['minus_di'].values

    for i in range(1, len(df)):
        if adx[i] > adx_threshold:
            if plus_di[i-1] < minus_di[i-1] and plus_di[i] > minus_di[i]:
                signals[i] = 1  # LONG
            elif plus_di[i-1] > minus_di[i-1] and plus_di[i] < minus_di[i]:
                signals[i] = -1  # SHORT

    return signals

def backtest_fixed_sl(df, signals, tp_pct, sl_pct, cooldown=6, leverage=4, fee_pct=0.05):
    """Backtest with fixed percentage stop-loss"""
    trades = []
    position = None
    last_trade_idx = -cooldown - 1

    for i in range(200, len(df)):
        if i - last_trade_idx < cooldown:
            continue

        if position is None and signals[i] != 0:
            direction = signals[i]
            entry_price = df['close'].iloc[i]
            position = {
                'direction': direction,
                'entry_price': entry_price,
                'entry_idx': i,
                'tp_price': entry_price * (1 + direction * tp_pct / 100),
                'sl_price': entry_price * (1 - direction * sl_pct / 100)
            }
            continue

        if position is not None:
            high = df['high'].iloc[i]
            low = df['low'].iloc[i]

            hit_tp = (position['direction'] == 1 and high >= position['tp_price']) or \
                     (position['direction'] == -1 and low <= position['tp_price'])
            hit_sl = (position['direction'] == 1 and low <= position['sl_price']) or \
                     (position['direction'] == -1 and high >= position['sl_price'])

            if hit_tp and hit_sl:
                exit_price = position['sl_price']
                exit_reason = 'SL'
            elif hit_tp:
                exit_price = position['tp_price']
                exit_reason = 'TP'
            elif hit_sl:
                exit_price = position['sl_price']
                exit_reason = 'SL'
            else:
                continue

            pnl_pct = position['direction'] * (exit_price / position['entry_price'] - 1) * 100 * leverage
            pnl_pct -= 2 * fee_pct * leverage

            trades.append({
                'entry_idx': position['entry_idx'],
                'exit_idx': i,
                'direction': position['direction'],
                'pnl_pct': pnl_pct,
                'exit_reason': exit_reason,
                'sl_type': 'FIXED',
                'sl_distance': sl_pct
            })

            position = None
            last_trade_idx = i

    return trades

def backtest_atr_sl(df, signals, tp_pct, atr_mult, cooldown=6, leverage=4, fee_pct=0.05):
    """Backtest with ATR-based stop-loss"""
    trades = []
    position = None
    last_trade_idx = -cooldown - 1

    for i in range(200, len(df)):
        if i - last_trade_idx < cooldown:
            continue

        if position is None and signals[i] != 0:
            direction = signals[i]
            entry_price = df['close'].iloc[i]
            atr = df['atr'].iloc[i]
            sl_distance = atr * atr_mult / entry_price * 100  # Convert to percentage

            position = {
                'direction': direction,
                'entry_price': entry_price,
                'entry_idx': i,
                'tp_price': entry_price * (1 + direction * tp_pct / 100),
                'sl_price': entry_price * (1 - direction * sl_distance / 100),
                'sl_distance': sl_distance
            }
            continue

        if position is not None:
            high = df['high'].iloc[i]
            low = df['low'].iloc[i]

            hit_tp = (position['direction'] == 1 and high >= position['tp_price']) or \
                     (position['direction'] == -1 and low <= position['tp_price'])
            hit_sl = (position['direction'] == 1 and low <= position['sl_price']) or \
                     (position['direction'] == -1 and high >= position['sl_price'])

            if hit_tp and hit_sl:
                exit_price = position['sl_price']
                exit_reason = 'SL'
            elif hit_tp:
                exit_price = position['tp_price']
                exit_reason = 'TP'
            elif hit_sl:
                exit_price = position['sl_price']
                exit_reason = 'SL'
            else:
                continue

            pnl_pct = position['direction'] * (exit_price / position['entry_price'] - 1) * 100 * leverage
            pnl_pct -= 2 * fee_pct * leverage

            trades.append({
                'entry_idx': position['entry_idx'],
                'exit_idx': i,
                'direction': position['direction'],
                'pnl_pct': pnl_pct,
                'exit_reason': exit_reason,
                'sl_type': f'ATR_{atr_mult}x',
                'sl_distance': position['sl_distance']
            })

            position = None
            last_trade_idx = i

    return trades

def backtest_swing_sl(df, signals, tp_pct, swing_buffer_pct=0.1, cooldown=6, leverage=4, fee_pct=0.05):
    """Backtest with swing low/high based stop-loss"""
    trades = []
    position = None
    last_trade_idx = -cooldown - 1

    for i in range(200, len(df)):
        if i - last_trade_idx < cooldown:
            continue

        if position is None and signals[i] != 0:
            direction = signals[i]
            entry_price = df['close'].iloc[i]

            if direction == 1:  # LONG: SL below swing low
                sl_price = df['swing_low'].iloc[i] * (1 - swing_buffer_pct / 100)
            else:  # SHORT: SL above swing high
                sl_price = df['swing_high'].iloc[i] * (1 + swing_buffer_pct / 100)

            sl_distance = abs(entry_price - sl_price) / entry_price * 100

            # Skip if SL distance is too large (>5%) or too small (<0.3%)
            if sl_distance > 5.0 or sl_distance < 0.3:
                continue

            position = {
                'direction': direction,
                'entry_price': entry_price,
                'entry_idx': i,
                'tp_price': entry_price * (1 + direction * tp_pct / 100),
                'sl_price': sl_price,
                'sl_distance': sl_distance
            }
            continue

        if position is not None:
            high = df['high'].iloc[i]
            low = df['low'].iloc[i]

            hit_tp = (position['direction'] == 1 and high >= position['tp_price']) or \
                     (position['direction'] == -1 and low <= position['tp_price'])
            hit_sl = (position['direction'] == 1 and low <= position['sl_price']) or \
                     (position['direction'] == -1 and high >= position['sl_price'])

            if hit_tp and hit_sl:
                exit_price = position['sl_price']
                exit_reason = 'SL'
            elif hit_tp:
                exit_price = position['tp_price']
                exit_reason = 'TP'
            elif hit_sl:
                exit_price = position['sl_price']
                exit_reason = 'SL'
            else:
                continue

            pnl_pct = position['direction'] * (exit_price / position['entry_price'] - 1) * 100 * leverage
            pnl_pct -= 2 * fee_pct * leverage

            trades.append({
                'entry_idx': position['entry_idx'],
                'exit_idx': i,
                'direction': position['direction'],
                'pnl_pct': pnl_pct,
                'exit_reason': exit_reason,
                'sl_type': 'SWING',
                'sl_distance': position['sl_distance']
            })

            position = None
            last_trade_idx = i

    return trades

def backtest_supertrend_sl(df, signals, tp_pct, cooldown=6, leverage=4, fee_pct=0.05):
    """Backtest with Supertrend line as stop-loss"""
    trades = []
    position = None
    last_trade_idx = -cooldown - 1

    for i in range(200, len(df)):
        if i - last_trade_idx < cooldown:
            continue

        if position is None and signals[i] != 0:
            direction = signals[i]
            entry_price = df['close'].iloc[i]

            # SL at Supertrend line
            sl_price = df['supertrend'].iloc[i]
            sl_distance = abs(entry_price - sl_price) / entry_price * 100

            # Skip if SL distance is too large (>5%) or too small (<0.3%)
            if sl_distance > 5.0 or sl_distance < 0.3:
                continue

            position = {
                'direction': direction,
                'entry_price': entry_price,
                'entry_idx': i,
                'tp_price': entry_price * (1 + direction * tp_pct / 100),
                'sl_price': sl_price,
                'sl_distance': sl_distance
            }
            continue

        if position is not None:
            high = df['high'].iloc[i]
            low = df['low'].iloc[i]

            # Update trailing SL with Supertrend
            current_st = df['supertrend'].iloc[i]
            if position['direction'] == 1:  # LONG: trail up
                position['sl_price'] = max(position['sl_price'], current_st)
            else:  # SHORT: trail down
                position['sl_price'] = min(position['sl_price'], current_st)

            hit_tp = (position['direction'] == 1 and high >= position['tp_price']) or \
                     (position['direction'] == -1 and low <= position['tp_price'])
            hit_sl = (position['direction'] == 1 and low <= position['sl_price']) or \
                     (position['direction'] == -1 and high >= position['sl_price'])

            if hit_tp and hit_sl:
                exit_price = position['sl_price']
                exit_reason = 'SL'
            elif hit_tp:
                exit_price = position['tp_price']
                exit_reason = 'TP'
            elif hit_sl:
                exit_price = position['sl_price']
                exit_reason = 'SL'
            else:
                continue

            pnl_pct = position['direction'] * (exit_price / position['entry_price'] - 1) * 100 * leverage
            pnl_pct -= 2 * fee_pct * leverage

            trades.append({
                'entry_idx': position['entry_idx'],
                'exit_idx': i,
                'direction': position['direction'],
                'pnl_pct': pnl_pct,
                'exit_reason': exit_reason,
                'sl_type': 'SUPERTREND',
                'sl_distance': position['sl_distance']
            })

            position = None
            last_trade_idx = i

    return trades

def backtest_di_reversal_sl(df, signals, tp_pct, fixed_sl_pct=3.0, cooldown=6, leverage=4, fee_pct=0.05):
    """Backtest with +DI/-DI reversal as exit signal (with safety fixed SL)"""
    trades = []
    position = None
    last_trade_idx = -cooldown - 1

    for i in range(200, len(df)):
        if i - last_trade_idx < cooldown:
            continue

        if position is None and signals[i] != 0:
            direction = signals[i]
            entry_price = df['close'].iloc[i]

            position = {
                'direction': direction,
                'entry_price': entry_price,
                'entry_idx': i,
                'tp_price': entry_price * (1 + direction * tp_pct / 100),
                'sl_price': entry_price * (1 - direction * fixed_sl_pct / 100),  # Safety SL
            }
            continue

        if position is not None:
            high = df['high'].iloc[i]
            low = df['low'].iloc[i]
            plus_di = df['plus_di'].iloc[i]
            minus_di = df['minus_di'].iloc[i]
            prev_plus_di = df['plus_di'].iloc[i-1]
            prev_minus_di = df['minus_di'].iloc[i-1]

            # Check for DI reversal (trend change signal)
            di_reversal = False
            if position['direction'] == 1:  # LONG position
                # Exit if +DI crosses below -DI (trend reversal to bearish)
                if prev_plus_di > prev_minus_di and plus_di < minus_di:
                    di_reversal = True
            else:  # SHORT position
                # Exit if -DI crosses below +DI (trend reversal to bullish)
                if prev_minus_di > prev_plus_di and minus_di < plus_di:
                    di_reversal = True

            hit_tp = (position['direction'] == 1 and high >= position['tp_price']) or \
                     (position['direction'] == -1 and low <= position['tp_price'])
            hit_sl = (position['direction'] == 1 and low <= position['sl_price']) or \
                     (position['direction'] == -1 and high >= position['sl_price'])

            if hit_tp:
                exit_price = position['tp_price']
                exit_reason = 'TP'
            elif hit_sl:
                exit_price = position['sl_price']
                exit_reason = 'SL'
            elif di_reversal:
                exit_price = df['close'].iloc[i]
                exit_reason = 'DI_REV'
            else:
                continue

            pnl_pct = position['direction'] * (exit_price / position['entry_price'] - 1) * 100 * leverage
            pnl_pct -= 2 * fee_pct * leverage

            sl_dist = abs(position['entry_price'] - position['sl_price']) / position['entry_price'] * 100

            trades.append({
                'entry_idx': position['entry_idx'],
                'exit_idx': i,
                'direction': position['direction'],
                'pnl_pct': pnl_pct,
                'exit_reason': exit_reason,
                'sl_type': 'DI_REVERSAL',
                'sl_distance': sl_dist
            })

            position = None
            last_trade_idx = i

    return trades

def calculate_metrics(trades):
    if not trades:
        return {'trades': 0, 'return': 0, 'win_rate': 0, 'mdd': 0, 'avg_sl_dist': 0}

    pnls = [t['pnl_pct'] for t in trades]
    wins = [p for p in pnls if p > 0]
    cum_return = sum(pnls)
    win_rate = len(wins) / len(pnls) * 100

    # MDD calculation
    equity = [100]
    for pnl in pnls:
        equity.append(equity[-1] * (1 + pnl / 100))
    peak = equity[0]
    mdd = 0
    for e in equity:
        if e > peak:
            peak = e
        dd = (peak - e) / peak * 100
        if dd > mdd:
            mdd = dd

    # Average SL distance
    avg_sl_dist = np.mean([t['sl_distance'] for t in trades])

    # Exit reason breakdown
    tp_count = len([t for t in trades if t['exit_reason'] == 'TP'])
    sl_count = len([t for t in trades if t['exit_reason'] == 'SL'])
    di_rev_count = len([t for t in trades if t['exit_reason'] == 'DI_REV'])

    return {
        'trades': len(trades),
        'return': cum_return,
        'win_rate': win_rate,
        'mdd': mdd,
        'avg_sl_dist': avg_sl_dist,
        'tp_count': tp_count,
        'sl_count': sl_count,
        'di_rev_count': di_rev_count,
        'risk_adj': cum_return / mdd if mdd > 0 else 0
    }

def main():
    print("=" * 80)
    print("DYNAMIC STOP-LOSS RESEARCH FOR ADX TREND STRATEGY")
    print("=" * 80)

    print("\nLoading data...")
    df = load_data()
    print(f"Data: {df['timestamp'].iloc[0]} to {df['timestamp'].iloc[-1]}")
    days = (df['timestamp'].iloc[-1] - df['timestamp'].iloc[0]).days
    print(f"Period: {days} days")

    print("\nAdding indicators...")
    df = add_indicators(df)

    print("\nGenerating ADX Trend signals...")
    signals = generate_signals(df, adx_threshold=20)
    print(f"Total signals: {np.sum(signals != 0)}")

    results = []

    # 1. FIXED PERCENTAGE SL (Baseline)
    print("\n" + "=" * 80)
    print("1. FIXED PERCENTAGE STOP-LOSS (Baseline)")
    print("=" * 80)

    for sl_pct in [1.0, 1.5, 2.0, 2.5]:
        trades = backtest_fixed_sl(df, signals, tp_pct=2.0, sl_pct=sl_pct)
        metrics = calculate_metrics(trades)
        results.append({
            'sl_type': f'FIXED_{sl_pct}%',
            **metrics
        })
        print(f"SL {sl_pct}%: {metrics['trades']} trades | WR {metrics['win_rate']:.1f}% | "
              f"Return {metrics['return']:+.1f}% | MDD {metrics['mdd']:.1f}% | "
              f"RiskAdj {metrics['risk_adj']:.2f}")

    # 2. ATR-BASED SL
    print("\n" + "=" * 80)
    print("2. ATR-BASED STOP-LOSS (Volatility Adaptive)")
    print("=" * 80)

    for atr_mult in [1.0, 1.5, 2.0, 2.5, 3.0]:
        trades = backtest_atr_sl(df, signals, tp_pct=2.0, atr_mult=atr_mult)
        metrics = calculate_metrics(trades)
        results.append({
            'sl_type': f'ATR_{atr_mult}x',
            **metrics
        })
        print(f"ATR {atr_mult}x: {metrics['trades']} trades | WR {metrics['win_rate']:.1f}% | "
              f"Return {metrics['return']:+.1f}% | MDD {metrics['mdd']:.1f}% | "
              f"Avg SL Dist {metrics['avg_sl_dist']:.2f}% | RiskAdj {metrics['risk_adj']:.2f}")

    # 3. SWING-BASED SL
    print("\n" + "=" * 80)
    print("3. SWING LOW/HIGH STOP-LOSS (Support/Resistance)")
    print("=" * 80)

    for buffer in [0.1, 0.2, 0.3, 0.5]:
        trades = backtest_swing_sl(df, signals, tp_pct=2.0, swing_buffer_pct=buffer)
        metrics = calculate_metrics(trades)
        results.append({
            'sl_type': f'SWING_{buffer}%buf',
            **metrics
        })
        print(f"Swing +{buffer}% buffer: {metrics['trades']} trades | WR {metrics['win_rate']:.1f}% | "
              f"Return {metrics['return']:+.1f}% | MDD {metrics['mdd']:.1f}% | "
              f"Avg SL Dist {metrics['avg_sl_dist']:.2f}% | RiskAdj {metrics['risk_adj']:.2f}")

    # 4. SUPERTREND TRAILING SL
    print("\n" + "=" * 80)
    print("4. SUPERTREND TRAILING STOP-LOSS (Dynamic Trend Line)")
    print("=" * 80)

    trades = backtest_supertrend_sl(df, signals, tp_pct=2.0)
    metrics = calculate_metrics(trades)
    results.append({
        'sl_type': 'SUPERTREND_TRAIL',
        **metrics
    })
    print(f"Supertrend Trail: {metrics['trades']} trades | WR {metrics['win_rate']:.1f}% | "
          f"Return {metrics['return']:+.1f}% | MDD {metrics['mdd']:.1f}% | "
          f"Avg SL Dist {metrics['avg_sl_dist']:.2f}% | RiskAdj {metrics['risk_adj']:.2f}")

    # 5. +DI/-DI REVERSAL EXIT
    print("\n" + "=" * 80)
    print("5. +DI/-DI REVERSAL EXIT (Trend Reversal Signal)")
    print("=" * 80)

    for safety_sl in [2.0, 3.0, 4.0]:
        trades = backtest_di_reversal_sl(df, signals, tp_pct=2.0, fixed_sl_pct=safety_sl)
        metrics = calculate_metrics(trades)
        results.append({
            'sl_type': f'DI_REV_safety{safety_sl}%',
            **metrics
        })
        print(f"DI Reversal (safety SL {safety_sl}%): {metrics['trades']} trades | "
              f"WR {metrics['win_rate']:.1f}% | Return {metrics['return']:+.1f}% | "
              f"MDD {metrics['mdd']:.1f}% | RiskAdj {metrics['risk_adj']:.2f}")
        print(f"  Exit breakdown: TP {metrics['tp_count']} | SL {metrics['sl_count']} | DI_REV {metrics['di_rev_count']}")

    # SUMMARY COMPARISON
    print("\n" + "=" * 80)
    print("COMPARISON SUMMARY (Sorted by Risk-Adjusted Return)")
    print("=" * 80)

    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('risk_adj', ascending=False)

    print(f"\n{'SL Type':<25} {'Trades':>7} {'WR%':>7} {'Return':>10} {'MDD':>8} {'AvgSL%':>8} {'RiskAdj':>9}")
    print("-" * 80)

    for _, row in results_df.head(15).iterrows():
        print(f"{row['sl_type']:<25} {row['trades']:>7} {row['win_rate']:>6.1f}% "
              f"{row['return']:>+9.1f}% {row['mdd']:>7.1f}% {row['avg_sl_dist']:>7.2f}% "
              f"{row['risk_adj']:>8.2f}")

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f"results/dynamic_sl_research_{timestamp}.csv"
    results_df.to_csv(output_file, index=False)
    print(f"\nResults saved to: {output_file}")

    # Best configuration analysis
    print("\n" + "=" * 80)
    print("BEST CONFIGURATION ANALYSIS")
    print("=" * 80)

    best = results_df.iloc[0]
    print(f"\nBest SL Type: {best['sl_type']}")
    print(f"  - Trades: {best['trades']} ({best['trades']/days:.2f}/day)")
    print(f"  - Win Rate: {best['win_rate']:.1f}%")
    print(f"  - Total Return: {best['return']:+.1f}%")
    print(f"  - Max Drawdown: {best['mdd']:.1f}%")
    print(f"  - Avg SL Distance: {best['avg_sl_dist']:.2f}%")
    print(f"  - Risk-Adjusted Return: {best['risk_adj']:.2f}")

    # Comparison with baseline
    baseline = results_df[results_df['sl_type'] == 'FIXED_1.5%'].iloc[0]
    print(f"\nComparison with FIXED_1.5% baseline:")
    print(f"  - Return: {best['return']:+.1f}% vs {baseline['return']:+.1f}% ({best['return'] - baseline['return']:+.1f}%)")
    print(f"  - MDD: {best['mdd']:.1f}% vs {baseline['mdd']:.1f}% ({best['mdd'] - baseline['mdd']:+.1f}%)")
    print(f"  - RiskAdj: {best['risk_adj']:.2f} vs {baseline['risk_adj']:.2f} ({best['risk_adj'] - baseline['risk_adj']:+.2f})")

if __name__ == "__main__":
    main()
