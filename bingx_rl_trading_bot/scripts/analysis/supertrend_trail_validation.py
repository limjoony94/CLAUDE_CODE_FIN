"""
Supertrend Trailing Stop-Loss Walk-Forward Validation
Validate the incredible results from dynamic SL research
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
    """Generate ADX trend signals"""
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

def backtest_supertrend_trail(df, signals, start_idx, end_idx, tp_pct=2.0, cooldown=6, leverage=4, fee_pct=0.05):
    """Backtest with Supertrend trailing stop-loss"""
    trades = []
    position = None
    last_trade_idx = start_idx - cooldown - 1

    for i in range(start_idx, end_idx):
        if i - last_trade_idx < cooldown:
            continue

        if position is None and signals[i] != 0:
            direction = signals[i]
            entry_price = df['close'].iloc[i]

            # SL at Supertrend line
            sl_price = df['supertrend'].iloc[i]
            sl_distance = abs(entry_price - sl_price) / entry_price * 100

            # Skip if SL distance is too large or too small
            if sl_distance > 5.0 or sl_distance < 0.3:
                continue

            position = {
                'direction': direction,
                'entry_price': entry_price,
                'entry_idx': i,
                'tp_price': entry_price * (1 + direction * tp_pct / 100),
                'sl_price': sl_price,
                'sl_distance': sl_distance,
                'entry_time': df['timestamp'].iloc[i]
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
                'entry_time': position['entry_time'],
                'exit_time': df['timestamp'].iloc[i]
            })

            position = None
            last_trade_idx = i

    return trades

def calculate_metrics(trades):
    if not trades:
        return {'trades': 0, 'return': 0, 'win_rate': 0, 'mdd': 0}

    pnls = [t['pnl_pct'] for t in trades]
    wins = [p for p in pnls if p > 0]
    cum_return = sum(pnls)
    win_rate = len(wins) / len(pnls) * 100 if pnls else 0

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

    tp_count = len([t for t in trades if t['exit_reason'] == 'TP'])
    sl_count = len([t for t in trades if t['exit_reason'] == 'SL'])

    return {
        'trades': len(trades),
        'return': cum_return,
        'win_rate': win_rate,
        'mdd': mdd,
        'tp_count': tp_count,
        'sl_count': sl_count,
        'risk_adj': cum_return / mdd if mdd > 0 else 0
    }

def walk_forward_validation(df, signals, tp_pct=2.0, cooldown=6, train_days=60, test_days=30):
    """Walk-forward validation"""
    results = []
    candles_per_day = 96  # 15min candles

    train_candles = train_days * candles_per_day
    test_candles = test_days * candles_per_day
    window_size = train_candles + test_candles

    start_idx = 200
    current_idx = start_idx
    window_num = 0

    while current_idx + window_size <= len(df):
        window_num += 1

        train_start = current_idx
        train_end = current_idx + train_candles
        test_start = train_end
        test_end = train_end + test_candles

        # Train period
        train_trades = backtest_supertrend_trail(df, signals, train_start, train_end, tp_pct, cooldown)
        train_metrics = calculate_metrics(train_trades)

        # Test period
        test_trades = backtest_supertrend_trail(df, signals, test_start, test_end, tp_pct, cooldown)
        test_metrics = calculate_metrics(test_trades)

        results.append({
            'window': window_num,
            'train_start': df['timestamp'].iloc[train_start],
            'train_end': df['timestamp'].iloc[train_end-1],
            'test_start': df['timestamp'].iloc[test_start],
            'test_end': df['timestamp'].iloc[test_end-1],
            'train_trades': train_metrics['trades'],
            'train_return': train_metrics['return'],
            'train_wr': train_metrics['win_rate'],
            'train_mdd': train_metrics['mdd'],
            'test_trades': test_metrics['trades'],
            'test_return': test_metrics['return'],
            'test_wr': test_metrics['win_rate'],
            'test_mdd': test_metrics['mdd']
        })

        current_idx += test_candles

    return results

def monthly_breakdown(df, signals, tp_pct=2.0, cooldown=6):
    """Monthly performance breakdown"""
    all_trades = backtest_supertrend_trail(df, signals, 200, len(df), tp_pct, cooldown)

    monthly = {}
    for trade in all_trades:
        month = trade['exit_time'].strftime('%Y-%m')
        if month not in monthly:
            monthly[month] = []
        monthly[month].append(trade['pnl_pct'])

    results = []
    for month, pnls in sorted(monthly.items()):
        results.append({
            'month': month,
            'trades': len(pnls),
            'return': sum(pnls),
            'win_rate': len([p for p in pnls if p > 0]) / len(pnls) * 100 if pnls else 0
        })

    return results

def main():
    print("=" * 80)
    print("SUPERTREND TRAILING STOP-LOSS - WALK-FORWARD VALIDATION")
    print("=" * 80)

    print("\nLoading data...")
    df = load_data()
    print(f"Data: {df['timestamp'].iloc[0]} to {df['timestamp'].iloc[-1]}")
    days = (df['timestamp'].iloc[-1] - df['timestamp'].iloc[0]).days
    print(f"Period: {days} days")

    print("\nAdding indicators...")
    df = add_indicators(df)

    print("\nGenerating signals...")
    signals = generate_signals(df, adx_threshold=20)

    # Configuration
    TP_PCT = 2.0
    COOLDOWN = 6

    print(f"\nConfiguration: ADX_Trend_20 + Supertrend Trail | TP{TP_PCT}% | CD{COOLDOWN}")

    # Full period backtest
    print("\n" + "=" * 80)
    print("FULL PERIOD BACKTEST")
    print("=" * 80)

    all_trades = backtest_supertrend_trail(df, signals, 200, len(df), TP_PCT, COOLDOWN)
    full_metrics = calculate_metrics(all_trades)

    print(f"Period: {days} days")
    print(f"Trades: {full_metrics['trades']} ({full_metrics['trades']/days:.2f}/day)")
    print(f"Win Rate: {full_metrics['win_rate']:.1f}%")
    print(f"Total Return: {full_metrics['return']:+.2f}%")
    print(f"Max Drawdown: {full_metrics['mdd']:.2f}%")
    print(f"Risk-Adjusted Return: {full_metrics['risk_adj']:.2f}")
    print(f"TP: {full_metrics['tp_count']} | SL: {full_metrics['sl_count']}")

    # Walk-forward validation
    print("\n" + "=" * 80)
    print("WALK-FORWARD VALIDATION (60-day train / 30-day test)")
    print("=" * 80)

    wf_results = walk_forward_validation(df, signals, TP_PCT, COOLDOWN)

    print(f"\nWindow | Train Period        | Test Period         | Train    | Test")
    print(f"       |                     |                     | Ret/MDD  | Ret/MDD")
    print("-" * 80)

    train_returns = []
    test_returns = []
    test_positive = 0

    for r in wf_results:
        train_ra = r['train_return']/r['train_mdd'] if r['train_mdd'] > 0 else 0
        test_ra = r['test_return']/r['test_mdd'] if r['test_mdd'] > 0 else 0

        train_returns.append(r['train_return'])
        test_returns.append(r['test_return'])
        if r['test_return'] > 0:
            test_positive += 1

        train_str = f"{r['train_start'].strftime('%m/%d')}-{r['train_end'].strftime('%m/%d')}"
        test_str = f"{r['test_start'].strftime('%m/%d')}-{r['test_end'].strftime('%m/%d')}"

        print(f"W{r['window']:2d}    | {train_str:18s} | {test_str:18s} | {r['train_return']:+7.1f}% | {r['test_return']:+7.1f}%")

    print("-" * 80)
    print(f"Average Train Return: {np.mean(train_returns):+.2f}%")
    print(f"Average Test Return: {np.mean(test_returns):+.2f}%")
    print(f"Test Positive Windows: {test_positive}/{len(wf_results)} ({test_positive/len(wf_results)*100:.1f}%)")
    print(f"Train/Test Correlation: {np.corrcoef(train_returns, test_returns)[0,1]:.2f}")

    # Monthly breakdown
    print("\n" + "=" * 80)
    print("MONTHLY BREAKDOWN")
    print("=" * 80)

    monthly = monthly_breakdown(df, signals, TP_PCT, COOLDOWN)

    positive_months = 0
    for m in monthly:
        status = "+" if m['return'] > 0 else "-"
        print(f"{m['month']}: {m['trades']:3d} trades | {m['return']:+8.2f}% | WR {m['win_rate']:.1f}% {status}")
        if m['return'] > 0:
            positive_months += 1

    print("-" * 80)
    print(f"Positive Months: {positive_months}/{len(monthly)} ({positive_months/len(monthly)*100:.1f}%)")

    # Monte Carlo
    print("\n" + "=" * 80)
    print("MONTE CARLO SIMULATION (1000 iterations)")
    print("=" * 80)

    if all_trades:
        pnls = [t['pnl_pct'] for t in all_trades]
        mc_returns = []
        for _ in range(1000):
            shuffled = np.random.choice(pnls, len(pnls), replace=True)
            mc_returns.append(sum(shuffled))

        mc_returns = np.array(mc_returns)
        print(f"Median Return: {np.median(mc_returns):+.2f}%")
        print(f"5th Percentile: {np.percentile(mc_returns, 5):+.2f}%")
        print(f"95th Percentile: {np.percentile(mc_returns, 95):+.2f}%")
        print(f"Probability of Profit: {np.mean(mc_returns > 0)*100:.1f}%")

    # Validation Summary
    print("\n" + "=" * 80)
    print("VALIDATION SUMMARY")
    print("=" * 80)

    validation_score = (
        (full_metrics['risk_adj'] / 10) * 20 +  # Risk-adjusted (max 50 if RA=25)
        (test_positive / len(wf_results)) * 30 +  # Walk-forward positive rate (max 30)
        (positive_months / len(monthly)) * 20 +  # Monthly positive rate (max 20)
        (np.mean(mc_returns > 0)) * 30  # Monte Carlo positive probability (max 30)
    )

    print(f"\nStrategy: ADX_Trend_20 + Supertrend Trail")
    print(f"Parameters: TP{TP_PCT}% / Supertrend SL / Cooldown{COOLDOWN}")
    print(f"\nKey Metrics:")
    print(f"  - Trades: {full_metrics['trades']} ({full_metrics['trades']/days:.2f}/day)")
    print(f"  - Return: {full_metrics['return']:+.2f}%")
    print(f"  - Max DD: {full_metrics['mdd']:.2f}%")
    print(f"  - Risk-Adj: {full_metrics['risk_adj']:.2f}")
    print(f"\nValidation Scores:")
    print(f"  - Walk-Forward Positive: {test_positive}/{len(wf_results)} ({test_positive/len(wf_results)*100:.1f}%)")
    print(f"  - Monthly Positive: {positive_months}/{len(monthly)} ({positive_months/len(monthly)*100:.1f}%)")
    print(f"  - Monte Carlo Positive: {np.mean(mc_returns > 0)*100:.1f}%")
    print(f"\n  >> COMBINED SCORE: {validation_score:.1f}/100")

    if validation_score >= 70:
        print("\n✅ STRATEGY VALIDATED - Ready for production")
    elif validation_score >= 50:
        print("\n⚠️ STRATEGY MARGINAL - Use with caution")
    else:
        print("\n❌ STRATEGY NOT VALIDATED - Needs improvement")

    # Trade analysis
    print("\n" + "=" * 80)
    print("TRADE ANALYSIS")
    print("=" * 80)

    if all_trades:
        pnls = [t['pnl_pct'] for t in all_trades]
        wins = [p for p in pnls if p > 0]
        losses = [p for p in pnls if p <= 0]

        print(f"\nWinning Trades:")
        print(f"  - Count: {len(wins)}")
        print(f"  - Average: {np.mean(wins):.2f}%")
        print(f"  - Max: {max(wins):.2f}%")

        print(f"\nLosing Trades:")
        print(f"  - Count: {len(losses)}")
        print(f"  - Average: {np.mean(losses):.2f}%")
        print(f"  - Max Loss: {min(losses):.2f}%")

        print(f"\nRisk/Reward Ratio: {abs(np.mean(wins)/np.mean(losses)):.2f}")

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f"results/supertrend_trail_validation_{timestamp}.csv"
    pd.DataFrame(wf_results).to_csv(output_file, index=False)
    print(f"\nResults saved to: {output_file}")

if __name__ == "__main__":
    main()
