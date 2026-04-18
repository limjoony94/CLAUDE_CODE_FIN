"""
C1 Reverse FULL-period backtest (332 days).

Same reverse logic as c1_reverse_30days_backtest.py, but evaluates on entire
dataset history. Produces WF-style segmentation (5 folds) for stability check.
"""
import sys, os, json, math
from datetime import datetime
from pathlib import Path
from collections import defaultdict

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

import pandas as pd
from scripts.production.c1_breakout.signals import C1BreakoutSignal
from scripts.production.c1_breakout.indicators import (
    compute_atr, compute_channel
)
from scripts.analysis.c1_reverse_30days_backtest import (
    CONFIG, FEE_RT_PCT, LEVERAGE, load_and_resample,
    reverse_check_entry, summarize,
)


def run_backtest(df15, start_idx, end_idx, timestamps, opens, highs, lows, closes,
                 atr, ch_high, ch_low):
    signal = C1BreakoutSignal(CONFIG)
    trades = []
    in_position = False
    cooldown_until = 0
    pos_direction = None
    pos_entry_price = None
    pos_entry_time = None
    pos_sl_price = None
    pos_best_price = None
    pos_bars_held = 0

    for i in range(start_idx, end_idx + 1):
        if in_position:
            pos_bars_held += 1
            if pos_direction == 'LONG':
                pos_best_price = max(pos_best_price, highs[i])
            else:
                pos_best_price = min(pos_best_price, lows[i])
            exit_result = signal.check_exit(
                direction=pos_direction, entry_price=pos_entry_price,
                best_price=pos_best_price, current_high=highs[i],
                current_low=lows[i], current_close=closes[i],
                sl_price=pos_sl_price,
                atr_val=atr[i] if not math.isnan(atr[i]) else atr[i - 1],
                bars_held=pos_bars_held,
            )
            if exit_result is not None:
                exit_price = exit_result['exit_price']
                reason = exit_result['reason']
                if pos_direction == 'LONG':
                    pnl_pct = (exit_price / pos_entry_price - 1) * 100
                else:
                    pnl_pct = (1 - exit_price / pos_entry_price) * 100
                pnl_pct -= FEE_RT_PCT
                trades.append({
                    'direction': pos_direction,
                    'entry_time': str(pos_entry_time),
                    'entry_price': round(pos_entry_price, 1),
                    'exit_time': str(timestamps[i]),
                    'exit_price': round(exit_price, 1),
                    'pnl_pct': round(pnl_pct, 4),
                    'pnl_pct_3x': round(pnl_pct * LEVERAGE, 4),
                    'reason': reason,
                    'bars_held': pos_bars_held,
                })
                in_position = False
                cooldown_until = i + CONFIG['min_bars_between']
                pos_direction = None

        if not in_position and i >= cooldown_until and i < end_idx:
            if math.isnan(atr[i]) or math.isnan(ch_high[i]):
                continue
            entry_signal = reverse_check_entry(
                opens[i], highs[i], lows[i], closes[i],
                ch_high[i], ch_low[i], atr[i],
            )
            if entry_signal is not None:
                next_i = i + 1
                if next_i > end_idx:
                    continue
                pos_direction = entry_signal['direction']
                pos_entry_price = opens[next_i]
                pos_entry_time = timestamps[next_i]
                pos_sl_price = entry_signal['sl_price']
                pos_bars_held = 0
                in_position = True
                if pos_direction == 'LONG':
                    pos_best_price = highs[next_i]
                else:
                    pos_best_price = lows[next_i]
                # immediate exit check (entry bar)
                exit_result = signal.check_exit(
                    direction=pos_direction, entry_price=pos_entry_price,
                    best_price=pos_best_price, current_high=highs[next_i],
                    current_low=lows[next_i], current_close=closes[next_i],
                    sl_price=pos_sl_price,
                    atr_val=atr[next_i] if not math.isnan(atr[next_i]) else atr[i],
                    bars_held=0,
                )
                if exit_result is not None:
                    exit_price = exit_result['exit_price']
                    reason = exit_result['reason']
                    if pos_direction == 'LONG':
                        pnl_pct = (exit_price / pos_entry_price - 1) * 100
                    else:
                        pnl_pct = (1 - exit_price / pos_entry_price) * 100
                    pnl_pct -= FEE_RT_PCT
                    trades.append({
                        'direction': pos_direction,
                        'entry_time': str(pos_entry_time),
                        'entry_price': round(pos_entry_price, 1),
                        'exit_time': str(timestamps[next_i]),
                        'exit_price': round(exit_price, 1),
                        'pnl_pct': round(pnl_pct, 4),
                        'pnl_pct_3x': round(pnl_pct * LEVERAGE, 4),
                        'reason': reason,
                        'bars_held': 0,
                    })
                    in_position = False
                    cooldown_until = next_i + CONFIG['min_bars_between']
                    pos_direction = None
    return trades


def main():
    csv_path = str(ROOT / 'data' / 'btc_5m_270days_reclassified.csv')
    df15 = load_and_resample(csv_path)
    opens = df15['open'].tolist()
    highs = df15['high'].tolist()
    lows = df15['low'].tolist()
    closes = df15['close'].tolist()
    timestamps = df15['timestamp'].tolist()
    n = len(closes)
    print(f"Total 15m bars: {n} | From {timestamps[0]} to {timestamps[-1]}")

    atr = compute_atr(highs, lows, closes, CONFIG['atr_period'])
    ch_high, ch_low = compute_channel(highs, lows, CONFIG['channel_period'])

    # Skip warmup (first 50 bars)
    warmup = 50
    trades = run_backtest(df15, warmup, n - 1, timestamps,
                          opens, highs, lows, closes, atr, ch_high, ch_low)

    summary = summarize(trades)
    days = (pd.Timestamp(timestamps[-1]) - pd.Timestamp(timestamps[warmup])).days
    daily_pnl = summary.get('total_pnl_1x', 0) / max(days, 1)
    tpd = summary.get('total_trades', 0) / max(days, 1)

    print()
    print("=" * 100)
    print(f"REVERSE STRATEGY — Full {days} days")
    print("=" * 100)
    for k, v in summary.items():
        print(f"  {k}: {v}")
    print(f"  daily_pnl_1x: {daily_pnl:+.4f}%")
    print(f"  trades_per_day: {tpd:.2f}")
    print()

    # WF-style 5-fold segmentation (expanding window not needed for stability view;
    # just split trades into 5 equal chronological segments)
    print("=" * 100)
    print("WF-style 5 chronological folds")
    print("=" * 100)
    n_trades = len(trades)
    fold_size = n_trades // 5
    folds_pnl = []
    for k in range(5):
        si = k * fold_size
        ei = (k + 1) * fold_size if k < 4 else n_trades
        fold_trades = trades[si:ei]
        fold_pnl = sum(t['pnl_pct'] for t in fold_trades)
        wins = sum(1 for t in fold_trades if t['pnl_pct'] > 0)
        wr = wins / max(len(fold_trades), 1) * 100
        folds_pnl.append(fold_pnl)
        start_t = fold_trades[0]['entry_time'][:10] if fold_trades else '-'
        end_t = fold_trades[-1]['entry_time'][:10] if fold_trades else '-'
        status = 'PASS' if fold_pnl > 0 else 'FAIL'
        print(f"  Fold {k+1}: {start_t} ~ {end_t}  "
              f"trades={len(fold_trades):4d}  PnL={fold_pnl:+8.2f}%  "
              f"WR={wr:4.1f}%  [{status}]")
    passed = sum(1 for p in folds_pnl if p > 0)
    print(f"  → {passed}/5 folds positive")

    # Compare normal baseline (v2.5 full-period)
    print()
    print("=" * 100)
    print("COMPARISON vs NORMAL C1 baseline (v2.5 full 333-day, additive 1x)")
    print("=" * 100)
    normal_baseline = {
        'trades': 1028, 'WR': 36.6, 'RR': 3.36,
        'PnL_1x': 169.5, 'MDD_1x': 5.4,
        'daily_pnl': 0.509, 'trades_per_day': 3.1,
    }
    print(f"  {'Metric':15s} {'Normal':>12s} {'Reverse':>12s}")
    print(f"  {'trades':15s} {normal_baseline['trades']:>12} {summary['total_trades']:>12}")
    print(f"  {'WR %':15s} {normal_baseline['WR']:>12} {summary['win_rate_pct']:>12}")
    print(f"  {'PnL 1x %':15s} {normal_baseline['PnL_1x']:>12} {summary['total_pnl_1x']:>12}")
    print(f"  {'MDD 1x %':15s} {normal_baseline['MDD_1x']:>12} {summary['mdd_1x']:>12}")
    print(f"  {'RR':15s} {normal_baseline['RR']:>12} {summary['risk_reward']:>12}")
    print(f"  {'daily_pnl 1x':15s} {normal_baseline['daily_pnl']:>12} {round(daily_pnl,3):>12}")
    print(f"  {'trades/day':15s} {normal_baseline['trades_per_day']:>12} {round(tpd,1):>12}")

    # BTC context
    btc_ret = (closes[-1] / closes[warmup] - 1) * 100
    print()
    print(f"BTC market context: return {btc_ret:+.2f}% over {days} days "
          f"(${closes[warmup]:.0f} → ${closes[-1]:.0f})")

    out = {
        'metadata': {
            'script': 'c1_reverse_full_backtest.py',
            'date_run': datetime.now().isoformat(),
            'total_bars': n, 'days': days,
            'btc_return_pct': round(btc_ret, 2),
        },
        'summary': summary,
        'daily_pnl_1x': round(daily_pnl, 4),
        'trades_per_day': round(tpd, 2),
        'folds_pnl': folds_pnl,
        'folds_passed': passed,
    }
    with open(ROOT / 'results' / 'c1_reverse_full_backtest.json', 'w') as f:
        json.dump(out, f, indent=2, default=str)


if __name__ == '__main__':
    main()
