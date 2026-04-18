"""
Live vs Backtest Trade-by-Trade Verification
=============================================
Fetches BTC/USDT 15m candles from BingX for the exact live trading period
(April 10-16, 2026) and runs the C1 Breakout v2 backtest.
Compares every backtest trade 1:1 with actual live trades.

Uses production signal/indicator code directly — no reimplementation.
"""

import sys
import os
import json
import math
from datetime import datetime, timezone

# Setup path to import production modules
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, REPO_ROOT)

from scripts.production.c1_breakout.indicators import (
    compute_atr, compute_channel, compute_fractal_swings
)
from scripts.production.c1_breakout.signals import C1BreakoutSignal

# ─── Configuration (must match production exactly) ────────────────────────
CONFIG = {
    'channel_period': 15,
    'body_min_ratio': 0.4,
    'atr_period': 14,
    'trail_K': 2.5,
    'max_sl_atr': 3.3,
    'emergency_sl_pct': 3.0,
    'max_hold_bars': 192,
    'sl_min_pct': 0.15,
    'sl_max_pct': 3.0,
    'min_bars_between': 2,
    'trail_activation_pct': 0.05,
}

LEVERAGE = 3
FEE_RT_PCT = 0.10  # Round-trip fee (taker 0.05% x 2)
FRACTAL_LOOKBACK = 10

# ─── Live trades to compare (v2.6, 3x leverage) ──────────────────────────
LIVE_TRADES = [
    {'id': 2, 'dir': 'SHORT', 'entry': 70560.0, 'exit': 71080.3, 'pnl_pct': -2.51,
     'reason': 'EXCHANGE_SL', 'bars': 16, 'date': 'Apr12'},
    {'id': 3, 'dir': 'LONG', 'entry': 71668.6, 'exit': 74026.6, 'pnl_pct': 9.57,
     'reason': 'EXCHANGE_TRAIL', 'bars': 44, 'date': 'Apr13-14'},
    {'id': 4, 'dir': 'SHORT', 'entry': 74226.0, 'exit': 74782.5, 'pnl_pct': -2.55,
     'reason': 'EXCHANGE_SL', 'bars': 4, 'date': 'Apr14'},
    {'id': 5, 'dir': 'SHORT', 'entry': 74300.0, 'exit': 74630.0, 'pnl_pct': -1.63,
     'reason': 'EXCHANGE_TRAIL', 'bars': 10, 'date': 'Apr14'},
    {'id': 6, 'dir': 'LONG', 'entry': 75112.0, 'exit': 75448.0, 'pnl_pct': 1.04,
     'reason': 'EXCHANGE_TRAIL', 'bars': 3, 'date': 'Apr14'},
    {'id': 7, 'dir': 'LONG', 'entry': 74521.9, 'exit': 74521.9, 'pnl_pct': -0.30,
     'reason': 'TRAIL_TP', 'bars': 12, 'date': 'Apr15'},
    {'id': 8, 'dir': 'SHORT', 'entry': 73948.3, 'exit': 73948.3, 'pnl_pct': -0.30,
     'reason': 'TRAIL_TP', 'bars': 12, 'date': 'Apr15'},
    {'id': 9, 'dir': 'LONG', 'entry': 74305.7, 'exit': 73854.8, 'pnl_pct': -2.12,
     'reason': 'EXCHANGE_TRAIL', 'bars': 3, 'date': 'Apr15'},
    {'id': 10, 'dir': 'LONG', 'entry': 74361.5, 'exit': 74622.4, 'pnl_pct': 0.75,
     'reason': 'EXCHANGE_TRAIL', 'bars': 4, 'date': 'Apr15-16'},
]


def fetch_candles():
    """Fetch BTC/USDT 15m candles from BingX via CCXT."""
    import ccxt

    exchange = ccxt.bingx({'options': {'defaultType': 'swap'}})

    # April 10 00:00 UTC to April 16 23:59 UTC
    start_ts = int(datetime(2026, 4, 10, 0, 0, tzinfo=timezone.utc).timestamp() * 1000)
    end_ts = int(datetime(2026, 4, 16, 23, 59, tzinfo=timezone.utc).timestamp() * 1000)

    all_candles = []
    since = start_ts

    while since < end_ts:
        candles = exchange.fetch_ohlcv('BTC-USDT', '15m', since=since, limit=1000)
        if not candles:
            break
        all_candles.extend(candles)
        last_ts = candles[-1][0]
        if last_ts <= since:
            break
        since = last_ts + 1  # next after last

    # Deduplicate by timestamp
    seen = set()
    unique = []
    for c in all_candles:
        if c[0] not in seen:
            seen.add(c[0])
            unique.append(c)

    # Filter to date range
    unique = [c for c in unique if start_ts <= c[0] <= end_ts]
    unique.sort(key=lambda c: c[0])

    print(f"Fetched {len(unique)} candles from BingX")
    print(f"  First: {datetime.fromtimestamp(unique[0][0]/1000, tz=timezone.utc)}")
    print(f"  Last:  {datetime.fromtimestamp(unique[-1][0]/1000, tz=timezone.utc)}")

    return unique


def run_backtest(candles):
    """Run C1 Breakout v2 backtest on the candle data.

    Returns list of trade dicts.
    """
    signal_gen = C1BreakoutSignal(CONFIG)

    # Extract OHLCV arrays
    timestamps = [c[0] for c in candles]
    opens = [c[1] for c in candles]
    highs = [c[2] for c in candles]
    lows = [c[3] for c in candles]
    closes = [c[4] for c in candles]
    n = len(candles)

    # Compute indicators (all at once, causal)
    atr = compute_atr(highs, lows, closes, CONFIG['atr_period'])
    ch_high, ch_low = compute_channel(highs, lows, CONFIG['channel_period'])
    sw_low, sw_high = compute_fractal_swings(highs, lows, FRACTAL_LOOKBACK)

    trades = []
    position = None  # {'dir', 'entry_price', 'sl_price', 'best_price', 'entry_bar', 'signal_bar'}
    last_exit_bar = -999  # For min_bars_between cooldown

    # Only process bars in the live trading window (April 12 onward)
    # but compute signals from the start for warmup
    apr12_start = int(datetime(2026, 4, 12, 0, 0, tzinfo=timezone.utc).timestamp() * 1000)

    for i in range(1, n - 1):  # Need i+1 for entry
        dt = datetime.fromtimestamp(timestamps[i] / 1000, tz=timezone.utc)

        if position is not None:
            # ─── Manage open position ───
            entry_bar = position['entry_bar']
            bars_held = i - entry_bar  # bars since entry bar

            # Update best_price with current bar's extreme
            if position['dir'] == 'LONG':
                position['best_price'] = max(position['best_price'], highs[i])
            else:
                position['best_price'] = min(position['best_price'], lows[i])

            # Check exit
            exit_result = signal_gen.check_exit(
                direction=position['dir'],
                entry_price=position['entry_price'],
                best_price=position['best_price'],
                current_high=highs[i],
                current_low=lows[i],
                current_close=closes[i],
                sl_price=position['sl_price'],
                atr_val=atr[i] if not math.isnan(atr[i]) else atr[i-1],
                bars_held=bars_held,
            )

            if exit_result is not None:
                exit_price = exit_result['exit_price']
                reason = exit_result['reason']

                # Compute PnL (additive 1x, no leverage in raw calc)
                if position['dir'] == 'LONG':
                    raw_pnl = (exit_price / position['entry_price'] - 1) * 100
                else:
                    raw_pnl = (1 - exit_price / position['entry_price']) * 100

                # With leverage and fees
                pnl_lev = raw_pnl * LEVERAGE
                pnl_net = pnl_lev - FEE_RT_PCT * LEVERAGE

                trade = {
                    'signal_bar_ts': timestamps[position['signal_bar']],
                    'signal_bar_dt': datetime.fromtimestamp(
                        timestamps[position['signal_bar']] / 1000, tz=timezone.utc
                    ).strftime('%Y-%m-%d %H:%M'),
                    'entry_bar_ts': timestamps[entry_bar],
                    'entry_bar_dt': datetime.fromtimestamp(
                        timestamps[entry_bar] / 1000, tz=timezone.utc
                    ).strftime('%Y-%m-%d %H:%M'),
                    'exit_bar_ts': timestamps[i],
                    'exit_bar_dt': datetime.fromtimestamp(
                        timestamps[i] / 1000, tz=timezone.utc
                    ).strftime('%Y-%m-%d %H:%M'),
                    'dir': position['dir'],
                    'entry_price': round(position['entry_price'], 1),
                    'exit_price': round(exit_price, 1),
                    'sl_price': round(position['sl_price'], 1),
                    'raw_pnl_pct': round(raw_pnl, 3),
                    'pnl_3x_pct': round(pnl_net, 2),
                    'reason': reason,
                    'bars_held': bars_held,
                    'best_price': round(position['best_price'], 1),
                }
                trades.append(trade)
                last_exit_bar = i
                position = None

        # ─── Check for new entry signal ───
        if position is None and timestamps[i] >= apr12_start:
            # Cooldown check
            if i - last_exit_bar < CONFIG['min_bars_between']:
                continue

            # Need valid indicators
            if math.isnan(ch_high[i]) or math.isnan(ch_low[i]) or math.isnan(atr[i]):
                continue
            if math.isnan(sw_low[i]) or math.isnan(sw_high[i]):
                continue

            entry_signal = signal_gen.check_entry(
                bar_open=opens[i],
                bar_high=highs[i],
                bar_low=lows[i],
                bar_close=closes[i],
                channel_high=ch_high[i],
                channel_low=ch_low[i],
                atr_val=atr[i],
                last_swing_low=sw_low[i],
                last_swing_high=sw_high[i],
            )

            if entry_signal is not None:
                # Entry at next bar open
                entry_bar_idx = i + 1
                entry_price = opens[entry_bar_idx]

                # best_price starts at entry price, then update with entry bar extreme
                if entry_signal['direction'] == 'LONG':
                    best_price = max(entry_price, highs[entry_bar_idx])
                else:
                    best_price = min(entry_price, lows[entry_bar_idx])

                position = {
                    'dir': entry_signal['direction'],
                    'entry_price': entry_price,
                    'sl_price': entry_signal['sl_price'],
                    'best_price': best_price,
                    'entry_bar': entry_bar_idx,
                    'signal_bar': i,
                }

                # Check exit on entry bar itself (SL hit on entry bar)
                exit_result = signal_gen.check_exit(
                    direction=position['dir'],
                    entry_price=position['entry_price'],
                    best_price=position['best_price'],
                    current_high=highs[entry_bar_idx],
                    current_low=lows[entry_bar_idx],
                    current_close=closes[entry_bar_idx],
                    sl_price=position['sl_price'],
                    atr_val=atr[entry_bar_idx] if not math.isnan(atr[entry_bar_idx]) else atr[i],
                    bars_held=1,
                )

                if exit_result is not None:
                    exit_price = exit_result['exit_price']
                    reason = exit_result['reason']

                    if position['dir'] == 'LONG':
                        raw_pnl = (exit_price / position['entry_price'] - 1) * 100
                    else:
                        raw_pnl = (1 - exit_price / position['entry_price']) * 100

                    pnl_lev = raw_pnl * LEVERAGE
                    pnl_net = pnl_lev - FEE_RT_PCT * LEVERAGE

                    trade = {
                        'signal_bar_ts': timestamps[i],
                        'signal_bar_dt': datetime.fromtimestamp(
                            timestamps[i] / 1000, tz=timezone.utc
                        ).strftime('%Y-%m-%d %H:%M'),
                        'entry_bar_ts': timestamps[entry_bar_idx],
                        'entry_bar_dt': datetime.fromtimestamp(
                            timestamps[entry_bar_idx] / 1000, tz=timezone.utc
                        ).strftime('%Y-%m-%d %H:%M'),
                        'exit_bar_ts': timestamps[entry_bar_idx],
                        'exit_bar_dt': datetime.fromtimestamp(
                            timestamps[entry_bar_idx] / 1000, tz=timezone.utc
                        ).strftime('%Y-%m-%d %H:%M'),
                        'dir': position['dir'],
                        'entry_price': round(position['entry_price'], 1),
                        'exit_price': round(exit_price, 1),
                        'sl_price': round(position['sl_price'], 1),
                        'raw_pnl_pct': round(raw_pnl, 3),
                        'pnl_3x_pct': round(pnl_net, 2),
                        'reason': reason,
                        'bars_held': 1,
                        'best_price': round(position['best_price'], 1),
                    }
                    trades.append(trade)
                    last_exit_bar = entry_bar_idx
                    position = None

    # Handle still-open position at end of data
    if position is not None:
        i = n - 1
        exit_price = closes[i]
        if position['dir'] == 'LONG':
            raw_pnl = (exit_price / position['entry_price'] - 1) * 100
        else:
            raw_pnl = (1 - exit_price / position['entry_price']) * 100
        pnl_lev = raw_pnl * LEVERAGE
        pnl_net = pnl_lev - FEE_RT_PCT * LEVERAGE

        trade = {
            'signal_bar_ts': timestamps[position['signal_bar']],
            'signal_bar_dt': datetime.fromtimestamp(
                timestamps[position['signal_bar']] / 1000, tz=timezone.utc
            ).strftime('%Y-%m-%d %H:%M'),
            'entry_bar_ts': timestamps[position['entry_bar']],
            'entry_bar_dt': datetime.fromtimestamp(
                timestamps[position['entry_bar']] / 1000, tz=timezone.utc
            ).strftime('%Y-%m-%d %H:%M'),
            'exit_bar_ts': timestamps[i],
            'exit_bar_dt': datetime.fromtimestamp(
                timestamps[i] / 1000, tz=timezone.utc
            ).strftime('%Y-%m-%d %H:%M'),
            'dir': position['dir'],
            'entry_price': round(position['entry_price'], 1),
            'exit_price': round(exit_price, 1),
            'sl_price': round(position['sl_price'], 1),
            'raw_pnl_pct': round(raw_pnl, 3),
            'pnl_3x_pct': round(pnl_net, 2),
            'reason': 'OPEN_EOD',
            'bars_held': i - position['entry_bar'],
            'best_price': round(position['best_price'], 1),
        }
        trades.append(trade)
        position = None

    return trades


def match_trades(bt_trades, live_trades):
    """Match backtest trades to live trades by direction and approximate entry time/price."""
    matches = []
    used_live = set()

    for bt in bt_trades:
        best_match = None
        best_score = float('inf')

        for j, lt in enumerate(live_trades):
            if j in used_live:
                continue
            if bt['dir'] != lt['dir']:
                continue

            # Score by entry price difference
            entry_diff_pct = abs(bt['entry_price'] - lt['entry']) / lt['entry'] * 100
            score = entry_diff_pct

            if score < best_score and score < 1.0:  # Max 1% entry diff
                best_score = score
                best_match = j

        if best_match is not None:
            used_live.add(best_match)
            lt = live_trades[best_match]
            entry_diff = (bt['entry_price'] - lt['entry']) / lt['entry'] * 100
            exit_diff = (bt['exit_price'] - lt['exit']) / lt['exit'] * 100
            pnl_diff = bt['pnl_3x_pct'] - lt['pnl_pct']

            # Reason match (BT uses SL/TRAIL_TP, live uses EXCHANGE_SL/EXCHANGE_TRAIL)
            reason_map = {
                'SL': 'EXCHANGE_SL',
                'TRAIL_TP': ['EXCHANGE_TRAIL', 'TRAIL_TP'],
            }
            bt_reason = bt['reason']
            lt_reason = lt['reason']
            if bt_reason in reason_map:
                expected = reason_map[bt_reason]
                if isinstance(expected, list):
                    reason_ok = lt_reason in expected
                else:
                    reason_ok = lt_reason == expected
            else:
                reason_ok = bt_reason == lt_reason

            matches.append({
                'bt_trade': bt,
                'live_trade': lt,
                'entry_diff_pct': round(entry_diff, 4),
                'exit_diff_pct': round(exit_diff, 4),
                'pnl_diff_pct': round(pnl_diff, 2),
                'reason_match': reason_ok,
                'direction_match': True,
            })
        else:
            matches.append({
                'bt_trade': bt,
                'live_trade': None,
                'note': 'NO LIVE MATCH — backtest-only signal',
            })

    # Check for unmatched live trades
    unmatched_live = []
    for j, lt in enumerate(live_trades):
        if j not in used_live:
            unmatched_live.append(lt)

    return matches, unmatched_live


def print_comparison(matches, unmatched_live, bt_trades):
    """Print detailed comparison table."""
    print("\n" + "=" * 130)
    print("LIVE vs BACKTEST TRADE-BY-TRADE COMPARISON")
    print("=" * 130)

    # Header
    fmt = "{:<4} {:<6} {:>10} {:>10} {:>8} {:>10} {:>10} {:>8} {:>8} {:>8} {:>10} {:>6}"
    print(fmt.format(
        "BT#", "Dir", "BT_Entry", "Liv_Entry", "Ent_Diff",
        "BT_Exit", "Liv_Exit", "Ext_Diff", "BT_PnL", "Liv_PnL", "BT_Reason", "Match"
    ))
    print("-" * 130)

    matched_count = 0
    total_entry_diff = 0
    total_pnl_diff = 0
    n_matched = 0

    for idx, m in enumerate(matches):
        bt = m['bt_trade']
        if m.get('live_trade') is not None:
            lt = m['live_trade']
            entry_diff_str = f"{m['entry_diff_pct']:+.3f}%"
            exit_diff_str = f"{m['exit_diff_pct']:+.3f}%"
            match_str = "Y" if m['reason_match'] else "~"
            if abs(m['entry_diff_pct']) < 0.5 and m['direction_match']:
                matched_count += 1

            total_entry_diff += abs(m['entry_diff_pct'])
            total_pnl_diff += abs(m['pnl_diff_pct'])
            n_matched += 1

            print(fmt.format(
                idx + 1, bt['dir'],
                f"{bt['entry_price']:.1f}", f"{lt['entry']:.1f}", entry_diff_str,
                f"{bt['exit_price']:.1f}", f"{lt['exit']:.1f}", exit_diff_str,
                f"{bt['pnl_3x_pct']:+.2f}%", f"{lt['pnl_pct']:+.2f}%",
                bt['reason'], match_str,
            ))
        else:
            print(fmt.format(
                idx + 1, bt['dir'],
                f"{bt['entry_price']:.1f}", "---", "---",
                f"{bt['exit_price']:.1f}", "---", "---",
                f"{bt['pnl_3x_pct']:+.2f}%", "---",
                bt['reason'], "BT_ONLY",
            ))

    # Unmatched live trades
    if unmatched_live:
        print("\n--- UNMATCHED LIVE TRADES (in live but not in backtest) ---")
        for lt in unmatched_live:
            print(f"  Live #{lt['id']}: {lt['dir']} entry={lt['entry']:.1f} "
                  f"exit={lt['exit']:.1f} pnl={lt['pnl_pct']:+.2f}% reason={lt['reason']}")

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Backtest trades:         {len(bt_trades)}")
    print(f"Live trades:             {len(LIVE_TRADES)}")
    print(f"Matched (same dir+bar):  {matched_count}/{min(len(bt_trades), len(LIVE_TRADES))}")
    if n_matched > 0:
        print(f"Avg entry price diff:    {total_entry_diff / n_matched:.4f}%")
        print(f"Avg |PnL diff| (3x):     {total_pnl_diff / n_matched:.2f}pp")

    # Verdict
    match_rate = matched_count / max(1, min(len(bt_trades), len(LIVE_TRADES)))
    if match_rate >= 0.8:
        verdict = "MATCH"
    elif match_rate >= 0.5:
        verdict = "PARTIAL"
    else:
        verdict = "MISMATCH"

    print(f"\nVerdict: {verdict} ({match_rate:.0%} match rate)")

    # Print all backtest trade details
    print("\n" + "=" * 80)
    print("ALL BACKTEST TRADES (detailed)")
    print("=" * 80)
    for idx, t in enumerate(bt_trades):
        print(f"\n  BT #{idx+1}: {t['dir']}")
        print(f"    Signal:  {t['signal_bar_dt']} UTC")
        print(f"    Entry:   {t['entry_bar_dt']} UTC @ {t['entry_price']:.1f}")
        print(f"    Exit:    {t['exit_bar_dt']} UTC @ {t['exit_price']:.1f}")
        print(f"    SL:      {t['sl_price']:.1f}")
        print(f"    Best:    {t['best_price']:.1f}")
        print(f"    PnL:     {t['raw_pnl_pct']:+.3f}% (1x) → {t['pnl_3x_pct']:+.2f}% (3x net)")
        print(f"    Reason:  {t['reason']}  |  Bars held: {t['bars_held']}")

    return verdict, match_rate


def save_results(matches, unmatched_live, bt_trades, verdict, match_rate):
    """Save results to JSON."""
    results = {
        'metadata': {
            'script': 'live_vs_backtest_verification.py',
            'date': datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC'),
            'strategy': 'C1 Breakout v2.6',
            'leverage': LEVERAGE,
            'fee_rt_pct': FEE_RT_PCT,
            'data_range': 'Apr 10-16, 2026 (15m BTC/USDT from BingX)',
        },
        'summary': {
            'bt_trade_count': len(bt_trades),
            'live_trade_count': len(LIVE_TRADES),
            'matched_count': sum(1 for m in matches if m.get('live_trade') is not None
                                 and abs(m.get('entry_diff_pct', 999)) < 0.5),
            'unmatched_bt_signals': sum(1 for m in matches if m.get('live_trade') is None),
            'unmatched_live_trades': len(unmatched_live),
            'verdict': verdict,
            'match_rate': round(match_rate, 4),
        },
        'backtest_trades': bt_trades,
        'matches': [
            {
                'bt_entry': m['bt_trade']['entry_price'],
                'bt_exit': m['bt_trade']['exit_price'],
                'bt_dir': m['bt_trade']['dir'],
                'bt_reason': m['bt_trade']['reason'],
                'bt_pnl_3x': m['bt_trade']['pnl_3x_pct'],
                'bt_signal_dt': m['bt_trade']['signal_bar_dt'],
                'bt_entry_dt': m['bt_trade']['entry_bar_dt'],
                'bt_bars': m['bt_trade']['bars_held'],
                'live_id': m['live_trade']['id'] if m.get('live_trade') else None,
                'live_entry': m['live_trade']['entry'] if m.get('live_trade') else None,
                'live_exit': m['live_trade']['exit'] if m.get('live_trade') else None,
                'live_pnl': m['live_trade']['pnl_pct'] if m.get('live_trade') else None,
                'live_reason': m['live_trade']['reason'] if m.get('live_trade') else None,
                'entry_diff_pct': m.get('entry_diff_pct'),
                'exit_diff_pct': m.get('exit_diff_pct'),
                'pnl_diff_pct': m.get('pnl_diff_pct'),
                'reason_match': m.get('reason_match'),
                'note': m.get('note'),
            }
            for m in matches
        ],
        'unmatched_live': unmatched_live,
    }

    out_path = os.path.join(REPO_ROOT, 'results', 'live_vs_backtest_verification.json')
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to {out_path}")
    return out_path


def main():
    print("=" * 80)
    print("C1 Breakout v2 — Live vs Backtest Verification")
    print("=" * 80)

    # 1. Fetch candles
    print("\n[1] Fetching 15m candles from BingX...")
    candles = fetch_candles()

    # 2. Run backtest
    print(f"\n[2] Running backtest on {len(candles)} bars...")
    bt_trades = run_backtest(candles)
    print(f"    Backtest produced {len(bt_trades)} trades")

    # 3. Match and compare
    print(f"\n[3] Comparing against {len(LIVE_TRADES)} live trades...")
    matches, unmatched_live = match_trades(bt_trades, LIVE_TRADES)

    # 4. Print comparison
    verdict, match_rate = print_comparison(matches, unmatched_live, bt_trades)

    # 5. Save results
    save_results(matches, unmatched_live, bt_trades, verdict, match_rate)


if __name__ == '__main__':
    main()
