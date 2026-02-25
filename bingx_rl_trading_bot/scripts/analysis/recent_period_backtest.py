"""
Recent Period Backtest: v1.33 (35pat Compact) vs v1.35 (51pat ATR)
=================================================================
Feb 23-25 구간에서 두 패턴셋 비교.
- v1.33: 9L+26S, Compact TP/SL (TP max 2.0%, SL max 2.5%)
- v1.35: 16L+35S, ATR-scaled TP/SL (ATR14/median576, clamp [0.6,1.7])

Standard Research Protocol:
- Entry: signal bar+1 open
- Exit: intrabar high/low (distance-based)
- Fee: 0.05% x2 = 0.10% (x leverage in capital space)
- Slippage buffer: 0.02%
- Leverage: 3x
- ATR scaling: ATR(14)/rolling_median(576), clamp [0.6, 1.7]
- Multi-position: N=9, 1/N sizing, direction cap 6
- Timeout: 864 bars (72h)
"""
import sys, os, json
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from scripts.production.pattern_5m.indicators import calculate_indicators
from scripts.production.pattern_5m.constants import CandleType
from scripts.scanner.pattern_scanner import classify_candle

# ─── Constants ───
LEVERAGE = 3
FEE_PCT = 0.10   # 0.05% x 2 (roundtrip)
SLIPPAGE_BUFFER = 0.02
MAX_BARS = 288
TIMEOUT_BARS = 864
MAX_POSITIONS = 9
DIRECTION_CAP = 6
ATR_PERIOD = 14
ATR_WINDOW = 576
CLAMP_LO = 0.6
CLAMP_HI = 1.7

# ─── Data Fetching ───
def fetch_ohlcv(symbol='BTC-USDT', timeframe='5m', limit=4000):
    """Fetch OHLCV from BingX via ccxt."""
    import yaml
    cfg_path = os.path.join(os.path.dirname(__file__), '..', '..', 'config', 'api_keys.yaml')
    with open(cfg_path) as f:
        keys = yaml.safe_load(f)

    bingx_keys = keys['bingx']['mainnet']
    import ccxt
    ex = ccxt.bingx({
        'apiKey': bingx_keys['api_key'],
        'secret': bingx_keys['secret_key'],
        'options': {'defaultType': 'swap'}
    })

    all_bars = []
    # Fetch backwards from now
    end_ts = None
    remaining = limit
    while remaining > 0:
        params = {}
        if end_ts is not None:
            params['endTime'] = end_ts
        batch = ex.fetch_ohlcv(symbol, timeframe, limit=min(1000, remaining), params=params)
        if not batch:
            break
        all_bars = batch + all_bars
        end_ts = batch[0][0] - 1
        remaining -= len(batch)
        if len(batch) < 100:
            break

    df = pd.DataFrame(all_bars, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df = df.drop_duplicates(subset='timestamp').sort_values('timestamp').reset_index(drop=True)
    df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms')
    return df


def compute_atr_ratio(highs, lows, closes, period=ATR_PERIOD, window=ATR_WINDOW):
    """ATR / rolling_median(ATR) ratio."""
    n = len(closes)
    tr = np.empty(n)
    tr[0] = highs[0] - lows[0]
    for i in range(1, n):
        tr[i] = max(highs[i] - lows[i],
                     abs(highs[i] - closes[i-1]),
                     abs(lows[i] - closes[i-1]))
    atr = pd.Series(tr).rolling(period, min_periods=period).mean().values
    med = pd.Series(atr).rolling(window, min_periods=window).median().values
    ratio = np.full(n, np.nan)
    valid = (~np.isnan(atr)) & (~np.isnan(med)) & (med > 0)
    ratio[valid] = atr[valid] / med[valid]
    return ratio


def classify_candles(df):
    """Use production classify_candle (same as scanner)."""
    df = df.copy()
    df['body'] = df['close'] - df['open']
    df['body_abs'] = df['body'].abs()
    df['avg_body_20'] = df['body_abs'].rolling(20, min_periods=1).mean()
    rctypes = []
    for _, row in df.iterrows():
        ct = classify_candle(row, row['avg_body_20'])
        rctypes.append(ct.value)
    return rctypes


# ─── Pattern Signal Generation ───
def generate_signals(rctypes, patterns_long, patterns_short):
    """Generate signals from 3-candle patterns. Returns list of (bar_idx, pattern, direction)."""
    signals = []
    n = len(rctypes)
    for i in range(2, n):
        pat = f"{rctypes[i-2]}-{rctypes[i-1]}-{rctypes[i]}"
        if pat in patterns_long:
            signals.append((i, pat, 'LONG'))
        if pat in patterns_short:
            signals.append((i, pat, 'SHORT'))
    return signals


# ─── Single Trade Simulation ───
def simulate_trade(signal_bar, direction, tp_pct, sl_pct, opens, highs, lows, n_bars,
                   atr_ratio=None, use_atr=False):
    """Simulate a single trade with optional ATR scaling.
    Returns dict with trade details or None if no entry possible."""
    entry_bar = signal_bar + 1
    if entry_bar >= n_bars:
        return None

    entry = opens[entry_bar]
    if entry <= 0:
        return None

    # ATR scaling
    if use_atr and atr_ratio is not None and not np.isnan(atr_ratio[signal_bar]):
        r = max(CLAMP_LO, min(CLAMP_HI, atr_ratio[signal_bar]))
    else:
        r = 1.0

    eff_tp_pct = tp_pct * r + SLIPPAGE_BUFFER
    eff_sl_pct = max(0.1, sl_pct * r - SLIPPAGE_BUFFER)

    if direction == 'LONG':
        tp_price = entry * (1 + eff_tp_pct / 100)
        sl_price = entry * (1 - eff_sl_pct / 100)
    else:
        tp_price = entry * (1 - eff_tp_pct / 100)
        sl_price = entry * (1 + eff_sl_pct / 100)

    # Scan forward for exit
    for j in range(entry_bar, min(entry_bar + TIMEOUT_BARS, n_bars)):
        h, l = highs[j], lows[j]

        if direction == 'LONG':
            hit_tp = h >= tp_price
            hit_sl = l <= sl_price
        else:
            hit_tp = l <= tp_price
            hit_sl = h >= sl_price

        if hit_tp and hit_sl:
            # Same-bar: distance from open
            if direction == 'LONG':
                tp_dist = abs(tp_price - opens[j])
                sl_dist = abs(sl_price - opens[j])
            else:
                tp_dist = abs(tp_price - opens[j])
                sl_dist = abs(sl_price - opens[j])
            if tp_dist <= sl_dist:
                exit_price = tp_price
                reason = 'TP'
            else:
                exit_price = sl_price
                reason = 'SL'
        elif hit_tp:
            exit_price = tp_price
            reason = 'TP'
        elif hit_sl:
            exit_price = sl_price
            reason = 'SL'
        else:
            continue

        # PnL calculation (leveraged, capital-space)
        if direction == 'LONG':
            pnl_pct = (exit_price / entry - 1) * 100 * LEVERAGE
        else:
            pnl_pct = (1 - exit_price / entry) * 100 * LEVERAGE
        pnl_pct -= FEE_PCT * LEVERAGE  # fee on notional

        return {
            'entry_bar': entry_bar,
            'exit_bar': j,
            'hold_bars': j - entry_bar,
            'entry_price': entry,
            'exit_price': exit_price,
            'tp_price': tp_price,
            'sl_price': sl_price,
            'atr_mult': r,
            'pnl_slot': pnl_pct,
            'reason': reason,
        }

    # Timeout — exit at market
    if entry_bar + TIMEOUT_BARS < n_bars:
        j = entry_bar + TIMEOUT_BARS
        exit_price = opens[j]
        if direction == 'LONG':
            pnl_pct = (exit_price / entry - 1) * 100 * LEVERAGE
        else:
            pnl_pct = (1 - exit_price / entry) * 100 * LEVERAGE
        pnl_pct -= FEE_PCT * LEVERAGE
        return {
            'entry_bar': entry_bar,
            'exit_bar': j,
            'hold_bars': TIMEOUT_BARS,
            'entry_price': entry,
            'exit_price': exit_price,
            'tp_price': tp_price,
            'sl_price': sl_price,
            'atr_mult': r,
            'pnl_slot': pnl_pct,
            'reason': 'TIMEOUT',
        }

    return None  # still open


# ─── Multi-Position Portfolio Simulation ───
def simulate_portfolio(signals, direction_map, tpsl_map, opens, highs, lows, n_bars,
                       atr_ratio=None, use_atr=False, label='',
                       direction_cap_override=None):
    """Simulate N=9 multi-position portfolio with direction cap."""
    dir_cap = direction_cap_override if direction_cap_override is not None else DIRECTION_CAP
    positions = []  # active: {slot, signal_bar, direction, pattern, tp_pct, sl_pct, ...}
    trades = []
    SIZE_PCT = 100.0 / MAX_POSITIONS  # 11.1% per slot

    # Sort signals by bar index
    signals_sorted = sorted(signals, key=lambda x: x[0])
    sig_idx = 0

    for bar in range(n_bars):
        # Check exits for active positions
        closed_slots = []
        for pos in positions:
            result = _check_exit(pos, bar, opens, highs, lows, n_bars, atr_ratio, use_atr)
            if result is not None:
                result['pattern'] = pos['pattern']
                result['direction'] = pos['direction']
                result['pnl_portfolio'] = result['pnl_slot'] * SIZE_PCT / 100
                result['signal_bar'] = pos['signal_bar']
                trades.append(result)
                closed_slots.append(pos['slot'])

        positions = [p for p in positions if p['slot'] not in closed_slots]

        # Process signals at this bar
        while sig_idx < len(signals_sorted) and signals_sorted[sig_idx][0] == bar:
            sig_bar, pat, direction = signals_sorted[sig_idx]
            sig_idx += 1

            # Skip if max positions reached
            if len(positions) >= MAX_POSITIONS:
                continue

            # Direction cap
            dir_count = sum(1 for p in positions if p['direction'] == direction)
            if dir_count >= dir_cap:
                continue

            # No duplicate pattern
            if any(p['pattern'] == pat for p in positions):
                continue

            # Entry on next bar
            entry_bar = sig_bar + 1
            if entry_bar >= n_bars:
                continue

            tp_sl = tpsl_map.get(pat)
            if tp_sl is None:
                continue

            tp_pct, sl_pct = tp_sl[0], tp_sl[1]

            positions.append({
                'slot': f"{pat}_{sig_bar}",
                'signal_bar': sig_bar,
                'entry_bar': entry_bar,
                'direction': direction,
                'pattern': pat,
                'tp_pct': tp_pct,
                'sl_pct': sl_pct,
            })

    return trades


def _check_exit(pos, bar, opens, highs, lows, n_bars, atr_ratio, use_atr):
    """Check if position should exit at this bar."""
    entry_bar = pos['entry_bar']
    if bar < entry_bar:
        return None

    # Not yet entered
    if bar == entry_bar - 1:
        return None

    entry = opens[entry_bar]
    if entry <= 0:
        return None

    tp_pct = pos['tp_pct']
    sl_pct = pos['sl_pct']
    direction = pos['direction']

    # ATR scaling at signal time
    sig_bar = pos['signal_bar']
    if use_atr and atr_ratio is not None and not np.isnan(atr_ratio[sig_bar]):
        r = max(CLAMP_LO, min(CLAMP_HI, atr_ratio[sig_bar]))
    else:
        r = 1.0

    eff_tp = tp_pct * r + SLIPPAGE_BUFFER
    eff_sl = max(0.1, sl_pct * r - SLIPPAGE_BUFFER)

    if direction == 'LONG':
        tp_price = entry * (1 + eff_tp / 100)
        sl_price = entry * (1 - eff_sl / 100)
    else:
        tp_price = entry * (1 - eff_tp / 100)
        sl_price = entry * (1 + eff_sl / 100)

    # Timeout check
    hold = bar - entry_bar
    if hold >= TIMEOUT_BARS:
        exit_price = opens[bar]
        if direction == 'LONG':
            pnl = (exit_price / entry - 1) * 100 * LEVERAGE
        else:
            pnl = (1 - exit_price / entry) * 100 * LEVERAGE
        pnl -= FEE_PCT * LEVERAGE
        return {'entry_bar': entry_bar, 'exit_bar': bar, 'hold_bars': hold,
                'entry_price': entry, 'exit_price': exit_price,
                'tp_price': tp_price, 'sl_price': sl_price,
                'atr_mult': r, 'pnl_slot': pnl, 'reason': 'TIMEOUT'}

    h, l = highs[bar], lows[bar]
    if direction == 'LONG':
        hit_tp = h >= tp_price
        hit_sl = l <= sl_price
    else:
        hit_tp = l <= tp_price
        hit_sl = h >= sl_price

    if not hit_tp and not hit_sl:
        return None

    if hit_tp and hit_sl:
        tp_dist = abs(tp_price - opens[bar])
        sl_dist = abs(sl_price - opens[bar])
        if tp_dist <= sl_dist:
            exit_price, reason = tp_price, 'TP'
        else:
            exit_price, reason = sl_price, 'SL'
    elif hit_tp:
        exit_price, reason = tp_price, 'TP'
    else:
        exit_price, reason = sl_price, 'SL'

    if direction == 'LONG':
        pnl = (exit_price / entry - 1) * 100 * LEVERAGE
    else:
        pnl = (1 - exit_price / entry) * 100 * LEVERAGE
    pnl -= FEE_PCT * LEVERAGE

    return {'entry_bar': entry_bar, 'exit_bar': bar, 'hold_bars': hold,
            'entry_price': entry, 'exit_price': exit_price,
            'tp_price': tp_price, 'sl_price': sl_price,
            'atr_mult': r, 'pnl_slot': pnl, 'reason': reason}


def _track_signals(signals, patterns_L, patterns_S, tpsl_map, opens, highs, lows, n_bars,
                    atr_ratio, df):
    """Replay signal processing with full blocking reason tracking."""
    positions = []
    SIZE_PCT = 100.0 / MAX_POSITIONS
    signals_sorted = sorted(signals, key=lambda x: x[0])
    traded_pats = set()

    long_signals = sum(1 for s in signals_sorted if s[2] == 'LONG')
    short_signals = sum(1 for s in signals_sorted if s[2] == 'SHORT')
    print(f"  Total signals: {len(signals_sorted)} (LONG={long_signals}, SHORT={short_signals})")

    blocked = {'max_pos': 0, 'dir_cap': 0, 'dup_pat': 0, 'no_tpsl': 0}
    entered = {'LONG': 0, 'SHORT': 0}
    blocked_details = []

    sig_idx = 0
    for bar in range(n_bars):
        # Close positions
        closed_slots = []
        for pos in positions:
            result = _check_exit(pos, bar, opens, highs, lows, n_bars, atr_ratio, True)
            if result is not None:
                closed_slots.append(pos['slot'])
        positions = [p for p in positions if p['slot'] not in closed_slots]

        while sig_idx < len(signals_sorted) and signals_sorted[sig_idx][0] == bar:
            sig_bar, pat, direction = signals_sorted[sig_idx]
            sig_idx += 1
            dt = df['datetime'].iloc[sig_bar]

            if len(positions) >= MAX_POSITIONS:
                blocked['max_pos'] += 1
                blocked_details.append((dt, pat, direction, 'MAX_POS', len(positions)))
                continue

            dir_count = sum(1 for p in positions if p['direction'] == direction)
            if dir_count >= DIRECTION_CAP:
                blocked['dir_cap'] += 1
                blocked_details.append((dt, pat, direction, 'DIR_CAP', dir_count))
                continue

            if any(p['pattern'] == pat for p in positions):
                blocked['dup_pat'] += 1
                blocked_details.append((dt, pat, direction, 'DUP_PAT', 0))
                continue

            tp_sl = tpsl_map.get(pat)
            if tp_sl is None:
                blocked['no_tpsl'] += 1
                blocked_details.append((dt, pat, direction, 'NO_TPSL', 0))
                continue

            entry_bar = sig_bar + 1
            if entry_bar >= n_bars:
                continue

            positions.append({
                'slot': f"{pat}_{sig_bar}",
                'signal_bar': sig_bar,
                'entry_bar': entry_bar,
                'direction': direction,
                'pattern': pat,
                'tp_pct': tp_sl[0],
                'sl_pct': tp_sl[1],
            })
            entered[direction] = entered.get(direction, 0) + 1

    print(f"  Entered: LONG={entered.get('LONG',0)}, SHORT={entered.get('SHORT',0)}")
    print(f"  Blocked: max_pos={blocked['max_pos']}, dir_cap={blocked['dir_cap']}, "
          f"dup_pat={blocked['dup_pat']}, no_tpsl={blocked['no_tpsl']}")

    if blocked_details:
        print(f"\n  Blocked signals detail:")
        print(f"  {'Time':<20} {'Pattern':<12} {'Dir':>5} {'Reason':>8} {'Info':>5}")
        print("  " + "-" * 55)
        for dt, pat, d, reason, info in blocked_details:
            print(f"  {str(dt):<20} {pat:<12} {d:>5} {reason:>8} {info:>5}")


# ─── Main ───
def main():
    base = os.path.join(os.path.dirname(__file__), '..', '..')

    # 1. Load pattern sets
    with open(os.path.join(base, 'results', 'dynamic_patterns.json')) as f:
        new_data = json.load(f)
    with open(os.path.join(base, 'results', 'dynamic_patterns_35pat_compact_backup.json')) as f:
        old_data = json.load(f)

    new_L = set(new_data['patterns']['long'])
    new_S = set(new_data['patterns']['short'])
    new_tpsl = new_data['patterns_tpsl']

    old_L = set(old_data['patterns']['long'])
    old_S = set(old_data['patterns']['short'])
    old_tpsl = old_data['patterns_tpsl']

    print("=" * 70)
    print("Recent Period Backtest: v1.33 Compact vs v1.35 ATR")
    print("=" * 70)
    print(f"v1.33 Compact: {len(old_L)}L + {len(old_S)}S = {len(old_L)+len(old_S)} patterns")
    print(f"v1.35 ATR:     {len(new_L)}L + {len(new_S)}S = {len(new_L)+len(new_S)} patterns")
    print()

    # 2. Fetch data (need ~4000 bars for ATR warmup + test period)
    print("Fetching OHLCV data from BingX...")
    df = fetch_ohlcv(limit=4000)
    print(f"Fetched {len(df)} bars: {df['datetime'].iloc[0]} ~ {df['datetime'].iloc[-1]}")

    # 3. Classify candles
    print("Classifying candles...")
    rctypes = classify_candles(df)
    print(f"Classified {len(rctypes)} candles")

    # 4. Compute ATR ratio
    highs = df['high'].values
    lows = df['low'].values
    closes = df['close'].values
    opens = df['open'].values
    n_bars = len(df)

    atr_ratio = compute_atr_ratio(highs, lows, closes, ATR_PERIOD, ATR_WINDOW)
    valid_pct = np.sum(~np.isnan(atr_ratio)) / len(atr_ratio) * 100
    print(f"ATR ratio: {valid_pct:.1f}% valid")

    # 5. Define test period: last 3 days (~864 bars)
    test_start_dt = df['datetime'].iloc[-1] - timedelta(days=3)
    test_start_idx = df[df['datetime'] >= test_start_dt].index[0]
    print(f"\nTest period: {df['datetime'].iloc[test_start_idx]} ~ {df['datetime'].iloc[-1]}")
    print(f"Test bars: {n_bars - test_start_idx}")
    print(f"BTC range: ${df['low'].iloc[test_start_idx:].min():,.0f} ~ ${df['high'].iloc[test_start_idx:].max():,.0f}")

    # Recent ATR ratio stats in test period
    test_atr = atr_ratio[test_start_idx:]
    test_atr_valid = test_atr[~np.isnan(test_atr)]
    if len(test_atr_valid) > 0:
        print(f"ATR ratio in test: mean={np.mean(test_atr_valid):.3f}, "
              f"min={np.min(test_atr_valid):.3f}, max={np.max(test_atr_valid):.3f}")

    # 6. Generate signals (full range, but only trade in test period)
    old_signals = []
    new_signals = []
    for i in range(max(2, test_start_idx), n_bars):
        pat = f"{rctypes[i-2]}-{rctypes[i-1]}-{rctypes[i]}"
        if pat in old_L:
            old_signals.append((i, pat, 'LONG'))
        if pat in old_S:
            old_signals.append((i, pat, 'SHORT'))
        if pat in new_L:
            new_signals.append((i, pat, 'LONG'))
        if pat in new_S:
            new_signals.append((i, pat, 'SHORT'))

    print(f"\nSignals in test period: Old={len(old_signals)}, New={len(new_signals)}")

    # 7. Multi-scenario simulation
    scenarios = [
        ('v1.35 cap6 (현재)', new_signals, new_tpsl, new_L, new_S, DIRECTION_CAP),
        ('v1.35 no-cap',       new_signals, new_tpsl, new_L, new_S, 99),
        ('v1.33 cap6',         old_signals, old_tpsl, old_L, old_S, DIRECTION_CAP),
        ('v1.33 no-cap',       old_signals, old_tpsl, old_L, old_S, 99),
    ]

    all_results = {}
    for label, sigs, tpsl, pL, pS, cap in scenarios:
        print(f"\n{'=' * 70}")
        print(f"SIMULATION: {label}  (dir_cap={cap})")
        print('=' * 70)
        trades = simulate_portfolio(
            sigs, {**{p:'LONG' for p in pL}, **{p:'SHORT' for p in pS}},
            tpsl, opens, highs, lows, n_bars,
            atr_ratio=atr_ratio, use_atr=True, label=label,
            direction_cap_override=cap,
        )
        print_results(trades, df, label)

        # Direction breakdown
        for d in ['LONG', 'SHORT']:
            dt = [t for t in trades if t['direction'] == d]
            if not dt:
                print(f"\n  {d}: No trades")
                continue
            w = sum(1 for t in dt if t['pnl_slot'] > 0)
            l = len(dt) - w
            pnl_d = sum(t['pnl_portfolio'] for t in dt)
            print(f"\n  {d}: {len(dt)} trades ({w}W/{l}L), WR={w/len(dt)*100:.1f}%, PnL={pnl_d:+.2f}%")
            for t in sorted(dt, key=lambda x: x['entry_bar']):
                dts = df['datetime'].iloc[t['entry_bar']]
                print(f"    {str(dts):<20} {t['pattern']:<12} ${t['entry_price']:>9,.1f}→${t['exit_price']:>9,.1f} "
                      f"{t['pnl_slot']:>+7.2f}% {t['reason']:>4} {t['hold_bars']:>4}b")

        all_results[label] = trades

    # 8. Comparison table
    print(f"\n{'=' * 70}")
    print("COMPARISON")
    print('=' * 70)
    header = f"{'Metric':<20}"
    for label in all_results:
        header += f" {label:>18}"
    print(header)
    print("-" * (20 + 19 * len(all_results)))

    for metric_name, metric_fn in [
        ('Trades', lambda ts: str(len(ts))),
        ('LONG trades', lambda ts: str(sum(1 for t in ts if t['direction']=='LONG'))),
        ('SHORT trades', lambda ts: str(sum(1 for t in ts if t['direction']=='SHORT'))),
        ('Win Rate', lambda ts: f"{sum(1 for t in ts if t['pnl_slot']>0)/max(1,len(ts))*100:.1f}%"),
        ('Portfolio PnL', lambda ts: f"{sum(t['pnl_portfolio'] for t in ts):+.2f}%"),
        ('LONG PnL', lambda ts: f"{sum(t['pnl_portfolio'] for t in ts if t['direction']=='LONG'):+.2f}%"),
        ('SHORT PnL', lambda ts: f"{sum(t['pnl_portfolio'] for t in ts if t['direction']=='SHORT'):+.2f}%"),
        ('Max DD (approx)', lambda ts: f"{_max_dd(ts):.2f}%"),
    ]:
        row = f"{metric_name:<20}"
        for label in all_results:
            row += f" {metric_fn(all_results[label]):>18}"
        print(row)


def _max_dd(trades):
    """Approximate max drawdown from portfolio PnL sequence."""
    if not trades:
        return 0.0
    sorted_trades = sorted(trades, key=lambda x: x['entry_bar'])
    equity = 100.0
    peak = equity
    max_dd = 0.0
    for t in sorted_trades:
        equity += t['pnl_portfolio']
        if equity > peak:
            peak = equity
        dd = (peak - equity) / peak * 100
        if dd > max_dd:
            max_dd = dd
    return max_dd


def print_results(trades, df, label):
    """Print trade results summary."""
    if not trades:
        print(f"  No trades for {label}")
        return

    wins = [t for t in trades if t['pnl_slot'] > 0]
    losses = [t for t in trades if t['pnl_slot'] <= 0]
    total_pnl = sum(t['pnl_portfolio'] for t in trades)
    wr = len(wins) / len(trades) * 100

    print(f"\n  Trades: {len(trades)} ({len(wins)}W / {len(losses)}L)")
    print(f"  Win Rate: {wr:.1f}%")
    print(f"  Portfolio PnL: {total_pnl:+.2f}%")
    if wins:
        print(f"  Avg Win (slot): {np.mean([t['pnl_slot'] for t in wins]):+.2f}%")
    if losses:
        print(f"  Avg Loss (slot): {np.mean([t['pnl_slot'] for t in losses]):+.2f}%")
    print(f"  Avg Hold: {np.mean([t['hold_bars'] for t in trades]):.0f} bars")

    # Trade-by-trade
    print(f"\n  {'Time':<20} {'Pat':<12} {'Dir':>5} {'Entry':>10} {'Exit':>10} "
          f"{'PnL(s)':>8} {'PnL(p)':>8} {'Rsn':>4} {'Hold':>5} {'ATR':>5}")
    print("  " + "-" * 95)
    for t in sorted(trades, key=lambda x: x['entry_bar']):
        dt = df['datetime'].iloc[t['entry_bar']]
        print(f"  {str(dt):<20} {t['pattern']:<12} {t['direction']:>5} "
              f"${t['entry_price']:>9,.1f} ${t['exit_price']:>9,.1f} "
              f"{t['pnl_slot']:>+7.2f}% {t['pnl_portfolio']:>+7.2f}% "
              f"{t['reason']:>4} {t['hold_bars']:>5} {t['atr_mult']:>5.3f}")


if __name__ == '__main__':
    main()
