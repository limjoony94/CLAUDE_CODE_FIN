"""
Direction Cap WF Study: cap6 vs no-cap
=======================================
270d CSV (77,760 bars) + API 최신 데이터로 장기 WF 검증.
v1.35 ATR 51 패턴 고정, 포트폴리오 구성만 변경 (cap 유무).

3-fold OOS (각 ~80d):
  Fold 1: bars 590-26550   (day 2~92)
  Fold 2: bars 26550-52510 (day 92~182)
  Fold 3: bars 52510-77760 (day 182~270)

+ 최근 26d API 데이터 추가 시 Fold 4 (day 270~296)

Standard Research Protocol:
- Entry: signal bar+1 open
- Exit: intrabar high/low (distance-based)
- Fee: 0.10% roundtrip (x3 leverage in capital space)
- Slippage buffer: 0.02%
- ATR scaling: ATR(14)/median(576), clamp [0.6, 1.7]
- N=9, 1/N=11.1% sizing, timeout 864 bars
"""
import sys, os, json
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from scripts.scanner.pattern_scanner import classify_candle

# ─── Constants ───
LEVERAGE = 3
FEE_PCT = 0.10
SLIPPAGE_BUFFER = 0.02
TIMEOUT_BARS = 864
MAX_POSITIONS = 9
ATR_PERIOD = 14
ATR_WINDOW = 576
CLAMP_LO = 0.6
CLAMP_HI = 1.7
ATR_WARMUP = ATR_PERIOD + ATR_WINDOW  # 590 bars


def compute_atr_ratio(highs, lows, closes):
    n = len(closes)
    tr = np.empty(n)
    tr[0] = highs[0] - lows[0]
    for i in range(1, n):
        tr[i] = max(highs[i] - lows[i],
                     abs(highs[i] - closes[i-1]),
                     abs(lows[i] - closes[i-1]))
    atr = pd.Series(tr).rolling(ATR_PERIOD, min_periods=ATR_PERIOD).mean().values
    med = pd.Series(atr).rolling(ATR_WINDOW, min_periods=ATR_WINDOW).median().values
    ratio = np.full(n, np.nan)
    valid = (~np.isnan(atr)) & (~np.isnan(med)) & (med > 0)
    ratio[valid] = atr[valid] / med[valid]
    return ratio


def load_csv_data(base):
    """Load 270d CSV + classify candles."""
    csv_path = os.path.join(base, 'data', 'btc_5m_270days_reclassified.csv')
    df = pd.read_csv(csv_path)
    df['datetime'] = pd.to_datetime(df['timestamp'])
    # type_code has short codes (BD, DN, U, etc.)
    rctypes = df['type_code'].values.tolist()
    return df, rctypes


def fetch_recent_data(df_csv):
    """Fetch recent data from BingX API beyond CSV end date."""
    import yaml, ccxt
    base = os.path.join(os.path.dirname(__file__), '..', '..')
    cfg_path = os.path.join(base, 'config', 'api_keys.yaml')
    with open(cfg_path) as f:
        keys = yaml.safe_load(f)
    bingx_keys = keys['bingx']['mainnet']
    ex = ccxt.bingx({
        'apiKey': bingx_keys['api_key'],
        'secret': bingx_keys['secret_key'],
        'options': {'defaultType': 'swap'}
    })

    csv_end = pd.to_datetime(df_csv['timestamp'].iloc[-1])
    since_ts = int(csv_end.timestamp() * 1000) + 300000  # +5min

    all_bars = []
    remaining = 10000  # ~35 days max
    while remaining > 0:
        batch = ex.fetch_ohlcv('BTC-USDT', '5m', since=since_ts, limit=min(1000, remaining))
        if not batch:
            break
        all_bars.extend(batch)
        since_ts = batch[-1][0] + 300000
        remaining -= len(batch)
        if len(batch) < 100:
            break

    if not all_bars:
        return None, None

    df_api = pd.DataFrame(all_bars, columns=['ts_ms', 'open', 'high', 'low', 'close', 'volume'])
    df_api = df_api.drop_duplicates(subset='ts_ms').sort_values('ts_ms').reset_index(drop=True)
    df_api['datetime'] = pd.to_datetime(df_api['ts_ms'], unit='ms')
    df_api['timestamp'] = df_api['datetime'].astype(str)

    # Classify candles
    df_api['body'] = df_api['close'] - df_api['open']
    df_api['body_abs'] = df_api['body'].abs()
    df_api['avg_body_20'] = df_api['body_abs'].rolling(20, min_periods=1).mean()
    rctypes = []
    for _, row in df_api.iterrows():
        ct = classify_candle(row, row['avg_body_20'])
        rctypes.append(ct.value)

    return df_api, rctypes


def generate_signals(rctypes, patterns_long, patterns_short, start_idx=0):
    """Generate signals. Returns list of (bar_idx, pattern, direction)."""
    signals = []
    n = len(rctypes)
    for i in range(max(2, start_idx), n):
        pat = f"{rctypes[i-2]}-{rctypes[i-1]}-{rctypes[i]}"
        if pat in patterns_long:
            signals.append((i, pat, 'LONG'))
        if pat in patterns_short:
            signals.append((i, pat, 'SHORT'))
    return signals


def simulate_portfolio(signals, tpsl_map, opens, highs, lows, n_bars,
                       atr_ratio, direction_cap, oos_start, oos_end):
    """Simulate N=9 multi-position portfolio on OOS period only.
    Signals before oos_start are ignored. Positions open at oos_end are force-closed."""
    positions = []
    trades = []
    SIZE_PCT = 100.0 / MAX_POSITIONS

    signals_in_range = [(s, p, d) for s, p, d in signals if oos_start <= s < oos_end]
    signals_sorted = sorted(signals_in_range, key=lambda x: x[0])
    sig_idx = 0

    for bar in range(oos_start, oos_end):
        # Check exits
        closed_slots = []
        for pos in positions:
            result = _check_exit(pos, bar, opens, highs, lows, n_bars, atr_ratio)
            if result is not None:
                result['pattern'] = pos['pattern']
                result['direction'] = pos['direction']
                result['pnl_portfolio'] = result['pnl_slot'] * SIZE_PCT / 100
                trades.append(result)
                closed_slots.append(pos['slot'])
        positions = [p for p in positions if p['slot'] not in closed_slots]

        # Process signals
        while sig_idx < len(signals_sorted) and signals_sorted[sig_idx][0] == bar:
            sig_bar, pat, direction = signals_sorted[sig_idx]
            sig_idx += 1

            if len(positions) >= MAX_POSITIONS:
                continue
            dir_count = sum(1 for p in positions if p['direction'] == direction)
            if dir_count >= direction_cap:
                continue
            if any(p['pattern'] == pat for p in positions):
                continue

            entry_bar = sig_bar + 1
            if entry_bar >= n_bars:
                continue
            tp_sl = tpsl_map.get(pat)
            if tp_sl is None:
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

    # Force-close remaining positions at OOS end
    for pos in positions:
        entry_bar = pos['entry_bar']
        if entry_bar >= n_bars:
            continue
        entry = opens[entry_bar]
        if entry <= 0:
            continue
        exit_bar = min(oos_end - 1, n_bars - 1)
        exit_price = opens[exit_bar] if exit_bar < n_bars else opens[-1]
        direction = pos['direction']
        if direction == 'LONG':
            pnl = (exit_price / entry - 1) * 100 * LEVERAGE
        else:
            pnl = (1 - exit_price / entry) * 100 * LEVERAGE
        pnl -= FEE_PCT * LEVERAGE
        trades.append({
            'entry_bar': entry_bar, 'exit_bar': exit_bar,
            'hold_bars': exit_bar - entry_bar,
            'entry_price': entry, 'exit_price': exit_price,
            'pnl_slot': pnl, 'reason': 'OOS_END',
            'pattern': pos['pattern'], 'direction': direction,
            'pnl_portfolio': pnl * SIZE_PCT / 100,
        })

    return trades


def _check_exit(pos, bar, opens, highs, lows, n_bars, atr_ratio):
    entry_bar = pos['entry_bar']
    if bar < entry_bar:
        return None

    entry = opens[entry_bar]
    if entry <= 0:
        return None

    tp_pct = pos['tp_pct']
    sl_pct = pos['sl_pct']
    direction = pos['direction']
    sig_bar = pos['signal_bar']

    if atr_ratio is not None and not np.isnan(atr_ratio[sig_bar]):
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
                'pnl_slot': pnl, 'reason': 'TIMEOUT'}

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
            'pnl_slot': pnl, 'reason': reason}


def compute_metrics(trades):
    """Compute portfolio metrics from trades."""
    if not trades:
        return {'trades': 0, 'wr': 0, 'pnl': 0, 'mdd': 0,
                'long_trades': 0, 'short_trades': 0,
                'long_pnl': 0, 'short_pnl': 0,
                'avg_win': 0, 'avg_loss': 0, 'pnl_per_trade': 0}

    wins = [t for t in trades if t['pnl_slot'] > 0]
    losses = [t for t in trades if t['pnl_slot'] <= 0]
    total_pnl = sum(t['pnl_portfolio'] for t in trades)
    wr = len(wins) / len(trades) * 100 if trades else 0

    # MDD
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

    long_t = [t for t in trades if t['direction'] == 'LONG']
    short_t = [t for t in trades if t['direction'] == 'SHORT']

    return {
        'trades': len(trades),
        'wr': wr,
        'pnl': total_pnl,
        'mdd': max_dd,
        'long_trades': len(long_t),
        'short_trades': len(short_t),
        'long_pnl': sum(t['pnl_portfolio'] for t in long_t),
        'short_pnl': sum(t['pnl_portfolio'] for t in short_t),
        'avg_win': np.mean([t['pnl_slot'] for t in wins]) if wins else 0,
        'avg_loss': np.mean([t['pnl_slot'] for t in losses]) if losses else 0,
        'pnl_per_trade': total_pnl / len(trades) if trades else 0,
    }


def main():
    base = os.path.join(os.path.dirname(__file__), '..', '..')

    # 1. Load patterns
    with open(os.path.join(base, 'results', 'dynamic_patterns.json')) as f:
        pat_data = json.load(f)
    pL = set(pat_data['patterns']['long'])
    pS = set(pat_data['patterns']['short'])
    tpsl = pat_data['patterns_tpsl']
    print(f"v1.35 ATR patterns: {len(pL)}L + {len(pS)}S = {len(pL)+len(pS)}")

    # 2. Load CSV data
    print("\nLoading 270d CSV...")
    df_csv, rctypes_csv = load_csv_data(base)
    print(f"CSV: {len(df_csv)} bars ({len(df_csv)/288:.0f}d), "
          f"{df_csv['datetime'].iloc[0]} ~ {df_csv['datetime'].iloc[-1]}")

    # 3. Fetch recent API data to extend
    print("\nFetching recent API data...")
    df_api, rctypes_api = fetch_recent_data(df_csv)
    if df_api is not None:
        print(f"API: {len(df_api)} bars ({len(df_api)/288:.1f}d), "
              f"{df_api['datetime'].iloc[0]} ~ {df_api['datetime'].iloc[-1]}")

        # Merge
        opens = np.concatenate([df_csv['open'].values, df_api['open'].values])
        highs = np.concatenate([df_csv['high'].values, df_api['high'].values])
        lows = np.concatenate([df_csv['low'].values, df_api['low'].values])
        closes = np.concatenate([df_csv['close'].values, df_api['close'].values])
        rctypes = rctypes_csv + rctypes_api
        datetimes = pd.concat([df_csv['datetime'], df_api['datetime']]).reset_index(drop=True)
        total_bars = len(opens)
        total_days = total_bars / 288
        print(f"Combined: {total_bars} bars ({total_days:.1f}d)")
    else:
        print("No API data fetched, using CSV only")
        opens = df_csv['open'].values
        highs = df_csv['high'].values
        lows = df_csv['low'].values
        closes = df_csv['close'].values
        rctypes = rctypes_csv
        datetimes = df_csv['datetime']
        total_bars = len(opens)
        total_days = total_bars / 288

    n_bars = len(opens)

    # 4. Compute ATR ratio
    print("\nComputing ATR ratio...")
    atr_ratio = compute_atr_ratio(highs, lows, closes)
    valid_pct = np.sum(~np.isnan(atr_ratio)) / len(atr_ratio) * 100
    print(f"ATR ratio: {valid_pct:.1f}% valid (warmup={ATR_WARMUP} bars)")

    # 5. Generate all signals
    all_signals = generate_signals(rctypes, pL, pS, start_idx=ATR_WARMUP)
    print(f"Total signals: {len(all_signals)}")

    # 6. Define OOS folds
    tradeable_start = ATR_WARMUP
    tradeable_bars = n_bars - tradeable_start
    tradeable_days = tradeable_bars / 288

    # 3-fold if 270d, 4-fold if extended
    n_folds = 3 if tradeable_days < 300 else 4
    fold_size = tradeable_bars // n_folds

    folds = []
    for i in range(n_folds):
        oos_start = tradeable_start + i * fold_size
        oos_end = tradeable_start + (i + 1) * fold_size if i < n_folds - 1 else n_bars
        folds.append((oos_start, oos_end))

    print(f"\nWF Folds: {n_folds} x ~{fold_size/288:.0f}d")
    for i, (s, e) in enumerate(folds):
        print(f"  Fold {i+1}: bars {s}-{e} ({(e-s)/288:.0f}d) "
              f"| {datetimes.iloc[s]} ~ {datetimes.iloc[min(e-1, n_bars-1)]}")

    # 7. Run WF for each cap scenario
    cap_scenarios = [
        ('cap6', 6),
        ('cap9 (no-cap)', 99),
        ('cap3', 3),
        ('cap4', 4),
        ('cap5', 5),
        ('cap7', 7),
        ('cap8', 8),
    ]

    results = {}  # {cap_label: [fold_metrics]}
    for cap_label, cap_val in cap_scenarios:
        results[cap_label] = []
        for fold_idx, (oos_start, oos_end) in enumerate(folds):
            trades = simulate_portfolio(
                all_signals, tpsl, opens, highs, lows, n_bars,
                atr_ratio, cap_val, oos_start, oos_end
            )
            m = compute_metrics(trades)
            results[cap_label].append(m)

    # 8. Print fold-by-fold results
    print(f"\n{'=' * 100}")
    print("FOLD-BY-FOLD RESULTS")
    print('=' * 100)

    for cap_label, cap_val in cap_scenarios:
        print(f"\n--- {cap_label} (dir_cap={cap_val}) ---")
        print(f"  {'Fold':<8} {'Trades':>7} {'L/S':>8} {'WR':>7} {'PnL':>9} {'MDD':>7} {'PnL/MDD':>8} "
              f"{'L_PnL':>8} {'S_PnL':>8} {'$/trade':>8}")
        print("  " + "-" * 85)
        for i, m in enumerate(results[cap_label]):
            pnl_mdd = m['pnl'] / m['mdd'] if m['mdd'] > 0 else 0
            print(f"  Fold {i+1}  {m['trades']:>7} {m['long_trades']:>3}/{m['short_trades']:<4} "
                  f"{m['wr']:>6.1f}% {m['pnl']:>+8.2f}% {m['mdd']:>6.2f}% {pnl_mdd:>7.2f}x "
                  f"{m['long_pnl']:>+7.2f}% {m['short_pnl']:>+7.2f}% {m['pnl_per_trade']:>+7.3f}%")

        # Aggregate
        all_trades = sum(m['trades'] for m in results[cap_label])
        total_pnl = sum(m['pnl'] for m in results[cap_label])
        avg_wr = np.mean([m['wr'] for m in results[cap_label] if m['trades'] > 0])
        max_mdd = max(m['mdd'] for m in results[cap_label])
        avg_mdd = np.mean([m['mdd'] for m in results[cap_label] if m['trades'] > 0])
        total_long_pnl = sum(m['long_pnl'] for m in results[cap_label])
        total_short_pnl = sum(m['short_pnl'] for m in results[cap_label])
        pnl_per_trade = total_pnl / all_trades if all_trades > 0 else 0
        pnl_max_mdd = total_pnl / max_mdd if max_mdd > 0 else 0

        print("  " + "-" * 85)
        print(f"  TOTAL  {all_trades:>7} {'':>8} {avg_wr:>6.1f}% {total_pnl:>+8.2f}% {max_mdd:>6.2f}% "
              f"{pnl_max_mdd:>7.2f}x {total_long_pnl:>+7.2f}% {total_short_pnl:>+7.2f}% {pnl_per_trade:>+7.3f}%")

    # 9. Comparison summary
    print(f"\n{'=' * 100}")
    print("COMPARISON SUMMARY (All Folds Aggregated)")
    print('=' * 100)

    header = f"{'Scenario':<18} {'Trades':>7} {'Avg WR':>8} {'Total PnL':>10} {'Max MDD':>8} {'PnL/MDD':>8} " \
             f"{'L Trades':>8} {'S Trades':>8} {'L PnL':>8} {'S PnL':>8} {'$/trade':>8}"
    print(header)
    print("-" * 110)

    for cap_label, cap_val in cap_scenarios:
        folds_m = results[cap_label]
        all_trades = sum(m['trades'] for m in folds_m)
        total_pnl = sum(m['pnl'] for m in folds_m)
        avg_wr = np.mean([m['wr'] for m in folds_m if m['trades'] > 0])
        max_mdd = max(m['mdd'] for m in folds_m)
        pnl_mdd = total_pnl / max_mdd if max_mdd > 0 else 0
        total_L = sum(m['long_trades'] for m in folds_m)
        total_S = sum(m['short_trades'] for m in folds_m)
        total_L_pnl = sum(m['long_pnl'] for m in folds_m)
        total_S_pnl = sum(m['short_pnl'] for m in folds_m)
        ppt = total_pnl / all_trades if all_trades > 0 else 0

        print(f"{cap_label:<18} {all_trades:>7} {avg_wr:>7.1f}% {total_pnl:>+9.2f}% {max_mdd:>7.2f}% "
              f"{pnl_mdd:>7.2f}x {total_L:>8} {total_S:>8} {total_L_pnl:>+7.2f}% {total_S_pnl:>+7.2f}% "
              f"{ppt:>+7.3f}%")

    # 10. Consistency check: which cap has ALL folds positive?
    print(f"\n{'=' * 100}")
    print("CONSISTENCY CHECK (All Folds Positive PnL?)")
    print('=' * 100)
    for cap_label, cap_val in cap_scenarios:
        folds_m = results[cap_label]
        positive_folds = sum(1 for m in folds_m if m['pnl'] > 0)
        total_folds = len(folds_m)
        status = "PASS" if positive_folds == total_folds else "FAIL"
        fold_pnls = " | ".join(f"{m['pnl']:+.2f}%" for m in folds_m)
        print(f"  {cap_label:<18} {positive_folds}/{total_folds} {status:>4}  [{fold_pnls}]")

    # 11. Delta analysis: no-cap vs cap6
    print(f"\n{'=' * 100}")
    print("DELTA ANALYSIS: no-cap vs cap6")
    print('=' * 100)
    for i in range(n_folds):
        m_cap6 = results['cap6'][i]
        m_nocap = results['cap9 (no-cap)'][i]
        d_trades = m_nocap['trades'] - m_cap6['trades']
        d_pnl = m_nocap['pnl'] - m_cap6['pnl']
        d_mdd = m_nocap['mdd'] - m_cap6['mdd']
        d_wr = m_nocap['wr'] - m_cap6['wr']
        print(f"  Fold {i+1}: trades {d_trades:+d}, WR {d_wr:+.1f}pp, PnL {d_pnl:+.2f}%, MDD {d_mdd:+.2f}%")

    # Total delta
    cap6_total_pnl = sum(m['pnl'] for m in results['cap6'])
    nocap_total_pnl = sum(m['pnl'] for m in results['cap9 (no-cap)'])
    cap6_max_mdd = max(m['mdd'] for m in results['cap6'])
    nocap_max_mdd = max(m['mdd'] for m in results['cap9 (no-cap)'])
    print(f"\n  TOTAL: PnL {nocap_total_pnl - cap6_total_pnl:+.2f}%, "
          f"MDD {nocap_max_mdd - cap6_max_mdd:+.2f}%")
    print(f"  cap6 PnL/MDD:  {cap6_total_pnl/cap6_max_mdd:.2f}x" if cap6_max_mdd > 0 else "")
    print(f"  no-cap PnL/MDD: {nocap_total_pnl/nocap_max_mdd:.2f}x" if nocap_max_mdd > 0 else "")


if __name__ == '__main__':
    main()
