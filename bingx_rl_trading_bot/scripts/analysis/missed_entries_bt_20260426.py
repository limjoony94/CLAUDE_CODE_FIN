"""
Missed Entries BT — 2026-04-25 ~ 04-26 Hedge mode 11개 silent fail 신호 재평가
=================================================================================
실측 BT (추정 X). 각 신호:
- Entry: bot 로그의 actual price (signal close 시점)
- Exit: BingX 15m 캔들로 봇 signals.py.check_exit를 cycle별 호출
- Config: 현재 production (max_sl 4.5, trail_K 2.5, progressive_trail enabled, max_hold 192)

목표: 놓친 11 entries 누적 PnL을 정확히 측정해서 보고.
"""
import sys, os, json, math, yaml
from datetime import datetime, timezone, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))
from scripts.production.c1_breakout.signals import C1BreakoutSignal
from scripts.production.c1_breakout.indicators import compute_atr

FEE_RT_PCT = 0.10  # 0.05% × 2 (taker)
LEVERAGE_DISPLAY = 3  # bot trading_leverage

# 11 missed signals from logs/c1_breakout.log* (Hedge mode 109400 fails)
SIGNALS = [
    # (signal_time_utc, direction, entry_price, sl_price, atr_at_signal)
    ("2026-04-24T22:45:00", "SHORT", 77244.00, 77727.90, 162.60),  # KST 04-25 07:45
    ("2026-04-25T09:45:00", "LONG",  77686.80, 77419.64,  80.96),  # KST 04-25 18:45
    ("2026-04-25T10:30:00", "LONG",  77778.80, 77510.54,  81.29),  # KST 04-25 19:30
    ("2026-04-25T12:45:00", "SHORT", 77517.60, 77845.30,  86.84),  # KST 04-25 21:45
    ("2026-04-25T16:00:00", "SHORT", 77290.10, 77715.30,  96.25),  # KST 04-26 01:00
    ("2026-04-25T21:00:00", "LONG",  77436.60, 77137.60, 100.19),  # KST 04-26 06:00
    ("2026-04-25T21:15:00", "LONG",  77461.10, 77137.60,  97.26),  # KST 04-26 06:15
    ("2026-04-25T22:45:00", "LONG",  77559.00, 77155.16,  89.74),  # KST 04-26 07:45
    ("2026-04-26T03:30:00", "SHORT", 77389.10, 77611.40,  85.65),  # KST 04-26 12:30
    ("2026-04-26T05:00:00", "LONG",  77732.80, 77284.70, 104.38),  # KST 04-26 14:00
    ("2026-04-26T05:15:00", "LONG",  78104.90, 77520.77, 129.81),  # KST 04-26 14:15
]


def fetch_bingx_15m_range(start_utc, end_utc):
    """Fetch BingX BTC 15m candles in [start, end] (inclusive end)."""
    import ccxt
    ex = ccxt.bingx({'options': {'defaultType': 'swap'}})
    st = int(start_utc.replace(tzinfo=timezone.utc).timestamp() * 1000)
    en = int(end_utc.replace(tzinfo=timezone.utc).timestamp() * 1000)
    out = []
    cur = st
    while cur <= en:
        cs = ex.fetch_ohlcv('BTC-USDT', '15m', since=cur, limit=1000)
        if not cs: break
        out.extend(cs)
        cur = cs[-1][0] + 15 * 60 * 1000
        if len(cs) < 1000: break
    seen = set(); uniq = []
    for c in out:
        if c[0] not in seen and c[0] <= en:
            seen.add(c[0]); uniq.append(c)
    return sorted(uniq, key=lambda x: x[0])


def simulate_trade(signal, candles, signal_obj, atr_warmup=14):
    """Simulate single trade from signal time using check_exit each bar.

    Approach:
    - Find bar where signal_time matches close time (signal bar)
    - Entry = next bar open (i+1) — but we use logged entry price (= signal close)
      because bot enters at the SAME bar close in real loop. Use logged entry.
    - From bar i+1 forward, call check_exit each bar with cumulative best_price.
    """
    sig_ts_ms = int(datetime.strptime(signal['ts'], "%Y-%m-%dT%H:%M:%S")
                    .replace(tzinfo=timezone.utc).timestamp() * 1000)
    # Locate signal bar
    sig_idx = None
    for i, c in enumerate(candles):
        if c[0] == sig_ts_ms:
            sig_idx = i; break
    if sig_idx is None:
        return {**signal, 'error': 'signal bar not found in candles'}
    if sig_idx + 1 >= len(candles):
        return {**signal, 'error': 'no bars after signal'}

    direction = signal['direction']
    entry = signal['entry']
    sl = signal['sl']
    atr_signal = signal['atr']

    # Compute rolling ATR for post-signal bars (uses candles only)
    highs = [c[2] for c in candles]
    lows  = [c[3] for c in candles]
    closes = [c[4] for c in candles]
    atrs = compute_atr(highs, lows, closes, signal_obj.atr_period)

    # Initialize best_price at entry
    best_price = entry
    bars_held = 0

    # Iterate from sig_idx+1 forward
    for j in range(sig_idx + 1, len(candles)):
        bars_held += 1
        c = candles[j]
        h, l, cl = c[2], c[3], c[4]

        # Update best_price
        if direction == 'LONG':
            best_price = max(best_price, h)
        else:
            best_price = min(best_price, l)

        # ATR for this bar (use signals' rolling ATR; fallback to signal-time ATR)
        atr_val = atrs[j] if not math.isnan(atrs[j]) else atr_signal

        result = signal_obj.check_exit(
            direction=direction,
            entry_price=entry,
            best_price=best_price,
            current_high=h,
            current_low=l,
            current_close=cl,
            sl_price=sl,
            atr_val=atr_val,
            bars_held=bars_held,
        )
        if result:
            xp = result['exit_price']
            if direction == 'LONG':
                pnl_1x = (xp / entry - 1) * 100
            else:
                pnl_1x = (1 - xp / entry) * 100
            pnl_1x -= FEE_RT_PCT  # round-trip fee
            return {
                **signal,
                'exit_time': datetime.fromtimestamp(c[0] / 1000).isoformat(),
                'exit_price': round(xp, 2),
                'reason': result['reason'],
                'bars_held': bars_held,
                'pnl_1x': round(pnl_1x, 4),
                'pnl_3x': round(pnl_1x * LEVERAGE_DISPLAY, 4),
                'best_price': round(best_price, 2),
            }
    # No exit in window — open
    return {
        **signal,
        'exit_time': None,
        'exit_price': closes[-1],
        'reason': 'OPEN',
        'bars_held': bars_held,
        'pnl_unrealized_1x': round(((closes[-1] / entry - 1) if direction == 'LONG' else (1 - closes[-1] / entry)) * 100 - FEE_RT_PCT, 4),
        'best_price': round(best_price, 2),
    }


def main():
    # Load production config
    cfg_path = ROOT / 'config' / 'c1_breakout_config.yaml'
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)
    sig_obj = C1BreakoutSignal(cfg['strategy'])
    print(f"Config: max_sl_atr={sig_obj.max_sl_atr}, trail_K={sig_obj.trail_K}, "
          f"prog_trail={sig_obj.prog_trail_enabled} (thr={sig_obj.prog_trail_threshold}, K_post={sig_obj.prog_trail_K_post})")
    print(f"Emergency={sig_obj.emergency_sl_pct}%, max_hold={sig_obj.max_hold_bars} bars\n")

    # Fetch enough candles to cover all signals + 192 bars (max_hold)
    first_sig = datetime.strptime(SIGNALS[0][0], "%Y-%m-%dT%H:%M:%S")
    last_sig = datetime.strptime(SIGNALS[-1][0], "%Y-%m-%dT%H:%M:%S")
    fetch_start = first_sig - timedelta(hours=4)  # ATR warmup
    fetch_end = datetime.utcnow()  # up to now
    print(f"Fetching BTC 15m: {fetch_start} ~ {fetch_end}")
    candles = fetch_bingx_15m_range(fetch_start, fetch_end)
    print(f"  {len(candles)} candles\n")

    if len(candles) < 50:
        print("ERROR: insufficient candles")
        return

    # Run each signal (each is independent — bot only holds N=1, but missed = no actual conflict)
    print("=" * 110)
    print(f"{'#':>2} {'signal_utc':<19} {'dir':<5} {'entry':>9} {'sl':>9} {'reason':<10} {'bars':>5} {'exit_p':>9} {'pnl_1x':>8} {'pnl_3x':>8}")
    print("=" * 110)

    results = []
    for i, (ts, dir_, entry, sl, atr) in enumerate(SIGNALS):
        sig = {'ts': ts, 'direction': dir_, 'entry': entry, 'sl': sl, 'atr': atr}
        r = simulate_trade(sig, candles, sig_obj)
        results.append(r)
        if 'error' in r:
            print(f"{i+1:>2} {ts:<19} {dir_:<5} {entry:>9.1f} {sl:>9.1f} ERR: {r['error']}")
            continue
        reason = r['reason']
        bars = r.get('bars_held', 0)
        xp = r.get('exit_price', 0)
        if reason == 'OPEN':
            pnl1 = r.get('pnl_unrealized_1x', 0)
            pnl3 = pnl1 * LEVERAGE_DISPLAY
            print(f"{i+1:>2} {ts:<19} {dir_:<5} {entry:>9.1f} {sl:>9.1f} {'OPEN':<10} {bars:>5} {xp:>9.1f} {pnl1:>+7.3f}% {pnl3:>+7.3f}%")
        else:
            pnl1 = r['pnl_1x']
            pnl3 = r['pnl_3x']
            print(f"{i+1:>2} {ts:<19} {dir_:<5} {entry:>9.1f} {sl:>9.1f} {reason:<10} {bars:>5} {xp:>9.1f} {pnl1:>+7.3f}% {pnl3:>+7.3f}%")

    # Aggregate
    print("=" * 110)
    closed = [r for r in results if 'pnl_1x' in r]
    open_ = [r for r in results if r.get('reason') == 'OPEN']

    if closed:
        sum1x = sum(r['pnl_1x'] for r in closed)
        sum3x = sum(r['pnl_3x'] for r in closed)
        wins = sum(1 for r in closed if r['pnl_1x'] > 0)
        wr = 100 * wins / len(closed)
        avg1x = sum1x / len(closed)
        # Reason breakdown
        from collections import Counter
        reason_counts = Counter(r['reason'] for r in closed)

        print(f"\nClosed: {len(closed)} trades")
        print(f"  Sum PnL 1x : {sum1x:+.3f}%")
        print(f"  Sum PnL 3x : {sum3x:+.3f}%")
        print(f"  Avg PnL 1x : {avg1x:+.3f}%/trade")
        print(f"  WR         : {wr:.1f}% ({wins}/{len(closed)})")
        print(f"  Reasons    : {dict(reason_counts)}")

    if open_:
        sum_open = sum(r['pnl_unrealized_1x'] for r in open_)
        print(f"\nOpen: {len(open_)} (unrealized 1x sum: {sum_open:+.3f}%)")

    # Save
    out = {
        'date': datetime.now().isoformat(),
        'config': {
            'max_sl_atr': sig_obj.max_sl_atr,
            'trail_K': sig_obj.trail_K,
            'progressive_trail': {
                'enabled': sig_obj.prog_trail_enabled,
                'threshold_pct': sig_obj.prog_trail_threshold,
                'trail_K_post': sig_obj.prog_trail_K_post,
            },
            'emergency_sl_pct': sig_obj.emergency_sl_pct,
            'max_hold_bars': sig_obj.max_hold_bars,
            'fee_rt_pct': FEE_RT_PCT,
        },
        'n_signals': len(SIGNALS),
        'closed': len(closed),
        'open': len(open_),
        'sum_pnl_1x_closed': round(sum(r['pnl_1x'] for r in closed), 4) if closed else 0,
        'sum_pnl_3x_closed': round(sum(r['pnl_3x'] for r in closed), 4) if closed else 0,
        'sum_pnl_1x_open_unrealized': round(sum(r['pnl_unrealized_1x'] for r in open_), 4) if open_ else 0,
        'wr': round(100 * sum(1 for r in closed if r['pnl_1x'] > 0) / len(closed), 1) if closed else 0,
        'trades': results,
    }
    p = ROOT / 'results' / f'missed_entries_bt_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
    with open(p, 'w') as f:
        json.dump(out, f, indent=2, default=str)
    print(f"\nSaved: {p}")


if __name__ == '__main__':
    main()
