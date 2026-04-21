"""
Baton-only Backtest — E Phase 1 (2026-04-22)
==============================================
Pre-activation 구간에도 baton STOP_MARKET intrabar-touch 로직 적용.
현재 BT(bar-close trail) + LIVE(tick-level TRAILING) 사이의 타협점을 BT로 재현.

Key differences vs dd_comparison_20260421.py (baseline BT):
1. Trail exit는 intrabar low/high touch 기반 (baton STOP_MARKET처럼)
2. Pre-activation 구간에도 baton trigger 적용 (activation_pct 가드 제거)
3. Fractal SL과 baton trigger 중 tighter 선택

Outputs:
1. 27 trades 기간 (04-12 ~ 04-21) — LIVE 비교용
2. 333일 full 기간 — baseline +170% preservation 검증
"""

import sys
import os
import json
import math
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from scripts.production.c1_breakout.indicators import (
    compute_atr, compute_channel, compute_fractal_swings
)
from scripts.production.c1_breakout.signals import C1BreakoutSignal

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
    'fractal_lookback': 10,
    'progressive_trail': {'enabled': True, 'threshold_pct': 0.9, 'trail_K_post': 0.5},
}

FEE_RT_PCT = 0.10
LEVERAGE = 3
START_BALANCE = 2100.0


def baton_trail_trigger(direction, entry, best, atr, trail_K):
    """Compute baton STOP_MARKET trigger price (no activation gate)."""
    if math.isnan(atr) or atr <= 0:
        return None
    k_atr = trail_K * atr
    if direction == 'LONG':
        disc = best * best - 4 * k_atr * entry
        if disc < 0:
            return None
        trigger = (best + math.sqrt(disc)) / 2
        return round(trigger, 1)
    else:
        disc = best * best + 4 * k_atr * entry
        trigger = (best + math.sqrt(disc)) / 2
        return round(trigger, 1)


def get_effective_trail_k(best_pnl, cfg):
    pt = cfg.get('progressive_trail', {}) or {}
    if pt.get('enabled', False) and best_pnl >= pt.get('threshold_pct', 0.9):
        return pt.get('trail_K_post', 0.5)
    return cfg['trail_K']


def baton_check_exit(direction, entry, best, high, low, sl_price, atr,
                    bars_held, cfg):
    """Intrabar-touch baton exit logic.

    Priority: SL > Emergency > Timeout > Baton trail
    """
    emergency_pct = cfg['emergency_sl_pct']
    max_hold = cfg['max_hold_bars']

    # 1) Fractal SL (intrabar touch)
    if direction == 'LONG':
        if low <= sl_price:
            return {'exit_price': sl_price, 'reason': 'SL', 'priority': 1}
    else:
        if high >= sl_price:
            return {'exit_price': sl_price, 'reason': 'SL', 'priority': 1}

    # 2) Emergency SL
    if direction == 'LONG':
        worst_pnl = (low / entry - 1) * 100
        if worst_pnl <= -emergency_pct:
            px = entry * (1 - emergency_pct / 100)
            return {'exit_price': px, 'reason': 'EMERGENCY', 'priority': 2}
    else:
        worst_pnl = (1 - high / entry) * 100
        if worst_pnl <= -emergency_pct:
            px = entry * (1 + emergency_pct / 100)
            return {'exit_price': px, 'reason': 'EMERGENCY', 'priority': 2}

    # 3) Timeout (at bar close approximation)
    if bars_held >= max_hold:
        return {'exit_price': (high + low) / 2, 'reason': 'TIMEOUT', 'priority': 3}

    # 4) Baton trail (no activation gate — E option)
    if direction == 'LONG':
        best_pnl = (best / entry - 1) * 100
    else:
        best_pnl = (1 - best / entry) * 100

    trail_K = get_effective_trail_k(best_pnl, cfg)
    trigger = baton_trail_trigger(direction, entry, best, atr, trail_K)
    if trigger is None:
        return None

    # Tighter-of-two: baton vs fractal SL
    if direction == 'LONG':
        effective_trig = max(trigger, sl_price)
        if low <= effective_trig:
            return {'exit_price': effective_trig, 'reason': 'BATON_TRAIL', 'priority': 4}
    else:
        effective_trig = min(trigger, sl_price)
        if high >= effective_trig:
            return {'exit_price': effective_trig, 'reason': 'BATON_TRAIL', 'priority': 4}

    return None


def fetch_candles_ccxt(start_dt, end_dt):
    import ccxt
    exchange = ccxt.bingx({'options': {'defaultType': 'swap'}})
    start_ts = int(start_dt.replace(tzinfo=timezone.utc).timestamp() * 1000)
    end_ts = int(end_dt.replace(tzinfo=timezone.utc).timestamp() * 1000)
    all_c, since = [], start_ts
    while since < end_ts:
        cs = exchange.fetch_ohlcv('BTC-USDT', '15m', since=since, limit=1000)
        if not cs: break
        all_c.extend(cs)
        if cs[-1][0] <= since: break
        since = cs[-1][0] + 1
    seen, uniq = set(), []
    for c in all_c:
        if c[0] not in seen:
            seen.add(c[0]); uniq.append(c)
    uniq = [c for c in uniq if start_ts <= c[0] < end_ts]
    uniq.sort(key=lambda x: x[0])
    return uniq


def load_candles_csv(path):
    import pandas as pd
    df = pd.read_csv(path, parse_dates=['timestamp'])
    df = df.sort_values('timestamp').reset_index(drop=True)
    df = df.set_index('timestamp')
    df15 = df.resample('15min').agg({
        'open': 'first', 'high': 'max', 'low': 'min',
        'close': 'last', 'volume': 'sum',
    }).dropna(subset=['open']).reset_index()
    candles = [
        [int(row['timestamp'].timestamp()*1000), row['open'], row['high'],
         row['low'], row['close'], row.get('volume', 0)]
        for _, row in df15.iterrows()
    ]
    return candles


def run_bt(candles, eval_start, eval_end, cfg, mode='baton'):
    """mode='baton': E option (intrabar baton-touch).
       mode='classic': baseline BT (bar-close trail via check_exit)."""
    signal = C1BreakoutSignal(cfg)

    ts = [datetime.fromtimestamp(c[0]/1000, tz=timezone.utc).replace(tzinfo=None) for c in candles]
    opens = [c[1] for c in candles]
    highs = [c[2] for c in candles]
    lows = [c[3] for c in candles]
    closes = [c[4] for c in candles]
    n = len(closes)

    atr = compute_atr(highs, lows, closes, cfg['atr_period'])
    ch_h, ch_l = compute_channel(highs, lows, cfg['channel_period'])
    sw_l, sw_h = compute_fractal_swings(highs, lows, cfg['fractal_lookback'])

    s_idx = next((i for i, t in enumerate(ts) if t >= eval_start), None)
    if s_idx is None:
        return []
    e_idx = next((i for i in range(n-1, -1, -1) if ts[i] < eval_end), n-1)

    trades = []
    in_pos, cd = False, 0
    pdir = pprice = ptime = psl = pbest = None
    pheld = 0

    for i in range(s_idx, e_idx + 1):
        if in_pos:
            pheld += 1
            pbest = max(pbest, highs[i]) if pdir == 'LONG' else min(pbest, lows[i])

            if mode == 'baton':
                er = baton_check_exit(
                    direction=pdir, entry=pprice, best=pbest,
                    high=highs[i], low=lows[i], sl_price=psl,
                    atr=atr[i] if not math.isnan(atr[i]) else atr[i-1],
                    bars_held=pheld, cfg=cfg,
                )
            else:
                er = signal.check_exit(
                    direction=pdir, entry_price=pprice, best_price=pbest,
                    current_high=highs[i], current_low=lows[i], current_close=closes[i],
                    sl_price=psl,
                    atr_val=atr[i] if not math.isnan(atr[i]) else atr[i-1],
                    bars_held=pheld,
                )
            if er:
                xp, rs = er['exit_price'], er['reason']
                pnl = (xp/pprice - 1)*100 if pdir == 'LONG' else (1 - xp/pprice)*100
                pnl -= FEE_RT_PCT
                trades.append({
                    'dir': pdir, 'entry': round(pprice,1), 'exit': round(xp,1),
                    'pnl1x': round(pnl,4), 'pnl3x': round(pnl*LEVERAGE,4),
                    'reason': rs, 'bars': pheld, 't_exit': str(ts[i])[:19],
                })
                in_pos, cd, pdir = False, i + cfg['min_bars_between'], None

        if not in_pos and i >= cd and i < e_idx:
            if math.isnan(atr[i]) or math.isnan(ch_h[i]):
                continue
            es = signal.check_entry(
                bar_open=opens[i], bar_high=highs[i], bar_low=lows[i], bar_close=closes[i],
                channel_high=ch_h[i], channel_low=ch_l[i], atr_val=atr[i],
                last_swing_low=sw_l[i], last_swing_high=sw_h[i],
            )
            if es:
                ni = i + 1
                if ni > e_idx: continue
                pdir = es['direction']
                pprice, ptime, psl = opens[ni], ts[ni], es['sl_price']
                pheld, in_pos = 0, True
                pbest = highs[ni] if pdir == 'LONG' else lows[ni]

                if mode == 'baton':
                    er = baton_check_exit(
                        direction=pdir, entry=pprice, best=pbest,
                        high=highs[ni], low=lows[ni], sl_price=psl,
                        atr=atr[ni] if not math.isnan(atr[ni]) else atr[i],
                        bars_held=0, cfg=cfg,
                    )
                else:
                    er = signal.check_exit(
                        direction=pdir, entry_price=pprice, best_price=pbest,
                        current_high=highs[ni], current_low=lows[ni], current_close=closes[ni],
                        sl_price=psl,
                        atr_val=atr[ni] if not math.isnan(atr[ni]) else atr[i],
                        bars_held=0,
                    )
                if er:
                    xp, rs = er['exit_price'], er['reason']
                    pnl = (xp/pprice - 1)*100 if pdir == 'LONG' else (1 - xp/pprice)*100
                    pnl -= FEE_RT_PCT
                    trades.append({
                        'dir': pdir, 'entry': round(pprice,1), 'exit': round(xp,1),
                        'pnl1x': round(pnl,4), 'pnl3x': round(pnl*LEVERAGE,4),
                        'reason': rs, 'bars': 0, 't_exit': str(ts[ni])[:19],
                    })
                    in_pos, cd, pdir = False, ni + cfg['min_bars_between'], None

    return trades


def stats(trades, start=START_BALANCE):
    if not trades:
        return {'trades': 0}
    n = len(trades)
    wins = sum(1 for t in trades if t['pnl3x'] > 0)
    sum3x = sum(t['pnl3x'] for t in trades)
    sum1x = sum(t['pnl1x'] for t in trades)
    bal = start; peak = start; mdd = 0.0
    for t in trades:
        bal *= (1 + t['pnl3x']/100)
        if bal > peak: peak = bal
        dd = (bal - peak) / peak * 100
        if dd < mdd: mdd = dd
    reasons = {}
    for t in trades:
        reasons[t['reason']] = reasons.get(t['reason'], 0) + 1
    return {
        'trades': n, 'wr_pct': round(100*wins/n, 2),
        'sum_pnl_1x': round(sum1x, 4),
        'sum_pnl_3x': round(sum3x, 4),
        'end_bal': round(bal, 2),
        'net_pct': round(100*(bal/start - 1), 4),
        'mdd_pct': round(mdd, 4),
        'reasons': reasons,
    }


def summarize(label, s, live_end=None):
    reasons = ' '.join(f"{k}:{v}" for k, v in sorted(s.get('reasons', {}).items()))
    print(f"{label:<35} trades={s['trades']:>3} WR={s.get('wr_pct', 0):>5.1f}% "
          f"PnL3x={s.get('sum_pnl_3x', 0):>+8.2f}% endBal=${s.get('end_bal', 0):>8.2f} "
          f"MDD={s.get('mdd_pct', 0):>+6.2f}% | {reasons}")


def main():
    # ── Part 1: 27 trades 구간 (LIVE 비교) ─────────────
    print("=" * 95)
    print("Part 1: 2026-04-12 ~ 2026-04-22 (27 trades 구간, LIVE 비교)")
    print("=" * 95)
    print("Fetching BingX 15m (ccxt)...")
    c27 = fetch_candles_ccxt(datetime(2026, 4, 8), datetime(2026, 4, 22))
    print(f"Got {len(c27)} candles\n")

    t_baton = run_bt(c27, datetime(2026, 4, 12), datetime(2026, 4, 22), CONFIG, mode='baton')
    t_classic = run_bt(c27, datetime(2026, 4, 12), datetime(2026, 4, 22), CONFIG, mode='classic')

    s_baton = stats(t_baton)
    s_classic = stats(t_classic)

    print()
    summarize("27-trade baseline (classic BT)", s_classic)
    summarize("27-trade E-option (baton BT)  ", s_baton)
    print(f"\nLIVE reference (from dd_comparison):")
    print(f"{'LIVE 27 trades':<35} trades= 27 WR= 25.9% PnL3x= -16.09% endBal=$ 1766.86 MDD= -15.86%")

    gap_baton_vs_live = round(s_baton.get('sum_pnl_3x', 0) - (-16.09), 2)
    gap_classic_vs_live = round(s_classic.get('sum_pnl_3x', 0) - (-16.09), 2)
    print(f"\nGap (Classic vs LIVE): {gap_classic_vs_live:+}pp")
    print(f"Gap (Baton vs LIVE):   {gap_baton_vs_live:+}pp  ← E option 기대 갭")

    # ── Part 2: 333일 full BT (baseline preservation) ─────────────
    print()
    print("=" * 95)
    print("Part 2: Full BT (333일) — baseline +170% preservation 검증")
    print("=" * 95)

    csv_path = ROOT / 'data' / 'btc_5m_270days_reclassified.csv'
    if not csv_path.exists():
        print(f"WARN: {csv_path} not found, skipping full BT")
    else:
        print(f"Loading {csv_path.name} + resampling to 15m...")
        c_full = load_candles_csv(str(csv_path))
        print(f"Got {len(c_full)} 15m candles")
        # Use entire dataset
        t_start = datetime.fromtimestamp(c_full[0][0]/1000)
        t_end = datetime.fromtimestamp(c_full[-1][0]/1000)
        print(f"Range: {t_start} ~ {t_end}\n")

        tf_classic = run_bt(c_full, t_start, t_end, CONFIG, mode='classic')
        tf_baton = run_bt(c_full, t_start, t_end, CONFIG, mode='baton')

        sf_classic = stats(tf_classic, start=100.0)  # additive 1x comparison
        sf_baton = stats(tf_baton, start=100.0)

        # Recompute sum_pnl_1x for additive view
        sf_classic['sum_pnl_1x_add'] = round(sum(t['pnl1x'] for t in tf_classic), 2)
        sf_baton['sum_pnl_1x_add'] = round(sum(t['pnl1x'] for t in tf_baton), 2)

        print()
        summarize("Full 333d baseline (classic)  ", sf_classic)
        summarize("Full 333d E-option (baton)    ", sf_baton)
        print(f"\nAdditive 1x PnL (baseline 검증):")
        print(f"  Classic: {sf_classic['sum_pnl_1x_add']:+}% (expected ~+170%)")
        print(f"  Baton:   {sf_baton['sum_pnl_1x_add']:+}% (E option)")

    # Save results
    out = {
        'date': datetime.now().isoformat(),
        'config': CONFIG,
        'part1_27_trades': {
            'classic': {'stats': s_classic, 'trades': t_classic},
            'baton': {'stats': s_baton, 'trades': t_baton},
            'live_ref': {'pnl3x': -16.09, 'trades': 27, 'wr': 25.9},
        },
    }
    path = ROOT / 'results' / f'baton_only_backtest_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
    path.parent.mkdir(exist_ok=True)
    with open(path, 'w') as f:
        json.dump(out, f, indent=2, default=str)
    print(f"\nSaved: {path}")


if __name__ == '__main__':
    main()
