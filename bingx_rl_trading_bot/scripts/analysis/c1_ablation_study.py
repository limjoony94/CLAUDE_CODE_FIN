"""
C1 Ablation Study — 타이밍 + Exit 구조 분리 연구
================================================
Tests which component generates alpha:
  E1: Random direction         (direction noise test)
  E2: Periodic entry (no signal)
  E3: Channel only (no body filter)
  E4: Body only (no channel filter)
  E5: Trail K sensitivity (1.0, 1.5, 2.0, 2.5, 3.0, 4.0)
  E6: Fixed TP/SL (no trail)
  E7: No SL (trail only)
  E8: Cooldown sensitivity (0, 2, 5, 10, 20)

All run on full 332-day BTC 15m data. Additive 1x PnL.
"""
import sys, os, json, math, random
from datetime import datetime
from pathlib import Path
from collections import defaultdict

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

import pandas as pd
from scripts.production.c1_breakout.indicators import (
    compute_atr, compute_channel, compute_fractal_swings
)

# ── Base config ──
BASE_CFG = {
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
}
FEE_RT_PCT = 0.10
LEVERAGE = 3


def load_data():
    csv = ROOT / 'data' / 'btc_5m_270days_reclassified.csv'
    df = pd.read_csv(csv, parse_dates=['timestamp']).sort_values('timestamp').reset_index(drop=True)
    df = df.set_index('timestamp').resample('15min').agg({
        'open': 'first', 'high': 'max', 'low': 'min',
        'close': 'last', 'volume': 'sum',
    }).dropna(subset=['open']).reset_index()
    return df


def default_entry(bar_open, bar_high, bar_low, bar_close,
                  ch_high, ch_low, atr_val, sw_low, sw_high, cfg):
    """Standard C1 entry (normal direction)."""
    if math.isnan(ch_high) or math.isnan(ch_low) or math.isnan(atr_val) or atr_val <= 0:
        return None
    if ch_high <= ch_low:
        return None
    direction = None
    if bar_close > ch_high:
        direction = 'LONG'
    elif bar_close < ch_low:
        direction = 'SHORT'
    if direction is None:
        return None
    rng = bar_high - bar_low
    if rng <= 0:
        return None
    body = bar_close - bar_open
    if abs(body) / rng < cfg['body_min_ratio']:
        return None
    if direction == 'LONG' and body <= 0:
        return None
    if direction == 'SHORT' and body >= 0:
        return None
    entry_approx = bar_close
    if direction == 'LONG':
        fractal_sl = sw_low if not math.isnan(sw_low) else entry_approx - cfg['max_sl_atr'] * atr_val
        atr_sl = entry_approx - cfg['max_sl_atr'] * atr_val
        sl_price = max(fractal_sl, atr_sl)
    else:
        fractal_sl = sw_high if not math.isnan(sw_high) else entry_approx + cfg['max_sl_atr'] * atr_val
        atr_sl = entry_approx + cfg['max_sl_atr'] * atr_val
        sl_price = min(fractal_sl, atr_sl)
    sl_pct = abs(entry_approx - sl_price) / entry_approx * 100
    if sl_pct < cfg['sl_min_pct'] or sl_pct > cfg['sl_max_pct']:
        return None
    return {'direction': direction, 'sl_price': sl_price, 'sl_pct': sl_pct}


def check_exit_std(direction, entry, best, hi, lo, cl, sl, atr, bars, cfg):
    """Standard C1 exit: SL → Emergency → Timeout → Trail."""
    if direction == 'LONG' and lo <= sl:
        return {'reason': 'SL', 'exit_price': sl}
    if direction == 'SHORT' and hi >= sl:
        return {'reason': 'SL', 'exit_price': sl}
    worst = (lo / entry - 1) * 100 if direction == 'LONG' else (1 - hi / entry) * 100
    if worst <= -cfg['emergency_sl_pct']:
        ep = entry * (1 - cfg['emergency_sl_pct'] / 100) if direction == 'LONG' \
             else entry * (1 + cfg['emergency_sl_pct'] / 100)
        return {'reason': 'EMERGENCY', 'exit_price': ep}
    if bars >= cfg['max_hold_bars']:
        return {'reason': 'TIMEOUT', 'exit_price': cl}
    # Trail
    if direction == 'LONG':
        best_pnl = (best / entry - 1) * 100
        cur_pnl = (cl / entry - 1) * 100
    else:
        best_pnl = (1 - best / entry) * 100
        cur_pnl = (1 - cl / entry) * 100
    if (best_pnl > cfg['trail_activation_pct']
            and not math.isnan(atr) and atr > 0 and cl > 0):
        td = cfg['trail_K'] * atr / cl * 100
        if best_pnl - cur_pnl >= td:
            realized = max(0, best_pnl - td)
            ep = entry * (1 + realized / 100) if direction == 'LONG' \
                 else entry * (1 - realized / 100)
            return {'reason': 'TRAIL_TP', 'exit_price': ep}
    return None


def run_backtest(df15, cfg, entry_fn=None, exit_fn=None,
                 random_direction=False, periodic_entry=None,
                 seed=None):
    """Generic backtest with pluggable entry/exit."""
    if seed is not None:
        random.seed(seed)
    opens = df15['open'].tolist()
    highs = df15['high'].tolist()
    lows = df15['low'].tolist()
    closes = df15['close'].tolist()
    timestamps = df15['timestamp'].tolist()
    n = len(closes)
    atr = compute_atr(highs, lows, closes, cfg['atr_period'])
    ch_high, ch_low = compute_channel(highs, lows, cfg['channel_period'])
    sw_low, sw_high = compute_fractal_swings(highs, lows, cfg['fractal_lookback'])

    entry_fn = entry_fn or default_entry
    exit_fn = exit_fn or check_exit_std

    trades = []
    in_pos = False
    cd_until = 0
    pd_, pep, pet, pspr, pbp, pbh = None, None, None, None, None, 0

    for i in range(50, n):
        if in_pos:
            pbh += 1
            if pd_ == 'LONG':
                pbp = max(pbp, highs[i])
            else:
                pbp = min(pbp, lows[i])
            ex = exit_fn(pd_, pep, pbp, highs[i], lows[i], closes[i], pspr,
                        atr[i] if not math.isnan(atr[i]) else atr[i-1],
                        pbh, cfg)
            if ex:
                ep = ex['exit_price']
                pnl = (ep/pep - 1)*100 if pd_ == 'LONG' else (1 - ep/pep)*100
                pnl -= FEE_RT_PCT
                trades.append({
                    'direction': pd_, 'entry_time': str(pet),
                    'entry_price': round(pep, 1), 'exit_price': round(ep, 1),
                    'pnl_pct': round(pnl, 4), 'reason': ex['reason'],
                    'bars_held': pbh,
                })
                in_pos = False
                cd_until = i + cfg['min_bars_between']
                pd_ = None

        if not in_pos and i >= cd_until and i < n - 1:
            if math.isnan(atr[i]) or math.isnan(ch_high[i]):
                continue
            if periodic_entry is not None:
                # Fire every N bars alternating LONG/SHORT
                if (i - 50) % periodic_entry == 0:
                    direction = 'LONG' if (len(trades) % 2 == 0) else 'SHORT'
                    entry_approx = closes[i]
                    sl_distance = cfg['max_sl_atr'] * atr[i]
                    sl_price = entry_approx - sl_distance if direction == 'LONG' \
                               else entry_approx + sl_distance
                    sl_pct_ = abs(entry_approx - sl_price) / entry_approx * 100
                    if sl_pct_ < cfg['sl_min_pct'] or sl_pct_ > cfg['sl_max_pct']:
                        continue
                    sig = {'direction': direction, 'sl_price': sl_price, 'sl_pct': sl_pct_}
                else:
                    continue
            else:
                sig = entry_fn(opens[i], highs[i], lows[i], closes[i],
                               ch_high[i], ch_low[i], atr[i],
                               sw_low[i], sw_high[i], cfg)
            if sig is None:
                continue
            if random_direction:
                # Flip direction with 50% probability
                if random.random() < 0.5:
                    d = sig['direction']
                    entry_approx = closes[i]
                    # Recompute SL on the flipped side using ATR cap
                    sl_distance = cfg['max_sl_atr'] * atr[i]
                    new_d = 'SHORT' if d == 'LONG' else 'LONG'
                    new_sl = entry_approx - sl_distance if new_d == 'LONG' \
                             else entry_approx + sl_distance
                    new_sl_pct = abs(entry_approx - new_sl) / entry_approx * 100
                    if new_sl_pct < cfg['sl_min_pct'] or new_sl_pct > cfg['sl_max_pct']:
                        continue
                    sig = {'direction': new_d, 'sl_price': new_sl, 'sl_pct': new_sl_pct}
            ni = i + 1
            if ni >= n:
                continue
            pd_ = sig['direction']
            pep = opens[ni]
            pet = timestamps[ni]
            pspr = sig['sl_price']
            pbh = 0
            in_pos = True
            pbp = highs[ni] if pd_ == 'LONG' else lows[ni]
            # immediate exit on entry bar
            ex = exit_fn(pd_, pep, pbp, highs[ni], lows[ni], closes[ni], pspr,
                        atr[ni] if not math.isnan(atr[ni]) else atr[i], 0, cfg)
            if ex:
                ep = ex['exit_price']
                pnl = (ep/pep - 1)*100 if pd_ == 'LONG' else (1 - ep/pep)*100
                pnl -= FEE_RT_PCT
                trades.append({
                    'direction': pd_, 'entry_time': str(pet),
                    'entry_price': round(pep, 1), 'exit_price': round(ep, 1),
                    'pnl_pct': round(pnl, 4), 'reason': ex['reason'],
                    'bars_held': 0,
                })
                in_pos = False
                cd_until = ni + cfg['min_bars_between']
                pd_ = None
    return trades


def summarize(trades):
    if not trades:
        return {'trades': 0, 'WR': 0, 'PnL_1x': 0, 'MDD_1x': 0, 'RR': 0}
    n = len(trades)
    wins = sum(1 for t in trades if t['pnl_pct'] > 0)
    pnl = sum(t['pnl_pct'] for t in trades)
    avg_win = sum(t['pnl_pct'] for t in trades if t['pnl_pct'] > 0) / max(wins, 1)
    avg_loss = sum(t['pnl_pct'] for t in trades if t['pnl_pct'] <= 0) / max(n - wins, 1)
    rr = abs(avg_win / avg_loss) if avg_loss else 0
    eq = 0; peak = 0; mdd = 0
    for t in trades:
        eq += t['pnl_pct']
        peak = max(peak, eq)
        mdd = max(mdd, peak - eq)
    return {'trades': n, 'WR': round(wins/n*100, 1),
            'PnL_1x': round(pnl, 2), 'MDD_1x': round(mdd, 2),
            'RR': round(rr, 2)}


# ── Entry variants ──

def entry_channel_only(bo, bh, bl, bc, ch, cl, atr, swl, swh, cfg):
    """Channel breakout WITHOUT body filter."""
    if math.isnan(ch) or math.isnan(cl) or math.isnan(atr) or atr <= 0:
        return None
    if ch <= cl:
        return None
    direction = None
    if bc > ch:
        direction = 'LONG'
    elif bc < cl:
        direction = 'SHORT'
    if direction is None:
        return None
    # Skip body filter
    entry_approx = bc
    if direction == 'LONG':
        fs = swl if not math.isnan(swl) else entry_approx - cfg['max_sl_atr']*atr
        atr_sl = entry_approx - cfg['max_sl_atr']*atr
        sl = max(fs, atr_sl)
    else:
        fs = swh if not math.isnan(swh) else entry_approx + cfg['max_sl_atr']*atr
        atr_sl = entry_approx + cfg['max_sl_atr']*atr
        sl = min(fs, atr_sl)
    sp = abs(entry_approx - sl) / entry_approx * 100
    if sp < cfg['sl_min_pct'] or sp > cfg['sl_max_pct']:
        return None
    return {'direction': direction, 'sl_price': sl, 'sl_pct': sp}


def entry_body_only(bo, bh, bl, bc, ch, cl, atr, swl, swh, cfg):
    """Body strength WITHOUT channel filter — direction = body sign."""
    if math.isnan(atr) or atr <= 0:
        return None
    rng = bh - bl
    if rng <= 0:
        return None
    body = bc - bo
    if abs(body) / rng < cfg['body_min_ratio']:
        return None
    direction = 'LONG' if body > 0 else 'SHORT'
    entry_approx = bc
    sl_distance = cfg['max_sl_atr'] * atr
    sl = entry_approx - sl_distance if direction == 'LONG' else entry_approx + sl_distance
    sp = abs(entry_approx - sl) / entry_approx * 100
    if sp < cfg['sl_min_pct'] or sp > cfg['sl_max_pct']:
        return None
    return {'direction': direction, 'sl_price': sl, 'sl_pct': sp}


# ── Exit variants ──

def exit_fixed_tp_sl(direction, entry, best, hi, lo, cl, sl, atr, bars, cfg):
    """Fixed 2% TP / standard SL — no trailing."""
    if direction == 'LONG' and lo <= sl:
        return {'reason': 'SL', 'exit_price': sl}
    if direction == 'SHORT' and hi >= sl:
        return {'reason': 'SL', 'exit_price': sl}
    tp_pct = cfg.get('fixed_tp_pct', 2.0)
    if direction == 'LONG':
        tp_price = entry * (1 + tp_pct / 100)
        if hi >= tp_price:
            return {'reason': 'FIXED_TP', 'exit_price': tp_price}
    else:
        tp_price = entry * (1 - tp_pct / 100)
        if lo <= tp_price:
            return {'reason': 'FIXED_TP', 'exit_price': tp_price}
    if bars >= cfg['max_hold_bars']:
        return {'reason': 'TIMEOUT', 'exit_price': cl}
    return None


def exit_no_sl(direction, entry, best, hi, lo, cl, sl, atr, bars, cfg):
    """Trail only — no SL (just emergency 3%)."""
    worst = (lo / entry - 1) * 100 if direction == 'LONG' else (1 - hi / entry) * 100
    if worst <= -cfg['emergency_sl_pct']:
        ep = entry * (1 - cfg['emergency_sl_pct'] / 100) if direction == 'LONG' \
             else entry * (1 + cfg['emergency_sl_pct'] / 100)
        return {'reason': 'EMERGENCY', 'exit_price': ep}
    if bars >= cfg['max_hold_bars']:
        return {'reason': 'TIMEOUT', 'exit_price': cl}
    if direction == 'LONG':
        best_pnl = (best / entry - 1) * 100
        cur_pnl = (cl / entry - 1) * 100
    else:
        best_pnl = (1 - best / entry) * 100
        cur_pnl = (1 - cl / entry) * 100
    if (best_pnl > cfg['trail_activation_pct']
            and not math.isnan(atr) and atr > 0 and cl > 0):
        td = cfg['trail_K'] * atr / cl * 100
        if best_pnl - cur_pnl >= td:
            realized = max(0, best_pnl - td)
            ep = entry * (1 + realized / 100) if direction == 'LONG' \
                 else entry * (1 - realized / 100)
            return {'reason': 'TRAIL_TP', 'exit_price': ep}
    return None


def main():
    df15 = load_data()
    print(f"Bars: {len(df15)} | Period: {df15['timestamp'].iloc[0]} ~ {df15['timestamp'].iloc[-1]}")

    results = {}

    # Baseline
    print("\nRunning BASELINE (C1 normal)...")
    t = run_backtest(df15, BASE_CFG)
    results['BASELINE'] = summarize(t)

    # E1: Random direction (MC: 5 seeds for stability)
    print("\nE1: Random direction (5 seeds)...")
    e1_results = []
    for seed in [1, 2, 3, 42, 100]:
        t = run_backtest(df15, BASE_CFG, random_direction=True, seed=seed)
        e1_results.append(summarize(t))
    # Average
    avg_pnl = sum(r['PnL_1x'] for r in e1_results) / 5
    avg_wr = sum(r['WR'] for r in e1_results) / 5
    avg_mdd = sum(r['MDD_1x'] for r in e1_results) / 5
    results['E1_RANDOM_DIR_avg'] = {'trades': e1_results[0]['trades'],
                                    'WR': round(avg_wr, 1),
                                    'PnL_1x': round(avg_pnl, 2),
                                    'MDD_1x': round(avg_mdd, 2),
                                    'RR': '-'}

    # E2: Periodic entry (every 15 bars = ~3.75h)
    print("\nE2: Periodic entry (every 15 bars)...")
    t = run_backtest(df15, BASE_CFG, periodic_entry=15)
    results['E2_PERIODIC_15bars'] = summarize(t)

    # E3: Channel only (no body filter)
    print("\nE3: Channel only (no body filter)...")
    t = run_backtest(df15, BASE_CFG, entry_fn=entry_channel_only)
    results['E3_CHANNEL_ONLY'] = summarize(t)

    # E4: Body only (no channel)
    print("\nE4: Body only (no channel filter)...")
    t = run_backtest(df15, BASE_CFG, entry_fn=entry_body_only)
    results['E4_BODY_ONLY'] = summarize(t)

    # E5: Trail K sensitivity
    print("\nE5: Trail K sensitivity...")
    for k in [1.0, 1.5, 2.0, 2.5, 3.0, 4.0]:
        cfg = {**BASE_CFG, 'trail_K': k}
        t = run_backtest(df15, cfg)
        results[f'E5_TRAIL_K_{k}'] = summarize(t)

    # E6: Fixed TP/SL
    print("\nE6: Fixed TP 2% / SL std...")
    t = run_backtest(df15, BASE_CFG, exit_fn=exit_fixed_tp_sl)
    results['E6_FIXED_TP_2pct'] = summarize(t)

    # E7: No SL (trail + emergency only)
    print("\nE7: No fractal SL (trail + 3% emergency only)...")
    t = run_backtest(df15, BASE_CFG, exit_fn=exit_no_sl)
    results['E7_NO_SL'] = summarize(t)

    # E8: Cooldown sensitivity
    print("\nE8: Cooldown sensitivity...")
    for cd in [0, 2, 5, 10, 20]:
        cfg = {**BASE_CFG, 'min_bars_between': cd}
        t = run_backtest(df15, cfg)
        results[f'E8_COOLDOWN_{cd}'] = summarize(t)

    # ── Report ──
    print("\n" + "=" * 100)
    print("ABLATION STUDY RESULTS (332 days, additive 1x)")
    print("=" * 100)
    print(f"{'Variant':30s} {'trades':>8s} {'WR':>6s} {'PnL_1x':>10s} {'MDD':>8s} {'RR':>6s}")
    print("-" * 100)
    for name, r in results.items():
        print(f"{name:30s} {r['trades']:>8} {r['WR']:>6} "
              f"{r['PnL_1x']:>10} {r['MDD_1x']:>8} {r['RR']:>6}")

    # Save
    out = {
        'date_run': datetime.now().isoformat(),
        'bars': len(df15),
        'results': results,
    }
    out_path = ROOT / 'results' / 'c1_ablation_study.json'
    with open(out_path, 'w') as f:
        json.dump(out, f, indent=2, default=str)
    print(f"\nSaved: {out_path}")


if __name__ == '__main__':
    main()
