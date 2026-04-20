#!/usr/bin/env python3
"""4-path exploration for F6 (outlier concentration) challenge.

Path A: Mean Reversion strategy (z-score based)
Path B: Relaxed F6 metric (top 20% remove + Sharpe retention)
Path C: 5m scalping (tighter parameters)
Path D: Current candidate + F6 exception (documentation)

모두 실행 후 비교.
"""
import sys, copy, random, math
from pathlib import Path
from statistics import mean, stdev, median

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))
import os; os.chdir(ROOT)

import pandas as pd
import scripts.analysis.intrabar_trail_impact as ibt
from scripts.production.c1_breakout.signals import C1BreakoutSignal
from scripts.production.c1_breakout.indicators import compute_atr
from scripts.analysis.c1_intrabar_parity import SLIPPAGE as SLIP_MED, apply_slippage
from scripts.analysis.regime_filter_lowvol_study import run_bt_with_regime
from scripts.analysis.regime_filter_trend_study import precompute_trend_pass

DATA_DAYS = ibt.n15 / 96

# ═══════════════════════════════════════════════════════════════════
# Path A — Mean Reversion (z-score Bollinger-like)
# ═══════════════════════════════════════════════════════════════════

def run_mean_reversion(ma_period=20, z_entry=2.0, z_exit=0.3, sl_z=3.5, max_hold=32):
    """Simple mean reversion:
       Entry: z = (close - ma) / std >= z_entry (SHORT) or <= -z_entry (LONG)
       Exit:  z returns to ±z_exit OR SL at z=±sl_z OR max_hold bars
    """
    n = ibt.n15
    c = ibt.c15; o = ibt.o15; h = ibt.h15; l = ibt.l15

    # Rolling mean/std
    ma = [float('nan')] * n
    sd = [float('nan')] * n
    for i in range(ma_period, n):
        window = c[i - ma_period:i]
        m = sum(window) / ma_period
        ma[i] = m
        var = sum((x - m) ** 2 for x in window) / ma_period
        sd[i] = math.sqrt(var)

    trades = []
    pos = None
    last_exit_bar = -3
    WARMUP = max(25, ma_period + 5)

    for bar in range(WARMUP, n - 1):
        if sd[bar] is None or math.isnan(sd[bar]) or sd[bar] <= 0:
            continue

        # Exit check (z-score based)
        if pos is not None:
            pos['bh'] += 1
            z = (c[bar] - ma[bar]) / sd[bar] if sd[bar] > 0 else 0
            # Normalize current z to entry z direction
            dir_sign = 1 if pos['d'] == 'LONG' else -1
            # LONG entered when z < -z_entry; exit when z > -z_exit (i.e., price normalized)
            # Use absolute distance to mean
            exit_reason = None
            exit_price = c[bar]
            if dir_sign * z >= -z_exit:
                exit_reason = 'REVERT'
            # SL at extreme z
            if pos['d'] == 'LONG' and z <= -sl_z:
                exit_reason = 'SL'
                exit_price = ma[bar] - sl_z * sd[bar]
            elif pos['d'] == 'SHORT' and z >= sl_z:
                exit_reason = 'SL'
                exit_price = ma[bar] + sl_z * sd[bar]
            if pos['bh'] >= max_hold:
                exit_reason = 'TIMEOUT'
                exit_price = c[bar]

            if exit_reason:
                raw = (exit_price / pos['ep'] - 1) * 100 if pos['d'] == 'LONG' \
                      else (1 - exit_price / pos['ep']) * 100
                trades.append({
                    'entry_bar': pos['eb'], 'exit_bar': bar,
                    'd': pos['d'], 'ep': pos['ep'], 'xp': exit_price,
                    'raw': raw, 'net': raw - 0.10,
                    'reason': exit_reason, 'bh': pos['bh'],
                })
                pos = None
                last_exit_bar = bar
                continue

        # Entry check
        if pos is None and bar - last_exit_bar >= 2:
            z = (c[bar] - ma[bar]) / sd[bar] if sd[bar] > 0 else 0
            direction = None
            if z <= -z_entry:
                direction = 'LONG'
            elif z >= z_entry:
                direction = 'SHORT'
            if direction and bar + 1 < n:
                pos = {
                    'd': direction, 'ep': o[bar + 1],
                    'eb': bar + 1, 'bh': 0,
                }

    return trades


# ═══════════════════════════════════════════════════════════════════
# Common: Stability gate + bootstrap
# ═══════════════════════════════════════════════════════════════════

def bootstrap_3day(trades, n_samples=1000, seed=42, start_min=220):
    WINDOW = 288
    START_MAX = ibt.n15 - WINDOW - 1
    rng = random.Random(seed)
    pnls = []
    for _ in range(n_samples):
        s = rng.randint(start_min, START_MAX)
        e = s + WINDOW
        pnls.append(sum(t['net'] for t in trades if s <= t['entry_bar'] < e))
    ps = sorted(pnls)
    return {
        'mean': mean(pnls), 'median': median(pnls), 'std': stdev(pnls),
        'pos_pct': sum(1 for p in pnls if p > 0) / n_samples * 100,
        'sharpe': mean(pnls) / stdev(pnls) if stdev(pnls) > 0 else 0,
        'p5': ps[int(0.05 * n_samples)],
        'p_loss_2pp': sum(1 for p in pnls if p < -2) / n_samples * 100,
    }


def full_metrics(name, trades):
    if not trades:
        return {'name': name, 'note': 'no trades', 'trades': 0}
    total = sum(t['net'] for t in trades)
    daily = total / DATA_DAYS
    per_trade = total / len(trades)
    tpd = len(trades) / DATA_DAYS

    # F6 original: top 5% exclusion
    n_top = max(1, int(len(trades) * 0.05))
    sorted_desc = sorted(trades, key=lambda t: t['net'], reverse=True)
    ex_top5 = sum(t['net'] for t in sorted_desc[n_top:])

    # F6 Path B variants
    n_top20 = max(1, int(len(trades) * 0.20))
    ex_top20 = sum(t['net'] for t in sorted_desc[n_top20:])
    # Sharpe retention after top-5% removal
    trades_ex5 = sorted_desc[n_top:]
    bs_full = bootstrap_3day(trades)
    bs_ex5 = bootstrap_3day(trades_ex5) if len(trades_ex5) > 10 else {'sharpe': 0, 'mean': 0, 'pos_pct': 0, 'p5': 0}
    sharpe_retention_pct = (bs_ex5['sharpe'] / bs_full['sharpe'] * 100) if bs_full['sharpe'] > 0 else 0

    # Path B alternative F6: Sharpe retention after top 10% removal
    n_top10 = max(1, int(len(trades) * 0.10))
    trades_ex10 = sorted_desc[n_top10:]
    bs_ex10 = bootstrap_3day(trades_ex10) if len(trades_ex10) > 10 else {'sharpe': 0, 'mean': 0}
    sharpe_retention_10 = (bs_ex10['sharpe'] / bs_full['sharpe'] * 100) if bs_full['sharpe'] > 0 else 0

    return {
        'name': name,
        'trades': len(trades),
        'total_pnl': round(total, 2),
        'daily_pnl': round(daily, 3),
        'per_trade': round(per_trade, 3),
        'trades_per_day': round(tpd, 2),
        'f6_top5_remove': round(ex_top5, 2),
        'f6_top20_remove': round(ex_top20, 2),
        'sharpe_retention_ex5_pct': round(sharpe_retention_pct, 1),
        'sharpe_retention_ex10_pct': round(sharpe_retention_10, 1),
        'boot_mean': round(bs_full['mean'], 3),
        'boot_pos_pct': round(bs_full['pos_pct'], 1),
        'boot_sharpe': round(bs_full['sharpe'], 3),
        'boot_p5': round(bs_full['p5'], 2),
        # Original gate: 6/6
        'gate_6core': {
            'f1_daily>=0.2': daily >= 0.2,
            'f2_per_trade>0': per_trade > 0,
            'f3_tpd>=2': tpd >= 2.0,
            'f4_pos>=55': bs_full['pos_pct'] >= 55,
            'f5_p5>=-3.5': bs_full['p5'] >= -3.5,
            'f6_top5>0': ex_top5 > 0,
        },
        # Relaxed F6 (Path B)
        'gate_pathB': {
            'f1_daily>=0.2': daily >= 0.2,
            'f2_per_trade>0': per_trade > 0,
            'f3_tpd>=2': tpd >= 2.0,
            'f4_pos>=55': bs_full['pos_pct'] >= 55,
            'f5_p5>=-3.5': bs_full['p5'] >= -3.5,
            'f6B_sharpe_retention>=50': sharpe_retention_pct >= 50,
        },
    }


# ═══════════════════════════════════════════════════════════════════
# Path C — 5m timeframe scalping (BT on raw 5m, not resampled)
# ═══════════════════════════════════════════════════════════════════

def run_5m_scalping(channel_p=30, body_min=0.5, trail_k=1.5, max_sl_atr=2.5, max_hold=48):
    """5m scalping — tighter parameters, shorter hold.
       Uses ibt.c5, h5, l5, o5 arrays (raw 5m data).
    """
    n = ibt.n5
    c5 = ibt.c5; o5 = ibt.o5; h5 = ibt.h5; l5 = ibt.l5

    # ATR and channel on 5m
    atr = [float('nan')] * n
    tr = [h5[0] - l5[0]]
    for i in range(1, n):
        tr.append(max(h5[i] - l5[i], abs(h5[i] - c5[i-1]), abs(l5[i] - c5[i-1])))
    period = 14
    if n >= period:
        atr[period-1] = sum(tr[:period]) / period
        for i in range(period, n):
            atr[i] = (atr[i-1] * (period - 1) + tr[i]) / period

    # Channel
    ch_h = [float('nan')] * n
    ch_l = [float('nan')] * n
    for i in range(channel_p, n):
        ch_h[i] = max(h5[i - channel_p:i])
        ch_l[i] = min(l5[i - channel_p:i])

    trades = []
    pos = None
    last_exit_bar = -3
    WARMUP = max(35, channel_p + 5)

    for bar in range(WARMUP, n - 1):
        a = atr[bar]
        if math.isnan(a) or a <= 0: continue

        if pos:
            pos['bh'] += 1
            if pos['d'] == 'LONG': pos['bp'] = max(pos['bp'], h5[bar])
            else: pos['bp'] = min(pos['bp'], l5[bar])

            # Exit priority: SL → Timeout → Trail
            exit_info = None
            if pos['d'] == 'LONG' and l5[bar] <= pos['sl']:
                exit_info = ('SL', pos['sl'])
            elif pos['d'] == 'SHORT' and h5[bar] >= pos['sl']:
                exit_info = ('SL', pos['sl'])
            elif pos['bh'] >= max_hold:
                exit_info = ('TIMEOUT', c5[bar])
            else:
                if pos['d'] == 'LONG':
                    best_pnl = (pos['bp']/pos['ep']-1)*100
                    cur_pnl = (c5[bar]/pos['ep']-1)*100
                else:
                    best_pnl = (1-pos['bp']/pos['ep'])*100
                    cur_pnl = (1-c5[bar]/pos['ep'])*100
                if best_pnl > 0.05:
                    td = trail_k * a / c5[bar] * 100
                    if best_pnl - cur_pnl >= td:
                        realized = max(0, best_pnl - td)
                        ep = pos['ep'] * (1 + realized/100) if pos['d']=='LONG' else pos['ep'] * (1 - realized/100)
                        exit_info = ('TRAIL_TP', ep)

            if exit_info:
                reason, xp = exit_info
                raw = (xp/pos['ep']-1)*100 if pos['d']=='LONG' else (1-xp/pos['ep'])*100
                trades.append({'entry_bar':pos['eb'], 'exit_bar':bar, 'd':pos['d'],
                               'ep':pos['ep'], 'xp':xp, 'raw':raw, 'net':raw-0.10,
                               'reason':reason, 'bh':pos['bh']})
                pos = None
                last_exit_bar = bar
                continue

        # Entry: Channel breakout + body filter
        if pos is None and bar - last_exit_bar >= 2:
            if math.isnan(ch_h[bar]) or math.isnan(ch_l[bar]): continue
            direction = None
            if c5[bar] > ch_h[bar]: direction = 'LONG'
            elif c5[bar] < ch_l[bar]: direction = 'SHORT'
            if direction:
                rng = h5[bar] - l5[bar]
                if rng <= 0: continue
                body = c5[bar] - o5[bar]
                if abs(body)/rng < body_min: continue
                if direction == 'LONG' and body <= 0: continue
                if direction == 'SHORT' and body >= 0: continue
                ep = o5[bar+1]
                sl_dist = max_sl_atr * a
                sl = ep - sl_dist if direction == 'LONG' else ep + sl_dist
                sl_pct = abs(ep - sl) / ep * 100
                if sl_pct < 0.10 or sl_pct > 2.0: continue
                pos = {'d':direction, 'ep':ep, 'sl':sl, 'bp':ep, 'bh':0, 'eb':bar+1}

    # NOTE: entry_bar here is 5m index. For compatibility with bootstrap (which uses 15m bar),
    # we need to map: 15m_bar_idx = 5m_bar_idx // 3
    # But this makes comparison across strategies difficult.
    # Keep 5m indexing but adjust bootstrap window.
    return trades


def bootstrap_3day_5m(trades, n_samples=1000, seed=42):
    """Bootstrap on 5m indexed trades. Window = 3 days × 288 5m bars = 864 bars."""
    WINDOW = 864
    START_MIN = 220 * 3
    START_MAX = ibt.n5 - WINDOW - 1
    rng = random.Random(seed)
    pnls = []
    for _ in range(n_samples):
        s = rng.randint(START_MIN, START_MAX)
        e = s + WINDOW
        pnls.append(sum(t['net'] for t in trades if s <= t['entry_bar'] < e))
    ps = sorted(pnls)
    return {
        'mean': mean(pnls), 'median': median(pnls), 'std': stdev(pnls),
        'pos_pct': sum(1 for p in pnls if p > 0) / n_samples * 100,
        'sharpe': mean(pnls) / stdev(pnls) if stdev(pnls) > 0 else 0,
        'p5': ps[int(0.05 * n_samples)],
    }


# ═══════════════════════════════════════════════════════════════════
# Main orchestration
# ═══════════════════════════════════════════════════════════════════

def main():
    print('=' * 100)
    print('  4-Path F6 Challenge — Exploration & Comparison')
    print(f'  Data: {ibt.n15} 15m bars, {ibt.n5} 5m bars, {DATA_DAYS:.1f} days')
    print('=' * 100)

    # === Path D: Current candidate (baseline for comparison) ===
    print('\n--- Path D: Current candidate (cand_C_b0.60 + trend_filter 1.0/192) ---')
    CAND = {'max_sl_atr':4.0,'trail_K':2.5,'max_hold_bars':192,'body_min_ratio':0.60}
    ibt.strat = copy.deepcopy(ibt.strat); ibt.strat.update(CAND)
    ibt.trail_K = 2.5; ibt.max_hold = 192; ibt.sig = C1BreakoutSignal(ibt.strat)
    passes = precompute_trend_pass(192, 1.0)
    cand_trades = run_bt_with_regime(mode='5m', regime_passes=passes, slippage=SLIP_MED)
    r_d = full_metrics('PathD_cand_C_b0.60+trend', cand_trades)
    print(f'  trades={r_d["trades"]}, PnL={r_d["total_pnl"]}, daily={r_d["daily_pnl"]}, '
          f'boot_sharpe={r_d["boot_sharpe"]}')
    print(f'  F6 (top5 remove): {r_d["f6_top5_remove"]:+.2f}, '
          f'F6B (sharpe retain): {r_d["sharpe_retention_ex5_pct"]}%')

    # === Path A: Mean Reversion ===
    print('\n--- Path A: Mean Reversion (z-score) — 다양 파라미터 ---')
    mr_configs = [
        ('MR_ma20_z2.0', 20, 2.0, 0.3, 3.5, 32),
        ('MR_ma20_z1.5', 20, 1.5, 0.3, 3.0, 32),
        ('MR_ma40_z2.0', 40, 2.0, 0.3, 3.5, 48),
        ('MR_ma20_z2.5', 20, 2.5, 0.5, 4.0, 64),
    ]
    mr_results = []
    for name, mp, ze, zx, sz, mh in mr_configs:
        trades = run_mean_reversion(mp, ze, zx, sz, mh)
        # Apply slippage
        from scripts.analysis.c1_intrabar_parity import apply_slippage
        for t in trades:
            entry_adv = SLIP_MED['entry_pct']/100
            if t['d']=='LONG':
                eff = t['ep']*(1+entry_adv)
                raw = (t['xp']/eff-1)*100
            else:
                eff = t['ep']*(1-entry_adv)
                raw = (1-t['xp']/eff)*100
            t['raw'] = raw
            t['net'] = apply_slippage(t, SLIP_MED, 3.0)
        r = full_metrics(name, trades)
        mr_results.append(r)
        print(f'  {name}: trades={r["trades"]}, PnL={r["total_pnl"]:+.2f}, '
              f'daily={r["daily_pnl"]:+.3f}, tpd={r["trades_per_day"]}, '
              f'pos={r["boot_pos_pct"]}%, sharpe={r["boot_sharpe"]}, '
              f'F6(top5)={r["f6_top5_remove"]:+.2f}')

    # === Path C: 5m Scalping ===
    print('\n--- Path C: 5m Scalping (tighter params) ---')
    c_configs = [
        ('SCALP_ch30_body50', 30, 0.50, 1.5, 2.5, 48),
        ('SCALP_ch45_body50', 45, 0.50, 2.0, 3.0, 64),
        ('SCALP_ch20_body60', 20, 0.60, 1.5, 2.5, 36),
    ]
    scalp_results = []
    for name, ch, bm, tk, ms, mh in c_configs:
        trades = run_5m_scalping(ch, bm, tk, ms, mh)
        # Apply slippage
        for t in trades:
            entry_adv = SLIP_MED['entry_pct']/100
            if t['d']=='LONG':
                eff = t['ep']*(1+entry_adv)
                raw = (t['xp']/eff-1)*100
            else:
                eff = t['ep']*(1-entry_adv)
                raw = (1-t['xp']/eff)*100
            t['raw'] = raw
            t['net'] = apply_slippage(t, SLIP_MED, 3.0)
        # Compute metrics with 5m bootstrap (entry_bar is 5m indexed)
        if not trades:
            print(f'  {name}: 0 trades'); continue
        total = sum(t['net'] for t in trades)
        daily = total / DATA_DAYS
        per_trade = total / len(trades)
        tpd = len(trades) / DATA_DAYS
        bs = bootstrap_3day_5m(trades)
        n_top = max(1, int(len(trades)*0.05))
        sd = sorted(trades, key=lambda t: t['net'], reverse=True)
        ex5 = sum(t['net'] for t in sd[n_top:])
        scalp_results.append({'name':name, 'trades':len(trades), 'total':total, 'daily':daily,
                               'tpd':tpd, 'bs':bs, 'ex5':ex5, 'per_trade':per_trade})
        print(f'  {name}: trades={len(trades)}, PnL={total:+.2f}, daily={daily:+.3f}, '
              f'tpd={tpd:.2f}, pos={bs["pos_pct"]:.1f}%, sharpe={bs["sharpe"]:.3f}, '
              f'F6(top5)={ex5:+.2f}')

    # === Path B: Relaxed F6 evaluation on existing combos ===
    print('\n--- Path B: Relaxed F6 (Sharpe retention ex-top5%) on existing C1 combos ---')
    print(f'  {"Combo":<28} {"Sharpe":>7} {"Ex5% Sharpe":>12} {"Retention":>10} {"PathB pass?"}')
    path_b_combos = [
        ('baseline_b0.40',  {'max_sl_atr':3.3,'trail_K':2.5,'max_hold_bars':192,'body_min_ratio':0.40}, False, 1.0, 192),
        ('cand_C_b0.40',    {'max_sl_atr':4.0,'trail_K':2.5,'max_hold_bars':192,'body_min_ratio':0.40}, False, 1.0, 192),
        ('cand_C_b0.60',    {'max_sl_atr':4.0,'trail_K':2.5,'max_hold_bars':192,'body_min_ratio':0.60}, False, 1.0, 192),
        ('cand_C_b0.60+trend', {'max_sl_atr':4.0,'trail_K':2.5,'max_hold_bars':192,'body_min_ratio':0.60}, True, 1.0, 192),
    ]
    for name, cfg, uf, thr, lb in path_b_combos:
        ibt.strat = copy.deepcopy(ibt.strat); ibt.strat.update(cfg)
        ibt.trail_K = cfg['trail_K']; ibt.max_hold = cfg['max_hold_bars']
        ibt.sig = C1BreakoutSignal(ibt.strat)
        p = precompute_trend_pass(lb, thr) if uf else [True]*ibt.n15
        trades = run_bt_with_regime(mode='5m', regime_passes=p, slippage=SLIP_MED)
        r = full_metrics(name, trades)
        b_pass = all(r['gate_pathB'].values())
        shr = r['boot_sharpe']; retn = r['sharpe_retention_ex5_pct']
        tag = 'PASS' if retn >= 50 else 'FAIL'
        print(f'  {name:<28} {shr:>6.3f} {retn:>11.1f}% {tag:>10} {b_pass}')

    # === Final Summary Table ===
    print()
    print('=' * 100)
    print('  FINAL COMPARISON SUMMARY')
    print('=' * 100)
    hdr = 'Approach'
    print(f'{hdr:<28} {"Trades":>7} {"Daily":>7} {"Sharpe":>7} {"F6_ex5":>9} {"pos%":>6}')

    # Path D
    lab_d = 'D:cand_C+trend'
    print(f'{lab_d:<28} {r_d["trades"]:>7} {r_d["daily_pnl"]:>+6.3f} {r_d["boot_sharpe"]:>+6.3f} {r_d["f6_top5_remove"]:>+8.2f} {r_d["boot_pos_pct"]:>5.1f}')

    # Path A MR
    for r in mr_results:
        lab = 'A:' + r['name']
        print(f'{lab:<28} {r["trades"]:>7} {r["daily_pnl"]:>+6.3f} {r["boot_sharpe"]:>+6.3f} {r["f6_top5_remove"]:>+8.2f} {r["boot_pos_pct"]:>5.1f}')

    # Path C Scalping
    for r in scalp_results:
        lab = 'C:' + r['name']
        print(f'{lab:<28} {r["trades"]:>7} {r["daily"]:>+6.3f} {r["bs"]["sharpe"]:>+6.3f} {r["ex5"]:>+8.2f} {r["bs"]["pos_pct"]:>5.1f}')

    print()
    print('주의: Path C는 5m bar 기준이라 trade count가 다름')


if __name__ == '__main__':
    main()
