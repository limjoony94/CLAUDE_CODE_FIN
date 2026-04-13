#!/usr/bin/env python3
"""
C1 Breakout — Fractal Lookback 5 vs 10 Full Comparison
========================================================
Both lookback=5 (backtest standard) and lookback=10 (production)
run through identical validation suite:
  1. Baseline performance (1x additive)
  2. Progressive look-ahead test
  3. 3-Way split (train/valid/test)
  4. Walk-Forward 5-fold
  5. MC Direction Shuffle (999 sims)
  6. Parameter sensitivity (channel ±20%)
  7. Rolling 60-day windows
  8. Top trade removal
"""
import os, sys, math
import pandas as pd, numpy as np
os.chdir(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, '.')

from scripts.production.c1_breakout.indicators import compute_atr, compute_channel, compute_fractal_swings
from scripts.production.c1_breakout.signals import C1BreakoutSignal
from scripts.production.c1_breakout.config import load_config

# ── Data ──
df = pd.read_csv('data/btc_5m_270days_reclassified.csv')
df['timestamp'] = pd.to_datetime(df['timestamp'])
df['group'] = df.index // 3
agg = df.groupby('group').agg(
    timestamp=('timestamp', 'first'), open=('open', 'first'),
    high=('high', 'max'), low=('low', 'min'),
    close=('close', 'last')).reset_index(drop=True)
o = agg['open'].tolist(); h = agg['high'].tolist()
l = agg['low'].tolist(); c = agg['close'].tolist()
ts = agg['timestamp'].values; N = len(c)
FEE = 0.10

cfg = load_config()
sig_template = cfg['strategy']


def bt(oo, hh, ll, cc, lookback, start_offset=0):
    """Run backtest on a slice with given fractal lookback."""
    nn = len(cc)
    a14 = compute_atr(hh, ll, cc, 14)
    ch_h, ch_l = compute_channel(hh, ll, 15)
    sw_l, sw_h = compute_fractal_swings(hh, ll, lookback)
    sig = C1BreakoutSignal(sig_template)

    trades = []; pos = None; last_exit = -3
    for bar in range(26, nn - 1):
        if pos is not None:
            if pos['d'] == 'LONG': pos['bp'] = max(pos['bp'], hh[bar])
            else: pos['bp'] = min(pos['bp'], ll[bar])
            pos['bh'] += 1
            ex = sig.check_exit(pos['d'], pos['ep'], pos['bp'],
                                hh[bar], ll[bar], cc[bar], pos['sl'], a14[bar], pos['bh'])
            if ex:
                xp = ex['exit_price']
                raw = ((xp / pos['ep'] - 1) if pos['d'] == 'LONG'
                       else (1 - xp / pos['ep'])) * 100
                trades.append(dict(raw=raw, r=ex['reason'],
                                   xb=bar + start_offset, eb=pos['ebar'] + start_offset,
                                   d=pos['d'], bh=pos['bh']))
                pos = None; last_exit = bar; continue
        if pos is None and bar - last_exit >= 2:
            e = sig.check_entry(oo[bar], hh[bar], ll[bar], cc[bar],
                                ch_h[bar], ch_l[bar], a14[bar], sw_l[bar], sw_h[bar])
            if e and bar + 1 < nn:
                pos = dict(d=e['direction'], ep=oo[bar + 1], sl=e['sl_price'],
                           bp=oo[bar + 1], bh=0, ebar=bar + 1)
    return trades


def stats(trades):
    if not trades: return dict(T=0, WR=0, RR=0, PnL=0, MDD=0, Daily=0)
    pnls = [t['raw'] - FEE for t in trades]
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p <= 0]
    total = sum(pnls)
    eq = 0; pk = 0; mdd = 0
    for p in pnls:
        eq += p; pk = max(pk, eq); mdd = max(mdd, pk - eq)
    wr = len(wins) / len(pnls) * 100 if pnls else 0
    wa = np.mean(wins) if wins else 0
    la = np.mean(losses) if losses else -1
    rr = abs(wa / la) if la else 0
    nd = N * 15 / 1440
    return dict(T=len(pnls), WR=round(wr, 1), RR=round(rr, 2),
                PnL=round(total, 1), MDD=round(mdd, 1),
                Daily=round(total / nd, 4) if nd else 0)


def run_suite(lookback):
    tag = f"Lookback={lookback}"
    print(f"\n{'#' * 70}")
    print(f"  {tag}")
    print(f"{'#' * 70}")

    # ── 1. Baseline ──
    all_t = bt(o, h, l, c, lookback)
    s = stats(all_t)
    nd = N * 15 / 1440
    print(f"\n  1. Baseline (1x, {nd:.0f}d)")
    print(f"     T={s['T']} WR={s['WR']}% R:R={s['RR']} PnL={s['PnL']:+.1f}% "
          f"MDD={s['MDD']:.1f}% Daily={s['Daily']:+.4f}%")

    # ── 2. Progressive Look-Ahead ──
    print(f"\n  2. Progressive Look-Ahead Test")
    la_ok = 0
    for pct in [0.25, 0.5, 0.75, 1.0]:
        end = int(N * pct)
        tr = bt(o[:end], h[:end], l[:end], c[:end], lookback)
        full_in = [t for t in all_t if t['xb'] < end]
        diff = abs(len(tr) - len(full_in))
        ok = diff <= 1
        if ok: la_ok += 1
        print(f"     {pct * 100:3.0f}%: trunc={len(tr)} full_in={len(full_in)} "
              f"diff={diff} {'OK' if ok else 'MISMATCH'}")
    print(f"     Result: {la_ok}/4 {'PASS' if la_ok == 4 else 'FAIL'}")

    # ── 3. 3-Way Split ──
    print(f"\n  3. 3-Way Split")
    s1 = int(N * 0.5); s2 = int(N * 0.75)
    splits_pass = 0
    for label, a, b in [('Train', 0, s1), ('Valid', s1, s2), ('Test', s2, N)]:
        tr = bt(o[a:b], h[a:b], l[a:b], c[a:b], lookback, a)
        ss = stats(tr)
        days = (b - a) * 15 / 1440
        ok = ss['PnL'] > 0
        if ok: splits_pass += 1
        print(f"     {label:>5} ({days:3.0f}d): T={ss['T']:>4} WR={ss['WR']}% R:R={ss['RR']} "
              f"PnL={ss['PnL']:>+7.1f}% {'PASS' if ok else 'FAIL'}")
    print(f"     Result: {splits_pass}/3 {'ALL PASS' if splits_pass == 3 else 'PARTIAL'}")

    # ── 4. Walk-Forward 5-Fold ──
    print(f"\n  4. Walk-Forward 5-Fold")
    oos_pass = 0; oos_total = 0
    for fi in range(5):
        ie = int(N * (fi + 1) / 6); oe = int(N * (fi + 2) / 6)
        t_is = bt(o[:ie], h[:ie], l[:ie], c[:ie], lookback)
        t_oos = bt(o[ie:oe], h[ie:oe], l[ie:oe], c[ie:oe], lookback, ie)
        si = stats(t_is); so = stats(t_oos)
        ok = so['PnL'] > 0
        if ok: oos_pass += 1
        oos_total += so['PnL']
        print(f"     F{fi + 1}: IS T={si['T']:>4} PnL={si['PnL']:>+7.1f}% | "
              f"OOS T={so['T']:>3} WR={so['WR']}% R:R={so['RR']} "
              f"PnL={so['PnL']:>+6.1f}% [{'P' if ok else 'F'}]")
    print(f"     Result: {oos_pass}/5 PASS, OOS total={oos_total:+.1f}%")

    # ── 5. MC Direction Shuffle (999 sims) ──
    print(f"\n  5. MC Direction Shuffle (999 sims)")
    real_pnl = sum(t['raw'] - FEE for t in all_t)
    better = 0; shuf_pnls = []
    rng = np.random.RandomState(42)
    for _ in range(999):
        sp = sum((t['raw'] if rng.random() > 0.5 else -t['raw']) - FEE for t in all_t)
        shuf_pnls.append(sp)
        if sp >= real_pnl: better += 1
    mc_p = better / 999
    print(f"     Real={real_pnl:+.1f}% Shuffle={np.mean(shuf_pnls):+.1f}% "
          f"p={mc_p:.4f} {'DISC' if mc_p < 0.05 else 'N-DISC'}")

    # ── 6. Parameter Sensitivity (channel ±20%) ──
    print(f"\n  6. Parameter Sensitivity (channel period)")
    sens_all_pass = True
    for ch in [12, 13, 14, 15, 16, 17, 18]:
        ch_h2, ch_l2 = compute_channel(h, l, ch)
        sw_l2, sw_h2 = compute_fractal_swings(h, l, lookback)
        sig2 = C1BreakoutSignal(sig_template)
        a14 = compute_atr(h, l, c, 14)
        trades2 = []; pos = None; last_exit = -3
        for bar in range(26, N - 1):
            if pos is not None:
                if pos['d'] == 'LONG': pos['bp'] = max(pos['bp'], h[bar])
                else: pos['bp'] = min(pos['bp'], l[bar])
                pos['bh'] += 1
                ex = sig2.check_exit(pos['d'], pos['ep'], pos['bp'],
                                     h[bar], l[bar], c[bar], pos['sl'], a14[bar], pos['bh'])
                if ex:
                    raw = ((ex['exit_price'] / pos['ep'] - 1) if pos['d'] == 'LONG'
                           else (1 - ex['exit_price'] / pos['ep'])) * 100
                    trades2.append(dict(raw=raw)); pos = None; last_exit = bar; continue
            if pos is None and bar - last_exit >= 2:
                e = sig2.check_entry(o[bar], h[bar], l[bar], c[bar],
                                     ch_h2[bar], ch_l2[bar], a14[bar], sw_l2[bar], sw_h2[bar])
                if e and bar + 1 < N:
                    pos = dict(d=e['direction'], ep=o[bar + 1], sl=e['sl_price'],
                               bp=o[bar + 1], bh=0)
        s2 = stats(trades2)
        m = ' <<<' if ch == 15 else ''
        ok = s2['PnL'] > 0
        if not ok: sens_all_pass = False
        print(f"     ch={ch:>2}: T={s2['T']:>4} PnL={s2['PnL']:>+7.1f}% "
              f"MDD={s2['MDD']:.1f}% {'PASS' if ok else 'FAIL'}{m}")
    print(f"     Plateau: {'YES' if sens_all_pass else 'NO'}")

    # ── 7. Rolling 60-day Windows ──
    print(f"\n  7. Rolling 60-day Windows")
    bars_60d = int(60 * 1440 / 15)
    roll_pass = 0; roll_total = 0
    for wi in range(N // bars_60d):
        a = wi * bars_60d; b = min(a + bars_60d, N)
        tr = bt(o[a:b], h[a:b], l[a:b], c[a:b], lookback, a)
        sr = stats(tr)
        roll_total += 1
        ok = sr['PnL'] > 0
        if ok: roll_pass += 1
        print(f"     W{wi + 1:>2} ({a // (1440 // 15):>3}-{b // (1440 // 15):>3}d): "
              f"T={sr['T']:>3} PnL={sr['PnL']:>+6.1f}% {'PASS' if ok else 'FAIL'}")
    print(f"     Result: {roll_pass}/{roll_total}")

    # ── 8. Top Trade Removal ──
    print(f"\n  8. Top Trade Removal")
    sorted_t = sorted(all_t, key=lambda x: -x['raw'])
    for rm in [0, 1, 3, 5, 10, 20]:
        remaining = sorted_t[rm:]
        sr = stats(remaining)
        print(f"     -{rm:>2}: T={sr['T']:>4} PnL={sr['PnL']:>+7.1f}% "
              f"MDD={sr['MDD']:.1f}%")

    return dict(tag=tag, baseline=s, la_ok=la_ok, splits_pass=splits_pass,
                oos_pass=oos_pass, oos_total=oos_total, mc_p=mc_p,
                sens_plateau=sens_all_pass, roll_pass=roll_pass, roll_total=roll_total)


# ═══════════════════════════════════════════════════════
#  Run both
# ═══════════════════════════════════════════════════════
print("=" * 70)
print("  C1 Breakout — Fractal Lookback 5 vs 10 Full Comparison")
print(f"  Data: {N} bars, {N * 15 / 1440:.0f} days")
print("=" * 70)

r5 = run_suite(5)
r10 = run_suite(10)

# ═══════════════════════════════════════════════════════
#  Side-by-Side Summary
# ═══════════════════════════════════════════════════════
print(f"\n{'=' * 70}")
print("  SIDE-BY-SIDE COMPARISON")
print(f"{'=' * 70}")
print(f"{'Metric':<30} {'Lookback=5':>18} {'Lookback=10':>18}")
print(f"{'-' * 30} {'-' * 18} {'-' * 18}")
b5 = r5['baseline']; b10 = r10['baseline']
print(f"{'Trades':<30} {b5['T']:>18} {b10['T']:>18}")
print(f"{'Win Rate':<30} {str(b5['WR'])+'%':>18} {str(b10['WR'])+'%':>18}")
print(f"{'R:R':<30} {b5['RR']:>18} {b10['RR']:>18}")
print(f"{'PnL (1x additive)':<30} {str(b5['PnL'])+'%':>18} {str(b10['PnL'])+'%':>18}")
print(f"{'MDD':<30} {str(b5['MDD'])+'%':>18} {str(b10['MDD'])+'%':>18}")
print(f"{'Daily':<30} {str(b5['Daily'])+'%':>18} {str(b10['Daily'])+'%':>18}")
print(f"{'Look-ahead test':<30} {str(r5['la_ok'])+'/4':>18} {str(r10['la_ok'])+'/4':>18}")
print(f"{'3-Way split':<30} {str(r5['splits_pass'])+'/3':>18} {str(r10['splits_pass'])+'/3':>18}")
print(f"{'WF 5-fold':<30} {str(r5['oos_pass'])+'/5':>18} {str(r10['oos_pass'])+'/5':>18}")
print(f"{'WF OOS total':<30} {str(round(r5['oos_total'],1))+'%':>18} {str(round(r10['oos_total'],1))+'%':>18}")
print(f"{'MC p-value':<30} {r5['mc_p']:>18.4f} {r10['mc_p']:>18.4f}")
print(f"{'Param plateau':<30} {'YES' if r5['sens_plateau'] else 'NO':>18} {'YES' if r10['sens_plateau'] else 'NO':>18}")
print(f"{'Rolling 60d':<30} {str(r5['roll_pass'])+'/'+str(r5['roll_total']):>18} {str(r10['roll_pass'])+'/'+str(r10['roll_total']):>18}")
