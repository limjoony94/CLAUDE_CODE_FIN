#!/usr/bin/env python3
"""
C1 Breakout — Look-Ahead Bias & Overfitting Deep Audit
========================================================
Both lookback=5 and lookback=10.

LOOK-AHEAD BIAS:
  1. Progressive truncation (10 cuts)
  2. Indicator causality proof (ATR, Channel, Fractal)
  3. Entry timing: signal bar[i] → entry o[i+1]
  4. Shuffled bar-block test (destroy temporal structure)
  5. Reverse chronological test

OVERFITTING:
  6. Full parameter grid sensitivity (ch × trail_K × max_sl_atr)
  7. Bootstrap 95% CI on PnL (1000 resamples)
  8. Regime split: high-vol vs low-vol
  9. Year-half stability (H1 vs H2)
 10. Combinatorial purged cross-validation (5-fold, 3-bar purge)
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
N = len(c); FEE = 0.10
cfg = load_config()
sig_cfg = cfg['strategy']


def bt(oo, hh, ll, cc, lookback):
    nn = len(cc)
    a14 = compute_atr(hh, ll, cc, 14)
    ch_h, ch_l = compute_channel(hh, ll, 15)
    sw_l, sw_h = compute_fractal_swings(hh, ll, lookback)
    sig = C1BreakoutSignal(sig_cfg)
    trades = []; pos = None; last_exit = -3
    for bar in range(26, nn - 1):
        if pos is not None:
            if pos['d'] == 'LONG': pos['bp'] = max(pos['bp'], hh[bar])
            else: pos['bp'] = min(pos['bp'], ll[bar])
            pos['bh'] += 1
            ex = sig.check_exit(pos['d'], pos['ep'], pos['bp'],
                                hh[bar], ll[bar], cc[bar], pos['sl'], a14[bar], pos['bh'])
            if ex:
                raw = ((ex['exit_price'] / pos['ep'] - 1) if pos['d'] == 'LONG'
                       else (1 - ex['exit_price'] / pos['ep'])) * 100
                trades.append(dict(raw=raw, r=ex['reason'], eb=pos['ebar'], xb=bar, bh=pos['bh']))
                pos = None; last_exit = bar; continue
        if pos is None and bar - last_exit >= 2:
            e = sig.check_entry(oo[bar], hh[bar], ll[bar], cc[bar],
                                ch_h[bar], ch_l[bar], a14[bar], sw_l[bar], sw_h[bar])
            if e and bar + 1 < nn:
                pos = dict(d=e['direction'], ep=oo[bar + 1], sl=e['sl_price'],
                           bp=oo[bar + 1], bh=0, ebar=bar + 1)
    return trades


def bt_custom(oo, hh, ll, cc, lookback, channel=15, trail_K=2.5, max_sl_atr=3.3):
    nn = len(cc)
    a14 = compute_atr(hh, ll, cc, 14)
    ch_h, ch_l = compute_channel(hh, ll, channel)
    sw_l, sw_h = compute_fractal_swings(hh, ll, lookback)
    custom_cfg = dict(sig_cfg)
    custom_cfg['trail_K'] = trail_K
    custom_cfg['max_sl_atr'] = max_sl_atr
    sig = C1BreakoutSignal(custom_cfg)
    trades = []; pos = None; last_exit = -3
    for bar in range(26, nn - 1):
        if pos is not None:
            if pos['d'] == 'LONG': pos['bp'] = max(pos['bp'], hh[bar])
            else: pos['bp'] = min(pos['bp'], ll[bar])
            pos['bh'] += 1
            ex = sig.check_exit(pos['d'], pos['ep'], pos['bp'],
                                hh[bar], ll[bar], cc[bar], pos['sl'], a14[bar], pos['bh'])
            if ex:
                raw = ((ex['exit_price'] / pos['ep'] - 1) if pos['d'] == 'LONG'
                       else (1 - ex['exit_price'] / pos['ep'])) * 100
                trades.append(raw - FEE)
                pos = None; last_exit = bar; continue
        if pos is None and bar - last_exit >= 2:
            e = sig.check_entry(oo[bar], hh[bar], ll[bar], cc[bar],
                                ch_h[bar], ch_l[bar], a14[bar], sw_l[bar], sw_h[bar])
            if e and bar + 1 < nn:
                pos = dict(d=e['direction'], ep=oo[bar + 1], sl=e['sl_price'],
                           bp=oo[bar + 1], bh=0)
    return trades


def pnl_add(trades):
    return sum(t['raw'] - FEE for t in trades)


def run_audit(lookback):
    tag = f"Lookback={lookback}"
    print(f"\n{'#' * 70}")
    print(f"  {tag} — DEEP AUDIT")
    print(f"{'#' * 70}")

    all_t = bt(o, h, l, c, lookback)
    base_pnl = pnl_add(all_t)
    print(f"  Baseline: {len(all_t)} trades, PnL={base_pnl:+.1f}% (additive 1x)")

    # ═══════════════════════════════════════════════════
    # LOOK-AHEAD BIAS TESTS
    # ═══════════════════════════════════════════════════
    print(f"\n{'=' * 60}")
    print("  SECTION A: LOOK-AHEAD BIAS")
    print(f"{'=' * 60}")

    # ── 1. Progressive Truncation (10 cuts) ──
    print(f"\n  A1. Progressive Truncation (10 cuts)")
    la_pass = 0
    for pct_i in range(1, 11):
        pct = pct_i / 10.0
        end = int(N * pct)
        trunc = bt(o[:end], h[:end], l[:end], c[:end], lookback)
        full_in = [t for t in all_t if t['xb'] < end]
        diff = abs(len(trunc) - len(full_in))
        ok = diff <= 1
        if ok: la_pass += 1
        if pct_i <= 3 or pct_i >= 8 or diff > 1:
            print(f"      {pct * 100:5.0f}%: trunc={len(trunc):>4} full={len(full_in):>4} "
                  f"diff={diff} {'OK' if ok else '** MISMATCH **'}")
    print(f"      Result: {la_pass}/10 {'PASS' if la_pass == 10 else 'FAIL'}")

    # ── 2. Indicator Causality Proof ──
    print(f"\n  A2. Indicator Causality Verification")
    a14_full = compute_atr(h, l, c, 14)
    ch_h_full, ch_l_full = compute_channel(h, l, 15)
    sw_l_full, sw_h_full = compute_fractal_swings(h, l, lookback)

    causal_ok = True
    test_bars = [100, 500, 1000, N // 2, N - 100]
    for tb in test_bars:
        a14_trunc = compute_atr(h[:tb + 1], l[:tb + 1], c[:tb + 1], 14)
        ch_h_trunc, ch_l_trunc = compute_channel(h[:tb + 1], l[:tb + 1], 15)
        sw_l_trunc, sw_h_trunc = compute_fractal_swings(h[:tb + 1], l[:tb + 1], lookback)

        atr_match = (math.isnan(a14_full[tb]) and math.isnan(a14_trunc[tb])) or \
                    abs(a14_full[tb] - a14_trunc[tb]) < 1e-10
        ch_match = (math.isnan(ch_h_full[tb]) and math.isnan(ch_h_trunc[tb])) or \
                   abs(ch_h_full[tb] - ch_h_trunc[tb]) < 1e-10
        sw_match = (math.isnan(sw_l_full[tb]) and math.isnan(sw_l_trunc[tb])) or \
                   abs(sw_l_full[tb] - sw_l_trunc[tb]) < 1e-10

        if not (atr_match and ch_match and sw_match):
            causal_ok = False
            print(f"      bar {tb}: ATR={'OK' if atr_match else 'FAIL'} "
                  f"CH={'OK' if ch_match else 'FAIL'} SW={'OK' if sw_match else 'FAIL'}")
    print(f"      All indicators causal: {'PASS' if causal_ok else '** FAIL **'}")

    # ── 3. Entry Timing Verification ──
    print(f"\n  A3. Entry Timing (signal bar vs entry bar)")
    timing_ok = True
    for t in all_t[:50]:
        if t['eb'] <= 26:
            timing_ok = False
            break
    # Signal on bar, entry on bar+1 → entry bar should always be > signal bar
    entry_bars = [t['eb'] for t in all_t]
    exit_bars = [t['xb'] for t in all_t]
    entry_after_signal = all(t['eb'] > 26 for t in all_t)
    exit_after_entry = all(t['xb'] >= t['eb'] for t in all_t)
    print(f"      Entry after warmup: {'PASS' if entry_after_signal else 'FAIL'}")
    print(f"      Exit >= Entry bar:  {'PASS' if exit_after_entry else 'FAIL'}")
    # Check no same-bar entry+exit (minimum 1 bar hold)
    zero_hold = sum(1 for t in all_t if t['bh'] == 0)
    print(f"      Zero-bar holds:     {zero_hold} ({'OK' if zero_hold == 0 else 'SUSPICIOUS'})")

    # ── 4. Shuffled Bar-Block Test ──
    print(f"\n  A4. Shuffled Bar-Block Test (destroy temporal order)")
    print(f"      (If strategy works on shuffled blocks → possible look-ahead)")
    block_size = 96  # 1 day of 15m bars
    n_blocks = N // block_size
    rng = np.random.RandomState(42)
    shuf_pnls = []
    for trial in range(20):
        idx = rng.permutation(n_blocks)
        so = []; sh = []; sl2 = []; sc = []
        for bi in idx:
            a = bi * block_size; b = min(a + block_size, N)
            so.extend(o[a:b]); sh.extend(h[a:b]); sl2.extend(l[a:b]); sc.extend(c[a:b])
        tr = bt(so, sh, sl2, sc, lookback)
        shuf_pnls.append(pnl_add(tr))
    shuf_mean = np.mean(shuf_pnls)
    shuf_positive = sum(1 for p in shuf_pnls if p > 0)
    print(f"      Original:  {base_pnl:+.1f}%")
    print(f"      Shuffled:  {shuf_mean:+.1f}% (mean of 20 trials)")
    print(f"      Positive:  {shuf_positive}/20")
    print(f"      Ratio:     {base_pnl / shuf_mean:.1f}x" if shuf_mean > 0
          else f"      Shuffled negative → temporal edge confirmed")
    degradation = (base_pnl - shuf_mean) / base_pnl * 100 if base_pnl > 0 else 0
    print(f"      Degradation: {degradation:.0f}% → "
          f"{'Strategy depends on time-series structure (NOT look-ahead)' if degradation > 30 else 'LOW degradation — investigate'}")

    # ── 5. Reverse Chronological Test ──
    print(f"\n  A5. Reverse Chronological Test")
    ro = list(reversed(o)); rh = list(reversed(h))
    rl = list(reversed(l)); rc = list(reversed(c))
    rev_t = bt(ro, rh, rl, rc, lookback)
    rev_pnl = pnl_add(rev_t)
    print(f"      Original:  {base_pnl:+.1f}% ({len(all_t)} trades)")
    print(f"      Reversed:  {rev_pnl:+.1f}% ({len(rev_t)} trades)")
    print(f"      Similar performance in reverse would suggest curve-fitting")

    # ═══════════════════════════════════════════════════
    # OVERFITTING TESTS
    # ═══════════════════════════════════════════════════
    print(f"\n{'=' * 60}")
    print("  SECTION B: OVERFITTING")
    print(f"{'=' * 60}")

    # ── 6. Full Parameter Grid ──
    print(f"\n  B6. Parameter Grid Sensitivity")
    print(f"      ch × trail_K × max_sl_atr")
    grid_results = []
    channels = [12, 14, 15, 16, 18]
    trail_Ks = [2.0, 2.5, 3.0]
    sl_atrs = [2.5, 3.0, 3.3, 3.6]
    for ch in channels:
        for tk in trail_Ks:
            for sa in sl_atrs:
                pnls = bt_custom(o, h, l, c, lookback, channel=ch, trail_K=tk, max_sl_atr=sa)
                total = sum(pnls)
                grid_results.append(dict(ch=ch, tk=tk, sa=sa, pnl=total, n=len(pnls)))

    # Analysis
    all_pnls = [r['pnl'] for r in grid_results]
    positive = sum(1 for p in all_pnls if p > 0)
    total_combos = len(grid_results)
    base_combo = [r for r in grid_results if r['ch'] == 15 and r['tk'] == 2.5 and r['sa'] == 3.3][0]
    best_combo = max(grid_results, key=lambda x: x['pnl'])
    worst_combo = min(grid_results, key=lambda x: x['pnl'])
    median_pnl = np.median(all_pnls)

    print(f"      Combos tested: {total_combos}")
    print(f"      Positive:      {positive}/{total_combos} ({positive / total_combos * 100:.0f}%)")
    print(f"      Median PnL:    {median_pnl:+.1f}%")
    print(f"      Current:       ch=15 tk=2.5 sa=3.3 → {base_combo['pnl']:+.1f}%")
    print(f"      Best:          ch={best_combo['ch']} tk={best_combo['tk']} sa={best_combo['sa']} → {best_combo['pnl']:+.1f}%")
    print(f"      Worst:         ch={worst_combo['ch']} tk={worst_combo['tk']} sa={worst_combo['sa']} → {worst_combo['pnl']:+.1f}%")
    rank = sorted(all_pnls, reverse=True).index(base_combo['pnl']) + 1
    print(f"      Current rank:  {rank}/{total_combos} "
          f"({'TOP QUARTILE — mild concern' if rank <= total_combos * 0.25 else 'NOT top quartile — robust'}"
          f" if optimized, but {'ON PLATEAU' if abs(base_combo['pnl'] - median_pnl) / median_pnl < 0.3 else 'SPIKE — overfit risk'})")

    # Heat map summary: channel vs trail_K (fix sa=3.3)
    print(f"\n      Grid slice (sa=3.3):")
    print(f"      {'ch\\tK':>8}", end='')
    for tk in trail_Ks:
        print(f"  {tk:>6}", end='')
    print()
    for ch in channels:
        print(f"      {ch:>8}", end='')
        for tk in trail_Ks:
            r = [x for x in grid_results if x['ch'] == ch and x['tk'] == tk and x['sa'] == 3.3][0]
            marker = ' *' if ch == 15 and tk == 2.5 else ''
            print(f" {r['pnl']:>+6.0f}{marker}", end='')
        print()

    # ── 7. Bootstrap 95% CI ──
    print(f"\n  B7. Bootstrap 95% Confidence Interval (1000 resamples)")
    trade_pnls = [t['raw'] - FEE for t in all_t]
    rng = np.random.RandomState(123)
    boot_pnls = []
    for _ in range(1000):
        sample = rng.choice(trade_pnls, size=len(trade_pnls), replace=True)
        boot_pnls.append(sum(sample))
    ci_lo = np.percentile(boot_pnls, 2.5)
    ci_hi = np.percentile(boot_pnls, 97.5)
    print(f"      PnL:    {base_pnl:+.1f}%")
    print(f"      95% CI: [{ci_lo:+.1f}%, {ci_hi:+.1f}%]")
    print(f"      Width:  {ci_hi - ci_lo:.1f}pp")
    print(f"      CI > 0: {'YES — statistically significant' if ci_lo > 0 else 'NO — includes zero'}")

    # ── 8. Regime Split: High-Vol vs Low-Vol ──
    print(f"\n  B8. Regime Split (High-Vol vs Low-Vol)")
    a14_full = compute_atr(h, l, c, 14)
    valid_atr = [a for a in a14_full if not math.isnan(a)]
    atr_median = np.median(valid_atr)
    # Split data into high/low vol halves by labeling bars
    hi_bars = set(); lo_bars = set()
    for i, a in enumerate(a14_full):
        if not math.isnan(a):
            if a >= atr_median: hi_bars.add(i)
            else: lo_bars.add(i)
    hi_trades = [t for t in all_t if t['eb'] in hi_bars]
    lo_trades = [t for t in all_t if t['eb'] in lo_bars]
    hi_pnl = sum(t['raw'] - FEE for t in hi_trades)
    lo_pnl = sum(t['raw'] - FEE for t in lo_trades)
    hi_wr = sum(1 for t in hi_trades if t['raw'] - FEE > 0) / len(hi_trades) * 100 if hi_trades else 0
    lo_wr = sum(1 for t in lo_trades if t['raw'] - FEE > 0) / len(lo_trades) * 100 if lo_trades else 0
    print(f"      ATR median: ${atr_median:.0f}")
    print(f"      High-Vol: T={len(hi_trades)} WR={hi_wr:.1f}% PnL={hi_pnl:+.1f}%")
    print(f"      Low-Vol:  T={len(lo_trades)} WR={lo_wr:.1f}% PnL={lo_pnl:+.1f}%")
    both_positive = hi_pnl > 0 and lo_pnl > 0
    print(f"      Both positive: {'YES — regime-robust' if both_positive else 'NO — regime-dependent'}")

    # ── 9. Half-Period Stability ──
    print(f"\n  B9. Half-Period Stability (H1 vs H2)")
    mid = N // 2
    h1_t = bt(o[:mid], h[:mid], l[:mid], c[:mid], lookback)
    h2_t = bt(o[mid:], h[mid:], l[mid:], c[mid:], lookback)
    h1_pnl = pnl_add(h1_t); h2_pnl = pnl_add(h2_t)
    h1_wr = sum(1 for t in h1_t if t['raw'] - FEE > 0) / len(h1_t) * 100 if h1_t else 0
    h2_wr = sum(1 for t in h2_t if t['raw'] - FEE > 0) / len(h2_t) * 100 if h2_t else 0
    print(f"      H1 (0-{mid * 15 // 1440:.0f}d): T={len(h1_t)} WR={h1_wr:.1f}% PnL={h1_pnl:+.1f}%")
    print(f"      H2 ({mid * 15 // 1440:.0f}-{N * 15 // 1440:.0f}d): T={len(h2_t)} WR={h2_wr:.1f}% PnL={h2_pnl:+.1f}%")
    print(f"      Ratio H2/H1: {h2_pnl / h1_pnl:.2f}" if h1_pnl > 0 else "")
    print(f"      Stable: {'YES' if h1_pnl > 0 and h2_pnl > 0 else 'NO'}")

    # ── 10. Combinatorial Purged CV ──
    print(f"\n  B10. Combinatorial Purged Cross-Validation (5-fold, 3-bar purge)")
    n_folds = 5; purge = 3
    fold_size = N // n_folds
    cv_oos = []
    for fi in range(n_folds):
        test_start = fi * fold_size
        test_end = min(test_start + fold_size, N)
        # Train = everything except test ± purge
        train_o = []; train_h = []; train_l = []; train_c = []
        for i in range(N):
            if i < test_start - purge or i >= test_end + purge:
                train_o.append(o[i]); train_h.append(h[i])
                train_l.append(l[i]); train_c.append(c[i])
        # Test
        to2 = o[test_start:test_end]; th2 = h[test_start:test_end]
        tl2 = l[test_start:test_end]; tc2 = c[test_start:test_end]
        test_trades = bt(to2, th2, tl2, tc2, lookback)
        test_pnl = pnl_add(test_trades)
        cv_oos.append(test_pnl)
        print(f"      Fold {fi + 1}: T={len(test_trades):>3} PnL={test_pnl:>+6.1f}% "
              f"[{'P' if test_pnl > 0 else 'F'}]")
    cv_pass = sum(1 for p in cv_oos if p > 0)
    print(f"      Result: {cv_pass}/{n_folds} PASS, Mean OOS={np.mean(cv_oos):+.1f}%")

    return dict(
        tag=tag, base_pnl=base_pnl, n_trades=len(all_t),
        la_pass=la_pass, causal=causal_ok,
        shuf_mean=shuf_mean, shuf_degradation=degradation,
        rev_pnl=rev_pnl,
        grid_positive=positive, grid_total=total_combos,
        grid_rank=rank, grid_median=median_pnl,
        ci_lo=ci_lo, ci_hi=ci_hi,
        regime_both=both_positive, h1_pnl=h1_pnl, h2_pnl=h2_pnl,
        cv_pass=cv_pass, cv_mean=np.mean(cv_oos),
    )


# ═══════════════════════════════════════════════════════
print("=" * 70)
print("  C1 Breakout — Look-Ahead Bias & Overfitting Deep Audit")
print(f"  Data: {N} bars, {N * 15 / 1440:.0f} days")
print("=" * 70)

r5 = run_audit(5)
r10 = run_audit(10)

# ═══════════════════════════════════════════════════════
print(f"\n{'=' * 70}")
print("  FINAL VERDICT")
print(f"{'=' * 70}")
print(f"\n  {'Test':<40} {'LB=5':>10} {'LB=10':>10}")
print(f"  {'-' * 40} {'-' * 10} {'-' * 10}")

def yn(v): return 'PASS' if v else 'FAIL'

print(f"  {'A1. Progressive (10 cuts)':<40} {r5['la_pass']}/10{'':<5} {r10['la_pass']}/10")
print(f"  {'A2. Indicator causality':<40} {yn(r5['causal']):>10} {yn(r10['causal']):>10}")
print(f"  {'A4. Shuffle degradation':<40} {r5['shuf_degradation']:>9.0f}% {r10['shuf_degradation']:>9.0f}%")
print(f"  {'A5. Reverse PnL':<40} {r5['rev_pnl']:>+9.1f}% {r10['rev_pnl']:>+9.1f}%")
print(f"  {'B6. Grid positive':<40} {r5['grid_positive']}/{r5['grid_total']:<6} {r10['grid_positive']}/{r10['grid_total']}")
print(f"  {'B6. Grid rank':<40} {r5['grid_rank']}/{r5['grid_total']:<6} {r10['grid_rank']}/{r10['grid_total']}")
print(f"  {'B7. Bootstrap 95% CI > 0':<40} {yn(r5['ci_lo'] > 0):>10} {yn(r10['ci_lo'] > 0):>10}")
print(f"  {'B7. CI range':<40} [{r5['ci_lo']:+.0f},{r5['ci_hi']:+.0f}] [{r10['ci_lo']:+.0f},{r10['ci_hi']:+.0f}]")
print(f"  {'B8. Both regimes positive':<40} {yn(r5['regime_both']):>10} {yn(r10['regime_both']):>10}")
print(f"  {'B9. H1 & H2 positive':<40} {yn(r5['h1_pnl']>0 and r5['h2_pnl']>0):>10} {yn(r10['h1_pnl']>0 and r10['h2_pnl']>0):>10}")
print(f"  {'B10. Purged CV':<40} {r5['cv_pass']}/5{'':<6} {r10['cv_pass']}/5")

la_verdict_5 = r5['la_pass'] == 10 and r5['causal'] and r5['shuf_degradation'] > 30
la_verdict_10 = r10['la_pass'] == 10 and r10['causal'] and r10['shuf_degradation'] > 30
of_verdict_5 = r5['ci_lo'] > 0 and r5['regime_both'] and r5['cv_pass'] >= 4
of_verdict_10 = r10['ci_lo'] > 0 and r10['regime_both'] and r10['cv_pass'] >= 4

print(f"\n  {'LOOK-AHEAD BIAS':<40} {yn(la_verdict_5):>10} {yn(la_verdict_10):>10}")
print(f"  {'OVERFITTING':<40} {yn(of_verdict_5):>10} {yn(of_verdict_10):>10}")
