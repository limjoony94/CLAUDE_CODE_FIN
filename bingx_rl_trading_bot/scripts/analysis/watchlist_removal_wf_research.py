"""
Watchlist Pattern Removal — Walk-Forward OOS Validation
========================================================
성과 저조 패턴 7개 제거 효과를 TRUE Walk-Forward로 검증.

기존 교훈:
  - 개별 MC 실패 ≠ 제거 근거 (포트폴리오 분산효과 > 개별 노이즈)
  - 패턴 조합 최적화 = in-sample 과적합 (TRUE WF에서 -13%)
  - "나쁜 패턴"은 기간마다 달라짐

이번 연구:
  1. 7개 워치리스트 패턴 리스트 확인
  2. In-sample (270일) 제거 시뮬레이션 (참고용)
  3. TRUE Walk-Forward (expanding window) OOS 검증
  4. 개별 패턴별 LOO (Leave-One-Out) 제거 WF
  5. Cumulative removal (최악→차악 순서)
  6. Bootstrap 신뢰구간

결론: WF OOS에서 제거가 개선되는지 여부로 판단.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from collections import defaultdict
import json
from datetime import datetime
import sys

sys.path.insert(0, str(Path(__file__).parent.parent / 'production'))
from pattern_5m.indicators import classify_candle
from pattern_5m.constants import (
    PATTERN_OPTIMAL_TPSL, VALIDATED_LONG_PATTERNS, VALIDATED_SHORT_PATTERNS,
    PATTERN_STATS,
)

# ============================================================
# Parameters
# ============================================================
MAX_BARS = 500
FEE_PCT = 0.10
LEVERAGE = 3
MC_SIMS = 10000
N_BOOTSTRAP = 5000

# Watchlist — 2+ issues from individual validation
WATCHLIST = [
    'MD-H-MD',   # H=6/8, MC=0.011, WF=3/5, 24 trades
    'MD-DN-MU',  # H=6/8, EdgeIncon, 8-streak
    'BD-BU-DN',  # H=6/8, MC=0.011, Mon 5/9
    'U-MD-MD',   # H=7/8, Streak8, DEGRADING, LowWR45
    'BD-MU-BD',  # H=7/8, WF=3/5, DEGRADING, 24 trades
    'DN-D-BD',   # H=7/8, MC=0.017, LowWR54
    'DN-MD-DN',  # H=7/8, Streak6, DEGRADING
]

print("=" * 80)
print("Watchlist Pattern Removal — Walk-Forward OOS Validation")
print(f"  Watchlist: {len(WATCHLIST)} patterns")
for w in WATCHLIST:
    print(f"    {w}")
print("=" * 80)

# ============================================================
# Load and classify data
# ============================================================
data_path = Path(__file__).resolve().parent / '../../data/btc_5m_270days_reclassified.csv'
df = pd.read_csv(data_path)
print(f"\nLoaded: {len(df)} bars ({len(df)/288:.0f} days)")

highs = df['high'].values
lows = df['low'].values
opens = df['open'].values
closes = df['close'].values
n_bars = len(df)

df['timestamp'] = pd.to_datetime(df['timestamp'])
dates = df['timestamp'].values

print("Classifying candles...")
body_abs_arr = np.abs(closes - opens)
avg_body_20_arr = pd.Series(body_abs_arr).rolling(20).mean().values

types = []
for i in range(n_bars):
    row = pd.Series({'open': opens[i], 'high': highs[i], 'low': lows[i], 'close': closes[i]})
    avg_b = avg_body_20_arr[i] if not pd.isna(avg_body_20_arr[i]) else 1.0
    types.append(classify_candle(row, avg_b).value)
print("Classification complete.")


# ============================================================
# Build pattern maps
# ============================================================
all_patterns = {}
for p in VALIDATED_LONG_PATTERNS:
    tp, sl = PATTERN_OPTIMAL_TPSL[p]
    all_patterns[p] = ('LONG', tp, sl)
for p in VALIDATED_SHORT_PATTERNS:
    tp, sl = PATTERN_OPTIMAL_TPSL[p]
    all_patterns[p] = ('SHORT', tp, sl)

print(f"\nTotal patterns: {len(all_patterns)} (baseline v1.27.2)")
print(f"Watchlist: {len(WATCHLIST)} patterns to test removal")
wl_set = set(WATCHLIST)
baseline_patterns = dict(all_patterns)
reduced_patterns = {p: v for p, v in all_patterns.items() if p not in wl_set}
print(f"After removal: {len(reduced_patterns)} patterns")


# ============================================================
# Engine functions
# ============================================================
def collect_trades_1pos(pattern_map, start_bar=0, end_bar=None):
    if end_bar is None:
        end_bar = n_bars
    raw_trades = []
    for i in range(max(2, start_bar), min(end_bar, n_bars)):
        pat = f"{types[i-2]}-{types[i-1]}-{types[i]}"
        if pat not in pattern_map:
            continue
        direction, tp, sl = pattern_map[pat]
        if i + 1 >= n_bars:
            continue
        entry = opens[i + 1]
        if entry <= 0:
            continue
        entry_bar = i + 1
        if direction == 'LONG':
            tpp = entry * (1 + tp / 100)
            slp = entry * (1 - sl / 100)
        else:
            tpp = entry * (1 - tp / 100)
            slp = entry * (1 + sl / 100)
        for j in range(i + 2, min(i + 2 + MAX_BARS, n_bars)):
            ht = (highs[j] >= tpp) if direction == 'LONG' else (lows[j] <= tpp)
            hs = (lows[j] <= slp) if direction == 'LONG' else (highs[j] >= slp)
            if ht and hs:
                pnl = (tp if abs(tpp - entry) <= abs(slp - entry) else -sl) * LEVERAGE - FEE_PCT
                raw_trades.append((entry_bar, j, pnl, pat))
                break
            elif ht:
                raw_trades.append((entry_bar, j, tp * LEVERAGE - FEE_PCT, pat))
                break
            elif hs:
                raw_trades.append((entry_bar, j, -sl * LEVERAGE - FEE_PCT, pat))
                break
    raw_trades.sort(key=lambda x: x[0])
    filtered = []
    last_exit = -1
    for eb, xb, pnl, pat in raw_trades:
        if eb > last_exit:
            filtered.append((eb, xb, pnl, pat))
            last_exit = xb
    return filtered


def eval_portfolio(trades):
    if not trades:
        return {'pnl': 0, 'trades': 0, 'wr': 0, 'mdd': 0, 'pf': 0,
                'avg_win': 0, 'avg_loss': 0, 'pnl_mdd': 0, 'max_consec_loss': 0}
    pnl_list = [t[2] for t in trades]
    cum = 0; peak = 0; mdd = 0; wins = 0; wps = []; lps = []
    streak = 0; max_streak = 0
    for p in pnl_list:
        cum += p
        if cum > peak: peak = cum
        dd = peak - cum
        if dd > mdd: mdd = dd
        if p > 0:
            wins += 1; wps.append(p)
            streak = 0
        else:
            lps.append(abs(p))
            streak += 1
            if streak > max_streak: max_streak = streak
    aw = np.mean(wps) if wps else 0
    al = np.mean(lps) if lps else 0
    tw = sum(wps); tl = sum(lps)
    wr = wins / len(pnl_list) * 100 if pnl_list else 0
    return {
        'pnl': round(cum, 2), 'trades': len(pnl_list),
        'wr': round(wr, 2), 'mdd': round(mdd, 2),
        'pf': round(tw / tl, 2) if tl > 0 else 999,
        'avg_win': round(aw, 2), 'avg_loss': round(al, 2),
        'pnl_mdd': round(cum / mdd, 2) if mdd > 0 else 999,
        'max_consec_loss': max_streak,
    }


def mc_portfolio(trades, n_sims=MC_SIMS):
    if not trades:
        return {'p_value': 1.0}
    pnls = np.array([t[2] for t in trades])
    real = np.sum(pnls)
    signs = np.random.choice([-1, 1], size=(n_sims, len(pnls)))
    sims = np.sum(pnls * signs, axis=1)
    p = float(np.mean(sims >= real))
    # MDD distribution
    cum_sims = np.cumsum(pnls * signs, axis=1)
    peak_sims = np.maximum.accumulate(cum_sims, axis=1)
    dd_sims = peak_sims - cum_sims
    mdd_sims = np.max(dd_sims, axis=1)
    return {
        'p_value': round(p, 6),
        'mdd_p50': round(float(np.percentile(mdd_sims, 50)), 1),
        'mdd_p90': round(float(np.percentile(mdd_sims, 90)), 1),
        'mdd_p95': round(float(np.percentile(mdd_sims, 95)), 1),
        'mdd_p99': round(float(np.percentile(mdd_sims, 99)), 1),
    }


# ============================================================
# Phase 1: In-Sample Comparison (270 days, reference only)
# ============================================================
print("\n" + "=" * 80)
print("PHASE 1: In-Sample Comparison (270 days — reference only)")
print("=" * 80)

trades_baseline = collect_trades_1pos(baseline_patterns)
trades_reduced = collect_trades_1pos(reduced_patterns)

eval_base = eval_portfolio(trades_baseline)
eval_red = eval_portfolio(trades_reduced)

# Count how many trades came from watchlist patterns
wl_trade_count = sum(1 for t in trades_baseline if t[3] in wl_set)
wl_win_count = sum(1 for t in trades_baseline if t[3] in wl_set and t[2] > 0)
wl_pnl = sum(t[2] for t in trades_baseline if t[3] in wl_set)
wl_wr = wl_win_count / wl_trade_count * 100 if wl_trade_count > 0 else 0

print(f"\n  Watchlist patterns contribution (in portfolio):")
print(f"    Trades from watchlist: {wl_trade_count}/{eval_base['trades']} ({wl_trade_count/eval_base['trades']*100:.1f}%)")
print(f"    WR from watchlist: {wl_wr:.1f}%")
print(f"    PnL from watchlist: {wl_pnl:+.1f}%")

# Per-watchlist pattern contribution
print(f"\n  Per-pattern contribution (in 1-pos portfolio):")
for wp in WATCHLIST:
    wp_trades = [(eb, xb, pnl) for eb, xb, pnl, pat in trades_baseline if pat == wp]
    wp_pnl = sum(t[2] for t in wp_trades)
    wp_w = sum(1 for t in wp_trades if t[2] > 0)
    wp_wr = wp_w / len(wp_trades) * 100 if wp_trades else 0
    print(f"    {wp:12s}: {len(wp_trades):3d} trades, WR {wp_wr:5.1f}%, PnL {wp_pnl:+7.1f}%")

print(f"\n  Baseline (51 patterns): PnL {eval_base['pnl']:+.1f}%, MDD {eval_base['mdd']:.1f}%, "
      f"PnL/MDD {eval_base['pnl_mdd']:.1f}x, WR {eval_base['wr']:.1f}%, Trades {eval_base['trades']}")
print(f"  Reduced  (44 patterns): PnL {eval_red['pnl']:+.1f}%, MDD {eval_red['mdd']:.1f}%, "
      f"PnL/MDD {eval_red['pnl_mdd']:.1f}x, WR {eval_red['wr']:.1f}%, Trades {eval_red['trades']}")
diff_pnl = eval_red['pnl'] - eval_base['pnl']
diff_ratio = eval_red['pnl_mdd'] - eval_base['pnl_mdd']
print(f"  Delta: PnL {diff_pnl:+.1f}%, PnL/MDD {diff_ratio:+.1f}x")
print(f"  ⚠️  WARNING: In-sample only — NOT actionable (circular reasoning)")

is_results = {
    'baseline': eval_base,
    'reduced': eval_red,
    'delta_pnl': round(diff_pnl, 1),
    'delta_pnl_mdd': round(diff_ratio, 1),
    'watchlist_contribution': {
        'trades': wl_trade_count,
        'wr': round(wl_wr, 1),
        'pnl': round(wl_pnl, 1),
    }
}


# ============================================================
# Phase 2: TRUE Walk-Forward (Expanding Window)
# ============================================================
print("\n" + "=" * 80)
print("PHASE 2: TRUE Walk-Forward (Expanding Window) OOS Validation")
print("  Train on first N days, test on next chunk — expanding forward")
print("=" * 80)

# Split: 4 folds of ~67 days each
bars_per_day = 288
total_days = n_bars // bars_per_day
fold_days = total_days // 5  # ~54 days per fold
n_folds = 4  # 4 OOS folds (first fold = initial train)

wf_results = {'folds': [], 'summary': {}}

print(f"\n  Total: {total_days} days, Fold size: {fold_days} days, OOS folds: {n_folds}")

oos_pnl_base_total = 0
oos_pnl_red_total = 0
fold_wins_base = 0
fold_wins_red = 0

for fold_i in range(n_folds):
    # Expanding window: train on folds 0..fold_i, test on fold_i+1
    train_end_bar = (fold_i + 1) * fold_days * bars_per_day
    oos_start_bar = train_end_bar
    oos_end_bar = min((fold_i + 2) * fold_days * bars_per_day, n_bars)

    if oos_end_bar <= oos_start_bar:
        continue

    train_days = (fold_i + 1) * fold_days
    oos_days = (oos_end_bar - oos_start_bar) // bars_per_day

    # OOS trades for baseline
    oos_trades_base = collect_trades_1pos(baseline_patterns, oos_start_bar, oos_end_bar)
    oos_trades_red = collect_trades_1pos(reduced_patterns, oos_start_bar, oos_end_bar)

    oos_eval_base = eval_portfolio(oos_trades_base)
    oos_eval_red = eval_portfolio(oos_trades_red)

    oos_pnl_base_total += oos_eval_base['pnl']
    oos_pnl_red_total += oos_eval_red['pnl']

    base_better = oos_eval_base['pnl'] >= oos_eval_red['pnl']
    if oos_eval_base['pnl'] > 0: fold_wins_base += 1
    if oos_eval_red['pnl'] > 0: fold_wins_red += 1

    winner = "BASE" if base_better else "REDUCED"
    print(f"\n  Fold {fold_i+1}: Train {train_days}d → OOS {oos_days}d (bars {oos_start_bar}-{oos_end_bar})")
    print(f"    Baseline:  PnL {oos_eval_base['pnl']:+7.1f}%, Trades {oos_eval_base['trades']:3d}, "
          f"WR {oos_eval_base['wr']:5.1f}%, MDD {oos_eval_base['mdd']:5.1f}%")
    print(f"    Reduced:   PnL {oos_eval_red['pnl']:+7.1f}%, Trades {oos_eval_red['trades']:3d}, "
          f"WR {oos_eval_red['wr']:5.1f}%, MDD {oos_eval_red['mdd']:5.1f}%")
    print(f"    Winner: {winner} (delta: {oos_eval_red['pnl'] - oos_eval_base['pnl']:+.1f}%)")

    wf_results['folds'].append({
        'fold': fold_i + 1,
        'train_days': train_days,
        'oos_days': oos_days,
        'baseline': oos_eval_base,
        'reduced': oos_eval_red,
        'winner': winner,
        'delta_pnl': round(oos_eval_red['pnl'] - oos_eval_base['pnl'], 2),
    })

wf_oos_delta = oos_pnl_red_total - oos_pnl_base_total
base_fold_wins = sum(1 for f in wf_results['folds'] if f['winner'] == 'BASE')
red_fold_wins = sum(1 for f in wf_results['folds'] if f['winner'] == 'REDUCED')

print(f"\n  === WF OOS SUMMARY ===")
print(f"  Baseline OOS total PnL: {oos_pnl_base_total:+.1f}%")
print(f"  Reduced  OOS total PnL: {oos_pnl_red_total:+.1f}%")
print(f"  Delta: {wf_oos_delta:+.1f}%")
print(f"  Fold wins: Baseline {base_fold_wins}/{n_folds}, Reduced {red_fold_wins}/{n_folds}")

wf_results['summary'] = {
    'baseline_oos_pnl': round(oos_pnl_base_total, 2),
    'reduced_oos_pnl': round(oos_pnl_red_total, 2),
    'delta': round(wf_oos_delta, 2),
    'baseline_fold_wins': base_fold_wins,
    'reduced_fold_wins': red_fold_wins,
    'verdict': 'REMOVAL_BENEFICIAL' if wf_oos_delta > 0 and red_fold_wins >= 3 else 'REMOVAL_HARMFUL' if wf_oos_delta < -20 else 'INCONCLUSIVE',
}
print(f"  VERDICT: {wf_results['summary']['verdict']}")


# ============================================================
# Phase 3: Individual LOO Removal WF
# ============================================================
print("\n" + "=" * 80)
print("PHASE 3: Individual Leave-One-Out Removal (WF OOS)")
print("  Remove each watchlist pattern one at a time, check WF OOS impact")
print("=" * 80)

loo_results = {}

for wp in WATCHLIST:
    loo_map = {p: v for p, v in all_patterns.items() if p != wp}
    oos_pnl_total = 0
    fold_details = []

    for fold_i in range(n_folds):
        train_end_bar = (fold_i + 1) * fold_days * bars_per_day
        oos_start_bar = train_end_bar
        oos_end_bar = min((fold_i + 2) * fold_days * bars_per_day, n_bars)
        if oos_end_bar <= oos_start_bar:
            continue

        oos_trades = collect_trades_1pos(loo_map, oos_start_bar, oos_end_bar)
        oos_eval = eval_portfolio(oos_trades)
        oos_pnl_total += oos_eval['pnl']
        fold_details.append(oos_eval['pnl'])

    delta = oos_pnl_total - oos_pnl_base_total
    direction = all_patterns[wp][0]
    tp, sl = all_patterns[wp][1], all_patterns[wp][2]

    loo_results[wp] = {
        'direction': direction,
        'tp_sl': f'{tp}/{sl}',
        'oos_pnl_without': round(oos_pnl_total, 2),
        'oos_pnl_baseline': round(oos_pnl_base_total, 2),
        'delta': round(delta, 2),
        'fold_pnls': [round(x, 2) for x in fold_details],
        'beneficial': delta > 0,
    }
    tag = "✅ REMOVAL HELPS" if delta > 0 else "❌ REMOVAL HURTS"
    print(f"  Remove {wp:12s} ({direction:5s} {tp}/{sl}): OOS PnL {oos_pnl_total:+7.1f}% (delta {delta:+6.1f}%) {tag}")

# Sort by benefit
loo_sorted = sorted(loo_results.items(), key=lambda x: x[1]['delta'], reverse=True)
print(f"\n  === LOO RANKING (best removal first) ===")
for wp, d in loo_sorted:
    tag = "✅" if d['beneficial'] else "❌"
    print(f"    {tag} {wp:12s}: delta {d['delta']:+6.1f}%")

beneficial_count = sum(1 for _, d in loo_results.items() if d['beneficial'])
print(f"\n  Beneficial removals: {beneficial_count}/{len(WATCHLIST)}")


# ============================================================
# Phase 4: Cumulative Removal (worst-first)
# ============================================================
print("\n" + "=" * 80)
print("PHASE 4: Cumulative Removal (worst issue count first)")
print("  Progressive removal: add one more pattern to remove each step")
print("=" * 80)

cumulative_results = []
removed_so_far = set()

for step, wp in enumerate(WATCHLIST):
    removed_so_far.add(wp)
    cum_map = {p: v for p, v in all_patterns.items() if p not in removed_so_far}

    oos_pnl_total = 0
    for fold_i in range(n_folds):
        train_end_bar = (fold_i + 1) * fold_days * bars_per_day
        oos_start_bar = train_end_bar
        oos_end_bar = min((fold_i + 2) * fold_days * bars_per_day, n_bars)
        if oos_end_bar <= oos_start_bar:
            continue
        oos_trades = collect_trades_1pos(cum_map, oos_start_bar, oos_end_bar)
        oos_eval = eval_portfolio(oos_trades)
        oos_pnl_total += oos_eval['pnl']

    # Also in-sample for reference
    is_trades = collect_trades_1pos(cum_map)
    is_eval = eval_portfolio(is_trades)

    delta = oos_pnl_total - oos_pnl_base_total
    cumulative_results.append({
        'step': step + 1,
        'removed': wp,
        'total_removed': list(removed_so_far),
        'remaining_patterns': len(cum_map),
        'oos_pnl': round(oos_pnl_total, 2),
        'oos_delta': round(delta, 2),
        'is_pnl': round(is_eval['pnl'], 2),
        'is_mdd': round(is_eval['mdd'], 2),
        'is_pnl_mdd': round(is_eval['pnl_mdd'], 2),
    })

    print(f"  Step {step+1}: Remove {wp:12s} → {len(cum_map)} patterns | "
          f"OOS PnL {oos_pnl_total:+7.1f}% (delta {delta:+6.1f}%) | "
          f"IS PnL {is_eval['pnl']:+7.1f}% MDD {is_eval['mdd']:5.1f}% PnL/MDD {is_eval['pnl_mdd']:5.1f}x")


# ============================================================
# Phase 5: Bootstrap Confidence Interval
# ============================================================
print("\n" + "=" * 80)
print("PHASE 5: Bootstrap Confidence Interval")
print(f"  {N_BOOTSTRAP} bootstrap samples of baseline vs reduced")
print("=" * 80)

base_pnls = np.array([t[2] for t in trades_baseline])
red_pnls = np.array([t[2] for t in trades_reduced])

np.random.seed(42)
base_boots = np.array([np.sum(np.random.choice(base_pnls, size=len(base_pnls), replace=True))
                        for _ in range(N_BOOTSTRAP)])
red_boots = np.array([np.sum(np.random.choice(red_pnls, size=len(red_pnls), replace=True))
                       for _ in range(N_BOOTSTRAP)])

# Probability that reduced > baseline
p_red_better = float(np.mean(red_boots > base_boots))

bootstrap_results = {
    'baseline_ci_95': [round(float(np.percentile(base_boots, 2.5)), 1),
                       round(float(np.percentile(base_boots, 97.5)), 1)],
    'reduced_ci_95': [round(float(np.percentile(red_boots, 2.5)), 1),
                      round(float(np.percentile(red_boots, 97.5)), 1)],
    'baseline_mean': round(float(np.mean(base_boots)), 1),
    'reduced_mean': round(float(np.mean(red_boots)), 1),
    'p_reduced_better': round(p_red_better, 4),
    'p_baseline_better': round(1 - p_red_better, 4),
}

print(f"  Baseline 95% CI: [{bootstrap_results['baseline_ci_95'][0]}, {bootstrap_results['baseline_ci_95'][1]}]")
print(f"  Reduced  95% CI: [{bootstrap_results['reduced_ci_95'][0]}, {bootstrap_results['reduced_ci_95'][1]}]")
print(f"  P(Reduced > Baseline): {p_red_better:.4f}")
print(f"  P(Baseline > Reduced): {1 - p_red_better:.4f}")


# ============================================================
# Phase 6: Portfolio MC Test
# ============================================================
print("\n" + "=" * 80)
print("PHASE 6: Portfolio Monte Carlo Test")
print("=" * 80)

mc_base = mc_portfolio(trades_baseline)
mc_red = mc_portfolio(trades_reduced)

print(f"  Baseline MC: p={mc_base['p_value']:.6f}, MDD P99={mc_base['mdd_p99']:.1f}%")
print(f"  Reduced  MC: p={mc_red['p_value']:.6f}, MDD P99={mc_red['mdd_p99']:.1f}%")


# ============================================================
# Final Verdict
# ============================================================
print("\n" + "=" * 80)
print("FINAL VERDICT")
print("=" * 80)

wf_better = wf_results['summary']['reduced_oos_pnl'] > wf_results['summary']['baseline_oos_pnl']
wf_majority = red_fold_wins > base_fold_wins
boot_better = p_red_better > 0.5
loo_majority = beneficial_count > len(WATCHLIST) // 2

criteria = [wf_better, wf_majority, boot_better, loo_majority]
pass_count = sum(criteria)

if pass_count >= 3:
    verdict = "REMOVAL_RECOMMENDED"
elif pass_count == 2:
    verdict = "MARGINAL — needs more evidence"
else:
    verdict = "KEEP_ALL — removal not beneficial"

print(f"  WF OOS better: {'YES' if wf_better else 'NO'} ({wf_results['summary']['delta']:+.1f}%)")
print(f"  WF fold majority: {'YES' if wf_majority else 'NO'} ({red_fold_wins}/{n_folds})")
print(f"  Bootstrap P(reduced>base): {'YES' if boot_better else 'NO'} ({p_red_better:.4f})")
print(f"  LOO majority beneficial: {'YES' if loo_majority else 'NO'} ({beneficial_count}/{len(WATCHLIST)})")
print(f"  Criteria passed: {pass_count}/4")
print(f"\n  >>> VERDICT: {verdict} <<<")


# ============================================================
# Save results
# ============================================================
results = {
    'meta': {
        'script': 'watchlist_removal_wf_research.py',
        'version': 'v1.27.2',
        'timestamp': datetime.now().isoformat(),
        'watchlist': WATCHLIST,
        'baseline_patterns': len(baseline_patterns),
        'reduced_patterns': len(reduced_patterns),
    },
    'in_sample': is_results,
    'walk_forward': wf_results,
    'loo_removal': loo_results,
    'cumulative_removal': cumulative_results,
    'bootstrap': bootstrap_results,
    'mc_test': {
        'baseline': mc_base,
        'reduced': mc_red,
    },
    'verdict': {
        'wf_oos_better': wf_better,
        'wf_fold_majority': wf_majority,
        'bootstrap_better': boot_better,
        'loo_majority': loo_majority,
        'criteria_passed': pass_count,
        'final': verdict,
    },
}

out_path = Path(__file__).resolve().parent / '../../results/watchlist_removal_wf_research.json'
with open(out_path, 'w') as f:
    json.dump(results, f, indent=2)
print(f"\nResults saved: {out_path}")
print("Done.")
