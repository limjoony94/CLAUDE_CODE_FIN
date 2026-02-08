"""
Portfolio Pruning Research v4
==============================
v1.26.0 (78 patterns) 가지치기 연구

목표: 78패턴 → 최적 부분집합 선별
방법: 1-position-at-a-time, simple returns, multi-tier progressive pruning

가지치기 기준:
  Tier 0 (Baseline): 현재 78패턴 전체
  Tier 1 (Moderate): trades >= 20, MC < 0.005, WF >= 4, periods >= 2
  Tier 2 (Strict):   trades >= 25, MC < 0.005, WF >= 5, periods >= 3
  Tier 3 (Elite):    trades >= 30, MC < 0.003, WF = 5, periods = 3, exp > 0
  Tier 4 (Marginal): 개별 패턴 제거 시 포트폴리오 악화 → 유지, 개선 → 제거
"""

import pandas as pd
import numpy as np
from pathlib import Path
from collections import defaultdict
import json
from datetime import datetime
import time
import sys

sys.path.insert(0, str(Path(__file__).parent.parent / 'production'))
from pattern_5m.indicators import classify_candle
from pattern_5m.constants import (
    VALIDATED_LONG_PATTERNS, VALIDATED_SHORT_PATTERNS,
    PATTERN_OPTIMAL_TPSL, PATTERN_STATS, BOT_VERSION,
)

# ============================================================
# Parameters
# ============================================================
MAX_BARS = 500
FEE_PCT = 0.10          # 0.05% * 2 (round trip)
SLIPPAGE = 0.02
LEVERAGE = 3
MC_SIMS = 10000

print("=" * 70)
print(f"Portfolio Pruning Research v4 (from v{BOT_VERSION})")
print(f"  Input: {len(VALIDATED_LONG_PATTERNS)}L + {len(VALIDATED_SHORT_PATTERNS)}S = "
      f"{len(VALIDATED_LONG_PATTERNS) + len(VALIDATED_SHORT_PATTERNS)} patterns")
print("=" * 70)

# ============================================================
# Build current portfolio map
# ============================================================
PORTFOLIO_78 = {}
for pat in VALIDATED_LONG_PATTERNS:
    tp, sl = PATTERN_OPTIMAL_TPSL[pat]
    PORTFOLIO_78[pat] = ("LONG", tp, sl)
for pat in VALIDATED_SHORT_PATTERNS:
    tp, sl = PATTERN_OPTIMAL_TPSL[pat]
    PORTFOLIO_78[pat] = ("SHORT", tp, sl)

print(f"Portfolio map: {len(PORTFOLIO_78)} patterns")

# ============================================================
# Load and classify data
# ============================================================
data_path = Path(__file__).resolve().parent / '../../data/btc_5m_270days_reclassified.csv'
df = pd.read_csv(data_path)
print(f"Loaded: {len(df)} bars")

highs = df['high'].values
lows = df['low'].values
opens = df['open'].values
closes = df['close'].values
n_bars = len(df)

df['timestamp'] = pd.to_datetime(df['timestamp'])
dates = df['timestamp'].values

period_masks = {
    'H1_May_Jul': (dates >= np.datetime64('2025-05-01')) & (dates < np.datetime64('2025-08-01')),
    'H2_Aug_Oct': (dates >= np.datetime64('2025-08-01')) & (dates < np.datetime64('2025-11-01')),
    'H3_Nov_Jan': (dates >= np.datetime64('2025-11-01')) & (dates < np.datetime64('2026-02-01')),
}

print("Classifying candles...")
body_abs_arr = np.abs(closes - opens)
avg_body_20_arr = pd.Series(body_abs_arr).rolling(20).mean().values

types = []
for i in range(n_bars):
    row = pd.Series({'open': opens[i], 'high': highs[i], 'low': lows[i], 'close': closes[i]})
    avg_b = avg_body_20_arr[i] if not pd.isna(avg_body_20_arr[i]) else 1.0
    types.append(classify_candle(row, avg_b).value)

pattern_indices = defaultdict(list)
for i in range(2, n_bars):
    pat = f"{types[i-2]}-{types[i-1]}-{types[i]}"
    pattern_indices[pat].append(i)
for k in pattern_indices:
    pattern_indices[k] = np.array(pattern_indices[k])

print("Classification done.")


# ============================================================
# Backtest engine
# ============================================================
def bt_fixed(indices, direction, tp_pct, sl_pct):
    pnls = []
    for idx in indices:
        if idx + 1 >= n_bars:
            continue
        entry = opens[idx + 1]
        if entry <= 0:
            continue
        if direction == 'LONG':
            tpp = entry * (1 + tp_pct / 100)
            slp = entry * (1 - sl_pct / 100)
        else:
            tpp = entry * (1 - tp_pct / 100)
            slp = entry * (1 + sl_pct / 100)
        for j in range(idx + 2, min(idx + 2 + MAX_BARS, n_bars)):
            ht = (highs[j] >= tpp) if direction == 'LONG' else (lows[j] <= tpp)
            hs = (lows[j] <= slp) if direction == 'LONG' else (highs[j] >= slp)
            if ht and hs:
                pnls.append((tp_pct if abs(tpp - entry) <= abs(slp - entry) else -sl_pct) * LEVERAGE - FEE_PCT)
                break
            elif ht:
                pnls.append(tp_pct * LEVERAGE - FEE_PCT)
                break
            elif hs:
                pnls.append(-sl_pct * LEVERAGE - FEE_PCT)
                break
    return pnls


def mc_test(pnl_list, n_sims=MC_SIMS):
    if len(pnl_list) < 5:
        return 1.0
    pa = np.array(pnl_list)
    real = np.sum(pa)
    signs = np.random.choice([-1, 1], size=(n_sims, len(pa)))
    return float(np.mean(np.sum(pa * signs, axis=1) >= real))


def walk_forward_test(indices, direction, tp_pct, sl_pct, n_folds=5):
    si = np.sort(indices)
    fs = len(si) // n_folds
    if fs < 3:
        return 0
    profitable = 0
    for f in range(n_folds):
        s = f * fs
        e = s + fs if f < n_folds - 1 else len(si)
        pnls = bt_fixed(si[s:e], direction, tp_pct, sl_pct)
        if pnls and sum(pnls) > 0:
            profitable += 1
    return profitable


def period_test(indices, direction, tp_pct, sl_pct):
    profitable = 0
    for mask in period_masks.values():
        pidx = indices[mask[indices]]
        if len(pidx) < 3:
            continue
        pnls = bt_fixed(pidx, direction, tp_pct, sl_pct)
        if pnls and sum(pnls) > 0:
            profitable += 1
    return profitable


def collect_trades_1pos(pattern_map):
    raw_trades = []
    for i in range(2, n_bars):
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
                raw_trades.append((entry_bar, j, pnl))
                break
            elif ht:
                raw_trades.append((entry_bar, j, tp * LEVERAGE - FEE_PCT))
                break
            elif hs:
                raw_trades.append((entry_bar, j, -sl * LEVERAGE - FEE_PCT))
                break
    raw_trades.sort(key=lambda x: x[0])
    filtered = []
    last_exit = -1
    for entry_bar, exit_bar, pnl in raw_trades:
        if entry_bar > last_exit:
            filtered.append((entry_bar, exit_bar, pnl))
            last_exit = exit_bar
    return filtered


def eval_trades_simple(trades):
    if not trades:
        return {'pnl': 0, 'trades': 0, 'wr': 0, 'mdd': 0, 'pf': 0,
                'avg_win': 0, 'avg_loss': 0, 'expectancy': 0, 'pnl_mdd': 0}
    pnl_list = [t[2] for t in trades]
    cum_pnl = 0
    peak_pnl = 0
    max_dd = 0
    wins = 0
    win_pnls = []
    loss_pnls = []
    for p in pnl_list:
        cum_pnl += p
        if cum_pnl > peak_pnl:
            peak_pnl = cum_pnl
        dd = peak_pnl - cum_pnl
        if dd > max_dd:
            max_dd = dd
        if p > 0:
            wins += 1
            win_pnls.append(p)
        else:
            loss_pnls.append(abs(p))
    avg_win = np.mean(win_pnls) if win_pnls else 0
    avg_loss = np.mean(loss_pnls) if loss_pnls else 0
    total_win = sum(win_pnls)
    total_loss = sum(loss_pnls)
    wr_pct = wins / len(pnl_list) * 100
    expectancy = (wr_pct / 100 * avg_win) - ((1 - wr_pct / 100) * avg_loss)
    return {
        'pnl': cum_pnl,
        'trades': len(pnl_list),
        'wr': wr_pct,
        'mdd': max_dd,
        'pf': total_win / total_loss if total_loss > 0 else 999,
        'avg_win': avg_win,
        'avg_loss': avg_loss,
        'expectancy': expectancy,
        'pnl_mdd': cum_pnl / max_dd if max_dd > 0 else 999,
    }


def eval_portfolio_wf(pmap, n_folds=5):
    fold_size = n_bars // n_folds
    wf_count = 0
    wf_details = []
    for f in range(n_folds):
        s = f * fold_size
        e = s + fold_size if f < n_folds - 1 else n_bars
        ft = []
        last_exit = -1
        for i in range(max(s, 2), e):
            pat = f"{types[i-2]}-{types[i-1]}-{types[i]}"
            if pat not in pmap:
                continue
            direction, tp, sl = pmap[pat]
            if i + 1 >= n_bars:
                continue
            entry_bar = i + 1
            if entry_bar <= last_exit:
                continue
            entry = opens[entry_bar]
            if entry <= 0:
                continue
            if direction == 'LONG':
                tpp = entry * (1 + tp / 100)
                slp = entry * (1 - sl / 100)
            else:
                tpp = entry * (1 - tp / 100)
                slp = entry * (1 + sl / 100)
            for j in range(entry_bar + 1, min(entry_bar + MAX_BARS, n_bars)):
                ht = (highs[j] >= tpp) if direction == 'LONG' else (lows[j] <= tpp)
                hs = (lows[j] <= slp) if direction == 'LONG' else (highs[j] >= slp)
                if ht and hs:
                    pnl = (tp if abs(tpp - entry) <= abs(slp - entry) else -sl) * LEVERAGE - FEE_PCT
                    ft.append((entry_bar, j, pnl))
                    last_exit = j
                    break
                elif ht:
                    ft.append((entry_bar, j, tp * LEVERAGE - FEE_PCT))
                    last_exit = j
                    break
                elif hs:
                    ft.append((entry_bar, j, -sl * LEVERAGE - FEE_PCT))
                    last_exit = j
                    break
        fr = eval_trades_simple(ft)
        wf_details.append(fr)
        if fr['pnl'] > 0:
            wf_count += 1
    return wf_count, wf_details


# ============================================================
# Phase 1: Per-pattern fresh backtest with strict validation
# ============================================================
print(f"\n{'='*70}")
print("Phase 1: Per-Pattern Fresh Backtest (strict re-validation)")
print(f"{'='*70}")

t0 = time.time()
pattern_data = {}  # pattern -> detailed backtest results

for pat, (direction, tp, sl) in PORTFOLIO_78.items():
    indices = pattern_indices.get(pat, np.array([]))
    if len(indices) == 0:
        print(f"  WARNING: {pat} not found in data!")
        continue

    pnls = bt_fixed(indices, direction, tp, sl)
    n_trades = len(pnls)
    if n_trades == 0:
        continue

    wr = sum(1 for x in pnls if x > 0) / n_trades * 100
    total_pnl = sum(pnls)
    mc = mc_test(pnls, MC_SIMS)
    wf = walk_forward_test(indices, direction, tp, sl)
    pp = period_test(indices, direction, tp, sl)

    avg_win = np.mean([x for x in pnls if x > 0]) if any(x > 0 for x in pnls) else 0
    avg_loss = np.mean([abs(x) for x in pnls if x <= 0]) if any(x <= 0 for x in pnls) else 0
    exp = (wr / 100) * avg_win - (1 - wr / 100) * avg_loss
    rr = tp / sl if sl > 0 else 999

    pattern_data[pat] = {
        'direction': direction,
        'tp': tp, 'sl': sl, 'rr': round(rr, 2),
        'trades': n_trades, 'wr': round(wr, 1),
        'mc': round(mc, 4), 'wf': wf, 'periods': pp,
        'total_pnl': round(total_pnl, 1),
        'avg_win': round(avg_win, 2), 'avg_loss': round(avg_loss, 2),
        'expectancy': round(exp, 3),
    }

elapsed = time.time() - t0
print(f"  Backtest complete in {elapsed:.1f}s for {len(pattern_data)} patterns")

# Print all patterns sorted by expectancy * trades
print(f"\n  {'#':>3} {'Pattern':<12} {'Dir':>5} {'TP/SL':>7} {'R:R':>5} {'Trades':>6} {'WR':>6} {'MC':>8} {'WF':>4} {'Per':>4} {'Exp':>8} {'PnL':>8}")
print("  " + "-" * 95)

sorted_pats = sorted(pattern_data.items(), key=lambda x: -x[1]['expectancy'] * x[1]['trades'])
for i, (pat, d) in enumerate(sorted_pats, 1):
    tpsl = f"{d['tp']}/{d['sl']}"
    flag = ""
    if d['mc'] >= 0.01:
        flag += " [MC!]"
    if d['wf'] < 4:
        flag += " [WF!]"
    if d['expectancy'] <= 0:
        flag += " [EXP-]"
    if d['trades'] < 15:
        flag += " [LOW-N]"
    print(f"  {i:>3} {pat:<12} {d['direction']:>5} {tpsl:>7} {d['rr']:>5.2f} {d['trades']:>6} {d['wr']:>5.1f}% {d['mc']:>8.4f} {d['wf']:>3}/5 {d['periods']:>3}/3 {d['expectancy']:>+7.3f}% {d['total_pnl']:>+7.1f}%{flag}")


# ============================================================
# Phase 2: Progressive Pruning Tiers
# ============================================================
print(f"\n{'='*70}")
print("Phase 2: Progressive Pruning Tiers")
print(f"{'='*70}")

pruning_tiers = {
    'T0_All78': {
        'min_trades': 0, 'max_mc': 1.0, 'min_wf': 0, 'min_periods': 0, 'min_exp': -999,
        'desc': 'All 78 patterns (baseline)',
    },
    'T1_Moderate': {
        'min_trades': 20, 'max_mc': 0.005, 'min_wf': 4, 'min_periods': 2, 'min_exp': -999,
        'desc': 'trades>=20, MC<0.005, WF>=4, Per>=2',
    },
    'T2_Strict': {
        'min_trades': 25, 'max_mc': 0.005, 'min_wf': 5, 'min_periods': 3, 'min_exp': -999,
        'desc': 'trades>=25, MC<0.005, WF=5, Per=3',
    },
    'T3_Elite': {
        'min_trades': 30, 'max_mc': 0.003, 'min_wf': 5, 'min_periods': 3, 'min_exp': 0,
        'desc': 'trades>=30, MC<0.003, WF=5, Per=3, Exp>0',
    },
    'T4_ExpPos': {
        'min_trades': 15, 'max_mc': 0.01, 'min_wf': 4, 'min_periods': 2, 'min_exp': 0,
        'desc': 'Original criteria + Exp>0 (expectancy filter)',
    },
}

tier_portfolios = {}

for tier_name, criteria in pruning_tiers.items():
    kept = {}
    removed = []
    for pat, d in pattern_data.items():
        if (d['trades'] >= criteria['min_trades'] and
            d['mc'] < criteria['max_mc'] and
            d['wf'] >= criteria['min_wf'] and
            d['periods'] >= criteria['min_periods'] and
            d['expectancy'] > criteria['min_exp']):
            kept[pat] = (d['direction'], d['tp'], d['sl'])
        else:
            removed.append(pat)

    tier_portfolios[tier_name] = kept

    n_long = sum(1 for v in kept.values() if v[0] == 'LONG')
    n_short = sum(1 for v in kept.values() if v[0] == 'SHORT')
    print(f"\n  [{tier_name}] {criteria['desc']}")
    print(f"    Kept: {len(kept)} ({n_long}L+{n_short}S), Removed: {len(removed)}")
    if removed and len(removed) <= 30:
        print(f"    Removed: {', '.join(removed)}")


# ============================================================
# Phase 3: Portfolio-Level Comparison (1-pos-at-a-time)
# ============================================================
print(f"\n{'='*70}")
print("Phase 3: Portfolio-Level Comparison (1-pos-at-a-time, simple returns)")
print(f"{'='*70}\n")

print(f"{'Tier':<14} {'Pats':>4} {'Trades':>6} {'Tr/d':>5} {'WR':>6} {'Exp/tr':>8} {'PnL':>10} {'MDD':>7} {'P/M':>6} {'PF':>6} {'AvgW':>6} {'AvgL':>6} {'W/L':>5} {'WF':>4}")
print("-" * 115)

tier_results = {}

for tier_name, pmap in tier_portfolios.items():
    if not pmap:
        print(f"{tier_name:<14} {'(empty)':>4}")
        continue

    trades = collect_trades_1pos(pmap)
    res = eval_trades_simple(trades)
    wf, wf_det = eval_portfolio_wf(pmap)
    tr_per_day = res['trades'] / 270 if res['trades'] > 0 else 0
    wipeout = res['avg_loss'] / res['avg_win'] if res['avg_win'] > 0 else 0

    n_long = sum(1 for v in pmap.values() if v[0] == 'LONG')
    n_short = sum(1 for v in pmap.values() if v[0] == 'SHORT')

    print(f"{tier_name:<14} {len(pmap):>4} {res['trades']:>6} {tr_per_day:>5.1f} {res['wr']:>5.1f}% {res['expectancy']:>+7.3f}% {res['pnl']:>+9.1f}% {res['mdd']:>6.1f}% {res['pnl_mdd']:>5.2f}x {res['pf']:>5.2f} {res['avg_win']:>5.2f}% {res['avg_loss']:>5.2f}% {wipeout:>4.1f}x {wf:>3}/5")

    # WF detail
    wf_str = "  WF: "
    for i, fr in enumerate(wf_det):
        s = "OK" if fr['pnl'] > 0 else "FAIL"
        wf_str += f"[F{i+1}: {fr['trades']}t {fr['pnl']:+.1f}% {s}] "
    print(wf_str)

    tier_results[tier_name] = {
        'patterns': len(pmap),
        'long': n_long, 'short': n_short,
        'result': res,
        'wf': wf,
        'wf_details': wf_det,
        'pattern_list': list(pmap.keys()),
    }


# ============================================================
# Phase 4: WR Resilience Analysis
# ============================================================
print(f"\n{'='*70}")
print("Phase 4: WR Drop Resilience (max tolerable WR drop)")
print(f"{'='*70}\n")

print(f"{'Tier':<14} {'WR':>6} | {'WR-5%':>10} {'WR-10%':>10} {'WR-15%':>10} {'WR-20%':>10} | {'MaxDrop':>8}")
print("-" * 90)

wr_resilience = {}

for tier_name, pmap in tier_portfolios.items():
    if not pmap:
        continue
    trades = collect_trades_1pos(pmap)
    if not trades:
        continue
    pnl_list = [t[2] for t in trades]
    actual_wr = sum(1 for p in pnl_list if p > 0) / len(pnl_list) * 100
    avg_win = np.mean([p for p in pnl_list if p > 0]) if any(p > 0 for p in pnl_list) else 0
    avg_loss = np.mean([abs(p) for p in pnl_list if p <= 0]) if any(p <= 0 for p in pnl_list) else 0

    results_str = f"{actual_wr:>5.1f}% |"
    exps = {}
    for wr_drop in [5, 10, 15, 20]:
        sim_wr = max(0, (actual_wr - wr_drop)) / 100
        exp = sim_wr * avg_win - (1 - sim_wr) * avg_loss
        exps[wr_drop] = exp
        status = "OK" if exp > 0 else "LOSS"
        results_str += f" {exp:>+7.3f}% {status:>4}"

    be_wr = avg_loss / (avg_win + avg_loss) * 100 if (avg_win + avg_loss) > 0 else 0
    max_tolerable = actual_wr - be_wr

    print(f"{tier_name:<14} {results_str} | {max_tolerable:>+6.1f}pp")
    wr_resilience[tier_name] = {
        'actual_wr': round(actual_wr, 1), 'be_wr': round(be_wr, 1),
        'max_drop': round(max_tolerable, 1),
        'sensitivities': {str(d): round(e, 3) for d, e in exps.items()},
    }


# ============================================================
# Phase 5: Marginal Value Analysis (leave-one-out)
# ============================================================
print(f"\n{'='*70}")
print("Phase 5: Marginal Value Analysis (best tier)")
print(f"{'='*70}")

# Find best tier by pnl_mdd
best_tier_name = max(tier_results.keys(),
                     key=lambda t: tier_results[t]['result'].get('pnl_mdd', 0)
                     if tier_results[t]['result']['trades'] > 0 else -999)
print(f"\n  Best tier by PnL/MDD: {best_tier_name}")

best_pmap = tier_portfolios[best_tier_name]
best_trades = collect_trades_1pos(best_pmap)
best_res = eval_trades_simple(best_trades)

print(f"  Baseline: {len(best_pmap)} patterns, {best_res['trades']} trades, "
      f"Exp={best_res['expectancy']:+.3f}%, PnL={best_res['pnl']:+.1f}%, PnL/MDD={best_res['pnl_mdd']:.2f}x")

# Leave-one-out: remove each pattern, check if portfolio improves
print(f"\n  Leave-One-Out Analysis:")
print(f"  {'Pattern':<12} {'Dir':>5} | {'Trades':>6} {'Exp':>8} {'PnL':>10} {'P/M':>6} | {'Impact':>10}")
print("  " + "-" * 80)

harmful_patterns = []
for pat in sorted(best_pmap.keys()):
    pmap_without = {k: v for k, v in best_pmap.items() if k != pat}
    trades_without = collect_trades_1pos(pmap_without)
    res_without = eval_trades_simple(trades_without)

    # Impact = metric WITH pattern - metric WITHOUT pattern
    # Positive impact = pattern helps, Negative = pattern hurts
    exp_impact = best_res['expectancy'] - res_without['expectancy']
    pnl_impact = best_res['pnl'] - res_without['pnl']
    pm_impact = best_res['pnl_mdd'] - res_without['pnl_mdd']

    d = pattern_data[pat]
    impact_label = "HELPFUL" if pm_impact > 0 else "HARMFUL"
    flag = " ***" if pm_impact < -0.05 else ""

    print(f"  {pat:<12} {d['direction']:>5} | {res_without['trades']:>6} {res_without['expectancy']:>+7.3f}% {res_without['pnl']:>+9.1f}% {res_without['pnl_mdd']:>5.2f}x | {impact_label:>8}{flag}")

    if pm_impact < 0:
        harmful_patterns.append((pat, pm_impact, pnl_impact, exp_impact))

# Build optimized portfolio (remove harmful patterns)
if harmful_patterns:
    harmful_patterns.sort(key=lambda x: x[1])
    print(f"\n  Harmful patterns (removing improves PnL/MDD): {len(harmful_patterns)}")
    for pat, pm, pnl, exp in harmful_patterns:
        d = pattern_data[pat]
        print(f"    {pat:<12} ({d['direction']}): PnL/MDD impact={pm:+.2f}x, PnL impact={pnl:+.1f}%, Exp impact={exp:+.3f}%")

    # Progressive removal: remove worst first, re-evaluate
    print(f"\n  Progressive Removal (worst-first):")
    current_pmap = best_pmap.copy()
    current_trades = collect_trades_1pos(current_pmap)
    current_res = eval_trades_simple(current_trades)
    print(f"    Start: {len(current_pmap)} patterns, PnL/MDD={current_res['pnl_mdd']:.2f}x, PnL={current_res['pnl']:+.1f}%")

    removed_in_order = []
    for pat, _, _, _ in harmful_patterns:
        test_pmap = {k: v for k, v in current_pmap.items() if k != pat}
        test_trades = collect_trades_1pos(test_pmap)
        test_res = eval_trades_simple(test_trades)

        if test_res['pnl_mdd'] >= current_res['pnl_mdd']:
            current_pmap = test_pmap
            current_res = test_res
            removed_in_order.append(pat)
            print(f"    Remove {pat}: {len(current_pmap)} pats, PnL/MDD={current_res['pnl_mdd']:.2f}x, PnL={current_res['pnl']:+.1f}%, Exp={current_res['expectancy']:+.3f}%")
        else:
            print(f"    Keep   {pat}: removal would worsen PnL/MDD")

    # Final optimized portfolio
    tier_portfolios['T5_Optimized'] = current_pmap
    opt_trades = collect_trades_1pos(current_pmap)
    opt_res = eval_trades_simple(opt_trades)
    opt_wf, opt_wf_det = eval_portfolio_wf(current_pmap)

    n_long = sum(1 for v in current_pmap.values() if v[0] == 'LONG')
    n_short = sum(1 for v in current_pmap.values() if v[0] == 'SHORT')

    tier_results['T5_Optimized'] = {
        'patterns': len(current_pmap),
        'long': n_long, 'short': n_short,
        'result': opt_res,
        'wf': opt_wf,
        'wf_details': opt_wf_det,
        'pattern_list': list(current_pmap.keys()),
    }

    # WR resilience for optimized
    pnl_list = [t[2] for t in opt_trades]
    if pnl_list:
        actual_wr = sum(1 for p in pnl_list if p > 0) / len(pnl_list) * 100
        avg_win = np.mean([p for p in pnl_list if p > 0]) if any(p > 0 for p in pnl_list) else 0
        avg_loss = np.mean([abs(p) for p in pnl_list if p <= 0]) if any(p <= 0 for p in pnl_list) else 0
        be_wr = avg_loss / (avg_win + avg_loss) * 100 if (avg_win + avg_loss) > 0 else 0
        max_tolerable = actual_wr - be_wr
        wr_resilience['T5_Optimized'] = {
            'actual_wr': round(actual_wr, 1), 'be_wr': round(be_wr, 1),
            'max_drop': round(max_tolerable, 1),
        }
else:
    print(f"\n  No harmful patterns found — all patterns contribute positively.")


# ============================================================
# Phase 6: Final Summary & Recommendation
# ============================================================
print(f"\n{'='*70}")
print("FINAL SUMMARY")
print(f"{'='*70}\n")

print(f"{'Tier':<14} {'Pats':>4} {'L':>3} {'S':>3} {'Trades':>6} {'Exp':>8} {'PnL':>10} {'MDD':>7} {'P/M':>6} {'PF':>6} {'WF':>4} {'MaxDrop':>8}")
print("-" * 100)

for tier_name in ['T0_All78', 'T1_Moderate', 'T2_Strict', 'T3_Elite', 'T4_ExpPos', 'T5_Optimized']:
    tr = tier_results.get(tier_name)
    if not tr:
        continue
    r = tr['result']
    wr = wr_resilience.get(tier_name, {})
    tr_per_day = r['trades'] / 270 if r['trades'] > 0 else 0
    print(f"{tier_name:<14} {tr['patterns']:>4} {tr.get('long',0):>3} {tr.get('short',0):>3} {r['trades']:>6} {r['expectancy']:>+7.3f}% {r['pnl']:>+9.1f}% {r['mdd']:>6.1f}% {r['pnl_mdd']:>5.2f}x {r['pf']:>5.2f} {tr['wf']:>3}/5 {wr.get('max_drop', 0):>+6.1f}pp")

# Recommend best tier
print(f"\n--- Recommendation ---")
best_overall = None
best_score = -999
for tier_name, tr in tier_results.items():
    r = tr['result']
    if r['trades'] < 50:
        continue
    wr = wr_resilience.get(tier_name, {})
    wf = tr.get('wf', 0)
    # Score: pnl_mdd * wf/5 * (1 + max_drop/100)
    md = wr.get('max_drop', 0)
    score = r['pnl_mdd'] * (wf / 5) * (1 + md / 100)
    if score > best_score:
        best_score = score
        best_overall = tier_name

if best_overall:
    tr = tier_results[best_overall]
    r = tr['result']
    wr = wr_resilience.get(best_overall, {})
    print(f"  RECOMMENDED: {best_overall}")
    print(f"    Patterns: {tr['patterns']} ({tr.get('long',0)}L + {tr.get('short',0)}S)")
    print(f"    Trades: {r['trades']} ({r['trades']/270:.1f}/day)")
    print(f"    Expectancy: {r['expectancy']:+.3f}%/trade")
    print(f"    PnL: {r['pnl']:+.1f}%, MDD: {r['mdd']:.1f}%, PnL/MDD: {r['pnl_mdd']:.2f}x")
    print(f"    PF: {r['pf']:.2f}, WF: {tr['wf']}/5")
    print(f"    WR Resilience: max {wr.get('max_drop', 0):+.1f}pp drop tolerable")

    print(f"\n  Pattern List ({best_overall}):")
    longs = sorted([p for p in tr['pattern_list'] if pattern_data[p]['direction'] == 'LONG'])
    shorts = sorted([p for p in tr['pattern_list'] if pattern_data[p]['direction'] == 'SHORT'])
    print(f"    LONG ({len(longs)}): {', '.join(longs)}")
    print(f"    SHORT ({len(shorts)}): {', '.join(shorts)}")


# ============================================================
# Save results
# ============================================================
output = {
    'timestamp': datetime.now().isoformat(),
    'research': 'Portfolio Pruning v4',
    'source_version': BOT_VERSION,
    'source_patterns': len(PORTFOLIO_78),
    'pattern_data': pattern_data,
    'tier_results': {},
    'wr_resilience': wr_resilience,
    'recommendation': best_overall,
}

for tier_name, tr in tier_results.items():
    r = tr['result']
    output['tier_results'][tier_name] = {
        'patterns': tr['patterns'],
        'long': tr.get('long', 0),
        'short': tr.get('short', 0),
        'pattern_list': tr['pattern_list'],
        'trades': r['trades'],
        'wr': round(r['wr'], 1),
        'expectancy': round(r['expectancy'], 3),
        'pnl': round(r['pnl'], 1),
        'mdd': round(r['mdd'], 1),
        'pnl_mdd': round(r['pnl_mdd'], 2),
        'pf': round(r['pf'], 2),
        'avg_win': round(r['avg_win'], 2),
        'avg_loss': round(r['avg_loss'], 2),
        'wf': tr['wf'],
    }

out_path = Path(__file__).resolve().parent / '../../results/portfolio_pruning_v4.json'
with open(out_path, 'w') as f:
    json.dump(output, f, indent=2, default=str)
print(f"\nResults saved to {out_path}")
print("\nDone.")
