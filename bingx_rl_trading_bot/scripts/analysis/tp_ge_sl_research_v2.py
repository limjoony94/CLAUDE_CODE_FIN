"""
TP >= SL Extended Research v2
=============================
v1 결과: TP>=SL + WR>=85% → 8개 패턴, 134거래 (너무 적음)
v2 개선: R:R에 맞춰 WR 기준 동적 조정 → 더 많은 패턴 발굴

핵심 인사이트: R:R=1.0이면 WR>50%만으로 수익.
  현재 WR>=85%는 R:R=0.34용 기준 → TP>=SL에 부적합.

연구 내용:
  1) R:R별 break-even WR 계산
  2) R:R 스펙트럼 (0.5, 0.75, 1.0, 1.5, 2.0) 패턴 발굴
  3) 거래 빈도 vs 안정성 트레이드오프 분석
  4) 최적 포트폴리오 구성 (빈도 + 안정성 균형)
"""

import pandas as pd
import numpy as np
from pathlib import Path
from collections import defaultdict, Counter
import json
from datetime import datetime
import time
import sys

sys.path.insert(0, str(Path(__file__).parent.parent / 'production'))
from pattern_5m.indicators import classify_candle

# ============================================================
# Parameters
# ============================================================
MAX_BARS = 500
FEE_PCT = 0.10
SLIPPAGE = 0.02
LEVERAGE = 3
MIN_TRADES = 15
MC_SIMS = 10000
TP_SL_GRID = [0.3, 0.5, 0.7, 1.0, 1.5, 2.0, 2.5, 3.0]

# R:R tiers to test
RR_TIERS = {
    'RR_0.5': {'min_rr': 0.5, 'max_rr': 0.74},   # Improved but still TP<SL
    'RR_0.75': {'min_rr': 0.75, 'max_rr': 0.99},  # Near-equal
    'RR_1.0': {'min_rr': 1.0, 'max_rr': 1.49},    # Equal or better
    'RR_1.5': {'min_rr': 1.5, 'max_rr': 1.99},    # Favorable
    'RR_2.0+': {'min_rr': 2.0, 'max_rr': 99},     # Very favorable
}

# Current portfolio for comparison
CURRENT_PORTFOLIO = {
    "MD-BU-U": ("LONG", 0.5, 2.0), "MU-MU-U": ("LONG", 0.7, 2.0),
    "MU-U-MU": ("LONG", 0.5, 2.0), "BU-BU-BD": ("LONG", 0.7, 2.0),
    "ST-MU-U": ("LONG", 0.7, 2.0), "IH-DN-DN": ("LONG", 0.7, 2.0),
    "MD-ST-ST": ("SHORT", 0.5, 2.0), "U-MU-BU": ("SHORT", 0.5, 2.0),
    "MU-BU-DN": ("SHORT", 0.7, 2.0), "ST-H-U": ("SHORT", 0.5, 2.0),
    "ST-DN-H": ("SHORT", 0.7, 2.0), "MD-MU-U": ("SHORT", 1.0, 2.0),
    "BU-U-ST": ("SHORT", 0.7, 2.0), "H-DN-ST": ("SHORT", 0.7, 2.0),
    "DN-BD-BU": ("SHORT", 0.7, 2.0), "DN-BU-U": ("SHORT", 1.0, 2.0),
    "DN-U-U": ("SHORT", 0.5, 2.0), "ST-DN-U": ("SHORT", 0.7, 2.0),
    "ST-DN-DN": ("SHORT", 0.7, 2.0), "U-ST-DN": ("SHORT", 0.7, 2.0),
    "U-U-ST": ("SHORT", 0.7, 2.0), "ST-U-DN": ("SHORT", 0.7, 2.0),
    "ST-ST-U": ("SHORT", 0.7, 2.0),
}

print("=" * 70)
print("TP >= SL Extended Research v2")
print("=" * 70)

# ============================================================
# Break-even WR calculation
# ============================================================
print("\n--- Break-Even WR by R:R ---")
print(f"{'R:R':>6} {'TP%':>6} {'SL%':>6} {'Win PnL':>10} {'Loss PnL':>10} {'BE WR':>8} {'Safe WR':>8}")
print("-" * 65)

be_table = []
for tp in TP_SL_GRID:
    for sl in TP_SL_GRID:
        rr = tp / sl
        win_pnl = tp * LEVERAGE - FEE_PCT - SLIPPAGE
        loss_pnl = sl * LEVERAGE + FEE_PCT + SLIPPAGE
        # break-even: WR * win = (1-WR) * loss → WR = loss / (win + loss)
        be_wr = loss_pnl / (win_pnl + loss_pnl) * 100
        safe_wr = be_wr + 10  # 10% safety margin
        be_table.append((rr, tp, sl, win_pnl, loss_pnl, be_wr, safe_wr))

# Print unique R:R values
seen_rr = set()
for rr, tp, sl, wp, lp, be, safe in sorted(be_table, key=lambda x: x[0]):
    rr_key = f"{rr:.2f}"
    if rr_key not in seen_rr and rr >= 0.5:
        seen_rr.add(rr_key)
        print(f"{rr:>6.2f} {tp:>5.1f}% {sl:>5.1f}% {wp:>+9.2f}% {lp:>9.2f}% {be:>7.1f}% {safe:>7.1f}%")

# ============================================================
# Load and classify
# ============================================================
data_path = Path(__file__).resolve().parent / '../../data/btc_5m_270days_reclassified.csv'
df = pd.read_csv(data_path)
print(f"\nLoaded: {len(df)} bars")

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

eligible = sum(1 for v in pattern_indices.values() if len(v) >= MIN_TRADES)
print(f"Patterns with >= {MIN_TRADES} trades: {eligible}")

# ============================================================
# Backtest engine (same as v1)
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


def collect_trades(pattern_map, bar_range=None):
    trades = []
    start = bar_range[0] if bar_range else 2
    end = bar_range[1] if bar_range else n_bars
    for i in range(max(start, 2), end):
        pat = f"{types[i-2]}-{types[i-1]}-{types[i]}"
        if pat not in pattern_map:
            continue
        direction, tp, sl = pattern_map[pat]
        if i + 1 >= n_bars:
            continue
        entry = opens[i + 1]
        if entry <= 0:
            continue
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
                trades.append((i + 1, (tp if abs(tpp - entry) <= abs(slp - entry) else -sl) * LEVERAGE - FEE_PCT))
                break
            elif ht:
                trades.append((i + 1, tp * LEVERAGE - FEE_PCT))
                break
            elif hs:
                trades.append((i + 1, -sl * LEVERAGE - FEE_PCT))
                break
    trades.sort(key=lambda x: x[0])
    return trades


def eval_trades(trades):
    if not trades:
        return {'pnl': 0, 'trades': 0, 'wr': 0, 'mdd': 0, 'pf': 0,
                'avg_win': 0, 'avg_loss': 0, 'expectancy': 0, 'pnl_mdd': 0}
    pnl_list = [t[1] for t in trades]
    equity = 100.0
    peak = 100.0
    max_dd = 0
    wins = 0
    win_pnls = []
    loss_pnls = []
    for p in pnl_list:
        equity *= (1 + p / 100)
        if equity > peak:
            peak = equity
        dd = (peak - equity) / peak * 100
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
        'pnl': (equity / 100 - 1) * 100,
        'trades': len(pnl_list),
        'wr': wr_pct,
        'mdd': max_dd,
        'pf': total_win / total_loss if total_loss > 0 else 999,
        'avg_win': avg_win,
        'avg_loss': avg_loss,
        'expectancy': expectancy,
        'pnl_mdd': (equity / 100 - 1) * 100 / max_dd if max_dd > 0 else 999,
    }


# ============================================================
# Baselines
# ============================================================
print("Computing baselines...")
sample_idx = np.arange(2, n_bars, 20)

baselines = {}
for tp in TP_SL_GRID:
    for sl in TP_SL_GRID:
        wins_l, wins_s, total_l, total_s = 0, 0, 0, 0
        for idx in sample_idx:
            if idx + 1 >= n_bars:
                continue
            entry = opens[idx + 1]
            if entry <= 0:
                continue
            for d in ['LONG', 'SHORT']:
                if d == 'LONG':
                    tpp = entry * (1 + tp / 100)
                    slp = entry * (1 - sl / 100)
                else:
                    tpp = entry * (1 - tp / 100)
                    slp = entry * (1 + sl / 100)
                for j in range(idx + 2, min(idx + 2 + MAX_BARS, n_bars)):
                    ht = (highs[j] >= tpp) if d == 'LONG' else (lows[j] <= tpp)
                    hs = (lows[j] <= slp) if d == 'LONG' else (highs[j] >= slp)
                    if ht and hs:
                        if abs(tpp - entry) <= abs(slp - entry):
                            if d == 'LONG': wins_l += 1
                            else: wins_s += 1
                        if d == 'LONG': total_l += 1
                        else: total_s += 1
                        break
                    elif ht:
                        if d == 'LONG': wins_l += 1; total_l += 1
                        else: wins_s += 1; total_s += 1
                        break
                    elif hs:
                        if d == 'LONG': total_l += 1
                        else: total_s += 1
                        break
        baselines[(tp, sl)] = {
            'LONG': wins_l / total_l * 100 if total_l else 50,
            'SHORT': wins_s / total_s * 100 if total_s else 50,
        }
print("  Done.")


# ============================================================
# Phase 1: R:R Spectrum Scan
# ============================================================
print(f"\n{'='*70}")
print("Phase 1: R:R Spectrum Pattern Discovery")
print("  WR threshold = dynamic (break-even + 10% margin)")
print(f"{'='*70}")

all_rr_results = {}  # tier_name -> list of valid patterns

t0 = time.time()

# Build all TP/SL combos with their R:R
all_combos = []
for tp in TP_SL_GRID:
    for sl in TP_SL_GRID:
        rr = tp / sl
        win_pnl = tp * LEVERAGE - FEE_PCT - SLIPPAGE
        loss_pnl = sl * LEVERAGE + FEE_PCT + SLIPPAGE
        be_wr = loss_pnl / (win_pnl + loss_pnl) * 100
        safe_wr = be_wr + 10  # 10% margin above break-even
        all_combos.append((tp, sl, rr, safe_wr))

# Scan for minimum R:R >= 0.5 (improvement over current 0.34)
# Use dynamic WR threshold based on R:R
valid_by_rr = {}  # key -> (pat, direction, tp, sl, rr, trades, wr, excess, mc, wf, pp, total_pnl)

for pat, indices in pattern_indices.items():
    if len(indices) < MIN_TRADES:
        continue
    for direction in ['LONG', 'SHORT']:
        for tp, sl, rr, min_wr in all_combos:
            if rr < 0.5:
                continue  # Skip very unfavorable R:R

            pnls = bt_fixed(indices, direction, tp, sl)
            if len(pnls) < MIN_TRADES:
                continue
            wr = sum(1 for x in pnls if x > 0) / len(pnls) * 100

            if wr < min_wr:
                continue

            base = baselines.get((tp, sl), {}).get(direction, 50)
            excess = wr - base
            if excess < 5:
                continue
            total_pnl = sum(pnls)
            if total_pnl <= 0:
                continue

            mc = mc_test(pnls, 3000)
            if mc >= 0.05:
                continue

            wf = walk_forward_test(indices, direction, tp, sl)
            if wf < 3:
                continue

            pp = period_test(indices, direction, tp, sl)
            if pp < 2:
                continue

            mc_full = mc_test(pnls, MC_SIMS)
            if mc_full >= 0.05:
                continue

            key = f"{pat}_{direction}_{tp}_{sl}"
            valid_by_rr[key] = (pat, direction, tp, sl, rr, len(pnls), wr, excess, mc_full, wf, pp, total_pnl)

elapsed = time.time() - t0
print(f"  Scan complete in {elapsed:.1f}s")
print(f"  Total valid entries (all R:R >= 0.5): {len(valid_by_rr)}")

# ============================================================
# Phase 2: Group by R:R tier and find best per pattern
# ============================================================
print(f"\n{'='*70}")
print("Phase 2: Best Pattern per R:R Tier")
print(f"{'='*70}")

tier_portfolios = {}

for tier_name, tier_cfg in RR_TIERS.items():
    min_rr = tier_cfg['min_rr']
    max_rr = tier_cfg['max_rr']

    # Filter to this tier
    tier_entries = {k: v for k, v in valid_by_rr.items()
                    if min_rr <= v[4] <= max_rr}

    # Best per pattern-direction (by total_pnl * rr)
    best_per_pat = {}
    for k, v in tier_entries.items():
        pat, direction = v[0], v[1]
        pat_key = f"{pat}_{direction}"
        score = v[11] * v[4] * (v[9] / 5)  # pnl * rr * wf_ratio
        if pat_key not in best_per_pat or score > best_per_pat[pat_key][1]:
            best_per_pat[pat_key] = (v, score)

    # Remove bidirectional
    pat_names = [v[0][0] for v in best_per_pat.values()]
    bidir = {p for p, c in Counter(pat_names).items() if c > 1}
    if bidir:
        to_remove = []
        for p in bidir:
            keys = [k for k, v in best_per_pat.items() if v[0][0] == p]
            keys.sort(key=lambda k: best_per_pat[k][1], reverse=True)
            to_remove.extend(keys[1:])
        for k in to_remove:
            del best_per_pat[k]

    # Sort by total_pnl
    sorted_tier = sorted(best_per_pat.values(), key=lambda x: -x[0][11])

    print(f"\n--- {tier_name} (R:R {min_rr:.1f}~{max_rr:.1f}) ---")
    print(f"  Valid patterns: {len(sorted_tier)}")

    if sorted_tier:
        print(f"  {'Pattern':<12} {'Dir':>5} {'TP':>4} {'SL':>4} {'R:R':>5} {'Trades':>6} {'WR':>6} {'MC':>8} {'WF':>4} {'Per':>4} {'PnL':>8}")
        print("  " + "-" * 80)
        for v, score in sorted_tier[:20]:  # Show top 20
            pat, d, tp, sl, rr, nt, wr, exc, mc, wf, pp, tpnl = v
            print(f"  {pat:<12} {d:>5} {tp:>4.1f} {sl:>4.1f} {rr:>5.2f} {nt:>6} {wr:>5.1f}% {mc:>8.4f} {wf:>3}/5 {pp:>3}/3 {tpnl:>+7.1f}%")

    # Build portfolio map
    pmap = {}
    for v, score in sorted_tier:
        pat, d, tp, sl = v[0], v[1], v[2], v[3]
        pmap[pat] = (d, tp, sl)
    tier_portfolios[tier_name] = pmap

# ============================================================
# Phase 3: Portfolio Comparison
# ============================================================
print(f"\n{'='*70}")
print("Phase 3: Portfolio-Level Comparison")
print(f"{'='*70}\n")

all_portfolios = {'Current (R:R=0.34)': CURRENT_PORTFOLIO}
all_portfolios.update(tier_portfolios)

summary_rows = []

print(f"{'Portfolio':<22} {'Pats':>4} {'Trades':>6} {'Tr/day':>6} {'WR':>6} {'Exp/tr':>8} {'PnL':>12} {'MDD':>6} {'PF':>6} {'WF':>4}")
print("-" * 95)

for label, pmap in all_portfolios.items():
    if not pmap:
        continue
    trades = collect_trades(pmap)
    res = eval_trades(trades)
    tr_per_day = res['trades'] / 270 if res['trades'] > 0 else 0

    # Walk-forward
    fold_size = n_bars // 5
    wf_count = 0
    for f in range(5):
        s = f * fold_size
        e = s + fold_size if f < 4 else n_bars
        ft = collect_trades(pmap, bar_range=(s, e))
        fr = eval_trades(ft)
        if fr['pnl'] > 0:
            wf_count += 1

    pnl_str = f"{res['pnl']:+.1f}%" if abs(res['pnl']) < 100000 else f"{res['pnl']:+.0f}%"
    print(f"{label:<22} {len(pmap):>4} {res['trades']:>6} {tr_per_day:>6.1f} {res['wr']:>5.1f}% {res['expectancy']:>+7.3f}% {pnl_str:>12} {res['mdd']:>5.1f}% {res['pf']:>5.2f} {wf_count:>3}/5")

    summary_rows.append({
        'portfolio': label,
        'patterns': len(pmap),
        'trades': res['trades'],
        'trades_per_day': round(tr_per_day, 1),
        'wr': round(res['wr'], 1),
        'expectancy': round(res['expectancy'], 3),
        'pnl': round(res['pnl'], 1),
        'mdd': round(res['mdd'], 1),
        'pf': round(res['pf'], 2),
        'wf': wf_count,
        'avg_win': round(res['avg_win'], 2),
        'avg_loss': round(res['avg_loss'], 2),
    })

# ============================================================
# Phase 4: WR Sensitivity Analysis
# ============================================================
print(f"\n{'='*70}")
print("Phase 4: WR Drop Sensitivity (Expectancy per trade)")
print(f"{'='*70}\n")

print(f"{'Portfolio':<22} {'WR':>6} {'WR-5%':>9} {'WR-10%':>9} {'WR-15%':>9} {'WR-20%':>9}")
print("-" * 65)

for label, pmap in all_portfolios.items():
    if not pmap:
        continue
    trades = collect_trades(pmap)
    if not trades:
        continue
    pnl_list = [t[1] for t in trades]
    actual_wr = sum(1 for p in pnl_list if p > 0) / len(pnl_list) * 100
    avg_win = np.mean([p for p in pnl_list if p > 0]) if any(p > 0 for p in pnl_list) else 0
    avg_loss = np.mean([abs(p) for p in pnl_list if p <= 0]) if any(p <= 0 for p in pnl_list) else 0

    results_row = f"{actual_wr:>5.1f}%"
    for wr_drop in [5, 10, 15, 20]:
        sim_wr = (actual_wr - wr_drop) / 100
        exp = sim_wr * avg_win - (1 - sim_wr) * avg_loss
        results_row += f" {exp:>+8.3f}%"

    print(f"{label:<22} {results_row}")

# ============================================================
# Phase 5: Optimal Combined Portfolio
# ============================================================
print(f"\n{'='*70}")
print("Phase 5: Optimal Combined Portfolio Construction")
print(f"{'='*70}\n")

# Strategy: Take patterns from RR >= 0.75 tiers
# Filter: WF >= 4, MC < 0.01, reasonable trade count
combined_map = {}
combined_details = []

# Collect all valid patterns with R:R >= 0.75
for k, v in valid_by_rr.items():
    pat, direction, tp, sl, rr, nt, wr, excess, mc, wf, pp, total_pnl = v
    if rr < 0.75:
        continue
    if wf < 4:
        continue
    if mc >= 0.01:
        continue
    if pp < 2:
        continue

    pat_key = f"{pat}_{direction}"
    score = total_pnl * rr * (wf / 5)
    combined_details.append((pat, direction, tp, sl, rr, nt, wr, mc, wf, pp, total_pnl, score))

# Best per pattern (avoid duplicates)
best_combined = {}
for entry in combined_details:
    pat, direction = entry[0], entry[1]
    pat_key = f"{pat}_{direction}"
    if pat_key not in best_combined or entry[-1] > best_combined[pat_key][-1]:
        best_combined[pat_key] = entry

# Remove bidirectional
pat_names = [v[0] for v in best_combined.values()]
bidir = {p for p, c in Counter(pat_names).items() if c > 1}
if bidir:
    to_remove = []
    for p in bidir:
        keys = [k for k, v in best_combined.items() if v[0] == p]
        keys.sort(key=lambda k: best_combined[k][-1], reverse=True)
        to_remove.extend(keys[1:])
    for k in to_remove:
        del best_combined[k]

# Sort by score
sorted_combined = sorted(best_combined.values(), key=lambda x: -x[-1])

print(f"Optimal Combined Portfolio (R:R>=0.75, WF>=4, MC<0.01, Period>=2):")
print(f"  Total patterns: {len(sorted_combined)}")
print()

longs = [v for v in sorted_combined if v[1] == 'LONG']
shorts = [v for v in sorted_combined if v[1] == 'SHORT']

print(f"LONG: {len(longs)} patterns")
print(f"  {'Pattern':<12} {'TP':>4} {'SL':>4} {'R:R':>5} {'Trades':>6} {'WR':>6} {'MC':>8} {'WF':>4} {'Per':>4} {'PnL':>8}")
print("  " + "-" * 75)
for v in longs:
    pat, d, tp, sl, rr, nt, wr, mc, wf, pp, tpnl, _ = v
    print(f"  {pat:<12} {tp:>4.1f} {sl:>4.1f} {rr:>5.2f} {nt:>6} {wr:>5.1f}% {mc:>8.4f} {wf:>3}/5 {pp:>3}/3 {tpnl:>+7.1f}%")

print(f"\nSHORT: {len(shorts)} patterns")
print(f"  {'Pattern':<12} {'TP':>4} {'SL':>4} {'R:R':>5} {'Trades':>6} {'WR':>6} {'MC':>8} {'WF':>4} {'Per':>4} {'PnL':>8}")
print("  " + "-" * 75)
for v in shorts:
    pat, d, tp, sl, rr, nt, wr, mc, wf, pp, tpnl, _ = v
    print(f"  {pat:<12} {tp:>4.1f} {sl:>4.1f} {rr:>5.2f} {nt:>6} {wr:>5.1f}% {mc:>8.4f} {wf:>3}/5 {pp:>3}/3 {tpnl:>+7.1f}%")

# Build combined map
for v in sorted_combined:
    pat, d, tp, sl = v[0], v[1], v[2], v[3]
    combined_map[pat] = (d, tp, sl)

# Evaluate combined portfolio
if combined_map:
    print(f"\n--- Combined Portfolio Evaluation ---")
    trades = collect_trades(combined_map)
    res = eval_trades(trades)
    tr_per_day = res['trades'] / 270

    # Walk-forward
    fold_size = n_bars // 5
    wf_count = 0
    wf_details = []
    for f in range(5):
        s = f * fold_size
        e = s + fold_size if f < 4 else n_bars
        ft = collect_trades(combined_map, bar_range=(s, e))
        fr = eval_trades(ft)
        wf_details.append(fr)
        if fr['pnl'] > 0:
            wf_count += 1

    # Period
    period_results = {}
    for pname, mask in period_masks.items():
        bidx = np.where(mask)[0]
        if len(bidx) == 0:
            continue
        pt = collect_trades(combined_map, bar_range=(bidx[0], bidx[-1] + 1))
        period_results[pname] = eval_trades(pt)
    pp_count = sum(1 for r in period_results.values() if r['pnl'] > 0)

    tp_vals = [v[1] for v in combined_map.values()]
    sl_vals = [v[2] for v in combined_map.values()]
    avg_rr = np.mean(tp_vals) / np.mean(sl_vals) if sl_vals else 0

    print(f"  Patterns: {len(combined_map)}, Avg R:R: {avg_rr:.2f}")
    print(f"  Trades: {res['trades']} ({tr_per_day:.1f}/day)")
    print(f"  WR: {res['wr']:.1f}%, Expectancy: {res['expectancy']:+.3f}%/trade")
    print(f"  Avg Win: {res['avg_win']:.2f}%, Avg Loss: {res['avg_loss']:.2f}%")
    if res['avg_win'] > 0:
        print(f"  Wipeout ratio: 1 loss = {res['avg_loss']/res['avg_win']:.1f}x wins")
    print(f"  PnL: {res['pnl']:+.1f}%, MDD: {res['mdd']:.1f}%, PF: {res['pf']:.2f}")
    print(f"  Walk-Forward: {wf_count}/5")
    for i, fr in enumerate(wf_details):
        s = "OK" if fr['pnl'] > 0 else "FAIL"
        print(f"    Fold {i+1}: {fr['trades']:>4}t, WR {fr['wr']:>5.1f}%, PnL {fr['pnl']:>+10.1f}%, MDD {fr['mdd']:>5.1f}% [{s}]")
    print(f"  Period stability: {pp_count}/3")
    for pn, pr in period_results.items():
        s = "OK" if pr['pnl'] > 0 else "FAIL"
        print(f"    {pn}: {pr['trades']:>4}t, WR {pr['wr']:>5.1f}%, PnL {pr['pnl']:>+10.1f}% [{s}]")

    # WR sensitivity
    pnl_list = [t[1] for t in trades]
    actual_wr = sum(1 for p in pnl_list if p > 0) / len(pnl_list) * 100
    avg_win = np.mean([p for p in pnl_list if p > 0]) if any(p > 0 for p in pnl_list) else 0
    avg_loss = np.mean([abs(p) for p in pnl_list if p <= 0]) if any(p <= 0 for p in pnl_list) else 0

    print(f"\n  WR Sensitivity:")
    for wr_drop in [5, 10, 15, 20]:
        sim_wr = (actual_wr - wr_drop) / 100
        exp = sim_wr * avg_win - (1 - sim_wr) * avg_loss
        print(f"    WR-{wr_drop}% ({actual_wr-wr_drop:.1f}%): Exp = {exp:+.3f}%/trade {'OK' if exp > 0 else 'LOSS'}")

# ============================================================
# Phase 6: Final Summary
# ============================================================
print(f"\n{'='*70}")
print("FINAL SUMMARY")
print(f"{'='*70}\n")

print("Current vs Proposed Portfolios:")
print(f"{'':>25} {'Current':>15} {'Combined':>15}")
print(f"{'':>25} {'(TP<SL)':>15} {'(R:R>=0.75)':>15}")
print("-" * 60)

cur_trades = collect_trades(CURRENT_PORTFOLIO)
cur_res = eval_trades(cur_trades)

if combined_map:
    comb_trades = collect_trades(combined_map)
    comb_res = eval_trades(comb_trades)
    comb_tr_day = comb_res['trades'] / 270

    print(f"{'Patterns':>25} {len(CURRENT_PORTFOLIO):>15} {len(combined_map):>15}")
    print(f"{'Trades':>25} {cur_res['trades']:>15} {comb_res['trades']:>15}")
    print(f"{'Trades/day':>25} {cur_res['trades']/270:>15.1f} {comb_tr_day:>15.1f}")
    print(f"{'WR':>25} {cur_res['wr']:>14.1f}% {comb_res['wr']:>14.1f}%")
    print(f"{'Expectancy/trade':>25} {cur_res['expectancy']:>+14.3f}% {comb_res['expectancy']:>+14.3f}%")
    print(f"{'Avg Win':>25} {cur_res['avg_win']:>14.2f}% {comb_res['avg_win']:>14.2f}%")
    print(f"{'Avg Loss':>25} {cur_res['avg_loss']:>14.2f}% {comb_res['avg_loss']:>14.2f}%")
    wipeout_cur = cur_res['avg_loss'] / cur_res['avg_win'] if cur_res['avg_win'] > 0 else 0
    wipeout_comb = comb_res['avg_loss'] / comb_res['avg_win'] if comb_res['avg_win'] > 0 else 0
    print(f"{'Wipeout (loss/win)':>25} {wipeout_cur:>14.1f}x {wipeout_comb:>14.1f}x")
    print(f"{'MDD':>25} {cur_res['mdd']:>14.1f}% {comb_res['mdd']:>14.1f}%")
    print(f"{'PF':>25} {cur_res['pf']:>14.2f} {comb_res['pf']:>14.2f}")
    print(f"{'Avg R:R':>25} {0.34:>14.2f} {avg_rr:>14.2f}")

# Save
output = {
    'timestamp': datetime.now().isoformat(),
    'research': 'TP>=SL Extended v2',
    'summary': summary_rows,
    'combined_portfolio': [],
}

for v in sorted_combined:
    pat, d, tp, sl, rr, nt, wr, mc, wf, pp, tpnl, score = v
    output['combined_portfolio'].append({
        'pattern': pat, 'direction': d, 'tp': tp, 'sl': sl,
        'rr': round(rr, 2), 'trades': nt, 'wr': round(wr, 1),
        'mc': round(mc, 4), 'wf': wf, 'periods': pp, 'total_pnl': round(tpnl, 1),
    })

out_path = Path(__file__).resolve().parent / '../../results/tp_ge_sl_research_v2.json'
with open(out_path, 'w') as f:
    json.dump(output, f, indent=2, default=str)
print(f"\nResults saved to {out_path}")
print("\nDone.")
