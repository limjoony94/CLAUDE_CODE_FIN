"""
R:R Optimization Research v1.26.4
==================================
TP/SL 비율 최적화 연구 — R:R >= 1.0 (TP >= SL) 강제

배경:
- 현재 52개 패턴 중 30개(58%)가 R:R < 1.0 (TP < SL)
- R:R < 1.0은 높은 승률에 의존하므로 승률 하락 시 급격한 수익성 악화
- R:R >= 1.0이면 BE WR <= 50% → 승률 50% 초과만으로도 양수 기대값

수학적 근거:
  Break-Even WR = 1 / (R:R + 1)
  R:R = 0.5 → BE WR = 66.7% (위험)
  R:R = 1.0 → BE WR = 50.0% (안전)
  R:R = 1.5 → BE WR = 40.0% (매우 안전)
  R:R = 2.0 → BE WR = 33.3% (극안전)

Phase 1: 현재 R:R 분포 및 수학적 분석
Phase 2: R:R >= 1.0 Grid Search (모든 패턴)
Phase 3: 시나리오 비교 (SL Cap별)
Phase 4: 포트폴리오 수준 비교
Phase 5: 권고사항
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
LEVERAGE = 3
MC_SIMS = 10000

# Grid search ranges
TP_GRID = [0.5, 0.7, 1.0, 1.5, 2.0, 2.5, 3.0]
SL_GRID = [0.3, 0.5, 0.7, 1.0, 1.5, 2.0, 2.5, 3.0]

# Validation thresholds
MIN_TRADES = 10
MAX_MC = 0.01
MIN_WF = 4

print("=" * 70)
print(f"R:R Optimization Research (v{BOT_VERSION})")
print(f"  Patterns: {len(VALIDATED_LONG_PATTERNS)}L + {len(VALIDATED_SHORT_PATTERNS)}S = "
      f"{len(VALIDATED_LONG_PATTERNS) + len(VALIDATED_SHORT_PATTERNS)}")
print(f"  Grid: TP {TP_GRID}, SL {SL_GRID}")
print(f"  Constraint: TP >= SL (R:R >= 1.0)")
print("=" * 70)

# ============================================================
# Build portfolio map
# ============================================================
PORTFOLIO = {}
for pat in VALIDATED_LONG_PATTERNS:
    tp, sl = PATTERN_OPTIMAL_TPSL[pat]
    PORTFOLIO[pat] = ("LONG", tp, sl)
for pat in VALIDATED_SHORT_PATTERNS:
    tp, sl = PATTERN_OPTIMAL_TPSL[pat]
    PORTFOLIO[pat] = ("SHORT", tp, sl)

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
timestamps = df['timestamp'].values

print("Classifying candles...")
body_abs_arr = np.abs(closes - opens)
avg_body_20_arr = pd.Series(body_abs_arr).rolling(20).mean().values

types = []
for i in range(n_bars):
    row = pd.Series({'open': opens[i], 'high': highs[i], 'low': lows[i], 'close': closes[i]})
    avg_b = avg_body_20_arr[i] if not pd.isna(avg_body_20_arr[i]) else 1.0
    types.append(classify_candle(row, avg_b).value)

# Build pattern indices
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
    """Per-pattern backtest. Returns list of PnL values."""
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


def collect_trades_1pos(pattern_map):
    """1-position-at-a-time portfolio backtest."""
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


def eval_trades(trades):
    if not trades:
        return {'pnl': 0, 'trades': 0, 'wr': 0, 'mdd': 0, 'pf': 0,
                'avg_win': 0, 'avg_loss': 0, 'expectancy': 0, 'pnl_mdd': 0}
    pnl_list = [t[2] for t in trades]
    cum = 0
    peak = 0
    max_dd = 0
    wins = 0
    win_pnls = []
    loss_pnls = []
    for p in pnl_list:
        cum += p
        if cum > peak:
            peak = cum
        dd = peak - cum
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
    wr = wins / len(pnl_list) * 100
    exp = (wr / 100 * avg_win) - ((1 - wr / 100) * avg_loss)
    return {
        'pnl': cum, 'trades': len(pnl_list), 'wr': wr, 'mdd': max_dd,
        'pf': total_win / total_loss if total_loss > 0 else 999,
        'avg_win': avg_win, 'avg_loss': avg_loss, 'expectancy': exp,
        'pnl_mdd': cum / max_dd if max_dd > 0 else 999,
    }


# ============================================================
# Phase 1: Current R:R Distribution Analysis
# ============================================================
print(f"\n{'='*70}")
print("Phase 1: Current R:R Distribution & Mathematical Analysis")
print(f"{'='*70}")

print(f"\n  R:R >= 1.0 → Break-Even WR <= 50% (수학적 안전 영역)")
print(f"  R:R < 1.0  → Break-Even WR > 50% (승률 의존적, 취약)")
print(f"\n  {'Pattern':<12} {'Dir':>5} {'TP':>5} {'SL':>5} {'R:R':>5} {'BE WR':>6} {'Actual':>7} {'Edge':>7} {'MaxLoss':>8} {'Status':>10}")
print("  " + "-" * 90)

rr_analysis = []
below_1 = []
above_1 = []

for pat in sorted(PORTFOLIO.keys()):
    direction, tp, sl = PORTFOLIO[pat]
    rr = tp / sl if sl > 0 else 999
    be_wr = 1 / (rr + 1) * 100  # break-even WR accounting for R:R
    # Adjust BE WR for fees: need (WR*TP*L - (1-WR)*SL*L - FEE) > 0
    # WR*(TP*L) - (1-WR)*(SL*L) > FEE  →  WR > (SL*L + FEE) / (TP*L + SL*L)
    be_wr_with_fees = (sl * LEVERAGE + FEE_PCT) / (tp * LEVERAGE + sl * LEVERAGE) * 100
    actual_wr = PATTERN_STATS.get(pat, {}).get('wr', 0)
    edge = actual_wr - be_wr_with_fees
    max_loss = sl * LEVERAGE + FEE_PCT

    status = "SAFE" if edge >= 15 else "OK" if edge >= 5 else "FRAGILE" if edge >= 0 else "NEGATIVE"
    flag = "" if rr >= 1.0 else " [R:R<1]"

    entry = {
        'pattern': pat, 'direction': direction,
        'tp': tp, 'sl': sl, 'rr': round(rr, 2),
        'be_wr': round(be_wr_with_fees, 1), 'actual_wr': actual_wr,
        'edge': round(edge, 1), 'max_loss': round(max_loss, 1),
        'status': status,
    }
    rr_analysis.append(entry)

    if rr < 1.0:
        below_1.append(entry)
    else:
        above_1.append(entry)

    print(f"  {pat:<12} {direction:>5} {tp:>5.1f} {sl:>5.1f} {rr:>5.2f} {be_wr_with_fees:>5.1f}% {actual_wr:>6.1f}% {edge:>+6.1f}pp {max_loss:>7.1f}% {status:>8}{flag}")

# Summary
print(f"\n  Summary:")
print(f"    R:R >= 1.0: {len(above_1)}/52 ({len(above_1)/52*100:.0f}%) — BE WR <= 50%")
print(f"    R:R < 1.0:  {len(below_1)}/52 ({len(below_1)/52*100:.0f}%) — BE WR > 50%, 승률 의존적")

fragile = [e for e in rr_analysis if e['status'] == 'FRAGILE']
negative = [e for e in rr_analysis if e['status'] == 'NEGATIVE']
print(f"    FRAGILE (edge < 5pp): {len(fragile)} patterns")
print(f"    NEGATIVE edge: {len(negative)} patterns")

avg_rr = np.mean([e['rr'] for e in rr_analysis])
avg_edge = np.mean([e['edge'] for e in rr_analysis])
avg_max_loss = np.mean([e['max_loss'] for e in rr_analysis])
print(f"    Avg R:R: {avg_rr:.2f}, Avg edge: {avg_edge:+.1f}pp, Avg max loss: {avg_max_loss:.1f}%")


# ============================================================
# Phase 2: R:R >= 1.0 Grid Search
# ============================================================
print(f"\n{'='*70}")
print("Phase 2: R:R >= 1.0 Grid Search (all 52 patterns)")
print(f"{'='*70}")

t0 = time.time()

# Generate valid TP/SL combinations where TP >= SL
valid_combos = [(tp, sl) for tp in TP_GRID for sl in SL_GRID if tp >= sl]
print(f"\n  Valid R:R >= 1.0 combinations: {len(valid_combos)}")
print(f"  Total backtests: {len(PORTFOLIO)} patterns × {len(valid_combos)} combos = {len(PORTFOLIO)*len(valid_combos)}")

grid_results = {}  # pat -> best result

for pat in sorted(PORTFOLIO.keys()):
    direction = PORTFOLIO[pat][0]
    indices = pattern_indices.get(pat, np.array([]))
    if len(indices) == 0:
        continue

    current_tp, current_sl = PATTERN_OPTIMAL_TPSL[pat]
    current_rr = current_tp / current_sl if current_sl > 0 else 999

    # Test all valid combos
    candidates = []
    for tp, sl in valid_combos:
        pnls = bt_fixed(indices, direction, tp, sl)
        n_trades = len(pnls)
        if n_trades < MIN_TRADES:
            continue
        wr = sum(1 for x in pnls if x > 0) / n_trades * 100
        total_pnl = sum(pnls)
        avg_win = np.mean([x for x in pnls if x > 0]) if any(x > 0 for x in pnls) else 0
        avg_loss = np.mean([abs(x) for x in pnls if x <= 0]) if any(x <= 0 for x in pnls) else 0
        exp = (wr / 100) * avg_win - (1 - wr / 100) * avg_loss
        rr = tp / sl

        if exp <= 0:
            continue

        candidates.append({
            'tp': tp, 'sl': sl, 'rr': round(rr, 2),
            'trades': n_trades, 'wr': round(wr, 1),
            'expectancy': round(exp, 3), 'total_pnl': round(total_pnl, 1),
        })

    if not candidates:
        grid_results[pat] = {
            'direction': direction,
            'current': {'tp': current_tp, 'sl': current_sl, 'rr': round(current_rr, 2)},
            'best_rr1': None,
            'status': 'NO_CANDIDATE',
        }
        continue

    # Sort by total PnL (primary) — this selects the most profitable R:R >= 1.0 option
    candidates.sort(key=lambda c: -c['total_pnl'])
    best = candidates[0]

    # Validate the best candidate
    mc = mc_test(bt_fixed(indices, direction, best['tp'], best['sl']), MC_SIMS)
    wf = walk_forward_test(indices, direction, best['tp'], best['sl'])

    valid = mc < MAX_MC and wf >= MIN_WF

    # Also get current performance for comparison
    current_pnls = bt_fixed(indices, direction, current_tp, current_sl)
    current_exp = 0
    if current_pnls:
        cwr = sum(1 for x in current_pnls if x > 0) / len(current_pnls) * 100
        caw = np.mean([x for x in current_pnls if x > 0]) if any(x > 0 for x in current_pnls) else 0
        cal = np.mean([abs(x) for x in current_pnls if x <= 0]) if any(x <= 0 for x in current_pnls) else 0
        current_exp = (cwr / 100) * caw - (1 - cwr / 100) * cal

    grid_results[pat] = {
        'direction': direction,
        'current': {
            'tp': current_tp, 'sl': current_sl, 'rr': round(current_rr, 2),
            'trades': len(current_pnls), 'wr': round(cwr, 1) if current_pnls else 0,
            'expectancy': round(current_exp, 3), 'total_pnl': round(sum(current_pnls), 1) if current_pnls else 0,
        },
        'best_rr1': {
            **best,
            'mc': round(mc, 4), 'wf': wf, 'valid': valid,
        },
        'all_candidates': len(candidates),
        'exp_change': round(best['expectancy'] - current_exp, 3),
        'pnl_change': round(best['total_pnl'] - (sum(current_pnls) if current_pnls else 0), 1),
        'status': 'VALID' if valid else 'MC_FAIL' if mc >= MAX_MC else 'WF_FAIL',
    }

elapsed = time.time() - t0
print(f"\n  Grid search done in {elapsed:.1f}s")

# Print results
valid_count = sum(1 for r in grid_results.values() if r.get('status') == 'VALID')
no_candidate = sum(1 for r in grid_results.values() if r.get('status') == 'NO_CANDIDATE')
mc_fail = sum(1 for r in grid_results.values() if r.get('status') == 'MC_FAIL')
wf_fail = sum(1 for r in grid_results.values() if r.get('status') == 'WF_FAIL')

print(f"\n  Results: {valid_count} VALID, {mc_fail} MC_FAIL, {wf_fail} WF_FAIL, {no_candidate} NO_CANDIDATE")

# Detailed table
print(f"\n  {'Pattern':<12} {'Dir':>5} | {'Cur TP/SL':>9} {'Cur R:R':>7} {'Cur Exp':>8} | {'New TP/SL':>9} {'New R:R':>7} {'New Exp':>8} {'MC':>7} {'WF':>4} | {'ExpΔ':>7} {'Status':>10}")
print("  " + "-" * 120)

for pat in sorted(grid_results.keys()):
    r = grid_results[pat]
    cur = r['current']
    cur_tpsl = f"{cur['tp']}/{cur['sl']}"

    if r['best_rr1'] is None:
        print(f"  {pat:<12} {r['direction']:>5} | {cur_tpsl:>9} {cur['rr']:>7.2f} {cur.get('expectancy',0):>+7.3f}% | {'---':>9} {'---':>7} {'---':>8} {'---':>7} {'---':>4} | {'---':>7} {'NO_CAND':>10}")
        continue

    best = r['best_rr1']
    new_tpsl = f"{best['tp']}/{best['sl']}"
    flag = " ***" if r['status'] == 'VALID' and r.get('exp_change', 0) < -0.5 else ""

    print(f"  {pat:<12} {r['direction']:>5} | {cur_tpsl:>9} {cur['rr']:>7.2f} {cur.get('expectancy',0):>+7.3f}% | "
          f"{new_tpsl:>9} {best['rr']:>7.2f} {best['expectancy']:>+7.3f}% {best['mc']:>7.4f} {best['wf']:>3}/5 | "
          f"{r.get('exp_change', 0):>+6.3f}% {r['status']:>10}{flag}")

# Patterns that already have R:R >= 1.0 — how do they change?
print(f"\n  --- Patterns already R:R >= 1.0 (unchanged or improved) ---")
already_rr1 = [(p, r) for p, r in grid_results.items() if r['current']['rr'] >= 1.0]
print(f"    {len(already_rr1)} patterns already have R:R >= 1.0")
improved = sum(1 for _, r in already_rr1 if r.get('exp_change', 0) > 0 and r.get('status') == 'VALID')
print(f"    {improved} could be further improved with R:R >= 1.0 search")

# Patterns that need conversion (R:R < 1.0 → R:R >= 1.0)
print(f"\n  --- Patterns needing conversion (R:R < 1.0 → >= 1.0) ---")
need_convert = [(p, r) for p, r in grid_results.items() if r['current']['rr'] < 1.0]
converted_valid = sum(1 for _, r in need_convert if r.get('status') == 'VALID')
converted_fail = sum(1 for _, r in need_convert if r.get('status') not in ('VALID', 'NO_CANDIDATE'))
no_option = sum(1 for _, r in need_convert if r.get('status') == 'NO_CANDIDATE')
print(f"    {len(need_convert)} patterns need conversion")
print(f"    {converted_valid} successfully converted (VALID)")
print(f"    {converted_fail} conversion failed (MC/WF fail)")
print(f"    {no_option} no viable R:R >= 1.0 option")


# ============================================================
# Phase 3: Scenario Comparison (SL Cap variants)
# ============================================================
print(f"\n{'='*70}")
print("Phase 3: Scenario Portfolios")
print(f"{'='*70}")


def build_scenario_portfolio(grid_res, sl_cap=None, allow_current_fallback=False):
    """Build portfolio from grid search results.

    Args:
        grid_res: grid search results dict
        sl_cap: max SL allowed (None = no cap)
        allow_current_fallback: if True, keep current TP/SL when R:R >= 1.0 conversion fails
    """
    pmap = {}
    included = 0
    dropped = 0
    fallback = 0
    converted = 0

    for pat, r in grid_res.items():
        direction = r['direction']
        cur = r['current']

        # Check if best R:R >= 1.0 candidate is valid and within SL cap
        best = r.get('best_rr1')
        use_new = False
        if best and r.get('status') == 'VALID':
            if sl_cap is None or best['sl'] <= sl_cap:
                use_new = True

        if use_new:
            pmap[pat] = (direction, best['tp'], best['sl'])
            if cur['rr'] < 1.0:
                converted += 1
            included += 1
        elif allow_current_fallback:
            # Fallback to current if within SL cap
            if sl_cap is None or cur['sl'] <= sl_cap:
                pmap[pat] = (direction, cur['tp'], cur['sl'])
                fallback += 1
                included += 1
            else:
                dropped += 1
        else:
            # Pure R:R >= 1.0 — drop patterns without valid conversion
            dropped += 1

    return pmap, {'included': included, 'dropped': dropped, 'fallback': fallback, 'converted': converted}


scenarios = {}

# Scenario A: Current portfolio (baseline)
scenario_a_map = {p: v for p, v in PORTFOLIO.items()}
trades_a = collect_trades_1pos(scenario_a_map)
stats_a = eval_trades(trades_a)
scenarios['A_Current'] = {
    'desc': 'Current v1.26.4 (no R:R constraint)',
    'patterns': len(scenario_a_map),
    'stats': stats_a,
    'meta': {'included': len(scenario_a_map), 'dropped': 0, 'fallback': 0, 'converted': 0},
}

# Scenario B: Pure R:R >= 1.0 (drop failures)
pmap_b, meta_b = build_scenario_portfolio(grid_results, sl_cap=None, allow_current_fallback=False)
trades_b = collect_trades_1pos(pmap_b)
stats_b = eval_trades(trades_b)
scenarios['B_RR1_Pure'] = {
    'desc': 'R:R >= 1.0 only (drop patterns without valid conversion)',
    'patterns': len(pmap_b),
    'stats': stats_b,
    'meta': meta_b,
}

# Scenario C: R:R >= 1.0 with fallback (keep current when conversion fails)
pmap_c, meta_c = build_scenario_portfolio(grid_results, sl_cap=None, allow_current_fallback=True)
trades_c = collect_trades_1pos(pmap_c)
stats_c = eval_trades(trades_c)
scenarios['C_RR1_Hybrid'] = {
    'desc': 'R:R >= 1.0 where possible, keep current otherwise',
    'patterns': len(pmap_c),
    'stats': stats_c,
    'meta': meta_c,
}

# Scenario D: R:R >= 1.0 + SL cap 2.0% (pure)
pmap_d, meta_d = build_scenario_portfolio(grid_results, sl_cap=2.0, allow_current_fallback=False)
trades_d = collect_trades_1pos(pmap_d)
stats_d = eval_trades(trades_d)
scenarios['D_RR1_SL2'] = {
    'desc': 'R:R >= 1.0, SL <= 2.0% (max loss 6.1%)',
    'patterns': len(pmap_d),
    'stats': stats_d,
    'meta': meta_d,
}

# Scenario E: R:R >= 1.0 + SL cap 2.0% with fallback
pmap_e, meta_e = build_scenario_portfolio(grid_results, sl_cap=2.0, allow_current_fallback=True)
trades_e = collect_trades_1pos(pmap_e)
stats_e = eval_trades(trades_e)
scenarios['E_RR1_SL2_Hybrid'] = {
    'desc': 'R:R >= 1.0 + SL <= 2.0% where possible, keep current otherwise',
    'patterns': len(pmap_e),
    'stats': stats_e,
    'meta': meta_e,
}

# Scenario F: R:R >= 1.0 + SL cap 1.5% (pure, max loss 4.6%)
pmap_f, meta_f = build_scenario_portfolio(grid_results, sl_cap=1.5, allow_current_fallback=False)
trades_f = collect_trades_1pos(pmap_f)
stats_f = eval_trades(trades_f)
scenarios['F_RR1_SL1.5'] = {
    'desc': 'R:R >= 1.0, SL <= 1.5% (max loss 4.6%)',
    'patterns': len(pmap_f),
    'stats': stats_f,
    'meta': meta_f,
}

# Scenario G: R:R >= 1.0 + SL cap 2.5% (pure)
pmap_g, meta_g = build_scenario_portfolio(grid_results, sl_cap=2.5, allow_current_fallback=False)
trades_g = collect_trades_1pos(pmap_g)
stats_g = eval_trades(trades_g)
scenarios['G_RR1_SL2.5'] = {
    'desc': 'R:R >= 1.0, SL <= 2.5% (max loss 7.6%)',
    'patterns': len(pmap_g),
    'stats': stats_g,
    'meta': meta_g,
}


# ============================================================
# Phase 4: Portfolio-Level Comparison
# ============================================================
print(f"\n{'='*70}")
print("Phase 4: Portfolio-Level Comparison")
print(f"{'='*70}")

print(f"\n  {'Scenario':<22} {'Pats':>4} {'Conv':>4} {'Drop':>4} {'FB':>4} | {'Trades':>6} {'WR':>6} {'Exp':>8} {'PnL':>10} {'MDD':>7} {'P/M':>7} {'PF':>6} {'MaxL':>6}")
print("  " + "-" * 115)

for name, sc in scenarios.items():
    s = sc['stats']
    m = sc['meta']
    # Calculate max single loss for this scenario
    if name == 'A_Current':
        pmap_ref = scenario_a_map
    elif name == 'B_RR1_Pure':
        pmap_ref = pmap_b
    elif name == 'C_RR1_Hybrid':
        pmap_ref = pmap_c
    elif name == 'D_RR1_SL2':
        pmap_ref = pmap_d
    elif name == 'E_RR1_SL2_Hybrid':
        pmap_ref = pmap_e
    elif name == 'F_RR1_SL1.5':
        pmap_ref = pmap_f
    elif name == 'G_RR1_SL2.5':
        pmap_ref = pmap_g
    else:
        pmap_ref = {}
    max_sl = max((v[2] for v in pmap_ref.values()), default=0)
    max_loss = max_sl * LEVERAGE + FEE_PCT

    print(f"  {name:<22} {sc['patterns']:>4} {m['converted']:>4} {m['dropped']:>4} {m['fallback']:>4} | "
          f"{s['trades']:>6} {s['wr']:>5.1f}% {s['expectancy']:>+7.3f}% {s['pnl']:>+9.1f}% {s['mdd']:>6.1f}% {s['pnl_mdd']:>6.2f}x {s['pf']:>5.2f} {max_loss:>5.1f}%")

# Highlight key comparisons
print(f"\n  Key Comparisons vs Current (A):")
for name, sc in scenarios.items():
    if name == 'A_Current':
        continue
    s = sc['stats']
    sa = scenarios['A_Current']['stats']
    pnl_d = s['pnl'] - sa['pnl']
    mdd_d = s['mdd'] - sa['mdd']
    wr_d = s['wr'] - sa['wr']
    pm_d = s['pnl_mdd'] - sa['pnl_mdd']
    print(f"    {name:<22}: PnL {pnl_d:>+8.1f}%, MDD {mdd_d:>+6.1f}%, WR {wr_d:>+5.1f}%, PnL/MDD {pm_d:>+6.2f}x")


# ============================================================
# Phase 5: Deep Analysis of Best Scenario
# ============================================================
print(f"\n{'='*70}")
print("Phase 5: Deep Analysis & Recommendations")
print(f"{'='*70}")

# Find best scenario by PnL/MDD (excluding current)
best_name = max(
    [n for n in scenarios if n != 'A_Current'],
    key=lambda n: scenarios[n]['stats']['pnl_mdd'] if scenarios[n]['stats']['trades'] >= 50 else -999
)
best_sc = scenarios[best_name]

print(f"\n  Best scenario by PnL/MDD: {best_name}")
print(f"    {best_sc['desc']}")
bs = best_sc['stats']
print(f"    Patterns: {best_sc['patterns']}, Trades: {bs['trades']}")
print(f"    PnL: {bs['pnl']:+.1f}%, MDD: {bs['mdd']:.1f}%, PnL/MDD: {bs['pnl_mdd']:.2f}x")
print(f"    WR: {bs['wr']:.1f}%, PF: {bs['pf']:.2f}, Exp: {bs['expectancy']:+.3f}%")

# Mathematical analysis of R:R >= 1.0
print(f"\n  Mathematical Framework:")
print(f"    R:R >= 1.0 guarantees: Break-Even WR <= 50%")
print(f"    Current portfolio avg WR: {scenarios['A_Current']['stats']['wr']:.1f}%")
print(f"    → With R:R >= 1.0, WR could drop by {scenarios['A_Current']['stats']['wr'] - 50:.1f}pp and still break even")

# Conversion feasibility summary
print(f"\n  R:R < 1.0 → R:R >= 1.0 Conversion Summary:")
need_convert = [(p, r) for p, r in grid_results.items() if r['current']['rr'] < 1.0]
for pat, r in sorted(need_convert, key=lambda x: x[1].get('exp_change', 0)):
    cur = r['current']
    best = r.get('best_rr1')
    if best is None:
        print(f"    {pat:<12} ({r['direction']:>5}): R:R {cur['rr']:.2f} → NO VIABLE R:R>=1.0 OPTION")
    else:
        delta_exp = r.get('exp_change', 0)
        direction_str = "↑" if delta_exp > 0 else "↓"
        print(f"    {pat:<12} ({r['direction']:>5}): R:R {cur['rr']:.2f} → {best['rr']:.2f}, "
              f"Exp {cur.get('expectancy',0):+.3f} → {best['expectancy']:+.3f}% ({direction_str}{abs(delta_exp):.3f}%), "
              f"{'VALID' if r['status'] == 'VALID' else r['status']}")

# Trade-off analysis
print(f"\n  Trade-off Analysis:")
sa = scenarios['A_Current']['stats']
sb = scenarios['B_RR1_Pure']['stats']
sc = scenarios['C_RR1_Hybrid']['stats']
print(f"    Pure R:R >= 1.0 (B) vs Current (A):")
print(f"      PnL: {sb['pnl']:+.1f}% vs {sa['pnl']:+.1f}% ({sb['pnl']-sa['pnl']:+.1f}%)")
print(f"      MDD: {sb['mdd']:.1f}% vs {sa['mdd']:.1f}% ({sb['mdd']-sa['mdd']:+.1f}%)")
print(f"      WR:  {sb['wr']:.1f}% vs {sa['wr']:.1f}% ({sb['wr']-sa['wr']:+.1f}%)")
print(f"      Patterns: {scenarios['B_RR1_Pure']['patterns']} vs {scenarios['A_Current']['patterns']} (-{scenarios['A_Current']['patterns']-scenarios['B_RR1_Pure']['patterns']})")
print(f"    Hybrid R:R >= 1.0 (C) vs Current (A):")
print(f"      PnL: {sc['pnl']:+.1f}% vs {sa['pnl']:+.1f}% ({sc['pnl']-sa['pnl']:+.1f}%)")
print(f"      MDD: {sc['mdd']:.1f}% vs {sa['mdd']:.1f}% ({sc['mdd']-sa['mdd']:+.1f}%)")
print(f"      WR:  {sc['wr']:.1f}% vs {sa['wr']:.1f}% ({sc['wr']-sa['wr']:+.1f}%)")

# Recommendations
print(f"\n  {'='*60}")
print(f"  RECOMMENDATIONS")
print(f"  {'='*60}")

# Check if pure R:R >= 1.0 is worth it
pnl_loss_pct = (sa['pnl'] - sb['pnl']) / sa['pnl'] * 100 if sa['pnl'] > 0 else 0
mdd_improvement = sa['mdd'] - sb['mdd']

print(f"\n  1. R:R >= 1.0 Pure (Scenario B):")
if sb['pnl_mdd'] >= sa['pnl_mdd']:
    print(f"     RECOMMENDED — PnL/MDD improves: {sb['pnl_mdd']:.2f}x vs {sa['pnl_mdd']:.2f}x")
else:
    print(f"     {'WORTH CONSIDERING' if pnl_loss_pct < 20 else 'NOT RECOMMENDED'} — "
          f"PnL/MDD: {sb['pnl_mdd']:.2f}x vs {sa['pnl_mdd']:.2f}x, "
          f"PnL loss: {pnl_loss_pct:.1f}%")

print(f"\n  2. R:R >= 1.0 Hybrid (Scenario C):")
print(f"     Keeps all {scenarios['C_RR1_Hybrid']['patterns']} patterns, "
      f"PnL/MDD: {sc['pnl_mdd']:.2f}x vs {sa['pnl_mdd']:.2f}x")

# Find best SL-capped scenario
for name in ['D_RR1_SL2', 'G_RR1_SL2.5', 'F_RR1_SL1.5']:
    sc_x = scenarios[name]
    sx = sc_x['stats']
    print(f"\n  3. {name} ({sc_x['desc']}):")
    print(f"     Patterns: {sc_x['patterns']}, PnL: {sx['pnl']:+.1f}%, PnL/MDD: {sx['pnl_mdd']:.2f}x")


# ============================================================
# Save Results
# ============================================================
output = {
    'timestamp': datetime.now().isoformat(),
    'research': 'R:R Optimization Research v1.26.4',
    'version': BOT_VERSION,
    'portfolio_size': len(PORTFOLIO),
    'constraint': 'TP >= SL (R:R >= 1.0)',
    'grid': {'tp': TP_GRID, 'sl': SL_GRID, 'valid_combos': len(valid_combos)},
    'phase1_rr_analysis': {
        'rr_above_1': len(above_1),
        'rr_below_1': len(below_1),
        'avg_rr': round(avg_rr, 2),
        'avg_edge': round(avg_edge, 1),
        'fragile_count': len(fragile),
    },
    'phase2_grid_search': {
        'valid': valid_count,
        'mc_fail': mc_fail,
        'wf_fail': wf_fail,
        'no_candidate': no_candidate,
        'per_pattern': {p: {
            'direction': r['direction'],
            'current_tp': r['current']['tp'],
            'current_sl': r['current']['sl'],
            'current_rr': r['current']['rr'],
            'current_exp': r['current'].get('expectancy', 0),
            'best_tp': r['best_rr1']['tp'] if r['best_rr1'] else None,
            'best_sl': r['best_rr1']['sl'] if r['best_rr1'] else None,
            'best_rr': r['best_rr1']['rr'] if r['best_rr1'] else None,
            'best_exp': r['best_rr1']['expectancy'] if r['best_rr1'] else None,
            'best_mc': r['best_rr1']['mc'] if r['best_rr1'] else None,
            'best_wf': r['best_rr1']['wf'] if r['best_rr1'] else None,
            'status': r['status'],
            'exp_change': r.get('exp_change', None),
        } for p, r in grid_results.items()},
    },
    'phase3_scenarios': {name: {
        'desc': sc['desc'],
        'patterns': sc['patterns'],
        'meta': sc['meta'],
        'stats': {k: round(v, 2) if isinstance(v, float) else v for k, v in sc['stats'].items()},
    } for name, sc in scenarios.items()},
    'best_scenario': best_name,
}

out_path = Path(__file__).resolve().parent / '../../results/rr_optimization_research.json'
with open(out_path, 'w') as f:
    json.dump(output, f, indent=2, default=str)
print(f"\nResults saved to {out_path}")
print("\nDone.")
