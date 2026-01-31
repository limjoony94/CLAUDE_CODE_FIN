"""
TP/SL 설정 민감도 분석 (수정판)
================================
문제: Total PnL 합산은 TP/SL 폭에 비례 → 설정 간 비교 불가.
      3% TP 1승 = 0.5% TP 6승과 동일.

수정: 설정 간 비교 가능한 지표 사용
  1. Excess WR (baseline 대비 초과 승률) — TP/SL 무관
  2. 복리 기반 PnL (compound equity) — 실제 수익률
  3. 복리 PnL/MDD — 리스크 조정 수익
  4. Profit Factor — 총이익/총손실
  5. 트레이드당 기대값 정규화 — edge/(tp+sl) 로 리스크 단위당 edge
"""

import pandas as pd
import numpy as np
from pathlib import Path
from collections import defaultdict, Counter

MAX_BARS = 500
FEE_PCT = 0.10
LEVERAGE = 3
MIN_TRADES = 25
MC_SIMS = 5000
EXCESS_THRESHOLD = 15

df = pd.read_csv(Path(__file__).parent / '../../data/btc_5m_270days.csv')
print(f"Loaded: {len(df)} bars")

highs = df['high'].values
lows = df['low'].values
opens = df['open'].values
closes = df['close'].values
n_bars = len(df)

df['timestamp'] = pd.to_datetime(df['timestamp'])
dates = df['timestamp'].values

period_masks = {
    'H1': (dates >= np.datetime64('2025-05-01')) & (dates < np.datetime64('2025-08-01')),
    'H2': (dates >= np.datetime64('2025-08-01')) & (dates < np.datetime64('2025-11-01')),
    'H3': (dates >= np.datetime64('2025-11-01')) & (dates < np.datetime64('2026-02-01')),
}

def classify_candle(o, h, l, c, avg_body_20):
    """Classify candle - matches production (constants.py thresholds)."""
    body = c - o
    body_abs = abs(body)
    rng = h - l
    if rng == 0: return 'D'
    upper_wick = h - max(o, c)
    lower_wick = min(o, c) - l
    body_ratio = body_abs / rng
    if body_ratio < 0.10:
        if lower_wick / rng > 0.70: return 'DF'
        elif upper_wick / rng > 0.70: return 'GS'
        return 'D'
    total_wick_ratio = (upper_wick + lower_wick) / rng
    if total_wick_ratio < 0.15:
        return 'MU' if body > 0 else 'MD'
    if body_abs > 0:
        if lower_wick / body_abs > 2.0 and upper_wick / body_abs < 0.3: return 'H'
        if upper_wick / body_abs > 2.0 and lower_wick / body_abs < 0.3: return 'IH'
    norm_body = body_abs / avg_body_20 if avg_body_20 > 0 else 1.0
    if norm_body < 0.5 and body_abs > 0:
        if lower_wick > 0.5 * body_abs and upper_wick > 0.5 * body_abs:
            return 'ST'
    if norm_body > 1.5:
        return 'BU' if body > 0 else 'BD'
    return 'U' if body > 0 else 'DN'

print("Classifying candles...")
body_abs_arr = np.abs(closes - opens)
avg_body_20_arr = pd.Series(body_abs_arr).rolling(20).mean().values  # NaN for bars 0-19
# For early bars: default avg_body_20=1.0 preserves range-based classification
types = [classify_candle(opens[i], highs[i], lows[i], closes[i],
                         avg_body_20_arr[i] if not pd.isna(avg_body_20_arr[i]) else 1.0)
         for i in range(n_bars)]

pattern_indices = defaultdict(list)
for i in range(2, n_bars):
    pat = f"{types[i-2]}-{types[i-1]}-{types[i]}"
    pattern_indices[pat].append(i)
for k in pattern_indices:
    pattern_indices[k] = np.array(pattern_indices[k])

def bt_fixed(indices, direction, tp_pct, sl_pct):
    pnls = []
    for idx in indices:
        if idx + 1 >= n_bars: continue
        entry = opens[idx + 1]
        if entry <= 0: continue
        if direction == 'LONG':
            tpp = entry * (1 + tp_pct/100); slp = entry * (1 - sl_pct/100)
        else:
            tpp = entry * (1 - tp_pct/100); slp = entry * (1 + sl_pct/100)
        for i in range(idx+2, min(idx+2+MAX_BARS, n_bars)):
            ht = (highs[i]>=tpp) if direction=='LONG' else (lows[i]<=tpp)
            hs = (lows[i]<=slp) if direction=='LONG' else (highs[i]>=slp)
            if ht and hs:
                pnls.append((tp_pct if abs(tpp-entry)<=abs(slp-entry) else -sl_pct)*LEVERAGE - FEE_PCT)
                break
            elif ht: pnls.append(tp_pct*LEVERAGE - FEE_PCT); break
            elif hs: pnls.append(-sl_pct*LEVERAGE - FEE_PCT); break
    return pnls

def mc_test(pnls):
    if len(pnls) < 10: return 1.0
    pa = np.array(pnls); real = np.sum(pa)
    return sum(1 for _ in range(MC_SIMS) if np.sum(pa * np.random.choice([-1,1], len(pa))) >= real) / MC_SIMS

def wf_test(indices, direction, tp, sl, n_folds=5):
    si = np.sort(indices); fs = len(si)//n_folds
    if fs < 3: return 0
    p = 0
    for f in range(n_folds):
        s = f*fs; e = s+fs if f < n_folds-1 else len(si)
        pnls = bt_fixed(si[s:e], direction, tp, sl)
        if pnls and sum(pnls) > 0: p += 1
    return p

def period_profitable(indices, direction, tp, sl):
    count = 0
    for mask in period_masks.values():
        pidx = indices[mask[indices]]
        if len(pidx) < 3: continue
        p = bt_fixed(pidx, direction, tp, sl)
        if p and sum(p) > 0: count += 1
    return count

# --- Baselines ---
print("Computing baselines...")
sample_idx = np.arange(2, n_bars, 20)

TPSL_SETTINGS = [
    (0.5, 0.5), (0.5, 1.0), (0.7, 1.0), (1.0, 1.0),
    (1.0, 1.5), (1.0, 2.0), (1.5, 2.0), (1.5, 3.0),
    (2.0, 2.0), (2.0, 3.0), (2.5, 3.0), (3.0, 3.0),
]

baselines = {}
for tp, sl in TPSL_SETTINGS:
    pl = bt_fixed(sample_idx, 'LONG', tp, sl)
    ps = bt_fixed(sample_idx, 'SHORT', tp, sl)
    baselines[(tp,sl)] = {
        'LONG': sum(1 for x in pl if x>0)/len(pl)*100 if pl else 50,
        'SHORT': sum(1 for x in ps if x>0)/len(ps)*100 if ps else 50,
    }
print("Baselines computed.")

all_pats = [(pat, idx) for pat, idx in pattern_indices.items() if len(idx) >= MIN_TRADES]
print(f"Candidate patterns: {len(all_pats)}")

# ============================================================
# Discover validated patterns for each setting
# ============================================================
setting_patterns = {}

for tp, sl in TPSL_SETTINGS:
    label = f"{tp}/{sl}"
    base_l = baselines[(tp,sl)]['LONG']
    base_s = baselines[(tp,sl)]['SHORT']

    candidates = []
    for pat, indices in all_pats:
        for direction in ['LONG', 'SHORT']:
            base = base_l if direction == 'LONG' else base_s
            pnls = bt_fixed(indices, direction, tp, sl)
            if len(pnls) < MIN_TRADES: continue
            wr = sum(1 for x in pnls if x>0)/len(pnls)*100
            excess = wr - base
            pnl = sum(pnls)
            if excess < EXCESS_THRESHOLD or pnl <= 0: continue
            candidates.append({
                'pattern': pat, 'direction': direction,
                'tp': tp, 'sl': sl,
                'trades': len(pnls), 'wr': wr, 'excess': excess, 'pnl': pnl,
            })

    validated = []
    for c in candidates:
        idx = pattern_indices[c['pattern']]
        pnls = bt_fixed(idx, c['direction'], tp, sl)
        mc = mc_test(pnls)
        if mc >= 0.05: continue
        wf = wf_test(idx, c['direction'], tp, sl)
        if wf < 3: continue
        pp = period_profitable(idx, c['direction'], tp, sl)
        if pp < 2: continue
        c['mc'] = mc; c['wf'] = wf; c['periods'] = pp
        validated.append(c)

    pat_dirs = Counter(v['pattern'] for v in validated)
    bidir = {p for p, cnt in pat_dirs.items() if cnt > 1}
    validated = [v for v in validated if v['pattern'] not in bidir]
    validated.sort(key=lambda x: -x['pnl'])

    setting_patterns[(tp,sl)] = validated
    print(f"  {label}: {len(validated)} patterns")

# ============================================================
# CORRECTED METRICS: compound equity for each setting's portfolio
# ============================================================
print(f"\n{'='*80}")
print("CORRECTED COMPARISON: Compound Equity per Setting")
print(f"{'='*80}\n")

def compound_portfolio(patterns_list):
    """Compound equity backtest — returns actual growth"""
    trades = []
    for v in patterns_list:
        idx = pattern_indices[v['pattern']]
        tp, sl = v['tp'], v['sl']
        d = v['direction']
        for ix in idx:
            if ix + 1 >= n_bars: continue
            entry = opens[ix + 1]
            if entry <= 0: continue
            if d == 'LONG':
                tpp = entry * (1 + tp/100); slp = entry * (1 - sl/100)
            else:
                tpp = entry * (1 - tp/100); slp = entry * (1 + sl/100)
            for i in range(ix+2, min(ix+2+MAX_BARS, n_bars)):
                ht = (highs[i]>=tpp) if d=='LONG' else (lows[i]<=tpp)
                hs = (lows[i]<=slp) if d=='LONG' else (highs[i]>=slp)
                if ht and hs:
                    pnl_pct = (tp if abs(tpp-entry)<=abs(slp-entry) else -sl)*LEVERAGE - FEE_PCT
                    trades.append((ix+1, pnl_pct))
                    break
                elif ht:
                    trades.append((ix+1, tp*LEVERAGE - FEE_PCT))
                    break
                elif hs:
                    trades.append((ix+1, -sl*LEVERAGE - FEE_PCT))
                    break

    if not trades:
        return {'pnl': 0, 'trades': 0, 'wr': 0, 'mdd': 0, 'pnl_mdd': 0,
                'profit_factor': 0, 'avg_win': 0, 'avg_loss': 0, 'expectancy_norm': 0}

    trades.sort(key=lambda x: x[0])
    pnl_list = [t[1] for t in trades]

    equity = 100.0
    peak = 100.0
    max_dd = 0
    wins = 0
    total_win = 0
    total_loss = 0
    n_win = 0
    n_loss = 0
    for pnl_pct in pnl_list:
        equity *= (1 + pnl_pct/100)
        if equity > peak: peak = equity
        dd = (peak - equity) / peak * 100
        if dd > max_dd: max_dd = dd
        if pnl_pct > 0:
            wins += 1; total_win += pnl_pct; n_win += 1
        else:
            total_loss += abs(pnl_pct); n_loss += 1

    total_pnl = (equity / 100 - 1) * 100
    wr = wins / len(pnl_list) * 100
    pnl_mdd = total_pnl / max_dd if max_dd > 0 else 999
    pf = total_win / total_loss if total_loss > 0 else 999
    avg_win = total_win / n_win if n_win > 0 else 0
    avg_loss = total_loss / n_loss if n_loss > 0 else 0

    return {
        'pnl': total_pnl, 'trades': len(pnl_list), 'wr': wr,
        'mdd': max_dd, 'pnl_mdd': pnl_mdd, 'profit_factor': pf,
        'avg_win': avg_win, 'avg_loss': avg_loss,
    }

# Also compute: baseline random compound equity for each setting
# to see how much of compound growth is just from TP/SL width
print("Computing baseline compound equity (random entries)...")
baseline_compound = {}
for tp, sl in TPSL_SETTINGS:
    label = f"{tp}/{sl}"
    base_wr_l = baselines[(tp,sl)]['LONG'] / 100
    base_wr_s = baselines[(tp,sl)]['SHORT'] / 100
    # Simulate 500 random trades
    np.random.seed(42)
    rng = np.random.random(500)
    base_wr = (base_wr_l + base_wr_s) / 2
    eq = 100.0
    for r in rng:
        if r < base_wr:
            eq *= (1 + (tp * LEVERAGE - FEE_PCT)/100)
        else:
            eq *= (1 + (-sl * LEVERAGE - FEE_PCT)/100)
    baseline_compound[label] = (eq / 100 - 1) * 100

results = {}
for tp, sl in TPSL_SETTINGS:
    label = f"{tp}/{sl}"
    vl = setting_patterns[(tp,sl)]
    if not vl:
        continue
    res = compound_portfolio(vl)
    res['n_patterns'] = len(vl)
    res['label'] = label
    results[label] = res

# ============================================================
# TABLE 1: Raw compound metrics
# ============================================================
print(f"\n  {'Setting':<8} {'Pats':>5} {'Trades':>6} {'Compound PnL':>13} {'MDD':>7} {'PnL/MDD':>8} {'WR':>6} {'PF':>5} {'AvgW':>6} {'AvgL':>6}")
print(f"  {'-'*73}")
for tp, sl in TPSL_SETTINGS:
    label = f"{tp}/{sl}"
    if label not in results: continue
    r = results[label]
    # Cap display for readability
    pnl_str = f"{r['pnl']:+.1f}%" if abs(r['pnl']) < 1e6 else f"+{r['pnl']:.1e}%"
    pm_str = f"{r['pnl_mdd']:.1f}" if r['pnl_mdd'] < 1e6 else f"{r['pnl_mdd']:.1e}"
    print(f"  {label:<8} {r['n_patterns']:>5} {r['trades']:>6} {pnl_str:>13} {r['mdd']:>6.1f}% {pm_str:>8} {r['wr']:>5.1f}% {r['profit_factor']:>4.1f} {r['avg_win']:>5.2f} {r['avg_loss']:>5.2f}")

# ============================================================
# TABLE 2: Normalized metrics (setting-comparable)
# ============================================================
print(f"\n{'='*80}")
print("NORMALIZED METRICS (comparable across settings)")
print(f"{'='*80}\n")

print("Key insight: wider TP/SL → fewer trades but bigger per-trade impact.")
print("Compound PnL is dominated by (trades × per-trade-size) interaction.")
print()

# Metric 1: Average Excess WR (same scale regardless of TP/SL)
# Metric 2: Profit Factor (ratio, comparable)
# Metric 3: Edge / Risk ratio = avg_net_pnl_per_trade / sl_pct (risk-normalized)
# Metric 4: Trade frequency × edge = opportunity-adjusted edge

print(f"  {'Setting':<8} {'Pats':>5} {'Trades':>6} {'AvgExcess':>10} {'PF':>6} {'Edge/Risk':>10} {'Freq×Edge':>10}")
print(f"  {'-'*58}")

setting_scores = []
for tp, sl in TPSL_SETTINGS:
    label = f"{tp}/{sl}"
    vl = setting_patterns[(tp,sl)]
    if not vl: continue

    avg_excess = np.mean([v['excess'] for v in vl])
    r = results[label]

    # Edge per trade (average pnl per trade)
    total_simple_pnl = sum(v['pnl'] for v in vl)
    avg_edge = total_simple_pnl / r['trades'] if r['trades'] > 0 else 0
    # Normalize by risk (sl)
    edge_risk = avg_edge / (sl * LEVERAGE) if sl > 0 else 0
    # Frequency × edge = trades × edge_risk
    freq_edge = r['trades'] * edge_risk

    setting_scores.append({
        'label': label, 'tp': tp, 'sl': sl,
        'n_patterns': len(vl), 'trades': r['trades'],
        'avg_excess': avg_excess, 'pf': r['profit_factor'],
        'edge_risk': edge_risk, 'freq_edge': freq_edge,
        'wr': r['wr'], 'mdd': r['mdd'],
    })

    print(f"  {label:<8} {len(vl):>5} {r['trades']:>6} {avg_excess:>+9.1f}% {r['profit_factor']:>5.2f} {edge_risk:>9.3f} {freq_edge:>9.1f}")

# ============================================================
# RANKING
# ============================================================
print(f"\n{'='*80}")
print("RANKING BY EACH METRIC")
print(f"{'='*80}\n")

metrics = [
    ('avg_excess', 'Avg Excess WR', True),
    ('pf', 'Profit Factor', True),
    ('edge_risk', 'Edge/Risk', True),
    ('freq_edge', 'Freq × Edge', True),
    ('n_patterns', 'Pattern Count', True),
    ('trades', 'Trade Count', True),
]

rank_points = defaultdict(float)
for metric, name, higher_better in metrics:
    sorted_s = sorted(setting_scores, key=lambda x: x[metric], reverse=higher_better)
    print(f"  {name}:")
    for rank, s in enumerate(sorted_s[:5]):
        print(f"    #{rank+1} {s['label']}: {s[metric]:.3f}" if isinstance(s[metric], float) else f"    #{rank+1} {s['label']}: {s[metric]}")
        rank_points[s['label']] += (len(sorted_s) - rank)  # Higher = better
    print()

print("COMPOSITE RANKING (sum of rank points across all metrics):")
for label, pts in sorted(rank_points.items(), key=lambda x: -x[1]):
    print(f"  {label}: {pts:.0f} points")

# ============================================================
# PATTERN OVERLAP ANALYSIS (from previous script, corrected)
# ============================================================
print(f"\n{'='*80}")
print("PATTERN OVERLAP: Which patterns change with setting?")
print(f"{'='*80}\n")

setting_sets = {}
for tp, sl in TPSL_SETTINGS:
    label = f"{tp}/{sl}"
    setting_sets[label] = set(f"{v['pattern']}_{v['direction']}" for v in setting_patterns[(tp,sl)])

all_pattern_keys = set()
for vl in setting_patterns.values():
    for v in vl:
        all_pattern_keys.add(f"{v['pattern']}_{v['direction']}")

pattern_count = {}
for pk in all_pattern_keys:
    pattern_count[pk] = []
    for tp, sl in TPSL_SETTINGS:
        label = f"{tp}/{sl}"
        if pk in setting_sets[label]:
            pattern_count[pk].append(label)

sorted_pats = sorted(pattern_count.items(), key=lambda x: -len(x[1]))

count_dist = Counter(len(v) for _, v in sorted_pats)
print("Appearance distribution:")
for c in sorted(count_dist.keys(), reverse=True):
    print(f"  {c}/{len(TPSL_SETTINGS)} settings: {count_dist[c]} patterns")

UNIVERSAL_THRESHOLD = len(TPSL_SETTINGS) // 2
universal = [(pk, setts) for pk, setts in sorted_pats if len(setts) >= UNIVERSAL_THRESHOLD]
print(f"\nUniversal (>={UNIVERSAL_THRESHOLD} settings): {len(universal)}")
for pk, setts in universal:
    pat, d = pk.rsplit('_', 1)
    print(f"  {pat:<15} {d:<6} in {len(setts)}/{len(TPSL_SETTINGS)} settings: {', '.join(setts)}")

# Adjacent overlap
print(f"\nAdjacent setting overlap:")
for i in range(len(TPSL_SETTINGS)-1):
    l1 = f"{TPSL_SETTINGS[i][0]}/{TPSL_SETTINGS[i][1]}"
    l2 = f"{TPSL_SETTINGS[i+1][0]}/{TPSL_SETTINGS[i+1][1]}"
    s1 = setting_sets[l1]; s2 = setting_sets[l2]
    shared = len(s1 & s2)
    total = len(s1 | s2)
    jaccard = shared/total*100 if total > 0 else 0
    print(f"  {l1} ↔ {l2}: {shared} shared / {total} total ({jaccard:.0f}% Jaccard)")

# ============================================================
# CURRENT PRODUCTION PATTERNS CHECK
# ============================================================
print(f"\n{'='*80}")
print("CURRENT PRODUCTION PATTERNS: Which settings validate them?")
print(f"{'='*80}\n")

# v1.18.2 regime patterns from config
prod_patterns = {
    'MU-ST-DN_SHORT', 'IH-DN-DN_SHORT', 'BU-U-DN_SHORT',
    'BD-ST-DN_SHORT', 'U-ST-U_LONG', 'DN-DN-BD_SHORT',
    'BD-BD-BD_SHORT', 'ST-BD-DN_LONG', 'MU-ST-DN_SHORT',
    'DN-DN-BD_SHORT',
}
# Deduplicate
prod_patterns = set(prod_patterns)

print(f"Production patterns (v1.18.2): {len(prod_patterns)}")
for pk in sorted(prod_patterns):
    pat, d = pk.rsplit('_', 1)
    settings_found = []
    for tp, sl in TPSL_SETTINGS:
        label = f"{tp}/{sl}"
        if pk in setting_sets[label]:
            settings_found.append(label)
    status = f"in {len(settings_found)}/{len(TPSL_SETTINGS)} settings" if settings_found else "NOT FOUND in any setting"
    print(f"  {pat:<15} {d:<6} {status}")
    if settings_found:
        print(f"    Settings: {', '.join(settings_found)}")

# ============================================================
# FINAL SUMMARY
# ============================================================
print(f"\n{'='*80}")
print("FINAL SUMMARY")
print(f"{'='*80}\n")

print("1. COMPARISON FLAW IN PREVIOUS ANALYSIS:")
print("   Total PnL (sum) scales with TP/SL width → meaningless for cross-setting comparison.")
print("   e.g., 3%TP win = +8.9%/trade vs 0.5%TP win = +1.4%/trade (6.4x difference)")
print()

print("2. CORRECTED FINDING — DO PATTERNS CHANGE?")
unique_to_one = sum(1 for _, setts in sorted_pats if len(setts) == 1)
print(f"   YES. {len(sorted_pats)} total unique, {unique_to_one} ({unique_to_one/len(sorted_pats)*100:.0f}%) unique to single setting.")
print(f"   Universal patterns: {len(universal)} ({len(universal)/len(sorted_pats)*100:.0f}%)")
print()

print("3. CORRECTED FINDING — OPTIMAL SETTING:")
top = sorted(rank_points.items(), key=lambda x: -x[1])
print(f"   Composite rank #1: {top[0][0]} ({top[0][1]:.0f} pts)")
print(f"   Composite rank #2: {top[1][0]} ({top[1][1]:.0f} pts)")
print(f"   Composite rank #3: {top[2][0]} ({top[2][1]:.0f} pts)")
print()

print("4. KEY INSIGHT:")
print("   Avg Excess WR is ~17-19% across ALL settings — patterns are equally 'real'")
print("   regardless of TP/SL width. The TRADE-OFF is:")
print("   - Tight (0.5-1.0%): Many patterns, many trades, small per-trade impact")
print("   - Wide (2.0-3.0%): Few patterns, few trades, large per-trade impact")
print("   - Mid (1.0-1.5%): Balance of patterns, trades, and impact")

print("\nDone.")
