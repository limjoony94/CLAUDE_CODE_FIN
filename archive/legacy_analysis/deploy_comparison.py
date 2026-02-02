"""
배포 후보 비교: v1.18.2 (Regime) vs v1.19.1 (Tight TP/SL) vs Per-Pattern Optimal
=================================================================================
동일 270일 데이터, 동일 백테스트 엔진으로 세 접근법 비교.

A. v1.18.2 Regime-Adaptive: 11 패턴, 레짐별 TP/SL (1.5-3.0%)
B. v1.19.1 Tight TP/SL: 21 패턴, tight TP/SL (0.3-1.0%)
C. Per-Pattern Optimal: 전체 그리드에서 per-pattern 최적 발굴

비교 지표 (모두 compound equity 기반):
  - Compound PnL, MDD, PnL/MDD
  - Walk-Forward (5-fold OOS)
  - Monte Carlo p-value
  - 기간별 안정성 (H1/H2/H3)
"""

import pandas as pd
import numpy as np
from pathlib import Path
from collections import defaultdict, Counter
import sys

# Import canonical classify_candle from production
sys.path.insert(0, str(Path(__file__).parent.parent))
from scripts.production.pattern_5m.indicators import classify_candle

MAX_BARS = 500
FEE_PCT = 0.10
LEVERAGE = 3
MIN_TRADES = 25  # per-pattern optimal 발굴용
MC_SIMS = 10000
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
    'H1_May_Jul': (dates >= np.datetime64('2025-05-01')) & (dates < np.datetime64('2025-08-01')),
    'H2_Aug_Oct': (dates >= np.datetime64('2025-08-01')) & (dates < np.datetime64('2025-11-01')),
    'H3_Nov_Jan': (dates >= np.datetime64('2025-11-01')) & (dates < np.datetime64('2026-02-01')),
}

print("Classifying candles...")
# Pre-compute avg_body_20 (rolling 20-period average of abs body)
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

# Regime detection for v1.18.2
def detect_regime(idx, lookback=100, threshold=1.5):
    start = max(0, idx - lookback)
    if start == idx: return 'SIDEWAYS'
    p_start = closes[start]
    p_end = closes[idx]
    if p_start <= 0: return 'SIDEWAYS'
    change = (p_end - p_start) / p_start * 100
    if change > threshold: return 'BULL'
    elif change < -threshold: return 'BEAR'
    return 'SIDEWAYS'

# ============================================================
# Define the three approaches
# ============================================================

# A: v1.18.2 Regime-Adaptive
regime_patterns = {
    "BULL": {
        "MU-ST-DN": ("SHORT", 3.0, 1.5),
        "IH-DN-DN": ("SHORT", 2.5, 1.5),
        "BU-U-DN": ("SHORT", 1.5, 2.0),
    },
    "BEAR": {
        "BD-ST-DN": ("SHORT", 1.5, 1.5),
        "BU-U-DN": ("SHORT", 3.0, 1.5),
        "U-ST-U": ("LONG", 1.5, 3.0),
        "DN-DN-BD": ("SHORT", 2.0, 2.5),
    },
    "SIDEWAYS": {
        "BD-BD-BD": ("SHORT", 3.0, 2.5),
        "ST-BD-DN": ("LONG", 3.0, 3.0),
        "MU-ST-DN": ("SHORT", 3.0, 2.0),
        "DN-DN-BD": ("SHORT", 1.5, 2.0),
    },
}

# B: v1.19.1 Tight TP/SL
tight_patterns = {
    'MD-H-MD':  ('LONG', 1.0, 1.0),
    'IH-DN-MD': ('LONG', 1.0, 1.0),
    'ST-D-MU':  ('LONG', 1.0, 1.0),
    'MU-MU-BU': ('LONG', 1.0, 1.0),
    'U-BU-DF':  ('LONG', 0.7, 0.7),
    'H-MU-MD':  ('LONG', 0.7, 1.0),
    'MD-DN-H':  ('LONG', 0.7, 1.0),
    'MU-IH-U':  ('LONG', 1.0, 1.0),
    'BU-MD-GS': ('LONG', 1.0, 1.0),
    'DF-MU-U':  ('LONG', 1.0, 1.0),
    'BD-DF-MD': ('SHORT', 0.7, 1.0),
    'MD-BD-MD': ('SHORT', 0.5, 0.3),
    'D-IH-DN':  ('SHORT', 0.3, 0.5),
    'DN-DF-BD': ('SHORT', 0.3, 0.5),
    'U-D-MD':   ('SHORT', 1.0, 1.0),
    'D-BD-H':   ('SHORT', 1.0, 1.0),
    'DN-GS-DN': ('SHORT', 0.7, 1.0),
    'D-BU-D':   ('SHORT', 0.3, 0.5),
    'IH-MU-IH': ('SHORT', 0.3, 0.3),
    'ST-MU-DF': ('SHORT', 0.5, 0.5),
    'DF-BU-MD': ('SHORT', 0.5, 0.3),
}

# ============================================================
# Unified trade collection
# ============================================================

def collect_trades_regime(bar_range=None):
    """v1.18.2: Regime-based trade collection"""
    trades = []  # (bar_idx, pnl_pct)
    start = bar_range[0] if bar_range else 2
    end = bar_range[1] if bar_range else n_bars

    for i in range(max(start, 2), end):
        pat = f"{types[i-2]}-{types[i-1]}-{types[i]}"
        regime = detect_regime(i)
        if regime not in regime_patterns: continue
        if pat not in regime_patterns[regime]: continue
        direction, tp, sl = regime_patterns[regime][pat]
        if i + 1 >= n_bars: continue
        entry = opens[i + 1]
        if entry <= 0: continue
        if direction == 'LONG':
            tpp = entry * (1 + tp/100); slp = entry * (1 - sl/100)
        else:
            tpp = entry * (1 - tp/100); slp = entry * (1 + sl/100)
        for j in range(i+2, min(i+2+MAX_BARS, n_bars)):
            ht = (highs[j]>=tpp) if direction=='LONG' else (lows[j]<=tpp)
            hs = (lows[j]<=slp) if direction=='LONG' else (highs[j]>=slp)
            if ht and hs:
                trades.append((i+1, (tp if abs(tpp-entry)<=abs(slp-entry) else -sl)*LEVERAGE - FEE_PCT))
                break
            elif ht: trades.append((i+1, tp*LEVERAGE - FEE_PCT)); break
            elif hs: trades.append((i+1, -sl*LEVERAGE - FEE_PCT)); break
    trades.sort(key=lambda x: x[0])
    return trades

def collect_trades_tight(bar_range=None):
    """v1.19.1: Tight TP/SL trade collection"""
    trades = []
    start = bar_range[0] if bar_range else 2
    end = bar_range[1] if bar_range else n_bars

    for i in range(max(start, 2), end):
        pat = f"{types[i-2]}-{types[i-1]}-{types[i]}"
        if pat not in tight_patterns: continue
        direction, tp, sl = tight_patterns[pat]
        if i + 1 >= n_bars: continue
        entry = opens[i + 1]
        if entry <= 0: continue
        if direction == 'LONG':
            tpp = entry * (1 + tp/100); slp = entry * (1 - sl/100)
        else:
            tpp = entry * (1 - tp/100); slp = entry * (1 + sl/100)
        for j in range(i+2, min(i+2+MAX_BARS, n_bars)):
            ht = (highs[j]>=tpp) if direction=='LONG' else (lows[j]<=tpp)
            hs = (lows[j]<=slp) if direction=='LONG' else (highs[j]>=slp)
            if ht and hs:
                trades.append((i+1, (tp if abs(tpp-entry)<=abs(slp-entry) else -sl)*LEVERAGE - FEE_PCT))
                break
            elif ht: trades.append((i+1, tp*LEVERAGE - FEE_PCT)); break
            elif hs: trades.append((i+1, -sl*LEVERAGE - FEE_PCT)); break
    trades.sort(key=lambda x: x[0])
    return trades

# ============================================================
# C: Per-Pattern Optimal Discovery (wide grid)
# ============================================================
print("\nDiscovering per-pattern optimal (wide grid)...")

sample_idx = np.arange(2, n_bars, 20)
fixed_grid = [(tp, sl) for tp in [0.3, 0.5, 0.7, 1.0, 1.5, 2.0, 2.5, 3.0]
              for sl in [0.3, 0.5, 0.7, 1.0, 1.5, 2.0, 2.5, 3.0]]

baselines = {}
for tp, sl in fixed_grid:
    pnls_l, pnls_s = [], []
    for idx in sample_idx:
        if idx + 1 >= n_bars: continue
        entry = opens[idx+1]
        if entry <= 0: continue
        for d, plist in [('LONG', pnls_l), ('SHORT', pnls_s)]:
            if d == 'LONG':
                tpp = entry*(1+tp/100); slp = entry*(1-sl/100)
            else:
                tpp = entry*(1-tp/100); slp = entry*(1+sl/100)
            for i in range(idx+2, min(idx+2+MAX_BARS, n_bars)):
                ht = (highs[i]>=tpp) if d=='LONG' else (lows[i]<=tpp)
                hs = (lows[i]<=slp) if d=='LONG' else (highs[i]>=slp)
                if ht and hs:
                    plist.append(1 if abs(tpp-entry)<=abs(slp-entry) else 0); break
                elif ht: plist.append(1); break
                elif hs: plist.append(0); break
    baselines[(tp,sl)] = {
        'LONG': sum(pnls_l)/len(pnls_l)*100 if pnls_l else 50,
        'SHORT': sum(pnls_s)/len(pnls_s)*100 if pnls_s else 50,
    }
print("  Baselines computed.")

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

def mc_test_pnls(pnl_list, n_sims=MC_SIMS):
    if len(pnl_list) < 5: return 1.0
    pa = np.array(pnl_list); real = np.sum(pa)
    signs = np.random.choice([-1,1], size=(n_sims, len(pa)))
    return np.mean(np.sum(pa * signs, axis=1) >= real)

all_pats = [(pat, idx) for pat, idx in pattern_indices.items() if len(idx) >= MIN_TRADES]

per_pattern_configs = {}  # pat -> (direction, tp, sl)

for pat, indices in all_pats:
    for direction in ['LONG', 'SHORT']:
        best_pnl = -999
        best_cfg = None
        for tp, sl in fixed_grid:
            base = baselines.get((tp,sl),{}).get(direction, 50)
            pnls = bt_fixed(indices, direction, tp, sl)
            if len(pnls) < MIN_TRADES: continue
            wr = sum(1 for x in pnls if x>0)/len(pnls)*100
            excess = wr - base
            pnl = sum(pnls)
            if excess < EXCESS_THRESHOLD or pnl <= 0: continue
            if pnl > best_pnl:
                best_pnl = pnl
                best_cfg = (direction, tp, sl, len(pnls), wr, excess, pnl)
        if best_cfg:
            d, tp, sl, nt, wr, exc, pnl = best_cfg
            # Validate: MC
            pnls = bt_fixed(indices, d, tp, sl)
            mc = mc_test_pnls(pnls, 5000)
            if mc >= 0.05: continue
            # WF
            si = np.sort(indices); fs = len(si)//5
            if fs < 3: continue
            wf = 0
            for f in range(5):
                s = f*fs; e = s+fs if f < 4 else len(si)
                fp = bt_fixed(si[s:e], d, tp, sl)
                if fp and sum(fp) > 0: wf += 1
            if wf < 3: continue
            # Period stability
            pp = 0
            for mask in period_masks.values():
                pidx = indices[mask[indices]]
                if len(pidx) < 3: continue
                fp = bt_fixed(pidx, d, tp, sl)
                if fp and sum(fp) > 0: pp += 1
            if pp < 2: continue

            key = f"{pat}_{d}"
            if key not in per_pattern_configs or pnl > per_pattern_configs[key][3]:
                per_pattern_configs[key] = (d, tp, sl, pnl, nt, wr, exc, mc, wf, pp)

# Remove bidirectional
pat_names = [k.rsplit('_',1)[0] for k in per_pattern_configs]
bidir = {p for p, c in Counter(pat_names).items() if c > 1}
per_pattern_configs = {k: v for k, v in per_pattern_configs.items()
                       if k.rsplit('_',1)[0] not in bidir}

print(f"  Per-pattern optimal: {len(per_pattern_configs)} patterns")

# Build per-pattern trade collector
pp_pattern_map = {}
for key, (d, tp, sl, *_) in per_pattern_configs.items():
    pat = key.rsplit('_',1)[0]
    pp_pattern_map[pat] = (d, tp, sl)

def collect_trades_perpattern(bar_range=None):
    trades = []
    start = bar_range[0] if bar_range else 2
    end = bar_range[1] if bar_range else n_bars
    for i in range(max(start, 2), end):
        pat = f"{types[i-2]}-{types[i-1]}-{types[i]}"
        if pat not in pp_pattern_map: continue
        direction, tp, sl = pp_pattern_map[pat]
        if i + 1 >= n_bars: continue
        entry = opens[i + 1]
        if entry <= 0: continue
        if direction == 'LONG':
            tpp = entry * (1 + tp/100); slp = entry * (1 - sl/100)
        else:
            tpp = entry * (1 - tp/100); slp = entry * (1 + sl/100)
        for j in range(i+2, min(i+2+MAX_BARS, n_bars)):
            ht = (highs[j]>=tpp) if direction=='LONG' else (lows[j]<=tpp)
            hs = (lows[j]<=slp) if direction=='LONG' else (highs[j]>=slp)
            if ht and hs:
                trades.append((i+1, (tp if abs(tpp-entry)<=abs(slp-entry) else -sl)*LEVERAGE - FEE_PCT))
                break
            elif ht: trades.append((i+1, tp*LEVERAGE - FEE_PCT)); break
            elif hs: trades.append((i+1, -sl*LEVERAGE - FEE_PCT)); break
    trades.sort(key=lambda x: x[0])
    return trades

# ============================================================
# Evaluation Functions
# ============================================================

def eval_trades(trades):
    """Compute all metrics from trade list"""
    if not trades:
        return {'pnl': 0, 'trades': 0, 'wr': 0, 'mdd': 0, 'pnl_mdd': 0,
                'pf': 0, 'avg_edge': 0, 'max_consec_loss': 0}
    pnl_list = [t[1] for t in trades]
    equity = 100.0; peak = 100.0; max_dd = 0
    wins = 0; total_win = 0; total_loss = 0
    consec_loss = 0; max_cl = 0
    for p in pnl_list:
        equity *= (1 + p/100)
        if equity > peak: peak = equity
        dd = (peak - equity)/peak*100
        if dd > max_dd: max_dd = dd
        if p > 0:
            wins += 1; total_win += p; consec_loss = 0
        else:
            total_loss += abs(p); consec_loss += 1
            if consec_loss > max_cl: max_cl = consec_loss

    total_pnl = (equity/100 - 1)*100
    return {
        'pnl': total_pnl,
        'trades': len(pnl_list),
        'wr': wins/len(pnl_list)*100,
        'mdd': max_dd,
        'pnl_mdd': total_pnl/max_dd if max_dd > 0 else 999,
        'pf': total_win/total_loss if total_loss > 0 else 999,
        'avg_edge': np.mean(pnl_list),
        'max_consec_loss': max_cl,
    }

def walk_forward(collect_fn, n_folds=5):
    """Walk-forward: split by bar index, count profitable folds"""
    fold_size = n_bars // n_folds
    results = []
    for f in range(n_folds):
        s = f * fold_size
        e = s + fold_size if f < n_folds - 1 else n_bars
        trades = collect_fn(bar_range=(s, e))
        res = eval_trades(trades)
        results.append(res)
    profitable = sum(1 for r in results if r['pnl'] > 0)
    return profitable, results

def monte_carlo(trades, n_sims=MC_SIMS):
    """MC test on compound equity"""
    if len(trades) < 5: return 1.0
    pnl_list = np.array([t[1] for t in trades])
    # Real compound PnL
    real_eq = np.prod(1 + pnl_list/100)
    count = 0
    for _ in range(n_sims):
        signs = np.random.choice([-1,1], len(pnl_list))
        shuffled = pnl_list * signs
        eq = np.prod(1 + shuffled/100)
        if eq >= real_eq: count += 1
    return count / n_sims

def period_analysis(collect_fn):
    """Per-period performance"""
    results = {}
    for name, mask in period_masks.items():
        bar_indices = np.where(mask)[0]
        if len(bar_indices) == 0: continue
        trades = collect_fn(bar_range=(bar_indices[0], bar_indices[-1]+1))
        results[name] = eval_trades(trades)
    return results

# ============================================================
# Run all evaluations
# ============================================================
approaches = {
    'A: v1.18.2 Regime': collect_trades_regime,
    'B: v1.19.1 Tight': collect_trades_tight,
    'C: Per-Pattern Opt': collect_trades_perpattern,
}

print(f"\n{'='*80}")
print("FULL-PERIOD COMPARISON (270 days)")
print(f"{'='*80}\n")

full_results = {}
for name, fn in approaches.items():
    trades = fn()
    res = eval_trades(trades)
    mc = monte_carlo(trades)
    wf_p, wf_folds = walk_forward(fn)
    periods = period_analysis(fn)
    full_results[name] = {
        'full': res, 'mc': mc, 'wf': wf_p, 'wf_folds': wf_folds,
        'periods': periods, 'trades_raw': trades,
    }
    n_long = sum(1 for _ in []) # placeholder
    print(f"--- {name} ---")

# Count patterns per approach
n_pats = {
    'A: v1.18.2 Regime': sum(len(v) for v in regime_patterns.values()),
    'B: v1.19.1 Tight': len(tight_patterns),
    'C: Per-Pattern Opt': len(per_pattern_configs),
}

# Print comparison table
print(f"\n{'Metric':<22}", end='')
for name in approaches:
    print(f" {name:>22}", end='')
print()
print(f"{'─'*90}")

metrics_to_show = [
    ('Patterns', lambda n: f"{n_pats[n]}"),
    ('Trades', lambda n: f"{full_results[n]['full']['trades']}"),
    ('Win Rate', lambda n: f"{full_results[n]['full']['wr']:.1f}%"),
    ('Compound PnL', lambda n: f"{full_results[n]['full']['pnl']:+.1f}%" if abs(full_results[n]['full']['pnl']) < 1e6 else f"+{full_results[n]['full']['pnl']:.1e}%"),
    ('MDD', lambda n: f"{full_results[n]['full']['mdd']:.1f}%"),
    ('PnL/MDD', lambda n: f"{full_results[n]['full']['pnl_mdd']:.1f}" if full_results[n]['full']['pnl_mdd'] < 1e6 else f"{full_results[n]['full']['pnl_mdd']:.1e}"),
    ('Profit Factor', lambda n: f"{full_results[n]['full']['pf']:.2f}"),
    ('Avg Edge/Trade', lambda n: f"{full_results[n]['full']['avg_edge']:+.3f}%"),
    ('Max Consec Loss', lambda n: f"{full_results[n]['full']['max_consec_loss']}"),
    ('MC p-value', lambda n: f"{full_results[n]['mc']:.4f}"),
    ('Walk-Forward', lambda n: f"{full_results[n]['wf']}/5"),
]

for label, fn in metrics_to_show:
    print(f"{label:<22}", end='')
    for name in approaches:
        print(f" {fn(name):>22}", end='')
    print()

# Walk-forward detail
print(f"\n{'='*80}")
print("WALK-FORWARD DETAIL (5 folds)")
print(f"{'='*80}\n")

for name in approaches:
    folds = full_results[name]['wf_folds']
    print(f"--- {name} ---")
    for i, f in enumerate(folds):
        status = "✅" if f['pnl'] > 0 else "❌"
        print(f"  Fold {i+1}: {f['trades']:>4} trades, {f['wr']:>5.1f}% WR, {f['pnl']:>+10.1f}% PnL, {f['mdd']:>5.1f}% MDD {status}")
    print(f"  → {full_results[name]['wf']}/5 profitable\n")

# Period detail
print(f"{'='*80}")
print("PERIOD DETAIL (H1=Bull, H2=Mixed, H3=Bear)")
print(f"{'='*80}\n")

for name in approaches:
    periods = full_results[name]['periods']
    print(f"--- {name} ---")
    for pname, r in periods.items():
        status = "✅" if r['pnl'] > 0 else "❌"
        print(f"  {pname}: {r['trades']:>4}t, {r['wr']:>5.1f}% WR, {r['pnl']:>+10.1f}% PnL, {r['mdd']:>5.1f}% MDD {status}")
    profitable = sum(1 for r in periods.values() if r['pnl'] > 0)
    print(f"  → {profitable}/3 periods profitable\n")

# ============================================================
# OVERFITTING RISK ASSESSMENT
# ============================================================
print(f"{'='*80}")
print("OVERFITTING RISK ASSESSMENT")
print(f"{'='*80}\n")

print("Degrees of freedom (pattern selection + TP/SL optimization):")
for name in approaches:
    if 'Regime' in name:
        dof = "Low: 8 unique patterns × regime-specific TP/SL, regime detection adds structure"
    elif 'Tight' in name:
        dof = f"Medium: 21 patterns × per-pattern TP/SL from tight grid (0.3-1.0%)"
    else:
        dof = f"High: {len(per_pattern_configs)} patterns × per-pattern TP/SL from FULL grid (0.3-3.0%)"
    print(f"  {name}: {dof}")

print()
print("In-sample optimization surface:")
print(f"  A (Regime):      8 patterns × 3 regimes × ~10 TP/SL combos = ~240 searches")
print(f"  B (Tight):      21 patterns from ~20 TP/SL tight combos × 1728 patterns = ~34k searches")
print(f"  C (Per-Pat Opt): {len(per_pattern_configs)} patterns from 64 TP/SL combos × 1728 patterns = ~110k searches")
print()
print("Larger search space → higher overfitting risk → need stronger OOS validation")

# ============================================================
# FINAL VERDICT
# ============================================================
print(f"\n{'='*80}")
print("DEPLOYMENT RECOMMENDATION")
print(f"{'='*80}\n")

# Score each approach
scores = {}
for name in approaches:
    r = full_results[name]
    score = 0
    # WF (most important for OOS)
    score += r['wf'] * 20  # 0-100
    # MC significance
    score += (50 if r['mc'] < 0.001 else 30 if r['mc'] < 0.01 else 10 if r['mc'] < 0.05 else 0)
    # Period stability
    p_profit = sum(1 for pr in r['periods'].values() if pr['pnl'] > 0)
    score += p_profit * 15  # 0-45
    # Overfitting penalty
    if 'Per-Pattern' in name:
        score -= 30  # Large search space penalty
    elif 'Tight' in name:
        score -= 10  # Medium penalty
    # Trade frequency bonus (more trades = more data = more reliable)
    trades = r['full']['trades']
    score += min(trades / 10, 30)  # Cap at 30
    # PnL/MDD bonus (capped)
    pm = r['full']['pnl_mdd']
    score += min(pm / 10, 20) if pm < 1e6 else 20

    scores[name] = score

print("Scoring (higher = better deployment candidate):")
for name in sorted(scores, key=lambda x: -scores[x]):
    print(f"  {name}: {scores[name]:.0f} pts")

winner = max(scores, key=lambda x: scores[x])
print(f"\n  ★ RECOMMENDED: {winner}")

# Explain reasoning
print(f"\nReasoning:")
for name in sorted(scores, key=lambda x: -scores[x]):
    r = full_results[name]
    pp = sum(1 for pr in r['periods'].values() if pr['pnl'] > 0)
    print(f"  {name}:")
    print(f"    WF={r['wf']}/5, MC={r['mc']:.4f}, Periods={pp}/3, Trades={r['full']['trades']}")

print("\nDone.")
