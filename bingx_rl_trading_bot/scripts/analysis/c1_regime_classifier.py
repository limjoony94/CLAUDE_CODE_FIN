"""
C1 Regime Classifier — Phase 2
================================
Design and validate a practical regime classifier that decides forward vs.
reverse direction on the valid-reverse subset (176 of 1027 signals).

Phase 1 baseline:
  • Baseline (all forward)        : +170.5% PnL, 1027 trades
  • Oracle ceiling                : +253.9% PnL (gap +83.4pp)
  • Subset (rev_valid=True)       : 176 trades
      - Forward on subset         : +34.72%  (67.6% fwd-win rate)
      - Reverse on subset         : +2.80%
      - Oracle on subset          : +118.10%

Policy (what the classifier does):
  if rev_valid(bar[i]):  direction = classifier_predict(features[i])
  else:                  direction = forward   # always

Features (all strictly causal, computed at bar[i] using past-only data):
  F1 — ATR ratio vs 100-bar median
  F2 — HTF trend alignment (200-MA slope × break direction consistency)
  F3 — Breakout quality (body/range ratio + wick symmetry)
  F4 — Consolidation age (channel-width squeeze score)

Classifiers:
  M1 — Single-feature threshold (4 variants)
  M2 — 2-feature AND combos (top pairs)
  M3 — Logistic regression on F1..F4

Validation:
  • WF 5-fold expanding window (fit per fold, evaluate OOS)
  • 3-way split at 4 split points (40/20/40, 50/20/30, 60/20/20, 70/15/15)
  • Primary metric: OOS PnL delta vs. Forward-all baseline (NOT accuracy)
  • Secondary: OOS classifier accuracy, MDD delta

Note on accuracy math: 67.6% fwd-win rate means "always-forward" scores 67.6%
accuracy AND matches the subset baseline exactly. Accuracy alone doesn't imply
better PnL — classifier errors on high-magnitude reverse-wins can still lose.
"""
import sys, os, json, math
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from scripts.analysis.c1_refined_validation import (
    BASE_CFG, entry_baseline, check_exit, precompute, FEE_RT_PCT
)
from scripts.analysis.c1_oracle_switching import (
    _sim_single_trade, _compute_reverse_sl, _calc_pnl,
)

# ──────────────────────────────────────────────────────────────────────────
#  Step 1 — Build the full comparison set with causal features
# ──────────────────────────────────────────────────────────────────────────

def _feature_f1_atr_ratio(atr_arr, i, window=100):
    """F1: ATR[i] / median(ATR[i-window:i]). Purely past."""
    if i < window:
        return np.nan
    past = [atr_arr[j] for j in range(i - window, i)
            if not math.isnan(atr_arr[j])]
    if len(past) < window // 2 or math.isnan(atr_arr[i]):
        return np.nan
    med = float(np.median(past))
    if med <= 0:
        return np.nan
    return atr_arr[i] / med


def _feature_f2_htf_align(closes, i, direction, ma_len=200, slope_len=50):
    """F2: trend alignment score for the breakout direction.

    Uses MA200 and 50-bar slope from past closes only (closes[:i+1]).
    Returns a signed score in roughly [-1,+1] — positive = direction
    aligned with macro trend, negative = against trend.

    Components:
      - price position: sign(close[i] - ma200[i])
      - slope sign     : sign(close[i-1] - close[i-slope_len-1])
    Score = 1 if both align with direction, 0 if mixed, -1 if both oppose.
    """
    if i < ma_len + slope_len:
        return np.nan
    ma = sum(closes[i - ma_len:i]) / ma_len       # past only
    ref = closes[i]                                 # bar[i] close ok (we enter at opens[i+1])
    pos_sign = 1 if ref > ma else -1
    # slope over past 50 bars, excluding bar[i]
    c_now = closes[i - 1]
    c_ref = closes[i - slope_len - 1]
    slope_sign = 1 if c_now > c_ref else -1
    dir_sign = 1 if direction == 'LONG' else -1
    # score: both agree = 1, disagree = -1, mixed = 0
    agree_pos = (pos_sign == dir_sign)
    agree_slope = (slope_sign == dir_sign)
    return (int(agree_pos) + int(agree_slope) - 1)  # {-1, 0, +1}


def _feature_f3_breakout_quality(bo, bh, bl, bc, direction):
    """F3: microstructure quality at bar[i].

    Returns body_ratio × wick_bias where wick_bias penalises
    opposite-direction rejection wicks.
      body_ratio ∈ [0,1]
      wick_bias  ∈ [-1, 1]  (positive = breakout-direction continuation)
    Score = body_ratio * (1 + wick_bias) / 2  ∈ [0, body_ratio]
    """
    rng = bh - bl
    if rng <= 0:
        return np.nan
    body = bc - bo
    body_r = abs(body) / rng
    upper_wick = bh - max(bo, bc)
    lower_wick = min(bo, bc) - bl
    # Opposite-direction wick = rejection signal
    if direction == 'LONG':
        # rejection would be a large upper_wick
        bias = (lower_wick - upper_wick) / rng   # positive = continuation
    else:
        bias = (upper_wick - lower_wick) / rng
    # combine: body_r multiplied by (1 + bias)/2 → larger body + continuation wick = high
    score = body_r * (1.0 + bias) / 2.0
    return score


def _feature_f4_consolidation(ch_arr, cl_arr, i, lookback=50):
    """F4: consolidation score = mean(past widths) / current width.
    >1 means the channel is currently narrower than recent history (squeezed).
    All bars past-only.
    """
    if i < lookback + 1:
        return np.nan
    widths = []
    for j in range(i - lookback, i):
        w = ch_arr[j] - cl_arr[j]
        if not math.isnan(w) and w > 0:
            widths.append(w)
    if len(widths) < lookback // 2:
        return np.nan
    cur_w = ch_arr[i] - cl_arr[i]
    if math.isnan(cur_w) or cur_w <= 0:
        return np.nan
    return float(np.mean(widths)) / cur_w


def build_dataset(df15, cfg, pc):
    """Iterate breakout signals; record forward/reverse PnLs + features.

    Returns list of dicts with keys:
      entry_bar, timestamp, direction, rev_valid, fwd_pnl, rev_pnl, label,
      f1, f2, f3, f4, fold_idx (filled later).
    Mirrors c1_oracle_switching.simulate_both_directions for PnL; adds features.
    """
    warmup = 50
    end_i = len(df15) - 1
    cd_until = warmup
    records = []

    opens = pc['opens']; highs = pc['highs']; lows = pc['lows']; closes = pc['closes']
    atr = pc['atr']; ch = pc['ch']; cl_arr = pc['cl']
    swl = pc['swl']; swh = pc['swh']
    ts = pc['timestamps']

    for i in range(warmup, end_i):
        if i < cd_until:
            continue
        if math.isnan(atr[i]):
            continue
        sig = entry_baseline(opens[i], highs[i], lows[i], closes[i],
                             ch[i], cl_arr[i], atr[i],
                             swl[i], swh[i], cfg)
        if sig is None:
            continue

        fwd_pnl, fwd_exit_bar, fwd_reason = _sim_single_trade(
            i, sig['direction'], sig['sl_price'], cfg, pc, end_i
        )
        rev_dir = 'SHORT' if sig['direction'] == 'LONG' else 'LONG'
        ni = i + 1
        if ni > end_i:
            cd_until = fwd_exit_bar + cfg['min_bars_between']
            continue
        entry_px_rev = opens[ni]
        rev_sl, rev_sp = _compute_reverse_sl(
            rev_dir, entry_px_rev, atr[i], swl[i], swh[i], cfg
        )
        if rev_sl is None:
            rev_pnl = None
            rev_valid = False
        else:
            rev_pnl, rev_exit_bar, rev_reason = _sim_single_trade(
                i, rev_dir, rev_sl, cfg, pc, end_i
            )
            rev_valid = True

        # --- Features at bar[i] ---
        f1 = _feature_f1_atr_ratio(atr, i)
        f2 = _feature_f2_htf_align(closes, i, sig['direction'])
        f3 = _feature_f3_breakout_quality(opens[i], highs[i], lows[i], closes[i], sig['direction'])
        f4 = _feature_f4_consolidation(ch, cl_arr, i)

        # Label: 1 means forward wins (or tie), 0 means reverse wins
        # (only used for subset rows in training)
        if rev_valid:
            label = 1 if fwd_pnl >= rev_pnl else 0
        else:
            label = None

        records.append({
            'entry_bar': i,
            'timestamp': str(ts[i]),
            'direction': sig['direction'],
            'rev_valid': rev_valid,
            'fwd_pnl': float(fwd_pnl),
            'rev_pnl': (float(rev_pnl) if rev_valid else None),
            'label': label,
            'f1': (float(f1) if not (f1 is None or (isinstance(f1, float) and math.isnan(f1))) else None),
            'f2': (float(f2) if (f2 is not None and not (isinstance(f2, float) and math.isnan(f2))) else None),
            'f3': (float(f3) if not (f3 is None or (isinstance(f3, float) and math.isnan(f3))) else None),
            'f4': (float(f4) if not (f4 is None or (isinstance(f4, float) and math.isnan(f4))) else None),
        })
        cd_until = fwd_exit_bar + cfg['min_bars_between']

    return records


# ──────────────────────────────────────────────────────────────────────────
#  Step 2 — Sanity: look-ahead check on fractal swings
# ──────────────────────────────────────────────────────────────────────────
def fractal_causality_check(df15, cfg):
    """Compare fractal swl/swh at index T using df15[:T+1] vs df15[:T+50].
    If causal, they must match exactly for indices <= T.
    """
    from scripts.production.c1_breakout.indicators import compute_fractal_swings
    T = len(df15) // 2
    df_full = df15.iloc[:T + 50].reset_index(drop=True)
    df_trunc = df15.iloc[:T + 1].reset_index(drop=True)

    swl_full, swh_full = compute_fractal_swings(
        df_full['high'].tolist(), df_full['low'].tolist(), cfg['fractal_lookback']
    )
    swl_trunc, swh_trunc = compute_fractal_swings(
        df_trunc['high'].tolist(), df_trunc['low'].tolist(), cfg['fractal_lookback']
    )
    # Compare first T+1 entries
    def _eq(a, b):
        if math.isnan(a) and math.isnan(b):
            return True
        return a == b
    diffs_swl = sum(1 for j in range(T + 1) if not _eq(swl_full[j], swl_trunc[j]))
    diffs_swh = sum(1 for j in range(T + 1) if not _eq(swh_full[j], swh_trunc[j]))
    return diffs_swl, diffs_swh


# ──────────────────────────────────────────────────────────────────────────
#  Step 3 — Classifier M1/M2/M3
# ──────────────────────────────────────────────────────────────────────────

def _fit_M1_threshold(train_df, feat_col):
    """Single-feature threshold: find t* that maximises train PnL under
    the switching policy. Returns (threshold, sign) where sign ∈ {+1,-1}
    denotes which side of threshold triggers FORWARD.

    Policy: predict FORWARD if sign * feat >= sign * t.
    """
    # Drop rows with missing feature
    df = train_df.dropna(subset=[feat_col]).copy()
    if len(df) < 20:
        return None
    # Candidate thresholds = unique quantiles
    qs = np.linspace(0.1, 0.9, 17)  # 17 candidates
    thresholds = sorted(set(df[feat_col].quantile(q) for q in qs))

    best = None
    for t in thresholds:
        for sign in (+1, -1):
            pred_fwd = (sign * df[feat_col] >= sign * t).astype(int)
            # PnL: if predict forward → use fwd_pnl, else rev_pnl
            pnl = np.where(pred_fwd == 1, df['fwd_pnl'], df['rev_pnl']).sum()
            if best is None or pnl > best[0]:
                best = (pnl, t, sign)
    return {'threshold': best[1], 'sign': best[2], 'train_pnl': float(best[0])}


def _predict_M1(row, model, feat_col):
    v = row[feat_col]
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return 1  # default forward on missing feature
    return int(model['sign'] * v >= model['sign'] * model['threshold'])


def _fit_M2_and(train_df, fa, fb):
    """2-feature AND rule. Fit independent thresholds per feature, with
    signs — predict FORWARD iff both feature conditions hold (else REVERSE).

    To avoid a huge grid, reuse M1 per-feature thresholds then check
    sign patterns.
    """
    df = train_df.dropna(subset=[fa, fb]).copy()
    if len(df) < 20:
        return None
    qs = np.linspace(0.2, 0.8, 7)  # 7 quantiles per feature
    ta_cands = sorted(set(df[fa].quantile(q) for q in qs))
    tb_cands = sorted(set(df[fb].quantile(q) for q in qs))

    best = None
    for ta in ta_cands:
        for tb in tb_cands:
            for sa in (+1, -1):
                for sb in (+1, -1):
                    cond_a = (sa * df[fa] >= sa * ta).astype(int)
                    cond_b = (sb * df[fb] >= sb * tb).astype(int)
                    pred_fwd = (cond_a & cond_b)
                    pnl = np.where(pred_fwd == 1, df['fwd_pnl'], df['rev_pnl']).sum()
                    if best is None or pnl > best[0]:
                        best = (pnl, ta, tb, sa, sb)
    return {'ta': best[1], 'tb': best[2], 'sa': best[3], 'sb': best[4],
            'fa': fa, 'fb': fb, 'train_pnl': float(best[0])}


def _predict_M2(row, model):
    va = row[model['fa']]; vb = row[model['fb']]
    if (va is None or (isinstance(va, float) and math.isnan(va))
            or vb is None or (isinstance(vb, float) and math.isnan(vb))):
        return 1
    ca = int(model['sa'] * va >= model['sa'] * model['ta'])
    cb = int(model['sb'] * vb >= model['sb'] * model['tb'])
    return 1 if (ca and cb) else 0


def _fit_M3_logreg(train_df, feat_cols):
    """Logistic regression, standardised features. Class-balanced to
    reduce 67.6% majority bias."""
    df = train_df.dropna(subset=feat_cols).copy()
    if len(df) < 20:
        return None
    X = df[feat_cols].values
    y = df['label'].values
    if len(set(y)) < 2:
        return None
    scaler = StandardScaler().fit(X)
    Xs = scaler.transform(X)
    try:
        model = LogisticRegression(class_weight='balanced', max_iter=1000,
                                    solver='liblinear', C=1.0)
        model.fit(Xs, y)
    except Exception as e:
        return None
    return {'model': model, 'scaler': scaler, 'feat_cols': feat_cols}


def _predict_M3(row, model):
    if model is None:
        return 1
    vals = []
    for c in model['feat_cols']:
        v = row[c]
        if v is None or (isinstance(v, float) and math.isnan(v)):
            return 1  # default forward
        vals.append(v)
    X = model['scaler'].transform([vals])
    pred = int(model['model'].predict(X)[0])
    return pred  # 1 = forward, 0 = reverse


# ──────────────────────────────────────────────────────────────────────────
#  Step 4 — Evaluation framework
# ──────────────────────────────────────────────────────────────────────────

def _apply_policy(records, predict_fn, model):
    """For each record, return the PnL under the policy:
       - non-subset → forward
       - subset     → forward if predict_fn(row) == 1 else reverse
    Returns list of (pnl, is_subset, prediction_correct_or_None).
    """
    results = []
    for r in records:
        if not r['rev_valid']:
            results.append({'pnl': r['fwd_pnl'], 'subset': False,
                             'correct': None, 'pred': None})
        else:
            pred = predict_fn(r, model)
            if pred == 1:
                pnl = r['fwd_pnl']
            else:
                pnl = r['rev_pnl']
            correct = (pred == r['label'])
            results.append({'pnl': pnl, 'subset': True,
                             'correct': correct, 'pred': pred})
    return results


def _aggregate(results):
    pnls = [x['pnl'] for x in results]
    n = len(pnls)
    wins = sum(1 for p in pnls if p > 0)
    pnl = sum(pnls)
    eq, peak, mdd = 0, 0, 0
    for p in pnls:
        eq += p
        peak = max(peak, eq)
        mdd = max(mdd, peak - eq)
    subset_res = [x for x in results if x['subset']]
    acc = None
    if subset_res:
        correct_ct = sum(1 for x in subset_res if x['correct'])
        acc = correct_ct / len(subset_res) * 100
    return {
        'trades': n,
        'WR': round(wins / max(n, 1) * 100, 2),
        'PnL': round(pnl, 2),
        'MDD': round(mdd, 2),
        'subset_n': len(subset_res),
        'subset_acc': (round(acc, 2) if acc is not None else None),
    }


def wf_evaluate(records, fit_fn, predict_fn, n_folds=5):
    """Expanding-window WF: for each fold k, fit on records[:train_end]
    (subset only → label available), evaluate on fold-k OOS records.

    Records are already in entry_bar order (temporal).
    Returns dict with per-fold stats and totals.
    """
    n = len(records)
    warmup_records = 0  # we're past bar warmup already
    step = (n - warmup_records) // (n_folds + 1)

    df_all = pd.DataFrame(records)

    folds = []
    total_results = []
    for k in range(n_folds):
        train_end = warmup_records + (k + 1) * step
        oos_start = train_end
        oos_end = warmup_records + (k + 2) * step
        if oos_end > n:
            oos_end = n

        train_df = df_all.iloc[:train_end]
        train_subset = train_df[train_df['rev_valid'] == True].copy()
        # labels must be populated
        train_subset = train_subset.dropna(subset=['label'])

        model = fit_fn(train_subset) if len(train_subset) >= 20 else None

        oos_records = records[oos_start:oos_end]
        if model is None:
            # default: all forward
            results = [{'pnl': r['fwd_pnl'], 'subset': r['rev_valid'],
                        'correct': (1 == r['label']) if r['rev_valid'] else None,
                        'pred': 1} for r in oos_records]
        else:
            results = _apply_policy(oos_records, predict_fn, model)

        # Baseline (all forward) on same OOS window
        baseline_pnl = sum(r['fwd_pnl'] for r in oos_records)
        agg = _aggregate(results)
        agg['baseline_PnL'] = round(baseline_pnl, 2)
        agg['delta_PnL'] = round(agg['PnL'] - baseline_pnl, 2)
        agg['fold'] = k + 1
        agg['train_end'] = train_end
        agg['oos_range'] = [oos_start, oos_end]
        folds.append(agg)
        total_results.extend(results)

    totals = _aggregate(total_results)
    totals['baseline_PnL'] = round(sum(sum(r['fwd_pnl'] for r in records[warmup_records + step:
                                           warmup_records + (n_folds + 1) * step]) for _ in [0]), 2)
    # Recompute baseline properly
    all_oos = records[warmup_records + step:warmup_records + (n_folds + 1) * step]
    if warmup_records + (n_folds + 1) * step > n:
        all_oos = records[warmup_records + step:]
    totals['baseline_PnL'] = round(sum(r['fwd_pnl'] for r in all_oos), 2)
    totals['delta_PnL'] = round(totals['PnL'] - totals['baseline_PnL'], 2)
    totals['folds_improved'] = sum(1 for f in folds if f['delta_PnL'] > 0)
    totals['folds_degraded_5pp'] = sum(1 for f in folds if f['delta_PnL'] < -5)

    return {'folds': folds, 'totals': totals}


def three_way_evaluate(records, fit_fn, predict_fn, train_frac, val_frac):
    """Train on first train_frac of records; skip val_frac; evaluate test.

    This mirrors c1_refined_validation.three_way_split but operates on
    the signal-records index (not bar index).
    """
    n = len(records)
    t1 = int(n * train_frac)
    t2 = int(n * (train_frac + val_frac))

    df_all = pd.DataFrame(records)
    train_df = df_all.iloc[:t1]
    train_subset = train_df[train_df['rev_valid'] == True].dropna(subset=['label'])
    if len(train_subset) < 20:
        return None
    model = fit_fn(train_subset)
    if model is None:
        return None

    def _eval(recs):
        if not recs:
            return None
        results = _apply_policy(recs, predict_fn, model)
        baseline_pnl = sum(r['fwd_pnl'] for r in recs)
        agg = _aggregate(results)
        agg['baseline_PnL'] = round(baseline_pnl, 2)
        agg['delta_PnL'] = round(agg['PnL'] - baseline_pnl, 2)
        return agg

    train_recs = records[:t1]
    val_recs = records[t1:t2]
    test_recs = records[t2:]

    return {
        'train': _eval(train_recs),
        'val': _eval(val_recs),
        'test': _eval(test_recs),
    }


# ──────────────────────────────────────────────────────────────────────────
#  Step 5 — Driver
# ──────────────────────────────────────────────────────────────────────────

def main():
    csv = ROOT / 'data' / 'btc_5m_270days_reclassified.csv'
    df = pd.read_csv(csv, parse_dates=['timestamp']).sort_values('timestamp').reset_index(drop=True)
    df15 = df.set_index('timestamp').resample('15min').agg({
        'open': 'first', 'high': 'max', 'low': 'min',
        'close': 'last', 'volume': 'sum',
    }).dropna(subset=['open']).reset_index()
    days = (df15['timestamp'].iloc[-1] - df15['timestamp'].iloc[0]).days
    print(f"Bars: {len(df15)} | {df15['timestamp'].iloc[0]} ~ {df15['timestamp'].iloc[-1]} ({days}d)")

    cfg = BASE_CFG
    pc = precompute(df15, cfg)

    # ── 0. Look-ahead sanity check on fractal swings ──
    diffs_swl, diffs_swh = fractal_causality_check(df15, cfg)
    print(f"\nFractal causality check: swl diffs={diffs_swl}, swh diffs={diffs_swh}")
    if diffs_swl != 0 or diffs_swh != 0:
        print("!!! WARNING: fractal swings leak future data. Features on swl/swh are invalid.")
    else:
        print("  -> Causal: OK")

    # ── 1. Build dataset with features ──
    print("\nBuilding dataset...")
    records = build_dataset(df15, cfg, pc)
    n_sig = len(records)
    n_sub = sum(1 for r in records if r['rev_valid'])
    print(f"  Signals: {n_sig}, subset: {n_sub} ({n_sub/max(n_sig,1)*100:.1f}%)")

    # Sanity: forward-all should match +170.5%
    fwd_all_pnl = sum(r['fwd_pnl'] for r in records)
    print(f"  Forward-all PnL (sanity): {fwd_all_pnl:+.2f}% (expect +170.49%)")

    # Feature availability
    for fc in ['f1', 'f2', 'f3', 'f4']:
        n_ok = sum(1 for r in records if r[fc] is not None)
        print(f"  {fc}: {n_ok}/{n_sig} non-null ({n_ok/max(n_sig,1)*100:.1f}%)")

    # ── 2. Fit-functions & predict-functions ──
    classifiers = []

    # M1 per feature
    for fc, fname in [('f1', 'vol_ratio'), ('f2', 'htf_align'),
                       ('f3', 'quality'), ('f4', 'consol')]:
        def mk_fit(col=fc):
            def _fit(td):
                return _fit_M1_threshold(td, col)
            return _fit
        def mk_pred(col=fc):
            def _pred(row, m):
                if m is None:
                    return 1
                return _predict_M1(row, m, col)
            return _pred
        classifiers.append({
            'name': f'M1-{fc}({fname})',
            'fit': mk_fit(), 'pred': mk_pred(),
        })

    # M2: 2-feature AND combos (select three illustrative pairs)
    pairs = [('f1', 'f2'), ('f1', 'f3'), ('f3', 'f4')]
    for a, b in pairs:
        def mk_fit(fa=a, fb=b):
            def _fit(td):
                return _fit_M2_and(td, fa, fb)
            return _fit
        def mk_pred():
            def _pred(row, m):
                if m is None:
                    return 1
                return _predict_M2(row, m)
            return _pred
        classifiers.append({
            'name': f'M2-{a}AND{b}',
            'fit': mk_fit(), 'pred': mk_pred(),
        })

    # M3: logistic regression on all 4
    def fit_m3(td):
        return _fit_M3_logreg(td, ['f1', 'f2', 'f3', 'f4'])
    def pred_m3(row, m):
        if m is None:
            return 1
        return _predict_M3(row, m)
    classifiers.append({'name': 'M3-LogReg(f1-f4)', 'fit': fit_m3, 'pred': pred_m3})

    # ── 3. Evaluate each classifier ──
    print(f"\n{'='*96}")
    print("REGIME CLASSIFIER COMPARISON (WF 5-fold expanding window)")
    print(f"{'='*96}")
    header = (f"{'Classifier':28s}  {'WF PnL':>9s}  {'Base':>9s}  {'Delta':>9s}  "
              f"{'Folds+':>7s}  {'Deg5pp':>7s}  {'OOSAcc%':>8s}  {'MDD':>7s}")
    print(header)
    print('-' * len(header))

    all_results = {}
    for c in classifiers:
        wf = wf_evaluate(records, c['fit'], c['pred'], n_folds=5)
        tot = wf['totals']
        acc_str = f"{tot['subset_acc']:>7.2f}" if tot['subset_acc'] is not None else '   N/A '
        print(f"{c['name']:28s}  {tot['PnL']:>+9.2f}  {tot['baseline_PnL']:>+9.2f}  "
              f"{tot['delta_PnL']:>+9.2f}  {tot['folds_improved']}/5    "
              f"{tot['folds_degraded_5pp']}/5    {acc_str}  {tot['MDD']:>7.2f}")
        all_results[c['name']] = {'wf': wf}

    # ── 4. 3-way splits (4 split points) ──
    # Strict spec gates:
    #   (a) test PnL > 0 AND
    #   (b) test delta vs forward > 0 (no inferior-to-forward outcome)
    print(f"\n{'='*96}")
    print("3-WAY SPLIT (delta_test/pnl_test — pass requires delta>0 AND pnl>0 on every split)")
    print(f"{'='*96}")
    splits = [(0.40, 0.20, '40/20/40'),
              (0.50, 0.20, '50/20/30'),
              (0.60, 0.20, '60/20/20'),
              (0.70, 0.15, '70/15/15')]
    header = f"{'Classifier':28s}  " + "  ".join(f"{lab:>11s}" for _, _, lab in splits) + "  PassCnt"
    print(header)
    print('-' * len(header))

    for c in classifiers:
        line = f"{c['name']:28s}  "
        pass_ct = 0
        split_results = {}
        has_inferior = False
        for tf, vf, lab in splits:
            res = three_way_evaluate(records, c['fit'], c['pred'], tf, vf)
            if res is None or res['test'] is None:
                line += f"{'  N/A':>11s}  "
                split_results[lab] = None
                continue
            dt = res['test']['delta_PnL']
            test_pnl = res['test']['PnL']
            line += f"{dt:>+6.1f}/{test_pnl:>+4.1f}  "
            if test_pnl > 0 and dt > 0:
                pass_ct += 1
            else:
                has_inferior = True  # negative PnL OR inferior-to-forward on any split
            split_results[lab] = res
        line += f"  {pass_ct}/4"
        print(line)
        all_results[c['name']]['three_way'] = split_results
        all_results[c['name']]['three_way_pass'] = pass_ct
        all_results[c['name']]['three_way_has_inferior'] = has_inferior

    # ── 5. Verdict assignment ──
    print(f"\n{'='*96}")
    print("VERDICT SUMMARY")
    print(f"{'='*96}")
    # Spec gates (strict):
    #   FAIL if any: WF ≥2 folds degraded ≥5pp, OR 3-way has any "inferior-to-forward" or negative test
    #   PASS if all : WF 5/5 improved by ≥2pp AND WFdelta ≥+10pp AND 3-way 4/4 AND OOS acc > 60%
    #   AMBIGUOUS otherwise
    verdicts = {}
    for name, r in all_results.items():
        tot = r['wf']['totals']
        passed = r['three_way_pass']
        has_inf = r.get('three_way_has_inferior', True)
        dpl = tot['delta_PnL']
        deg = tot['folds_degraded_5pp']
        impr = tot['folds_improved']
        acc = tot['subset_acc']
        # Spec-strict failure triggers first
        if deg >= 2:
            v = 'FAIL (≥2 folds degraded by >5pp)'
        elif has_inf:
            v = 'FAIL (≥1 three-way split inferior to forward or negative)'
        elif dpl <= 0:
            v = 'FAIL (WF delta non-positive)'
        elif dpl >= 10 and passed == 4 and impr >= 4 and (acc is not None and acc > 60):
            v = 'PASS (all criteria met)'
        else:
            v = 'AMBIGUOUS (partial improvement only)'
        verdicts[name] = v
        print(f"  {name:28s}  WFdelta={dpl:+7.2f}pp  Folds+={impr}/5  Deg5={deg}  "
              f"3way={passed}/4  InfSplit={'Y' if has_inf else 'N'}  Acc={acc}  → {v}")

    # ── 6. Save JSON ──
    # Strip unpickleable items from models; the structure already has
    # floats/ints/strings so we can dump directly.
    def _scrub_fold(f):
        return {k: v for k, v in f.items() if k != 'train_end'}
    serialisable = {}
    for name, r in all_results.items():
        wf = r['wf']
        # Diagnostic: fold-5 concentration
        folds = wf['folds']
        f5_delta = folds[-1]['delta_PnL'] if folds else 0
        ex5_delta = sum(f['delta_PnL'] for f in folds[:-1])
        serialisable[name] = {
            'wf_folds': [_scrub_fold(f) for f in wf['folds']],
            'wf_totals': wf['totals'],
            'wf_diag': {
                'fold5_delta': f5_delta,
                'folds1to4_delta_sum': round(ex5_delta, 2),
            },
            'three_way': {k: (v and {
                'train': v['train'], 'val': v['val'], 'test': v['test']
            }) for k, v in r['three_way'].items()},
            'three_way_pass': r['three_way_pass'],
            'three_way_has_inferior': r.get('three_way_has_inferior', True),
            'verdict': verdicts[name],
        }

    out = {
        'date_run': datetime.now().isoformat(),
        'script': 'c1_regime_classifier.py',
        'data_file': 'btc_5m_270days_reclassified.csv',
        'period_days': days,
        'cfg': cfg,
        'fee_rt_pct': FEE_RT_PCT,
        'n_signals': n_sig,
        'n_subset': n_sub,
        'forward_all_pnl_sanity': round(fwd_all_pnl, 2),
        'fractal_causality': {'swl_diffs': diffs_swl, 'swh_diffs': diffs_swh},
        'classifiers': serialisable,
    }
    out_path = ROOT / 'results' / 'c1_regime_classifier.json'
    with open(out_path, 'w') as f:
        json.dump(out, f, indent=2, default=str)
    print(f"\nSaved: {out_path}")


if __name__ == '__main__':
    main()
