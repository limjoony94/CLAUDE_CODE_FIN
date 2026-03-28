#!/usr/bin/env python3
"""
Loop Evaluator — Reusable evaluation module for Strategy Loop iterations.
Provides: IS quick check, WF validation, discrimination test, cascade dependency.
"""
import os, sys, json, warnings
import numpy as np
from collections import defaultdict
warnings.filterwarnings('ignore')

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, _PROJECT_ROOT)

from scripts.scanner.pattern_scanner import (
    load_and_classify, build_signal_index,
    portfolio_npos, calc_stats_compound,
    compute_atr_ratio, compute_ema_slope,
    DEFAULT_TP_DECAY_RATE,
    DEFAULT_N_SLOTS, DEFAULT_DIRECTION_CAP,
    DEFAULT_AGG_RISK_COUNTER, DEFAULT_AGG_RISK_WITH,
    DEFAULT_MOMENTUM_LOOKBACK, DEFAULT_MOMENTUM_THRESHOLD, DEFAULT_MOMENTUM_COOLDOWN,
    DEFAULT_REGIME_MULT,
)

DATA_FILE = os.path.join(_PROJECT_ROOT, 'data', 'btc_5m_270days_reclassified.csv')
PATTERNS_FILE = os.path.join(_PROJECT_ROOT, 'results', 'dynamic_patterns.json')

# Cached globals (loaded once)
_df = None
_n = 0
_opens = _highs = _lows = _closes = _types = _atr = _ema = _sidx = None
_sigs = None


def _ensure_loaded():
    global _df, _n, _opens, _highs, _lows, _closes, _types, _atr, _ema, _sidx, _sigs
    if _df is not None:
        return

    _df = load_and_classify(DATA_FILE)
    _n = len(_df)
    _opens = _df['open'].values
    _highs = _df['high'].values
    _lows = _df['low'].values
    _closes = _df['close'].values
    _types = _df['candle_type'].values
    _atr = compute_atr_ratio(_highs, _lows, _closes, atr_period=14, window=576)
    _ema = compute_ema_slope(_closes, period=20, lookback=5)
    _sidx = build_signal_index(_types, _n)

    with open(PATTERNS_FILE) as f:
        dp = json.load(f)
    details = dp['pattern_details']
    _sigs = []
    for key, pd_ in details.items():
        pat, d = pd_['pattern'], pd_['direction']
        tp = pd_.get('exc_stats', {}).get('mfe_median') or pd_['tp']
        sl = pd_['sl']
        for b in _sidx.get(pat, []):
            _sigs.append((b, pat, d, tp, sl))


def get_data_info():
    _ensure_loaded()
    return {'bars': _n, 'days': _n / 288, 'patterns': len(set(s[1] for s in _sigs)),
            'signals': len(_sigs)}


def _run_npos(start, end, params, sigs_override=None):
    """Run portfolio_npos with given params."""
    _ensure_loaded()
    sigs = sigs_override or _sigs
    t, st = portfolio_npos(
        sigs, _opens, _highs, _lows, _closes, _n, _atr, _ema,
        start_bar=start, end_bar=end,
        n_slots=params.get('n_slots', DEFAULT_N_SLOTS),
        direction_cap=params.get('direction_cap', DEFAULT_DIRECTION_CAP),
        regime_mult=params.get('regime_mult', DEFAULT_REGIME_MULT),
        agg_risk_counter=params.get('agg_risk_counter', DEFAULT_AGG_RISK_COUNTER),
        agg_risk_with=params.get('agg_risk_with', DEFAULT_AGG_RISK_WITH),
        momentum_lookback=params.get('momentum_lookback', DEFAULT_MOMENTUM_LOOKBACK),
        momentum_threshold=params.get('momentum_threshold', DEFAULT_MOMENTUM_THRESHOLD),
        momentum_cooldown=params.get('momentum_cooldown', DEFAULT_MOMENTUM_COOLDOWN),
        cascade_tighten_pct=params.get('cascade_tighten_pct', 95),
        preemptive_cascade_pct=params.get('preemptive_cascade_pct', 3.0),
        preemptive_tighten_pct=params.get('preemptive_tighten_pct', 95),
        timeout_bars=params.get('timeout_bars', 288),
        tp_decay_rate=params.get('tp_decay_rate', DEFAULT_TP_DECAY_RATE),
        vol_adapt_clamp=params.get('vol_adapt_clamp', (0.0, 0.0)),
    )
    cs = calc_stats_compound(t)
    cs['mdd_mtm'] = st.get('mdd_mtm', 0)
    if t:
        wins = [x['pnl_slot'] for x in t if x['pnl_slot'] > 0]
        loses = [x['pnl_slot'] for x in t if x['pnl_slot'] <= 0]
        aw = np.mean(wins) if wins else 0
        al = abs(np.mean(loses)) if loses else 1
        cs['rr'] = round(aw / al, 3) if al > 0 else 0
        cs['avg_win'] = round(aw, 3)
        cs['avg_loss'] = round(-abs(np.mean(loses)) if loses else 0, 3)
        be = al / (aw + al) * 100 if (aw + al) > 0 else 50
        cs['wr_margin'] = round(cs['wr'] - be, 1)
        reasons = defaultdict(int)
        for x in t:
            reasons[x.get('reason', '?')] += 1
        cs['exit_reasons'] = dict(reasons)
    return cs


def is_quick_check(baseline_params, treatment_params):
    """Quick IS check on last 30 days. Returns (baseline, treatment, pass?)"""
    _ensure_loaded()
    recent = _n - 8640
    b = _run_npos(recent, _n, baseline_params)
    t = _run_npos(recent, _n, treatment_params)
    passed = t['pnl'] > b['pnl'] and t.get('wr_margin', 0) > 5
    return b, t, passed


def wf_validation(params, n_folds=5):
    """5-fold expanding window WF. Returns list of fold stats."""
    _ensure_loaded()
    folds = []
    for fi in range(n_folds):
        is_end = int(_n * (fi + 1) / (n_folds + 1))
        oos_end = int(_n * (fi + 2) / (n_folds + 1))
        s = _run_npos(is_end, oos_end, params)
        folds.append(s)
    return folds


def discrimination_test(params, n_seeds=10):
    """Shuffle directions and run WF. Returns (real_folds, shuffle_pass_count, shuffle_totals)."""
    _ensure_loaded()
    real_folds = wf_validation(params)
    real_total = sum(f['pnl'] for f in real_folds)

    shuffle_pass = 0
    shuffle_totals = []
    for seed in range(42, 42 + n_seeds):
        rng = np.random.RandomState(seed)
        shuffled = [(b, p, ('LONG' if rng.random() < 0.5 else 'SHORT'), tp, sl)
                    for (b, p, d, tp, sl) in _sigs]
        sf = []
        for fi in range(5):
            is_end = int(_n * (fi + 1) / 6)
            oos_end = int(_n * (fi + 2) / 6)
            s = _run_npos(is_end, oos_end, params, sigs_override=shuffled)
            sf.append(s)
        st = sum(f['pnl'] for f in sf)
        shuffle_totals.append(st)
        if all(f['pnl'] > 0 for f in sf):
            shuffle_pass += 1

    disc = "DISC" if shuffle_pass < 7 else "NON-DISC"
    return real_folds, shuffle_pass, shuffle_totals, disc


def cascade_dependency(baseline_params, treatment_params):
    """Test if treatment effect vanishes with cascade OFF."""
    _ensure_loaded()
    # ON
    on_base = wf_validation(baseline_params)
    on_treat = wf_validation(treatment_params)
    on_gap = sum(f['pnl'] for f in on_treat) - sum(f['pnl'] for f in on_base)

    # OFF
    off_base_params = {**baseline_params, 'cascade_tighten_pct': 0,
                       'preemptive_cascade_pct': 999, 'preemptive_tighten_pct': 0}
    off_treat_params = {**treatment_params, 'cascade_tighten_pct': 0,
                        'preemptive_cascade_pct': 999, 'preemptive_tighten_pct': 0}
    off_base = wf_validation(off_base_params)
    off_treat = wf_validation(off_treat_params)
    off_gap = sum(f['pnl'] for f in off_treat) - sum(f['pnl'] for f in off_base)

    ratio = abs(off_gap) / max(abs(on_gap), 0.01) if on_gap != 0 else 999
    verdict = 'CASCADE-DEPENDENT' if ratio < 0.3 else 'INDEPENDENT' if ratio > 0.7 else 'PARTIAL'

    return {
        'on_gap': round(on_gap, 1),
        'off_gap': round(off_gap, 1),
        'dependency_ratio': round(ratio, 2),
        'verdict': verdict,
    }


def pure_oos_check(params, oos_bars=6549):
    """Run on truly new data (post-training)."""
    _ensure_loaded()
    start = _n - oos_bars
    return _run_npos(start, _n, params)


def full_evaluation(baseline_params, treatment_params, hypothesis_name=""):
    """Run complete evaluation pipeline. Returns verdict dict."""
    _ensure_loaded()
    info = get_data_info()
    print(f"\n{'='*60}")
    print(f"EVALUATION: {hypothesis_name}")
    print(f"Data: {info['bars']} bars, {info['patterns']} patterns")
    print(f"{'='*60}")

    # Gate 0: IS Quick Check
    print("\n[Gate 0] IS Quick Check (last 30d)...")
    b_is, t_is, is_pass = is_quick_check(baseline_params, treatment_params)
    print(f"  Baseline: PnL {b_is['pnl']:+.1f}%, WR {b_is['wr']:.1f}%, R:R {b_is.get('rr',0):.3f}")
    print(f"  Treatment: PnL {t_is['pnl']:+.1f}%, WR {t_is['wr']:.1f}%, R:R {t_is.get('rr',0):.3f}")
    print(f"  IS Pass: {'YES' if is_pass else 'NO'}")

    if not is_pass:
        return {
            'verdict': 'STOP',
            'reason': 'IS quick check failed',
            'baseline_is': {'pnl': b_is['pnl'], 'wr': b_is['wr']},
            'treatment_is': {'pnl': t_is['pnl'], 'wr': t_is['wr']},
        }

    # Gate 1: WF
    print("\n[Gate 1] 5-Fold WF Validation...")
    b_folds = wf_validation(baseline_params)
    t_folds = wf_validation(treatment_params)

    b_pnls = [f['pnl'] for f in b_folds]
    t_pnls = [f['pnl'] for f in t_folds]
    b_total = sum(b_pnls)
    t_total = sum(t_pnls)

    wf_pass = all(p > 0 for p in t_pnls) and t_total > b_total * 1.05

    print(f"  Baseline: [{', '.join(f'{p:+.1f}' for p in b_pnls)}] = {b_total:+.1f}%")
    print(f"  Treatment: [{', '.join(f'{p:+.1f}' for p in t_pnls)}] = {t_total:+.1f}%")
    print(f"  Delta: {t_total - b_total:+.1f}% | WF Pass: {'YES' if wf_pass else 'NO'}")

    if not wf_pass:
        return {
            'verdict': 'STOP',
            'reason': 'WF failed or treatment < baseline',
            'baseline_oos': round(b_total, 1),
            'treatment_oos': round(t_total, 1),
        }

    # Gate 2: Discrimination
    print("\n[Gate 2] Discrimination Test (10 seeds)...")
    _, shuf_pass, shuf_totals, disc = discrimination_test(treatment_params, n_seeds=10)
    print(f"  Shuffled PASS: {shuf_pass}/10 | Verdict: {disc}")

    # Gate 3: Cascade Dependency
    print("\n[Gate 3] Cascade Dependency...")
    dep = cascade_dependency(baseline_params, treatment_params)
    print(f"  ON gap: {dep['on_gap']:+.1f}%, OFF gap: {dep['off_gap']:+.1f}%, "
          f"Ratio: {dep['dependency_ratio']:.2f} | {dep['verdict']}")

    # Pure OOS
    print("\n[Bonus] Pure OOS (23d post-training)...")
    oos = pure_oos_check(treatment_params)
    print(f"  PnL {oos['pnl']:+.1f}%, WR {oos['wr']:.1f}%, R:R {oos.get('rr',0):.3f}")

    # Final verdict
    wr_margin_ok = t_folds[-1].get('wr_margin', 0) > 10
    rr_ok = np.mean([f.get('rr', 0) for f in t_folds]) > np.mean([f.get('rr', 0) for f in b_folds]) * 0.8

    go = wf_pass and wr_margin_ok and rr_ok
    verdict = 'GO' if go else 'STOP'

    print(f"\n  VERDICT: {verdict}")
    if not go:
        reasons = []
        if not wr_margin_ok:
            reasons.append("WR margin < 10pp")
        if not rr_ok:
            reasons.append("R:R degraded > 20%")
        print(f"  Reason: {', '.join(reasons)}")

    return {
        'verdict': verdict,
        'baseline_oos': round(b_total, 1),
        'treatment_oos': round(t_total, 1),
        'delta': round(t_total - b_total, 1),
        'discrimination': disc,
        'shuffle_pass': shuf_pass,
        'cascade_dependency': dep,
        'pure_oos': {'pnl': round(oos['pnl'], 1), 'wr': oos['wr'], 'rr': oos.get('rr', 0)},
        'treatment_folds': [round(p, 1) for p in t_pnls],
        'baseline_folds': [round(p, 1) for p in b_pnls],
    }
