#!/usr/bin/env python3
"""
Adversarial Evaluator v2 — GAN-inspired, system-aware, asymmetric-cost evaluation.
===================================================================================

v1 문제점 해결:
  1. 시스템 본질(변동성 수확기) 인식: 방향 무관성은 결함이 아님
  2. 가중 점수제: 과적합 위험 > 견고성 > 안정성
  3. 적응형 threshold: 파라미터 스케일에 맞는 동적 기준
  4. 비대칭 비용: Type 1 (false GO) 비용 >> Type 2 (false STOP)
  5. 새 공격 벡터: Tail Risk, Fold Decay, Trade Density, Redundancy, Multiple Testing

Architecture:
  Tier 1 (필수, 탈락형): 통과 못하면 즉시 REJECT
    G1. WF OOS positive (5/5 folds)
    G2. WR margin > breakeven
    G3. 과적합 점수 < 0.5

  Tier 2 (가중, 점수형): 10점 만점에서 차감
    A1. Time Bootstrap stability        (weight 2.0)
    A2. Noise Robustness                (weight 1.5)
    A3. Regime Worst-Case               (weight 2.0)
    A4. Fold Decay (최근 fold 열화)      (weight 2.5) ← NEW
    A5. Tail Risk (max DD 구간 행동)     (weight 2.0) ← NEW
    A6. Parameter Neighborhood          (weight 1.5)
    A7. Trade Count Stability           (weight 1.0) ← NEW
    A8. Multiple Testing Correction     (weight 1.5) ← NEW

  Final Verdict:
    Tier 1 ALL PASS + Tier 2 >= 7.0 → STRONG GO
    Tier 1 ALL PASS + Tier 2 >= 5.0 → CONDITIONAL GO
    Tier 1 ANY FAIL or Tier 2 < 5.0  → REJECT
"""
import os, sys, json, warnings
import numpy as np
from collections import defaultdict
from datetime import datetime
warnings.filterwarnings('ignore')

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, _PROJECT_ROOT)

import scripts.analysis.loop_evaluator as _le
from scripts.scanner.pattern_scanner import (
    portfolio_npos, calc_stats_compound,
)

LOG_FILE = os.path.join(_PROJECT_ROOT, 'results', 'strategy_loop_log.json')
STATE_FILE = os.path.join(_PROJECT_ROOT, 'results', 'strategy_loop_state.json')

DEFAULT_TP_DECAY_RATE = _le.DEFAULT_TP_DECAY_RATE


def _ensure_loaded():
    _le._ensure_loaded()

def _get_sigs():
    _le._ensure_loaded()
    return _le._sigs

def _get_n():
    _le._ensure_loaded()
    return _le._n

def _get_arrays():
    _le._ensure_loaded()
    return _le._opens, _le._highs, _le._lows, _le._closes, _le._atr, _le._ema

def _run_npos(start, end, params):
    return _le._run_npos(start, end, params)


def _wf_folds(params, n_folds=5, sigs_override=None):
    """Run WF, return list of per-fold stat dicts."""
    _ensure_loaded()
    n = _get_n()
    opens, highs, lows, closes, atr, ema = _get_arrays()
    sigs = sigs_override or _get_sigs()
    folds = []
    for fi in range(n_folds):
        is_end = int(n * (fi + 1) / (n_folds + 1))
        oos_end = int(n * (fi + 2) / (n_folds + 1))
        t, st = portfolio_npos(sigs, opens, highs, lows, closes, n, atr, ema,
                               start_bar=is_end, end_bar=oos_end, **params)
        cs = calc_stats_compound(t)
        cs['mdd_mtm'] = st.get('mdd_mtm', 0)
        if t:
            wins = [x['pnl_slot'] for x in t if x['pnl_slot'] > 0]
            loses = [x['pnl_slot'] for x in t if x['pnl_slot'] <= 0]
            aw = np.mean(wins) if wins else 0
            al = abs(np.mean(loses)) if loses else 1
            cs['rr'] = round(aw / al, 3) if al > 0 else 0
            be = al / (aw + al) * 100 if (aw + al) > 0 else 50
            cs['wr_margin'] = round(cs['wr'] - be, 1)
            cs['n_trades'] = len(t)
        else:
            cs['rr'] = 0
            cs['wr_margin'] = 0
            cs['n_trades'] = 0
        cs['fold_idx'] = fi
        cs['is_end'] = is_end
        cs['oos_end'] = oos_end
        folds.append(cs)
    return folds


# ====================================================================
# TIER 1: Mandatory Gates (instant REJECT if any fails)
# ====================================================================

def gate_wf_positive(treatment_folds):
    """G1: All 5 WF folds must be positive."""
    pnls = [f['pnl'] for f in treatment_folds]
    passed = all(p > 0 for p in pnls)
    return {
        'gate': 'G1_WF_POSITIVE',
        'pnls': [round(p, 1) for p in pnls],
        'all_positive': passed,
        'passed': passed,
        'reason': '' if passed else f"Fold(s) negative: {[i+1 for i, p in enumerate(pnls) if p <= 0]}"
    }


def gate_wr_margin(treatment_folds, min_margin=5.0):
    """G2: Average WR margin must exceed breakeven by min_margin pp."""
    margins = [f.get('wr_margin', 0) for f in treatment_folds]
    avg = np.mean(margins)
    worst = min(margins)
    passed = avg >= min_margin and worst >= 0
    return {
        'gate': 'G2_WR_MARGIN',
        'avg_margin': round(avg, 1),
        'worst_margin': round(worst, 1),
        'threshold': min_margin,
        'passed': passed,
        'reason': '' if passed else f"Avg margin {avg:.1f}pp < {min_margin}pp or worst {worst:.1f}pp < 0"
    }


def gate_overfit_score(baseline_params, treatment_params):
    """G3: IS-OOS overfit ratio must be < 0.5."""
    _ensure_loaded()
    n = _get_n()

    b_is = _run_npos(576, n, baseline_params)
    t_is = _run_npos(576, n, treatment_params)
    is_improve = t_is['pnl'] - b_is['pnl']

    b_folds = _wf_folds(baseline_params)
    t_folds = _wf_folds(treatment_params)
    oos_improve = sum(f['pnl'] for f in t_folds) - sum(f['pnl'] for f in b_folds)

    if is_improve > 0:
        overfit = 1.0 - (oos_improve / is_improve)
    else:
        overfit = 0.0
    overfit = max(0, min(1, overfit))

    passed = overfit < 0.5
    return {
        'gate': 'G3_OVERFIT',
        'is_improvement': round(is_improve, 1),
        'oos_improvement': round(oos_improve, 1),
        'overfit_score': round(overfit, 3),
        'risk': 'LOW' if overfit < 0.25 else 'MODERATE' if overfit < 0.5 else 'HIGH',
        'passed': passed,
        'reason': '' if passed else f"Overfit {overfit:.3f} >= 0.5"
    }


# ====================================================================
# TIER 2: Weighted Attacks (scored 0-10)
# ====================================================================

def attack_time_bootstrap(baseline_params, treatment_params, n_seeds=10):
    """A1: Random time splits — does improvement persist? (weight 2.0)"""
    _ensure_loaded()
    n = _get_n()
    gaps = []
    for seed in range(42, 42 + n_seeds):
        rng = np.random.RandomState(seed)
        split = rng.randint(int(n * 0.3), int(n * 0.6))
        oos_end = min(split + int(n * 0.25), n)
        b = _run_npos(split, oos_end, baseline_params)
        t = _run_npos(split, oos_end, treatment_params)
        gaps.append(t['pnl'] - b['pnl'])

    pos_ratio = sum(1 for g in gaps if g > 0) / len(gaps)
    # Score: 0-1 based on positive ratio (0.5=chance, 1.0=all positive)
    raw_score = max(0, (pos_ratio - 0.5) * 2)  # 0.5→0, 0.75→0.5, 1.0→1.0
    return {
        'attack': 'A1_TIME_BOOTSTRAP',
        'weight': 2.0,
        'positive_ratio': round(pos_ratio, 2),
        'mean_gap': round(np.mean(gaps), 1),
        'std_gap': round(np.std(gaps), 1),
        'raw_score': round(raw_score, 3),
        'weighted_score': round(raw_score * 2.0, 3),
    }


def attack_noise_robustness(baseline_params, treatment_params):
    """A2: TP/SL noise injection — does improvement survive? (weight 1.5)"""
    _ensure_loaded()
    n = _get_n()
    opens, highs, lows, closes, atr, ema = _get_arrays()
    sigs = _get_sigs()

    noise_results = {}
    for noise in [0.05, 0.10, 0.20]:
        rng = np.random.RandomState(42)
        noisy = [(b, p, d, max(0.3, tp * (1 + rng.uniform(-noise, noise))),
                  max(0.5, sl * (1 + rng.uniform(-noise, noise))))
                 for (b, p, d, tp, sl) in sigs]

        b_folds = _wf_folds(baseline_params, sigs_override=noisy)
        t_folds = _wf_folds(treatment_params, sigs_override=noisy)
        gap = sum(f['pnl'] for f in t_folds) - sum(f['pnl'] for f in b_folds)
        noise_results[f'{noise:.0%}'] = {'gap': round(gap, 1), 'positive': gap > 0}

    survive_count = sum(1 for v in noise_results.values() if v['positive'])
    raw_score = survive_count / len(noise_results)
    return {
        'attack': 'A2_NOISE_ROBUSTNESS',
        'weight': 1.5,
        'noise_levels': noise_results,
        'survive_ratio': round(raw_score, 2),
        'raw_score': round(raw_score, 3),
        'weighted_score': round(raw_score * 1.5, 3),
    }


def attack_regime_worst(baseline_params, treatment_params):
    """A3: Worst regime window performance (weight 2.0)"""
    _ensure_loaded()
    n = _get_n()
    window = 2880

    gaps = []
    for i in range(576, n - window, window):
        b = _run_npos(i, i + window, baseline_params)
        t = _run_npos(i, i + window, treatment_params)
        gaps.append(t['pnl'] - b['pnl'])

    worst = min(gaps) if gaps else -999
    mean = np.mean(gaps) if gaps else 0
    negative_pct = sum(1 for g in gaps if g < 0) / max(len(gaps), 1)

    # Score: penalize severe worst-case, reward low negative_pct
    # worst > 0: perfect (1.0), worst > -5: good (0.7), worst > -15: poor (0.3), else: bad (0)
    if worst >= 0:
        worst_score = 1.0
    elif worst >= -5:
        worst_score = 0.7 + 0.3 * (worst + 5) / 5
    elif worst >= -15:
        worst_score = 0.3 + 0.4 * (worst + 15) / 10
    else:
        worst_score = max(0, 0.3 * (worst + 30) / 15)

    neg_score = 1.0 - negative_pct
    raw_score = 0.6 * worst_score + 0.4 * neg_score

    return {
        'attack': 'A3_REGIME_WORST',
        'weight': 2.0,
        'worst_gap': round(worst, 1),
        'mean_gap': round(mean, 1),
        'negative_pct': round(negative_pct * 100, 1),
        'n_windows': len(gaps),
        'raw_score': round(raw_score, 3),
        'weighted_score': round(raw_score * 2.0, 3),
    }


def attack_fold_decay(baseline_folds, treatment_folds):
    """A4: Does improvement DECAY in recent folds? (weight 2.5) — NEW
    Most important: if improvement fades over time, it's adapting to past, not future."""
    b_pnls = [f['pnl'] for f in baseline_folds]
    t_pnls = [f['pnl'] for f in treatment_folds]
    gaps = [t - b for t, b in zip(t_pnls, b_pnls)]

    # Check if last 2 folds have smaller gap than first 2
    if len(gaps) >= 4:
        early_avg = np.mean(gaps[:2])
        late_avg = np.mean(gaps[-2:])
        decay_ratio = late_avg / max(early_avg, 0.01) if early_avg > 0 else 0

        # Score: decay_ratio > 1.0 = improving over time (1.0), = 1.0 = stable (0.7), < 0.5 = decaying (0)
        if decay_ratio >= 1.0:
            raw_score = 1.0
        elif decay_ratio >= 0.7:
            raw_score = 0.7 + 0.3 * (decay_ratio - 0.7) / 0.3
        elif decay_ratio >= 0.3:
            raw_score = 0.3 + 0.4 * (decay_ratio - 0.3) / 0.4
        else:
            raw_score = max(0, decay_ratio / 0.3 * 0.3)
    else:
        decay_ratio = 1.0
        early_avg = late_avg = np.mean(gaps)
        raw_score = 0.5

    return {
        'attack': 'A4_FOLD_DECAY',
        'weight': 2.5,
        'gaps_by_fold': [round(g, 1) for g in gaps],
        'early_avg': round(early_avg, 1),
        'late_avg': round(late_avg, 1),
        'decay_ratio': round(decay_ratio, 3),
        'trend': 'IMPROVING' if decay_ratio > 1.1 else 'STABLE' if decay_ratio > 0.7 else 'DECAYING',
        'raw_score': round(raw_score, 3),
        'weighted_score': round(raw_score * 2.5, 3),
    }


def attack_tail_risk(baseline_params, treatment_params):
    """A5: Performance during max drawdown periods (weight 2.0) — NEW
    Does the improvement help during the WORST times, or only during good times?"""
    _ensure_loaded()
    n = _get_n()
    closes = _get_arrays()[3]

    # Find the 3 worst drawdown periods in the data
    # Simple: sliding 1440-bar (5-day) windows, find worst return
    window = 1440
    dd_windows = []
    for i in range(576, n - window, window // 2):
        ret = (closes[i + window - 1] - closes[i]) / closes[i] * 100
        dd_windows.append((i, i + window, ret))

    dd_windows.sort(key=lambda x: x[2])
    worst_3 = dd_windows[:3]

    gaps = []
    for start, end, ret in worst_3:
        b = _run_npos(start, end, baseline_params)
        t = _run_npos(start, end, treatment_params)
        gap = t['pnl'] - b['pnl']
        gaps.append({'start': start, 'end': end, 'btc_return': round(ret, 1),
                     'baseline_pnl': round(b['pnl'], 1), 'treatment_pnl': round(t['pnl'], 1),
                     'gap': round(gap, 1)})

    # Score: improvement should help MOST during bad times
    pos_in_bad = sum(1 for g in gaps if g['gap'] > 0)
    avg_gap = np.mean([g['gap'] for g in gaps])

    if pos_in_bad == 3:
        raw_score = 1.0
    elif pos_in_bad == 2:
        raw_score = 0.7
    elif pos_in_bad == 1:
        raw_score = 0.3
    else:
        raw_score = 0.0

    # Bonus for large positive gaps
    if avg_gap > 5:
        raw_score = min(1.0, raw_score + 0.2)

    return {
        'attack': 'A5_TAIL_RISK',
        'weight': 2.0,
        'worst_periods': gaps,
        'positive_in_worst': pos_in_bad,
        'avg_gap_in_worst': round(avg_gap, 1),
        'raw_score': round(raw_score, 3),
        'weighted_score': round(raw_score * 2.0, 3),
    }


def attack_param_neighborhood(baseline_params, treatment_params, param_name, param_value):
    """A6: Adaptive neighborhood test (weight 1.5)
    Uses parameter-specific step sizes, not fixed ±10%."""
    _ensure_loaded()

    # Adaptive step sizes by parameter type
    step_map = {
        'preemptive_cascade_pct': 0.5,   # ±0.5 absolute
        'cascade_tighten_pct': 2,        # ±2 absolute
        'timeout_bars': 24,              # ±24 bars (2h)
        'direction_cap': 1,              # ±1
        'agg_risk_counter': 1.0,         # ±1%
        'agg_risk_with': 1.5,            # ±1.5%
        'momentum_threshold': 0.25,      # ±0.25%
        'momentum_cooldown': 3,          # ±3 bars
        'tp_decay_rate': 0.001,          # ±0.001
    }

    step = step_map.get(param_name, abs(param_value) * 0.1)
    neighbors = {
        'low': param_value - step,
        'center': param_value,
        'high': param_value + step,
    }

    b_total = sum(f['pnl'] for f in _wf_folds(baseline_params))
    results = {}
    for label, val in neighbors.items():
        p = {**treatment_params, param_name: val}
        t_total = sum(f['pnl'] for f in _wf_folds(p))
        results[label] = round(t_total - b_total, 1)

    # Score: all neighbors positive + low variance = stable
    all_pos = all(g > 0 for g in results.values())
    gap_range = max(results.values()) - min(results.values())
    center_gap = results['center']

    if all_pos and gap_range < center_gap * 0.5:
        raw_score = 1.0
    elif all_pos:
        raw_score = 0.7
    elif results['center'] > 0:
        raw_score = 0.4
    else:
        raw_score = 0.0

    return {
        'attack': 'A6_PARAM_NEIGHBORHOOD',
        'weight': 1.5,
        'param': param_name,
        'step': step,
        'neighbors': neighbors,
        'gaps': results,
        'range': round(gap_range, 1),
        'all_positive': all_pos,
        'raw_score': round(raw_score, 3),
        'weighted_score': round(raw_score * 1.5, 3),
    }


def attack_trade_count(baseline_folds, treatment_folds):
    """A7: Does trade count change dramatically? (weight 1.0) — NEW
    If treatment adds/removes >30% trades, it's changing the system fundamentally."""
    b_trades = sum(f.get('n_trades', f.get('trades', 0)) for f in baseline_folds)
    t_trades = sum(f.get('n_trades', f.get('trades', 0)) for f in treatment_folds)

    if b_trades > 0:
        change_pct = abs(t_trades - b_trades) / b_trades * 100
    else:
        change_pct = 100

    # Score: <10% change = perfect, 10-30% = ok, >30% = suspicious
    if change_pct < 10:
        raw_score = 1.0
    elif change_pct < 20:
        raw_score = 0.8
    elif change_pct < 30:
        raw_score = 0.5
    else:
        raw_score = max(0, 1.0 - change_pct / 50)

    return {
        'attack': 'A7_TRADE_COUNT',
        'weight': 1.0,
        'baseline_trades': b_trades,
        'treatment_trades': t_trades,
        'change_pct': round(change_pct, 1),
        'raw_score': round(raw_score, 3),
        'weighted_score': round(raw_score * 1.0, 3),
    }


def attack_multiple_testing(iteration_number, n_total_params=9):
    """A8: Bonferroni-like correction for multiple testing (weight 1.5) — NEW
    After testing N hypotheses, chance of at least 1 false positive increases."""
    # Effective significance: 0.05 / n_tests (Bonferroni)
    # But we use a softer correction: (1 - 0.05)^N chance of no false positive
    p_false_positive = 1 - (1 - 0.05) ** iteration_number
    required_confidence = 1 - 0.05 / max(iteration_number, 1)

    # Score: penalize later iterations more heavily
    if iteration_number <= 3:
        raw_score = 1.0  # early testing: no penalty
    elif iteration_number <= 6:
        raw_score = 0.8
    elif iteration_number <= 10:
        raw_score = 0.6
    else:
        raw_score = max(0.3, 0.6 - (iteration_number - 10) * 0.05)

    return {
        'attack': 'A8_MULTIPLE_TESTING',
        'weight': 1.5,
        'iteration': iteration_number,
        'total_params_tested': n_total_params,
        'false_positive_prob': round(p_false_positive, 3),
        'required_confidence': round(required_confidence, 3),
        'raw_score': round(raw_score, 3),
        'weighted_score': round(raw_score * 1.5, 3),
    }


# ====================================================================
# ORCHESTRATOR
# ====================================================================

class AdversarialEvaluatorV2:
    """Orchestrates Tier 1 gates + Tier 2 weighted attacks."""

    def __init__(self, baseline_params, treatment_params, param_name, param_value, iteration=1):
        self.baseline = baseline_params
        self.treatment = treatment_params
        self.param_name = param_name
        self.param_value = param_value
        self.iteration = iteration

    def run(self):
        _ensure_loaded()
        print(f"\n{'='*65}")
        print(f"ADVERSARIAL EVALUATOR v2")
        print(f"Change: {self.param_name} → {self.param_value}")
        print(f"{'='*65}")

        # ── Tier 1: Mandatory Gates ──
        print(f"\n{'─'*65}")
        print("TIER 1: Mandatory Gates (instant REJECT on failure)")
        print(f"{'─'*65}")

        t_folds = _wf_folds(self.treatment)
        b_folds = _wf_folds(self.baseline)

        g1 = gate_wf_positive(t_folds)
        g2 = gate_wr_margin(t_folds)
        g3 = gate_overfit_score(self.baseline, self.treatment)

        tier1_results = [g1, g2, g3]
        tier1_pass = all(g['passed'] for g in tier1_results)

        for g in tier1_results:
            icon = "✅" if g['passed'] else "❌"
            print(f"  {icon} {g['gate']}: {'PASS' if g['passed'] else 'FAIL'}")
            if not g['passed']:
                print(f"     → {g['reason']}")

        if not tier1_pass:
            print(f"\n  ❌ TIER 1 FAILED — REJECT")
            return {
                'verdict': 'REJECT',
                'tier1': {g['gate']: g for g in tier1_results},
                'reason': 'Tier 1 gate failure',
                'score': 0,
                'max_score': 14.0,
            }

        # ── Tier 2: Weighted Attacks ──
        print(f"\n{'─'*65}")
        print("TIER 2: Adversarial Attacks (weighted scoring)")
        print(f"{'─'*65}")

        attacks = [
            attack_time_bootstrap(self.baseline, self.treatment),
            attack_noise_robustness(self.baseline, self.treatment),
            attack_regime_worst(self.baseline, self.treatment),
            attack_fold_decay(b_folds, t_folds),
            attack_tail_risk(self.baseline, self.treatment),
            attack_param_neighborhood(self.baseline, self.treatment,
                                       self.param_name, self.param_value),
            attack_trade_count(b_folds, t_folds),
            attack_multiple_testing(self.iteration),
        ]

        total_weighted = sum(a['weighted_score'] for a in attacks)
        max_weighted = sum(a['weight'] for a in attacks)

        for a in attacks:
            bar_len = int(a['raw_score'] * 20)
            bar = "█" * bar_len + "░" * (20 - bar_len)
            print(f"  {a['attack']:28s} [{bar}] {a['weighted_score']:.1f}/{a['weight']:.1f}")

        print(f"\n  Total Score: {total_weighted:.1f} / {max_weighted:.1f} "
              f"({total_weighted/max_weighted*100:.0f}%)")

        # ── Final Verdict ──
        normalized = total_weighted / max_weighted * 10  # scale to 0-10

        if normalized >= 7.0:
            verdict = "STRONG GO"
        elif normalized >= 5.0:
            verdict = "CONDITIONAL GO"
        else:
            verdict = "REJECT"

        print(f"\n{'='*65}")
        print(f"  VERDICT: {verdict} (score {normalized:.1f}/10)")
        print(f"{'='*65}")

        # WF summary
        t_total = sum(f['pnl'] for f in t_folds)
        b_total = sum(f['pnl'] for f in b_folds)
        print(f"  Baseline OOS: {b_total:+.1f}% → Treatment OOS: {t_total:+.1f}% "
              f"(delta {t_total-b_total:+.1f}%)")

        return {
            'verdict': verdict,
            'normalized_score': round(normalized, 2),
            'raw_score': round(total_weighted, 2),
            'max_score': round(max_weighted, 1),
            'tier1': {g['gate']: g for g in tier1_results},
            'tier2': {a['attack']: a for a in attacks},
            'baseline_oos': round(b_total, 1),
            'treatment_oos': round(t_total, 1),
            'delta': round(t_total - b_total, 1),
            'treatment_folds': [round(f['pnl'], 1) for f in t_folds],
        }


# ====================================================================
# INTEGRATED HARNESS
# ====================================================================

def run_iteration(param_name=None, new_value=None):
    """Run one adversarial iteration (auto-generates if no params given)."""
    _ensure_loaded()

    with open(STATE_FILE) as f:
        state = json.load(f)

    current = state.get('config_snapshot', {})
    baseline = {**current, 'tp_decay_rate': DEFAULT_TP_DECAY_RATE}

    if param_name and new_value is not None:
        treatment = {**baseline, param_name: new_value}
        iteration = state.get('iteration', 0) + 1
    else:
        # Use AdaptiveGenerator from v1
        from scripts.analysis.adversarial_evaluator import AdaptiveGenerator
        gen = AdaptiveGenerator()
        candidate = gen.generate_next()
        if candidate is None:
            print("Generator exhausted.")
            return None
        param_name = candidate['param']
        new_value = candidate['new']
        treatment = {**baseline, param_name: new_value}
        iteration = state.get('iteration', 0) + 1

    print(f"\n{'#'*65}")
    print(f"ITERATION {iteration}: {param_name} {current.get(param_name, '?')} → {new_value}")
    print(f"{'#'*65}")

    # IS quick gate
    n = _get_n()
    recent = n - 8640
    b_is = _run_npos(recent, n, baseline)
    t_is = _run_npos(recent, n, treatment)
    print(f"\n[Quick IS] Baseline: {b_is['pnl']:+.1f}% | Treatment: {t_is['pnl']:+.1f}%")

    if t_is['pnl'] <= b_is['pnl']:
        print("  ❌ IS FAILED — skipping adversarial battery")
        result = {'verdict': 'REJECT', 'reason': 'IS failed', 'normalized_score': 0}
    else:
        ev = AdversarialEvaluatorV2(baseline, treatment, param_name, new_value, iteration)
        result = ev.run()

    # Record
    with open(LOG_FILE) as f:
        log = json.load(f)
    log.append({
        'iteration': iteration,
        'hypothesis': f"{param_name} {current.get(param_name, '?')} → {new_value}",
        'parameter': param_name,
        'change': f"{current.get(param_name, '?')} → {new_value}",
        'result': result,
        'timestamp': datetime.now().isoformat(),
        'evaluator_version': 'v2',
    })
    with open(LOG_FILE, 'w') as f:
        json.dump(log, f, indent=2, default=str)

    # Update state
    state['iteration'] = iteration
    state['status'] = result['verdict']
    if 'GO' in result.get('verdict', ''):
        state['applied_changes'].append({
            'iteration': iteration, 'param': param_name,
            'old': current.get(param_name), 'new': new_value,
            'score': result.get('normalized_score', 0),
        })
    else:
        state['rejected_hypotheses'].append({
            'iteration': iteration,
            'hypothesis': f"{param_name} → {new_value}",
            'score': result.get('normalized_score', 0),
        })
    with open(STATE_FILE, 'w') as f:
        json.dump(state, f, indent=2, default=str)

    return result


if __name__ == '__main__':
    # Test: run with pre-emptive cascade 2.0 (iter 3 GO candidate)
    result = run_iteration('preemptive_cascade_pct', 2.0)
    if result:
        print(f"\nFinal: {result['verdict']} (score {result.get('normalized_score', 0):.1f}/10)")
