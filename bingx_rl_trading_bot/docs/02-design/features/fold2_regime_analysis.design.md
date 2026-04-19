# Design: Fold 2 Regime Analysis

> **Feature**: fold2_regime_analysis
> **Date**: 2026-04-19
> **Phase**: Design
> **Plan**: `docs/01-plan/features/fold2_regime_analysis.plan.md`

---

## 1. Architecture

```
scripts/analysis/fold2_regime_analysis.py  (NEW, ~300 lines)
 ├─ 재사용:
 │   ├─ intrabar_trail_impact (data, indicators)
 │   └─ c1_intrabar_parity (run_slip, set_combo, reset_combo)
 │
 ├─ 신규:
 │   ├─ compute_regime_metrics()   — per-fold ATR/std/range/trend/sideways
 │   ├─ compute_strategy_metrics() — per-fold trades 통계 (WR, R:R, exit 분포)
 │   ├─ sub_window_microscopy()    — 5일 rolling
 │   ├─ evaluate_regime_filter()   — H7 regime classifier calibration
 │   └─ summarize_hypotheses()     — H1~H7 verdict
 │
 └─ Output: results/fold2_regime_analysis_{timestamp}.json
```

---

## 2. Fold 경계 정의 (candidate_c_validation과 동일)

```python
# 실제 trade entry_bar 기반 분할 (candidate_C slip_med run 결과)
FOLD_BOUNDARIES = {
    'fold_1': {'bars': (31, 6407),      'dates': ('2025-05-05', '2025-07-11')},
    'fold_2': {'bars': (6407, 12783),   'dates': ('2025-07-11', '2025-09-15')},  # 약점
    'fold_3': {'bars': (12783, 19159),  'dates': ('2025-09-15', '2025-11-21')},
    'fold_4': {'bars': (19159, 25535),  'dates': ('2025-11-21', '2026-01-26')},
    'fold_5': {'bars': (25535, 31916),  'dates': ('2026-01-26', '2026-04-03')},
}
```

---

## 3. Regime Metrics (per fold)

```python
def compute_regime_metrics(fold_lo, fold_hi):
    """Fold 구간의 시장 레짐 지표."""
    h = ibt.h15[fold_lo:fold_hi]
    l = ibt.l15[fold_lo:fold_hi]
    c = ibt.c15[fold_lo:fold_hi]
    o = ibt.o15[fold_lo:fold_hi]
    atr_slice = ibt.atr14[fold_lo:fold_hi]

    # 변동성
    atr_avg = mean(x for x in atr_slice if not math.isnan(x))
    atr_pct_avg = atr_avg / mean(c) * 100

    # Close returns 14-bar rolling std
    returns = [(c[i]/c[i-1] - 1) for i in range(1, len(c))]
    ret_std = stdev(returns) * 100

    # Range pct (high-low)/close
    range_pct = mean((h[i] - l[i]) / c[i] * 100 for i in range(len(c)))

    # 추세 (EMA20-EMA50 기울기 proxy)
    # Simplified: first half close vs second half close
    half = len(c) // 2
    trend_pct = (c[-1]/c[half] - 1) * 100

    # Sideways index: (fold_max_high - fold_min_low) / atr_avg
    sideways_idx = (max(h) - min(l)) / atr_avg if atr_avg > 0 else 0

    return {
        'atr_avg': round(atr_avg, 2),
        'atr_pct_avg': round(atr_pct_avg, 3),  # %
        'returns_std_pct': round(ret_std, 3),
        'range_pct_avg': round(range_pct, 3),
        'trend_pct_half_to_full': round(trend_pct, 2),
        'sideways_index': round(sideways_idx, 2),
        'price_first': round(c[0], 1),
        'price_last': round(c[-1], 1),
        'price_max': round(max(h), 1),
        'price_min': round(min(l), 1),
    }
```

---

## 4. Strategy Metrics (per fold, per combo)

```python
def compute_strategy_metrics(combo_name, combo_cfg, fold_lo, fold_hi, slip='med'):
    """Fold 내 해당 combo의 전략 거동 metrics."""
    # run_slip을 통해 trades 생성 후 fold 필터링
    all_trades = run_slip(combo_cfg, slip)  # full period
    fold_trades = [t for t in all_trades if fold_lo <= t['entry_bar'] < fold_hi]

    if not fold_trades:
        return {'count': 0}

    wins = [t for t in fold_trades if t['net'] > 0]
    losses = [t for t in fold_trades if t['net'] <= 0]

    # Exit reason 분포
    reasons = {}
    for t in fold_trades:
        r = t.get('reason_effective', t['reason'])
        reasons[r] = reasons.get(r, 0) + 1
    reason_pct = {k: round(v/len(fold_trades)*100, 1) for k, v in reasons.items()}

    # Max consecutive loss streak
    streak = max_streak = 0
    for t in fold_trades:
        if t['net'] <= 0:
            streak += 1
            max_streak = max(max_streak, streak)
        else:
            streak = 0

    # Days in fold
    days = (fold_hi - fold_lo) / 96  # 15m bars / 96 = days

    return {
        'count': len(fold_trades),
        'trades_per_day': round(len(fold_trades) / days, 2),
        'wr_pct': round(len(wins) / len(fold_trades) * 100, 1),
        'pnl_sum': round(sum(t['net'] for t in fold_trades), 2),
        'avg_win': round(sum(t['net'] for t in wins) / len(wins), 3) if wins else 0,
        'avg_loss': round(sum(t['net'] for t in losses) / len(losses), 3) if losses else 0,
        'rr': round(abs(sum(t['net'] for t in wins) / len(wins) /
                        (sum(t['net'] for t in losses) / len(losses))), 2)
              if wins and losses else 0,
        'exit_reason_pct': reason_pct,
        'max_consec_loss': max_streak,
        'median_bars_held': sorted(t.get('bh', 0) for t in fold_trades)[len(fold_trades)//2],
    }
```

---

## 5. Sub-window Microscopy (fold 2 전용)

```python
def sub_window_microscopy(combo_cfg, fold_lo, fold_hi, window_days=5):
    """Fold 내부 5일 rolling PnL로 worst sub-window 특정."""
    all_trades = run_slip(combo_cfg, 'med')
    fold_trades = [t for t in all_trades if fold_lo <= t['entry_bar'] < fold_hi]
    ts_col = ibt.agg15['ts']

    bars_per_window = window_days * 96  # 15m bars
    stride = bars_per_window // 2  # 50% overlap

    windows = []
    cur = fold_lo
    while cur + bars_per_window <= fold_hi:
        w_trades = [t for t in fold_trades if cur <= t['entry_bar'] < cur + bars_per_window]
        w_pnl = sum(t['net'] for t in w_trades)
        windows.append({
            'start_date': str(ts_col.iloc[cur])[:10],
            'end_date':   str(ts_col.iloc[cur + bars_per_window - 1])[:10],
            'trades': len(w_trades),
            'pnl': round(w_pnl, 2),
            'wr': round(sum(1 for t in w_trades if t['net']>0)/len(w_trades)*100, 1)
                  if w_trades else 0,
        })
        cur += stride

    # Sort by worst
    windows_sorted = sorted(windows, key=lambda w: w['pnl'])
    return {
        'all_windows': windows,
        'worst_3': windows_sorted[:3],
        'best_3': windows_sorted[-3:],
    }
```

---

## 6. Regime Filter Calibration (H7)

```python
def evaluate_regime_filter(fold_metrics_dict):
    """
    Candidates for simple regime filter.
    For each threshold rule, check:
      - Does it flag fold 2 as 'risky'?
      - How many other folds get flagged? (cost)
      - Expected PnL improvement if we skip flagged periods
    """
    candidates = [
        {'name': 'low_atr_pct', 'metric': 'atr_pct_avg',
         'op': '<', 'threshold_values': [0.3, 0.4, 0.5, 0.6]},
        {'name': 'low_returns_std', 'metric': 'returns_std_pct',
         'op': '<', 'threshold_values': [0.05, 0.08, 0.10, 0.15]},
        {'name': 'high_sideways', 'metric': 'sideways_index',
         'op': '>', 'threshold_values': [40, 50, 60, 80]},
    ]

    results = []
    for c in candidates:
        for th in c['threshold_values']:
            flagged = []
            for fn, fm in fold_metrics_dict.items():
                val = fm[c['metric']]
                triggered = (val < th) if c['op'] == '<' else (val > th)
                if triggered:
                    flagged.append(fn)
            results.append({
                'rule': f"{c['metric']} {c['op']} {th}",
                'flagged_folds': flagged,
                'flags_fold_2': 'fold_2' in flagged,
                'also_flags': [f for f in flagged if f != 'fold_2'],
            })
    return results
```

---

## 7. Hypothesis Summary

```python
def summarize_hypotheses(regime_by_fold, strategy_by_fold_combo, sub_windows,
                         regime_filters):
    """Evaluate H1-H7 with numerical evidence."""
    r2 = regime_by_fold['fold_2']
    r_others = [regime_by_fold[f'fold_{i}'] for i in (1,3,4,5)]
    s2_cand = strategy_by_fold_combo['candidate_C']['fold_2']
    s2_base = strategy_by_fold_combo['baseline']['fold_2']

    def avg(lst, key): return sum(x[key] for x in lst) / len(lst)

    h = {}

    # H1 low vol
    h['H1_low_vol'] = {
        'fold_2_atr_pct': r2['atr_pct_avg'],
        'others_avg': round(avg(r_others, 'atr_pct_avg'), 3),
        'verdict': r2['atr_pct_avg'] < avg(r_others, 'atr_pct_avg') * 0.9,
    }

    # H2 low breakout freq
    others_tpd = [strategy_by_fold_combo['candidate_C'][f'fold_{i}']['trades_per_day']
                  for i in (1,3,4,5)]
    h['H2_low_breakout'] = {
        'fold_2_trades_per_day': s2_cand['trades_per_day'],
        'others_avg': round(sum(others_tpd)/len(others_tpd), 2),
        'verdict': s2_cand['trades_per_day'] < sum(others_tpd)/len(others_tpd) * 0.9,
    }

    # H3 high SL exit
    sl_pct = s2_cand.get('exit_reason_pct', {}).get('SL', 0)
    others_sl = [strategy_by_fold_combo['candidate_C'][f'fold_{i}']
                 .get('exit_reason_pct', {}).get('SL', 0) for i in (1,3,4,5)]
    h['H3_high_sl_exit'] = {
        'fold_2_sl_pct': sl_pct,
        'others_avg': round(sum(others_sl)/len(others_sl), 1),
        'verdict': sl_pct > sum(others_sl)/len(others_sl) * 1.1,
    }

    # H4 poor R:R
    h['H4_poor_rr'] = {
        'fold_2_rr': s2_cand['rr'],
        'others_avg': round(avg([strategy_by_fold_combo['candidate_C'][f'fold_{i}']
                                 for i in (1,3,4,5)], 'rr'), 2),
        'verdict': s2_cand['rr'] < avg([strategy_by_fold_combo['candidate_C'][f'fold_{i}']
                                         for i in (1,3,4,5)], 'rr') * 0.8,
    }

    # H5 baseline vs candidate divergence
    h['H5_widening_sl_amplifies'] = {
        'fold_2_baseline_pnl': s2_base['pnl_sum'],
        'fold_2_candidate_pnl': s2_cand['pnl_sum'],
        'diff': round(s2_cand['pnl_sum'] - s2_base['pnl_sum'], 2),
        'verdict_baseline_better': s2_base['pnl_sum'] > s2_cand['pnl_sum'],
    }

    # H6 concentrated sub-window
    worst = sub_windows['worst_3']
    worst_sum = sum(w['pnl'] for w in worst)
    h['H6_concentrated_loss'] = {
        'worst_3_windows_sum_pnl': round(worst_sum, 2),
        'fold_2_total_pnl': s2_cand['pnl_sum'],
        'worst_3_share_pct': round(worst_sum / s2_cand['pnl_sum'] * 100, 1)
                              if s2_cand['pnl_sum'] != 0 else 0,
        'worst_1_window': worst[0],
        'verdict_concentrated': worst_sum < s2_cand['pnl_sum'] * 0.5  # worst 3 > 50% loss
                                if s2_cand['pnl_sum'] < 0 else False,
    }

    # H7 regime filter viable
    clean_filters = [f for f in regime_filters
                     if f['flags_fold_2'] and len(f['also_flags']) <= 1]
    h['H7_regime_filter_viable'] = {
        'clean_filter_count': len(clean_filters),
        'best_candidates': clean_filters[:3],
        'verdict': len(clean_filters) > 0,
    }

    return h
```

---

## 8. Output Schema

```json
{
  "timestamp": "2026-04-19T...",
  "fold_boundaries": {...},
  "regime_by_fold": {
    "fold_1": {atr_avg, atr_pct_avg, returns_std_pct, ...},
    ...
  },
  "strategy_by_fold_combo": {
    "baseline": {"fold_1": {...}, "fold_2": {...}, ...},
    "candidate_C": {...}
  },
  "fold_2_sub_windows": {"all_windows":[...], "worst_3":[...], "best_3":[...]},
  "regime_filter_candidates": [...],
  "hypothesis_summary": {
    "H1_low_vol": {...verdict...},
    ...
    "H7_regime_filter_viable": {...}
  },
  "elapsed_sec": ...
}
```

---

## 9. Implementation Order

1. Fold boundary 정의 (constants)
2. `compute_regime_metrics()` 
3. `compute_strategy_metrics()` with set_combo/run_slip
4. `sub_window_microscopy()`
5. `evaluate_regime_filter()` (threshold sweep)
6. `summarize_hypotheses()` with H1~H7 verdict
7. Main orchestration + JSON output

---

## 10. Performance Estimate

- 5 folds × 2 combos × run_slip = 10 runs × 0.15초 ≈ 1.5초
- Sub-window microscopy: ~2초
- Regime filter 12 thresholds: <1초
- 총 **5~10초**

---

## 11. Risks & Mitigations

| Risk | Mitigation |
|------|-----------|
| `set_combo`가 full 기간 run 후 reset 안하면 다음 fold 영향 | 매 호출 후 `reset_combo()` |
| run_slip이 trade list를 cache하지 않음 → 10회 중복 실행 | 동일 combo_cfg는 1회 run 후 fold 필터링 |
| Regime metric이 sample 수 부족으로 noise 클 수 있음 | 최소 30일 window 사용 |
| 해석 주관성 (verdict True/False 임계) | 수치 반올림·임계 Plan 반영 명시 |

---

## 12. Non-Goals

- Regime filter production 적용 (diagnostic만)
- 다른 fold 약점 분석 (본 PDCA는 fold 2 전용)
- 다변량 regime classifier (ML 기반 등)

---

## 13. Reference

- Plan: `docs/01-plan/features/fold2_regime_analysis.plan.md`
- 선행 결과: `results/candidate_c_validation_20260419_151610.json`
- 재사용: `scripts/analysis/c1_intrabar_parity.py`, `intrabar_trail_impact.py`
