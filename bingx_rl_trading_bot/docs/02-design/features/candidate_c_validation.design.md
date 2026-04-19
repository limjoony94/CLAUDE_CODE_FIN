# Design: Candidate_C Validation

> **Feature**: candidate_c_validation
> **Date**: 2026-04-19
> **Phase**: Design
> **Plan**: `docs/01-plan/features/candidate_c_validation.plan.md`

---

## 1. Architecture

```
scripts/analysis/candidate_c_validation.py  (NEW, ~400 lines)
 ├─ 재사용:
 │   ├─ c1_refined_validation.BASE_CFG, entry_baseline, run_bt, summarize,
 │   │                          wf_5fold, three_way_split, precompute
 │   ├─ c1_refined_bootstrap_mdd.bootstrap_mdd, _stationary_bootstrap_indices
 │   └─ c1_intrabar_parity.run_bt_with_slippage, apply_slippage,
 │                          wf_on_adjusted_trades, compute_mdd_additive,
 │                          SLIPPAGE (as SLIPPAGE_MED base)
 │
 ├─ 신규:
 │   ├─ SLIPPAGE_LOW / SLIPPAGE_MED / SLIPPAGE_HIGH (3 시나리오)
 │   ├─ bootstrap_pnl_on_trades() — 기존 intrabar_parity의 bootstrap_mdd 병행
 │   ├─ mc_direction_pvalue()
 │   ├─ neighborhood_eval() (±1 step, 6-neighbor)
 │   └─ evaluate_go_9flags() (Plan §4 9-flag)
 │
 └─ Output: results/candidate_c_validation_{stamp}.json
```

**전략**: baseline (3.3, 2.5, 192)과 candidate_C (4.0, 2.5, 192)을 동일 파이프라인으로 평가 → **단일 변수 max_sl_atr 효과 순수 측정**.

---

## 2. Combos & Slippage Matrix

### 2.1 Combos (단일 파라미터 차이)
```python
COMBOS = {
    'baseline':     {'max_sl_atr': 3.3, 'trail_K': 2.5, 'max_hold_bars': 192},
    'candidate_C':  {'max_sl_atr': 4.0, 'trail_K': 2.5, 'max_hold_bars': 192},
}
```

### 2.2 Slippage 시나리오 3개
```python
SLIPPAGE_LOW  = {'entry_pct':0.025,'exit_sl_pct':0.075,'exit_trail_pct':0.025,
                 'exit_emergency_pct':0.15,'exit_timeout_pct':0.025}
SLIPPAGE_MED  = {'entry_pct':0.05, 'exit_sl_pct':0.15, 'exit_trail_pct':0.05,
                 'exit_emergency_pct':0.30,'exit_timeout_pct':0.05}
SLIPPAGE_HIGH = {'entry_pct':0.10, 'exit_sl_pct':0.30, 'exit_trail_pct':0.10,
                 'exit_emergency_pct':0.60,'exit_timeout_pct':0.10}
```

### 2.3 실행 매트릭스 (8 run + validation)
| # | Combo | BT mode | Slippage | 용도 |
|---|-------|---------|----------|------|
| 1 | baseline | bar_close | clean | Clean 기준 |
| 2 | candidate_C | bar_close | clean | Clean 비교 |
| 3 | baseline | 5m | LOW | Slip sensitivity |
| 4 | candidate_C | 5m | LOW | Slip sensitivity |
| 5 | baseline | 5m | MED | Primary 평가 |
| 6 | candidate_C | 5m | MED | Primary 평가 |
| 7 | baseline | 5m | HIGH | Slip sensitivity |
| 8 | candidate_C | 5m | HIGH | Slip sensitivity |

+ WF/MC/Bootstrap/Neighborhood는 MED 시나리오에서 candidate_C만 수행.

---

## 3. 9-flag GO Conditions (Plan §4 1:1 매핑)

```python
def evaluate_go_9flags(baseline_results, cand_results):
    """
    baseline_results / cand_results: dict with keys
      clean, slip_low, slip_med, slip_high, wf_clean, wf_slip,
      three_way_clean, three_way_slip, mc_p, bootstrap, neighborhood
    """
    f = {}

    # 1. wf_clean_pass — candidate clean BT WF 5/5
    f['wf_clean_pass'] = sum(1 for fold in cand_results['wf_clean']
                              if fold['PnL'] > 0) == 5

    # 2. wf_slip_pass — candidate slip MED BT WF 5/5
    f['wf_slip_pass'] = cand_results['wf_slip_positive_count'] == 5

    # 3. tw_pass — 3-way train/val/test 모두 양수 (slip 기준)
    tw = cand_results['three_way_slip']
    f['tw_pass'] = all(tw[s]['PnL'] > 0 for s in ('train', 'val', 'test'))

    # 4. test_not_worse — clean AND slip 둘 다 만족
    base_test_clean = baseline_results['three_way_clean']['test']['PnL']
    cand_test_clean = cand_results['three_way_clean']['test']['PnL']
    base_test_slip  = baseline_results['three_way_slip']['test']['PnL']
    cand_test_slip  = cand_results['three_way_slip']['test']['PnL']
    f['test_not_worse'] = (
        cand_test_clean >= base_test_clean - 5.0 and
        cand_test_slip  >= base_test_slip  - 5.0
    )

    # 5. nbr_pass — 6-neighbor ≥ 75% positive (= ≥5)
    f['nbr_pass'] = cand_results['neighborhood']['positive_count'] >= 5

    # 6. mc_pass — MC p < 0.01
    f['mc_pass'] = cand_results['mc_p'] < 0.01

    # 7. ci_pass — Bootstrap PnL CI lower > 0
    f['ci_pass'] = cand_results['bootstrap']['pnl_ci_lo'] > 0

    # 8. train_not_degraded — candidate train slip ≥ baseline train slip − 2pp
    f['train_not_degraded'] = (
        cand_results['three_way_slip']['train']['PnL']
        >= baseline_results['three_way_slip']['train']['PnL'] - 2.0
    )

    # 9. slip_sensitivity — LOW/MED/HIGH 세 시나리오 모두 PnL/MDD 우위
    all_wins = True
    for s in ('slip_low', 'slip_med', 'slip_high'):
        b_r = baseline_results[s]['PnL'] / baseline_results[s]['MDD'] \
              if baseline_results[s]['MDD'] > 0 else 0
        c_r = cand_results[s]['PnL'] / cand_results[s]['MDD'] \
              if cand_results[s]['MDD'] > 0 else 0
        if c_r <= b_r:
            all_wins = False
            break
    f['slip_sensitivity'] = all_wins

    return f
```

### Verdict
```python
CORE = ['wf_clean_pass', 'wf_slip_pass', 'tw_pass', 'train_not_degraded',
        'slip_sensitivity']

def verdict(flags):
    for c in CORE:
        if not flags[c]:
            return 'STOP', f'core flag {c} failed'
    total = sum(1 for v in flags.values() if v)
    if total == 9:
        return 'GO', 'all 9 flags pass'
    return 'STOP', f'{total}/9 (need 9/9)'
```

**9/9 전부 PASS만 GO**. 1개라도 fail 시 STOP → baseline 유지.

---

## 4. Neighborhood 정의

Candidate_C `(4.0, 2.5, 192)` 기준 6개 축방향 ±1 이웃:

```python
AXES = {
    'max_sl_atr':    [2.8, 3.0, 3.3, 3.6, 4.0, 4.5],  # 4.0 이웃: 3.6, 4.5
    'trail_K':       [2.0, 2.2, 2.5, 2.8, 3.0],       # 2.5 이웃: 2.2, 2.8
    'max_hold_bars': [96, 144, 192, 288],              # 192 이웃: 144, 288
}
# Candidate_C neighbors (6개): 
#   (3.6,2.5,192), (4.5,2.5,192),
#   (4.0,2.2,192), (4.0,2.8,192),
#   (4.0,2.5,144), (4.0,2.5,288)
```

각 이웃을 `5m+SLIP_MED`로 평가하여 PnL > 0 개수 카운트. 기준: ≥5/6 (75%+).

---

## 5. Script Skeleton

```python
#!/usr/bin/env python3
"""Candidate_C (4.0, 2.5, 192) validation — 9-flag GO protocol."""
import sys, os, json, math, copy, random
from pathlib import Path
from datetime import datetime, timezone
from statistics import mean, median, stdev
from itertools import product

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT)); os.chdir(ROOT)

import pandas as pd

from scripts.analysis.c1_refined_validation import (
    BASE_CFG, entry_baseline, run_bt, summarize,
    wf_5fold, three_way_split, precompute, FEE_RT_PCT,
)
from scripts.analysis.c1_refined_bootstrap_mdd import (
    _stationary_bootstrap_indices, bootstrap_mdd,
)
import scripts.analysis.intrabar_trail_impact as ibt
from scripts.analysis.c1_intrabar_parity import (
    run_bt_with_slippage, apply_slippage, wf_on_adjusted_trades,
    compute_mdd_additive, set_combo, reset_combo,
)

COMBOS = {
    'baseline':     {'max_sl_atr': 3.3, 'trail_K': 2.5, 'max_hold_bars': 192},
    'candidate_C':  {'max_sl_atr': 4.0, 'trail_K': 2.5, 'max_hold_bars': 192},
}

SLIP_SCENARIOS = {
    'low':  dict(entry_pct=0.025, exit_sl_pct=0.075, exit_trail_pct=0.025,
                 exit_emergency_pct=0.15, exit_timeout_pct=0.025),
    'med':  dict(entry_pct=0.05,  exit_sl_pct=0.15,  exit_trail_pct=0.05,
                 exit_emergency_pct=0.30, exit_timeout_pct=0.05),
    'high': dict(entry_pct=0.10,  exit_sl_pct=0.30,  exit_trail_pct=0.10,
                 exit_emergency_pct=0.60, exit_timeout_pct=0.10),
}

def load_15m():
    df5 = pd.read_csv(ROOT / 'data' / 'btc_5m_270days_reclassified.csv')
    df5['timestamp'] = pd.to_datetime(df5['timestamp'])
    df5 = df5.sort_values('timestamp').set_index('timestamp')
    df15 = df5.resample('15min').agg(
        {'open':'first','high':'max','low':'min','close':'last'}
    ).dropna().reset_index()
    return df15

def bootstrap_pnl_on_trades(trades, n_boot=1000, seed=42, mean_block_len=20):
    rng = random.Random(seed)
    n = len(trades)
    if n == 0:
        return {'obs_pnl':0, 'pnl_ci_lo':0, 'pnl_ci_hi':0}
    obs = sum(t['net'] for t in trades)
    vals = []
    for _ in range(n_boot):
        idx = _stationary_bootstrap_indices(n, mean_block_len, rng)
        vals.append(sum(trades[j]['net'] for j in idx))
    vals.sort()
    return {'obs_pnl': round(obs, 3),
            'pnl_ci_lo': round(vals[int(0.025*n_boot)], 3),
            'pnl_ci_hi': round(vals[int(0.975*n_boot)], 3)}

def mc_direction_pvalue(trades, n_sims=999, seed=42):
    if not trades: return 1.0
    actual = sum(t['net'] for t in trades)
    rng = random.Random(seed)
    pnls = [t['net'] for t in trades]
    cnt = sum(1 for _ in range(n_sims)
              if sum((p if rng.random()<0.5 else -p) for p in pnls) >= actual)
    return (cnt + 1) / (n_sims + 1)

def evaluate_single_combo(combo_name, combo_cfg, slip, mode='5m'):
    """combo × slippage 1회 평가. Returns summary dict."""
    set_combo(**combo_cfg)
    if slip == 'clean':
        trades = ibt.run_backtest(mode=mode if mode != 'bar_close' else 'bar_close')
        pnl = sum(t['net'] for t in trades)
    else:
        # swap SLIPPAGE then restore
        import scripts.analysis.c1_intrabar_parity as cip
        old_slip = cip.SLIPPAGE
        cip.SLIPPAGE = SLIP_SCENARIOS[slip]
        trades = run_bt_with_slippage(mode=mode)
        cip.SLIPPAGE = old_slip
        pnl = sum(t['net'] for t in trades)
    mdd = compute_mdd_additive(trades) if slip != 'clean' \
          else compute_mdd_additive(trades)  # trades have 'net' either way
    wr = sum(1 for t in trades if t['net'] > 0) / len(trades) * 100 if trades else 0
    reset_combo()
    return {'trades': trades, 'PnL': round(pnl,2), 'MDD': round(mdd,2),
            'WR': round(wr,1), 'count': len(trades)}

def three_way_on_trades(trades, boundary_train, boundary_val):
    """Partition trades by entry_bar → train/val/test."""
    train = [t for t in trades if t['entry_bar'] <= boundary_train]
    val   = [t for t in trades if boundary_train < t['entry_bar'] <= boundary_val]
    test  = [t for t in trades if t['entry_bar'] > boundary_val]
    def summarize_slice(ts):
        if not ts: return {'PnL':0, 'MDD':0, 'count':0}
        pnl = sum(t['net'] for t in ts)
        return {'PnL':round(pnl,2), 'MDD':round(compute_mdd_additive(ts),2),
                'count':len(ts)}
    return {'train':summarize_slice(train),
            'val':summarize_slice(val),
            'test':summarize_slice(test)}

def neighborhood_eval(combo, slip_scenario):
    AXES = {
        'max_sl_atr':    [2.8, 3.0, 3.3, 3.6, 4.0, 4.5],
        'trail_K':       [2.0, 2.2, 2.5, 2.8, 3.0],
        'max_hold_bars': [96, 144, 192, 288],
    }
    nbs = []
    for ax, vals in AXES.items():
        cur_i = vals.index(combo[ax])
        for di in (-1, +1):
            j = cur_i + di
            if 0 <= j < len(vals):
                nb = dict(combo); nb[ax] = vals[j]
                nbs.append(nb)
    results = []
    pos = 0
    for nb in nbs:
        r = evaluate_single_combo(f'nbr_{nb}', nb, slip_scenario)
        results.append({'combo':nb, 'PnL':r['PnL'], 'MDD':r['MDD']})
        if r['PnL'] > 0: pos += 1
    return {'neighbors':results, 'positive_count':pos, 'total':len(nbs)}

def main():
    # ... assemble all 8 runs + WF/MC/bootstrap/3way/neighborhood
    # ... evaluate_go_9flags + verdict
    # ... write results JSON

if __name__ == '__main__':
    main()
```

---

## 6. Output Schema

```json
{
  "timestamp": "2026-04-19T...",
  "combos": {...},
  "slippage_scenarios": {...},
  "runs": {
    "baseline_clean":     {"PnL":..., "MDD":..., "WR":..., "count":...},
    "baseline_slip_low":  {...},
    "baseline_slip_med":  {...},
    "baseline_slip_high": {...},
    "candidate_C_clean":     {...},
    "candidate_C_slip_low":  {...},
    "candidate_C_slip_med":  {...},
    "candidate_C_slip_high": {...}
  },
  "candidate_C_validation": {
    "wf_clean":  [...5 folds],
    "wf_slip_med": {"positive_count":..., "folds":[...]},
    "three_way_clean": {...},
    "three_way_slip":  {...},
    "mc_p": 0.001,
    "bootstrap": {"obs_pnl":..., "pnl_ci_lo":..., "pnl_ci_hi":...},
    "neighborhood": {"neighbors":[...], "positive_count":5, "total":6}
  },
  "comparison": {
    "pnl_mdd_ratio_by_slip": {
      "low":  {"baseline":..., "candidate_C":..., "winner":...},
      "med":  {...}, "high": {...}
    }
  },
  "go_flags": {...9 keys...},
  "verdict": {"outcome":"GO|STOP", "reason":"..."}
}
```

---

## 7. Files Touched

### NEW
- `scripts/analysis/candidate_c_validation.py` (~400 lines)
- `results/candidate_c_validation_{timestamp}.json`

### READ ONLY 재사용
- `scripts/analysis/c1_refined_validation.py`
- `scripts/analysis/c1_refined_bootstrap_mdd.py`
- `scripts/analysis/c1_intrabar_parity.py`
- `scripts/analysis/intrabar_trail_impact.py`
- `data/btc_5m_270days_reclassified.csv`

### CONDITIONAL (GO 판정 후에만)
- `config/c1_breakout_config.yaml` — `max_sl_atr: 3.3 → 4.0` 단일 변경
- `CLAUDE.md` — Version History + 빠른 참조

---

## 8. Performance Estimate

| 단계 | 예상 |
|------|------|
| 8 primary runs (full 332일) | ~16초 |
| WF (candidate clean + slip_med) | ~10초 |
| 3-way split × 2 | ~2초 |
| MC 999 sims | ~1초 |
| Bootstrap PnL+MDD 1000×2 | ~2초 |
| Neighborhood 6 combos × 5m slip | ~12초 |
| **합계** | **~45초** |

---

## 9. Testing / Regression

### 회귀 확인
Baseline clean bar_close PnL ≈ +170.49% (±0.5%) — 기존 sl_trail_tuning 결과 매칭.
Baseline slip_med PnL ≈ +46.09% (intrabar_parity 결과 매칭).

### 단위 확인
- 9-flag 평가 함수: 모든 flag가 True인 fake input → verdict GO
- 1개라도 False → verdict STOP + failed flag 명시

---

## 10. Risks & Mitigations

| Risk | Mitigation |
|------|-----------|
| Slippage 시나리오 임의성 | 3 시나리오 모두 우위 요구 (보수적) |
| Single-combo overfit | Neighborhood + MC + WF × clean/slip 이중 검증 |
| Data OOS (04-03까지) | GO 판정 시 30일 LIVE 재확인 필수 명시 |
| Production risk | Single-param change (max_sl_atr), 즉시 rollback 가능 |

---

## 11. Decision Protocol

```
Plan → Design(이 문서) → Do(실행) → Check(결과+gap)
  → GO(9/9)   → Report + config 변경 제안 + 30일 LIVE 재확인 대기
  → STOP      → Report + 교훈 영구화 + baseline 유지
```

Production 변경은 **9/9 PASS + 사용자 승인**의 2중 조건.

---

## 12. Reference

- Plan: `docs/01-plan/features/candidate_c_validation.plan.md`
- Baseline 데이터: `results/c1_refined_variants.json`
- intrabar 엔진: `scripts/analysis/c1_intrabar_parity.py`
- 표준 규칙: `memory/research_protocol_overfit_guards.md`
