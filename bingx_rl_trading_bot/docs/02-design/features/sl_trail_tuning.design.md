# Design: SL/Trail 파라미터 튜닝

> **Feature**: sl_trail_tuning
> **Date**: 2026-04-18
> **Phase**: Design
> **Plan**: `docs/01-plan/features/sl_trail_tuning.plan.md`

---

## 1. Architecture

연구 전용(research-only) 설계. **production 코드 변경 없음** — 기존 validation 인프라를 재사용하여 grid 스크립트만 신규 작성.

```
┌────────────────────────────────────────────────────────────┐
│ scripts/analysis/sl_trail_grid.py  (NEW)                   │
│                                                            │
│  ┌─ main() ─────────────────────────────────────────────┐  │
│  │ 1. 데이터 로딩 (btc_5m → 15m resample)                │  │
│  │ 2. precompute_cache_per_atr_period(cfg)              │  │
│  │ 3. run_grid_train()   → rank_by PnL/MDD, 상위 10     │  │
│  │ 4. run_grid_val()     → top-10 재평가, 상위 3         │  │
│  │ 5. run_grid_test()    → top-3 사후 검증 (재선정 X)    │  │
│  │ 6. run_wf_5fold()     → top-3                         │  │
│  │ 7. run_mc_direction() → top-3 (999 sims)              │  │
│  │ 8. run_bootstrap()    → top-3 (1000 sims)             │  │
│  │ 9. run_neighborhood() → top-3 ±1 step                 │  │
│  │10. write_results_json()                              │  │
│  └──────────────────────────────────────────────────────┘  │
│                                                            │
│  Imports from c1_refined_validation:                       │
│    BASE_CFG, entry_baseline, check_exit,                   │
│    run_bt, summarize, wf_5fold, three_way_split,           │
│    precompute                                              │
│                                                            │
│  Imports from c1_refined_bootstrap_mdd:                    │
│    bootstrap_mdd, compute_mdd_from_trades                  │
│                                                            │
│  Output: results/sl_trail_grid_{timestamp}.json            │
└────────────────────────────────────────────────────────────┘
```

---

## 2. Grid Specification

### Axes (120 combos)
```python
GRID = {
    'max_sl_atr':    [2.8, 3.0, 3.3, 3.6, 4.0, 4.5],   # 6
    'trail_K':       [2.0, 2.2, 2.5, 2.8, 3.0],         # 5
    'max_hold_bars': [96, 144, 192, 288],               # 4
}
# total = 6 * 5 * 4 = 120
```

### Constant (변경 없음)
```python
FIXED = {
    'channel_period': 15, 'body_min_ratio': 0.4, 'atr_period': 14,
    'emergency_sl_pct': 3.0, 'sl_min_pct': 0.15, 'sl_max_pct': 3.0,
    'min_bars_between': 2, 'trail_activation_pct': 0.05,
    'fractal_lookback': 10,
}
```

### Precompute 최적화
`atr_period=14` 고정이므로 **ATR/Channel/Fractal 인디케이터는 1회만 계산**하여 120 combos에서 공유 (`precompute(df15, BASE_CFG)` 재사용).

---

## 3. Selection Protocol (Selection-After-Peek 방지)

### 3.1 데이터 분할 (기존과 동일)
```python
warmup = 50
t1 = warmup + int((n - warmup) * 0.6)   # train end
t2 = warmup + int((n - warmup) * 0.8)   # val end
# train: [warmup, t1]
# val:   [t1+1, t2]
# test:  [t2+1, n-1]
```

### 3.2 단계별 선정
| 단계 | 입력 | 평가 데이터 | 출력 | 주의 |
|------|------|-------------|------|------|
| S1 | 120 combos | **train only** | 상위 10 (PnL/MDD 기준) | val/test 미사용 |
| S2 | top-10 | **val only** | 상위 3 | 재선정 O (val 전용) |
| S3 | top-3 | **test** | 사후 기록 | 재선정 X |
| S4 | top-3 | **전체 (WF)** | PASS/FAIL | expanding window |
| S5 | top-3 | **전체 (MC/BS/NBR)** | 통계 검정 | |

### 3.3 Ranking Metric
Primary: `PnL/MDD`
Tiebreaker: `PnL`

### 3.4 Trade Count Filter (통계 유의성)
| 단계 | 최소 거래 수 | 비율 |
|------|-------------|------|
| Train (~60%) | 300 | baseline ~615 대비 ~49% |
| Val (~20%) | 100 | baseline ~204 대비 ~49% |
| Test (~20%) | 100 | baseline ~208 대비 ~48% |

미달 combo는 해당 단계에서 **제외**(다음 단계로 전파되지 않음).

---

## 4. Validation Methods (top-3 대상)

### 4.1 WF 5-fold (기존 `wf_5fold` 재사용)
PASS 조건: 5 folds 모두 OOS PnL > 0.

### 4.2 MC Direction (신규 함수)
```python
def mc_direction_pvalue(trades, n_sims=999, seed=42):
    """Sign-randomization null. p = P(random_pnl >= actual_pnl)."""
    actual = sum(t['pnl_pct'] for t in trades)
    rng = random.Random(seed)
    pnls = [t['pnl_pct'] for t in trades]
    count = sum(1 for _ in range(n_sims)
                if sum(p if rng.random() < 0.5 else -p for p in pnls) >= actual)
    return (count + 1) / (n_sims + 1)
```
PASS: `p < 0.01`.

### 4.3 Bootstrap CI (**신규 함수 — bootstrap_mdd는 MDD만 반환**)

기존 `bootstrap_mdd`는 MDD 분포만 반환(ci_low/ci_high = MDD CI). PnL CI를 얻으려면 **신규 `bootstrap_pnl()`** 필요. 동일 stationary block 인프라(`_stationary_bootstrap_indices`) 재사용.

```python
# scripts/analysis/sl_trail_grid.py 내 정의
from statistics import mean, median, stdev
import random
from scripts.analysis.c1_refined_bootstrap_mdd import _stationary_bootstrap_indices

def bootstrap_pnl(trades, n_boot=1000, seed=42, mean_block_len=20):
    """Stationary block bootstrap for additive PnL distribution."""
    rng = random.Random(seed)
    n = len(trades)
    if n == 0:
        return {'obs_pnl': 0, 'pnl_ci_lo': 0, 'pnl_ci_hi': 0}
    obs = sum(t['pnl_pct'] for t in trades)
    pnls = []
    for _ in range(n_boot):
        idx = _stationary_bootstrap_indices(n, mean_block_len, rng)
        pnls.append(sum(trades[j]['pnl_pct'] for j in idx))
    pnls.sort()
    return {
        'obs_pnl':   round(obs, 3),
        'pnl_mean':  round(mean(pnls), 3),
        'pnl_ci_lo': round(pnls[int(0.025 * n_boot)], 3),
        'pnl_ci_hi': round(pnls[int(0.975 * n_boot)], 3),
        'method':    f'stationary_bootstrap(mean_block={mean_block_len})',
    }
```

MDD CI도 함께 필요하므로 `bootstrap_mdd`도 동시에 호출하여 두 결과를 `r['bootstrap']` 아래 병합.
PASS: `pnl_ci_lo > 0`.

### 4.4 Parameter Neighborhood

최적 combo의 3D 좌표 인덱스를 `(i_sl, i_K, i_bars)`라 할 때, 6개 인접 축방향 이웃(각 축에서 ±1)을 대상으로 함:
```python
AXES = list(GRID.items())  # [('max_sl_atr',[...]), ('trail_K',[...]), ('max_hold_bars',[...])]

def neighbors_of(combo):
    """6개 축방향 ±1 이웃 (가장자리는 가능한 이웃만)."""
    out = []
    for axis_name, values in AXES:
        cur = values.index(combo[axis_name])
        for di in (-1, +1):
            j = cur + di
            if 0 <= j < len(values):
                nb = dict(combo)
                nb[axis_name] = values[j]
                out.append(nb)
    return out  # 최소 3개 (corner), 최대 6개 (interior)
```

각 이웃은 **전체 기간(full)**에서 평가. Positive 정의: `summarize(...).PnL > 0`.
PASS: `positive_count >= ceil(0.75 × total_neighbors)` → interior(6개) 시 ≥5, edge(5개) 시 ≥4, corner(3개) 시 ≥3.
(Plan §4.5의 "6/8"은 3×3×3 큐브 가정이었으나 실제 ±1 축방향 이웃은 최대 6개이므로 기준 조정)

### 4.5 3-way Split (기존 `three_way_split`)
PASS: train/val/test 모두 PnL > 0, 그리고 `test_new.PnL >= baseline_test.PnL - 5.0`.

---

## 5. Baseline 기준점

기존 결과(`c1_refined_variants.json` BASELINE) 재사용:
```
BASELINE = {
  'full':  {'trades': 1027, 'WR': 36.6, 'PnL': 170.49, 'MDD': 5.38},
  '3way':  {'train': +95.07, 'val': +21.21, 'test': +54.20},
  'wf_passed': 5, 'wf_total_oos': 153.5,
  'pnl_mdd_ratio': 170.49 / 5.38,   # = 31.69
}
```
GO 기준 `PnL/MDD >= 31.69 * 1.10 = 34.86`.

---

## 6. Output Schema

```json
{
  "timestamp": "2026-04-18T...",
  "baseline": { ... },
  "grid_config": { "axes": {...}, "fixed": {...} },
  "train_ranking": [
    {"combo": {"max_sl_atr": 3.6, "trail_K": 2.5, "max_hold_bars": 192},
     "train": {"trades":..., "WR":..., "PnL":..., "MDD":..., "ratio":...}},
    ... (top 10)
  ],
  "val_ranking": [... (top 3 from top 10)],
  "test_verification": [... (top 3, no re-rank)],
  "wf_results":        [{"combo":..., "folds":[...], "passed":5, "total_oos":...}],
  "mc_pvalues":        [{"combo":..., "p": 0.001}],
  "bootstrap_ci":      [{"combo":..., "pnl_lo":..., "pnl_hi":...}],
  "neighborhoods":     [{"combo":..., "positive":7, "total":8}],
  "verdict": {
    "selected": {"max_sl_atr":..., "trail_K":..., "max_hold_bars":...}
                 | null,
    "reason": "GO|STOP: <rationale>",
    "go_conditions": {
      "pnl_mdd_improved":   true|false,
      "wf_5of5":            true|false,
      "three_way_all_pos":  true|false,
      "test_not_worse_5p":  true|false,
      "mc_p_lt_001":        true|false,
      "neighborhood_6of8":  true|false,
      "bootstrap_ci_pos":   true|false
    }
  }
}
```

---

## 7. Script Skeleton (`scripts/analysis/sl_trail_grid.py`)

```python
"""SL/Trail 3D parameter grid tuning for C1 Breakout v2.6."""
import sys, os, json, math, random, copy
from datetime import datetime
from itertools import product
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

import pandas as pd
from scripts.analysis.c1_refined_validation import (
    BASE_CFG, entry_baseline, run_bt, summarize, wf_5fold,
    three_way_split, precompute, FEE_RT_PCT,
)
from scripts.analysis.c1_refined_bootstrap_mdd import (
    bootstrap_mdd, compute_mdd_from_trades,
)

GRID = {
    'max_sl_atr':    [2.8, 3.0, 3.3, 3.6, 4.0, 4.5],
    'trail_K':       [2.0, 2.2, 2.5, 2.8, 3.0],
    'max_hold_bars': [96, 144, 192, 288],
}
MIN_TRADES_TRAIN = 300

def load_data():
    df5 = pd.read_csv(ROOT / 'data' / 'btc_5m_270days_reclassified.csv')
    df5['timestamp'] = pd.to_datetime(df5['timestamp'])
    df5 = df5.sort_values('timestamp').set_index('timestamp')
    df15 = df5.resample('15min').agg(
        {'open':'first','high':'max','low':'min','close':'last'}
    ).dropna().reset_index()
    return df15

def make_cfg(combo):
    cfg = copy.deepcopy(BASE_CFG)
    cfg.update(combo)
    return cfg

def eval_combo(df15, cfg, pre, start, end):
    trades = run_bt(df15, cfg, entry_baseline, start, end, **pre)
    s = summarize(trades)
    s['ratio'] = s['PnL'] / s['MDD'] if s['MDD'] > 0 else 0
    return s, trades

def train_rank(df15, pre, t1):
    results = []
    for sl, k, bars in product(*GRID.values()):
        combo = {'max_sl_atr': sl, 'trail_K': k, 'max_hold_bars': bars}
        cfg = make_cfg(combo)
        s, _ = eval_combo(df15, cfg, pre, 50, t1)
        if s['trades'] < MIN_TRADES_TRAIN: continue
        results.append({'combo': combo, 'train': s})
    results.sort(key=lambda r: (r['train']['ratio'], r['train']['PnL']),
                 reverse=True)
    return results[:10]

def val_rerank(df15, pre, t1, t2, top10):
    for r in top10:
        cfg = make_cfg(r['combo'])
        s, _ = eval_combo(df15, cfg, pre, t1+1, t2)
        r['val'] = s
    top10.sort(key=lambda r: (r['val']['ratio'], r['val']['PnL']),
               reverse=True)
    return top10[:3]

def test_verify(df15, pre, t2, n, top3):
    for r in top3:
        cfg = make_cfg(r['combo'])
        s, _ = eval_combo(df15, cfg, pre, t2+1, n-1)
        r['test'] = s
    return top3

def mc_direction(trades, n_sims=999, seed=42):
    actual = sum(t['pnl_pct'] for t in trades)
    rng = random.Random(seed)
    pnls = [t['pnl_pct'] for t in trades]
    count = sum(1 for _ in range(n_sims)
                if sum((p if rng.random() < 0.5 else -p) for p in pnls)
                   >= actual)
    return (count + 1) / (n_sims + 1)

def neighborhood(top3):
    # returns { combo_tuple: (positive_count, total_count) }
    ...

def main():
    df15 = load_data()
    pre = precompute(df15, BASE_CFG)
    n = len(df15)
    t1 = 50 + int((n - 50) * 0.6)
    t2 = 50 + int((n - 50) * 0.8)

    top10 = train_rank(df15, pre, t1)
    top3  = val_rerank(df15, pre, t1, t2, top10)
    top3  = test_verify(df15, pre, t2, n, top3)

    # Full-dataset validation on top-3
    for r in top3:
        cfg = make_cfg(r['combo'])
        full_t = run_bt(df15, cfg, entry_baseline, 50, n-1, **pre)
        r['full']      = summarize(full_t)
        r['wf_oos']    = wf_5fold(df15, cfg, entry_baseline, pre)
        r['three_way'] = three_way_split(df15, cfg, entry_baseline, pre)
        r['mc_p']      = mc_direction(full_t)
        r['bootstrap'] = bootstrap_mdd(full_t, n_boot=1000,
                                        seed=42, mean_block_len=20)

    # Neighborhood
    nbh = neighborhood(top3, df15, pre, n)

    # Verdict
    verdict = decide_verdict(top3, nbh, BASELINE_RATIO=31.69)

    out = { ... }
    out_path = ROOT / 'results' / \
               f'sl_trail_grid_{datetime.now():%Y%m%d_%H%M%S}.json'
    out_path.write_text(json.dumps(out, indent=2, default=str))
    print(f'Written: {out_path}')

if __name__ == '__main__':
    main()
```

---

## 8. Performance Estimate

단위: full_time = 전체 기간 1회 `run_bt` 실행 시간.

| 단계 | 호출 수 | 슬라이스 | full 환산 |
|------|---------|----------|-----------|
| Train grid | 120 | 0.6 × full | 72 |
| Val rerank | 10 | 0.2 × full | 2 |
| Test verify | 3 | 0.2 × full | 0.6 |
| Full on top-3 | 3 | 1.0 × full | 3 |
| WF 5-fold × top-3 | 15 | ~0.167 × full each | 2.5 |
| MC direction | 3 (no run_bt) | — | ~0 (shuffle only) |
| Bootstrap PnL+MDD | 3 × 2 × 1000 | resample only | ~3 (trade list) |
| Neighborhood | top-3 × 최대 6 | 1.0 × full | **≤ 18** |
| **합계** | | | **~100** |

`extended_param_grid.py`가 1D 35run에 수 초 소요 → **수 분 이내** 예상. 스모크 테스트(§11)로 5분 초과 시 조기 중단.

---

## 9. GO/STOP Decision Logic

```python
def decide_verdict(top3, nbh, BASELINE_RATIO, BASELINE_TEST_PNL=54.20):
    """
    GO 조건 7개 (Plan §2 확장 — test-not-worse는 3-way에서 분리하여 가시화):
      (1) ratio_ok:   PnL/MDD >= BASELINE_RATIO * 1.10
      (2) wf_pass:    WF 5 folds all OOS PnL > 0
      (3) tw_pass:    train/val/test 모두 PnL > 0
      (4) test_ok:    test PnL >= baseline_test_PnL - 5.0 pp
      (5) mc_pass:    MC direction p < 0.01
      (6) ci_pass:    bootstrap PnL 95% CI lower > 0
      (7) nbr_pass:   neighborhood positive ratio >= 0.75
    """
    for r in top3:
        ratio     = r['full']['PnL'] / r['full']['MDD'] if r['full']['MDD'] > 0 else 0
        wf_pass   = all(f['PnL'] > 0 for f in r['wf_oos'])
        tw_pass   = all(r['three_way'][s]['PnL'] > 0
                        for s in ['train','val','test'])
        test_ok   = r['three_way']['test']['PnL'] >= BASELINE_TEST_PNL - 5.0
        mc_pass   = r['mc_p'] < 0.01
        ci_pass   = r['bootstrap']['pnl_ci_lo'] > 0
        pos, tot  = nbh[tuple(r['combo'].values())]
        nbr_pass  = pos >= math.ceil(0.75 * tot)
        ratio_ok  = ratio >= BASELINE_RATIO * 1.10

        flags = {
            'ratio_ok': ratio_ok, 'wf_pass': wf_pass, 'tw_pass': tw_pass,
            'test_ok': test_ok, 'mc_pass': mc_pass, 'ci_pass': ci_pass,
            'nbr_pass': nbr_pass,
        }
        if all(flags.values()):
            return {'outcome': 'GO', 'combo': r['combo'], 'flags': flags}
    return {'outcome': 'STOP', 'combo': None,
            'reasons': 'no top-3 candidate met all 7 GO conditions',
            'last_flags': flags}
```

> **Plan 반영**: Plan §2-6개 조건 중 "test PnL 하락 ≤ 5%p"를 3-way에서 분리한 가시화 플래그(`test_ok`)로 만들어 총 7개. Plan 문서 §2를 7개 조건으로 동기화 필요.

---

## 10. Files Touched

### NEW
- `scripts/analysis/sl_trail_grid.py`

### READ ONLY (재사용)
- `scripts/analysis/c1_refined_validation.py`
- `scripts/analysis/c1_refined_bootstrap_mdd.py`
- `scripts/production/c1_breakout/indicators.py`
- `data/btc_5m_270days_reclassified.csv`

### WRITE (결과)
- `results/sl_trail_grid_{timestamp}.json`

### CONDITIONAL (GO 판정 시에만)
- `config/c1_breakout_config.yaml` — 3개 파라미터 값만 교체
- `CLAUDE.md` — Version History + 빠른 참조
- `claudedocs/c1_breakout_v2_design.md` — 검증 수치 갱신

---

## 11. Testing

### 스모크 테스트 (grid 축소판)
```python
# 개발 중에는 GRID를 2×2×2로 축소하여 흐름 검증
GRID_SMOKE = {
    'max_sl_atr':    [3.3, 4.0],
    'trail_K':       [2.5, 2.8],
    'max_hold_bars': [192, 288],
}
```

### 회귀 확인
Baseline combo `(3.3, 2.5, 192)`가 grid에 포함되어 있으므로, 그 결과를 `c1_refined_variants.json`의 BASELINE과 비교하여 ±0.5% 오차 이내인지 확인 (인프라 버그 탐지).

---

## 12. Non-Goals

- entry 로직 변경 (channel/body_min 등) — Plan §6에서 제외
- exchange SL/TP 배치 방식 변경
- risk halt 도입
- per-regime 조건부 파라미터
- 다중 자산 적용
