# Design: Breakeven Trail

> **Feature**: breakeven_trail
> **Date**: 2026-04-19
> **Phase**: Design
> **Plan**: `docs/01-plan/features/breakeven_trail.plan.md`

---

## 1. Architecture

```
scripts/analysis/breakeven_trail_study.py  (NEW, ~350 lines)
 ├─ 재사용:
 │   ├─ intrabar_trail_impact (data, indicators, run_backtest engine)
 │   └─ c1_intrabar_parity (slippage injection, WF, neighborhood helpers)
 │
 ├─ 신규:
 │   ├─ _check_exit_5m_be()      — 기존 _check_exit_5m을 fork, breakeven guard 추가
 │   ├─ set_breakeven_buffer()   — module-level BUFFER 교체
 │   ├─ run_bt_with_be_and_slip()— engine을 BUFFER-aware로 실행
 │   ├─ run_matrix()             — 2 combos × 5 BUFFER × (clean+slip) = 20 runs
 │   └─ evaluate_go_8flags()     — core 5 + non-core 3 평가
 │
 └─ Output: results/breakeven_trail_{timestamp}.json
```

---

## 2. Core 수식 변경

### 기존 (`_check_exit_5m` from `intrabar_trail_impact.py:259`)
```python
# Trail TP on 5m bar
if best_pnl > trail_act and atr_val > 0:
    trail_dist_pct = tk * atr_val / c5[i5] * 100
    drawdown = best_pnl - cur_pnl
    if drawdown >= trail_dist_pct:
        realized = max(0, best_pnl - trail_dist_pct)   # ← 0으로 clamp
        return {'reason': 'TRAIL_TP', 'exit_price': entry*(1+realized/100)}
```

### 신규 (`_check_exit_5m_be`)
```python
# Trail TP with breakeven guard
if best_pnl > trail_act and atr_val > 0:
    trail_dist_pct = tk * atr_val / c5[i5] * 100
    projected = best_pnl - trail_dist_pct

    # NEW: skip trail if net-loss zone
    if projected < BREAKEVEN_BUFFER:
        continue  # keep iterating sub-bars; fractal SL / emergency handle downside

    drawdown = best_pnl - cur_pnl
    if drawdown >= trail_dist_pct:
        realized = projected  # already >= BUFFER >= 0
        return {'reason': 'TRAIL_TP', 'exit_price': entry*(1+realized/100)}
```

**핵심**: `projected < BUFFER` 체크로 trail이 fee+slip 차감 후 loss 영역에 떨어지는 경우 미발동. SL과 emergency는 정상 작동.

---

## 3. 엔진 Monkey Patch

```python
import scripts.analysis.intrabar_trail_impact as ibt

BREAKEVEN_BUFFER = 0.0  # module-level

def _orig_5m():
    return ibt._check_exit_5m

_ORIG_CHECK_EXIT_5M = _orig_5m()

def _check_exit_5m_be(pos, bar15, tk):
    """Fork of ibt._check_exit_5m with breakeven guard on trail."""
    d, ep, sl = pos['d'], pos['ep'], pos['sl']
    bh = pos['bh']
    start_5m = bar15 * 3
    end_5m = min(start_5m + 3, ibt.n5)
    atr_val = ibt.atr14[bar15]

    for i5 in range(start_5m, end_5m):
        # Update best_price
        if d == 'LONG':
            pos['bp'] = max(pos['bp'], ibt.h5[i5])
        else:
            pos['bp'] = min(pos['bp'], ibt.l5[i5])
        bp = pos['bp']

        # 1. Fractal SL
        if d == 'LONG' and ibt.l5[i5] <= sl:
            return {'reason': 'SL', 'exit_price': sl}
        elif d == 'SHORT' and ibt.h5[i5] >= sl:
            return {'reason': 'SL', 'exit_price': sl}

        # 2. Emergency
        if d == 'LONG':
            worst = (ibt.l5[i5] / ep - 1) * 100
        else:
            worst = (1 - ibt.h5[i5] / ep) * 100
        if worst <= -ibt.emergency_sl:
            exit_p = ep*(1 - ibt.emergency_sl/100) if d=='LONG' \
                     else ep*(1 + ibt.emergency_sl/100)
            return {'reason': 'EMERGENCY', 'exit_price': exit_p}

        # 3. Timeout
        if bh >= ibt.max_hold:
            return {'reason': 'TIMEOUT', 'exit_price': ibt.c5[i5]}

        # 4. Trail TP with breakeven guard
        if d == 'LONG':
            best_pnl = (bp / ep - 1) * 100
            cur_pnl = (ibt.c5[i5] / ep - 1) * 100
        else:
            best_pnl = (1 - bp / ep) * 100
            cur_pnl = (1 - ibt.c5[i5] / ep) * 100

        if best_pnl > ibt.trail_act and not math.isnan(atr_val) and atr_val > 0:
            trail_dist_pct = tk * atr_val / ibt.c5[i5] * 100
            projected = best_pnl - trail_dist_pct

            # Breakeven guard
            if projected < BREAKEVEN_BUFFER:
                continue  # skip trail, let fractal SL or emergency fire

            drawdown = best_pnl - cur_pnl
            if drawdown >= trail_dist_pct:
                realized = projected  # already >= BUFFER
                exit_p = ep*(1+realized/100) if d=='LONG' else ep*(1-realized/100)
                return {'reason': 'TRAIL_TP', 'exit_price': exit_p}

    return None

def set_breakeven_buffer(buffer_pct):
    """Install monkey patch for given buffer value."""
    global BREAKEVEN_BUFFER
    BREAKEVEN_BUFFER = buffer_pct
    ibt._check_exit_5m = _check_exit_5m_be

def reset_breakeven():
    global BREAKEVEN_BUFFER
    BREAKEVEN_BUFFER = 0.0
    ibt._check_exit_5m = _ORIG_CHECK_EXIT_5M
```

---

## 4. 실행 매트릭스 (20 runs)

| Combo | BUFFER | Mode | Run # |
|-------|--------|------|-------|
| baseline | 0.00 | clean / slip_med | 1-2 |
| baseline | 0.10 | clean / slip_med | 3-4 |
| baseline | 0.20 | clean / slip_med | 5-6 |
| baseline | 0.30 | clean / slip_med | 7-8 |
| baseline | 0.40 | clean / slip_med | 9-10 |
| candidate_C | 0.00 | clean / slip_med | 11-12 |
| candidate_C | 0.10 | clean / slip_med | 13-14 |
| candidate_C | 0.20 | clean / slip_med | 15-16 |
| candidate_C | 0.30 | clean / slip_med | 17-18 |
| candidate_C | 0.40 | clean / slip_med | 19-20 |

Top-2 performers에 대해 WF 5-fold + 3-way split + neighborhood 추가.

---

## 5. GO 조건 8-flag (Plan §4 1:1)

```python
def evaluate_go_8flags(base_ref, cand_result):
    """
    base_ref: baseline (BUFFER=0) results — 비교 기준
    cand_result: 평가 대상 (특정 combo × BUFFER)
    """
    f = {}

    # 1. wf_clean_pass (5/5)
    f['wf_clean_pass'] = cand_result['wf_clean_positive_count'] == 5

    # 2. wf_slip_pass (5/5)
    f['wf_slip_pass'] = cand_result['wf_slip_positive_count'] == 5

    # 3. tw_pass: 3-way slip 전 양수
    tw = cand_result['three_way_slip']
    f['tw_pass'] = all(tw[s]['PnL'] > 0 for s in ('train', 'val', 'test'))

    # 4. train_not_degraded: candidate train slip ≥ base train slip − 2pp
    f['train_not_degraded'] = (
        cand_result['three_way_slip']['train']['PnL']
        >= base_ref['three_way_slip']['train']['PnL'] - 2.0
    )

    # 5. pnl_improvement: slip_med PnL ≥ base + 5pp
    f['pnl_improvement'] = (
        cand_result['slip_med']['PnL'] >= base_ref['slip_med']['PnL'] + 5.0
    )

    # 6. ratio_ok: PnL/MDD ≥ baseline × 1.0
    c_r = cand_result['slip_med']['PnL'] / cand_result['slip_med']['MDD'] \
          if cand_result['slip_med']['MDD'] > 0 else 0
    b_r = base_ref['slip_med']['PnL'] / base_ref['slip_med']['MDD'] \
          if base_ref['slip_med']['MDD'] > 0 else 0
    f['ratio_ok'] = c_r >= b_r * 1.0

    # 7. buffer_stable: (외부에서 여러 BUFFER 결과 집계 후 판정)
    f['buffer_stable'] = None  # filled in main()

    # 8. rollback_ready: by design
    f['rollback_ready'] = True

    return f


CORE = ['wf_clean_pass', 'wf_slip_pass', 'tw_pass', 'train_not_degraded',
        'pnl_improvement']

def verdict(flags):
    for c in CORE:
        if not flags.get(c):
            return 'STOP', f'core flag {c} failed'
    true_cnt = sum(1 for k, v in flags.items() if v is True)
    if true_cnt == 8:
        return 'GO', '8/8 flags pass'
    return 'STOP', f'{true_cnt}/8 (need 8/8)'
```

---

## 6. Output Schema

```json
{
  "timestamp": "...",
  "combos": {...},
  "buffer_values": [0.0, 0.1, 0.2, 0.3, 0.4],
  "slippage_used": {...},
  "runs": {
    "baseline_b0.00_clean":     {PnL, MDD, WR, count},
    "baseline_b0.00_slip":      {...},
    ...
    "candidate_C_b0.40_slip":   {...}
  },
  "trail_exit_breakdown": {
    "baseline_b0.00": {
      "trail_total": 954, "trail_loss": 630, "trail_profit": 324,
      "trail_loss_sum_pct": -115.04, "trail_profit_sum_pct": +289.09
    },
    "baseline_b0.20": {...},
    ...
  },
  "top_candidates": [
    {"combo": "baseline", "buffer": 0.20,
     "slip_pnl": ..., "slip_mdd": ...}
  ],
  "top_candidate_full_validation": {
    "wf_clean": [...5 folds],
    "wf_slip": [...5 folds],
    "three_way_clean": {...},
    "three_way_slip": {...}
  },
  "go_flags": {...},
  "buffer_stable_check": {
    "baselines_better_count": 3,  # out of BUFFER > 0
    "passed": true
  },
  "verdict": {"outcome": "GO|STOP", "combo": ..., "buffer": ..., "reason": ...}
}
```

---

## 7. Implementation Order

1. Monkey patch setup (`_check_exit_5m_be`, `set_breakeven_buffer`)
2. Primary matrix 20 runs (2 combos × 5 buffers × 2 modes)
3. Trail exit breakdown (loss vs profit vs avoided)
4. Top performer 선별 (slip_med PnL 최대)
5. Top에 대해 WF + 3-way + neighborhood
6. 8-flag + buffer_stable 평가
7. Verdict + JSON output

---

## 8. Files Touched

### NEW
- `scripts/analysis/breakeven_trail_study.py`
- `results/breakeven_trail_{timestamp}.json`

### READ ONLY
- `scripts/analysis/intrabar_trail_impact.py` (monkey patched but not written)
- `scripts/analysis/c1_intrabar_parity.py`

### CONDITIONAL (GO 시)
- `config/c1_breakout_config.yaml` — `trail_breakeven_buffer: 0.20` 추가
- `scripts/production/c1_breakout/signals.py::check_exit` — breakeven guard
- `CLAUDE.md` — Version History

---

## 9. Performance Estimate

- 20 primary runs × 0.5초 = 10초
- WF × 2 top candidates = 2초
- 3-way × 2 = <1초
- Neighborhood (optional): +5초
- **총 ~20초**

---

## 10. Risks & Mitigations

| Risk | Mitigation |
|------|-----------|
| Monkey patch이 다른 테스트 오염 | `reset_breakeven()` 함수 호출로 항상 복원 |
| BUFFER=0 run이 원본 baseline과 다를 가능성 | Regression check: BUFFER=0 결과가 기존 slip_med +46.09와 일치 |
| Loss trail 회피가 SL 집중으로 악화 | Trail→SL 전환 비율 측정, exit_reason 분포 비교 |
| Buffer_stable 평가가 single BUFFER overfit | 3개 이상 BUFFER에서 개선 요구 |

---

## 11. Regression Check (회귀 확인)

Baseline × BUFFER=0 × slip_med 실행 결과가 기존 intrabar_parity baseline slip_med와 **완전 일치** 해야 함 (+46.09 / MDD 18.78 / WR 30.2 / 1074 trades).

불일치 시 monkey patch 버그.

---

## 12. Non-Goals

- SL/max_hold_bars/Emergency 조정
- Tick-level simulation
- Multi-asset
- LIMIT order 연구

---

## 13. Reference

- Plan: `docs/01-plan/features/breakeven_trail.plan.md`
- 사전 진단: baseline slip_med trail loss 630건 -115.04pp
- 재사용: `scripts/analysis/intrabar_trail_impact.py`, `c1_intrabar_parity.py`, `candidate_c_validation.py`
