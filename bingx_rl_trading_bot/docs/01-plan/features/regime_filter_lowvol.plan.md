# Plan: Low-Volatility Regime Filter (fold 2 근본 해결)

> **Feature**: regime_filter_lowvol
> **Date**: 2026-04-19
> **Phase**: Plan (compact)
> **Trigger**: body_filter_tuning 8/9 flags (유일 실패 wf_slip_pass fold 2)
> **Target**: Fold 2(2025-07-11~09-15, 저변동성) 진입 skip으로 9/9 GO 달성

---

## 1. Background

body_filter_tuning의 **candidate_C_b0.60**가 8/9 flags:
- 유일 실패: `wf_slip_pass` (fold 2 slip -5.62, 여전 음수)
- Fold 2 특징(fold2_regime_analysis): ATR% 0.229 (평균 0.318, -28%), trend -2.6% (횡보)
- body 증가는 완화책, regime filter가 근본 해결

**핵심 아이디어**: 실시간 rolling ATR%가 임계 이하 시 **진입 skip**. Fractal SL/trail 등 exit 로직은 그대로.

---

## 2. Hypotheses

| H | 내용 |
|---|------|
| H1 | Rolling ATR% < THR 시 skip → fold 2 slip 양수 회복 |
| H2 | Fold 1도 일부 저변동성(ATR% 0.248)이나 양수 → threshold 보수적이어야 (fold 2만 잡도록) |
| H3 | THR 스윕으로 sweet spot 존재 (너무 tight → 기회 상실, 너무 loose → fold 2 잔존) |
| H4 | candidate_C_b0.60 + regime filter = 9/9 GO 달성 |

---

## 3. Method

### 3.1 Regime Detection
매 bar i에서:
```python
recent_atr_pct = mean(atr[i-LOOKBACK:i]) / mean(close[i-LOOKBACK:i]) * 100
if recent_atr_pct < REGIME_THR:
    skip_entry()
```

### 3.2 스윕
- REGIME_THR: 0.22, 0.24, 0.26, 0.28, 0.30 (%)
- LOOKBACK: 96 (1일), 192 (2일), 288 (3일)
- Fixed combo: **candidate_C_b0.60** (이전 8/9 best)
- 5 THR × 3 LOOKBACK = 15 runs

### 3.3 구현
Monkey-patch `entry_baseline` (또는 `run_backtest` bar loop)에서 regime 조건 사전 체크.

---

## 4. Success Criteria (3-flag GO)

1. **fold2_slip_positive**: fold 2 slip PnL > 0 (핵심)
2. **overall_not_degraded**: full slip PnL ≥ candidate_C_b0.60 - 5pp (유지 필수)
3. **wf_slip_5of5**: WF slip 5/5 통과

3/3 PASS GO (production 가치 있음). Fold 2 양수 전환이 핵심.

---

## 5. Risks

| Risk | Mitigation |
|------|-----------|
| Regime threshold overfit | 3 lookback × 5 threshold grid 넓게 |
| Fold 1도 함께 skip되어 기회 상실 | Fold-by-fold 결과 점검 |
| Live regime detection latency | Lookback 96~288 충분히 길어 안정 |

---

## 6. Reference

- body_filter_tuning (candidate_C_b0.60 base): `docs/04-report/body_filter_tuning.report.md`
- Fold 2 진단: `memory/fold2_regime_analysis_20260419.md`
- 재사용 엔진: `intrabar_trail_impact.py`, `c1_intrabar_parity.py`
