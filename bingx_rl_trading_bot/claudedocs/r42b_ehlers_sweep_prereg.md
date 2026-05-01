# R42b Pre-Registration — Ehlers Cycle Mean Reversion SWEEP

**Date pre-registered**: 2026-05-01
**Status**: PRE-COMMIT (sweep grid + split LOCK, BT 미실행)
**Trigger**: 사용자 critique — single-config 평가 부당, parameter sweep으로 mechanism potential 측정 의무.
**Honest prior**: ~70% no-OOS-pass (32 surface-tested falsified evidence). Sweep + multi-stage validation으로 envelope 한계 정확히 측정.

---

## What changed vs R42

R42 single config (cycle_thr=0.7, sma=50, atr=1.0, timeout=0.75, smooth=4, detrend=20) → 0 signals (vacuous, internal contradiction).

R42b는 같은 mechanism이지만:
- **sma_trend_window=0** (trend filter OFF) 포함 → cycle reversion 단독 평가
- **cycle_threshold = [0.3, 0.5, 0.7]** → R42에서 0.7 너무 strict 가능성
- 6 parameter × 2-3 levels = 72 configs

이는 silent pivot이 아님 — 사용자 명시 critique에 따른 systematic sweep. R42 vacuous는 별도 lesson 보존.

---

## Locked Parameter Grid (FROZEN)

| Param | Levels | Rationale |
|-------|--------|-----------|
| cycle_threshold | 0.3 / 0.5 / 0.7 | Cycle wave extreme 정의. R42 0.7만 → component contradiction |
| sma_trend_window | 0 (OFF) / 50 / 200 | 0 = trend filter 제거 (R42 contradiction 해소) |
| atr_stop_mult | 1.0 / 2.0 | 너무 tight (1.0) vs lenient (2.0) |
| timeout_mult | 0.5 / 1.0 | cycle period의 절반 vs 1배 |
| smooth_window | 4 / 8 | Hilbert input smoothing |
| detrend_window | 20 / 40 | Detrend period |

**Total**: 3 × 3 × 2 × 2 × 2 × 2 = **72 configs**

---

## Multi-Stage Validation (Anti-fishing)

### Data split
- **720d BTC 1h** (17,280 bars)
- **50% IS** (360d) → all 72 configs evaluation
- **25% VAL** (180d) → IS top-5 by daily_net only
- **25% Fresh OOS** (180d) → val-PASS configs only, 1회 평가

### Stage criteria (per config)
- **F1 IS**: avg_gross > 0.07% (friction floor)
- **F6 IS**: n_trades ≥ 50 (statistical sufficiency)
- **Bootstrap IS**: 사용자 6 criteria (mean_daily ≥0.20%, p5 ≥0, pos_rate ≥0.50, avg_per_trade > 0.07%, sufficient trades/window, p_beats_baseline ≥0.55)

### Promotion gates
- IS PASS (F1+F6+bootstrap) → eligible for VAL top-5
- VAL PASS → eligible for OOS
- OOS PASS → DEPLOYABLE

### Bonferroni 처리
- Per-mechanism pre-reg (R42b 단위, 72 configs는 다른 mechanism 아님)
- Multi-stage promotion 자체가 multiple-testing 보호 (IS top-5 → VAL → OOS)
- VAL/OOS는 IS와 다른 데이터, FWER 통제

---

## Deployable Threshold

OOS PASS 시:
- 추가 검증 (regime test, BT-LIVE parity dry-run)
- 5-Gate protocol 시작 (memory: strategy_deploy_5gate_protocol.md)
- LIVE deploy는 별도 결정

OOS FAIL 시:
- R42b sweep falsified (mechanism 자체 falsification)
- 카운트: surface-tested 32 + sweep-tested 1 (R42b)

---

## Honest Prior Distribution

- ~70% no IS PASS (mechanism 자체 invalid, sweep 안에 있는 grid 점도 통과 못함)
- ~20% IS PASS but VAL/OOS FAIL (overfitting)
- ~8% VAL PASS, OOS FAIL (lookbiased borderline)
- ~2% OOS PASS (deployable)

---

## Anti-fishing commitments

1. **Grid LOCKED**: 위 6 parameter × levels 변경 금지. 추가 parameter (예: ATR window, cycle_norm_window) 추가도 금지 (새 pre-reg 필요).
2. **Top-K=5 LOCKED**: IS top-5만 VAL로. top-10 / top-3 변경 금지.
3. **OOS once**: val-PASS configs OOS 평가는 1회만. Re-evaluation 금지.
4. **All results reported**: 72 configs IS 결과 모두 보고 (cherry-pick 차단).
5. **Failure mode reporting**: F1 fail / F6 fail / bootstrap fail 분리 카운트.

---

## Result template (sweep 후 채움)

```
IS PASS:  ___ / 72
VAL PASS: ___ / 5
OOS PASS: ___ / N

DEPLOYABLE: ✅ YES / 🔴 NO

If DEPLOYABLE:
  - List of OOS-passing configs (daily_net, avg_gross, n_trades, WR)
  - Next step: regime test + 5-Gate protocol
If NOT:
  - R42b mechanism sweep falsified
  - Best IS config metrics for envelope reference
```
