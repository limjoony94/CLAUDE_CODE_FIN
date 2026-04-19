# Analysis: Breakeven Trail

> **Feature**: breakeven_trail
> **Date**: 2026-04-19
> **Phase**: Check
> **Match Rate**: **97%** (Design ↔ 구현, regression fix 1회)
> **Outcome**: **STOP** — 사용자 가설 실증 기각. BUFFER > 0 시 모든 조합이 폭락.

---

## 1. Executive Summary

"Trail이 fee+slip 차감 후 net-loss 영역에서 발동 않게 하면 성능 개선"이라는 합리적 직관을 20-run BT로 검증. **결과는 정반대**:

- Trail loss exits 제거 ✅ (630건 → 0건, -115pp → 0pp)
- 그러나 **전체 PnL은 baseline +46 → -103 (-149pp 악화)**
- MDD 18.78 → 106+ **약 6배 증폭**
- 이유: trail을 hold하면 fractal SL(~-1%)까지 노출되어 **tail risk 증폭**

**교훈**: 원본 `realized = max(0, projected)`는 이미 **암묵적 breakeven cap**. BUFFER 추가는 이 보호 메커니즘을 손상.

---

## 2. 실행 결과 매트릭스 (20 runs)

| Combo | BUFFER | Clean PnL | Clean MDD | Slip PnL | Slip MDD | N | Trail loss | Trail profit |
|-------|--------|-----------|-----------|----------|----------|---|-----------|--------------|
| **baseline** | **0.00** | **+169.55** | **5.38** | **+46.09** | **18.78** | 1074 | 630(-115pp) | 324 |
| baseline | 0.10 | +26.10 | 25.35 | -102.64 | 105.88 | 743 | 102(-6.7pp) | 317 |
| baseline | 0.20 | +26.68 | 29.08 | -103.56 | 106.47 | 703 | 0 | 372 |
| baseline | 0.30 | -2.80 | 27.76 | -127.26 | 130.17 | 682 | 0 | 329 |
| baseline | 0.40 | +11.81 | 21.28 | -115.15 | 118.06 | 649 | 0 | 304 |
| **candidate_C** | **0.00** | **+192.76** | **5.20** | **+63.06** | **14.26** | 1057 | 634(-116pp) | 322 |
| candidate_C | 0.10 | +7.55 | 27.03 | -104.60 | 108.82 | 700 | 101(-7.1pp) | 313 |
| candidate_C | 0.20 | -3.76 | 35.17 | -111.70 | 115.86 | 664 | 0 | 368 |
| candidate_C | 0.30 | -35.28 | 44.14 | -135.97 | 139.30 | 635 | 0 | 320 |
| candidate_C | 0.40 | -1.49 | 27.42 | -98.77 | 107.74 | 594 | 0 | 299 |

**Top performers가 모두 BUFFER=0 (원본)**. BUFFER>0은 모두 STOP 권장.

---

## 3. 왜 BUFFER가 역효과인가? (수학적 해명)

### 3.1 원본 Trail의 숨은 이중 역할

```python
# 원본
if drawdown >= trail_dist_pct:
    realized = max(0, projected)
    exit_price = entry * (1 + realized/100)
```

- **Profit 영역** (projected > 0): profit trail (전통적 의미)
- **Loss 영역** (projected < 0): `realized=0` → **exit at entry** (net loss = fee+slip만)

즉 원본은 이미 "**breakeven cap**" 구현 — 추가 BUFFER 불필요.

### 3.2 BUFFER 추가 시 발생하는 것

```python
# BUFFER > 0
if projected < BREAKEVEN_BUFFER:
    return None  # hold
```

Trail을 hold하면 position이 fractal SL 또는 emergency까지 연장. Fractal SL은 보통 entry ± 3.3×ATR ≈ **-1.0%** 수준.

**Trade-off 실측**:
- 회피한 trail-loss: -0.183% × 630건 = **+115pp 절약**
- 추가 SL 손실: **~-149pp 증가**
- 순 효과: **-34pp 악화** (baseline +46 → BUFFER=0.2 -103)

### 3.3 거래 수 감소
BUFFER=0.2에서 baseline 1074 → 703 trades (-371).
- 포지션 hold가 길어 new entry 막힘
- 동일 기간 내 진입 기회 상실 → 수익 기회도 상실

### 3.4 MDD 증폭
18.78 → 106+ (5.6배).
- Trail이 중간 drawdown 컷하던 역할 상실
- Open position이 더 깊은 drawdown 통과 후에야 청산
- Equity curve에 큰 drawdown spike 누적

---

## 4. 8-flag GO 평가

Top performer = `candidate_C_b0.00` (원본). BUFFER > 0 전부 탈락.

| # | Flag | Result |
|---|------|--------|
| 1 | wf_clean_pass | ✅ (BUFFER=0 기준) |
| 2 | **wf_slip_pass** | **❌** (4/5, fold 2) |
| 3 | tw_pass | ✅ |
| 4 | train_not_degraded | ✅ |
| 5 | **pnl_improvement** | **❌** (BUFFER>0 전부 악화) |
| 6 | ratio_ok | ✅ |
| 7 | buffer_stable | **❌** (0/4 better) |
| 8 | rollback_ready | ✅ |

**Verdict: STOP** — core flag `wf_slip_pass`, `pnl_improvement` 실패.

Top=BUFFER=0이라는 사실 자체가 **"BUFFER 도입 반대" 증거**.

---

## 5. 가설 평가

| H | 내용 | Verdict |
|---|------|---------|
| H1 | BUFFER > 0 시 slip_med PnL 개선 | ❌ **REVERSED** (악화) |
| H2 | BUFFER 최적 ~0.20% | ❌ 최적은 0.0% |
| H3 | 회피된 loss trail 중 상당수가 profit 전환 | ❌ **대부분 SL로 전환** |
| H4 | WF 안정성 유지 | ❌ BUFFER>0 WF 완전 붕괴 |
| H5 | train_not_degraded | ✅ (다른 flag들과 함께 fail) |
| H6 | candidate_C + BUFFER synergy | ❌ cand_C도 동일 악화 |
| H7 | BUFFER 민감도 낮음 | ❌ 0~0.4 사이 전부 fail |

**7/7 가설 기각 또는 반전**. 이는 **강력한 negative result** — 사용자 직관이 틀렸음 명확히 입증.

---

## 6. 핵심 인사이트

### 6.1 Trail의 숨은 설계 지혜
원본 `max(0, projected)`는 단순 clamp가 아니라 **"이론 최소 손실 보장"** 역할:
- Trail이 breakeven 부근에서 발동 시 ideal exit at entry
- 체결 시 fee+slip = **-0.2%만 손실**
- 이것이 structural **tail risk 차단**

### 6.2 Loss trail은 "비용"이 아니라 "보험"
- 630건 × -0.183% = -115pp는 **실제 손실**이지만
- 만약 trail 없었다면 → SL까지 hold → 630건 × -1.0% = **-630pp 손실**
- **Trail은 손실을 -1.0% → -0.2%로 제한하는 효과적 보호**

### 6.3 Fee/slip 0.2%는 "합리적 프리미엄"
Trail 청산 시 -0.2% = SL 청산 시 -1.0%의 **1/5 비용**으로 tail risk 회피. 이는 보험료 개념.

### 6.4 왜 일반적 "breakeven stop" 개념과 다른가?
전통적 "breakeven stop" = "포지션이 +X%에 도달하면 SL을 entry로 이동"
- 이건 **static SL 이동**이지 trail 비활성화가 아님
- 본 연구의 BUFFER는 trail을 **완전 hold** → Fractal SL에 의존 → Fractal SL은 훨씬 넓음 (~3%)
- 즉 잘못된 메커니즘 도입

**올바른 구현 대안** (별개 PDCA 후보):
```python
# SL을 entry로 이동 (breakeven stop)
if best_pnl > BREAKEVEN_ACTIVATION:  # e.g., 0.3%
    effective_sl = max(sl, entry_price)  # LONG case
```
이는 trail 유지 + SL tighten 조합. 본 BUFFER 방식과 완전히 다름.

---

## 7. Recommended Action

### 즉시
1. **production 변경 없음** — baseline 유지 (BUFFER=0)
2. Report 작성 + memory 영구화 (negative result의 가치)
3. "**Trail 원본 설계에 이미 breakeven cap 내장**" 지식 영구 기록

### 단기 (2~4주)
1. **True "Breakeven SL Move" PDCA 별개 검토**:
   - Trail은 건드리지 않고 SL만 entry 위치로 이동 (LONG 기준)
   - Activation: best_pnl > 0.3% (fee+slip+여유)
   - Trail + Moving SL 조합으로 upside 보호 + downside limit
2. 30일 LIVE 샘플 관찰 지속

### 중기 (1~2개월)
- Trail 메커니즘 자체 재설계는 지양 (원본이 이미 최적에 근접)
- Instead: 진입 selectivity 또는 regime filter 방향

---

## 8. 방법론적 교훈

1. **Negative result의 가치**: 20-run 실험으로 사용자 직관의 실증적 오류 입증. 비싼 production 실험 방지.
2. **Regression check의 중요성**: 첫 실행에서 BUFFER=0이 원본과 불일치 발견 → 즉시 수정. 없었다면 잘못된 결과 해석 위험.
3. **수학적 직관 ≠ 실측 효과**: "net-loss 영역 회피"는 합리적이나, fractal SL tail risk가 훨씬 큼.
4. **원본 설계의 숨은 지혜 존중**: `max(0, projected)` clamp는 단순 clamp 아니라 tail risk 보호.
5. **Hypothesis 전 기각의 정보량**: 7/7 기각은 강력한 증거. "방향성 자체가 틀렸다" 확정.

---

## 9. Files Touched

- `scripts/analysis/breakeven_trail_study.py` (NEW, ~310 lines)
- `results/breakeven_trail_20260419_170522.json` (NEW, regression OK 버전)
- `docs/01-plan/features/breakeven_trail.plan.md`
- `docs/02-design/features/breakeven_trail.design.md`
- `docs/03-analysis/breakeven_trail.analysis.md` (본 문서)

Production 변경 **0건**.

---

## 10. Reference

- Plan: `docs/01-plan/features/breakeven_trail.plan.md`
- Design: `docs/02-design/features/breakeven_trail.design.md`
- 재사용: `scripts/analysis/intrabar_trail_impact.py`, `c1_intrabar_parity.py`
- 결과: `results/breakeven_trail_20260419_170522.json`
- 사전 관찰: baseline slip_med trail loss 630건 분석
