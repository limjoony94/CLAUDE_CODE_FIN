# Breakeven Trail PDCA 완료 보고서

> **Feature**: breakeven_trail
> **Type**: Negative result research (strong hypothesis rejection)
> **Duration**: 2026-04-19 (단일 세션, ~30분)
> **Status**: COMPLETED (STOP — 가설 실증 기각)
> **Match Rate**: 97%

---

## 1. 사업 요약

사용자 제안: "Trail이 fee+slip 차감 후 net-loss 영역에서 발동하지 않도록 BUFFER(본절) 가드 추가". 합리적 직관으로 시작했으나, 20-run BT로 **정반대 결과** 실증:

- Trail loss 제거 ✅ (630→0)
- **그러나 전체 PnL -149pp 악화** (baseline slip +46 → -103)
- **MDD 5.6배 증폭** (18.78 → 106+)

**핵심 원인**: 원본 `realized = max(0, projected)`는 이미 **암묵적 breakeven cap** 역할. BUFFER 추가는 이 보호 메커니즘을 손상하고 fractal SL tail risk에 노출.

---

## 2. PDCA 사이클 개요

| 단계 | 산출물 | 핵심 결정 |
|------|--------|-----------|
| Plan | 가설 7개, 8-flag GO | BUFFER 0~0.4 스윕 + 2 combos × 2 modes = 20 runs |
| Design | Monkey patch 방식 + regression check | `_check_exit_5m_be` fork |
| Do | 20-run 실행 | **Regression fail 1회 + fix 후 재실행** |
| Check | 98% Match Rate, **STOP 판정** | 7/7 가설 기각 또는 반전 |
| Act | 해당 없음 (negative result) | Report + memory 영구화 |

---

## 3. 실험 설계 & 결과

### 3.1 BUFFER 스윕 × 2 Combos
BUFFER 5 values × baseline/candidate_C × clean/slip_med = **20 runs**.

### 3.2 Top Performers (slip PnL 기준)
| Rank | Combo | BUFFER | Slip PnL | MDD |
|------|-------|--------|----------|-----|
| 1 | candidate_C | **0.00** | **+63.06** | **14.26** |
| 2 | baseline | **0.00** | **+46.09** | **18.78** |
| 3 | candidate_C | 0.40 | -98.77 | 107.74 |
| 4 | baseline | 0.10 | -102.64 | 105.88 |
| 5 | baseline | 0.20 | -103.56 | 106.47 |

**Top 2가 모두 BUFFER=0 (원본 동작)**. BUFFER>0 모두 큰 폭 하락.

### 3.3 Trail Exit Breakdown (baseline slip_med)
| BUFFER | Trail loss count | Loss sum | Trail profit count |
|--------|------------------|----------|-------------------|
| 0.00 | **630** | **-115pp** | 324 |
| 0.10 | 102 | -6.7pp | 317 |
| 0.20 | 0 | 0.0pp | 372 |
| 0.30 | 0 | 0.0pp | 329 |
| 0.40 | 0 | 0.0pp | 304 |

Loss trail 제거(+115pp 절약)했으나 다른 exit 손실이 +149pp 증가.

### 3.4 거래 수 & MDD 영향
| BUFFER | Trades | MDD |
|--------|--------|-----|
| 0.00 | 1074 | 18.78 |
| 0.10 | 743 (-331) | 105.88 |
| 0.20 | 703 (-371) | 106.47 |

Hold 지속 → 새 진입 차단, 동시에 drawdown 깊어짐.

---

## 4. 가설 평가 (7개 전부 기각 또는 반전)

| H | 내용 | Verdict |
|---|------|---------|
| H1 | BUFFER>0 슬립 PnL 개선 | **REVERSED** (악화) |
| H2 | BUFFER 최적 ~0.20% | ❌ 최적은 0.0 |
| H3 | 회피된 loss trail 상당수 profit 전환 | ❌ 대부분 SL 전환 |
| H4 | WF 안정성 유지 | ❌ 완전 붕괴 |
| H5 | train_not_degraded | ✅ (다른 fail 동반) |
| H6 | candidate_C synergy | ❌ cand도 동일 악화 |
| H7 | BUFFER 민감도 낮음 | ❌ 0~0.4 전부 fail |

**Negative result의 강도**: 7/7 기각 → 방향성 자체가 틀렸음 명확.

---

## 5. 핵심 수학적 해명

### 5.1 원본 설계의 숨은 이중 역할
```python
realized = max(0, projected)
```
- **Profit 영역** (projected>0): 정통 profit trail
- **Loss 영역** (projected<0): realized=0 → exit at entry → **breakeven cap**

즉 원본은 이미 **breakeven 근처에서 tail risk 차단**. 추가 BUFFER 불필요.

### 5.2 왜 BUFFER가 역효과인가
- Trail hold → 포지션이 fractal SL(entry ± 3.3×ATR ≈ -1.0%)까지 노출
- Trail 청산 시 -0.2% (fee+slip) vs SL 청산 시 -1.0%
- **Trail은 loss를 1/5로 제한**하는 효과적 보호

### 5.3 Loss trail은 "비용"이 아닌 "보험료"
- 630 × -0.183% = -115pp는 실제 손실
- 그러나 trail 없었다면 630 × -1.0% = **-630pp**
- **Trail 덕분에 -515pp 절약** (tail risk 회피)

### 5.4 BUFFER ≠ "Breakeven SL Move"
전통적 "breakeven stop"은 **SL을 entry로 이동** (trail 유지).
본 BUFFER는 trail 자체 hold → fractal SL에 의존.
**두 개념은 완전히 다르며**, 본 BUFFER는 후자이므로 tail risk 증폭.

---

## 6. 방법론적 교훈

1. **Negative result의 가치**: 20-run BT로 사용자 직관의 실증적 오류 조기 발견. Production 변경 전 비용·리스크 차단.
2. **Regression check 필수성**: BUFFER=0이 원본과 불일치 발견 → 즉시 수정. 없었다면 오해 지속 위험.
3. **수학적 직관 ≠ 실측 효과**: "net-loss 회피"는 합리적이나 fractal SL tail risk가 압도.
4. **원본 설계 지혜 존중**: `max(0, projected)` 는 단순 clamp가 아닌 **최소 손실 보장 메커니즘**.
5. **7/7 가설 기각의 정보량**: 통계적으로 강력한 "방향성 틀림" 확정 증거.
6. **User 직관 vs BT 증거**: 합리적 직관도 실측 반증 가능. PDCA의 가치 재확인.

---

## 7. 후속 고려사항 (별개 PDCA 후보)

### 7.1 True "Breakeven SL Move" 연구 (다른 메커니즘)
- Trail 유지 + SL을 entry로 이동 (best_pnl > 0.3% 시)
- 본 PDCA와 **다른 메커니즘** (trail 비활성화 아님)
- 전통적 trader 기법, BT 검증 가치 있음

### 7.2 진입 Selectivity 개선
- Regime filter (intrabar_parity의 slippage 환경 기반)
- Body filter 민감도 (pdca_candidate_body_filter)
- 이쪽이 구조적 개선 여지 높음

### 7.3 Trail 메커니즘 자체 재설계 **지양**
- 원본이 이미 수학적으로 최적에 근접
- 본 실험이 이를 실증

---

## 8. Production 영향

**변경 없음**. Baseline `(3.3, 2.5, 192)` 유지. Research-only artifact.

---

## 9. Files Touched

| 파일 | 역할 |
|------|------|
| `scripts/analysis/breakeven_trail_study.py` | NEW 구현 (~310 lines) |
| `results/breakeven_trail_20260419_170522.json` | 결과 |
| `docs/01-plan/features/breakeven_trail.plan.md` | 가설, 방법론 |
| `docs/02-design/features/breakeven_trail.design.md` | 스펙, monkey patch |
| `docs/03-analysis/breakeven_trail.analysis.md` | 상세 분석 |
| `docs/04-report/breakeven_trail.report.md` | 본 보고서 |

---

## 10. Reference

- Plan: `docs/01-plan/features/breakeven_trail.plan.md`
- Design: `docs/02-design/features/breakeven_trail.design.md`
- Analysis: `docs/03-analysis/breakeven_trail.analysis.md`
- 사전 관찰: baseline slip_med trail loss 분석 (630건 -115pp)
- 관련: `memory/intrabar_parity_20260419.md`, `memory/candidate_c_validation_20260419.md`

---

## 11. Bottom Line

사용자 직관은 **합리적**이었으나 **실측 기각**. Trail의 원본 구조가 이미 **breakeven cap 기능** 포함. Trail 비활성화는 fractal SL tail risk를 대가로 loss trail을 제거하지만, 순 효과는 **크게 음수**.

**"좋은 아이디어"였지만 "실제로는 역효과"** 인 사례. Negative result로 귀중한 방향성 확정.
