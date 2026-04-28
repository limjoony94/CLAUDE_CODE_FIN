# M3-R9c — α N=4 Fixed Exit OOS Pre-registration (단일 검증)

> **Date**: 2026-04-28
> **Authority**: 사용자 옵션 F + 추가 심층 + advisor strict OOS 권고
> **Anti-pattern declaration**: R9b 126-config sweep (3 mech × 7 N × 6 friction)에서 positive 나온 candidate. **Multiple-comparison risk**. 따라서 단일 OOS 검증으로 "noise vs real" 결정.

---

## 1. R9b 결과 (sweep, retrofit risk)

| Candidate | Best config | Daily @ f=0.04 | n | WR | RR |
|-----------|-------------|----------------|---|------|----|
| **α N=4 fixed** | N=4 timeout-only exit | **+0.0195%** | 195 | 46.7% | 1.31 |
| ι N=6 fixed | n=49 (sample 너무 작음 — drop) | +0.0200% | 49 | 53.1% | 1.37 |

**Drop ι**: 0.068 trades/day는 noise territory (advisor 지적).
**Pick α N=4**: n=195 (0.27/day) 가장 robust sample.

## 2. New pre-registration (R9b sweep 후 commit, OOS 결과 보기 전)

**Spec**: α entry rules (ETH-lag + 고변동성, 1h+4h trend) + **fixed N=4 timeout exit only** (no trail, no SL, only emergency 1.5% + N=4 timeout, min_bars_between=2).

**Hypothesis**: 17 mechanisms는 trail framework + structural SL이 entry alpha를 잠식. Fixed N=4 (1h hold)는 alpha 보존 + friction-frequency 균형 양호. 단 R9b sweep에서 발견됐으므로 **OOS test로 noise vs real 결정**.

### 합격 조건 (사전 등록, 결과 보기 전)

| # | Test | 조건 |
|---|------|------|
| 1 | WF 5-fold (expanding) @ f=0.04 | **3/5 folds positive daily** |
| 2 | 3-way split (train/val/test) @ f=0.04 | **test split positive daily** |
| 3 | 3-day bootstrap (200 windows) @ f=0.04 | **mean > 0 AND pos_rate ≥ 50%** |
| 4 | Friction sensitivity (0.02, 0.04, 0.06) | 모두 양수 daily |
| 5 | Robustness | WR ≥40, RR ≥1.0 (원본 spec, 1.5 아님), n ≥ 150 |

**전부 통과해야 PASS**. **하나라도 fail → drop the claim**.

### 추가 정직성 안전장치

- **N=4는 변경 금지**: WF/3-way 결과 무관하게 N=3, N=5 등 추가 sweep 금지 (fix-impulse trap)
- **Parameter optimization 금지**: ETH thresh 0.3, BTC lag 0.1, ATR pctile 70 그대로
- **Result 후 criterion 추가 조정 금지**: 본 문서 commit 후 deep test 실행

## 3. Predictions

| Test | Predicted | Confidence | Rationale |
|------|-----------|-----------|-----------|
| WF 5-fold | borderline 2~3/5 | LOW | Original α R3 trail version WF 0/5. Fixed N=4 가능성 더 강함 |
| 3-way test | borderline | LOW | If WF 3+ holds, test split likely positive |
| Bootstrap | borderline | LOW | Mean +0.0195 × 1.2 (BT period factor) 가능 |
| Friction sens | likely ALL pass | MED | 0.02/0.04/0.06 in R9b 모두 positive |
| WR/RR/n | PASS | HIGH | n=195, WR 46.7, RR 1.31 |

**Most likely outcome**: 1 of WF/3-way/bootstrap fails → drop claim.
**Most likely surprise**: 3+ tests pass → real edge survives OOS, narrow but actionable finding.

## 4. Post-test branches

- **All 5 PASS**: 진짜 OOS-stable edge. PDCA Plan 작성하여 production canary 검토 (단 maker-rebate infra 필요).
- **1+ FAIL**: Sweep noise. Drop claim. Move to 사용자 옵션 A/B/E.
- **WF 3/5 + 3-way fail**: WF는 path-dependent 가능. 3-way fail = generalization issue → drop.
- **WF 2/5 + bootstrap pos**: WF는 경계, bootstrap stable. Borderline → 사용자 결정.

## 5. Anti-fix-impulse commitment

본 OOS test 결과 무관하게:
- N grid 추가 sweep 안 함
- Parameter optimization 안 함
- α′를 같은 framework로 재시도 안 함 (이미 R2에서 N=16 fixed로 fail)
- 결과 받아들이고 user options A/B/E로 진행

**Memory ref**: `lessons_fix_impulse_pattern_20260427.md` — 본 round은 그 패턴의 명시적 회피 시도.
