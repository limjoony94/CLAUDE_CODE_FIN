# Session Start — Game-Theoretic Force-Flow BTC v2

**Mandate version**: v2 (revised 2026-05-01)
**Session start**: 2026-05-01
**Project root**: `C:\Users\J\OneDrive\CLAUDE_CODE_FIN\game_theory_btc_v2\`
**Git tracking**: parent repo `CLAUDE_CODE_FIN/` (no separate repo init — OneDrive nested-git complications)

---

## Q1/Q2 사용자 결정 (2026-05-01)

| 항목 | 결정 | 비고 |
|------|------|------|
| Q1 프로젝트 root | **A안** — `game_theory_btc_v2/` 신규 top-level | 기존 `bingx_rl_trading_bot/`와 완전 분리. R26/C1 잔재 영향 차단 |
| Q2 BingX API | 기존 `bingx_rl_trading_bot/config/api_keys.yaml` 재사용 | R26 shelved 직후 같은 키 OK |

---

## Zero-Base Assumption — 운영 정의

| 항목 | 운영 의미 |
|------|----------|
| 코드 | 기존 `bingx_rl_trading_bot/scripts/`의 어떤 코드도 drop-in 재사용 안 함. 처음부터 빌드 |
| 데이터 | 기존 `data/btc_5m_*.parquet` 등은 **참조는 가능, 신뢰는 안 함**. P0.2에서 fresh fetch + audit 후 PASS 시 사용 |
| Memory | 기존 memory는 **읽되 가정 금지**. 본 mandate가 ground truth |
| 검증 결과 | "32-sweep 0/32 fail", "Funding Arb V5 +3.6%/yr", "L2 hindsight +1.9%/day ceiling" 등 prior conclusion은 **무가정**. P0-P6에서 재검증 |

---

## Mandate v2 핵심 요약

### 1. Theoretical Framework (학문 매핑)
- **통찰 A** (long-short 줄다리기): Brock-Hommes HAM, Kyle 1985, Brunnermeier-Pedersen 2005
- **통찰 B (v2 정정)** (저가 long advantage): OI 4-case 분해. `long_open=long_close` rotation이 핵심. Wyckoff Accumulation Phase B-C, predator-absorber framework
- **통찰 C** (SL forced flow): Brunnermeier-Pedersen predatory trading
- **통찰 D** (TP voluntary force): Wyckoff Distribution, Cartea-Sánchez-Betancourt mean-field
- **통찰 E** (multi-niche): NK-model PLS-complete, MAP-Elites, Minority Game

### 2. 검증 가능 가설 (H1-H9)
- H1: long/short net imbalance × Kyle's λ → 1-12h price direction
- H2: Kyle's λ low (non-toxic) constituent = retail entry safest
- H3-revised: 큰 long-liquidation + low quantile + 음 funding → mean reversion long edge
- H4-revised: liquidation cascade 5-30분 window edge > 0.16% RT friction
- H5: short-side mirror (BTC drift bias 약화)
- H6: 가격 상승 + funding spike + OI ↑ → distribution short candidate
- H7: spot-perp basis extreme + L/S ratio extreme → fade
- H8: MAP-Elites archive ≥3 cells 6-criteria 통과
- H9: Risk-Aware Thompson Sampling > static archive

### 3. Capital-Stage Strategy
- **현재 scope**: S0 ($1.5K) → S1 ($10K-$100K)
- S2-S5 (mid/whale) 별도 mandate

### 4. Anti-Fishing Charter
1. Single-attempt pre-commit: hypothesis + PASS/FAIL + stopping rule 사전 등록. silent pivot 금지
2. Fresh OOS holdout: last 90d (또는 25%) sealed during fitting
3. Lookahead audit: lag-shift sensitivity (lag 0 vs lag 1 dramatic difference → leak 의심)
4. Realistic friction: BingX taker 0.045%/side + slippage 0.02-0.05%/side = RT 0.13-0.20%. Funding 8h 별도
5. **6-Criteria gate** (3-day random window bootstrap, B=10000):
   - mean ≥ target_daily
   - p5 ≥ 0
   - pos_rate ≥ 0.5
   - p_beats_baseline ≥ 0.55
   - MaxDD ≥ -X%
   - Sharpe ≥ 1.5 (annualized)
6. Priority closure 명시 (`experiments/p{N}/result.md`): 3개월 self-audit 가능
7. Honesty as terminal value: PARTIAL은 PARTIAL로

### 5. Priority Schedule (P0 → P6)
```
P0  Zero-base inventory + theory grounding + env setup     (7d)
P1  BingX API + 공개 데이터 인벤토리                        (1d)
P2  Force-Flow Reversal Hypothesis (H3-H4 검증)             (3-5d)
P3  MAP-Elites on Mechanism × Regime Grid (H8)              (5-7d)
P4  Risk-Aware Thompson Sampling Bandit (H9)                (3-5d)
P5  Force-Flow Detection 정밀화 (P2 PASS/PARTIAL 시만)      (5-10d)
P6  통합 Portfolio + LIVE-readiness                          (3-5d)
```

### 6. P0 6-Substep + 7 Deliverables
| Sub | Title | Days |
|-----|-------|------|
| P0.1 | Environment & repo setup | 1 |
| P0.2 | Market data acquisition (BTC 1h/5m/1d/funding/OI/liquidation + multi-asset) | 1-2 |
| P0.3 | friction_model.py + bootstrap_six_criteria.py + unit tests | 1 |
| P0.4 | Buy-and-hold baseline + 1× constant long + random entry | 1 |
| P0.5 | H1-H9 가설 정량 정의 (memory/p0_hypothesis_registry.md) | 1-2 |
| P0.6 | Reference candidate strategies (~30 mechanism minimal def) | 1 |

**Day 7 closure**: 7-deliverable status table + GO/NOGO for P1.

### 7. 복리 수학 정직성 (§ 10)
- 시나리오 A (Funding Arb only): 99% prob, ~100년 to whale-tier ($1M)
- 시나리오 B (mandate full success 0.10-0.20%/day): 30-40% prob, 6.5-13년
- 시나리오 C (aggressive 0.50%/day): 5-10% prob, 3.6년
- 시나리오 D (1%/day, anecdotal whales): <2% prob, 2.5년
- **표는 envelope 인식용. honest closure 우선**
- Survival bias 반대편 항상 인지

---

## P0 Pre-Commit (자세한 내용은 `experiments/p0/precommit.md`)

가설: 7일 내에 7-deliverable 모두 working code + documented + reproducible.
PASS: 7개 deliverable 통과.
FAIL: 7일 후 1개 이상 미완성 → 본 mandate 재검토 필요.
Stopping rule: 7일 hard limit.

---

## Honesty Pledge

본 session에서 다음을 약속:
1. PASS 못한 priority를 PASS로 포장하지 않음
2. silent pivot 없음 (pre-committed criteria만 인정)
3. lookahead leak 발견 시 즉시 retract
4. BT-LIVE parity 의심 시 deploy 금지 (R26/C1 두 번 같은 패턴 재발 방지)
5. Strict criterion 통과 못 해도 정직한 closure로 다음 session 진행

---

## Memory File Index (P0 진행 중 추가 예정)
- `000_session_start.md` (이 파일) — 시작 + mandate
- `p0_hypothesis_registry.md` — H1-H9 정량 정의 (P0.5에서 생성)
- `p0_summary.md` — P0 closure 결과 (Day 7)
- 후속 priority별 추가 예정
