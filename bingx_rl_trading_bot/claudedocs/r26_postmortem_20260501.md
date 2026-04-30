# R26 Grid Postmortem — BT-LIVE Parity Audit

**Date**: 2026-05-01 UTC
**Trigger**: User intuition ("거래소 5+5 주문 넘어 거래 안 하나?") → 4-cycle advisor BT-LIVE parity review → bot halted
**Status**: 🪦 **R26 SHELVED**. Open orders cancelled, watchdog deregistered, state.json deleted, $500 → final balance preserved at exchange.
**Reference**: C1 Breakout postmortem (`c1_breakout_postmortem_20260427.md`) — 동일 root cause pattern 재발

---

## 0. 사건 요약 (TL;DR)

R26 grid bot은 사전 검증 BT 결과 +0.21%/day, +169.5% PnL/333d로 deploy(2026-04-30 18:03 KST). 그러나 사용자가 "거래소 5+5 주문 넘어 거래 안 하나?"라는 한 문장으로 BT-LIVE parity 결정적 결함을 식별. 4-cycle advisor 검증을 통해 사전 검증 BT가 LIVE 봇의 6개 핵심 동작을 누락한 것이 확인됨.

**balance-aware LIVE-parity 모델 재실행 결과**:
- BingX 100d (LIVE 거래소): **-46.46%** / -0.46%/day, MaxDD -50.8%, 20 halts
- Binance 689d (긴 기간): **사실상 RUIN** (balance $24.94 = 5%, 2025-12-11 도달), -95.01% / -0.14%/day, 85 halts

LIVE 14d 실측 -12.86% (-0.92%/day)는 BT의 2× worse — 추가 friction 0.46%/day 존재.

**결론**: 사전 검증 +0.21%/day는 BT 모델 결함의 인공물. 실제 R26 grid는 negative-EV. 봇 정지 결정.

---

## 1. 4-Layer 결함 분석 (advisor synthesis)

### Layer 1 — 모델 충실도 결함 (BT가 LIVE 동작 누락)

| # | LIVE 동작 (코드 위치) | BT 누락 결과 |
|---|---|---|
| **M1** | `grid.py:_replace_grid_level()` — TP/SL 후 같은 level 재배치 무한 cycle | BT는 한 grid lifetime에 level당 1회 fill. 추세장 SL 누적 손실 0배 표시 |
| **M2** | LIMIT @ P를 시장가가 P 아래일 때 placement → marketable taker @ open price | BT는 무조건 maker @ level price. 추세장 fill underestimate |
| **M3** | `bot.py:check_halts()` — daily 3% / 30d 10% / emergency 20% halt | BT는 halt 모델 없음 |
| **M4** | `bot.py:cycle()` — 5min poll로 fill check, intrabar SL은 STOP_MARKET | BT는 1h candle 단위. intra-1h 가격 변동 무시 → optimistic fill |
| **M5** | 8h funding fee | BT 무시 |
| **M6** | `auto_size_from_balance: true` — balance 변동 → per_level 자동 축소 | BT는 capital fixed, per_level fixed. 자동 deleveraging 미모델 |

**핵심**: M1 (re-arm)이 가장 결정적. 사전 검증 모델은 본질적으로 **다른 전략**을 시뮬레이션 — "각 level은 한 번만 발사" vs LIVE의 "무한 cycle".

### Layer 2 — 검증 절차 결함 (process가 양수 결과를 만든 메커니즘)

| # | 결함 | 양수 결과 인과 |
|---|---|---|
| **P1** | 단일 window (720d sequential) | 평균 양수, multi-window n=20에서 3/20 양수만 (15%) |
| **P2** | Pre-reg가 parameter만 lock, 모델 가정 lock 안 함 | "BT 모델 = LIVE 코드"라는 가정 자체는 검증 안 됨 |
| **P3** | 데이터 source mismatch (Binance for BT, BingX for LIVE) | 5-30bps 갭, 0.30% spacing에서 fill timing 영향 |
| **P4** | GO criterion이 단일 metric (cum %, daily %) | path-dependent metrics (MaxDD, halt frequency) 누락 |
| **P5** | round34 leverage tuning이 같은 결함 모델 사용 | "L=4× DEPLOYABLE" 결론도 같은 결함의 인공물 |

**핵심**: P2가 메타-결함. pre-reg가 parameter를 lock하는데 **모델이 LIVE를 정확히 시뮬하는지**는 lock 항목 아님.

### Layer 3 — 시스템/운영 결함 (deploy 후 발견 시스템 부재)

| # | 결함 | 결과 |
|---|---|---|
| **S1** | D-1/D-3/D-7 자동 distribution check 없음 | LIVE 14d -12.86%까지 진행 후 D3 진단 수동 실행. memory에 학습은 있었으나 적용 안 됨 |
| **S2** | Paper trade gate 없음 | 코드 변경 → BT 양수 → 즉시 LIVE deploy |
| **S3** | 자동 kill switch 부재 | -12.86% 도달까지 봇 계속 운영 |

### Layer 4 — 인지/의사결정 결함

| # | 결함 | 증상 |
|---|---|---|
| **C1** | Confirmation bias | round26 +0.21% → round34 leverage 검증 → 즉시 deploy. "검증" rounds이 같은 모델 재실행 |
| **C2** | Past precedent 무시 | CLAUDE.md "C1 LIVE failure precedent applies" 명시되어 있었으나 R26은 같은 pattern 재발 |
| **C3** | User intuition이 유일한 escape | 4-cycle advisor 검토에서 발견된 6개 결함 중 M1을 처음 짚은 것은 사용자 한 문장 |

---

## 2. 잠재 결함 — 시스템에 같은 패턴 점검

🔴 **확실한 같은 패턴**:
- **C1 Breakout v2** (이미 shelved 2026-04-27) — D3 진단으로 LIVE -12.86% 발견. 같은 process 결함이 두 번째 deploy에서 같은 결과 만듦.

🟡 **점검 권장**:
- `scripts/tests/` (113 pytest cases) — BUG#1~65 regression guard. **BT-LIVE parity test는 없음**. test cases는 internal logic correctness만 검증.
- `claudedocs/BACKTEST_LIVE_PARITY.md` (22-item, 20/22) — **C1 Breakout 기준**. R26 grid의 re-arm/marketable LIMIT/balance compounding 항목 누락. strategy-specific 아님.
- `trading-researcher` agent의 Standard Research Protocol — additive PnL/expanding WF 명시되어 있으나 **모델 충실도 검증 step 없음**.
- 28+ rounds 누적 결과 — 대부분 directional/breakout이라 R26 grid-cycle bug와 다른 카테고리. 새 grid 계열 만들면 같은 bug 재발 가능.

🟢 **Safer**:
- R5 single-coin BTC carry — directional 아니라 carry라 grid bug 무관

---

## 3. 5-Gate Prevention Protocol (HARD criteria — pre-deploy)

### Gate 1 — 모델 충실도 audit (HARD)
- [ ] LIVE bot 코드의 모든 `_on_fill`, `_replace_*`, `force_close_*`, `check_halt*`, balance update path → BT 모델에 explicit analog 존재
- [ ] **Code review with side-by-side mapping**: bot.py line N → BT script line M (1:1 표 작성)
- [ ] auto_size/compounding이 LIVE에 있으면 BT도 balance 변동 추적

### Gate 2 — 통계적 검증 (HARD)
- [ ] **n ≥ 20 비겹침 multi-window**, mean+median 둘 다 양수
- [ ] one-sided sign test p < 0.05
- [ ] 단일 sequential 720d cum PnL은 **necessary but not sufficient**
- [ ] regime metric vs PnL 상관관계 plot — 단일 regime dependence 시 caveat

### Gate 3 — 데이터 source 일치 (HARD)
- [ ] BT data는 LIVE 거래소와 동일. depth limit으로 불가하면 multi-source 비교 + caveat
- [ ] LIVE config snapshot (yaml hash)을 BT script 내부에 검증 (drift 감지)

### Gate 4 — Paper trade gate (HARD) ⭐ **advisor 단일 판단**
- [ ] 코드를 paper-mode (testnet 또는 mark-price-only fill) **최소 7일** 운영
- [ ] BT 예측 vs paper 결과 비교, gap > 0.1%/day면 BT 모델에 누락 동작 있음 → 발견 + 수정 + 재BT 후 LIVE
- [ ] paper mode는 LIVE 코드 동일 path 사용 (mock 거래소만 다름) — 모델 parity 자동 보장
- **이유**: M1 같은 결함은 사후 검증으로는 발견하기 어려움. paper trade는 같은 code path를 LIVE-like 환경에서 돌려 **모델 결함 + 코드 결함을 동시 발견**. R26과 C1 두 incident 모두 paper trade가 있었으면 deploy 전에 발견됐을 것.

### Gate 5 — Deploy 후 D-checkpoint (자동화)
- [ ] D-1: equity ≤ BT P5 → automatic halt + alert
- [ ] D-3: rolling 3d ≤ BT P5 → halt
- [ ] D-7: rolling 7d ≤ BT P5 → halt
- [ ] D-checkpoint은 cron job으로 자동 — human review 의존 X (S1 결함 방지)

### `scripts/tests/`에 BT-LIVE parity test 추가 (지속 적용)
- [ ] 새 test category: "주어진 5m candle sequence에서 BT 모델 결과 vs LIVE 봇 동일 sequence 시뮬 결과가 매 step에서 1bp 이내 일치" 자동 검증

---

## 4. 정직한 인정사항 (advisor self-critique)

**4-cycle advisor 검토에서 첫 review에서 일부 결함을 underweight**했음. 특히 user 직관 전엔 re-arm (M1) 자체를 제기 안 함. advisor도 process layer의 일부고, paper trade gate가 advisor judgment보다 robust.

**3-layer가 모두 필요**:
1. User intuition (도메인 직관)
2. Paper trade gate (실증 검증)
3. Multi-layer review (advisor + multi-window + parity test)

이번처럼 user intuition이 첫 신호인 경우, agent/advisor는 그 신호를 amplify하는 역할까지가 reliable한 contribution.

---

## 5. 사건 시퀀스 (timeline)

| 일시 (UTC) | 사건 |
|---|---|
| 2026-04-30 18:03 KST | R26 LIVE deploy (PID 44332, balance $500, mid $76,008) |
| 2026-05-01 ~07:00 | 사용자 1주일 BT 요청 → 1주일 +2.24% (legacy model) |
| 2026-05-01 ~07:30 | 사용자 직관: "거래소 5+5 주문 넘어 거래 안 하나?" → re-arm 결함 식별 |
| 2026-05-01 ~07:50 | 1달 BT 요청 → -3158% (BT 모델 결함, advisor flag) |
| 2026-05-01 ~08:00 | advisor cycle 1: re-arm fill model, halt, lookahead, funding 수정 plan |
| 2026-05-01 ~08:15 | advisor cycle 2: halt anchor session-cum + equity 수정 |
| 2026-05-01 ~08:25 | advisor cycle 3: 720d multi-window n=20 = 3/20 양수, p=0.0013 |
| 2026-05-01 ~08:35 | 사용자: "더 긴 기간 BT" 요청 → BingX 100d + Binance 720d |
| 2026-05-01 ~08:45 | 결과 -212% 도출 → 사용자 지적 "비현실적, 재점검" |
| 2026-05-01 ~09:00 | advisor cycle 4: balance compounding + ruin detection 수정 |
| 2026-05-01 ~09:15 | 최종 결과: BingX 100d -46%, Binance 689d RUIN at 2025-12-11 |
| 2026-05-01 ~09:30 | **사용자 결정: 봇 정지** |
| 2026-05-01 ~09:35 | Watchdog unregister, PID 78052 kill, 10 LIMIT cancel, state.json 삭제 |
| 2026-05-01 ~09:40 | advisor cycle 5: 4-layer 분석 + 5-gate protocol |
| 2026-05-01 (이 문서) | Postmortem 작성, MEMORY/CLAUDE.md 업데이트 |

---

## 6. user judgment 영역 (다음 결정)

advisor는 의도적으로 다음 중 어떤 것을 하라고 권장하지 않음. 위 protocol이 정착된 후 진행이 안전.

- **R5 single-coin BTC carry 운영 유지** (deployable, $49/yr)
- **새 전략 개발 시작** (Gate 1-5 적용 시)
- **휴식 / 시스템 재구축**

---

## 7. 결과 파일 references

- BT scripts: `scripts/analysis/r26_grid_5m_pastweek.py`, `r26_grid_long_bt.py`, `r26_grid_multi_window.py`
- Result JSONs: `results/r26_grid_5m_multi_20260501_073148.json` (1주/1달), `results/r26_grid_multi_window_20260501_073153.json` (n=20), `results/r26_grid_long_bt_20260501_074556.json` (long, balance-aware)
- Memory: `memory/r26_bt_live_parity_finding_20260501.md`
- Predecessor postmortem: `claudedocs/c1_breakout_postmortem_20260427.md`
