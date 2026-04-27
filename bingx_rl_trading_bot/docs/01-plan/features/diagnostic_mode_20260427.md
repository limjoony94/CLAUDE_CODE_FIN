# Diagnostic Mode — Post-Halt LIVE Investigation

**Date**: 2026-04-27
**Trigger**: Bot halted at -12.86%/14d (n=46) per advisor escalation
**Goal**: Determine if strategy can be redesigned to LIVE-profitable, or shelved

---

## 1. Halt Context

**LIVE 실측 (2026-04-12 ~ 2026-04-27, 14일)**:
- Trades: 46 | WR: 28.3% (13/46) | daily PnL 1x: **-0.92%**
- Sum 1x: **-12.86%** | Sum 3x: -38.58%
- Balance: $1,617 → $1,495.22 (실제 -$623 손실, 추정 시작 $2,118)

**BT theoretical (2025-07 ~ 2026-04, 272일)**:
- Trades: 943 | WR: 39.7% | daily PnL 1x: +0.83%
- Sum 1x: +218.20%

**Gap**: -0.51%/trade 1x systematic, -11.4pp WR. Selection variance (±5pp/30) **2배 이상**.

---

## 2. Diagnostic Priorities (Advisor 권고 3개)

### Diagnostic 1: WR 28% vs BT 40% 원인

**Hypothesis**: Bar-close 평가 vs intrabar tick reality. SL이 BT가 생각하는 것보다 tight하게 작동.

**검증 방법**:
- 각 LIVE SL hit trade의 5m intrabar 데이터 fetch
- BT가 cycle (15m bar close)에 평가했을 때와 intrabar tick 도달 시점 비교
- "Price wicked below SL intrabar then recovered" 비율 측정

**판정 기준**:
- 50%+ trades에서 wick-recover 발생 → SL placement issue 확인
- 30%+ → tighter SL 필요 (max_sl_atr 재고)
- < 30% → 다른 원인 우세

### Diagnostic 2: All trail variants bleed 원인

**Hypothesis**: Trail TP가 break-even cap에 자주 걸림 → best case = fees only. SL hit -0.52% 보충 못함.

**검증 방법**:
- LIVE 11 TRAIL_TP exits의 best_pnl 분포 vs BT (이미 부분 측정: avg best_pnl prog 0.771%)
- Break-even cap 도달 비율 LIVE vs BT
- Trail K (2.5 → 0.5 progressive)의 LIVE 효과 측정

**판정 기준**:
- Trail이 best_pnl < 0.9% 영역에서 작동 (progressive 미발동) 비율 ≥ BT 예상보다 큼 → trail 설계 문제
- Best_pnl 분포가 BT와 큰 차이 → entry timing 또는 hold duration issue

### Diagnostic 3: Regime mismatch

**Hypothesis**: 2026-04 LIVE 기간 BTC regime이 2025-07~2026-04 BT 평균과 substantially 다름.

**검증 방법**:
- 272d BT에서 rolling 14d windows 누적 PnL 분포 측정
- LIVE -12.86%/14d가 BT 분포의 어느 percentile인가?
- 2026-04 시기 BTC volatility, trend, channel range를 BT 다른 14d window와 비교

**판정 기준**:
- LIVE -12.86%가 BT 분포 P5 미만 → unprecedented regime, BT 자체 generalization 문제
- LIVE가 BT P5~P25 → bad luck regime, 변경 필요할 수 있음
- BT 분포 내 정상 → 다른 원인 (selection, slippage, parity)

---

## 3. Decision Matrix (Diagnostic 결과별)

| Diagnostic 결과 | 판단 | Next Action |
|----------------|------|-------------|
| WR gap = SL placement (intrabar wick) | 전략 fundamental issue | SL 재설계 (intrabar-aware) → 새 BT |
| Trail bleed = break-even cap 자주 도달 | Trail 설계 부적합 | Trail 재설계 (e.g., R:R 우선, ATR 기반 fixed TP) |
| Regime mismatch unprecedented | BT generalization 실패 | 더 긴 BT (2년+) + WF 재실행 |
| 모두 weak signal | 복합 원인 | Strategy shelve, 새 가설 탐색 |

---

## 4. Production 재개 조건 (Hard Gates)

다음 모두 충족해야 LIVE 재개:

1. ✅ **Diagnostic 3개 결과 명확** (한 가지 dominant 원인 식별)
2. ✅ **새 BT 모델이 LIVE-realistic**:
   - Intrabar slippage 모델 포함 (5m or 1m tick resolution)
   - Execution layer 비용 (MARKET slip, latency) 정량 모델
3. ✅ **새 BT가 LIVE 14일 -12.86% 재현 가능** (mechanism validated)
4. ✅ **재설계 전략의 새 BT가 양수** (수정된 BT 모델 기준)
5. ✅ **Out-of-sample 검증** (2026-05 이후 데이터로 paper test 14일)

5/5 충족 전까지 capital 복귀 X.

---

## 5. Process 원칙 보존

- ❌ Anxiety candidate (F v3 enable, max_sl 재튜닝 등 즉시 시도) 금지
- ❌ Re-tune existing knobs without diagnosis
- ✅ Offline diagnostic 우선
- ✅ Hard gate 5/5 통과 후 재개 결정
- ✅ Single-variable 원칙 유지 (한 번에 한 변경만)

---

## 6. F v3 코드 상태

`f_v3_limit_close.enabled: false` 보존. Diagnostic 결과로 trail issue 확인 시 재평가. 단독 enable로 fix 불가 (advisor 분석).

---

## 7. Open Questions

- LIVE 14d가 BT의 어느 percentile? (Diagnostic 3 답)
- F v2 cycle slippage 외 다른 BT-LIVE divergence source?
- 봇 운영 13일 데이터 + 4월 26일 hedge-mode 휴면 11 missed entries 포함 시 정확한 BT-LIVE 갭?
- Retraining BT with intrabar data is feasible? (5m data 720d 보유)
- Strategy shelve vs redesign 결정 기준?

---

## 8. Reference

- [bot_halt_20260427.md](../../../../.claude/projects/C--Users-J-OneDrive-CLAUDE-CODE-FIN/memory/bot_halt_20260427.md) — Halt rationale + lessons
- [BACKTEST_LIVE_PARITY.md](../../../claudedocs/BACKTEST_LIVE_PARITY.md) — 22-item parity (이론적 정합성)
- [selection_variance_20260426.md](../../../../.claude/projects/C--Users-J-OneDrive-CLAUDE-CODE-FIN/memory/selection_variance_20260426.md) — N=1 adoption 37.3%
- [results/c1_breakout_state.json](../../../results/c1_breakout_state.json) — 46 trades raw data
