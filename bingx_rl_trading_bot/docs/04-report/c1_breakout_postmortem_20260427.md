# C1 Breakout v2.6 — Postmortem

**Date**: 2026-04-27
**Status**: 🪦 SHELVED
**Period**: 2026-04-12 ~ 2026-04-27 (LIVE 14 days)
**Decision**: Shelve, do not redesign (advisor recommendation)

---

## 1. Summary

C1 Breakout v2.6은 272일 BT에서 검증된 양수 전략이었으나, 14일 LIVE 운영에서 -12.86% (1x) / -38.58% (3x) 누적 손실. **BT가 예측한 분포 (P0 = -0.72%) 밖의 outcome**을 시장이 생성. Advisor 결론: 모델 자체의 foundation problem, parameter tuning으로 해결 불가. C1 폐기 결정.

| 지표 | BT theoretical (272d) | LIVE actual (14d) | Gap |
|------|----------------------|-------------------|-----|
| trades/day | 3.47 | 3.29 | -5% |
| **WR** | **39.7%** | **28.3%** | **-11.4pp** |
| avg PnL/trade 1x | +0.231% | -0.279% | **-0.51pp** |
| daily PnL 1x | +0.83% | -0.92% | -1.75pp |
| 14d cumulative 1x | (BT min -0.72%) | **-12.86%** | UNPRECEDENTED |

---

## 2. What BT Predicted

| 검증 | 결과 |
|------|------|
| Look-ahead Progressive | 10/10 PASS |
| Indicator causality | ATR/Channel/Fractal causal |
| MC Direction (999 sims) | p=0.0000 DISC |
| WF 5-fold | 5/5 PASS, OOS +153.9% |
| 3-Way Split | Train+61, Valid+54, Test+55 ALL PASS |
| Param grid (60 combos) | 60/60 양수 |
| Rolling 60d | 5/5 양수 |
| Bootstrap 95% CI | [+109%, +234%] |
| Regime (High/Low vol) | 둘 다 양수 |
| Purged CV 5-fold | 5/5 PASS |
| **Rolling 14d** (272d 939 windows, post-mortem 측정) | **min -0.72%, P0 = -0.72%, 99.5% positive** |

→ BT 모든 검증 통과. 939개 14d windows 중 -5% 도달 0건.

## 3. What LIVE Delivered

| Reason | n | sum 3x | avg 3x | avg 1x |
|--------|---|--------|--------|--------|
| EXCHANGE_SL | 14 | -21.77% | -1.55% | -0.52% |
| EXCHANGE_TRAIL (pre-F v2) | 21 | -11.72% | -0.56% | -0.19% |
| TRAIL_TP (F v2) | 11 | -5.09% | -0.46% | -0.15% |
| **Total** | **46** | **-38.58%** | **-0.84%** | **-0.28%** |

→ **모든 exit type bleed**. SL은 가장 큰 단일 cost (-21.77% / 46 trades). Trail variants도 평균 음수.

---

## 4. Diagnostic Findings (3개 종합)

### D1: WR Gap — Intrabar SL Replay (`results/diag1_intrabar_sl_*.json`)

```
14 LIVE SL hits 분석:
  Wicked (recover > 50% sl_pct OR crossed back to entry): 12/14 (85.7%)
  Crossed back to entry: 8/14
```

→ SL이 BTC 5m noise에 너무 tight. BT는 high/low로 SL 잡으므로 LIVE-divergence 아니지만, **전략 fundamental 설계 문제** — SL 거리 < BTC noise 폭.

### D2: Trail Variants Bleed — Best PnL Distribution (`results/diag2_best_pnl_*.json`)

```
BT 943 TRAIL_TP   : avg +0.37% 1x, cap rate 54%
LIVE 7 TRAIL_TP   : avg +0.32% 1x (no-slip), cap rate 28.6%
Gap (no-slip)     : -0.05pp
```

→ **Trail 자체 설계는 BT와 비슷**. F v2 cycle MARKET slippage가 dominant cause:
- avg slippage -0.51% 1x, n=6 TRAIL_TP samples
- LIVE 실제 fill 기준 = +0.32 - 0.51 = **-0.19% 1x** (-0.30 fee 포함 시 -0.49%)

### D3: Regime Mismatch — Rolling 14d Distribution (`results/diag3_rolling_14d_*.json`)

```
272d BT, 939 rolling 14d windows:
  Mean +11.05%, std 6.52
  P5 +2.18%, P25 +6.31%, P50 +10.05%, P95 +23.52%
  Min -0.72%, Max +32.74%
  Positive rate: 99.5%

LIVE 14d: -12.86%
BT windows below LIVE: 0/939 (0.00%)
```

→ **🚨 LIVE -12.86%은 BT 분포 P0 미만**. BT는 "이런 outcome은 거의 불가능"이라고 했지만 LIVE 첫 14일에 발생. **Foundation problem**.

---

## 5. Why This Failed (Advisor Diagnosis)

### Priority 1: Compounded Slippage (BT zeros out)

BT는 slippage 0% 가정. 실제 LIVE:
- Entry MARKET ~0.03% (favorable / unfavorable mixed)
- Cycle TRAIL_TP MARKET ~-0.51% (systematic adverse, n=6)
- 46 trades × ~0.3% systematic = **-13.8%/14d execution friction**

BT theoretical edge: +0.83%/day = +11.62%/14d.
**Friction floor (-13.8%) > Strategy edge (+11.6%)** → 음수.

### Priority 2: Regime Non-stationarity

WF 5-fold는 모두 2025-07~2026-04 distribution에서 sample. LIVE 2026-04-12~27은 새 distribution에서 sample 가능성. BT 999개 MC simulation이 same period로부터 generated되어 진정한 OOS 검증 안 됨.

### Priority 3: Look-Ahead (Less Likely)

Parity 20/22 audit됐지만 P0 miss 크기는 가능성을 leave open. 단 D1/D2의 mechanism으로 대부분 설명되므로 lower priority.

---

## 6. Why Redesign Was Not Chosen

| 시도 가능 fix | 예상 회복 | 잔존 문제 |
|--------------|----------|----------|
| Wider SL (max_sl 4.5 — 이미 적용) | WR 회복 안 됨 (D1 측정) | 손실 trail로 migrate |
| F v3 LIMIT close | trail slippage ~3-5pp 회복 | -12.86% → -8 ~ -10% (여전 음수) |
| Intrabar BT 재구축 + 새 WF | 이론상 LIVE-realistic | 2-4주 engineering, signal alpha 보장 X |

> "Rebuilding BT framework before knowing signal works is likely more weeks for a coin flip." — Advisor

→ Multiple structural issues 동시 발생, simple fix 불가. Signal 자체의 LIVE alpha 미확인. 추가 투자 가치 불명.

---

## 7. Lessons Learned (Cross-Cutting)

### Process Lessons (이 PDCA loop)

1. **WF GO criterion은 hard threshold로 절대 reframe 금지** (`lessons_process_audit_20260425.md`)
   - max_sl 3.3→4.5 변경 시 WF 2/5만 통과한 것을 "moderate robust"로 reframe → process bug
2. **Single-variable change 원칙** — 24h 이상 간격
   - F v2 + max_sl 4.5 동시 적용 → cohort contamination
3. **모든 "X% recovery" 주장에 control group**
   - Stop hunt 97.1% 헤드라인은 control group 100% trivial이라 무의미
4. **Brake/Validation 분리**
   - Brake (catastrophe?) ≠ Validation (effective?). 둘 다 필요
5. **🚨 가장 중요**: **Distribution check이 cohort delta보다 우선**
   - "LIVE가 BT 분포 안에 있는가"가 30분 짜리 질문. -$623 막을 수 있었음
   - Advisor 본인 인정: cohort-relative thinking에 anchor한 framing 잘못

### Technical Lessons

1. **BT theoretical slippage 0 가정 위험**
   - Parity 20/22 통과해도 execution friction은 별개
   - 평균 -0.51% TRAIL_TP slippage = 일일 strategy edge 잠식
2. **WF가 same distribution sample이면 OOS 검증 약함**
   - 5 folds 모두 2025-07~2026-04 → 2026-04 LIVE는 새 distribution일 수 있음
   - Out-of-time + out-of-regime 별도 필요
3. **Channel breakout + 40% body filter는 LIVE alpha 보장 X**
   - BT에서 +218%/272d 양수, LIVE에서 음수
   - Signal 자체가 BT artifact일 수 있음

### Capital Cost

- 시작 추정 잔고: ~$2,118
- 종료 잔고: $1,495.22
- **총 -$623 (-29.4%)**
- 시간 비용: ~3주 PDCA loop (분석, 변경, 모니터링, halt)

---

## 8. Preserved Assets (Reference for Next Strategy)

| Asset | 위치 | 가치 |
|-------|------|------|
| 봇 코드 | `scripts/production/c1_breakout/` | BT-LIVE parity framework, BUG#1~66 fixes |
| BT scripts | `scripts/analysis/*_bt_*.py` | 재사용 가능 검증 도구 |
| Test suite | `scripts/tests/` (165 cases) | regression guard |
| Parity doc | `claudedocs/BACKTEST_LIVE_PARITY.md` | 22-item checklist |
| BUG_HISTORY | `claudedocs/BUG_HISTORY.md` | BUG#1~66 학습 자료 |
| MEMORY entries | `.claude/projects/.../memory/` | process bug + technical lessons |
| Git tag | `v4.9.0-halted` | snapshot for archeological reference |

---

## 9. What's NOT Decided Yet (User-level)

- 다음 전략 class (다른 indicators, timeframe, asset)?
- Manual research mode (no live capital)?
- Trading 자체 중단?
- 같은 framework 재사용 vs 처음부터?

→ **Capital risk 0, time pressure 0**. 천천히 결정.

⚠️ **회피해야 할 anxiety pattern** (advisor 명시):
- "다음에는 intrabar BT부터 시작" — 같은 anxiety pattern
- "더 많은 rigor 추가하면 실패 안 함" — 함정
- 새 전략 즉시 시작 — humility 필요

---

## 10. References

- [bot_halt_20260427.md](../../../../.claude/projects/C--Users-J-OneDrive-CLAUDE-CODE-FIN/memory/bot_halt_20260427.md)
- [diagnostic_mode_20260427.md](../01-plan/features/diagnostic_mode_20260427.md)
- [BACKTEST_LIVE_PARITY.md](../../claudedocs/BACKTEST_LIVE_PARITY.md)
- [BUG_HISTORY.md](../../claudedocs/BUG_HISTORY.md)
- Diagnostic results:
  - `results/diag1_intrabar_sl_20260427_172107.json`
  - `results/diag2_best_pnl_20260427_172030.json`
  - `results/diag3_rolling_14d_20260427_172030.json`
- LIVE state: `results/c1_breakout_state.json` (46 trades raw)

---

## 11. Final Word

> "I had reasonable tools, I had a passing BT, and reality didn't match. The next strategy needs the same humility about LIVE deployment, not 10x more upfront engineering." — Advisor

C1 Breakout v2.6은 BT 기준 정상 검증된 전략이었지만 LIVE에서 작동하지 않았다. 이 실패는 도구의 부족이 아닌 **BT-LIVE distribution 차이의 본질적 한계**를 보여준다. 다음 전략은 이 humility로 접근.
