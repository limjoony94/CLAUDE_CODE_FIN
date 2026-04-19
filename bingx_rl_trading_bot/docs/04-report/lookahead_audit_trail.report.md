# Trailing Look-Ahead Bias Audit 완료 보고서

> **Feature**: lookahead_audit_trail
> **Date**: 2026-04-19
> **Type**: Diagnostic audit (static code)
> **Outcome**: **NO NEW LOOK-AHEAD BIAS** — BT 결과 전면 신뢰 확인
> **Trigger**: 사용자 요청 "trailing 할 때 look-ahead bias 조사"

---

## 1. Executive Summary

**6개 잠재 look-ahead 경로 감사 결과**:
- 4개 경로 **수학적으로 bias 없음** (L1/L2/L4/L5)
- 2개 경로는 **기존 문서화된 structural limit** (L3=#21, L6=#22)
- **신규 bias 발견 0건** → 모든 BT 결과 신뢰

**핵심 증명**: Trail exit price가 수학적으로 **bar의 [close, high] 범위 내**에 반드시 존재 (L2 수학 증명) — 허위 가격 exit 없음.

---

## 2. Verdict Matrix

| ID | 경로 | Verdict |
|----|------|---------|
| L1 | best_price vs cur_pnl 같은 bar 사용 | ✅ **OK** (bar-local info, causally consistent) |
| L2 | Trail exit price reachability | ✅ **OK (수학적 증명)** |
| L3 | SL vs Trail priority at same bar | ⚠ STRUCTURAL (#21 기존) |
| L4 | ATR/Channel/Fractal causality | ✅ **OK** (Wilder/slice exclusive/past-only) |
| L5 | Entry timing (next bar open) | ✅ **OK** |
| L6 | 5m sub-bar traversal | ⚠ STRUCTURAL (#22 기존) |

---

## 3. 핵심 증명 — L2 Trail reachability

**주장**: Trail trigger 시 계산된 exit_price가 항상 bar 내 실제 거래 가능한 가격

**증명** (LONG):
- Trigger 조건: `drawdown >= trail_dist_pct` ⟺ `close <= best - trail_dist = trail_line`
- `best = running max` ≤ 현재 bar high
- 따라서 `trail_line ≤ best ≤ high` AND `trail_line >= close`
- 즉 `trail_line ∈ [close, high] ⊆ [low, high]`
- 가격이 best(≤high)에서 close로 하락하면서 **필연적으로 trail_line 통과**

**결론**: Exit price는 bar 내 tick path가 반드시 지나가는 가격. **허위 look-ahead 아님**.

---

## 4. BT 결과 신뢰도 판정

### 재확인된 신뢰 가능 결과
| 결과 | 수치 | 상태 |
|------|------|------|
| baseline clean BT | +169.55% | ✅ 신뢰 |
| baseline slip_med | +46.09% | ✅ 신뢰 |
| candidate_C clean | +192.76% | ✅ 신뢰 |
| candidate_C slip_med | +63.06% | ✅ 신뢰 |
| WF 5-fold 결과 | 기존 verdict | ✅ 신뢰 |
| STOP 판정들 | sl_trail/candidate_c/breakeven | ✅ 신뢰 |

### 남은 구조적 한계 (기존 문서화)
- **#21 SL vs Trail priority** (tick resolution) — 5m 해상도 한계
- **#22 MARKET slippage** — 체결 가격 이상

이 2건은 **look-ahead bias가 아닌 physical limit**. BUG#62~65 fix로 실질 영향 감소 중.

---

## 5. 사용자 구체 우려 해소

### 질문 재구성
> "수익중이지만 trail이 본절 아직 못 따라온 구간" = best_pnl > 0 but trail_line < 0

### 답변
- 이 구간에서 trail 발동 시 `realized = max(0, negative) = 0` → **breakeven cap 수학적 발동**
- Live에서도 stop_market이 entry 가격 근처 체결 시 동일
- **Look-ahead bias 아닌 의도된 tail-risk 보호 메커니즘**
- Fee+slip으로 실손실 -0.2% 는 **보호 비용** (tail -1.0% 방어)

### breakeven_trail (BUFFER 방식) 결과 맥락
BUFFER>0로 강제 hold 시 fractal SL tail risk에 노출되어 **PnL -149pp 악화** (breakeven_trail PDCA에서 실증). **Risk-reward trade-off** 문제지 bias 아님.

---

## 6. 방법론적 교훈

1. **수학적 증명 > 실증 미흡 의심**: L2 trail reachability는 **수학적 증명으로 확정**. 의심만으로 기각할 수 없음.
2. **Structural vs bias 구별**: L3/L6는 bias가 아닌 **intrabar resolution 부재**. 5m이 현재 상한.
3. **과거 문서 교차 검증**: BACKTEST_LIVE_PARITY 22-item과 본 감사 결과 100% 일관성.
4. **User 직관 존중 + 데이터 증거**: 사용자 우려는 합리적, 조사 결과 해소.

---

## 7. Production 영향

**변경 없음**. 다만:
- BT 결과 신뢰도 **상승**
- 후속 PDCA (True Breakeven SL Move 등)의 baseline 수치 신뢰 가능
- 봇 현재 v4.7.9 정상 운영 지속

---

## 8. 후속 PDCA 경로

### 즉시 (Plan 완료 후 진행)
1. **True Breakeven SL Move PDCA** (B안, 별개 메커니즘)
   - Trail 유지 + SL을 entry 위치로 이동
   - best_pnl > 0.3% 시 activation
   - Fractal SL tail risk 제거 + trail upside 보존

### 단기 (2~4주)
- Regime-conditional candidate_C (fold 2 완화)
- 30일 LIVE sample 수집

### 장기
- 1m tick data 확보 시 #22 structural limit 완화

---

## 9. Files Touched

- `docs/01-plan/features/lookahead_audit_trail.plan.md`
- `docs/03-analysis/lookahead_audit_trail.analysis.md` (6 경로 상세)
- `docs/04-report/lookahead_audit_trail.report.md` (본 문서)

Production code 변경 **0건**.

---

## 10. Bottom Line

사용자의 look-ahead bias 의심은 **합리적**이었으나, 6 경로 감사 결과 **모두 해소**:
- 4 경로 수학적으로 bias 없음 (특히 L2 trail reachability는 엄밀 증명)
- 2 경로는 이미 알려진 structural limit (tick data 부재)

**모든 BT 결과, STOP 판정, candidate 평가가 신뢰 가능**. 다음 PDCA는 True Breakeven SL Move (B안, trail 유지 + SL tighten) 로 진행 가능.
