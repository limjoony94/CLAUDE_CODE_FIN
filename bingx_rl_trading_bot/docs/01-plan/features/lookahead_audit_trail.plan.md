# Plan: Trailing Look-Ahead Bias Audit

> **Feature**: lookahead_audit_trail
> **Date**: 2026-04-19
> **Phase**: Plan
> **Type**: Diagnostic audit (read-only)
> **Trigger**: 사용자 요청 — "trailing 할 때 look-ahead bias가 존재할 수 있음 관련 조사"

---

## 1. Background

BT 결과(baseline slip +46%, candidate_C +63%)가 실제 edge인지, 아니면 look-ahead bias로 인한 허상인지 체계적 검증 필요. 특히 **trailing** 영역에서 다음 가능성:

1. **같은 15m bar의 high·close 동시 사용** → tick path 순서 무시 가능
2. **best_price 업데이트가 bar 완료 정보에 의존** → bar 시작 시점에 알 수 없음
3. **Trail exit price가 실제 tick path에 존재했는지 검증 안함** → bar 내 reachability 문제
4. **SL vs Trail priority 가정이 intrabar 순서 무관**
5. **ATR/Channel/Fractal 지표가 현재 bar 정보 포함 여부**
6. **5m sub-bar 모드의 intrabar 가정 정합성**

본 감사는 **코드 경로별 수학적 증명 또는 실증 반박**으로 각 의혹 해소.

---

## 2. Goal

1. **Trail 로직이 수학적으로 look-ahead free** 증명
2. 6개 잠재 경로 각각 verdict (BIAS / OK / STRUCTURAL LIMIT)
3. Bias 발견 시 수정 제안
4. BACKTEST_LIVE_PARITY 문서 업데이트

---

## 3. 감사 항목 (6가지 잠재 경로)

### L1 — Trail best_price 업데이트 순서
```python
pos['bp'] = max(pos['bp'], highs[i])  # bar i의 high 사용
cur_pnl = (closes[i] / entry - 1) * 100  # bar i의 close 사용
```
**질문**: 같은 bar의 high와 close 동시 사용이 bias인가?

### L2 — Trail trigger → Exit price reachability
- Trigger: `drawdown >= trail_dist_pct` (close 기준)
- Exit price: `entry × (1 + realized/100)` (수식적 가격)
- **질문**: 이 exit price가 bar의 [low, high] 범위 내에 반드시 존재하는가?

### L3 — SL vs Trail priority at same bar
- BT: SL → Emergency → Timeout → Trail 순 (코드 우선순위)
- 실제 tick path: 무작위 (e.g., high 먼저 vs low 먼저)
- **질문**: 우선순위 가정이 본질적 look-ahead인가?

### L4 — ATR/Channel/Fractal causality
- `compute_atr(highs, lows, closes, 14)`
- `compute_channel(highs, lows, 15)` — `shift(1)`로 causal 설계
- `compute_fractal_swings(highs, lows, 10)`
- **질문**: atr[i] 계산 시 bar i의 close 사용이 bias? channel[i]에 bar i의 high 포함?

### L5 — Entry timing
- Signal: bar i close 확인
- Entry: bar i+1 open
- **질문**: bar i close 시점에 bar i+1 open 정보를 미리 알 수 있는 BT 경로가 있는가?

### L6 — 5m sub-bar traversal 가정
- 15m bar = 3 × 5m sub-bars, 순차 순회
- best_price를 sub-bar 단위 update
- **질문**: 5m path가 실제 tick path를 대표하는가? 단일 5m bar 안에서 또 미세 path 가정?

---

## 4. Method

### 4.1 Static code audit (수학적 증명)
각 경로에 대해:
- 해당 함수 line-by-line 읽고 dependency 추적
- 시간축 정렬(t → t+1) 위반 여부
- Bar-local info만 사용하는지, future info 포함되는지

### 4.2 Empirical test (실증)
의심되는 경로에 대해:
- **Synthetic bar**: 알려진 tick path로 BT 실행 → exit price가 실제 가격 경로에 존재하는지 검증
- **Restricted BT**: future info를 인공적으로 지운 BT 결과와 비교
  - 예: close = (high+low+close)/3 로 교체 → drawdown 계산 변경 영향

### 4.3 Reference docs 대조
기존 `BACKTEST_LIVE_PARITY.md` 22-item의 L1~L6 관련 row 재검토.

---

## 5. Hypotheses

| L | 예상 verdict |
|---|------|
| L1 | **OK** (bar-local info only, causally consistent) |
| L2 | **OK (수학적 증명)** - trail_line ∈ [close, high] 필연 |
| L3 | **STRUCTURAL LIMIT** (intrabar tick 모름, BACKTEST_LIVE_PARITY #21과 동일) |
| L4 | **OK** (Wilder ATR는 전 bar 포함 inclusive 표준, causal) |
| L5 | **OK** (opens[i+1] 참조가 bar i close 이후 시점) |
| L6 | **STRUCTURAL LIMIT** (tick < 5m resolution 모름) |

---

## 6. Success Criteria

1. 6개 경로 각각 **verdict 명시** + 증거
2. True bias 발견 시 **구체적 수정 제안**
3. Structural limits는 `BACKTEST_LIVE_PARITY.md`와 일관되게 문서화
4. BT 결과 신뢰성 **최종 판정** (신뢰 가능 / 재검증 필요 / 폐기)

---

## 7. Implementation Plan

신규 스크립트 없음. **문서 중심 감사**:
- 코드 읽기 (Grep/Read)
- 수학 증명 (markdown)
- 필요 시 micro-test (on-the-fly Python)

### 예상 소요
- Static audit: 30~40분
- Empirical micro-test: 10분 (필요 시)
- 문서 작성: 20분
- **총 1시간 이내**

---

## 8. Non-Goals

- Production 코드 수정
- 새 BT 엔진 작성
- Candidate_C 재평가
- Tick-level simulation 재구현

---

## 9. Deliverable

- `docs/03-analysis/lookahead_audit_trail.analysis.md` — 6 경로 각 verdict + 증거
- `docs/04-report/lookahead_audit_trail.report.md` — 요약 + BT 신뢰성 판정
- 필요 시 `BACKTEST_LIVE_PARITY.md` 업데이트 (cross-ref)

---

## 10. Reference

- Core BT code: `scripts/analysis/c1_refined_validation.py`, `intrabar_trail_impact.py`
- Exit 로직: `scripts/production/c1_breakout/signals.py::check_exit`
- 관련 문서: `claudedocs/BACKTEST_LIVE_PARITY.md` (22-item)
- Structural #21 (tick resolution), #22 (MARKET slip) — 이미 확정된 한계
