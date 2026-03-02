# Gap Analysis: pattern_5m — Scanner vs Production Backtest Fidelity

**Date**: 2026-03-01
**Feature**: pattern_5m (Distribution TP study + Scanner-Production backtest gap)
**Analyst**: Claude Opus 4.6

---

## 1. Distribution TP Study 결과

### Verdict: **INFORMATIONAL** (Percentile 유지, Distribution은 옵션으로 보존)

| Phase | 결과 |
|-------|------|
| Phase 1: Fitting | Log-normal good+acceptable **97.7%** — MFE 데이터에 적합 |
| Phase 2: TP Range | Distribution 100% 더 넓은 탐색 (10.9개 vs 5개) |
| Phase 3: IS | Pctile **+1,385%** vs Dist +819% (44pat, MDD 19.4%) |
| Phase 4: WF OOS | Pctile **+872.7%** vs Dist +280.7% (3/3 PASS both) |
| Phase 5: Sensitivity | 20pt grid → 90pat PnL/MDD **76.1x** (IS, WF 미검증) |

### Distribution 열위 원인 분석
- Log-normal ISF가 보수적 TP 도출 (median **1.10%** vs percentile 1.70%)
- 작은 TP → 높은 hit rate(88.1%)이지만 per-trade 수익 부족
- PnL/MDD scoring이 자연스럽게 percentile 쪽 선호 (더 큰 TP = 더 큰 per-trade 수익)
- **44패턴 vs 137패턴** — distribution이 더 엄격한 필터

### 향후 연구 가치
- 20pt 고해상도 + BWR 75% 조합 → IS PnL/MDD 76.1x (WF 검증 필요)
- Gamma/Weibull 분포 (81.2%에서 log-normal보다 우수)
- 현재는 `--tp-method distribution` 옵션으로 유지

---

## 2. 핵심 갭: Scanner 백테스트 ≠ Production 현실

### Match Rate: **~35%** (Critical Gap)

사용자 지적 핵심: **Scanner 백테스트가 프로덕션 실행 조건을 전혀 반영하지 않음**.

### 2.1 Scanner 백테스트 현황 (현재)

```
bt_signals() / bt_signals_atr():
  - 단일 패턴 독립 백테스트 (패턴 간 상호작용 없음)
  - 무제한 동시 포지션 가능 (overlap 검사 없음)
  - 100% capital per trade (compound 아님)
  - 필터 없음 (regime, momentum, aggregate risk, loss burst 등)

portfolio_1pos():
  - 사후 1-position-at-a-time 필터 (시간순 정렬 후 overlap 제거)
  - 전체 패턴 trades를 합산 후 단일 포트폴리오로 처리

calc_stats():
  - 단순 PnL 합산 (additive, not compound)
  - MDD = peak - trough (additive % space)
```

### 2.2 Production 현실 (config + bot.py)

```
Production 실행 조건:
  - N=9 가상 슬롯, 1/N=11.1% sizing per slot
  - Hedge mode (LONG/SHORT 독립)
  - Direction Cap=7 (max 7 same-direction)
  - Position Timeout=864 bars (72h)
  - Regime Sizing: counter-regime ×0.3 (EMA20 slope)
  - Aggregate Risk Cap: counter 3%, with 7% SL exposure
  - Momentum Guard: BTC >1%/30min → 역방향 30min 차단
  - Loss Burst Brake: 동일 방향 2회 손실/24h → 12h 차단
  - MDD Sizing: DD 5%→full, 20%→25% 선형
  - Compound sizing (equity 기반)
  - Cooldown: 30초
  - Daily loss limit: 13%
```

### 2.3 갭 매트릭스

| 기능 | Scanner | Production | 영향도 | 구현 난이도 |
|------|---------|-----------|-------|-----------|
| **N slots** | 1pos filter (사후) | N=9 동시 | **CRITICAL** | Medium |
| **1/N sizing** | 100% capital | 11.1% per slot | **CRITICAL** | Easy |
| **Direction Cap** | 없음 | Max 7 same-dir | HIGH | Easy |
| **Compound equity** | Additive | Compound | HIGH | Medium |
| **Regime Sizing** | 없음 | ×0.3 counter | MEDIUM | Medium |
| **Aggregate Risk Cap** | 없음 | 3%/7% SL cap | MEDIUM | Medium |
| **Momentum Guard** | 없음 | 1%/30min block | LOW | Easy |
| **Loss Burst Brake** | 없음 | 2-loss/24h block | LOW | Easy |
| **MDD Sizing** | 없음 | DD→size reduction | LOW | Medium |
| **Position Timeout** | DROP (no count) | Market close | MEDIUM | Already partial |
| **Hedge mode** | Direction-aware | LONG/SHORT 독립 | LOW | Already correct |
| **Cooldown** | 없음 | 30s inter-trade | NEGLIGIBLE | - |

### 2.4 정량적 영향 추정

기존 연구 데이터 기반 추정:

| 지표 | Scanner IS | Scanner WF OOS | Live 22d |
|------|-----------|----------------|----------|
| WR | 95.0% | 88.8% | **52.7%** |
| Edge/trade | ~1.0% | ~0.8% | **0.04%** |

**IS → OOS 드롭**: WR -6.2pp, Edge ~20% 감소 (정상적 OOS decay)
**OOS → Live 드롭**: WR **-36.1pp**, Edge **95% 감소** (비정상)

이 거대한 Live 갭의 주요 원인:
1. **N=9 동시 포지션 + 방향 집중** → 1 BTC 움직임에 다수 동시 SL (correlated loss)
2. **Edge decay** (half-life 30d) — scanner는 IS 전체 기간 최적화, live는 최근 데이터
3. **1-pos filter** ≠ N-pos 현실 — scanner가 overlap 제거로 correlated loss 과소평가
4. **Additive PnL** ≠ compound — 실제 1/N sizing의 compound 효과 미반영

---

## 3. 권장 액션

### Priority 1: N-Position Portfolio Simulator (CRITICAL)

Scanner에 `portfolio_npos()` 함수 추가:

```python
def portfolio_npos(all_trades, n_slots=9, direction_cap=7):
    """
    N-slot concurrent position simulator.
    - Sort all trades by entry bar
    - Track open slots (entry_bar, exit_bar, direction, pnl)
    - 1/N sizing per slot
    - Direction cap enforcement
    - Compound equity tracking
    """
```

**핵심**:
- `all_trades`를 시간순 정렬 → 각 entry 시점에서 open slots 확인
- slot 부족하면 SKIP (현재 portfolio_1pos와 유사하지만 N개)
- 동일 방향 cap 초과면 SKIP
- PnL은 `pnl_pct * (1/N) * current_equity` (compound)
- MDD는 equity curve의 peak-to-trough

### Priority 2: Regime-Aware Sizing (MEDIUM)

ATR ratio처럼 EMA(20) slope를 precompute → 백테스트에서 counter-regime trades의 sizing ×0.3

### Priority 3: Aggregate Risk Cap (MEDIUM)

N-pos simulator에 방향별 SL exposure 합산 → cap 초과 시 진입 skip

### Priority 4: Entry Filters (LOW)

Momentum Guard, Loss Burst Brake — 효과가 작으므로 후순위

---

## 4. 구현 계획

### Phase 1: `portfolio_npos()` + `calc_stats_compound()`
- Scanner에 N-pos portfolio simulator 추가
- compound equity curve + 1/N sizing
- direction cap 파라미터
- `--n-slots` CLI 옵션

### Phase 2: WF에 N-pos 적용
- `expanding_window_wf()`의 OOS 평가를 `portfolio_npos()`로 교체
- IS scan은 개별 패턴 평가 유지 (패턴 선별은 개별)
- 포트폴리오 수준 검증만 N-pos로

### Phase 3: Regime + Risk Cap
- EMA slope precompute → sizing multiplier
- aggregate_risk_cap logic in portfolio simulator

### Phase 4: 연구 스크립트
- `npos_portfolio_study.py`: 1-pos vs N-pos 비교 + WF 검증
- GO/STOP 판정: N-pos WF가 더 보수적(현실적) 결과 제공하는지

---

## 5. 검증 기준

| 체크 | 기준 |
|------|------|
| N-pos WF OOS WR | Live WR (52.7%)에 가까워야 함 |
| N-pos MDD | Live MDD보다 과소추정하지 않아야 함 |
| 기존 테스트 | 1061+ tests pass |
| 기존 1-pos 경로 | `--n-slots 1`로 100% 보존 |
| 패턴 선별 | 개별 패턴 edge/MC는 변경 없음 |

---

## 6. 요약

| 영역 | Match Rate | 상태 |
|------|-----------|------|
| Distribution TP 구현 | **100%** | ✅ 완료, INFORMATIONAL verdict |
| Scanner 개별 패턴 평가 | **90%** | ✅ ATR, MC, WF 잘 반영 |
| **Scanner ↔ Production 포트폴리오** | **~35%** | ❌ CRITICAL GAP |
| Production 모듈 변경 | N/A | 변경 없음 (JSON 소비만) |

**Overall Match Rate: ~60%** — N-pos portfolio simulator가 해결되면 ~85% 예상.
