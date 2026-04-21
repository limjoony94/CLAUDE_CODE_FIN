# Plan: Slippage 원인 진단 (LIVE 체결 데이터 정량 분해)

> **Feature**: slippage_diagnosis
> **Date**: 2026-04-21
> **Phase**: Plan
> **Scope**: 측정·분해 중심의 진단 PDCA (해결책은 별도 후속 PDCA)
> **기반**: `dd_comparison_20260421_235028.json` (BT +9.37% vs LIVE -15.86%, 갭 25.23pp), `bt_live_gap_20260419` memory

---

## 1. Background

### 결정적 관찰 (2026-04-21)

27 trades, 신호 100% 일치 (BT 27 ↔ LIVE 27), 같은 10일 구간에서:

| | BT (compound 3x) | LIVE (compound 3x) | 갭 |
|---|---|---|---|
| Net | **+9.37%** | **-15.86%** | **-25.23pp** |
| WR | 37.0% | 25.9% | -11.1pp |
| End Balance ($2100 start) | $2,296.75 | $1,766.86 | -$529.89 |
| MDD | -11.53% (복구됨) | -15.86% (현재 진행) | -4.33pp |

### Fix 효과 미미

BUG#62~65 (v4.7.8, 2026-04-18) 후에도 per-trade 갭이 -0.94pp → -1.06pp로 **개선 없음**:

| 구간 | Trades | LIVE 3x 합 | BT 3x 합 | per-trade 갭 |
|---|---|---|---|---|
| Pre-fix (4/12~18) | 19 | -11.87% | +6.01% | **-0.94pp** |
| Post-fix (4/19~21) | 8 | -4.32% | +4.15% | **-1.06pp** |

**→ fix들은 state/PnL 기록 정확성은 개선했으나 slippage 자체를 줄이지 못함.**

### 핵심 공백: Slippage 구성요소별 정량화 부재

현재 memory의 `bt_live_gap_20260419`는 추정치(attribution heuristic)로 분해:
- Entry MARKET: 1.50pp (19%)
- Exit STOP_MARKET: 2.50pp (32%)
- Intrabar TRAIL vs SL 혼동: 0.80pp (10%)

하지만 이는 **측정 기반이 아닌 가정 기반 분해**. Production에서 이미 기록하고 있는 데이터:
- `Slippage: -0.017% (signal=75499.2 fill=75486.3)` (Entry, BUG#38 fix)
- `exit_slippage_pct` (Exit, BUG#65 fix — 27 trades 중 1건만 기록됨)

**→ 실제 trade별 slippage 측정 데이터로 진짜 attribution 도출 필요.**

### 체결 경로별 슬리피지 발생 지점 (현행)

| 경로 | Order Type | Slippage 측정 중? | 갭 원인 후보 |
|---|---|---|---|
| Entry | MARKET | ✅ (BUG#38) | Spread, depth |
| SL hit | Exchange STOP_MARKET (자동) | ❌ | STOP → MARKET 변환 시 점프 |
| Trail (pre-activation) | TRAILING_STOP_MARKET (callback=0.4~0.7%) | ❌ | Callback re-arm 지연, mid-bar trigger |
| Trail (post-activation) | Baton-touch STOP_MARKET (BUG#61) | ❌ (BUG#65 부분 기록) | STOP → MARKET 변환 |
| Emergency | MARKET (bot 내부) | ❌ | 없음 (해당 구간 미발생) |

---

## 2. Goal

**LIVE 27 trades 각각의 slippage를 4-way attribution으로 정량화하고, 다음 PDCA의 타겟 컴포넌트를 결정한다.**

출력물:
1. `results/slippage_attribution_{date}.json` — trade별 4 슬리피지 성분
2. `claudedocs/slippage_diagnosis_{date}.md` — 통계 + 해석 + 차기 PDCA 트리거
3. 결정 트리: 주원인 컴포넌트 → 대응 PDCA 제안 (예: `post_activation_only_trail`, `limit_entry_order`, `emergency_sl_tightening`)

**Non-Goal**: Slippage 자체 감소 (본 PDCA는 진단). 해결책은 결과 기반 후속 PDCA.

---

## 3. Hypotheses

| 가설 | 내용 | 검증 방법 |
|---|---|---|
| **H1** | Entry MARKET slippage는 평균 |0.05%| 미만 → 갭 기여 < 10% | 27 trades entry slippage 분포 |
| **H2** | **Pre-activation TRAILING_STOP_MARKET callback re-arm이 주원인** (40%+ 기여) | Pre/post-activation exit별 slippage 분리 측정 |
| **H3** | SL STOP_MARKET은 ATR 연동 (high vol에서 trigger 시 큼) | ATR vs SL slippage 회귀분석 |
| **H4** | Post-activation baton STOP_MARKET slippage는 < 0.1% (가정 일치) | 해당 exit들의 `exit_slippage_pct` |
| **H5** | Exit reason 분류의 BT-LIVE 불일치 존재 (LIVE TRAIL이 BT SL로 찍힘 등) | 27 trades reason match rate |

---

## 4. Success Criteria (GO 조건 6개)

1. **measurement_completeness**: 27 trades 중 최소 25건에 대해 entry/exit slippage 둘 다 정량 확보
2. **attribution_sum**: 4-way 분해 합이 LIVE-BT 갭의 ≥ 80% 설명 (잔여 unexplained ≤ 20%)
3. **hypothesis_resolution**: H1~H5 중 ≥ 3건이 데이터로 확정 또는 기각
4. **component_ranking**: 주원인 컴포넌트 1~2개 명확 식별 (기여도 ≥ 40%)
5. **next_pdca_trigger**: 주원인별 해결 PDCA 후보 문서화 (최소 2안, cost-benefit 포함)
6. **reproducibility**: 측정 로직이 config-driven하여 30일 후 재평가 가능

5/6 이상 GO. 단, 1, 2 중 하나라도 실패 시 샘플 확보를 위한 대기 (30일 후 재개).

---

## 5. Methodology

### 5.1 데이터 소스

**1차: 기존 로그 + state.json**
- `logs/c1_breakout.log.2026-04-*`: `Slippage:` 라인 추출 (Entry)
- `results/c1_breakout_state.json`: trade_history 27건 (exit_price, exit_slippage_pct 일부)

**2차: CCXT 추가 fetch (필요 시)**
- 각 trade의 exit 시점 ±30초 구간 BingX orderbook + 1m OHLCV
- 실제 체결 시점 trade 조회 (`fetch_my_trades`)

### 5.2 Slippage 성분 정의

각 LIVE trade에 대해 4-way 분해:

| 성분 | 정의 | 계산 |
|---|---|---|
| **s_entry** | Signal price → MARKET 체결가 차이 | `(fill - signal) / signal * 100 * direction_sign` |
| **s_exit_execution** | 거래소 trigger price → 실제 체결가 차이 | `(fill - trigger) / trigger * 100` |
| **s_bar_resolution** | BT가 가정한 bar close exit vs LIVE intrabar exit 차이 | `(live_exit - bt_exit_at_same_event) / live_exit * 100` |
| **s_reason_mismatch** | BT reason ≠ LIVE reason으로 인한 구조적 차이 | Binary flag × BT/LIVE exit price delta |

합산: `s_total = s_entry + s_exit_execution + s_bar_resolution + s_reason_mismatch`

### 5.3 분석 스크립트 설계

`scripts/analysis/slippage_attribution.py` (신규):

```python
# Pseudocode
for live_trade in load_live_trades():
    # 1. Entry slippage (from log)
    s_entry = parse_log_slippage(live_trade['entry_time'])
    
    # 2. Exit slippage — trigger vs fill
    exit_order = fetch_exchange_order_history(live_trade['exit_time'])
    s_exit_exec = (exit_order.fill_price - exit_order.trigger_price) / trigger * 100
    
    # 3. Bar resolution — compare to BT for same signal
    bt_trade = match_bt_trade(live_trade, bt_results)
    s_bar_res = (live_trade['exit'] - bt_trade['exit']) / live_trade['exit'] * 100
    
    # 4. Reason mismatch flag
    s_reason = 1 if live_trade['reason'] != bt_trade['reason'] else 0
    
    attribution[trade_id] = {
        's_entry': s_entry, 's_exit_exec': s_exit_exec,
        's_bar_res': s_bar_res, 's_reason': s_reason,
        's_total': sum(all), 'gap_actual': live_pnl - bt_pnl
    }
```

### 5.4 분석 결과 포맷

```
Slippage Attribution (27 trades, 3x leverage)
==================================================
Component          Mean      Stdev    Σ Impact   % of Gap
s_entry            X.XX%     X.XX%    X.XXpp     XX%
s_exit_execution   X.XX%     X.XX%    X.XXpp     XX%
s_bar_resolution   X.XX%     X.XX%    X.XXpp     XX%
s_reason_mismatch  X/27              X.XXpp     XX%
---------------------------------------------------
TOTAL                                 X.XXpp     100%
Observed gap                          25.23pp

By exit reason (pre vs post activation):
- Pre-activation TRAILING: N trades, mean s_exit X.XX%
- Post-activation baton:   N trades, mean s_exit X.XX%
- SL hit:                  N trades, mean s_exit X.XX%
```

---

## 6. Implementation Plan

### Phase 1: 데이터 수집 (0.5일)
1. 로그 파일 9개 (`c1_breakout.log.2026-04-12` ~ `2026-04-20`) + 현행 로그 병합
2. 정규식으로 `Slippage:`, `ENTRY`, `EXCHANGE_(SL|TRAIL)`, `TRAIL_TP` 라인 추출
3. `bot_pos_id` 수준으로 정렬하여 trade별 레코드 생성

### Phase 2: CCXT 보강 fetch (0.5일)
1. `fetch_my_trades('BTC/USDT:USDT', since=04-12)`로 exchange 체결 이력 전체 수집
2. Trigger price vs fill price 추출 (BUG#65 fallback 동일 로직)
3. Entry/Exit 시점 ±30초 ticker 데이터 수집 (spread 측정)

### Phase 3: Attribution 계산 (0.5일)
1. 4-way 분해 per trade
2. BT trade list (`dd_comparison_20260421` 재사용)와 1:1 매칭
3. 통계: mean, stdev, median, distribution per component

### Phase 4: 가설 판정 + 차기 PDCA 후보 (0.5일)
1. H1~H5 data-driven 판정
2. 주원인 1~2개 식별
3. 해결 PDCA 후보 문서화:
   - **A안 (H2 성립 시)**: `post_activation_only_trail` — Pre-activation 구간에도 baton STOP_MARKET 사용
   - **B안 (H1 성립 실패 시)**: `limit_entry_order` — MARKET → LIMIT (1 tick better) 전환 실험
   - **C안 (H5 성립 시)**: BT 모델에 reason classification fix
4. 6 GO 조건 평가

### Phase 5: 결과 정리 및 commit (0.5일)
1. `results/slippage_attribution_{date}.json`
2. `claudedocs/slippage_diagnosis_{date}.md`
3. Memory write: `slippage_diagnosis_20260421.md`
4. git commit + push

**총 2.5일 예상.**

---

## 7. Non-Goals

- 실제 slippage 감소 조치 (별도 PDCA)
- LIMIT order 구현 (본 PDCA의 후속)
- BT 모델 intrabar 개선 (intrabar_parity PDCA가 별도 존재)
- BingX 스펙 문의/지원 요청 (가능하면 반영, 본 PDCA 범위 외)

---

## 8. Rollback

본 PDCA는 **분석 전용** — 프로덕션 코드 변경 없음. Rollback 불필요.

- `scripts/analysis/slippage_attribution.py` 생성만, 실행 1회성
- config 변경 없음
- 기존 bot 코드 건드리지 않음

---

## 9. Risks

| 리스크 | 완화 |
|---|---|
| `fetch_my_trades` rate limit | 7일치 데이터 한 번에 fetch, 재시도 로직 |
| Exchange order history 일부 expire | 가능한 범위만 수집 + 갭은 unexplained로 남김 |
| Sample size 27이 통계적으로 부족 | Mean ± stdev만 보고, 개별 outlier 분석 병행 |
| BUG#65 미적용 구간 `exit_slippage_pct` 부재 | `fetch_my_trades`로 사후 복원, 실패 시 로그 기반 추정 |
| Reason mismatch 판정의 ambiguity (pre-activation TRAIL이 BT에선 TRAIL_TP) | 명시적 mapping table 작성 |

---

## 10. Reference

- `results/dd_comparison_20260421_235028.json` — BT vs LIVE 27 trades (본 PDCA의 직접 근거)
- `claudedocs/bt_live_gap_deep_review_20260419.md` — 이전 갭 분석 (가정 기반)
- `claudedocs/BACKTEST_LIVE_PARITY.md` — 22-item 체크리스트
- `memory/bt_live_gap_20260419.md` — 기존 추정 분해
- `memory/backtest_live_parity_20260418.md` — BUG#62~65 fix 이력
- `scripts/production/c1_breakout/bot.py:660` — `_resolve_ghost_exit` (체결가 복원 로직, fetch_my_trades 활용 참고)
- `logs/c1_breakout.log.2026-04-12` ~ `2026-04-20` — 9일치 원시 로그

---

## 11. Connection to Other PDCAs

- **Upstream (근거 제공)**: `dd_comparison_20260421` (본 PDCA 직전), `bt_live_gap_20260419` (초기 가설)
- **Downstream (본 PDCA 결과로 촉발)**: 
  - `post_activation_only_trail` (H2 성립 시)
  - `limit_entry_order` (H1 기각 시)
  - `intrabar_parity` (진행 중, H3 결과 참고)
- **Parallel**: `progressive_trail` (v4.8.0 enabled=false 상태, 본 진단 결과에 따라 활성화 재검토)
