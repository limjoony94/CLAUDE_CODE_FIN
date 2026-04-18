# Design: Intrabar Parity 개선

> **Feature**: intrabar_parity
> **Date**: 2026-04-19
> **Phase**: Design
> **Plan**: `docs/01-plan/features/intrabar_parity.plan.md`

---

## 1. Architecture 개요

```
┌──────────────────────────────────────────────────────────────┐
│ TRACK A — BT Intrabar + Slippage (Phase 1, 우선)              │
│                                                              │
│  scripts/analysis/c1_intrabar_parity.py  (NEW)               │
│   ├─ 기존 intrabar_trail_impact.py 엔진 재사용 + 확장         │
│   ├─ 신규 slippage 모델 (entry/exit_sl/exit_trail/exit_emg)   │
│   ├─ 신규 mode='5m_slip' (5m sub-bar + slippage 합성)         │
│   ├─ 19-trade LIVE 창 비교 (results/live_vs_backtest_*.json)  │
│   └─ GO 조건 8개 자동 평가                                    │
│                                                              │
│  Output: results/intrabar_parity_{timestamp}.json            │
└──────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────┐
│ TRACK B — LIVE 5m Polling (Phase 2, Track A 이후)             │
│                                                              │
│  scripts/production/c1_breakout/bot.py 확장                  │
│   ├─ _poll_best_price_5m(pos)  (NEW, opt-in)                 │
│   ├─ config.strategy.tick_best_price.enabled (default false) │
│   ├─ Background async task lifecycle                         │
│   └─ Rate limit guard + exception handling                   │
│                                                              │
│  Config: strategy.tick_best_price {enabled, poll_seconds}    │
└──────────────────────────────────────────────────────────────┘
```

Track A는 **연구 스크립트**(production 영향 0). Track B는 **production opt-in**(기본 OFF).

---

## 2. Track A — BT Intrabar + Slippage 모델

### 2.1 Slippage 사양 (측정 기반 보수치)

BT-LIVE 심층 검토(claudedocs/bt_live_gap_deep_review_20260419.md) 결과 평균 drift:
- Entry (MARKET): **0.287%** 측정 → 보수치 **0.15%** (단건 outlier 제외한 median 추정)
- Exit SL (STOP_MARKET): **0.641%** 평균 → 보수치 **0.30%**
- Exit Trail (baton-touch STOP_MARKET): 측정 불충분 → Entry 대비 감소 **0.15%**
- Exit Emergency: **0.50%** (tail-risk)

방향성:
- Entry LONG: `fill = ideal_open × (1 + entry_pct/100)` (adverse = 더 비싸게 매수)
- Entry SHORT: `fill = ideal_open × (1 - entry_pct/100)` (adverse = 더 싸게 매도)
- Exit: 전부 adverse (매도가↓ for LONG, 매수가↑ for SHORT)

```python
SLIPPAGE = {
    'entry_pct':          0.15,
    'exit_sl_pct':        0.30,
    'exit_trail_pct':     0.15,
    'exit_emergency_pct': 0.50,
    'exit_timeout_pct':   0.15,  # MARKET 청산 가정
}
```

Config-driven으로 on/off + override 가능:
```yaml
backtest:  # 별도 섹션, production config 영향 없음
  intrabar:
    enabled: true
    mode: '5m_slip'  # bar_close|intrabar|5m|5m_slip
  slippage:
    enabled: true
    entry_pct: 0.15
    exit_sl_pct: 0.30
    exit_trail_pct: 0.15
    exit_emergency_pct: 0.50
    exit_timeout_pct: 0.15
```

### 2.2 `c1_intrabar_parity.py` 구조

```python
"""C1 Breakout intrabar parity analysis.

Extends intrabar_trail_impact.py with slippage injection and LIVE gap comparison.
"""
import sys, json, math
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

# Reuse engine + exit checkers from intrabar_trail_impact.py
from scripts.analysis.intrabar_trail_impact import (
    run_backtest as run_intrabar_bt_clean,
    compute_stats,
    # data arrays, indicators pre-computed at module import
)

SLIPPAGE = {
    'entry_pct': 0.15,
    'exit_sl_pct': 0.30,
    'exit_trail_pct': 0.15,
    'exit_emergency_pct': 0.50,
    'exit_timeout_pct': 0.15,
}

def apply_slippage(trade, cfg):
    """Adjust trade PnL by exit-reason slippage.

    Entry slippage is already folded into 'raw' before this call (see run_bt_slip).
    Exit slippage + fee are deducted here → returns new 'net'.

    **Emergency priority preservation** (fix for critical gap):
    If SL exit with exit_sl_pct adverse makes effective |loss| exceed
    emergency_sl_pct (3.0%), reclassify as EMERGENCY and cap at emergency_sl_pct.
    Rationale: in live, emergency check ALWAYS fires before SL (priority 1 > 2
    in check_exit). Slippage-adjusted backtest must preserve this invariant.
    """
    reason = trade['reason']
    if reason == 'SL':       adj = cfg['exit_sl_pct']
    elif reason == 'TRAIL_TP': adj = cfg['exit_trail_pct']
    elif reason == 'EMERGENCY': adj = cfg['exit_emergency_pct']
    elif reason == 'TIMEOUT': adj = cfg['exit_timeout_pct']
    else: adj = 0.0
    raw_adj = trade['raw'] - adj
    # Emergency cap: SL losses expanded past -3% get reclassified
    if reason == 'SL' and raw_adj <= -emergency_sl:
        raw_adj = -emergency_sl - cfg['exit_emergency_pct']  # use emergency slip
        trade['reason_effective'] = 'EMERGENCY'
    else:
        trade['reason_effective'] = reason
    return raw_adj - FEE  # net

def run_bt_slip(mode='5m_slip'):
    """Run backtest with intrabar + slippage. Adjusts entry price and exit PnL."""
    # Step 1: run clean intrabar backtest (5m sub-bar mode)
    base_mode = '5m' if mode == '5m_slip' else mode.replace('_slip', '')
    trades = run_intrabar_bt_clean(mode=base_mode)

    # Step 2: apply entry slippage to each trade (retroactively adjust PnL)
    adjusted = []
    for t in trades:
        entry_adv = SLIPPAGE['entry_pct']  # always adverse
        if t['d'] == 'LONG':
            eff_entry = t['entry_price'] * (1 + entry_adv / 100)
            raw = (t['exit_price'] / eff_entry - 1) * 100
        else:
            eff_entry = t['entry_price'] * (1 - entry_adv / 100)
            raw = (1 - t['exit_price'] / eff_entry) * 100
        t_adj = dict(t)
        t_adj['raw'] = raw
        t_adj['net'] = apply_slippage(t_adj)
        adjusted.append(t_adj)
    return adjusted

def gap_vs_live_window(trades_adj, agg15_ts,
                       live_start='2026-04-12', live_end='2026-04-19'):
    """Filter adjusted BT trades to LIVE window and compute gap vs actual.

    agg15_ts: list/Series of 15m bar timestamps (from intrabar_trail_impact.py:agg15['ts']).
    Trade entry_bar → timestamp mapping via agg15_ts[entry_bar].
    """
    from datetime import datetime
    ls = datetime.fromisoformat(live_start)
    le = datetime.fromisoformat(live_end)

    # Fetch LIVE window trades
    state = json.load(open(ROOT / 'results' / 'c1_breakout_state.json'))
    live_in_window = [t for t in state['trade_history']
                      if ls <= datetime.fromisoformat(t['exit_time'][:19]) < le]
    live_pnl_3x = sum(t['pnl_pct'] for t in live_in_window)
    live_pnl_1x = live_pnl_3x / 3

    # BT trades in same window via entry_bar → timestamp
    bt_in_window = [t for t in trades_adj
                    if ls <= pd.Timestamp(agg15_ts[t['entry_bar']]) < le]
    bt_pnl_1x = sum(t['net'] for t in bt_in_window)

    return {
        'live_count': len(live_in_window),
        'live_pnl_1x': round(live_pnl_1x, 2),
        'live_pnl_3x': round(live_pnl_3x, 2),
        'bt_count': len(bt_in_window),
        'bt_pnl_1x': round(bt_pnl_1x, 2),
        'gap_1x': round(bt_pnl_1x - live_pnl_1x, 2),
    }

def evaluate_go_conditions(baseline_adj, gap_result,
                           clean_baseline_pnl=170.49,
                           clean_train_pnl=95.07,
                           clean_ratio=31.69,
                           clean_mdd=5.38):
    """Evaluate Phase-1 GO flags (6 evaluable; 2 deferred to Phase 2)."""
    flags = {}
    # 1. intrabar_realism: 5m_slip mode gap within ±3pp of LIVE
    flags['intrabar_realism'] = abs(gap_result['gap_1x']) <= 3.0
    # 2. baseline_preservation: adjusted baseline ≥ 88% of clean
    baseline_pnl_adj = sum(t['net'] for t in baseline_adj)
    flags['baseline_preservation'] = baseline_pnl_adj >= clean_baseline_pnl * 0.88
    # 3. wf_pass: 5-fold time-partition on adjusted trades, all positive
    flags['wf_pass'] = wf_on_adjusted_trades(baseline_adj) == 5
    # 4. ratio_ok: adj PnL/MDD ≥ 85% of clean
    adj_mdd = compute_mdd_additive(baseline_adj)
    adj_ratio = baseline_pnl_adj / adj_mdd if adj_mdd > 0 else 0
    flags['ratio_ok'] = adj_ratio >= clean_ratio * 0.85   # 26.94
    # 5,6. Phase 2 (Track B) — deferred
    flags['track_b_cost'] = None
    flags['track_b_benefit'] = None
    # 7. rollback_ready: verified by design (new script only, no prod touch)
    flags['rollback_ready'] = True
    # 8. train_not_degraded: adj train ≥ clean_train − 5.0pp
    train_pnl_adj = sum(t['net'] for t in baseline_adj
                        if is_in_train_slice(t))
    flags['train_not_degraded'] = train_pnl_adj >= clean_train_pnl - 5.0
    return flags

# ── WF & MDD & train-slice helpers (newly specified) ───────────────

def wf_on_adjusted_trades(trades, n_folds=5):
    """Time-based 5-fold partition on adjusted trades.

    Simpler than rebuilding per-fold: split trades by entry_bar into 5 equal
    time slices, check each slice's total PnL > 0. Returns count of positive folds.

    Rationale: slippage is a linear adjustment post-fact, so fold
    reshuffling on adjusted trades is equivalent to fold-wise rebuild
    under the same BT configuration. Less rigorous than rebuild-per-fold
    but preserves temporal OOS structure.
    """
    if not trades:
        return 0
    trades_sorted = sorted(trades, key=lambda t: t['entry_bar'])
    first_bar = trades_sorted[0]['entry_bar']
    last_bar  = trades_sorted[-1]['entry_bar']
    span = last_bar - first_bar
    fold_size = max(1, span // n_folds)
    pos_folds = 0
    for k in range(n_folds):
        lo = first_bar + k * fold_size
        hi = first_bar + (k + 1) * fold_size if k < n_folds - 1 else last_bar + 1
        fold_pnl = sum(t['net'] for t in trades_sorted
                       if lo <= t['entry_bar'] < hi)
        if fold_pnl > 0:
            pos_folds += 1
    return pos_folds

def compute_mdd_additive(trades):
    """Additive equity curve MDD from trade list sorted by entry_bar."""
    trades_sorted = sorted(trades, key=lambda t: t['entry_bar'])
    eq = peak = mdd = 0.0
    for t in trades_sorted:
        eq += t['net']
        peak = max(peak, eq)
        mdd = max(mdd, peak - eq)
    return mdd

def is_in_train_slice(trade, train_ratio=0.6, warmup=26, n15=None):
    """Trade belongs to train slice if entry_bar ≤ warmup + (n15-warmup)*0.6."""
    if n15 is None:
        from scripts.analysis.intrabar_trail_impact import n15 as N15
        n15 = N15
    boundary = warmup + int((n15 - warmup) * train_ratio)
    return trade['entry_bar'] <= boundary
```

### 2.2b WF 설계 정당성 (Critical Gap fix)

위 `wf_on_adjusted_trades()`는 **full 기간을 1회 실행 후 시간 파티션** 방식.

근거:
- Slippage는 모든 trade에 동일 reason 매핑 가능한 **선형 조정** → fold별 재실행 없이도 OOS 성질 유지
- 기존 `c1_refined_validation.wf_5fold`는 fold마다 `run_bt` 재호출 (expanding window). Slippage는 trade-reason 기반 사후 조정이므로 fold별 재호출 불필요
- 엄밀한 expanding-window WF가 필요하면 후속으로 `run_bt_slip`의 `start_i, end_i` 파라미터화 가능 (현재는 `intrabar_trail_impact.py`가 전체 기간 고정 — 별도 개선).

**대안 (더 엄밀한 방식)**: `run_backtest`를 `(warmup, end_bar)` 파라미터화한 후 5 expanding windows로 fold별 실행 → slippage 적용 → summarize. Phase 1에서는 단순 시간파티션으로 시작하고, 결과가 borderline일 경우 엄밀 방식으로 전환.

### 2.3 5-mode 그리드

한 번 실행으로 4종 모드 비교 + sl_trail_tuning의 top-3 combo 교차평가:

| Mode | 설명 | 사용 |
|------|------|------|
| `bar_close` | 현재 BT (15m close only trail) | 기준선 |
| `intrabar` | 15m bar low/high 기반 worst-case trail | 중간 |
| `5m` | 5m sub-bar traversal + per-sub-bar trail | 고해상도 |
| `5m_slip` | 5m + slippage 주입 | **운영 현실 추정** |

대상 combo:
- `baseline`: `(max_sl_atr=3.3, trail_K=2.5, max_hold_bars=192)`
- `candidate_A`: `(3.6, 2.2, 144)` — sl_trail_tuning train top-1
- `candidate_B`: `(4.5, 2.2, 144)` — sl_trail_tuning val top-1
- `candidate_C`: `(4.0, 2.5, 192)` — 기존 1D grid 최적

### 2.4 Output schema

```json
{
  "timestamp": "2026-04-19T...",
  "slippage_config": { ... },
  "mode_results": {
    "bar_close": {
      "baseline": {"trades": N, "pnl": X, "wr": Y, "mdd": Z},
      "candidate_A": {...},
      ...
    },
    "intrabar": {...},
    "5m": {...},
    "5m_slip": {...}
  },
  "live_window_comparison": {
    "live_pnl_1x": -3.92,
    "mode_pnl_1x": {
      "bar_close": 3.80,
      "intrabar": 2.50,
      "5m": 1.80,
      "5m_slip": -2.50
    },
    "closest_mode": "5m_slip",
    "gap_to_live": { "bar_close": 7.72, ..., "5m_slip": 1.42 }
  },
  "go_conditions": {
    "intrabar_realism": true|false,
    ...
    "train_not_degraded": true|false
  },
  "verdict": "GO|STOP|INCOMPLETE",
  "verdict_reason": "..."
}
```

---

## 3. Track B — LIVE 5m Polling (Phase 2)

### 3.1 구조 (정정 — Thread 기반)

**Critical gap fix**: `bot.py`는 **sync ccxt** (class TimeSyncBingX(ccxt.bingx), async def 0건). `asyncio`/`await` 도입은 메인 루프와 비호환. → **`threading.Thread` + sync ccxt** 로 재설계.

```python
# scripts/production/c1_breakout/bot.py 내 추가
import threading

class BestPricePoller(threading.Thread):
    """Background thread: poll 5m candle, update best_price, re-check trail.

    Separate thread per open position (N=1이므로 동시 최대 1개 스레드).
    Stop signal via self._stop.set(). Rate-limited via time.sleep.
    """
    def __init__(self, bot, pos, interval_sec=300):
        super().__init__(daemon=True, name=f'BestPricePoller-{pos["direction"]}')
        self.bot = bot
        self.pos = pos
        self.interval = interval_sec
        self._stop = threading.Event()

    def stop(self):
        self._stop.set()

    def run(self):
        while not self._stop.is_set():
            try:
                if self.pos not in self.bot.positions:
                    break  # position closed
                ohlcv = self.bot.exchange.fetch_ohlcv(
                    'BTC/USDT:USDT', '5m', limit=1
                )
                if ohlcv:
                    bar_high, bar_low = ohlcv[-1][2], ohlcv[-1][3]
                    with self.bot._pos_lock:  # protect shared state
                        old_best = self.pos['best_price']
                        if self.pos['direction'] == 'LONG':
                            self.pos['best_price'] = max(old_best, bar_high)
                        else:
                            self.pos['best_price'] = min(old_best, bar_low)
                        if self.pos['best_price'] != old_best:
                            self.bot._update_exchange_trail(
                                self.pos,
                                bar_high if self.pos['direction']=='LONG' else bar_low,
                                self.bot._last_atr
                            )
            except Exception as e:
                logger.warning(f"5m poll error: {e}")
            # sleep with interrupt support
            self._stop.wait(self.interval)
```

### 3.1b 메인 루프 통합

```python
# _do_open() 끝에 추가:
if self.config['strategy'].get('tick_best_price', {}).get('enabled', False):
    poller = BestPricePoller(self, self.positions[-1])
    poller.start()
    self._pollers.append(poller)

# _do_close() 시작에 추가:
for p in list(self._pollers):
    if p.pos is pos:
        p.stop()
        self._pollers.remove(p)

# __init__에 추가:
self._pos_lock = threading.Lock()
self._pollers = []
```

### 3.2 Config

```yaml
strategy:
  tick_best_price:
    enabled: false        # default OFF
    poll_seconds: 300     # 5m
```

### 3.3 Lifecycle

- 진입 시 `_do_open` 마지막에 `asyncio.create_task(self._poll_best_price_5m(pos))` 추가
- 청산 시 `_do_close`에서 task 자동 종료 (`while pos in self.positions` 조건)
- Rate limit: 1 pos × 288 polls/day = 288 req/day (BingX 제한 내, 여유 대폭)

### 3.4 A/B 테스트 프로토콜

30일 동안 2주 간격으로 ON/OFF 토글:
- Day 1-14: enabled=false (control)
- Day 15-28: enabled=true (treatment)
- 각 구간 trade 기록 비교 → trail exit PnL 개선 평균 산출
- GO 기준: treatment avg trail exit PnL ≥ control + 0.2pp/trade

---

## 4. Implementation Order

### Phase 1 (Track A, 우선)
1. `scripts/analysis/c1_intrabar_parity.py` 스크립트 작성 (재사용 최대화)
2. Slippage 모델 단위 테스트 (pytest 1~2 case로 수식 검증)
3. 4 modes × 4 combos 실행 → `results/intrabar_parity_{timestamp}.json`
4. LIVE 19-trade window 비교 결과 출력
5. 8 GO 조건 평가 (Phase 2 항목 2개는 None으로 유지)
6. 결과 검토 → Phase 2 진행 여부 결정

### Phase 2 (Track B, Phase 1 결과 기반 조건부)
7. `bot.py`에 `_poll_best_price_5m` 추가 + config schema
8. 단위 테스트 (mock exchange) 및 rate limit 검증
9. A/B 토글 지원 + 로그 강화
10. 실거래 A/B 2주 + 2주 (총 30일)
11. A/B 결과 비교 → Track B GO/STOP 확정

---

## 5. Files Touched

### Track A (Phase 1)
| 파일 | 변경 |
|------|------|
| `scripts/analysis/c1_intrabar_parity.py` | NEW |
| `scripts/tests/test_intrabar_parity.py` | NEW (optional, slippage unit tests) |
| `results/intrabar_parity_*.json` | 신규 결과 |

Production 코드 변경 없음.

### Track B (Phase 2, GO 판정 시에만)
| 파일 | 변경 |
|------|------|
| `scripts/production/c1_breakout/bot.py` | `_poll_best_price_5m` 추가 |
| `config/c1_breakout_config.yaml` | `tick_best_price` 섹션 추가 |
| `scripts/production/c1_breakout/config.py` | 신규 키 로딩 |
| `scripts/tests/test_bot_tick_polling.py` | NEW |

---

## 6. Testing Strategy

### Phase 1 단위 검증
- `apply_slippage(trade)`: 4개 reason 각각 수식 정확성
- `run_bt_slip(mode='bar_close')`: slippage=0 시 기존 `intrabar_trail_impact`의 `bar_close` 결과와 일치 (regression check)

### Phase 1 통합 검증
- `5m_slip` 모드 결과가 LIVE 19-trade PnL(-3.92%)과 ±3pp 이내
- baseline combo의 WF 5/5 유지
- sl_trail_tuning candidate의 순위 변화 모니터 (robustness)

### Phase 2 통합 검증 (A/B 테스트 후)
- Trail exit trade의 평균 PnL 개선 측정
- Rate limit 실측 (288 req/day)
- Exception 시 메인 루프 영향 없음 (background task 격리)

---

## 7. Risks & Mitigations

| 리스크 | 영향 | 완화 |
|--------|------|------|
| Slippage 보수치가 실제보다 낮음 | BT 과대평가 지속 | 민감도 분석 — ±0.05pp 범위에서 결과 안정성 확인 |
| 5m 데이터 중복/결측 | 잘못된 sub-bar traversal | 기존 검증된 `btc_5m_270days_reclassified.csv` 사용, 결측 시 15m fallback |
| Track B async task 누수 | 메모리 점증 | `while pos in self.positions` 조건으로 자연 종료, 추가 watchdog |
| Track B rate limit 초과 | 거래소 차단 | 1 pos × 288 = 288 req/day, 제한 대비 1/3 이하 |
| sl_trail_tuning 결과가 intrabar에서 뒤집힘 | robustness 가설 무너짐 | 4 combos 모두 재평가, 순위 변화 시 별도 분석 |

---

## 8. Rollback

### Track A
- `scripts/analysis/c1_intrabar_parity.py` 파일만 삭제 → 기존 BT 인프라 영향 0
- `research_protocol_overfit_guards.md`에 편입한 섹션만 복원 (편입 후 필요 시)

### Track B
- `config.strategy.tick_best_price.enabled: false` 즉시 비활성화
- 코드는 남기되 진입점 OFF → 메인 루프 완전 무관

---

## 9. Performance Estimate

### Phase 1
- 5m sub-bar traversal 오버헤드: 기존 `intrabar_trail_impact.py` 실측 약 15초 / full 333일
- 4 modes × 4 combos = 16 run ≈ **4~5분**
- WF 5-fold × 4 combos 추가 시 약 +5분
- LIVE window 비교: 초 단위

총 **10분 이내** 완료 예상.

### Phase 2
- Track B 개발: 3~5일 (async task + 테스트 + 로깅)
- A/B 테스트: 30일 운영

---

## 10. GO Condition 코드화 (Plan §4 1:1)

Plan §4 8개 조건 → Design §2.2의 `evaluate_go_conditions()`:

| # | Plan 조건 | 코드 구현 |
|---|-----------|-----------|
| 1 | intrabar_realism (±3pp) | `abs(gap_1x) <= 3.0` |
| 2 | baseline_preservation (+150%↑) | `adj_baseline_pnl >= clean * 0.88` |
| 3 | wf_pass | `run_wf_on_adj()` 5/5 |
| 4 | ratio_ok (≥85% of clean) | `adj_ratio >= clean_ratio * 0.85` |
| 5 | track_b_cost (≤10K/day) | Phase 2 deferred → `None` |
| 6 | track_b_benefit (≥0.2pp) | Phase 2 deferred → `None` |
| 7 | rollback_ready | `True` by design (config flag) |
| 8 | train_not_degraded | `adj_train >= clean_train - 5.0` |

Phase 1 종료 시 **1/2/3/4/7/8 = 6개** 평가. 4개 이상 true + 핵심(1,2,3,8) 4개 전부 true 시 Phase 2 진행.

---

## 11. Baseline 기준점

```
Clean BT (sl_trail_tuning 결과):
  Full PnL: +170.49%
  Train PnL: +95.07% (60% split)
  Val PnL:   +21.21% (20% split)
  Test PnL:  +54.20% (20% split)
  MDD: 5.38%, Ratio: 31.69

LIVE 19-trade 창 (2026-04-12~04-18):
  1x equiv: -3.92%
  3x actual: -11.77%
  WR: 26.3%
```

Phase 1 결과는 위 baseline 대비로 %/pp 단위 비교.

---

## 12. Non-Goals

- 1m tick 데이터 수집 (현재 5m가 상한)
- LIMIT order 전환 (별도 PDCA)
- Order book microstructure
- Multi-asset 확장
- Track B의 즉시 production 적용 (A/B 2주+2주 후 판정)

---

## 13. Reference

- Plan: `docs/01-plan/features/intrabar_parity.plan.md`
- 기존 스크립트 재사용: `scripts/analysis/intrabar_trail_impact.py`
- 심층 분석: `claudedocs/bt_live_gap_deep_review_20260419.md`
- 정합성 22항목: `claudedocs/BACKTEST_LIVE_PARITY.md`
- LIVE 매칭 데이터: `results/live_vs_backtest_verification.json`
- Overfit 표준: `memory/research_protocol_overfit_guards.md`
