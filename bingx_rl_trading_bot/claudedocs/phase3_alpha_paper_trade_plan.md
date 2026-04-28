# Phase 3 — α Paper Trade PDCA Plan

> **Date**: 2026-04-28
> **Status**: PLAN (pre-registered before any implementation)
> **Authority**: 사용자 명시 — "paper trade Phase 3 PDCA Plan" (2회 emphasis)
> **Origin**: R19 cluster 발견. R9b/R15 동일 statistical signature 2회 fail. Forward paper = 유일한 untouched OOS.

---

## 1. Background

### 19-round arc summary
- 17 mechanisms × 12 rounds + R14 6-family sweep + R15 timeframe + R17 8-family + R18 α WF + R19 native regime
- 2 surface-positive findings (R9b α N=4, R15 4h) failed strict OOS (R9c, R16)
- R19: 3rd surface-positive finding with same statistical signature
- 720일 dataset 모든 부분 test에 사용됨 → fresh OOS = forward time only

### R19 finding (validate target)
**Convergent cluster** (Approach A 3-way + Approach B WF 4-fold both):
- eth_thresh = **0.40**
- btc_lag_thresh = **0.10**
- atr_pctile = **60** (medium volatility regime)
- N_exit = **6** (1.5 hours hold @ 15m bars)
- Friction assumption: 0.04% maker-tier per leg

**R19 metrics**:
- Approach A: train +0.0080, val +0.0780, test holdout +0.0485 (% daily)
- Approach B WF mean: +0.0354%/day (4/4 folds positive)
- Cluster median expected: ~+0.03~0.04%/day forward

### 50% fail prior 명시
**R9b α N=4 (R9c WF FAIL)**:
- Surface positive train +0.0195/day @ 0.04
- WF 5-fold: 2/5 positive, bootstrap pos_rate 9% → drop

**R15 4h timeframe (R16 WF FAIL)**:
- Surface 60/40 split: 80% configs cross-stable
- WF 5-fold: 0/144 robust (vs ~4.5 expected) → drop

**R19 (current candidate, untested in fresh OOS)**:
- Same shape as both. Honest prior: ~50% probability of fail in forward.
- 진행 가치: 통과 시 진짜 deployable 발견. Fail 시 paradigm 자체 confirmed.

---

## 2. Plan — Hypothesis & Criteria

### Primary Hypothesis (사전 등록)
α 전략 (et=0.40, bl=0.10, ap=60, N=6, fixed timeout exit only)이:
- Forward 60-90일 동안 net daily PnL > 0 @ assumed friction 0.04%
- Random entry control 대비 Mann-Whitney p < 0.10 우월
- 위 두 조건 ALL pass → real edge, deployable

### Pre-registered Success Criteria (90-day forward)

| 지표 | 합격 기준 |
|------|----------|
| **Cumulative net daily PnL** | > 0 @ day 90 |
| **vs Random control** | Mann-Whitney U test p < 0.10 (one-sided) |
| **Trade frequency** | ≥ 0.3 trades/day (일치 R19 evidence) |
| **WR** | ≥ 40% |
| **Max drawdown** | < 3% (cumulative -3% halt) |
| **No fold collapse** | 30-day rolling daily PnL min > -0.10% |

### Pre-registered Halt Criteria (early termination)

| Trigger | 조치 |
|---------|------|
| Cumulative PnL < -1% at day 30 | **HALT** (50% fail prior 정합) |
| Cumulative PnL < -2% at day 60 | HALT |
| Drawdown -3% any time | HALT (catastrophic) |
| 0 trades in any 30-day rolling window | PAUSE + review (regime void) |
| 5 consecutive losing trades | Inspect (not auto-halt) |

### Pre-registered Decision Gates

**Day 30 review**:
- PnL > 0 AND ≥ 8 trades → continue to day 60
- PnL ∈ [-1%, 0%] AND ≥ 8 trades → continue with monitoring
- PnL < -1% OR < 4 trades → halt + report

**Day 60 review**:
- PnL > +0.5% (cumulative) → continue to day 90, prepare ramp-up
- PnL ∈ [0%, +0.5%] → continue, no ramp-up
- PnL < 0% → halt + report (R9b/R15 pattern confirmed)

**Day 90 final**:
- ALL success criteria met → Phase 3b (BingX 0.1× live ramp-up)
- ANY criterion fail → drop claim, accept 19-round cumulative finding

---

## 3. Architecture & Design

### Mode: Paper Journal (no real orders)
**Rationale**: Cleanest first step. Live API testnet adds auth/order-mgmt complexity. Pure observation script:
1. Poll BingX 15m candle data live (every 15min cycle)
2. Compute α signals using R19 optimal config
3. Log "would-be" trades + simulated fills + PnL
4. Compare against simultaneous random-entry control
5. Daily aggregation + decision-gate metrics

### Stack & Files
```
bingx_rl_trading_bot/
├── scripts/paper_trade/
│   ├── alpha_paper_trader.py       # Main script (15m cycle loop)
│   ├── alpha_signal.py              # R19 optimal config α detector
│   ├── random_control.py            # Random entry baseline (paired)
│   ├── paper_journal.py             # JSON log writer
│   └── daily_report.py              # Stats + criterion check
├── config/
│   └── alpha_paper_config.yaml      # Optimal params, halt criteria
├── results/
│   ├── paper_journal_alpha.jsonl    # Append-only trade log
│   ├── paper_journal_random.jsonl
│   └── paper_daily_<date>.json      # Daily aggregated metrics
├── logs/
│   └── alpha_paper.log
└── claudedocs/
    └── phase3_alpha_paper_trade_plan.md  # this doc
```

### Signal config (R19 optimal — locked)
```yaml
strategy:
  name: alpha_native_regime
  entry:
    eth_thresh: 0.40
    btc_lag_thresh: 0.10
    atr_pctile: 60
    require_trend_alignment: true   # 1h+4h
  exit:
    timeout_bars: 6                  # 1.5h on 15m
    use_sl: false
    use_trail: false
    emergency_pct: 1.5
    min_bars_between: 2
  friction:
    assumption: 0.04                 # maker-tier RT per leg, 0.08% total

random_control:
  mode: paired_random_entry
  target_n_per_day: same_as_alpha   # match α's daily count
  trend_aligned: true
  seed: 42
```

### Live data feed
```
BingX REST API (or websocket): 15m BTC perp + 5m ETH spot/perp + 1h+4h aggregation
Polling every 5 minutes (15m bar close + 5min buffer)
Source: same as bot infra used for C1 (already proven)
```

### Daily report metrics
- Trade count (α, random)
- Cumulative net PnL
- Daily PnL distribution (mean, std, max, min)
- 30/60/90-day rolling
- vs random control (Mann-Whitney U p-value)
- Halt criteria check

---

## 4. Do — Phased Deployment

### Phase 3a — Paper Journal (Day 0~90)
- **Mode**: 가상 거래만 (no real money)
- **Target**: 60-90일 forward observation
- **Cost**: 0 capital
- **Decision gate**: Day 90 success criteria

### Phase 3b — BingX Live 0.1× (Day 91~120, if 3a PASS)
- **Mode**: Real orders, **0.1× position size** (= ~$10/trade at $1000 capital)
- **Target**: 30일 confirmation, real-execution slippage measure
- **Cost**: ~$15 expected friction (worst case)
- **Decision gate**: 0.1× cumulative > 0 AND slippage < 2× assumed friction

### Phase 3c — BingX Live 0.5× (Day 121~150, if 3b PASS)
- **Mode**: Real orders, 0.5× position
- **Target**: Scale verification
- **Decision gate**: 0.5× cumulative > 0 AND no infrastructure issues

### Phase 3d — Full position (Day 151+, if 3c PASS)
- **Mode**: Full size based on user's capital decision
- **Ongoing**: Quarterly review for regime changes

### Estimated timeline
- Plan commit: today (2026-04-28)
- Phase 3a start: 2026-04-29 (after script implementation, 1-2 days)
- Phase 3a end: ~2026-07-28 (90 days forward)
- Phase 3b start: ~2026-07-29 (if 3a pass)
- Phase 3d earliest: ~2026-09-30 (if all phases pass)

---

## 5. Check — Monitoring & Comparison

### Daily artifacts
- `results/paper_daily_YYYY-MM-DD.json`:
  - α: trades_today, daily_pnl, cumulative_pnl, n_open_positions
  - random: same fields
  - statistical: rolling_mean_30d, mann_whitney_p, drawdown
  - halt_status: bool flags (criterion violations)

### Weekly summary
- Auto-generated markdown report
- Push to git (transparent log)

### Statistical comparison
- α PnL vs random control: Mann-Whitney U test (one-sided, paired by day)
- Bootstrap 90-day window pos_rate
- Sharpe ratio (annualized)

### Halt automation
- Script checks halt criteria daily
- If triggered: 즉시 alert (write halt_triggered.flag) + paper trades 중지
- 사용자 review 필요

---

## 6. Risks & Mitigation

### Risk 1: BT-LIVE parity gap (C1에서 이미 -25pp 갭 경험)
- **Mitigation**: Paper mode는 BT와 본질적으로 동일 (시뮬레이션). 단 Phase 3b live에서 측정.
- **Pre-registered slippage budget**: 실제 slippage > 2× assumed (0.08%) → 즉시 halt

### Risk 2: Regime change (α는 native regime conditional)
- **Mitigation**: 30-day rolling 0 trades = pause + review
- **Detection**: ATR percentile + h1/h4 trend filter는 자동 regime detector 역할

### Risk 3: Selection bias (R19 native regime post-hoc)
- **Mitigation**: Forward time = pristine OOS by definition
- **Acceptance**: 50% fail prior 명시. Halt criteria가 fail case 빠르게 감지

### Risk 4: Same-pattern failure (R9b/R15 → R19 third instance)
- **Mitigation**: 30/60/90-day decision gates. Day 30 PnL < -1% → halt.
- **Honesty**: User에게 "통과 확률 ~50% best case, 더 낮을 수도" 명시.

### Risk 5: 작은 sample 분석 결정 영향
- **Mitigation**: Pre-registered statistical tests (Mann-Whitney) 사용. Cherry-picking 방지.

### Risk 6: Maker rebate assumption (0.04% RT)
- **Mitigation**: Paper journal는 0.04% 가정. Live 단계에서 실제 maker fill rate 측정. 50% maker fill = ~0.07% RT 가정으로 재평가 필요.
- **Note**: BingX taker 0.05%, maker 0.02%. Pure maker = 0.04% RT. 50:50 = 0.07%.

---

## 7. Act — Decision Tree

### Day 30 outcomes
| 조건 | Action |
|------|--------|
| PnL > 0, ≥ 8 trades | Continue to Day 60 |
| PnL ∈ [-1%, 0], ≥ 8 trades | Continue with note |
| PnL < -1% OR < 4 trades | **HALT, report** |

### Day 60 outcomes
| 조건 | Action |
|------|--------|
| PnL > +0.5% cumulative | Continue, prepare 3b ramp-up |
| PnL ∈ [0, +0.5%] | Continue 3a only |
| PnL < 0% | **HALT, report (R9b/R15 pattern confirmed)** |

### Day 90 final
| 조건 | Action |
|------|--------|
| Success ALL criteria | **Phase 3b GO** (BingX 0.1× live) |
| Any criterion fail | **DROP claim**, finalize 19-round arc |

### Phase 3b/c/d decision gates same structure
- 각 단계 마지막 날 cumulative > threshold AND no infrastructure issue → next phase
- 어느 phase fail → halt + report

---

## 8. Resources

### Compute & Infra
- Existing C1 bot infra (BingX API auth, polling, logging) → 재활용
- 신규 코드: paper journal script (~300 lines), random control, daily reporter
- 시간: 1-2일 implementation

### Capital
- Phase 3a: $0 (paper)
- Phase 3b: ~$10/trade × ~30 days × ~0.3 trades/day ≈ $90 cycling. Net friction cost ~$15-30 expected.
- Phase 3c: $50/trade × scale-up
- Phase 3d: 사용자 결정 영역

### 사용자 시간 투입
- 일일 daily report 확인 (~5분)
- 30/60/90-day decision review (~30분 each)
- Halt 발생 시 즉시 review

---

## 9. Stop Conditions (override)

본 Phase 3 plan은 다음 조건 중 하나 발생 시 즉시 종료:
1. Halt criteria 발동 (drawdown, day-30/60 PnL)
2. BingX API 또는 인프라 fundamentals 변경
3. 사용자 명시 stop
4. 다른 paradigm으로 redirect

---

## 10. Anti-fix-impulse Commitments

본 plan 시작 후:
- Day 30/60 fail → grid 재최적화 안 함 (memory file `lessons_fix_impulse_pattern_20260427` 명시 패턴)
- Halt 후 "다른 N으로 재시도" 안 함
- Forward time 결과는 본 cluster의 진짜 verdict
- Fail = R9b/R15와 같은 결론, accept

---

## 11. Pre-registered checksum

본 plan에 포함된 모든 number는 실행 전 확정:
- N=6, friction=0.04, atr_pctile=60, eth_thresh=0.40, btc_lag=0.10
- Halt: -1% day 30, -2% day 60, -3% any time
- Success: PnL > 0 day 90, MW p < 0.10
- Decision gates: 0/+0.5/+0% thresholds at 30/60/90

위 number들은 결과 무관 변경 안 함.

---

## 12. Implementation TODOs

- [ ] PDCA Plan commit (이 문서 — 우선)
- [ ] `scripts/paper_trade/alpha_paper_trader.py` 작성 (R19 entry + fixed N=6 exit + paper journal)
- [ ] `scripts/paper_trade/random_control.py` (paired random control)
- [ ] `config/alpha_paper_config.yaml` 정리
- [ ] `scripts/paper_trade/daily_report.py` (Mann-Whitney + halt check)
- [ ] BingX live data feed 검증 (5min polling, 15m candle 정합)
- [ ] Day 1 launch dry-run
- [ ] Day 30 first decision gate

본 TODOs는 사용자 GO 후 진행. 각 step별 unit tests (`scripts/tests/`) 추가 권장.
