# R26 LIVE Deployment Plan — Phased Approach with BT-LIVE Parity Validation

**Date**: 2026-04-30
**Trigger**: User decision to LIVE deploy R26 1× $1500
**LIVE-parity prior**: 0/1 (C1 broke catastrophically)

---

## Critical Precedent — C1 LIVE Failure (2026-04-27)

C1 Breakout v2.6 was deployed LIVE → -12.86%/14d at n=46. Despite:
- 6/7 PASS BT criteria (better than R26)
- WF 5/5 PASS, 3-Way ALL PASS, MC p=0.000
- 20/22 BT-LIVE parity checks at code level

**LIVE catastrophically failed** because BT model didn't represent LIVE market accurately.
Postmortem (`docs/04-report/c1_breakout_postmortem_20260427.md`) confirms the issue
was foundational, not a parameter problem.

**This precedent demands we CANNOT trust BT result blindly. R26 needs LIVE validation
under reduced size before scaling.**

---

## Deployment Phases

### Phase A — Paper Trading Simulation (1 week)

**Goal**: Verify bot infrastructure correctness without real money.

**Setup**:
- Paper trade on real BingX BTC perp price feed
- Simulate fills as if maker limits at price levels filled when price touches
- Use BingX testnet OR fully internal simulator with real prices
- Capital: $1500 simulated

**Success criteria**:
- Bot stays online 100% uptime over 7 days
- Detects ranging regime correctly
- Places grid orders correctly
- Records TP fills correctly
- Logs match BT expectation (~2 trades/day, ~50% time in regime)

**Failure criteria**:
- Bot crashes / API errors > 5%
- Order placement bugs
- State management errors
- LOG anomalies

**If Phase A fails**: fix bugs, restart Phase A.

---

### Phase B — LIVE Small Size ($150-$300)

**Goal**: Real money at small size to test BT-LIVE parity without risk of major loss.

**Capital**: $150 initially (10% of target)
**Per-level size**: $15 (instead of $150)
**Other params**: identical to baseline R26

**Duration**: 14 days minimum

**Success criteria** (ALL must hold):
- BT-LIVE daily PnL parity: |LIVE − BT| < 0.05%/day mean over 14d
- No catastrophic loss (worst 5d ≥ -1%)
- Trade count consistent with BT (~30 trades / 14d = 2.1/day)
- Grid behavior visually matches BT (TP cycles, occasional trend exits)

**Failure criteria** (any of):
- LIVE daily mean < -0.10% (significantly negative)
- LIVE - BT divergence > 0.10%/day
- > 2 forced trend exits in 14d (BT shows 27/720d = 0.04/day, would expect <1 in 14d)
- Bot operational issues (downtime > 2 hours total)

**If Phase B fails**: do post-mortem. Likely C1-style structural BT-LIVE gap. SHELVE.

---

### Phase C — BT-LIVE Parity Inspection (manual gates)

**At +24h after Phase B start**:
- Compare LIVE trades vs BT-expected
- Check ranging filter activation
- Verify grid placement
- Decision: GREEN (continue) / YELLOW (continue + log issue) / RED (stop, investigate)

**At +7 days**:
- Statistical comparison: LIVE daily distribution vs BT 14-day windows from history
- LIVE percentile in BT distribution should be > 5%, ideally > 30%
- Per advisor pattern from C1: any LIVE percentile < 5% triggers immediate halt

**At +14 days**:
- Final BT-LIVE parity verdict
- Decision: scale to Phase D / extend Phase B / shelve

---

### Phase D — Scale to $1500 (1× full)

**Trigger**: Phase B-C all GREEN, BT-LIVE parity confirmed.

**Capital**: $1500 (10× scale-up from Phase B)
**Per-level**: $150
**Monitoring**: weekly health check, monthly full review

**Ongoing criteria**:
- Daily PnL within BT distribution (5th percentile minimum)
- Drift drawdown events: ≤ 1/month (BT baseline)
- Grid uptime > 99%

**Halt triggers**:
- Worst 5d > -3%
- LIVE 30d cum < BT 5th percentile of 30d windows
- Operational issues

---

## Implementation

### File Structure (mirrors C1)

```
scripts/production/r26_grid_bot.py          # entry point
scripts/production/r26_grid/                 # module
    bot.py                                    # main loop
    grid.py                                   # grid placement / fill monitoring
    regime.py                                 # ATR ranging filter
    config.py                                 # config loader
config/r26_grid_config.yaml                  # LOCKED params
results/r26_grid_state.json                  # state
logs/r26_grid.log                            # log
```

### Key Components

**bot.py**:
- Main loop @ 1 minute interval (less aggressive than C1's 1m for grid stability)
- Init exchange connection (CCXT BingX, one-way mode, 1× leverage)
- Load state from json
- Each cycle:
  1. Fetch current candle (1h close)
  2. Compute ATR ranging filter
  3. If no active grid AND ranging:  - Setup grid (5 buy + 5 sell limit orders)
     - Save state
  4. If active grid:
     - Check fills (compare existing orders to current order book status via API)
     - On fill: cancel that level's order, place TP limit on opposite side
     - On TP fill: cancel grid level, replace at original level
     - Trend exit check: if |price - init_mid| > 1.5% AND ranging filter off:
       - Cancel all open orders
       - Market close all open positions (taker)
       - Reset grid

**grid.py**:
- `setup_grid(mid, levels, spacing)` → place limit orders
- `monitor_fills(grid_state)` → API poll, detect fills, update state
- `place_tp(filled_position)` → place TP limit
- `force_close_all()` → cancel orders + market close positions

**regime.py**:
- `compute_atr(periods)` from recent klines
- `is_ranging(atr_pct, lookback_atr_pct_history)` → 30d trailing median check
- `compute_trend_exit_signal(price, init_mid, ranging)` → boolean

**config.py**:
- Load `config/r26_grid_config.yaml`
- Validate parameters

### Configuration File

```yaml
# config/r26_grid_config.yaml — R26 LOCKED params

asset: BTC/USDT
timeframe: 1h
exchange: bingx
api_keys_path: config/api_keys.yaml
position_mode: one-way  # use BOTH positionSide
leverage: 1

# Grid parameters (LOCKED, do not modify without re-validation)
grid_spacing_pct: 0.30
grid_levels_each_side: 5
trend_exit_distance_pct: 1.5
max_grid_lifetime_bars: 168

# Capital management
phase_b_capital_usd: 150       # Phase B small size
phase_b_per_level_usd: 15
phase_d_capital_usd: 1500      # Phase D full size
phase_d_per_level_usd: 150
current_phase: A                # A / B / C / D

# Regime filter
atr_period: 20
atr_pct_median_lookback_bars: 720  # 30d

# Logging
log_path: logs/r26_grid.log
state_path: results/r26_grid_state.json
log_rotate_daily: true
log_retention_days: 30
```

---

## Risk Management (LOCKED)

### Hard halts (auto-trigger sys.exit)
- Daily NAV < -2% on capital → halt
- Cumulative NAV < -10% over 30d → halt
- API consecutive errors > 10 → halt
- Position mode auto-switched by exchange → halt and alert (BUG#66 precedent)

### Soft monitors (log + alert)
- Forced trend exit > 2 in 7d → log warning
- Spread > 0.10% on grid placement → wait for narrower spread
- Order rejection > 5% → log + investigate

---

## Comparison to C1

| Aspect | C1 (SHELVED) | R26 (DEPLOY) |
|--------|--------------|--------------|
| Strategy class | Directional breakout | Volatility harvest grid |
| Maker % of executions | ~0% | 98.2% |
| Slippage exposure | Every trade | 1/month forced exits only |
| BT positives | 6/7 PASS | 3/4 user criteria PASS |
| LIVE-parity prior | 0/1 (FAIL -12.86%) | 0/1 (untested) |
| Behavior pattern | With-trend (chase) | Anti-FOMO + anti-panic (88%) |
| BT cum_net 720d | +169.5% (additive 1×) | +36.19% |
| BT daily | 0.51%/day | 0.05%/day |
| Drawdown | 5.4% | 0.7% (worst 5d) |

R26 has STRUCTURALLY better LIVE-parity prospect than C1:
- Maker-heavy execution avoids slippage cascade
- Behavior is contrarian to retail emotion (less likely to encounter adversarial flow)
- Smaller daily expectation = less pressure for breakthrough
- Slippage tested robust at 0-0.20% range

---

## Timeline

| Phase | Duration | Total elapsed |
|-------|----------|---------------|
| Implementation | 2-3 days | 0-3d |
| Phase A (paper) | 7 days | 3-10d |
| Phase B (LIVE $150) | 14 days | 10-24d |
| Phase C (parity check) | parallel to B | 24d |
| Phase D (scale $1500) | ongoing | 24d+ |

Earliest full deploy: ~3.5 weeks from now.

---

## User Confirmation Required

This plan requires user explicit approval on:

1. **Phased approach acceptable?** Or skip phases (more aggressive)?
2. **Phase B size**: $150 (suggested) / different amount?
3. **Phase B duration**: 14 days / shorter / longer?
4. **Halt triggers acceptable?** Or different thresholds?

Per advisor / BUG#66-style memos: hard halts protect against unanticipated LIVE
behavior. Reducing/removing halts increases risk of large loss.

---

## Implementation Start Trigger

User says "Phase A 진행" → start coding bot infrastructure.

If user wants to skip Phase A and go straight to LIVE → I will state explicitly that
this carries higher operational risk (untested code in LIVE) and request explicit
acceptance.
