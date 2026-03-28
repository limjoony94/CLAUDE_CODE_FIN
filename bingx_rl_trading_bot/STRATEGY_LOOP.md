# Strategy Improvement Loop — Ralph Loop Prompt
# ================================================
# Usage: /ralph-loop "$(cat STRATEGY_LOOP.md)" --max-iterations 10 --completion-promise "CYCLE COMPLETE"

## Your Role
You are a quantitative strategy researcher for a BTC 5m pattern trading bot.
Each iteration of this loop = one PDCA improvement cycle.

## System Context
- Bot: Pattern 5m v1.67.1, 111 patterns (51L+60S), BingX Hedge mode
- Core: TP/SL asymmetry + mechanism stack (cascade, timeout, momentum guard)
- Nature: Volatility harvester (not directional predictor)
- Scanner: `scripts/scanner/pattern_scanner.py` — `portfolio_npos()` with parity (timeout PnL + TP decay)
- Data: `data/btc_5m_270days_reclassified.csv`
- Patterns: `results/dynamic_patterns.json`

## Protocol: Read → Plan → Do → Check → Act

### Step 0: READ PREVIOUS STATE
```
Read results/strategy_loop_state.json (if exists)
Read results/strategy_loop_log.json (if exists)
Identify: current iteration number, previous findings, rejected hypotheses
```

### Step 1: PLAN — Generate Hypothesis
Based on previous iterations and current system knowledge:
1. Identify ONE specific improvement hypothesis
2. Must be testable with existing backtest infrastructure
3. Must NOT repeat previously rejected hypotheses
4. Examples:
   - "SL floor at 2.0% improves R:R without killing WR"
   - "Timeout at 192 bars reduces timeout losses"
   - "Direction cap 5 reduces cluster exposure"
   - "Pre-emptive cascade at 2% triggers more frequently"

Write hypothesis to `results/strategy_loop_state.json`:
```json
{
  "iteration": N,
  "hypothesis": "...",
  "parameter_changes": {"key": "old → new"},
  "expected_effect": "...",
  "status": "PLANNING"
}
```

### Step 2: DO — Implement & Backtest
Write a research script at `scripts/analysis/loop_iter_N.py` that:

1. **Baseline**: Run parity scanner (timeout PnL + TP decay 0.9975) with current params
2. **Treatment**: Run parity scanner with proposed change
3. **Compare on last 30 days** (IS quick check):
   - PnL, WR, R:R, MDD, PnL/MDD, WR margin
   - Exit reason breakdown (especially TIMEOUT count)

If IS check FAILS (treatment worse), skip to STOP.

4. **5-fold WF validation** (both baseline and treatment):
   - Expanding window, fold formula: `is_end = int(n * (fi + 1) / (n_folds + 1))`
   - All folds positive = PASS

### Step 3: CHECK — Validate (3 gates)

**Gate 1: WF PASS**
- All 5 folds positive for treatment
- Total OOS > baseline total OOS

**Gate 2: Discrimination Test**
- Shuffle signal directions 10 times
- Run WF on shuffled
- If >7/10 shuffled PASS → NON-DISCRIMINATING (mechanical)
- Mechanical improvements are OK for this system (volatility harvester)
- But improvement must survive shuffling (shuffled treatment > shuffled baseline)

**Gate 3: Cascade Independence**
- Run treatment with cascade OFF
- If treatment effect vanishes → CASCADE-DEPENDENT
- Record dependency ratio

### Step 4: ACT — Decision

**GO criteria** (ALL must pass):
- WF 5/5 PASS
- Treatment OOS > Baseline OOS by > 5%
- WR margin remains > +10pp
- R:R does not degrade > 20%

**STOP criteria** (ANY triggers stop):
- WF FAIL (any fold negative)
- Treatment OOS < Baseline OOS
- WR margin < +5pp
- Already tested in previous iteration (check log)

### Step 5: RECORD
Append to `results/strategy_loop_log.json`:
```json
{
  "iteration": N,
  "hypothesis": "...",
  "result": "GO" | "STOP",
  "baseline_oos": X,
  "treatment_oos": Y,
  "delta": Y-X,
  "discrimination": "DISC" | "NON-DISC",
  "cascade_dependency": ratio,
  "detail": {...}
}
```

Update `results/strategy_loop_state.json` with status.

### Step 6: NEXT or COMPLETE
- If GO: Apply change to config (note in state), increment iteration, continue
- If STOP: Increment iteration, generate new hypothesis, continue
- If 3 consecutive STOPs with no GO: Output <promise>CYCLE COMPLETE</promise>
- If improvement found and applied: Output <promise>CYCLE COMPLETE</promise>

## Constraints
- NEVER modify production bot code (`scripts/production/pattern_5m/`)
- ONLY modify `config/pattern_5m_config.yaml` for GO decisions
- Research scripts go in `scripts/analysis/loop_iter_N.py`
- Use parity scanner: `tp_decay_rate=0.9975` in `portfolio_npos()`
- Fee: 0.10%, Leverage: 3x, Compound sizing
- WF fold formula: `is_end = int(n * (fi + 1) / (n_folds + 1))` (NOT fi+2)
- Signal tuple format: `(bar, pattern_name, direction, tp_pct, sl_pct)`

## Hypothesis Priority Queue (suggested, not mandatory)
1. Timeout reduction (288 → 192 or 240)
2. Pre-emptive cascade threshold (3% → 2%)
3. Direction cap adjustment (6 → 5)
4. SL floor enforcement (min 2.0%)
5. Aggregate risk tightening (10/15 → 7/12)
6. Cascade tighten % (95 → 98)

## Anti-Overfitting Rules
- Each hypothesis changes EXACTLY ONE parameter
- No grid search over multiple values (pick ONE educated guess)
- If GO, validate on the 23-day pure OOS segment (n - 6549 to n)
- Maximum 10 iterations per cycle
- Previously rejected hypotheses cannot be retried with same value
