# Standard Research Protocol

## CRITICAL — Must Follow in Every Research Script

### Mandatory Imports
```python
from scripts.production.pattern_5m.indicators import classify_candle  # NEVER self-implement
LEVERAGE = 3  # MUST apply to PnL
FEE_PCT = 0.10  # 0.05% x 2 sides, apply as FEE_PCT * LEVERAGE in backtest
```

### Backtest Rules
- Entry: signal bar's NEXT bar open
- Exit: intrabar high/low distance from bar OPEN (not entry)
- Same-bar: `abs(tpp - opens[j])` NOT `abs(tpp - entry)`
- Sizing: compound (multiplicative)
- Timeout: DROP (exclude from PnL)
- MAX_BARS: 288 (24h)

### Validation
- Monte Carlo: sign randomization, 10k sims, 3 seeds (42, 123, 7), max p < 0.01
- Walk-Forward: EXPANDING WINDOW ONLY (IS=[0..T], OOS=[T..T+1])
  - NEVER cross-validation (produces false positives)
- min_trades >= 25

### Quality Thresholds
- Edge >= 21.8pp (WR - baseline_WR)
- WR >= 60%
- SL >= 1.0% (below = execution risk)
- MC p-value < 0.01 (3-seed conservative)

### Known Bugs to Avoid
- Self-implemented classify_candle() — misses HAMMER constraint
- Single-seed MC — use 3 seeds
- FEE_PCT without LEVERAGE — BingX charges on notional
- Cross-validation WF — future data leaks into IS
- `abs(tpp - entry)` for same-bar — always resolves to TP win
