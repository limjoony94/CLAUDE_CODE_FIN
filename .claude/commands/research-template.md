Create a new research script following the Standard Research Protocol.

The script MUST follow these mandatory rules:

## Mandatory Imports & Constants
```python
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from scripts.production.pattern_5m.indicators import classify_candle  # NEVER self-implement
from scripts.production.pattern_5m.constants import PATTERN_DIRECTIONS

LEVERAGE = 3  # MUST apply to PnL
FEE_PCT = 0.10  # 0.05% x 2 sides
```

## Backtest Rules
- Entry: signal bar's NEXT bar open (no look-ahead)
- Exit: intrabar high/low distance-based resolution
- Same-bar TP/SL: use `abs(tpp - opens[j])` (bar open), NOT `abs(tpp - entry)`
- Sizing: compound (multiplicative returns)
- Fees: `FEE_PCT * LEVERAGE` per side (BingX charges on notional)
- Timeout trades: DROP (do not include in PnL)
- MAX_BARS: 288 (24h)

## Validation Requirements
- Monte Carlo: sign randomization, 10k sims, 3 seeds (42, 123, 7), max p-value < 0.01
- Walk-Forward: Expanding window only (IS=[0..T], OOS=[T..T+1]), NEVER cross-validation
- min_trades >= 25 for statistical validity

## Forbidden Patterns
- `df['col'].shift(-1)` — look-ahead bias
- `df.rolling(n, center=True)` — centered window bias
- Self-implemented `classify_candle()` — MUST use production version
- Single seed MC tests — MUST use 3 seeds

## Output
- Save results to `bingx_rl_trading_bot/results/` as JSON
- Include metadata: date, script version, parameters used
- Print summary table to stdout

Ask the user: What is the research question or hypothesis to investigate?
