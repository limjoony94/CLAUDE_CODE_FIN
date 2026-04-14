Create a new research script following the Standard Research Protocol.

The script MUST follow these mandatory rules:

## Mandatory Imports & Constants
```python
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from scripts.production.c1_breakout.indicators import calculate_atr, calculate_channel, find_fractal_swings

FEE_PCT = 0.10  # 0.05% x 2 sides (RT)
```

## C1 Breakout Backtest Rules
- Timeframe: 15m (synthesized from 5m data)
- Entry: signal bar[i] close > channel high AND body > 40% range → next bar open[i+1]
- SL: Fractal swing point (lookback=10, max 3.3x ATR cap)
- TP: Trailing — best_price drawdown >= trail_K x ATR / close x 100
- Exit: Intrabar High/Low (distance-based same-bar resolution)
- Fee: 0.10% RT (taker 0.05% x 2)
- PnL: Additive (compound distortion prevention)
- Timeout: 48h (192 bars at 15m)
- min_bars_between: 2 bars after exit

## Validation Requirements
- Monte Carlo: sign randomization (>=999 sims), p < 0.01
- Walk-Forward: 5-fold expanding window, ie = int(n*(fi+1)/(n_folds+1))
- Look-ahead: Progressive test mandatory (truncated vs full comparison)
- 3-way split: train/val/test
- min_trades >= 25 for statistical validity

## Forbidden Patterns
- `df['col'].shift(-1)` — look-ahead bias
- `df.rolling(n, center=True)` — centered window bias
- Single seed MC tests — MUST use multiple seeds

## Output
- Save results to `bingx_rl_trading_bot/results/` as JSON
- Include metadata: date, script version, parameters used
- Print summary table to stdout

Ask the user: What is the research question or hypothesis to investigate?
