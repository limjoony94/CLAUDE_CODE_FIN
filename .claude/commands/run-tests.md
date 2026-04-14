Run project tests if available.

## Current Status
C1 Breakout v2 does NOT have a dedicated test suite yet.
The previous 1139+ tests were for Pattern 5m (now in archive/legacy_bots/).

## What You Can Verify
1. **Syntax check**: `cd bingx_rl_trading_bot && python -c "from scripts.production.c1_breakout import bot, signals, indicators, config; print('All modules import OK')"`
2. **Config validity**: `cd bingx_rl_trading_bot && python -c "from scripts.production.c1_breakout.config import load_config; c = load_config(); print(f'Config loaded: {len(c)} sections')"`
3. **Indicator sanity**: Run a quick backtest snippet to verify indicators produce expected output

## If Tests Are Created
Place in `bingx_rl_trading_bot/scripts/tests/` and run:
```bash
cd bingx_rl_trading_bot && python -m pytest scripts/tests/ -v --tb=short 2>&1 | tail -80
```

## Priority Test Areas for C1 Breakout
- `signals.py`: Channel breakout detection, body filter, direction logic
- `indicators.py`: ATR calculation, channel high/low, fractal swing finding
- `bot.py`: State management, position lifecycle, exchange order sync
- `config.py`: Config loading, default values, validation
