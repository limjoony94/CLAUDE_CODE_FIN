# Pattern 5m Bot — v1.28.42 Current State (2026-02-21)

## Strategy
- BTC 5m 3-candle pattern trading, 3x leverage, BingX exchange
- 59 patterns (12L + 47S), MAE/MFE discovery method
- ATR-scaled TP/SL: ATR(14)/median(ATR,576), clamp [0.6, 1.7]
- SL cap: proportional vol_mult cap (max_daily_loss/leverage = 4.33%)
- TP: 1.0-3.3%, SL: 1.7-4.2%
- Quality: Edge>=21.8pp, WR>=60%, SL>=1.0%, MC<0.01, min_trades>=25
- WF 3/3 PASS (720d expanding window, OOS +320.5%, avg OOS WR 73.9%)
- 90-day OOS live test in progress (target 2026-04-30)

## Key Paths
- Entry: `scripts/production/pattern_5m_bot.py`
- Modules: `scripts/production/pattern_5m/` (14 files)
- Config: `config/pattern_5m_config.yaml`
- Constants: `scripts/production/pattern_5m/constants.py`
- Scanner: `scripts/scanner/pattern_scanner.py`
- State: `results/pattern_5m_bot_state.json`
- Metrics: `results/pattern_5m_metrics.json`
- Patterns: `results/dynamic_patterns.json`
- Data: `data/btc_5m_270days_reclassified.csv`

## Module Responsibilities
- `bot.py`: Main loop (5m cycle)
- `config.py`: Config loading + dynamic pattern injection
- `constants.py`: Pattern definitions + per-pattern TP/SL
- `exchange.py`: BingX API wrapper (retry, circuit breaker)
- `indicators.py`: Candle classification (12-type) + ATR + vol_mult
- `models.py`: Dataclasses (TradeRecord, Metrics)
- `orders.py`: TP/SL placement + verify + cancel
- `position.py`: Facade
- `position_open.py`: Entry logic + ATR-scaled TP/SL
- `position_monitor.py`: Position sync + direction check
- `position_close.py`: Exit + PnL calc + recovery
- `signals.py`: Pattern detection + confidence scoring
- `state.py`: Atomic state save/load + corruption recovery
- `utils/`: lock.py (file lock), logging_config.py

## Config Structure (pattern_5m_config.yaml)
- `pattern_source: dynamic` — reads from dynamic_patterns.json
- `tp_sl_mode: per_pattern` — individual TP/SL per pattern
- `strategy.atr_scale`: enabled, window=576, clamp_lo=0.6, clamp_hi=1.7
- `risk.max_daily_loss_pct: 13`
- `risk.consecutive_loss_pause: 3`
