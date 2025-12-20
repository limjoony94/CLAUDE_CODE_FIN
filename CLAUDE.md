# CLAUDE_CODE_FIN - Workspace Overview

**Last Updated**: 2025-12-19 KST

---

## Quick Reference

| Item | Value |
|------|-------|
| **Active Bot** | RSI Trend Filter Bot v1.0 |
| **Bot File** | `bingx_rl_trading_bot/scripts/production/rsi_trend_filter_bot.py` |
| **Config** | `bingx_rl_trading_bot/config/rsi_trend_filter_config.yaml` |
| **State File** | `bingx_rl_trading_bot/results/rsi_trend_filter_bot_state.json` |
| **Logs** | `bingx_rl_trading_bot/logs/rsi_trend_filter_bot_YYYYMMDD.log` |

---

## Active Bot: RSI Trend Filter v1.0

### Strategy Parameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| **Entry (LONG)** | RSI crosses above 40 + Close > EMA100 | Uptrend + RSI bounce |
| **Entry (SHORT)** | RSI crosses below 60 + Close < EMA100 | Downtrend + RSI decline |
| RSI Period | 14 | Standard RSI |
| EMA Period | 100 | Trend filter |
| **Take Profit** | 3.0% | Fixed |
| **Stop Loss** | 2.0% | Fixed |
| Cooldown | 4 candles | 1 hour (15m × 4) |
| Leverage | 4x | |
| Timeframe | 15m | |
| Symbol | BTC-USDT | |

### Entry Logic

```
LONG Entry:
  - Price > EMA(100) → Uptrend confirmed
  - RSI(14) crosses above 40 → Bounce from oversold

SHORT Entry:
  - Price < EMA(100) → Downtrend confirmed
  - RSI(14) crosses below 60 → Decline from overbought
```

### Validation Results (Walk-Forward 7 Windows)

| Metric | Value |
|--------|-------|
| **Profitable Windows** | 6/7 (86%) |
| **Total PnL** | +120.8% |
| **Sharpe Ratio** | 1.31 |
| **P-value** | 0.013 (statistically significant) |
| **Monte Carlo** | 100% profit probability |
| **Worst Window** | -4.8% |
| **Validation Score** | 10/10 PASSED |

### Commands

```bash
# Start bot (Windows)
cd bingx_rl_trading_bot
START_RSI_TREND_FILTER.bat

# Start bot (Linux/Direct)
cd bingx_rl_trading_bot
python scripts/production/rsi_trend_filter_bot.py

# Monitor bot
python scripts/monitoring/rsi_trend_filter_monitor.py

# Check state
cat results/rsi_trend_filter_bot_state.json

# Check config
cat config/rsi_trend_filter_config.yaml

# View logs
tail -f logs/rsi_trend_filter_bot_$(date +%Y%m%d).log
```

### Parameter Modification Examples

```yaml
# In config/rsi_trend_filter_config.yaml:

# Change TP to 2.5%
strategy:
  tp_pct: 2.5

# Change RSI thresholds to 35/65
strategy:
  rsi_long_threshold: 35
  rsi_short_threshold: 65

# Change EMA period to 200
strategy:
  ema_period: 200

# Change leverage to 5x
leverage: 5
```

---

## Project Structure

```
CLAUDE_CODE_FIN/
├── CLAUDE.md                    ← This file
├── CLAUDE_ARCHIVE_OCT2025.md    ← Historical context
├── .gitignore
│
└── bingx_rl_trading_bot/        ← Main bot directory
    │
    ├── config/
    │   ├── rsi_trend_filter_config.yaml  ← ACTIVE
    │   ├── adx_supertrend_trail_config.yaml  ← DEPRECATED
    │   └── config.yaml                   ← Legacy
    │
    ├── scripts/
    │   ├── production/                   ← Production bots
    │   │   ├── rsi_trend_filter_bot.py   ← ACTIVE (v1.0)
    │   │   ├── adx_supertrend_trail_bot.py  ← DEPRECATED
    │   │   ├── download_historical_data.py
    │   │   ├── check_current_positions.py
    │   │   └── ... (legacy bots)
    │   │
    │   ├── monitoring/                   ← Monitor scripts
    │   │   ├── rsi_trend_filter_monitor.py  ← ACTIVE
    │   │   ├── adx_supertrend_trail_monitor.py
    │   │   └── quant_monitor.py
    │   │
    │   └── analysis/                     ← Research & backtesting
    │       ├── alternative_strategies_research.py
    │       ├── rsi_strategy_deep_research.py
    │       ├── best_strategy_validation.py
    │       └── ... (100+ analysis scripts)
    │
    ├── src/                              ← Core library code
    │   ├── api/
    │   │   └── bingx_client.py           ← BingX API wrapper
    │   ├── indicators/
    │   │   └── technical_indicators.py   ← TA calculations
    │   ├── models/
    │   │   └── xgboost_trader.py         ← ML models (legacy)
    │   ├── data/
    │   │   ├── data_collector.py
    │   │   └── data_processor.py
    │   ├── environment/
    │   │   └── trading_env_v6.py         ← RL environment
    │   ├── risk/
    │   │   └── risk_manager.py
    │   └── utils/
    │       ├── logger.py
    │       └── config_loader.py
    │
    ├── results/                          ← Bot state & results
    │   ├── rsi_trend_filter_bot_state.json  ← ACTIVE
    │   ├── backups/
    │   └── ... (optimization results)
    │
    ├── models/                           ← Trained ML models (legacy)
    │   ├── *.pkl                         ← Scalers
    │   └── *.keras                       ← LSTM models
    │
    ├── data/
    │   ├── cache/                        ← API cache
    │   ├── features/                     ← Feature data
    │   └── trained_models/               ← Model checkpoints
    │
    ├── logs/                             ← Bot logs
    │   └── rsi_trend_filter_bot_YYYYMMDD.log
    │
    ├── claudedocs/                       ← Research documentation
    │   ├── DYNAMIC_STOPLOSS_RESEARCH_20251213.md
    │   └── ... (200+ research docs)
    │
    ├── archive/                          ← Archived files
    │
    ├── .env.example                      ← Environment template
    ├── requirements.txt                  ← Python dependencies
    ├── START_RSI_TREND_FILTER.bat        ← ACTIVE
    ├── MONITOR_RSI_TREND_FILTER.bat      ← ACTIVE
    └── README.md
```

---

## Development Workflow

### Environment Setup

```bash
# 1. Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows

# 2. Install dependencies
cd bingx_rl_trading_bot
pip install -r requirements.txt

# 3. Configure API keys
cp .env.example .env
# Edit .env with your BingX API keys
```

### Key Dependencies

- **ccxt**: Exchange API (BingX)
- **pandas/numpy**: Data manipulation
- **ta/pandas-ta**: Technical indicators
- **stable-baselines3**: RL (legacy)
- **torch**: Deep learning (legacy)
- **pyyaml**: Configuration

### Git Workflow

```bash
# View status
git status

# Commit changes
git add .
git commit -m "feat: Description of change"

# Push (use branch name from context)
git push -u origin <branch-name>
```

---

## API & Exchange Notes

### BingX Hedge Mode Constraints

1. **reduce_only NOT supported**: Cannot use `reduce_only=True` in Hedge Mode
2. **Position Side Required**: Must specify `positionSide='LONG'` or `positionSide='SHORT'`
3. **Conditional Orders**: Use Raw API, not CCXT for conditional orders

### Order Creation Pattern

```python
# Correct Hedge Mode order
exchange.create_order(
    symbol='BTC/USDT:USDT',
    type='market',
    side='buy',
    amount=qty,
    params={
        'positionSide': 'LONG',  # Required in Hedge Mode
        # 'reduceOnly': False  # DO NOT use reduce_only
    }
)
```

### Position Sizing

```python
# Calculate position size (4x leverage)
EFFECTIVE_LEVERAGE = 4
available_margin = balance * 0.95  # Use 95%
position_value = available_margin * EFFECTIVE_LEVERAGE
qty = position_value / current_price
```

---

## Legacy Bots (Reference Only)

### ADX Supertrend Trail Bot - DEPRECATED

**Status**: Backtest bug discovered. Results were invalid (+1276% fake → -234% real).

| Parameter | Value |
|-----------|-------|
| Entry | ADX > 20 + DI Crossover |
| TP | 2.0% |
| SL | Supertrend Trail (dynamic) |

**Bug**: Exit price calculated using Supertrend value instead of actual candle price.

### Other Legacy Bots

| Bot | File | Status |
|-----|------|--------|
| Supertrend Regime Bot | `supertrend_regime_bot.py` | LEGACY |
| RSI Zone Bot v2.2 | `rsi_zone_bot.py` | LEGACY |
| EMA Crossover Bot | `ema_crossover_bot.py` | LEGACY |
| VWAP Band Bot | `vwap_band_bot.py` | LEGACY |
| Donchian Scalping Bot | `donchian_scalping_bot.py` | LEGACY |

---

## Strategy Research History

### December 2025 Research

1. **ADX Supertrend Bug Discovery** (2025-12-16)
   - Backtest had exit price bug
   - +1276% result was fake → Real: -234%

2. **Alternative Strategy Comparison** (2025-12-16)
   - Tested 8 strategies × 5 TP/SL combinations
   - RSI Trend Filter selected as best performer

3. **RSI Parameter Optimization**
   | Variant | PnL | P-value |
   |---------|-----|---------|
   | RSI 35/65 EMA200 | +54.4% | 0.38 |
   | **RSI 40/60 EMA100** | **+120.8%** | **0.013** |
   | RSI 45/55 EMA100 | +87.3% | 0.08 |
   | RSI 30/70 EMA100 | +23.1% | 0.52 |

### Key Research Documents

| Document | Content |
|----------|---------|
| `claudedocs/DYNAMIC_STOPLOSS_RESEARCH_20251213.md` | Dynamic SL research |
| `scripts/analysis/alternative_strategies_research.py` | 8 strategy comparison |
| `scripts/analysis/best_strategy_validation.py` | Final validation |

---

## AI Assistant Instructions

### Critical Rules

1. **NEVER modify code without reading it first**
2. **Hedge Mode**: No `reduce_only` parameter
3. **Position Sizing**: Always use EFFECTIVE_LEVERAGE (4x)
4. **State Files**: Back up before modification
5. **CCXT Limitations**: Use Raw API for conditional orders

### Common Tasks

#### Check Bot Status
```bash
cat bingx_rl_trading_bot/results/rsi_trend_filter_bot_state.json
```

#### Modify Strategy Parameters
Edit `config/rsi_trend_filter_config.yaml` - see Parameter Modification Examples above.

#### View Logs
```bash
tail -100 bingx_rl_trading_bot/logs/rsi_trend_filter_bot_$(date +%Y%m%d).log
```

#### Check Current Position
```bash
python scripts/production/check_current_positions.py
```

### File Naming Conventions

| Type | Pattern | Example |
|------|---------|---------|
| Bot | `{name}_bot.py` | `rsi_trend_filter_bot.py` |
| Config | `{name}_config.yaml` | `rsi_trend_filter_config.yaml` |
| Monitor | `{name}_monitor.py` | `rsi_trend_filter_monitor.py` |
| State | `{name}_state.json` | `rsi_trend_filter_bot_state.json` |
| Log | `{name}_YYYYMMDD.log` | `rsi_trend_filter_bot_20251219.log` |
| Research | `{TOPIC}_YYYYMMDD.md` | `DYNAMIC_STOPLOSS_RESEARCH_20251213.md` |

---

## Known Issues & Solutions

| Issue | Solution |
|-------|----------|
| `reduce_only` error in Hedge Mode | Remove `reduce_only` param, use `positionSide` |
| Conditional order fails with CCXT | Use exchange's Raw API directly |
| State file corruption | Restore from `results/backups/` |
| API rate limiting | Increase retry delay in config |

---

## Changelog

| Date | Change |
|------|--------|
| 2025-12-19 | Updated documentation, improved structure |
| 2025-12-16 | RSI Trend Filter v1.0 deployed |
| 2025-12-16 | ADX Supertrend bug discovered, deprecated |
| 2025-12-16 | 8-strategy comparison research |
| 2025-12-13 | ADX Supertrend Trail v1.0 (later found buggy) |
| 2025-12-12 | Entry Signal Research (21 methods) |

---

**Last Updated**: 2025-12-19 KST
