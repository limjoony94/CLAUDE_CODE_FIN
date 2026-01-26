# BingX RL Trading Bot

**Current System**: Pattern 5m Bot v1.17 (3-Candle Pattern Strategy)

## Documentation

For complete, up-to-date documentation, see the root **[CLAUDE.md](../CLAUDE.md)** file.

## Quick Start

```bash
# Start Bot
START_PATTERN_5M.bat
# or: python -m scripts.production.pattern_5m_bot

# Stop Bot
STOP_PATTERN_5M.bat

# Monitor
MONITOR_PATTERN_5M.bat
```

## Strategy Overview

**Pattern 5m** uses a 3-candle pattern recognition system with 12 candle types:

| Code | Type | Description |
|------|------|-------------|
| D | DOJI | body < 10% of range |
| ST | SPINNING_TOP | small body, balanced wicks |
| H | HAMMER | lower wick > 2x body |
| IH | INV_HAMMER | upper wick > 2x body |
| MU | MARUBOZU_UP | bullish, wicks < 15% |
| MD | MARUBOZU_DOWN | bearish, wicks < 15% |
| BU | BIG_UP | normalized body > 1.5 |
| BD | BIG_DOWN | normalized body > 1.5 |
| U | MED_UP | medium bullish |
| DN | MED_DOWN | medium bearish |

**v1.17 Stats**: 18 validated patterns (8 LONG + 10 SHORT), 83.9% avg WR, 14/18 statistically significant

## Key Files

| Purpose | Location |
|---------|----------|
| Entry Point | `scripts/production/pattern_5m_bot.py` |
| Bot Package | `scripts/production/pattern_5m/` |
| Config | `config/pattern_5m_config.yaml` |
| State | `results/pattern_5m_bot_state.json` |
| Metrics | `results/pattern_5m_metrics.json` |
| Logs | `logs/pattern_5m_bot_*.log` |
| API Client | `src/api/bingx_client.py` |

## Directory Structure

```
bingx_rl_trading_bot/
├── scripts/
│   ├── production/
│   │   ├── pattern_5m_bot.py    # Entry point
│   │   └── pattern_5m/          # Bot package (14 modules)
│   ├── analysis/                # Research scripts
│   └── utils/                   # Utility scripts
├── src/api/                     # BingX API client
├── config/                      # Configuration files
├── results/                     # State & results
├── logs/                        # Log files
├── claudedocs/                  # Analysis documentation
└── archive/                     # Deprecated bots
```

## Key Features (v1.17)

- **18 Validated Patterns**: Statistically validated with WF ≥4/5
- **Pattern-Specific TP/SL**: Optimized per pattern
- **TP/SL Auto-Adjustment**: On bot startup, adjusts existing position's TP/SL to match config
- **Early Exit Signal**: 3x consecutive BD/BU reversal detection
- **Context Filters**: RSI/Vol/Trend-based pattern filtering
- **Crash Recovery**: Orphan position detection and recovery
- **API Caching**: 5s TTL for ticker/balance/positions
- **Circuit Breaker**: 5 failures → 60s block

---
**Last Updated**: 2026-01-26 | **Version**: v1.17
