Run project tests.

## Quick Start

```bash
cd bingx_rl_trading_bot && python -m pytest scripts/tests/ -v --tb=short
```

Expected: **58 passed** in ~1s. No failures allowed.

## Test Suite Overview (v4.7.2+)

| Module | Count | Covers |
|--------|-------|--------|
| `test_indicators.py` | 10 | ATR (Wilder), Channel (causal), Fractal swings. Warmup NaN, flat-data, no-future-leak |
| `test_signals.py` | 19 | check_entry (breakout + body + SL clamp), check_exit (priority SL→EMG→TO→TRAIL), BUG#53/60 guards |
| `test_config.py` | 11 | Defaults, leverage validation (BUG#52), SL bounds, deep merge |
| `test_bot.py` | 18 | Orphan SL restoration (BUG#48, 7 cases), wall-clock reconciliation (BUG#54), I/O defense (BUG#58), time helpers (BUG#57), history trim (BUG#56) |
| **Total** | **58** | 4-angle critical-evaluation cases |

## Smoke Test (import only)

```bash
cd bingx_rl_trading_bot && python -c "from scripts.production.c1_breakout import bot, signals, indicators, config; print('imports OK')"
```

## Deprecation-Strict Check (Python 3.12+)

```bash
cd bingx_rl_trading_bot && python -W error::DeprecationWarning -c "from scripts.production.c1_breakout import bot; print('no deprecation')"
```

## Regression Guards per BUG

Each BUG fix has a dedicated test case — identity mapping from bug → test:

- BUG#48 (orphan SL) → `TestResolveOrphanSL` (7 tests)
- BUG#52 (leverage validation) → `TestLeverageValidation` (5 tests)
- BUG#53 (channel sanity) → `test_bug53_flat_channel_rejected`
- BUG#54 (wall-clock) → `TestLastExitTimeReconciliation` (5 tests)
- BUG#56 (memory cap) → `TestTradeHistoryTrim` (2 tests)
- BUG#57 (naive ISO) → `TestTimeHelpers` (2 tests)
- BUG#58 (I/O defense) → `TestStateIOResilience` (2 tests)
- BUG#60 (NaN/zero close) → `test_bug60_*` (3 tests)

## Adding New Tests

Each test should document its critical-evaluation angle in the docstring:
- **A. Edge conditions** (null/empty/extreme)
- **B. Backtest parity** (live changes must not alter backtest)
- **C. Bug interaction** (new fix doesn't break old)
- **D. Rollback safety** (failures degrade gracefully)

Place in appropriate `test_*.py`. Share fixtures via `conftest.py`.
