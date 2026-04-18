Run project tests.

## Quick Start

```bash
cd bingx_rl_trading_bot && python -m pytest scripts/tests/ -v --tb=short
```

Expected: **113 passed** in ~5s (includes property-based tests + coverage). No failures allowed.

## Coverage (v4.7.5, v4.7.9 기준 유효)

```
Name                                           Stmts   Miss  Cover
------------------------------------------------------------------
scripts\production\c1_breakout\__init__.py         0      0   100%
scripts\production\c1_breakout\bot.py            669    235    65%
scripts\production\c1_breakout\config.py          24      0   100%
scripts\production\c1_breakout\indicators.py      37      0   100%
scripts\production\c1_breakout\signals.py         79      0   100%
------------------------------------------------------------------
TOTAL                                            809    235    71%
```

signals/indicators/config: **100%**. bot.py: main loop + init 제외 핵심 로직 커버.

Run with coverage: `python -m pytest scripts/tests/ --cov=scripts/production/c1_breakout --cov-report=term-missing`

## Test Suite Overview (v4.7.5+)

| Module | Count | Covers |
|--------|-------|--------|
| `test_indicators.py` | 10 | ATR (Wilder), Channel (causal), Fractal swings. Warmup NaN, flat-data, no-future-leak |
| `test_signals.py` | 19 | check_entry (breakout + body + SL clamp), check_exit (priority SL→EMG→TO→TRAIL), BUG#53/60 guards |
| `test_config.py` | 11 | Defaults, leverage validation (BUG#52), SL bounds, deep merge |
| `test_bot.py` | 18 | Orphan SL restoration (BUG#48, 7 cases), wall-clock reconciliation (BUG#54), I/O defense (BUG#58), time helpers (BUG#57), history trim (BUG#56) |
| `test_bot_methods.py` | — | Internal method tests (`_calc_amount`, `_exchange_close`, etc.) |
| `test_exchange_open.py` | 7 | MARKET retry (BUG#38), SL filled_qty (BUG#28), emergency close (BUG#26), partial fill (BUG#55) |
| `test_sync_exchange.py` | 8 | Orphan adoption end-to-end (BUG#48), ghost orderType classification (BUG#50), entry_time filter (BUG#36), exchange timestamp (BUG#45), API error preservation |
| `test_trail_update.py` | 8 | force_reset one-shot (BUG#35), LOOSEN-only policy (BUG#46), failure streak (BUG#59) |
| `test_process_candles.py` | — | fetch_candles streak (BUG#51), process_candles integration |
| `test_properties.py` | 6 | Hypothesis-based: ATR causality invariant, signal sl_pct bounds, exit priority invariant, timeout guarantee. ~400 random cases |
| **Total** | **113** | 4-angle critical-evaluation + property-based |

> BUG#61~65 (2026-04-18 정합성) — `test_signals.py` trail formula 테스트가 regress 방지 보증. 전용 integration 테스트는 향후 추가 여지.

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
