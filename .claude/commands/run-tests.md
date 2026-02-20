Run the project test suite and report results.

```bash
cd bingx_rl_trading_bot && python -m pytest scripts/tests/ -v --tb=short 2>&1 | tail -80
```

Expected: 1139+ tests all passing (v1.28.42 baseline)

After running:
1. Report total tests passed/failed/skipped
2. If any failures: show the failing test names and error messages
3. Compare against baseline (1139 tests expected)
4. If new tests were added, note the count increase
5. If tests were removed, flag for review

For specific test files:
- `test_patterns.py` — Pattern validation and stats
- `test_pure_functions.py` — PnL calculation, pattern extraction, scale-out
- `test_config.py` — Dynamic pattern loading, config validation
- `test_indicators.py` — Candle classification accuracy

Do NOT skip or disable any failing tests. Investigate root cause instead.
