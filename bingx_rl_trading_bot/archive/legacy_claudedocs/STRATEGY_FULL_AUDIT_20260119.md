# Strategy Full Audit Report

**Generated**: 2026-01-19 04:49
**Protocol**: Standard Research Protocol v1.0

---

## Executive Summary

| Metric | Value |
|--------|-------|
| Total Strategies | 7 |
| Passed All Tests | 0 |
| Failed | 7 |

---

## Detailed Results

### ❌ Engulf 5m v1.9

| Test | Result | Status |
|------|--------|--------|
| Type 1: Signals | 20 | ❌ |
| Type 1: Win Rate | 30.0% | ❌ |
| Type 1: Expected Value | -2.25% | ❌ |
| Type 2: PnL | +70.62% | ✅ |
| Type 2: Trades | 39 | - |
| Type 2: Max DD | 16.5% | ✅ |
| Walk-Forward | 62% | ✅ |
| Monte Carlo | 100.0% | ✅ |

### ❌ EMA Crossover v1.5

| Test | Result | Status |
|------|--------|--------|
| Type 1: Signals | 444 | ❌ |
| Type 1: Win Rate | 43.0% | ❌ |
| Type 1: Expected Value | -0.38% | ❌ |
| Type 2: PnL | -67.16% | ❌ |
| Type 2: Trades | 231 | - |
| Type 2: Max DD | 85.2% | ❌ |
| Walk-Forward | 38% | ❌ |
| Monte Carlo | 0.0% | ❌ |

### ❌ Supertrend Regime v1.0

| Test | Result | Status |
|------|--------|--------|
| Type 1: Signals | 0 | ❌ |
| Type 1: Win Rate | 0.0% | ❌ |
| Type 1: Expected Value | 0.00% | ❌ |
| Type 2: PnL | +0.00% | ❌ |
| Type 2: Trades | 0 | - |
| Type 2: Max DD | 0.0% | ❌ |
| Walk-Forward | 0% | ❌ |
| Monte Carlo | 0.0% | ❌ |

### ❌ RSI Zone v2.2

| Test | Result | Status |
|------|--------|--------|
| Type 1: Signals | 2548 | ❌ |
| Type 1: Win Rate | 40.9% | ❌ |
| Type 1: Expected Value | -0.67% | ❌ |
| Type 2: PnL | -96.10% | ❌ |
| Type 2: Trades | 398 | - |
| Type 2: Max DD | 97.8% | ❌ |
| Walk-Forward | 25% | ❌ |
| Monte Carlo | 0.0% | ❌ |

### ❌ MACD Crossover

| Test | Result | Status |
|------|--------|--------|
| Type 1: Signals | 1487 | ❌ |
| Type 1: Win Rate | 41.2% | ❌ |
| Type 1: Expected Value | -0.64% | ❌ |
| Type 2: PnL | -79.53% | ❌ |
| Type 2: Trades | 399 | - |
| Type 2: Max DD | 95.2% | ❌ |
| Walk-Forward | 25% | ❌ |
| Monte Carlo | 0.0% | ❌ |

### ❌ BB + Stochastic

| Test | Result | Status |
|------|--------|--------|
| Type 1: Signals | 1897 | ❌ |
| Type 1: Win Rate | 39.7% | ❌ |
| Type 1: Expected Value | -0.84% | ❌ |
| Type 2: PnL | -99.31% | ❌ |
| Type 2: Trades | 432 | - |
| Type 2: Max DD | 99.3% | ❌ |
| Walk-Forward | 0% | ❌ |
| Monte Carlo | 0.0% | ❌ |

### ❌ ATR Breakout

| Test | Result | Status |
|------|--------|--------|
| Type 1: Signals | 264 | ❌ |
| Type 1: Win Rate | 32.2% | ❌ |
| Type 1: Expected Value | -0.60% | ❌ |
| Type 2: PnL | +61.70% | ❌ |
| Type 2: Trades | 179 | - |
| Type 2: Max DD | 55.8% | ❌ |
| Walk-Forward | 50% | ✅ |
| Monte Carlo | 100.0% | ✅ |

---

## Summary Table

| Strategy | Status | Type1 | Type2 | WF | MC | Overall |
|----------|--------|-------|-------|----|----|---------|
| Engulf 5m v1.9 | ACTIVE | ❌ | ✅ | ✅ | ✅ | ❌ |
| EMA Crossover v1.5 | ARCHIVED | ❌ | ❌ | ❌ | ❌ | ❌ |
| Supertrend Regime v1.0 | ARCHIVED | ❌ | ❌ | ❌ | ❌ | ❌ |
| RSI Zone v2.2 | ARCHIVED | ❌ | ❌ | ❌ | ❌ | ❌ |
| MACD Crossover | ARCHIVED | ❌ | ❌ | ❌ | ❌ | ❌ |
| BB + Stochastic | ARCHIVED | ❌ | ❌ | ❌ | ❌ | ❌ |
| ATR Breakout | ARCHIVED | ❌ | ❌ | ✅ | ✅ | ❌ |

---

## Recommendations

### Not Recommended
- **Engulf 5m v1.9**: Type 1 failed
- **EMA Crossover v1.5**: Type 1 failed, Type 2 failed, Walk-Forward failed, Monte Carlo failed
- **Supertrend Regime v1.0**: Type 1 failed, Type 2 failed, Walk-Forward failed, Monte Carlo failed
- **RSI Zone v2.2**: Type 1 failed, Type 2 failed, Walk-Forward failed, Monte Carlo failed
- **MACD Crossover**: Type 1 failed, Type 2 failed, Walk-Forward failed, Monte Carlo failed
- **BB + Stochastic**: Type 1 failed, Type 2 failed, Walk-Forward failed, Monte Carlo failed
- **ATR Breakout**: Type 1 failed, Type 2 failed
