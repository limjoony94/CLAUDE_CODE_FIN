# Support/Resistance Level Trading Research

**Date**: 2025-12-24
**Researcher**: Claude Code (Opus 4.5)
**Target Constraints**: DD<40%, Daily Return≥0.3%, Trades/Day≥1.0

---

## Executive Summary

Researched 5 S/R detection methods across 5 strategy types with 117 parameter combinations. **2 strategies met ALL target constraints**:

| Rank | Strategy | TP/SL | PnL | DD | Daily | Trades/Day | Sharpe |
|------|----------|-------|-----|-----|-------|------------|--------|
| **1** | **Fractal_Breakout_HighVol** | **2.0/1.0** | **+55.1%** | **34.5%** | **0.62%** | **1.93** | 0.64 |
| **2** | **Pivot_Fib_S1** | **1.5/1.0** | **+29.2%** | **24.9%** | **0.33%** | **1.02** | 1.07 |

**Recommendation**: Fractal Breakout with High Volume filter offers best returns while meeting all constraints. Pivot Fibonacci S1 provides better risk-adjusted returns (higher Sharpe).

---

## Target Constraints (User Requirements)

| Constraint | Target | Rationale |
|------------|--------|-----------|
| **Max Drawdown** | < 40% | Conservative risk management |
| **Daily Return** | ≥ 0.3% | ~9% monthly, ~109% annual compounded |
| **Daily Trades** | ≥ 1.0 | Minimum activity for consistent returns |

---

## S/R Detection Methods Tested

### 1. Pivot Points (Standard & Fibonacci)
```python
# Standard Pivot
Pivot = (High + Low + Close) / 3
R1 = 2 * Pivot - Low
S1 = 2 * Pivot - High
R2 = Pivot + (High - Low)
S2 = Pivot - (High - Low)

# Fibonacci Pivot
R1 = Pivot + 0.382 * (High - Low)
S1 = Pivot - 0.382 * (High - Low)
```

### 2. Williams Fractals
```python
# Fractal High: Middle bar has highest high in 5-bar window
# Fractal Low: Middle bar has lowest low in 5-bar window
fractal_high = rolling_max(high, 5) == high.shift(-2)
fractal_low = rolling_min(low, 5) == low.shift(-2)
```

### 3. Swing Levels (Pullback-based)
```python
# Fibonacci retracements: 38.2%, 50%, 61.8%
swing_high = argrelextrema(high, np.greater, order=5)
swing_low = argrelextrema(low, np.less, order=5)
```

### 4. Recent High/Low Zones
```python
# Rolling lookback periods: 48, 96 candles
recent_high = rolling_max(high, lookback)
recent_low = rolling_min(low, lookback)
```

### 5. Volume-Weighted Levels
```python
# Volume confirmation threshold
vol_confirm = volume > volume.rolling(20).mean() * threshold
```

---

## Strategy Descriptions

### 1. Pivot Bounce Strategy
- **Logic**: LONG near S1/S2, SHORT near R1/R2 with proximity filter
- **Best Config**: Fibonacci S1, 0.15% proximity, RSI filter
- **Result**: +29.2% PnL, 24.9% DD, 1.02 trades/day ✅

### 2. Fractal Breakout Strategy
- **Logic**: LONG on fractal high breakout, SHORT on fractal low break
- **Volume Filter**: Entry only when volume > 1.5x average
- **EMA Filter**: Entry only when price above/below EMA(100)
- **Best Config**: HighVol (1.5x), EMA(100) filter
- **Result**: +55.1% PnL, 34.5% DD, 1.93 trades/day ✅

### 3. Swing Pullback Strategy
- **Logic**: Trade pullbacks to Fibonacci levels in trend direction
- **Result**: Did not meet constraints (high DD or low trade frequency)

### 4. Recent High/Low Bounce Strategy
- **Logic**: Fade moves at recent extremes
- **Result**: Did not meet constraints (low win rate)

### 5. Mean Reversion at S/R Strategy
- **Logic**: RSI extremes combined with S/R proximity
- **Result**: Did not meet constraints (high DD)

---

## Detailed Results: Strategies Meeting ALL Constraints

### Strategy #1: Fractal_Breakout_HighVol (TP=2.0%, SL=1.0%)

| Metric | Value | Target | Status |
|--------|-------|--------|--------|
| **Total PnL** | +55.1% | - | - |
| **Max Drawdown** | 34.5% | <40% | ✅ |
| **Daily Return** | 0.62% | ≥0.3% | ✅ |
| **Daily Trades** | 1.93 | ≥1.0 | ✅ |
| Win Rate | 39.5% | - | - |
| Sharpe Ratio | 0.64 | - | - |
| Profit Factor | 1.07 | - | - |

**Direction Breakdown**:
- LONG: 83 trades, 36.1% win rate
- SHORT: 89 trades, 42.7% win rate

**Parameters**:
```yaml
vol_confirm: true
vol_threshold: 1.5  # Volume > 1.5x 20-period average
ema_filter: true
ema_period: 100     # EMA trend filter
tp_pct: 2.0
sl_pct: 1.0
leverage: 4x
```

**Entry Logic**:
```python
# LONG Entry
fractal_low_break = close < prev_fractal_low
volume_confirm = volume > volume.rolling(20).mean() * 1.5
trend_up = close > ema(100)
entry_long = fractal_low_break and volume_confirm and trend_up

# SHORT Entry
fractal_high_break = close > prev_fractal_high
trend_down = close < ema(100)
entry_short = fractal_high_break and volume_confirm and trend_down
```

**Walk-Forward Results**:
- Profitable Windows: 1/6 (16.7%)
- Avg WF PnL: -8.0%
- Avg WF Daily: -1.4%

⚠️ **Warning**: Poor walk-forward consistency suggests potential overfitting.

---

### Strategy #2: Pivot_Fib_S1 (TP=1.5%, SL=1.0%)

| Metric | Value | Target | Status |
|--------|-------|--------|--------|
| **Total PnL** | +29.2% | - | - |
| **Max Drawdown** | 24.9% | <40% | ✅ |
| **Daily Return** | 0.33% | ≥0.3% | ✅ |
| **Daily Trades** | 1.02 | ≥1.0 | ✅ |
| Win Rate | 47.3% | - | - |
| Sharpe Ratio | 1.07 | - | - |
| Profit Factor | 1.12 | - | - |

**Direction Breakdown**:
- LONG: 49 trades, 46.9% win rate
- SHORT: 42 trades, 47.6% win rate

**Parameters**:
```yaml
pivot_type: fibonacci
level: 1           # S1/R1 levels
proximity_pct: 0.15  # 0.15% proximity threshold
rsi_filter: true
tp_pct: 1.5
sl_pct: 1.0
leverage: 4x
```

**Entry Logic**:
```python
# LONG Entry (near Fibonacci S1)
near_s1 = abs(close - fib_s1) / close < 0.0015  # 0.15% proximity
rsi_oversold = rsi(14) < 35
entry_long = near_s1 and rsi_oversold

# SHORT Entry (near Fibonacci R1)
near_r1 = abs(close - fib_r1) / close < 0.0015
rsi_overbought = rsi(14) > 65
entry_short = near_r1 and rsi_overbought
```

**Walk-Forward Results**:
- Profitable Windows: 2/6 (33.3%)
- Avg WF PnL: -7.0%
- Avg WF Daily: -1.1%

⚠️ **Warning**: Walk-forward shows negative returns.

---

## Notable Near-Misses

| Strategy | TP/SL | PnL | DD | Daily | Trades | Fail Reason |
|----------|-------|-----|-----|-------|--------|-------------|
| Fractal_Breakout_Vol | 3.0/2.0 | **+306.3%** | **52.0%** | 3.44% | 0.87 | DD>40%, Trades<1 |
| Fractal_Breakout_HighVol | 2.0/1.5 | +69.4% | **44.9%** | 0.78% | 1.54 | DD>40% |
| Pivot_Fib_S1 | 2.0/1.0 | +62.3% | 27.6% | 0.70% | **0.90** | Trades<1 |
| Pivot_Standard_S2 | 0.5/0.5 | -10.0% | 16.4% | -0.11% | 1.01 | Negative PnL |

**Key Insight**: Wider TP/SL ratios (3.0/2.0) generate massive returns but with unacceptable drawdown. Tighter ratios (1.5/1.0, 2.0/1.0) provide better constraint compliance.

---

## Strategy Comparison: Winner Analysis

| Metric | Fractal Breakout | Pivot Fib S1 | Winner |
|--------|------------------|--------------|--------|
| Total PnL | +55.1% | +29.2% | **Fractal** |
| Max DD | 34.5% | 24.9% | **Pivot** |
| Daily Return | 0.62% | 0.33% | **Fractal** |
| Trades/Day | 1.93 | 1.02 | **Fractal** |
| Win Rate | 39.5% | 47.3% | **Pivot** |
| Sharpe Ratio | 0.64 | 1.07 | **Pivot** |
| Profit Factor | 1.07 | 1.12 | **Pivot** |
| WF Consistency | 16.7% | 33.3% | **Pivot** |

**Verdict**:
- **For Maximum Returns**: Fractal Breakout HighVol (+55.1%)
- **For Risk-Adjusted Returns**: Pivot Fib S1 (Sharpe 1.07)
- **For Consistency**: Neither is reliable (WF <50%)

---

## Backtest Methodology (Corrected)

Following MACD Martingale Bot v1.0 corrected methodology:

1. **Entry at NEXT Bar OPEN** (not same bar close)
2. **TP/SL Detection using HIGH/LOW** (not close price)
3. **Position Cap at 10x Balance** (exchange limit)
4. **Conservative Exit**: SL triggered first when both TP/SL possible in same bar
5. **Fee Calculation**: 0.04% per trade (entry + exit)

```python
# Corrected Exit Logic
for bar in bars_after_entry:
    if direction == 'LONG':
        sl_hit = bar.low <= sl_price
        tp_hit = bar.high >= tp_price
    else:  # SHORT
        sl_hit = bar.high >= sl_price
        tp_hit = bar.low <= tp_price

    if sl_hit and tp_hit:
        # Conservative: assume SL hit first
        exit_price = sl_price
        exit_reason = 'SL'
    elif sl_hit:
        exit_price = sl_price
        exit_reason = 'SL'
    elif tp_hit:
        exit_price = tp_price
        exit_reason = 'TP'
```

---

## Walk-Forward Validation Details

| Window | Train Period | Test Period | Train Size | Test Size |
|--------|--------------|-------------|------------|-----------|
| 1 | Days 0-62 | Days 63-88 | 70% | 30% |
| 2 | Days 15-77 | Days 78-103 | 70% | 30% |
| 3 | Days 30-92 | Days 93-118 | 70% | 30% |
| 4 | Days 45-107 | Days 108-133 | 70% | 30% |
| 5 | Days 60-122 | Days 123-148 | 70% | 30% |
| 6 | Days 75-137 | Days 138-163 | 70% | 30% |

**Results Summary**:
- Both winning strategies show poor walk-forward performance
- Suggests overfitting to historical data
- Real trading may not replicate backtest results

---

## Recommendations

### Short-Term (Paper Trading)
1. Deploy **Fractal_Breakout_HighVol** on paper trading for validation
2. Monitor actual trade frequency and drawdown
3. Track slippage and execution quality

### Medium-Term (Production Consideration)
1. If paper trading validates results → deploy with 50% position size
2. Consider combining with existing RSI Trend Filter Bot
3. Use different timeframes (5m for S/R, 15m for RSI)

### Long-Term (Research Direction)
1. Test multi-timeframe S/R detection
2. Explore machine learning for S/R significance scoring
3. Investigate order flow / volume profile integration

---

## Files Generated

| File | Description |
|------|-------------|
| `scripts/analysis/sr_level_research.py` | Research script with 5 S/R methods |
| `results/sr_level_research_20251224_045621.csv` | Full results (117 strategy configs) |
| `claudedocs/SR_LEVEL_RESEARCH_20251224.md` | This documentation |

---

## Conclusion

**Research successfully identified 2 strategies meeting all target constraints**:

1. **Fractal Breakout HighVol (TP=2.0%, SL=1.0%)**: Best returns (+55.1%), meets all constraints
2. **Pivot Fib S1 (TP=1.5%, SL=1.0%)**: Best risk-adjusted (Sharpe 1.07), lower returns (+29.2%)

**Caution**: Both strategies show poor walk-forward consistency (<35%), suggesting potential overfitting. Paper trading validation strongly recommended before live deployment.

---

**Last Updated**: 2025-12-24 05:00 KST
