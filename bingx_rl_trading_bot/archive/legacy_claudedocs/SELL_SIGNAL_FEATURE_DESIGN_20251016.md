# SELL Signal Feature Design - BUY/SELL Paradigm

**Date**: 2025-10-16
**Purpose**: Unified feature design for SELL signals (SHORT Entry + LONG Exit)

---

## Paradigm Shift: BUY/SELL Model Pairs

**OLD Thinking** (WRONG):
- LONG models: LONG Entry + LONG Exit (separate from SHORT)
- SHORT models: SHORT Entry + SHORT Exit (separate from LONG)

**NEW Thinking** (CORRECT):
- **BUY pair**: LONG Entry + SHORT Exit (both are BUYING actions)
- **SELL pair**: SHORT Entry + LONG Exit (both are SELLING actions)

---

## SELL Signal Features (22 Enhanced Features)

### Core Principle
SHORT Entry and LONG Exit should use **IDENTICAL features** because both identify good SELL opportunities.

### Feature List (from LONG Exit Enhanced Model)

**1. Base Indicators (3)**
```
- rsi: Momentum indicator
- macd: Trend following
- macd_signal: MACD signal line
```

**2. Volume Analysis (2)**
```
- volume_ratio: Current volume / 20-period average
- volume_surge: Volume > 1.5x average (binary)
```

**3. Price Momentum (3)**
```
- price_acceleration: Second derivative of price (turning points)
- price_vs_ma20: Distance from 20-period MA
- price_vs_ma50: Distance from 50-period MA
```

**4. Volatility Metrics (2)**
```
- volatility_20: 20-period returns std
- volatility_regime: High/low volatility (binary)
```

**5. RSI Dynamics (4)**
```
- rsi_slope: Rate of RSI change (3 candles)
- rsi_overbought: RSI > 70 (SELL signal)
- rsi_oversold: RSI < 30 (not SELL signal)
- rsi_divergence: Placeholder (0)
```

**6. MACD Dynamics (3)**
```
- macd_histogram_slope: MACD histogram rate of change
- macd_crossover: Bullish MACD cross (not SELL signal)
- macd_crossunder: Bearish MACD cross (SELL signal)
```

**7. Price Patterns (2)**
```
- higher_high: Higher high vs previous (not SELL signal)
- lower_low: Lower low vs previous (SELL signal)
```

**8. Support/Resistance (2)**
```
- near_resistance: Close > resistance * 0.98 (SELL signal)
- near_support: Close < support * 1.02 (not SELL signal)
```

**9. Bollinger Bands (1)**
```
- bb_position: Normalized position in BB range [0-1]
```

**Total: 22 features**

---

## Why These Features Identify SELL Signals

### For LONG Exit (Selling to Close)
| Feature | Why Good SELL Signal |
|---------|----------------------|
| `rsi_overbought` | Price overextended, time to take profit |
| `near_resistance` | Price at ceiling, likely reversal |
| `macd_crossunder` | Bearish momentum shift |
| `lower_low` | Downtrend forming, exit before further drop |
| `volume_surge` | Potential reversal, take profit |
| `price_acceleration` < 0 | Momentum turning down |

### For SHORT Entry (Selling to Open)
| Feature | Why Good SELL Signal |
|---------|----------------------|
| `rsi_overbought` | Price overextended, good short entry |
| `near_resistance` | Price at ceiling, good short entry |
| `macd_crossunder` | Bearish momentum, good short entry |
| `lower_low` | Downtrend starting, good short entry |
| `volume_surge` | Reversal signal, good short entry |
| `price_acceleration` < 0 | Momentum turning down, good short entry |

**Same features, same interpretation → Perfect alignment!**

---

## Comparison: Current vs Enhanced

### Current SHORT Entry (67 features)
**Problems**:
- Multi-timeframe general indicators (15min, 1h, 4h, 1d)
- No specific SELL signal focus
- Result: 1% signal rate, 20% win rate

**Features**:
```
close_change_1, close_change_2, ..., close_change_5
sma_10, sma_20, ema_10, ema_3, ema_5
rsi, rsi_5, rsi_7, rsi_15min, rsi_1h, rsi_4h, rsi_1d
macd, macd_signal, macd_diff, macd_1h, macd_4h
bb_high, bb_low, bb_mid, bb_position_1h, bb_position_4h
volatility, volatility_5, volatility_10
volume_ratio, volume_spike, volume_trend
... (67 total)
```

### Enhanced SHORT Entry (22 features)
**Advantages**:
- **Specific SELL signal indicators**
- **Aligned with LONG Exit** (SELL pair consistency)
- **Proven success** (+14.44% return in LONG Exit)

**Features**:
```
rsi, macd, macd_signal
volatility_regime, volatility_20
volume_surge, volume_ratio
price_acceleration, price_vs_ma20, price_vs_ma50
rsi_slope, rsi_overbought, rsi_oversold, rsi_divergence
macd_histogram_slope, macd_crossover, macd_crossunder
higher_high, lower_low
near_resistance, near_support
bb_position
```

---

## Expected Improvements

### Signal Quality
**Current**: 1% signal rate → **Expected**: 10-20% signal rate
- Better labeling (2of3 scoring vs Peak/Trough)
- SELL-focused features (not general multi-timeframe)

### Win Rate
**Current**: 20% win rate → **Expected**: 60-70% win rate
- Features actually identify SELL opportunities
- Aligned with LONG Exit success (67.4% win rate)

### Model Consistency
**Current**: 4 independent models → **Expected**: 2 model pairs
- SELL pair: SHORT Entry + LONG Exit (shared features)
- BUY pair: LONG Entry + SHORT Exit (to be designed later)

---

## Implementation Plan

### Phase 1: Retrain SHORT Entry with Enhanced Features
1. Use improved labeling (2of3 scoring)
2. Use LONG Exit enhanced features (22 features)
3. Train on same data as current model
4. Compare performance: old vs enhanced

### Phase 2: Optimize Labeling Parameters
1. Test with real BTC data
2. Find optimal profit_threshold (0.5% - 1.0%)
3. Find optimal lead_time window (3-24 candles)
4. Target: 10-20% positive rate

### Phase 3: Validation
1. Backtest enhanced SHORT Entry model
2. Compare with current model (1% signal, 20% win)
3. Target: >10% signal rate, >60% win rate
4. Validate SELL pair consistency (SHORT Entry + LONG Exit)

### Phase 4: Deployment
1. Update production bot with enhanced SHORT Entry
2. Monitor SELL pair performance
3. Weekly retraining schedule
4. Consider BUY pair enhancement (LONG Entry + SHORT Exit)

---

## Key Insights

### 1. Feature Sharing is Critical
- SELL signals (SHORT Entry + LONG Exit) need same features
- BUY signals (LONG Entry + SHORT Exit) need same features
- This is NOT about position type (LONG/SHORT) but action type (BUY/SELL)

### 2. EXIT Features Work for ENTRY
- LONG Exit enhanced features identify SELL opportunities
- SHORT Entry needs SELL opportunities
- Therefore: SHORT Entry should use LONG Exit features

### 3. Multi-Timeframe ≠ Better
- Current 67-feature approach: general indicators across timeframes
- Enhanced 22-feature approach: specific SELL signal indicators
- Quality > Quantity: Focused features outperform general features

---

## Next Steps

1. ✅ Design complete: Use LONG Exit 22 features for SHORT Entry
2. ⏳ Test labeling with real BTC data (parameter optimization)
3. ⏳ Retrain SHORT Entry model (2of3 scoring + 22 enhanced features)
4. ⏳ Backtest validation (compare old vs enhanced)
5. ⏳ Deploy enhanced SHORT Entry to production
6. ⏳ Consider BUY pair enhancement (LONG Entry + SHORT Exit)

---

**Status**: ✅ **DESIGN COMPLETE**
**Expected Outcome**: 10-20% signal rate, 60-70% win rate (aligned with LONG Exit success)
**Paradigm**: BUY/SELL pairs, not LONG/SHORT pairs
