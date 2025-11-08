# New Feature Architecture - Complete Rebuild from Scratch

**Date**: 2025-10-29
**Goal**: Achieve 3-8 trades/day at threshold 0.75 with returns beating market benchmark

---

## Design Philosophy

**Forget all 171 existing features. Start fresh.**

### Core Principles
1. **Multi-Timeframe Awareness**: Capture trends across 5m, 15m, 1h, 4h
2. **Market Regime Adaptation**: Detect and adapt to different market phases
3. **Momentum Quality**: Not just direction, but strength and persistence
4. **Microstructure**: Proxy for order flow and buyer/seller pressure
5. **Dynamic Patterns**: Support/resistance that evolves with price action

### Target Performance
- **Frequency**: 3-8 trades/day at threshold 0.75 (75% confidence)
- **Returns**: Beat (market_change × 4x leverage) benchmark
- **Win Rate**: 60-75%
- **Balance**: LONG/SHORT ~50/50 split

---

## Feature Categories (5 Major Groups)

### 1. Multi-Timeframe Trend & Momentum (30 features)

**Philosophy**: Align trades with multi-timeframe trend structure

**5-Minute Base (6 features)**:
```yaml
mtf_5m_trend_strength:
  Calculate: (close - sma_50) / atr_14
  Purpose: Normalized distance from trend center

mtf_5m_momentum:
  Calculate: (close - close.shift(12)) / close.shift(12)
  Purpose: 1-hour momentum on 5m scale

mtf_5m_volume_trend:
  Calculate: (volume / volume.rolling(20).mean()) - 1
  Purpose: Volume surge detection

mtf_5m_volatility:
  Calculate: atr_14 / close
  Purpose: Current volatility as % of price

mtf_5m_price_acceleration:
  Calculate: momentum.diff(3)
  Purpose: Rate of momentum change

mtf_5m_trend_consistency:
  Calculate: (close > sma_20).rolling(12).mean()
  Purpose: % of time above trend in last hour
```

**15-Minute Aggregate (6 features)**:
```yaml
mtf_15m_trend_strength:
  Resample: 5m → 15m, calculate same as 5m

mtf_15m_momentum:
  Resample: 5m → 15m, (close - close.shift(4)) / close.shift(4)
  Purpose: 1-hour momentum on 15m scale

mtf_15m_breakout:
  Calculate: (high - high.rolling(16).max()) / high.rolling(16).max()
  Purpose: Distance from 4-hour high

mtf_15m_volume_trend:
  Resample: 5m → 15m, volume analysis

mtf_15m_volatility:
  Resample: 5m → 15m, ATR analysis

mtf_15m_trend_alignment:
  Calculate: close > sma_20 and sma_20 > sma_50
  Purpose: Multi-MA alignment
```

**1-Hour Aggregate (6 features)**:
```yaml
mtf_1h_trend_strength:
  Resample: 5m → 1h

mtf_1h_momentum:
  Calculate: (close - close.shift(4)) / close.shift(4)
  Purpose: 4-hour momentum on 1h scale

mtf_1h_trend_direction:
  Calculate: 1 if sma_10 > sma_20 else -1
  Purpose: Clear directional signal

mtf_1h_volume_profile:
  Calculate: volume.rolling(24).mean() percentile
  Purpose: Current volume vs recent history

mtf_1h_volatility_regime:
  Calculate: atr_14.rolling(24).mean() percentile
  Purpose: High/low volatility regime

mtf_1h_price_position:
  Calculate: (close - low.rolling(24).min()) / (high.rolling(24).max() - low.rolling(24).min())
  Purpose: Position in recent range (0-1)
```

**4-Hour Aggregate (6 features)**:
```yaml
mtf_4h_trend_strength:
  Resample: 5m → 4h

mtf_4h_momentum:
  Calculate: (close - close.shift(6)) / close.shift(6)
  Purpose: 24-hour momentum

mtf_4h_support_distance:
  Calculate: (close - low.rolling(48).min()) / close
  Purpose: % distance from 8-day low

mtf_4h_resistance_distance:
  Calculate: (high.rolling(48).max() - close) / close
  Purpose: % distance to 8-day high

mtf_4h_volume_strength:
  Calculate: volume.rolling(48).mean() / volume.rolling(168).mean()
  Purpose: Recent volume vs 1-week average

mtf_4h_trend_persistence:
  Calculate: (close > sma_6).rolling(12).sum() / 12
  Purpose: % of time in uptrend over 2 days
```

**Cross-Timeframe Alignment (6 features)**:
```yaml
mtf_all_trends_aligned:
  Calculate: (5m_trend > 0) & (15m_trend > 0) & (1h_trend > 0) & (4h_trend > 0)
  Purpose: Strong directional signal when all aligned

mtf_momentum_divergence:
  Calculate: 5m_momentum - 1h_momentum
  Purpose: Detect short-term divergence from main trend

mtf_volatility_expansion:
  Calculate: 5m_volatility / 4h_volatility
  Purpose: Detect volatility regime changes

mtf_volume_acceleration:
  Calculate: 5m_volume_trend - 1h_volume_trend
  Purpose: Sudden volume changes vs background

mtf_trend_reversal_signal:
  Calculate: (5m_trend < 0) & (15m_trend < 0) & (1h_trend > 0)
  Purpose: Short-term pullback in uptrend

mtf_breakout_confluence:
  Calculate: count of timeframes breaking recent highs
  Purpose: Multi-timeframe breakout strength
```

---

### 2. Market Regime Detection (25 features)

**Philosophy**: Different strategies for different market conditions

**Trend Regime (8 features)**:
```yaml
regime_trend_strength:
  Calculate: ADX (Average Directional Index)
  Interpretation: < 20 weak, 20-40 moderate, > 40 strong

regime_trend_direction:
  Calculate: +DI vs -DI from ADX
  Interpretation: +1 uptrend, -1 downtrend

regime_trend_slope:
  Calculate: slope of 50-period SMA over 20 periods
  Purpose: Rate of trend change

regime_trend_consistency:
  Calculate: % of last 50 candles above/below SMA
  Purpose: Choppy vs clean trend

regime_trend_age:
  Calculate: candles since last trend reversal
  Purpose: Mature vs young trend

regime_trend_exhaustion:
  Calculate: RSI extreme + volume decline
  Purpose: Trend losing momentum

regime_channel_position:
  Calculate: price position in Donchian Channel
  Purpose: 0-1 scale within recent range

regime_breakout_potential:
  Calculate: Bollinger Band squeeze + low ATR
  Purpose: Compression before expansion
```

**Volatility Regime (8 features)**:
```yaml
regime_volatility_level:
  Calculate: ATR percentile (0-100 scale)
  Interpretation: < 30 low, 30-70 normal, > 70 high

regime_volatility_trend:
  Calculate: ATR.diff(10)
  Purpose: Expanding or contracting volatility

regime_volatility_skew:
  Calculate: upside volatility / downside volatility
  Purpose: Directional bias in volatility

regime_garch_forecast:
  Calculate: Simple GARCH(1,1) forecast
  Purpose: Expected volatility next period

regime_realized_vs_implied:
  Calculate: Recent ATR vs historical average
  Purpose: Quiet before storm detection

regime_intraday_range:
  Calculate: (high - low) / open
  Purpose: Daily volatility proxy

regime_overnight_gap:
  Calculate: open - prev_close
  Purpose: Weekend/overnight risk

regime_volatility_regime_change:
  Calculate: 5m_volatility > 2 × 4h_volatility
  Purpose: Sudden regime shift
```

**Volume Regime (9 features)**:
```yaml
regime_volume_level:
  Calculate: volume percentile (0-100 scale)
  Interpretation: < 30 low, 30-70 normal, > 70 high

regime_volume_trend:
  Calculate: volume.rolling(20).mean().diff(10)
  Purpose: Accumulation or distribution

regime_volume_concentration:
  Calculate: top 20% candles' volume / total volume
  Purpose: Concentrated vs distributed volume

regime_volume_price_correlation:
  Calculate: correlation(volume, abs(returns), 20)
  Purpose: Price-driven vs non-price volume

regime_buying_pressure:
  Calculate: (close - low) / (high - low) weighted by volume
  Purpose: Net buying pressure proxy

regime_selling_pressure:
  Calculate: (high - close) / (high - low) weighted by volume
  Purpose: Net selling pressure proxy

regime_volume_divergence:
  Calculate: price trend vs volume trend
  Purpose: Weak hands vs strong hands

regime_large_trade_frequency:
  Calculate: count of volume > 2 × average
  Purpose: Institutional activity proxy

regime_volume_profile_shape:
  Calculate: kurtosis of volume distribution
  Purpose: Normal vs heavy-tailed activity
```

---

### 3. Momentum Quality & Persistence (20 features)

**Philosophy**: Not just momentum direction, but quality and sustainability

**Momentum Strength (7 features)**:
```yaml
momentum_roc_1h:
  Calculate: (close - close.shift(12)) / close.shift(12)
  Purpose: 1-hour rate of change

momentum_roc_4h:
  Calculate: (close - close.shift(48)) / close.shift(48)
  Purpose: 4-hour rate of change

momentum_roc_24h:
  Calculate: (close - close.shift(288)) / close.shift(288)
  Purpose: 24-hour rate of change

momentum_acceleration:
  Calculate: momentum_roc_1h.diff(6)
  Purpose: Rate of momentum change

momentum_magnitude:
  Calculate: abs(returns).rolling(20).mean()
  Purpose: Average price change regardless of direction

momentum_asymmetry:
  Calculate: positive_returns.mean() / negative_returns.mean()
  Purpose: Up vs down move strength

momentum_efficiency:
  Calculate: net_change / sum(abs(changes))
  Purpose: Direct vs choppy movement
```

**Momentum Persistence (7 features)**:
```yaml
momentum_consecutive_up:
  Calculate: count of consecutive positive returns
  Purpose: Strength of current move

momentum_consecutive_down:
  Calculate: count of consecutive negative returns
  Purpose: Weakness of current move

momentum_persistence_score:
  Calculate: correlation(returns[i], returns[i-1], 20)
  Purpose: Trending vs mean-reverting

momentum_trend_days:
  Calculate: days since last trend reversal
  Purpose: Age of current momentum

momentum_higher_highs:
  Calculate: count of higher highs in last 20 candles
  Purpose: Uptrend structure

momentum_lower_lows:
  Calculate: count of lower lows in last 20 candles
  Purpose: Downtrend structure

momentum_zigzag_pattern:
  Calculate: detect higher-high/higher-low pattern
  Purpose: Clean uptrend vs choppy
```

**Momentum Exhaustion (6 features)**:
```yaml
momentum_rsi_extreme:
  Calculate: RSI > 70 or RSI < 30
  Purpose: Overbought/oversold detection

momentum_rsi_divergence:
  Calculate: price new high but RSI not new high
  Purpose: Momentum weakening

momentum_volume_divergence:
  Calculate: price new high but volume declining
  Purpose: Weak follow-through

momentum_parabolic_extension:
  Calculate: distance from parabolic SAR
  Purpose: Overextension detection

momentum_climax_volume:
  Calculate: volume > 3 × average + price reversal
  Purpose: Capitulation or blow-off top

momentum_exhaustion_score:
  Calculate: composite of above signals
  Purpose: Overall exhaustion measure
```

---

### 4. Microstructure & Order Flow Proxies (20 features)

**Philosophy**: Infer institutional activity and order flow from OHLCV data

**Price Action Microstructure (8 features)**:
```yaml
micro_body_wick_ratio:
  Calculate: abs(close - open) / (high - low)
  Purpose: Conviction vs indecision

micro_upper_wick_ratio:
  Calculate: (high - max(open, close)) / (high - low)
  Purpose: Rejection at highs

micro_lower_wick_ratio:
  Calculate: (min(open, close) - low) / (high - low)
  Purpose: Support at lows

micro_close_position:
  Calculate: (close - low) / (high - low)
  Purpose: Where close ended in range (0-1)

micro_open_gap:
  Calculate: (open - prev_close) / prev_close
  Purpose: Overnight positioning

micro_intracandle_momentum:
  Calculate: (close - open) / open
  Purpose: Within-period strength

micro_price_rejection:
  Calculate: high == close or low == close
  Purpose: Failed breakout/breakdown

micro_doji_pattern:
  Calculate: abs(close - open) < 0.1 × (high - low)
  Purpose: Indecision signal
```

**Volume Microstructure (8 features)**:
```yaml
micro_volume_at_price:
  Calculate: volume when close > open vs close < open
  Purpose: Buying vs selling volume proxy

micro_trade_size:
  Calculate: volume / num_ticks (proxy using range)
  Purpose: Average trade size estimate

micro_large_trade_indicator:
  Calculate: volume > 2 × average and small range
  Purpose: Large orders with minimal price impact

micro_volume_imbalance:
  Calculate: (uptick_volume - downtick_volume) / total
  Purpose: Net buying/selling pressure

micro_volume_concentration:
  Calculate: volume in upper half vs lower half of range
  Purpose: Where volume occurred in candle

micro_aggressive_buying:
  Calculate: volume when price closes near high
  Purpose: Aggressive buyers lifting offers

micro_aggressive_selling:
  Calculate: volume when price closes near low
  Purpose: Aggressive sellers hitting bids

micro_passive_volume:
  Calculate: volume when price closes mid-range
  Purpose: Range-bound trading
```

**Order Flow Proxies (4 features)**:
```yaml
micro_delta:
  Calculate: aggressive_buying - aggressive_selling
  Purpose: Net order flow direction

micro_cumulative_delta:
  Calculate: delta.rolling(20).sum()
  Purpose: Sustained buying/selling pressure

micro_delta_divergence:
  Calculate: price up but cumulative_delta down
  Purpose: Weak buying despite price rise

micro_vpin:
  Calculate: Volume-synchronized Probability of Informed Trading
  Formula: abs(buy_volume - sell_volume) / total_volume
  Purpose: Informed trader activity proxy
```

---

### 5. Dynamic Pattern Recognition (20 features)

**Philosophy**: Patterns that adapt to current market structure

**Dynamic Support/Resistance (8 features)**:
```yaml
pattern_recent_high:
  Calculate: max(high, 20 periods)
  Purpose: Near-term resistance

pattern_recent_low:
  Calculate: min(low, 20 periods)
  Purpose: Near-term support

pattern_distance_to_resistance:
  Calculate: (recent_high - close) / close
  Purpose: % to nearest resistance

pattern_distance_to_support:
  Calculate: (close - recent_low) / close
  Purpose: % to nearest support

pattern_swing_high_count:
  Calculate: count of swing highs in last 50 candles
  Purpose: Resistance density

pattern_swing_low_count:
  Calculate: count of swing lows in last 50 candles
  Purpose: Support density

pattern_pivot_strength:
  Calculate: volume at pivot points
  Purpose: Strength of S/R levels

pattern_range_position:
  Calculate: (close - recent_low) / (recent_high - recent_low)
  Purpose: Position in recent range (0-1)
```

**Breakout Detection (6 features)**:
```yaml
pattern_breakout_above:
  Calculate: close > recent_high
  Purpose: Upside breakout signal

pattern_breakout_below:
  Calculate: close < recent_low
  Purpose: Downside breakout signal

pattern_breakout_volume:
  Calculate: volume on breakout vs average
  Purpose: Breakout conviction

pattern_false_breakout:
  Calculate: breakout followed by reversal within 3 candles
  Purpose: Failed breakout detection

pattern_consolidation_duration:
  Calculate: candles in tight range before breakout
  Purpose: Compression duration

pattern_breakout_strength:
  Calculate: distance beyond breakout level
  Purpose: Magnitude of breakout
```

**Reversal Patterns (6 features)**:
```yaml
pattern_double_top:
  Calculate: detect two similar highs with valley
  Purpose: Reversal pattern

pattern_double_bottom:
  Calculate: detect two similar lows with peak
  Purpose: Reversal pattern

pattern_head_shoulders:
  Calculate: detect H&S pattern
  Purpose: Trend reversal

pattern_v_shaped_reversal:
  Calculate: sharp reversal with high volume
  Purpose: Capitulation signal

pattern_gradual_reversal:
  Calculate: slope change over 10+ candles
  Purpose: Trend exhaustion

pattern_reversal_confirmation:
  Calculate: reversal signal + volume + momentum
  Purpose: High-probability reversal
```

---

## Feature Engineering Implementation Plan

### Phase 1: Multi-Timeframe Aggregation (4 hours)
```python
# scripts/features/calculate_multitimeframe_features.py

def aggregate_timeframe(df_5m, timeframe='15min'):
    """Resample 5m data to higher timeframes"""
    df_resampled = df_5m.resample(timeframe, on='timestamp').agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum'
    })

    # Calculate indicators on resampled data
    df_resampled['sma_20'] = df_resampled['close'].rolling(20).mean()
    df_resampled['atr_14'] = calculate_atr(df_resampled, 14)
    # ... more indicators

    # Merge back to 5m (forward fill to align timestamps)
    df_5m_with_mtf = df_5m.merge(
        df_resampled.add_prefix(f'{timeframe}_'),
        left_on='timestamp',
        right_index=True,
        how='left'
    ).ffill()

    return df_5m_with_mtf
```

### Phase 2: Market Regime Detection (3 hours)
```python
# scripts/features/calculate_regime_features.py

def calculate_trend_regime(df):
    """ADX-based trend regime"""
    df['plus_di'] = calculate_plus_di(df)
    df['minus_di'] = calculate_minus_di(df)
    df['adx'] = calculate_adx(df)

    df['regime_trend_strength'] = pd.cut(
        df['adx'],
        bins=[0, 20, 40, 100],
        labels=['weak', 'moderate', 'strong']
    )

    df['regime_trend_direction'] = np.where(
        df['plus_di'] > df['minus_di'], 1, -1
    )

    return df

def calculate_volatility_regime(df):
    """ATR percentile-based volatility regime"""
    df['atr_percentile'] = df['atr'].rolling(100).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1]
    )

    df['regime_volatility_level'] = pd.cut(
        df['atr_percentile'],
        bins=[0, 0.3, 0.7, 1.0],
        labels=['low', 'normal', 'high']
    )

    return df
```

### Phase 3: Momentum Quality (2 hours)
```python
# scripts/features/calculate_momentum_features.py

def calculate_momentum_persistence(df):
    """Measure momentum quality and persistence"""

    # Rate of change across multiple horizons
    df['momentum_roc_1h'] = df['close'].pct_change(12)
    df['momentum_roc_4h'] = df['close'].pct_change(48)
    df['momentum_roc_24h'] = df['close'].pct_change(288)

    # Momentum acceleration (second derivative)
    df['momentum_acceleration'] = df['momentum_roc_1h'].diff(6)

    # Consecutive moves
    df['momentum_consecutive_up'] = (
        df['close'].diff() > 0
    ).rolling(20).apply(
        lambda x: max(sum(1 for _ in takewhile(lambda y: y, x)), 0)
    )

    return df
```

### Phase 4: Microstructure (3 hours)
```python
# scripts/features/calculate_microstructure_features.py

def calculate_order_flow_proxies(df):
    """Proxy order flow from OHLCV"""

    # Body and wick ratios
    df['body'] = abs(df['close'] - df['open'])
    df['range'] = df['high'] - df['low']
    df['micro_body_wick_ratio'] = df['body'] / df['range'].replace(0, np.nan)

    # Close position in candle
    df['micro_close_position'] = (
        (df['close'] - df['low']) / df['range'].replace(0, np.nan)
    )

    # Volume attribution (buying vs selling)
    df['micro_aggressive_buying'] = np.where(
        df['micro_close_position'] > 0.7,
        df['volume'],
        0
    )
    df['micro_aggressive_selling'] = np.where(
        df['micro_close_position'] < 0.3,
        df['volume'],
        0
    )

    # Delta (net order flow)
    df['micro_delta'] = (
        df['micro_aggressive_buying'] - df['micro_aggressive_selling']
    )
    df['micro_cumulative_delta'] = df['micro_delta'].rolling(20).sum()

    return df
```

### Phase 5: Dynamic Patterns (2 hours)
```python
# scripts/features/calculate_pattern_features.py

def calculate_dynamic_support_resistance(df):
    """Calculate adaptive S/R levels"""

    # Recent swing highs/lows
    df['pattern_recent_high'] = df['high'].rolling(20).max()
    df['pattern_recent_low'] = df['low'].rolling(20).min()

    # Distance to S/R
    df['pattern_distance_to_resistance'] = (
        (df['pattern_recent_high'] - df['close']) / df['close']
    )
    df['pattern_distance_to_support'] = (
        (df['close'] - df['pattern_recent_low']) / df['close']
    )

    # Breakout detection
    df['pattern_breakout_above'] = (
        df['close'] > df['pattern_recent_high'].shift(1)
    ).astype(float)
    df['pattern_breakout_below'] = (
        df['close'] < df['pattern_recent_low'].shift(1)
    ).astype(float)

    return df
```

---

## Expected Impact

### Hypothesis: New Features → Threshold 0.75 Achievable

**Current Problem**:
- Calibrated models: max prob 16.91% (LONG), 32.43% (SHORT)
- Cannot reach threshold 0.75

**Expected Solution**:
- Rich multi-timeframe context → Capture high-quality setups
- Regime-adaptive signals → Avoid bad market conditions
- Momentum quality filters → Only strong persistent moves
- Microstructure confirmation → Institutional support
- Dynamic patterns → Entry at optimal levels

**Target Outcome**:
- Model probabilities naturally reach 75%+ for best setups
- 3-8 trades/day with 75% threshold (no artificial adjustment)
- Returns beat (market_change × 4x leverage) consistently
- Win rate 60-75% with balanced LONG/SHORT

---

## Implementation Timeline

**Total Estimate: 14 hours**

| Phase | Task | Hours | Priority |
|-------|------|-------|----------|
| 1 | Multi-Timeframe Aggregation | 4 | HIGH |
| 2 | Market Regime Detection | 3 | HIGH |
| 3 | Momentum Quality | 2 | MEDIUM |
| 4 | Microstructure | 3 | MEDIUM |
| 5 | Dynamic Patterns | 2 | LOW |
| 6 | Generate Full Dataset | 1 | HIGH |
| 7 | Retrain Entry Models | 2 | HIGH |
| 8 | Backtest & Validate | 1 | HIGH |

**Recommended Approach**: Implement in priority order, test incrementally.

---

## Success Metrics

### Model Performance (Threshold 0.75)
- [ ] Trade Frequency: 3-8 trades/day ✅
- [ ] Returns: Beat (market_change × 4x) ✅
- [ ] Win Rate: 60-75% ✅
- [ ] LONG/SHORT: 40-60% each ✅
- [ ] Sharpe Ratio: > 2.0 ✅

### Probability Distribution
- [ ] Max probabilities reach 75%+ regularly
- [ ] Smooth distribution (no discrete clusters)
- [ ] LONG and SHORT both reach 75%+ (balance)

### Feature Quality
- [ ] No multicollinearity (VIF < 10)
- [ ] All features have non-zero importance
- [ ] Feature distributions stable across time

---

**Created**: 2025-10-29
**Status**: DESIGN COMPLETE - READY FOR IMPLEMENTATION
**Next Step**: Implement Phase 1 (Multi-Timeframe Aggregation)
