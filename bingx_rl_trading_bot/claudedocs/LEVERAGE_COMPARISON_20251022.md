# Leverage Comparison: 10x @ 40% vs 4x @ 100%

**Date**: 2025-10-22 04:45 KST
**Analysis**: Capital exposure equivalence and practical differences

---

## Mathematical Equivalence

```yaml
Capital Exposure (Same):
  10x leverage × 40% position = 4.0x capital exposure
  4x leverage × 100% position = 4.0x capital exposure

Example ($584 capital):
  10x @ 40%: $234 margin → $2,340 exposure (10x) → $936 effective (4x)
  4x @ 100%: $584 margin → $2,336 exposure (4x) → $2,336 effective (4x)
```

**Mathematical Answer**: YES, they have the same capital exposure (4x).

---

## Practical Differences

### 1. Margin Reserve (Capital Flexibility)

```yaml
10x @ 40%:
  Used Capital: 40% ($234)
  Reserve Capital: 60% ($350)
  Flexibility: Can enter additional trades
  Buffer: Has cushion for volatility

4x @ 100%:
  Used Capital: 100% ($584)
  Reserve Capital: 0%
  Flexibility: Cannot enter additional trades
  Buffer: No cushion, fully committed
```

**Winner**: 10x @ 40% (flexibility and buffer)

---

### 2. Liquidation Risk

```yaml
10x Leverage:
  Liquidation Threshold: ~-10% unrealized loss
  Price Movement: Tight tolerance
  Risk: High liquidation risk

4x Leverage:
  Liquidation Threshold: ~-25% unrealized loss
  Price Movement: Wide tolerance
  Risk: Lower liquidation risk
```

**Winner**: 4x @ 100% (safer liquidation threshold)

---

### 3. Stop Loss Behavior

```yaml
Stop Loss: -3% of total balance ($584 × -3% = -$17.52)

10x @ 40% Position:
  Position Size: $234
  Leveraged Exposure: $2,340 (10x)
  SL Trigger: -$17.52 / $2,340 = -0.748% price movement

4x @ 100% Position:
  Position Size: $584
  Leveraged Exposure: $2,336 (4x)
  SL Trigger: -$17.52 / $2,336 = -0.750% price movement
```

**Key Insight**: Balance-Based SL → Same capital exposure → Same SL trigger (~0.75%)

**Winner**: TIE (virtually identical SL behavior)

---

### 4. Real-World Example

```yaml
Initial Capital: $584
BTC Price: $107,000

Scenario 1: 10x @ 40%
  Margin: $234
  Leveraged Position: $2,340 (10x)
  Effective Exposure: $936 (4x equivalent)
  BTC Amount: 0.0087 BTC
  Reserve: $350 (60%)

Scenario 2: 4x @ 100%
  Margin: $584
  Leveraged Position: $2,336 (4x)
  Effective Exposure: $2,336 (4x)
  BTC Amount: 0.0218 BTC
  Reserve: $0 (0%)

Price Drops -5% (to $101,650):
  10x @ 40%: -$46.80 (-8.0%), Remaining: $537.20
  4x @ 100%: -$116.80 (-20.0%), Remaining: $467.20

Price Drops -10% (to $96,300):
  10x @ 40%: -$93.60 (-16.0%), Remaining: $490.40
  4x @ 100%: -$233.60 (-40.0%), Remaining: $350.40
```

**Winner**: 10x @ 40% (smaller absolute losses)

---

### 5. Backtest Performance (30 Days)

```yaml
4x Leverage (Dynamic Sizing, Avg 49.3% Position):
  Return: +75.58%
  Max Drawdown: -12.2%
  Sharpe Ratio: 0.336
  Win Rate: 63.6%
  Trades: 55
  Risk: BEST

10x v2 (Conservative Sizing, Avg 40% Position):
  Return: +67.49%
  Max Drawdown: -16.48%
  Sharpe Ratio: 0.270
  Win Rate: 60.7%
  Trades: 57
  Risk: GOOD

10x v1 (Simple Sizing, 20-30% Range):
  Return: +64.36%
  Max Drawdown: -21.32%
  Sharpe Ratio: 0.229
  Win Rate: 56.5%
  Trades: 58
  Risk: ACCEPTABLE

10x (50% Base Position):
  Return: +90.25%
  Max Drawdown: -35.72%
  Sharpe Ratio: 0.187
  Win Rate: 50.0%
  Trades: 60
  Risk: TOO HIGH
```

**Winner**: 4x leverage (best risk-adjusted returns)

---

## Comparison Matrix

| Metric | 10x @ 40% | 4x @ 100% | Winner |
|--------|-----------|-----------|--------|
| Capital Exposure | 4x | 4x | TIE (Same) |
| Margin Reserve | 60% | 0% | 10x (Flexibility) |
| Liquidation Risk | -10% | -25% | 4x (Safer) |
| SL Trigger Speed | -0.748% | -0.750% | TIE (Same) |
| Loss on -5% Drop | -$46.80 | -$116.80 | 10x (Smaller) |
| Loss on -10% Drop | -$93.60 | -$233.60 | 10x (Smaller) |
| 30d Return | +67.49% | +75.58% | 4x (Higher) |
| Max Drawdown | -16.48% | -12.2% | 4x (Lower) |
| Sharpe Ratio | 0.270 | 0.336 | 4x (Better) |
| Win Rate | 60.7% | 63.6% | 4x (Higher) |

---

## Conclusions

### Mathematical Answer
**YES**, 10x @ 40% and 4x @ 100% have the **same capital exposure** (4x).

### Practical Answer
**NO**, they behave very differently in real trading:

**10x @ 40% Advantages**:
- ✅ 60% capital reserve (flexibility)
- ✅ Smaller absolute losses on adverse moves
- ✅ Can add positions if opportunities arise
- ❌ Higher liquidation risk (-10% threshold)
- ❌ Lower historical performance

**4x @ 100% Advantages**:
- ✅ Lower liquidation risk (-25% threshold)
- ✅ Better backtest performance (+75.58%, -12.2% DD)
- ✅ Higher Sharpe ratio (0.336 vs 0.270)
- ✅ Higher win rate (63.6% vs 60.7%)
- ❌ No capital reserve (0% flexibility)
- ❌ Larger absolute losses on adverse moves

**Note**: SL trigger speed is virtually identical (~0.75%) due to Balance-Based SL

---

## Recommendation

**Based on 30-day backtest results:**

**🥇 4x Leverage (Dynamic Sizing) - BEST CHOICE**
```yaml
Why:
  - Highest Sharpe Ratio (0.336) → Best risk-adjusted returns
  - Lowest Max Drawdown (-12.2%) → Safest
  - Highest Win Rate (63.6%) → Most consistent
  - Highest Return (+75.58%) → Most profitable
  - Dynamic sizing (20-95%) → Adaptive risk management

Tradeoff:
  - Uses more capital per trade
  - Less flexibility for multiple positions
  - But: Current strategy is single-position anyway
```

**🥈 10x v2 (Conservative Sizing) - GOOD ALTERNATIVE**
```yaml
Why:
  - Good returns (+67.49%)
  - Acceptable drawdown (-16.48%)
  - 60% capital reserve
  - Advanced risk management features

When to Use:
  - Want capital flexibility for multiple strategies
  - Plan to scale up capital (reserve matters more)
  - Need buffer for manual trading
  - Prioritize flexibility over max returns
```

**⚠️ Not Recommended: 10x @ 50% Base**
```yaml
Why:
  - Too aggressive (-35.72% drawdown)
  - High stop loss rate (27.2%)
  - Low Sharpe ratio (0.187)
  - Despite +90% returns, risk too high
```

---

## Implementation Note

**Current Production Bot**:
```python
# opportunity_gating_bot_4x.py
LEVERAGE = 4
POSITION_SIZE_PCT = Dynamic (20-95%)  # via DynamicPositionSizer

# Optimized 2025-10-22
EMERGENCY_STOP_LOSS = 0.03  # -3% balance
EMERGENCY_MAX_HOLD_TIME = 120  # 10 hours
ML_EXIT_THRESHOLD_LONG = 0.75
ML_EXIT_THRESHOLD_SHORT = 0.75
```

**If Switching to 10x**:
```python
# Would need to modify:
LEVERAGE = 10
POSITION_SIZE_PCT = Use DynamicPositionSizer10xV2
  # Conservative: base 20%, max 30%, min 10%
  # Advanced features: drawdown scaling, Kelly, frequency throttling
```

---

## Key Takeaway

> "10x @ 40% and 4x @ 100% are mathematically equivalent in capital exposure,
> but 4x @ 100% is safer (liquidation) and performs better (backtest results)."

**Current Setup (4x + Dynamic Sizing) is optimal based on evidence.**

---

**Documentation**: `/claudedocs/LEVERAGE_COMPARISON_20251022.md`
**Related**: `BALANCE_BASED_SL_DEPLOYMENT_20251021.md`, `EXIT_PARAMETER_OPTIMIZATION_20251022.md`
