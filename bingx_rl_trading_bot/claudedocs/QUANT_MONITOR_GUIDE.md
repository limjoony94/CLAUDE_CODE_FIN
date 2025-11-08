# Professional Quantitative Trading Monitor - User Guide
**Updated**: 2025-10-17
**Version**: 2.0 (Opportunity Gating System)
**Level**: Institutional-Grade Monitoring

---

## Overview

Professional quantitative trading monitor with institutional-grade metrics and real-time analytics for the **Opportunity Gating + 4x Leverage** system.

### System Information

**Current Strategy**: Opportunity Gating
- **Innovation**: SHORT only when EV(SHORT) > EV(LONG) + gate (0.001)
- **Leverage**: 4x
- **Position Sizing**: Dynamic (20-95%)
- **Capital Protection**: Gate prevents low-quality SHORT trades

### Key Features

1. **Risk-Adjusted Performance Metrics**
   - Sharpe Ratio (annualized)
   - Sortino Ratio (downside risk focus)
   - Calmar Ratio (return/drawdown)
   - Maximum Drawdown tracking
   - **NEW**: Gate effectiveness tracking

2. **Real-Time Risk Analytics**
   - Value at Risk (VaR 95%)
   - Conditional VaR (CVaR 95%)
   - Current drawdown monitoring
   - Position exposure tracking
   - Volatility analysis
   - **NEW**: Leverage multiplier tracking

3. **Signal Quality & Model Diagnostics**
   - LONG signal rate vs expected
   - SHORT signal rate vs expected
   - **NEW**: SHORT gate block rate
   - **NEW**: Opportunity cost analysis
   - Model confidence tracking
   - Signal accuracy over time

4. **Execution Quality**
   - Average holding time
   - Win rate analysis (LONG vs SHORT)
   - Profit factor tracking
   - Win/loss distribution
   - **NEW**: LONG/SHORT trade distribution

5. **Market Regime Analysis**
   - Regime detection (Bullish/Bearish/Sideways)
   - Volatility regime
   - Price action sparklines
   - Trend strength

6. **Alert System**
   - High drawdown alerts
   - Low Sharpe ratio warnings
   - Position risk notifications
   - Win rate degradation alerts
   - **NEW**: Gate malfunction alerts
   - **NEW**: LONG/SHORT distribution alerts

7. **ASCII Visualization**
   - Drawdown progress bars
   - Exposure level bars
   - Signal history sparklines
   - Price action sparklines

---

## Quick Start

### Windows

```bash
# Simply double-click:
QUANT_MONITOR.bat

# Or from command line:
cd C:\Users\J\OneDrive\CLAUDE_CODE_FIN\bingx_rl_trading_bot
QUANT_MONITOR.bat
```

### Linux/Mac

```bash
cd /path/to/bingx_rl_trading_bot
python scripts/monitoring/quant_monitor.py
```

---

## Requirements

### Python Packages

```bash
# Core requirements (should already be installed)
pip install numpy

# Optional for enhanced features
pip install scipy  # For advanced statistical tests
```

### System Requirements

- Python 3.7+
- NumPy
- 2MB RAM for monitoring process
- Terminal with ANSI color support (recommended)

---

## Display Sections

### 1. Performance Metrics Section

```
┌─ PERFORMANCE METRICS ─────────────────────────────────────────────────────┐
│ Total Return       :              +18.5%  │  Trades:   28  │  Win Rate:  64.3%  │
│ Sharpe Ratio       :                 2.45  │  Sortino:   3.12  │  Calmar:   6.22  │
│ Max Drawdown       :                 2.97% │  Current DD:  1.2%  │             │
│ DD Progress        : [████░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░] 1.2%  │
│ Leverage           :                   4x  │  Avg Position:  51.4%  │             │
└───────────────────────────────────────────────────────────────────────────┘
```

**Metrics Explained**:

- **Total Return**: Overall portfolio return (leveraged, 4x)
- **Sharpe Ratio**: Risk-adjusted return (>2.0 excellent, >1.0 good)
- **Sortino Ratio**: Similar to Sharpe but only penalizes downside volatility
- **Calmar Ratio**: Return divided by maximum drawdown (>3.0 excellent)
- **Max Drawdown**: Largest peak-to-trough decline
- **Current DD**: Current distance from equity peak
- **Leverage**: 4x multiplier on all positions
- **Avg Position**: Average position size (target: 51.4%)

**Expected Performance (from 105-day backtest)**:
- Return: 18.13% per 5-day window
- Win Rate: 63.9%
- Trades: 18.5 per window (~3.7/day, ~26/week)

**Color Coding**:
- 🟢 Green: Excellent (Sharpe >2.0, DD <5%, Win Rate >60%)
- 🟡 Yellow: Good (Sharpe >1.0, DD <10%, Win Rate >55%)
- 🔴 Red: Poor (Sharpe <1.0, DD >10%, Win Rate <55%)

---

### 2. Trading Statistics Section

```
┌─ TRADING STATISTICS ──────────────────────────────────────────────────────┐
│ Win/Loss           :   18W /   10L  │  Profit Factor:   2.85  │                │
│ Avg Win/Loss       :  $285.50 / $125.30  │  Avg Hold:   2.5h  │                │
│ Largest Win/Loss   :  $890.20 / $245.60  │                                       │
│ LONG Trades        :   24 (85.7%)  │  SHORT Trades:    4 (14.3%)  │            │
│ LONG Win Rate      :   66.7%       │  SHORT Win Rate:  50.0%  │                │
└───────────────────────────────────────────────────────────────────────────┘
```

**Metrics Explained**:

- **Profit Factor**: Total profits / Total losses (>2.0 excellent, >1.5 good)
- **Avg Win/Loss**: Average profit on winners vs average loss on losers
- **Largest Win/Loss**: Biggest single trade profit and loss (leveraged)
- **Avg Hold**: Average position holding time (target: 2-4 hours)
- **LONG/SHORT Distribution**: Should be ~85%/15% (from backtest)

**Expected Distribution**:
- LONG: 85% of trades (15.7 per window)
- SHORT: 15% of trades (2.8 per window)

---

### 3. Risk Analytics Section

```
┌─ RISK ANALYTICS ──────────────────────────────────────────────────────────┐
│ VaR (95%)          :                 1.5%  │  CVaR (95%):   2.2%  │                │
│ Position Exposure  :                52.0%  │  Volatility:   38.5%  │                │
│ Exposure Level     : [██████████████████████████░░░░░░░░░░░░░░░░░░░░░░░░] 52.0%  │
│ Leverage Multiplier:                   4x  │  Effective Exp: 208%  │                │
└───────────────────────────────────────────────────────────────────────────┘
```

**Metrics Explained**:

- **VaR (95%)**: Maximum expected loss over next trade with 95% confidence
- **CVaR (95%)**: Expected loss when loss exceeds VaR (tail risk)
- **Position Exposure**: Current position value as % of balance
- **Leverage Multiplier**: 4x on all positions
- **Effective Exposure**: Position × Leverage (can exceed 100%)
- **Volatility**: Annualized price volatility

**Thresholds**:
- VaR: <2% excellent, <3% acceptable (leveraged system)
- Position Exposure: 20-95% (dynamic sizing)
- Effective Exposure: <380% (4x × 95% max position)
- CVaR: Should be <2x VaR

---

### 4. Signal Quality & Opportunity Gating Section

```
┌─ SIGNAL QUALITY & OPPORTUNITY GATING ─────────────────────────────────────┐
│ Signal History     : ▂▃▅▄▃▂▁▂▃▅▆▇█▇▆▅▃▂▁▂▃▄▅▆▇▆▅▄▃▂▁▂▃▄▅▆▅▄▃▂  │
│ Current Position   :   LONG  │  Entry Prob: 0.687  │  Regime:   Sideways  │
│ LONG Signal Rate   :  8.2%   │  Expected: 8.5%     │  Deviation: -3.5%    │
│ SHORT Signal Rate  :  2.1%   │  Expected: 2.0%     │  Deviation: +5.0%    │
│ Gate Effectiveness :          │  Blocks:  15 (75%)  │  Allows:   5 (25%)   │
└───────────────────────────────────────────────────────────────────────────┘
```

**Metrics Explained**:

- **Signal History**: Sparkline of recent signal probabilities
- **Current Position**: Active position details
- **Entry Prob**: Entry signal probability (threshold-adjusted)
- **LONG Signal Rate**: LONG signals above 0.65 threshold
- **SHORT Signal Rate**: SHORT signals above 0.70 threshold
- **Gate Blocks**: SHORT signals blocked by opportunity cost gate
- **Gate Allows**: SHORT signals that passed gate check

**Opportunity Gating Logic**:
```python
if short_prob >= 0.70:
    long_ev = long_prob × 0.0041
    short_ev = short_prob × 0.0047
    opportunity_cost = short_ev - long_ev

    if opportunity_cost > 0.001:  # Gate threshold
        Enter SHORT  # Passed gate
    else:
        Block SHORT  # Failed gate (wait for LONG)
```

**Expected Behavior**:
- Gate should block 70-80% of SHORT signals
- Final SHORT trades: ~15% of total trades
- Gate prevents capital lock from low-quality SHORTs

---

### 5. Market Regime Section

```
┌─ MARKET REGIME ANALYSIS ──────────────────────────────────────────────────┐
│ Price Action       : ▃▄▅▆▇█▇▆▅▄▃▂▁▂▃▄▅▆▇▆▅▄▃▂▁▂▃▄▅▆▅▄▃▂  $108,432.90  │
│ Market Regime      :   Sideways  │  Volatility:  38.5%  │                │
│ Trend Strength     :   Medium    │  Strategy Fit:  Good  │                │
└───────────────────────────────────────────────────────────────────────────┘
```

**Regimes**:
- **Bullish**: Upward trending (LONG favored, SHORT gate stricter)
- **Bearish**: Downward trending (SHORT may increase if gate allows)
- **Sideways**: Range-bound (balanced LONG/SHORT, optimal for strategy)

---

### 6. Alerts Section

```
┌─ ⚠️  ALERTS ──────────────────────────────────────────────────────────────┐
│ ⚠️  SHORT DISTRIBUTION: 5.2% (expected: 15%, threshold: 10-20%)            │
│ ⚠️  GATE BLOCKS: 95% (expected: 70-80%, possible gate malfunction)         │
│ 🟢 WIN RATE: 64.3% (target: >60%)                                          │
│ 🟢 SHARPE: 2.45 (target: >2.0)                                             │
└───────────────────────────────────────────────────────────────────────────┘
```

**Alert Thresholds**:

| Alert Type | Threshold | Action Required |
|------------|-----------|-----------------|
| 🚨 High Drawdown | >10% (leveraged) | Reduce position size |
| ⚠️ Low Sharpe | <1.5 | Review strategy parameters |
| ⚠️ High Exposure | >85% | Reduce position size |
| ⚠️ Low Win Rate | <55% | Investigate signal quality |
| ⚠️ SHORT Distribution | <10% or >25% | Check gate effectiveness |
| ⚠️ Gate Malfunction | <60% or >90% blocks | Review gate logic |
| ⚠️ Leverage Risk | Effective exp >350% | Critical: reduce immediately |

---

## Configuration

Edit `scripts/monitoring/quant_monitor.py`:

```python
# Monitoring parameters
REFRESH_INTERVAL = 30  # seconds (default: 30)
MAX_HISTORY = 1000     # data points to keep (default: 1000)
RISK_FREE_RATE = 0.04  # 4% annual (for Sharpe calculation)

# Alert thresholds (adjusted for Opportunity Gating)
ALERT_MAX_DRAWDOWN = 0.10           # 10% (leveraged system)
ALERT_MIN_SHARPE = 1.5              # Minimum acceptable Sharpe
ALERT_POSITION_RISK = 0.85          # 85% of balance
ALERT_MIN_WIN_RATE = 0.55           # 55%
ALERT_SHORT_DIST_MIN = 0.10         # 10% (expected 15%)
ALERT_SHORT_DIST_MAX = 0.25         # 25%
ALERT_GATE_BLOCK_MIN = 0.60         # 60% (expected 70-80%)
ALERT_GATE_BLOCK_MAX = 0.90         # 90%

# State file path (UPDATED for Opportunity Gating)
STATE_FILE = "results/opportunity_gating_bot_4x_state.json"
```

---

## Interpretation Guide

### Sharpe Ratio (Leveraged System)

| Range | Rating | Interpretation |
|-------|--------|----------------|
| >3.0 | Exceptional | Outstanding risk-adjusted returns |
| 2.0-3.0 | Excellent | Very good performance (target) |
| 1.5-2.0 | Good | Acceptable performance |
| 1.0-1.5 | Subpar | Below average |
| <1.0 | Poor | Consider strategy review |

**Note**: 4x leverage increases both returns and risk. Sharpe >2.0 is excellent for leveraged system.

### Win Rate (Opportunity Gating)

| Range | Rating | Interpretation |
|-------|--------|----------------|
| >70% | Exceptional | Excellent signal quality |
| 60-70% | Excellent | Target range (backtest: 63.9%) |
| 55-60% | Good | Acceptable performance |
| 50-55% | Subpar | Review needed |
| <50% | Poor | Strategy not working |

### Gate Effectiveness

| Blocks | Rating | Interpretation |
|--------|--------|----------------|
| 70-80% | Optimal | Gate working as designed |
| 60-70% | Good | Slightly permissive |
| 80-90% | Good | Slightly restrictive |
| <60% | Alert | Too permissive (investigate) |
| >90% | Alert | Too restrictive (investigate) |

### SHORT Trade Distribution

| Range | Rating | Interpretation |
|-------|--------|----------------|
| 10-20% | Optimal | Expected range (backtest: 15%) |
| 5-10% | Suboptimal | Gate too restrictive |
| 20-30% | Suboptimal | Gate too permissive |
| <5% | Alert | Gate malfunction likely |
| >30% | Alert | Gate not working |

---

## Opportunity Gating System Monitoring

### Key Metrics to Watch

**1. LONG/SHORT Distribution**:
- Target: 85% LONG, 15% SHORT
- Alert if: SHORT <10% or >25%
- Indicates: Gate effectiveness

**2. Gate Block Rate**:
- Target: 70-80% of SHORT signals blocked
- Alert if: <60% or >90%
- Indicates: Gate logic functioning

**3. Win Rate by Direction**:
- LONG: Should be ~64-65%
- SHORT: Should be ~63-65%
- Similar win rates validate strategy

**4. Opportunity Cost Average**:
- Track: Average opportunity_cost for allowed SHORTs
- Should be: >0.001 (gate threshold)
- Monitor: Trend over time

### Troubleshooting Gate Issues

**If SHORT trades <10%**:
- Check: Gate threshold too high?
- Review: Recent opportunity cost values
- Action: Consider lowering gate to 0.0005

**If SHORT trades >25%**:
- Check: Gate threshold too low?
- Review: LONG signal quality
- Action: Consider raising gate to 0.0015

**If gate blocks >90%**:
- Check: LONG signals strong relative to SHORT
- Review: Market regime (might be normal in strong trends)
- Action: Monitor, adjust if persists >48h

---

## Troubleshooting

### Monitor won't start

**Check**:
1. Trading bot is running
2. State file exists: `results/opportunity_gating_bot_4x_state.json`
3. Log files exist in `logs/`
4. Python and NumPy installed

### No data showing

**Check**:
1. At least 1 completed trade exists
2. State file has valid JSON
3. Log file is being written to
4. Verify file path updated for Opportunity Gating

### Metrics show 0.00

**Normal** if:
- Less than 2 trades completed (need data for calculations)
- All trades break-even (rare)

**Problem** if:
- Multiple trades exist but metrics still 0.00
- Check state file for corrupted data

### Colors not showing

**Terminal doesn't support ANSI colors**:
- Use Windows Terminal (recommended)
- Or use CMD with ANSI support enabled
- Linux/Mac: Should work by default

---

## Best Practices

### Daily Routine

1. **Morning**: Check QUANT_MONITOR for overnight performance
2. **Check**: Gate effectiveness (should block 70-80% of SHORTs)
3. **Verify**: LONG/SHORT distribution trending toward 85%/15%
4. **Review**: Any alerts, especially gate-related

### Weekly Review

1. **Sunday**: Full metrics review
2. Check if Sharpe >2.0 (target for leveraged system)
3. Review win rate trend (target: 63.9%)
4. Verify LONG/SHORT distribution (85%/15%)
5. Analyze gate block rate (target: 70-80%)
6. Compare to backtest expectations

### Monthly Review

1. Calculate monthly Sharpe, Sortino, Calmar
2. Review maximum drawdown trend (target: <10%)
3. Analyze signal quality drift
4. Check gate effectiveness over time
5. Verify leverage calculations accurate
6. Compare to backtest performance (18.13% per window)

---

## FAQ

**Q: What's different in Opportunity Gating vs previous system?**
A: SHORT trades are gated by opportunity cost. Only enter SHORT when significantly more profitable than waiting for LONG.

**Q: Why are SHORT trades only 15%?**
A: The gate blocks ~75% of SHORT signals, only allowing high-quality SHORTs that beat LONG expected value by >0.1%.

**Q: What if I see no SHORT trades?**
A: Could be normal if market strongly bullish. Alert if >24h with no SHORT and SHORT signals exist.

**Q: What refresh rate should I use?**
A: 30s is optimal. Faster creates noise, slower misses events.

**Q: When should I worry about Sharpe ratio?**
A: If consistently <1.5 after 20+ trades, review strategy (target: >2.0).

**Q: How much drawdown is acceptable with 4x leverage?**
A: <10% is good, <5% is excellent. >10% requires action.

**Q: Should I stop trading if gate alerts fire?**
A: Not immediately. Monitor for 24-48h. Frequent gate alerts = investigate.

---

## Integration with Opportunity Gating Deployment

### Week 1 (Current - Validation Phase)

Monitor baseline:
- Win rate: Unknown (target >60%)
- LONG/SHORT distribution: Unknown (target 85%/15%)
- Gate effectiveness: Unknown (target 70-80% blocks)
- Trade frequency: Unknown (target ~3.7/day)

### Week 2 (If Successful)

Watch for:
- Win rate stabilizing around 63.9%
- LONG/SHORT distribution settling at 85%/15%
- Gate consistently blocking 70-80% of SHORTs
- Sharpe ratio trending toward >2.0

### Week 3+ (Mainnet Consideration)

Verify:
- Sharpe ratio >2.0 consistently
- Win rate >60% over 30+ trades
- Gate effectiveness 70-80%
- Max drawdown <10%
- All metrics align with backtest

---

## Support

For issues or questions:
1. Check logs: `logs/opportunity_gating_bot_4x_*.log`
2. Verify state file: `results/opportunity_gating_bot_4x_state.json`
3. Review configuration in `scripts/monitoring/quant_monitor.py`
4. Check deployment docs: `claudedocs/OPPORTUNITY_GATING_DEPLOYMENT_20251017.md`

---

**Status**: Updated for Opportunity Gating System
**Version**: 2.0
**Last Updated**: 2025-10-17
**Recommended**: Essential for Week 1 validation monitoring
**Next**: Monitor first trades and gate effectiveness
