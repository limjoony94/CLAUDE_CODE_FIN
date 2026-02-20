Check live trading performance and compare against expectations.

1. Read metrics from `bingx_rl_trading_bot/results/pattern_5m_metrics.json`
2. Read state from `bingx_rl_trading_bot/results/pattern_5m_bot_state.json`
3. Calculate:
   - Days since bot started
   - Total trades and trade frequency (trades/day)
   - Overall win rate vs expected (68%)
   - Total PnL vs expected trajectory
   - Actual edge (total_pnl / total_trades) vs expected (0.27%)
   - Maximum drawdown
   - Current streak (wins/losses)
   - Daily loss status vs limit (13%)

4. Performance assessment:
   - GREEN: WR >= 65%, positive PnL, MDD < 25%
   - YELLOW: WR 60-65%, or MDD 25-35%, or daily loss > 8%
   - RED: WR < 60%, or MDD > 35%, or daily loss > 10%

5. Pattern-level analysis:
   - Which patterns have traded most
   - Which patterns have highest/lowest WR
   - Any patterns with 0 trades (never triggered)

6. Recommendations based on current state

Note: `recent_wins/losses` are rolling buffers — use total_pnl/trade counts for accurate metrics.
PnL in bot logs includes leverage (3x). TP/SL % are pre-leverage price distances.
