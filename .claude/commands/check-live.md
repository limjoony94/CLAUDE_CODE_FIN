Check live C1 Breakout v2 trading performance and compare against expectations.

1. Read state from `bingx_rl_trading_bot/results/c1_breakout_state.json`
2. Read last 200 lines from `bingx_rl_trading_bot/logs/c1_breakout.log`
3. Calculate:
   - Days since bot started
   - Total trades and trade frequency (trades/day)
   - Overall win rate vs expected (~36.6%)
   - R:R ratio vs expected (~3.36)
   - Total PnL trajectory
   - Maximum drawdown vs expected (MDD ~5.4% additive 1x)
   - Current streak (wins/losses)

4. Performance assessment:
   - GREEN: R:R >= 2.5, positive PnL, MDD < 8% (additive 1x)
   - YELLOW: R:R 1.5-2.5, or MDD 8-15%
   - RED: R:R < 1.5, or MDD > 15%, or no trades for 24h

5. Exit type analysis:
   - TRAIL_TP exits vs SL exits (expected: ~85% trail, ~15% SL)
   - Emergency exits (should be 0)
   - Timeout exits (should be 0)

6. Recommendations based on current state

Note: PnL in bot logs includes leverage (3x). Strategy uses additive PnL.
Expected: WR ~36.6% with R:R ~3.36 yields +0.509%/day (additive 1x).
