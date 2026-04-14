C1 Breakout v2 does not use pattern scanning (single strategy, no dynamic patterns).

This command is DEPRECATED for C1 Breakout. The strategy uses fixed parameters:
- 15m Channel Breakout (lookback=15)
- Fractal SL (lookback=10, max 3.3x ATR cap)
- ATR Trailing TP (trail_K=2.5)

For parameter sensitivity analysis, use `/research-template` instead.
For walk-forward validation, use `/wf-validate`.
