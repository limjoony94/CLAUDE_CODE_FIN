Run the pattern scanner to discover/validate trading patterns.

Default mode (MAE/MFE discovery, current production settings):
```bash
cd bingx_rl_trading_bot && python scripts/scanner/pattern_scanner.py --discovery-method mae_mfe --edge-threshold 21.8
```

With Walk-Forward validation:
```bash
cd bingx_rl_trading_bot && python scripts/scanner/pattern_scanner.py --discovery-method mae_mfe --edge-threshold 21.8 --wf-folds 3
```

With IS window optimization (Edge Decay Study: 135d optimal):
```bash
cd bingx_rl_trading_bot && python scripts/scanner/pattern_scanner.py --discovery-method mae_mfe --edge-threshold 21.8 --is-days 135 --wf-folds 3
```

After scanning:
1. Compare results against current production patterns (59 patterns, 12L+47S)
2. Report new patterns discovered or patterns lost
3. Report any significant changes in TP/SL values
4. Report WF results if --wf-folds was used
5. Do NOT auto-deploy results without explicit user approval

Critical reminders:
- Scanner uses MAX_BARS=288 (24h timeout)
- Quality filter: Edge>=21.8pp + WR>=60% + SL>=1.0% + MC<0.01 + min_trades>=25
- Fee calculation includes leverage (FEE_PCT * LEVERAGE = 0.30%)
- Production patterns backup before any deployment: `results/dynamic_patterns_backup.json`
