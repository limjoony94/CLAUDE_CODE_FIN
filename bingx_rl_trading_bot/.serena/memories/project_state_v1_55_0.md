# Pattern 5m Bot — v1.55.0 Current State (2026-03-08)

## Strategy
- BTC 5m 3-candle pattern trading, **fixed 3x leverage**, BingX Hedge mode
- **131 patterns (59L + 72S)**, MAE/MFE + ATR-scaled discovery
- ATR config: a14/w576/clamp [0.5, 1.5]
- TP 0.85-2.80%, SL 1.44-5.95%
- Quality: Edge>=18pp, WR>=60%, SL>=1.0%, MC<0.01, min_trades>=25, Holdout 7d
- N=9 slots, 1/N=11.1% sizing, Direction Cap=7, Hedge mode
- Position Timeout: 288 bars (24h)
- Cascade SL Tightening: 85% (SL hit → same-dir SL dist ×0.15)
- Aggregate Risk Cap: counter 8%, with 15%
- Momentum Guard: >1.5%/15min → 1h block
- WF 3/3 PASS (N-pos+Cascade, OOS +128.9%, avg WR ~61.6%)
- N-pos IS: WR 71.3%, PnL +236.4%, MDD 1.37%, PnL/MDD 172.4x
- Data: 303d (btc_5m_270days_reclassified.csv)

## v1.55.0 Changes (03-08)
- N/A pattern recovery: crash recovery 시 trade_history에서 pattern 복원
- Exit classification: near-SL 40%/near-TP 30% proximity 분류
- Mass closure guard: 3+ 동시 청산 시 API force refresh

## Live Performance (03-08 기준)
- Full history (154t): WR 54.5%, PnL -118.85% (pre-03-05 오염 포함)
- **Clean baseline (post-03-05, 44t)**: WR 65.9%, PnL +40.89%, R:R 0.793
- Post-03-06 (29t): WR 82.8%, PnL +50.23%
- Breakeven WR 55.8%, margin +10.1pp (양의 기대값)
- LONG WR 0% (하락 레짐 편향, 소표본), SHORT WR 100%

## Disabled Mechanisms (5)
- Regime Sizing, Adaptive Leverage, Equity Curve, Correlation-Aware, Loss Burst Brake
- 각 config `enabled: true`로 재활성화 가능

## Key Paths (unchanged from v1.28.42, see that memory)
- Scanner: `scripts/scanner/pattern_scanner.py` (v2.4, npos default)
- Dynamic Patterns: `results/dynamic_patterns.json` (131pat, v1.53.0 rescan)
