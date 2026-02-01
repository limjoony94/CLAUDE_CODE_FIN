# Pattern 5m v1.21.0 - Conservative Per-Pattern TP/SL

**Date**: 2026-02-01
**Version**: v1.21.1

## Key Change
- v1.20.1 uniform 1.0/1.0 → **Conservative per-pattern TP/SL**
- MC<0.01 패턴만 개별 최적화 (8/13), 나머지 uniform 유지 (5/13)
- 13 patterns (7L+6S), regime disabled

## Version Chain
- v1.20.0: 연구/프로덕션 분류 불일치 수정 (avg_body_20), 21→13 패턴 재발굴
- v1.20.1: Early-bar classification fix (default avg_body_20=1.0 for bars 0-19)
- v1.21.0: Conservative per-pattern TP/SL optimization
- v1.21.1: Leverage side fix (`params={'side': 'BOTH'}`), stale regime state cleanup, Serena memory cleanup

## Per-Pattern TP/SL Config
```
LONG:
  U-MU-H:   1.5/1.5  (MC=0.0000, optimized)
  MD-ST-MD: 2.0/2.0  (MC=0.0078, optimized)
  GS-U-BD:  1.0/1.0  (MC=0.0372, uniform - conservative)
  MD-MD-ST: 1.5/2.0  (MC=0.0002, optimized)
  BU-IH-DN: 1.5/2.0  (MC=0.0022, optimized)
  MD-H-MD:  1.0/1.0  (MC=0.0014, uniform best)
  IH-MD-MD: 1.5/2.0  (MC=0.0020, optimized)
SHORT:
  DN-D-BD:  1.0/1.0  (MC=0.2390, uniform - MC fail)
  BD-U-GS:  1.5/2.0  (MC=0.0042, optimized)
  DN-GS-H:  1.0/1.0  (MC=0.0176, uniform - conservative)
  U-DF-BU:  1.0/1.5  (MC=0.0010, optimized)
  BD-GS-BD: 1.0/1.0  (MC=0.0120, uniform - conservative)
  DN-IH-IH: 1.0/1.5  (MC=0.0000, optimized)
```

## Portfolio Backtest (270-day, compound equity)
| Config | Trades | WR | PnL | MDD | PF | WF |
|--------|--------|-----|-----|-----|-----|-----|
| A: Uniform 1.0/1.0 | 353 | 73.7% | +8,330% | 14.9% | 2.58 | 5/5 |
| B: Full Optimal | 310 | 78.1% | +83,510% | 20.6% | 3.33 | 5/5 |
| **C: Conservative** | **312** | **78.5%** | **+56,722%** | **20.6%** | **3.19** | **5/5** |

Selected: **C (Conservative)** — MC<0.01만 최적화, 나머지 uniform

## MC p-value Note
- Portfolio-level MC=0.0000 for all configs is mathematically correct
- With 300+ trades at 73%+ WR, sign randomization never beats actual (compound equity)
- Per-pattern MC (15-57 trades) provides meaningful discrimination
- Portfolio WF and Period Stability are more useful validation at portfolio level

## Research Scripts
- `per_pattern_tpsl_optimization.py`: Grid search TP/SL per pattern with WF+MC
- `portfolio_tpsl_comparison.py`: A/B/C portfolio comparison

## v1.21.1 Fixes
- `position_open.py:278`: `exchange.set_leverage(lev, sym, params={'side': 'BOTH'})` — BingX API requires side arg
- `pattern_5m_bot_state.json`: Removed stale `current_regime`/`regime_tp_sl` fields (regime disabled since v1.19.0)
- Deleted 5 obsolete Serena memories (engulf×3, old session, old compendium)

## Live Status (as of v1.21.1)
- 8 trades, 50.0% WR (still small sample)
- ~0.84 trades/day expected (13/1728 patterns)
- Next milestone: 30-50 trades for backtest vs live comparison
