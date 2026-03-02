# PDCA Report: MDD Reduction v1.40.0

## Overview
| Item | Value |
|------|-------|
| Feature | MDD Reduction (Equity Curve Trading + Correlation-Aware Entry) |
| Version | v1.40.0 |
| PDCA Cycle | Plan → Design → Do → Check(100%) |
| Duration | ~4 hours (research included) |
| Date | 2026-03-02 |

## Research Phase
- **Script**: `scripts/analysis/mdd_reduction_study.py`
- **Scope**: 7 hypotheses, 45 scenarios, 6 phases, 66s runtime
- **Data**: N-pos portfolio simulator (compound, N=9, 130pat)

### Hypotheses Summary
| ID | Name | MDD Δ | PnL Δ | PnL/MDD | WF | Verdict |
|----|------|-------|-------|---------|-----|---------|
| H1 | EqCurve_half_ema30 | -32.1% | -6.8% | +36.3% | 3/3 PASS | GO |
| H2 | EqCurve_quarter_ema30 | -22.5% | -3.9% | +24.1% | 3/3 PASS | candidate |
| H3 | Dynamic_0.5-1.5_ema20 | -28.4% | -5.2% | +31.0% | 3/3 PASS | candidate |
| H4 | Streak_L3_0.5 | -18.9% | -2.1% | +20.6% | 3/3 PASS | candidate |
| H5 | CorrAware_dir60 | -29.7% | -8.1% | +31.3% | 3/3 PASS | candidate |
| H6 | CorrAware_dir70 | -36.4% | -4.5% | +51.0% | 3/3 PASS | GO |
| H7 | VaR_pct95_0.5 | -25.8% | -6.0% | +26.5% | 3/3 PASS | candidate |

### Combo Analysis (Phase 5)
| Combo | Components | MDD Δ | PnL Δ | PnL/MDD | WF |
|-------|-----------|-------|-------|---------|-----|
| Combo1 | H1+H5 | -42.3% | -8.9% | +58.2% | 3/3 |
| **Combo2** | **H1+H6** | **-47.8%** | **-3.9%** | **+84.1%** | **3/3** |
| Combo3 | H2+H6 | -39.1% | -2.8% | +59.5% | 3/3 |

**Selection: Combo2** — MDD 감소 폭(-47.8%)이 PnL 감소 폭(-3.9%)을 압도적으로 초과. PnL/MDD +84.1% 향상.

## Implementation

### Files Modified (5)

| File | Changes |
|------|---------|
| `state.py` | `equity_curve_tracker` default (`long_cum_pnls`, `short_cum_pnls`) |
| `models.py` | `BotState` type extension |
| `bot.py` | 4 new functions + 3 integration points |
| `position_open.py` | `equity_curve_scale` parameter (size/open/slot) |
| `config.yaml` | `equity_curve_trading` + `correlation_aware_entry` sections |

### New Functions in bot.py
1. `_check_equity_curve_sizing(state, config, direction)` → size multiplier (1.0 or 0.5)
2. `_check_correlation_aware_entry(state, config, signal_direction, df)` → True=BLOCK
3. `_update_equity_curve_tracker(state, direction, pnl_pct)` → append cum_pnl
4. `_record_trade_for_equity_curve(state, config)` → dedup wrapper

### Guard Chain (Updated)
```
momentum_guard → loss_burst_brake → correlation_aware_entry(NEW) → aggregate_risk_cap → adaptive_leverage → equity_curve_sizing(NEW) → open_position
```

### Key Design Decision: Per-Direction Tracking
사용자 요구에 따라 LONG/SHORT 독립 에퀴티 커브:
- `long_cum_pnls[]`: LONG 거래만의 누적 PnL 시퀀스
- `short_cum_pnls[]`: SHORT 거래만의 누적 PnL 시퀀스
- 최대 100개 유지, SMA(30) lookback

### Config
```yaml
risk:
  equity_curve_trading:
    enabled: true
    ema_trades: 30
    size_mult: 0.5
  correlation_aware_entry:
    enabled: true
    dir_pct_threshold: 0.70
```

### Rollback
- `equity_curve_trading.enabled: false` → 에퀴티 커브 사이징 비활성화
- `correlation_aware_entry.enabled: false` → 상관 가드 비활성화
- 각각 독립 롤백 가능

## Verification
- **Tests**: 1061 passed (all existing tests pass)
- **Match Rate**: 100% (design ↔ implementation)
- **Regression**: None

## Expected Impact (Production)
| Metric | Before (v1.39.0) | After (v1.40.0) | Change |
|--------|-------------------|------------------|--------|
| MDD | 6.5% | 3.4% | -47.8% |
| PnL | 106.8% | 102.6% | -3.9% |
| PnL/MDD | 16.45 | 30.29 | +84.1% |
| WF | 3/3 PASS | 3/3 PASS | maintained |

## Lessons Learned
1. Per-direction 에퀴티 커브가 combined보다 정밀한 방향별 리스크 관리 제공
2. Combo가 개별 가설보다 비선형 시너지 효과 (MDD -47.8% > H1 -32.1% + H6 -36.4%)
3. Correlation guard는 aggregate risk cap과 상호보완 (방향 비율 vs SL 노출 합산)
4. Dedup 패턴 (_last_recorded timestamp)으로 중복 기록 방지 — 기존 burst brake와 동일
