# Analysis Scripts

> **Updated**: 2026-03-12 | **217 scripts** | **Bot Version**: v1.56.2

연구, 백테스트, 검증, 최적화를 위한 분석 스크립트입니다.

## v1.56.x 연구 (2026-03-11~12)

| 스크립트 | 설명 |
|---------|------|
| `mechanism_cross_validation_study.py` | 6-mechanism 교차검증 (15-seed, NON-DISC) |
| `timeout_cross_validation_study.py` | Timeout 독립 효과 6-phase 검증 |
| `mechanism_disc_followup.py` | Mechanism discriminating power 후속 분석 |
| `na_contamination_study.py` | N/A 오염 정화 + Duplicate Guard 연구 |
| `candle_classification_consistency.py` | 캔들 분류 일관성 점검 |

## v1.54.0~v1.55.0 연구 (2026-03-05~08)

| 스크립트 | 설명 |
|---------|------|
| `cascade_sl_optimization.py` | Cascade SL scanner 구현 검증 |
| `wr_gap_study.py` | Live WR gap 분석 (mechanism dominance 86%) |
| `live_pattern_audit.py` | Live 패턴 성과 감사 |

## v1.44.0~v1.53.0 파라미터 Sweep (2026-03-05)

| 스크립트 | 설명 |
|---------|------|
| `atr_infra_sweep_study.py` | ATR period/window + momentum threshold 4-Phase |
| `atr_mdd_param_sweep_study.py` | ATR clamp_hi + MDD sizing 5-Phase |
| `atr_clamp_resweep_study.py` | ATR clamp 2D grid resweep |
| `aggrisk_resweep_study.py` | AggRisk counter re-sweep |
| `timeout_sweep_study.py` | Timeout 12-config sweep |
| `position_sizing_study.py` | Position sizing 연구 |
| `pattern_sl_cooldown_study.py` | SL cooldown 연구 |
| `time_of_day_study.py` | 시간대별 필터 연구 |
| `interaction_effect_study.py` | 파라미터 상호작용 효과 |
| `nslots_sweep_study.py` | N-slots sweep |
| `counter_regime_cap_study.py` | Counter-regime cap 연구 |

## v1.38.0~v1.42.0 메커니즘 연구 (2026-03-01~03)

| 스크립트 | 설명 |
|---------|------|
| `correlated_loss_study.py` | Cascade SL Tightening 발견 (H5_Cascade) |
| `guard_ablation_study.py` | Guard mechanism ablation (3개 비활성화) |
| `equity_curve_mdd_study.py` | Equity Curve Trading + Correlation-Aware |
| `npos_scanner_validation.py` | N-pos Scanner production alignment |
| `loss_burst_brake_study.py` | Loss Burst Brake 연구 |

## v1.35.0~v1.36.x 연구 (2026-02-25~27)

| 스크립트 | 설명 |
|---------|------|
| `neutral_window_discovery.py` | Neutral window 자동 발견 |
| `direction_cap_study.py` | Direction Cap portfolio 최적화 |
| `momentum_guard_study.py` | Momentum Guard spike protection |
| `aggregate_risk_study.py` | Directional risk cap 연구 |
| `emergency_sl_study.py` | Emergency SL overhaul |
| `hedge_vs_oneway_reverification.py` | Hedge vs One-Way 재검증 |
| `direction_regime_study.py` | Direction regime filter 연구 |

## v1.28.x~v1.34.0 기반 연구 (2026-02-12~24)

| 스크립트 | 설명 |
|---------|------|
| `mae_mfe_discovery.py` | MAE/MFE TP/SL discovery |
| `holdout_validation.py` | Holdout 7일 OOS 검증 |
| `mdd_sizing_study.py` | MDD-based position sizing |
| `h7_critical_validation.py` | Entry Optimization (ROLLBACK) |
| `strategy_foundation_study.py` | WF non-discrimination 발견 |

## v1.26.x~v1.27.x 기초 연구 (2026-02-08~12)

| 스크립트 | 설명 |
|---------|------|
| `uniform_tp_validation.py` | Uniform TP 70% 검증 |
| `risk_management_research.py` | Daily limit sweep, MC MDD, Kelly |
| `tp_sl_optimization_v1264.py` | 52패턴 TP/SL grid search |
| `tp_sl_deep_validation.py` | 5-phase deep validation |
| `context_filter_research_v2.py` | Context filter 연구 (FAIL) |
| `distance_edge_decomposition.py` | WR = distance + edge 분해 |
| `portfolio_pruning_v4.py` | Leave-one-out 포트폴리오 프루닝 |

## Usage

```bash
cd bingx_rl_trading_bot
python scripts/analysis/<script_name>.py
```

결과는 `results/` 디렉토리에 JSON으로 저장됩니다.
모든 연구는 CLAUDE.md의 Standard Research Protocol을 따릅니다.
