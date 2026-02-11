# Analysis Scripts

> **Updated**: 2026-02-11 | 45+ scripts

연구, 백테스트, 검증, 최적화를 위한 분석 스크립트입니다.

## v1.27.0 핵심 연구 (2026-02-10)

| 스크립트 | 설명 |
|---------|------|
| `uniform_tp_validation.py` | Uniform TP 70% 검증 (8-phase validation) |
| `risk_management_research.py` | Daily limit sweep, MC MDD, Kelly criterion |
| `dual_tp_stability_research.py` | Dual-TP FWR → D_Uniform_70pct 발견 |
| `trade_microstructure_research.py` | TP vs SL race dynamics, MFE/MAE |
| `distance_edge_decomposition.py` | WR = distance + edge 분해 분석 |
| `rr_optimization_research.py` | R:R >= 1.0 최적화 분석 |
| `context_filter_research_v2.py` | Context filter 심층 연구 (8-phase, BH FDR — FAIL) |

## v1.26.x 최적화 (2026-02-09)

| 스크립트 | 설명 |
|---------|------|
| `tp_sl_optimization_v1264.py` | 52패턴 TP/SL grid search |
| `tp_sl_deep_validation.py` | 5-phase deep validation (CV, plateau, edge, OOS, composite) |
| `wf_validation_v1263.py` | v1.26.3 WF 검증 |
| `reopt_comparison.py` | Re-optimization 비교 |

## v1.26.0~v1.26.1 R:R 연구 (2026-02-08)

| 스크립트 | 설명 |
|---------|------|
| `tp_ge_sl_research.py` | R:R >= 1.0 연구 v1 |
| `tp_ge_sl_research_v2.py` | v2 (compound overflow 발견) |
| `tp_ge_sl_research_v3.py` | v3 (simple returns 수정) |
| `portfolio_pruning_v4.py` | Leave-one-out 포트폴리오 프루닝 |
| `tp_sl_bias_research.py` | 랜덤 baseline binomial test |
| `strategy_deep_review.py` | 전략 심층 리뷰 |

## 패턴 발견/검증 (v1.24.0~v1.25.x)

| 스크립트 | 설명 |
|---------|------|
| `full_270d_revalidation.py` | 270일 전수 재검증 |
| `unified_pattern_discovery.py` | 통합 패턴 발견 |
| `pattern_discovery.py` | 패턴 발견 |
| `pattern_validation_comprehensive.py` | 종합 검증 |
| `v125_validation_fixed.py` | v1.25.0 검증 |
| `deep_portfolio_analysis.py` | 포트폴리오 분석 |

## 모니터링/유틸리티

| 스크립트 | 설명 |
|---------|------|
| `analyze_recent_signals.py` | 최근 신호 분석 |
| `current_market_analysis.py` | 현재 시장 분석 |
| `validate_data_quality.py` | 데이터 품질 검증 |
| `calculate_buyhold_baseline.py` | Buy&Hold 베이스라인 |

## Usage

```bash
cd bingx_rl_trading_bot
python scripts/analysis/<script_name>.py
```

결과는 `results/` 디렉토리에 JSON으로 저장됩니다.
