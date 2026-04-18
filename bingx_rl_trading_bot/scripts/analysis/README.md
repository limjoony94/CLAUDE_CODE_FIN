# Analysis Scripts (C1 Breakout v2.6)

> **Updated**: 2026-04-18 | **Active Strategy**: C1 Breakout v2.6
> **레거시 (Pattern 5m, MAVS-15, CP, BTV, Volspike 등)**: `archive/cleanup_20260418/analysis/` 참조

연구·백테스트·검증·정합성 스크립트. C1 관련만 유지.

## 카테고리

### 백테스트 (C1 baseline 검증)
| 스크립트 | 내용 |
|---------|------|
| `c1_last_7days_backtest.py` | 최근 7일 |
| `c1_last_30days_backtest.py` | 최근 30일 |
| `c1_reverse_30days_backtest.py` | 30일 역방향 |
| `c1_reverse_full_backtest.py` | 전체 역방향 |
| `c1_v2_deep_validation.py` | v2 전반 심층 검증 |
| `c1_v25_verify.py` | v2.5 확인 |

### 적대적 감사 (Overfit / Look-ahead / Fee)
| 스크립트 | 내용 |
|---------|------|
| `c1_bias_overfit_audit.py` | 전반 bias/overfit |
| `c1_lookahead_overfit_fee_audit.py` | 3-section 18-test 감사 |
| `c1_deep_stress_test.py` | 10 stress tests |
| `c1_emergency_audit.py` | Emergency SL 엣지 |
| `c1_extreme_audit.py` | 극단 시나리오 |

### 비판 평가 (Critical Cycles)
| 스크립트 | 내용 |
|---------|------|
| `c1_critical_3x.py` | 3배 레버리지 크리티컬 |
| `c1_critical_new_angles.py` | 새 각도 비판 |
| `c1_critical_new2.py` | 추가 비판 |
| `c1_loss_verification.py` | 손실 검증 |
| `c1_compound_reality_check.py` | Compound 수익 현실 검증 |

### Trail · SL 메커니즘
| 스크립트 | 내용 |
|---------|------|
| `c1_trail_comparison.py` | Trail 변종 비교 |
| `c1_trail_math_verify.py` | Trail 수식 검증 |
| `trail_alternatives_comparison.py` | 9 trail variants |
| `intrabar_trail_impact.py` | Intrabar bar vs tick 영향 |

### Refined 변종 연구 (최종 baseline 유지 결론)
| 스크립트 | 내용 |
|---------|------|
| `c1_refined_variants.py` | A/B/C/D 변종 |
| `c1_refined_validation.py` | 변종 검증 |
| `c1_refined_bootstrap_mdd.py` | Stationary bootstrap MDD |
| `c1_refined_dmining_check.py` | Data mining 차단 |
| `c1_refined_stress.py` | Stress 확장 |

### 라이브 vs 백테스트 정합성
| 스크립트 | 내용 |
|---------|------|
| `live_vs_backtest_verification.py` | 1:1 trade matching |
| `live_window_analysis.py` | 13-trade windows |
| `live_pattern_analysis.py` | 15-trade pattern breakdown |
| `live_pattern_audit.py` | Live 패턴 감사 |
| `live_atr_regime_check.py` | Live ATR 레짐 확인 |
| `shake_out_pattern_verification.py` | 털어내기 패턴 검증 |
| `forward_path_simulation.py` | MC forward simulation |

### 파라미터 / 레짐 / 리스크
| 스크립트 | 내용 |
|---------|------|
| `extended_param_grid.py` | 35/35 ±50% 양수 검증 |
| `c1_lookback_comparison.py` | Lookback 비교 |
| `c1_regime_classifier.py` | 레짐 분류기 |
| `regime_asymmetry_test.py` | LONG/SHORT WR by regime |
| `low_vol_same_price_regime.py` | 저변동성 레짐 |
| `liquidation_risk_check.py` | 청산 리스크 (0/1028) |
| `c1_npos_leverage.py` | N-pos / 레버리지 |
| `c1_oracle_switching.py` | Oracle switching |

### 대체 전략 비교
| 스크립트 | 내용 |
|---------|------|
| `mean_reversion_vs_breakout_research.py` | 하따/상따 vs 돌파 |
| `c1_vs_mavs15_critical_eval.py` | C1 vs MAVS-15 |
| `c1_ablation_study.py` | Ablation study |

## 사용법

```bash
cd bingx_rl_trading_bot
python scripts/analysis/<script_name>.py
```

결과는 `results/`에 JSON으로 저장. 모든 연구는 `claudedocs/STANDARD_RESEARCH_PROTOCOL.md` 준수.
