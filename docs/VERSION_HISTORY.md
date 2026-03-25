# Version History (Full)

> CLAUDE.md에서는 최근 주요 버전만 표시. 전체 히스토리는 이 파일 참조.
> **Last Updated**: 2026-03-25 | **Current Version**: v1.67.1

| 버전 | 날짜 | 변경사항 |
|------|------|---------|
| **v1.67.1** | 03-25 | **Cascade SL original distance fix**: Reactive cascade(`_cascade_tighten_sls`)와 pre-emptive cascade(`_apply_preemptive_cascade`)가 `original_sl_distance`를 vol_adapt 이후 SL에서 계산 → cascade SL이 entry 근처에 배치 → 가격 유효성 검증 SKIP → `cascade_tightened=True`로 마킹되지만 SL 미변경 → cascade 무력화. Fix: `_sl_price_original`(진입 시 원래 SL) 우선 사용. BT 영향: +0.3%(무). 라이브: cascade SKIP 비율 감소로 실효성 향상 기대. 연구: sim_production_parity_study(SL 4배 괴리 발견 → cascade 90% vs live 16%), realistic_sim_study(1-bar delay + vol_adapt + decay + timeout_pnl realistic sim), cascade_tighten_sweep(95% 최적 재확인), overfit_check(FP=0, IS-OOS gap 동일, 과적합 아님). **1111 tests ALL PASSED**. |
| **v1.67.0** | 03-24 | **MFE Median TP + OPP_SIGNAL Exit**: (1) `tp_mode: mfe_median` — 각 패턴의 MFE(최대유리편향) 중앙값을 TP로 직접 사용. 자유파라미터 0개(데이터 직접 사용), 과적합 위험 최저(overfit score 30.6, 12전략 중 최저). IS P/M 149.3x (baseline ×0.72 134.8x), OOS5 +385% (+380%). 패턴별 적응적 scale 0.37~1.44 (median 0.71). `config.py` load_dynamic_patterns에 tp_mode 분기 추가, `pattern_details.exc_stats.mfe_median` 참조. Rollback: `tp_mode: scale`. (2) OPP_SIGNAL Exit (v1.66.0): `_check_opposite_signal_exit()` — 2회 연속 반대 신호 + 수익>0.1% → 시장가 청산 (opp_partial_close_study: 100% 전량청산 최적, 50% 부분청산 OOS -18.6%). (3) high_rr_pattern_discovery: TP>SL 고R:R 패턴 3개만 발견, DISCRIMINATING이나 표본 부족(78건). TP<SL 구조 유지 확정. (4) tp_calibration_study: 12개 TP 보정 전략 비교 — uniform, RR-bucket, WR-adaptive, MFE-median, edge-weight, trade-count 등. Joint TP+SL 동시 보정(SL 축소)은 WR margin 급감으로 비추. (5) tp_calibration_overfit_diag: 전략별 IS-OOS gap, fold CV, 자유 파라미터, 3→5 fold 열화 정량 평가. **1111 tests ALL PASSED**. |
| **v1.65.0** | 03-15 | **Pre-emptive Cascade SL + 파라미터 재최적화 + 시스템 본질 재정의**: (1) `_apply_preemptive_cascade()`: 방향별 미실현 손실 > 4% 시 SL 선제 95% 축소 — SL 피격 전 클러스터 방어. (2) DirCap 7→6 (동시 방향 노출 감소). (3) AggRisk counter 999→10%, with 999→15% 재활성화. (4) 13개 심층 연구 기반 시스템 본질 재정의: "BTC 5m 변동성 수확기" — 랜덤 진입 86% 성과, 방향 반전 100% 수익, 레짐 무관. COMBO_B: WF OOS +410.7% (v1.63.1 +308.4%), IS P/M 127.3x, R:R 1.291. Scanner: pre-emptive cascade + DirCap=6 + AggRisk 10 정합. `claudedocs/system_deep_analysis_20260315.md`. **1111 tests ALL PASSED**. |
| v1.63.1 | 03-14 | **Decay Rate 0.997→0.9975**: 10-point sweep 최적화. OOS +339.5%. Option B (TP→SL) 열위 확인. |
| v1.63.0 | 03-14 | **Exp-Decay TP (0.9975^bars)**: Bar 0부터 TP 지수 감소. IS +457.5%, OOS +332.5%. trades +40%. |
| v1.62.0 | 03-14 | Time-Decay TP (linear_144). → v1.63.0에서 exp_decay로 교체. |
| v1.61.0 | 03-13 | **TP Scale Factor 0.5→0.72**: R:R 1.009, BE WR 49.8%, WR margin +27.0pp. |
| v1.60.1 | 03-13 | Scanner _effective_vol_mult Cap (Production Parity). |
| v1.60.0 | 03-13 | **Soft-Delete Mass Closure** (Two-Cycle Confirmation). |
| v1.59.6 | 03-13 | CCXT Type Adopt Fix. |
| v1.59.5 | 03-12 | Emergency SL Update Fix (Cancel-first + EXCHANGE_MANAGED 해소). |
| v1.59.4 | 03-12 | N/A Pattern Prevention + Orphan Order Retry. |
| v1.59.3 | 03-12 | Emergency SL Race Condition Fix. |
| v1.59.2 | 03-12 | Median TP/SL Fallback. |
| v1.59.1 | 03-12 | Position Tracking Fix + SL×1.1 Revert. |
| v1.59.0 | 03-12 | Orphan Prevention 3-layer defense. |
| v1.57.0 | 03-12 | TP Scale Factor 0.5. |
| **v1.56.2** | 03-12 | **Code Audit 전수점검 + 7건 수정**: (1) `update_single_sl` Place-first/Cancel-after (Cascade SL 보호 갭 제거) (2) SL 실패 시 Emergency SL 즉시 호출 (3) Momentum cooldown `save_state` 영속화 (4) `datetime.fromisoformat('')` 방어 (5) `except Exception: pass`→로깅 (6) always-truthy `or {}` 수정 (7) Hardcoded 300s→`CANDLE_DURATION_MS//1000`. 교차검증으로 57건 중 4건 FALSE POSITIVE 확인 (C-10,H-9,H-11,L-1일부). **1061 tests ALL PASSED**. |
| **v1.56.1** | 03-11 | **N/A 오염 정화 + Duplicate Guard**: (1) trade_history 22건 N/A+dup 제거 (2) `record_closed_position` duplicate guard 추가 (3) `pattern_name` 필드 우선 사용. TP+SL WR 61.6→67.4%, gap 11.2→5.4pp (p=0.10, NOT significant). |
| **v1.56.0** | 03-11 | **Scanner regime_mult 정합 + 메커니즘 교차검증**: Scanner DEFAULT_REGIME_MULT 0.3→1.0 (production v1.42.0에서 비활성화한 Regime Sizing을 scanner에서도 정합). IS PnL/MDD 48.3→88.5x (+83%). WF OOS +128.9→+206.1% (+60%). 6-mechanism 교차검증: M6 Regime 유해(-121.7%), M5 Timeout 필수(+80.7%), TOP3(Timeout+Cascade+Momentum) 최적. |
| **v1.55.0** | 03-08 | **Live 안정성 3종 개선**: (1) N/A 패턴 방지: crash recovery 시 trade_history에서 pattern 복원 (2) Exit 분류 강화: near-SL 40%/near-TP 30% proximity 분류 (3) Mass closure guard: 3+ 동시 청산 시 API 재확인. |
| **v1.54.0** | 03-07 | **Scanner Cascade SL 구현 + EXIT 분류 개선**: N-pos IS: WR 71.3%, PnL +236.4%, MDD 4.87%(MTM), PnL/MDD 48.3x. WF 3/3 PASS (OOS +128.9%). CASCADE_SL exit reason 추가. |
| **v1.53.0** | 03-05 | **Data Extension 303d + Pattern Rescan (131pat, 59L+72S)**: 데이터 297d→303d (87,315 rows, 2026-03-04까지). BTC +10.3% 상승장 기간 포함으로 SHORT 패턴 선별 개선. **131패턴 (59L+72S)**, v1.52.0(125) 대비 +6S. Neutral 259d (drift -0.72%). IS: WR 95.4%, PnL +1,420%, MDD 27.0%. N-pos IS: 831 trades, WR 73.6%, PnL +120.4%, MDD 5.3%, PnL/MDD 22.7x. Holdout 10 FAIL 제거 (vs 6 previously). WF 3/3 PASS: F1 +19.0%, F2 +12.0%, F3 +30.4% (total +61.4%), 6 stable patterns. TP 0.85-2.8%, SL 1.44-5.95%. |
| **v1.52.0** | 03-05 | **Pattern Rescan (125pat) — Scanner-Production ATR 정합성 확보**: Scanner 기본값 ATR clamp [0.6,1.7] → [0.5,1.5] (production v1.47.0+v1.50.0 일치). 재스캔 결과: **125패턴 (59L+66S)**, 기존 130 대비 -5. IS: WR 95.1%, PnL +1,419%, MDD 28.3%. N-pos IS: 848 trades, WR 74.9%, PnL +123.8%, MDD 6.42%, PnL/MDD 19.3x. Holdout 6 SHORT 제거. WF N-pos 3/3 PASS: F1 +14.05%(WR 71.9%), F2 +12.39%(WR 68.1%), F3 +37.75%(WR 73.1%). TP 0.85-2.84%, SL 1.44-4.84%. Scanner defaults 업데이트: `DEFAULT_ATR_CLAMP_LO=0.5, DEFAULT_ATR_CLAMP_HI=1.5`. Backup: `dynamic_patterns_130pat_v1420_backup.json`. **1061 tests passed**. |
| **v1.51.0** | 03-05 | **Momentum Guard threshold 1.0→1.5% + ATR infra KEEP**: `atr_infra_sweep_study.py` 4-Phase (ATR period 7값 + window 8값 + 2D grid 9configs + momentum threshold 6값). **ATR infra**: p14/w576 현행 최적. **Momentum threshold 1.5%**: IS PnL/MDD 1137.7 (현행 1080.5 대비 +5.3%), MDD 2.92% (현행 3.00% -3%), OOS min fold 93.9 (동일), WF 3/3 PASS. config `momentum_guard.threshold_pct: 1.5`. **1061 tests passed**. |
| **v1.50.0** | 03-05 | **ATR clamp_hi 1.7→1.5 + MDD sizing KEEP**: `atr_mdd_param_sweep_study.py` 5-Phase 연구. **ATR clamp_hi**: 1.5 IS PnL/MDD **1112.5** (현행 731.2 대비 +52%), PnL +3337%, MDD 3.00%. WF 3/3 PASS. config `atr_scale.clamp_hi: 1.5`. **1061 tests passed**. |
| **v1.49.0** | 03-05 | **AggRisk counter 5→8%**: `aggrisk_resweep_study.py` v1.48.0 baseline re-sweep. c8 IS PnL/MDD 731.2 (현행 568.2 +29%). Blocks 1693→735 (-57%). config `counter_cap: 8.0`. **1061 tests passed**. |
| **v1.48.0** | 03-05 | **Timeout 864→288 (72h→24h) + Early Exit KEEP**: `timeout_sweep_study.py` 12-config sweep. 288 (24h) 선택: IS PnL/MDD 568.2, **OOS min fold 82.0 (현행 69.8 대비 +17.5%)**, WF 3/3 PASS. Scanner MAX_BARS=288 일치. config `timeout_bars: 288`. **1061 tests passed**. |
| **v1.47.0** | 03-05 | **ATR clamp_lo 0.6→0.5**: `atr_clamp_sweep_study.py` 2D grid. lo=0.5/hi=1.7: IS PnL/MDD 384.7→554.7 (+44%), MDD 4.45→3.32% (-25%), WF 3/3 PASS. config `atr_scale.clamp_lo: 0.5`. **1061 tests passed**. |
| **v1.46.0** | 03-05 | **Momentum Guard parameter tuning — lookback 6→3, cooldown 6→12**: 73-config 3D grid. IS PnL/MDD 291→385 (+32%), OOS avg +101→+110%. config `lookback_bars: 3`, `cooldown_bars: 12`. **1061 tests passed**. |
| **v1.45.0** | 03-05 | **Cascade SL tighten_pct 75→85%**: 4-Phase 연구 (7개 tighten_pct × 3개 패턴 필터). t85_full_130 PnL/MDD 291.5 (현행 201.9의 +44%), **WF 3/3 PASS OOS min fold +64.4%**. config `tighten_pct: 85`. **1061 tests passed**. |
| **v1.44.0** | 03-05 | **AggRisk Relaxation — counter 3→5%, with 7→15%**: 5개 연구 연쇄 기반 비판적 검증 통과. IS PnL/MDD 97.0→201.9 (+108%), OOS avg +55.2→+82.7%. config `counter_cap: 5.0`, `with_cap: 15.0`. **1061 tests passed**. |
| **v1.42.0** | 03-03 | **Mechanism Stack Optimization — M2 Regime + M4 AdaptiveLev 비활성화**: P3 CascadeSL이 MDD 방어 대체. M2+M4 redundancy -46.94. IS PnL/MDD 34.36→123.67 (+260%), MDD 4.50%→3.82% (-15%), OOS +77.3%→+160.7%. config `regime_sizing.enabled: false`, `adaptive_leverage.enabled: false`. **1061 tests passed**. |
| **v1.41.0** | 03-03 | **Cascade SL Tightening — 동일 방향 SL 연쇄 축소**: `correlated_loss_study.py` H5_Cascade_t75 (PnL/MDD 79.4, baseline 27.4의 2.9배). `orders.py`: `update_single_sl()`. `position_monitor.py`: `_cascade_tighten_sls()`. config `cascade_sl_tightening`. **1061 tests passed**. |
| **v1.40.1** | 03-03 | **Guard Ablation — 중복 메커니즘 3개 비활성화**: M3 Equity Curve(유해), G4 Correlation-Aware(M2 중복), G3 Loss Burst(G2 중복). config `enabled: false` × 3. **1061 tests passed**. |
| **v1.40.0** | 03-02 | **MDD Reduction — Equity Curve Trading + Correlation-Aware Entry (Combo2)**: MDD -47.8% (6.5→3.4%), PnL -3.9%, PnL/MDD +84.1%, WF 3/3 PASS. **v1.40.1에서 비활성화**. |
| **v1.39.0** | 03-02 | **Adaptive Leverage — WR Confidence × Edge Quality**: MDD -32%, Calmar +39%, OOS consistency 0.823. **v1.42.0에서 비활성화** (M2+M4 redundancy). |
| **v1.38.1** | 03-02 | **N-pos Scanner default 전환 + Fold 2 데이터 슬라이싱 버그 수정**: `--npos` default=True. Fold 2 +13.35% (안정). 10가설 검증 → 현 baseline 유지. |
| **v1.38.0** | 03-01 | **N-pos Scanner — production-aligned portfolio simulator**: Scanner 백테스트에 N=9/compound/dir_cap/regime/agg_risk/momentum 통합. Live WR gap 32.3pp→15.4pp (52% 감소). **1061 tests passed**. |
| **v1.37.0** | 03-01 | **Loss Burst Brake + Live WR gap research suite**: 동일 방향 2회 손실 in 24h → 12h 차단. **v1.40.1에서 비활성화됨**. |
| **v1.36.6** | 03-01 | **Emergency SL overhaul — closePosition + proactive health check + cascade defense**: 2026-02-27 연쇄 청산 사건 근본 해결. **1061 tests passed**. |
| **v1.36.5** | 02-27 | **ATR scaling bug fix (2 critical paths)**: 전수조사(17 code paths) 기반 2개 크리티컬 버그 수정. **1061 tests passed**. |
| v1.36.4 | 02-27 | **Edge threshold 21.8→18pp (130 patterns)**: Neutral window에서 18pp 최적 — 130패턴(61L+69S), WF 3/3 PASS. |
| v1.36.3 | 02-27 | **Neutral window pattern discovery + Filter ablation study**: 257d neutral, 51pat(22L+29S), MDD 13.5%, PnL/MDD 79.6x, WF 3/3 PASS. |
| v1.36.2 | 02-27 | **Momentum Guard — counter-dir spike protection**: BTC >1%/30min 변동 시 역방향 진입 차단. PnL/MDD +9%, ALL WF 3/3 PASS. |
| v1.36.1 | 02-27 | **Direction Cap 8→7 + Data Extension Rescan (54pat)**: Live 괴리 분석 → cap7. 데이터 270d→297d. |
| v1.36.0 | 02-27 | **Multi-TF infra + 15m 실험 + 15m 비활성화**: 15m 거래 빈도 부족. Multi-TF Filter 7/7 STOP. |
| v1.35.6 | 02-27 | **Remove consecutive loss pause (deadlock fix)**: N=9에서 데드락 발생 → 제거. |
| v1.35.5 | 02-26 | **Aggregate directional risk cap**: dynamic_3_7 MDD -52%, PnL/MDD +16%, WF 3/3 PASS. |
| v1.35.4 | 02-26 | **SL breach auto-recalculation**: 3-bug chain 디버깅. |
| v1.35.3 | 02-26 | **Regime-aware position sizing**: EMA(20) counter-regime ×0.3. MDD 6.71% (기존 18.70%). |
| v1.35.2 | 02-26 | **Consecutive loss fix + tighter MDD sizing**: full_size_below_dd 5→3%, min_size_above_dd 20→15%. |
| **v1.35.1** | 02-25 | **Direction Cap 6→8**: 296d, 3-fold, cap8 유일한 3/3 PASS. |
| v1.35.0 | 02-25 | **ATR Scanner v2.2 + 51pat deploy**: 51패턴(16L+35S), WF 3/3 PASS, OOS +443.9%. |
| v1.34.0 | 02-24 | **Holdout + Clean Protocol v3.0 + MDD sizing + Trade history** |
| v1.33.0 | 02-23 | **35pat Compact TP/SL + 15 STRONG SHORT 복원** |
| v1.32.0 | 02-23 | **7 STRONG LONG 복원 + Direction Cap 6** |
| **v1.31.1** | 02-23 | **Optimal TP/SL (WR Excess maximizing grid search)** |
| v1.31.0 | 02-23 | **15-pattern WR Excess filter + T864 timeout** |
| v1.30.1 | 02-23 | **N=5→9 멀티포지션 확장** |
| **v1.30.0** | 02-22 | **Production Hedge 모드 전환**: Hedge N=5 +174.0%/MDD 29.6% vs FIFO +57.1%/MDD 58.8% |
| v1.29.4 | 02-22 | Hedge mode infrastructure |
| v1.29.3 | 02-22 | CLOSE_OLDEST immediate re-entry + emergency SL amount fix |
| v1.29.2 | 02-21 | adjust_tpsl_to_config dynamic per-pattern mode |
| v1.29.1 | 02-21 | Crash recovery per-pattern TP/SL preservation + per-slot fill detection |
| **v1.29.0** | 02-21 | **N=5 멀티포지션 (One-Way BOTH mode)**: 가상 슬롯 기반. |
| **v1.28.42** | 02-21 | **ATR-scaled TP/SL + proportional vol_mult cap** |
| v1.28.41 | 02-19 | Atomic write retry + .new fallback for OneDrive PermissionError |
| **v1.28.40** | 02-19 | **Deploy MAE/MFE 59 patterns to production** |
| v1.28.39 | 02-19 | Scanner MAE/MFE discovery method |
| v1.28.38 | 02-18 | Scanner fee calculation bug fix |
| v1.28.37 | 02-18 | Extract calculate_pnl() + test pure functions |
| v1.28.36 | 02-18 | Dynamic pattern confidence scoring |
| v1.28.35 | 02-18 | Fix edge metric — unify actual/expected definition |
| v1.28.34 | 02-18 | Untrack state.json from git — prevent state corruption |
| v1.28.33 | 02-18 | Dead constants cleanup |
| v1.28.32 | 02-18 | PnL=0 consistency + dead code removal + sync .bak fix |
| v1.28.31 | 02-18 | Code cleanup |
| v1.28.30 | 02-18 | 3 code quality fixes |
| v1.28.29 | 02-18 | Fix load_dynamic_patterns partial config mutation |
| v1.28.28 | 02-18 | Fix recovery NoneType crash |
| v1.28.27 | 02-18 | position_open sentinel + refill guard |
| v1.28.26 | 02-18 | Sentinel consistency + lock cleanup |
| v1.28.25 | 02-18 | 3 production hardening fixes |
| v1.28.24 | 02-18 | Scanner MAX_BARS 500→288 (24h timeout study) |
| v1.28.23 | 02-18 | Extend EXCHANGE_MANAGED to initial TP/SL placement |
| v1.28.22 | 02-18 | Fix TP/SL verify infinite retry loop |
| v1.28.21 | 02-18 | Silent except → debug logging + backup consistency |
| v1.28.20 | 02-18 | Utils cleanup + outdated docstring |
| v1.28.19 | 02-18 | Unused import cleanup |
| v1.28.18 | 02-18 | Dead code removal + import cleanup |
| **v1.28.17** | 02-18 | **State corruption resilience + data restoration** |
| v1.28.13 | 02-17 | Resilience improvements — 3 fixes |
| v1.28.12 | 02-17 | Safety fixes — 6 production bugs |
| v1.28.11 | 02-17 | SL<1.0% pattern removal |
| v1.28.10 | 02-17 | Safety patch — 4 critical trading logic gaps |
| v1.28.9 | 02-16 | Edge>=21.8pp + WR>=60% quality filter |
| v1.28.8 | 02-16 | Logging system improvement |
| v1.28.7 | 02-16 | Production code review + dual-direction bug fix |
| v1.28.6 | 02-16 | PP discovery scanner |
| v1.28.5 | 02-15 | Dynamic per-pattern TP/SL optimization |
| v1.28.4 | 02-15 | Statistical rigor filter |
| v1.28.3 | 02-14 | Statistical significance filter |
| v1.28.2 | 02-14 | WF frontier optimal TP |
| v1.28.1 | 02-14 | Fine grid TP + distance fix + min_trades filter |
| v1.28.0 | 02-12 | Static→Dynamic 프로덕션 전환 |
| v1.27.3 | 02-12 | Expectation reset + Dynamic WF Pattern Selection 인프라 |
| v1.27.2 | 02-12 | Low-WR pattern review |
| v1.27.1 | 02-12 | Legacy pattern re-optimization |
| v1.27.0 | 02-10 | Uniform TP 70% + 리스크 관리 |
| v1.26.4 | 02-09 | Full TP/SL optimization |
| v1.26.3 | 02-09 | R:R re-optimization |
| v1.26.2 | 02-09 | MC/edge cleanup |
| v1.26.1 | 02-08 | T5_Optimized 58패턴 |
| v1.26.0 | 02-08 | R:R>=0.75 포트폴리오 마이그레이션 |
| v1.25.6 | 02-08 | Opus 4.6 코드 리뷰 |
| v1.25.5 | 02-07 | CWD 의존 경로 버그 수정 |
| v1.25.4 | 02-07 | 품질 필터 적용 (WR >= 85%) |
| v1.25.3 | 02-07 | 전수 패턴 발굴 + 심층 가지치기 분석 |
| v1.25.2 | 02-07 | 전수검사 후 2패턴 제거 |
| v1.25.0 | 02-04 | Moderate-B-20 포트폴리오 |
| v1.24.0 | 02-04 | Ground Truth Classification 통일 |
| v1.23.0 | 02-02 | 안정성 강화 |
| v1.22.0 | 02-01 | DN-D-BD 제거 (과적합) |
| v1.21.1 | 02-01 | Leverage side fix + state cleanup |
| v1.21.0 | 02-01 | Conservative Per-Pattern TP/SL |
| v1.20.1 | 02-01 | Improved early-bar classification |
| v1.20.0 | 01-31 | Unified Classification Re-discovery |
| v1.19.x | 01-30 | Tight TP/SL, Uniform 1.0/1.0, 21-Pattern |
| v1.18.x | 01-27~30 | Regime-Adaptive Strategy |
| v1.17 | 01-26 | Statistical Validation + TP/SL Auto-Adjust |
| v1.13~16 | 01-25~26 | Early Exit, Context Filters, Pattern Discovery |
| v1.0~12 | 01-22~25 | 초기 릴리스~개선 |

---

## 연구 히스토리

| 날짜 | 연구 | 결과 |
|------|------|------|
| 03-12 | Code Audit 전수점검 (57건 검출, 교차검증) | **7건 실제 수정**, 4건 FALSE POSITIVE, 5건 severity 하향 |
| 03-11 | Timeout 교차검증 6-phase | 독립 효과 확인, Cascade 무관 (+193.7%), slot liberation 2124건 |
| 03-11 | Mechanism 교차검증 (15-seed) | 전부 NON-DISC, AggRisk/DirCap은 IS 감소시키나 live risk guard 역할 |
| 03-05 | 5개 파라미터 Sweep (SL Cooldown, Sizing, ATR clamp resweep, Time-of-day, MDD Sizing) | **ALL KEEP baseline** — 최적화 공간 소진 |
| 03-04 | Entry Optimization (h7_critical_validation) | **ROLLBACK** — WF 94% PASS rate (비판별), 95% Cascade 의존 |
| 03-04 | Strategy Foundation Critical Study | WF 100% non-discriminating (30 random all PASS), 패턴 기여 32.6% |
| 03-04 | AggRisk Cascade Cross-Validation | Genuine effect — 랜덤 0/15 PASS, Cascade-OFF 독립 개선 |
| 02-23 | ATR-Scaled Backtest Study | Scanner ATR scaling → 리스크 조정 성과 +57.5% |
