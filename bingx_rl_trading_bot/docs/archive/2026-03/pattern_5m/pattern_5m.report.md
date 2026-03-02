# PDCA Completion Report: pattern_5m v1.38.0~v1.38.1

> Date: 2026-03-02 | Feature: N-Position Portfolio Simulator | Tests: 1061 passed

---

## 1. Executive Summary

v1.38.0~v1.38.1은 Scanner의 백테스트 환경을 프로덕션과 정합시키는 N-Position Portfolio Simulator를 구현하고, default로 전환한 릴리스입니다.

**핵심 문제**: Scanner는 1-pos additive 백테스트 → WF OOS WR 88.8%. 프로덕션은 N=9 compound + 6종 필터 → Live WR 52.7%. **36pp 괴리**의 주요 원인.

**핵심 성과:**
- Scanner에 `portfolio_npos()` 구현 — 프로덕션과 동일한 N=9, compound, direction cap, regime sizing, aggregate risk cap, momentum guard 시뮬레이션
- **Live WR gap 52% 감소**: 32.3pp → 15.4pp (1-pos WR 88.8% → N-pos WR 68.4% → Live 52.7%)
- WF 3/3 PASS 유지 (Fold1 +22.8%, Fold2 +2.9%, Fold3 +12.1%)
- 패턴 선별 변경 없음 (130패턴 동일) — WF 메트릭만 정직하게 교체
- v1.38.1에서 `--npos` default 전환, `--no-npos`로 legacy 모드 보존

---

## 2. Plan → Implementation Trace

### 2.1 계획된 변경사항 (Plan mode 문서 기반)

| # | 변경 영역 | 대상 파일 | 상태 |
|---|----------|----------|------|
| 1 | 상수 추가 (N_SLOTS, DIRECTION_CAP 등) | `pattern_scanner.py` | DONE |
| 2 | `compute_ema_slope()` 함수 | `pattern_scanner.py` | DONE |
| 3 | `_check_exit_npos()` 함수 | `pattern_scanner.py` | DONE |
| 4 | `portfolio_npos()` 함수 — 핵심 | `pattern_scanner.py` | DONE |
| 5 | `calc_stats_compound()` 함수 | `pattern_scanner.py` | DONE |
| 6 | WF OOS 평가 N-pos 옵션 | `pattern_scanner.py` | DONE |
| 7 | Full scan IS에 N-pos 요약 | `pattern_scanner.py` | DONE |
| 8 | CLI 옵션 (`--npos`, `--n-slots` 등) | `pattern_scanner.py` | DONE |
| 9 | Output JSON npos 메타데이터 | `pattern_scanner.py` | DONE |

### 2.2 추가 구현

| # | 변경 영역 | 발견 계기 |
|---|----------|----------|
| 10 | `--no-npos` flag (v1.38.1) | Default 전환 결정 |
| 11 | `npos_portfolio_study.py` (연구 스크립트) | 3-Phase 비교 연구 |
| 12 | `npos_fold2_diagnosis_study.py` (진단 스크립트) | Fold 2 +2.91% 진단 |
| 13 | 진단 스터디 데이터 슬라이싱 버그 수정 | Fold 2 원인 분석 중 발견 |

---

## 3. Technical Details

### 3.1 portfolio_npos() — 핵심 함수

**출처**: `entry_improvement_study.py:210` simulate_portfolio (21 시나리오 WF 검증 완료)

```
시그니처:
  portfolio_npos(signal_tuples, opens, highs, lows, closes, n_bars,
                 atr_ratio, ema_slope, start_bar, end_bar,
                 n_slots=9, direction_cap=7, ...)

입력: signal_tuples = [(signal_bar, pattern, direction, tp_pct, sl_pct), ...]

로직 (bar-by-bar):
  1. Exit: _check_exit_npos() — ATR-scaled TP/SL + slippage + timeout(DROP) + intrabar + fee
  2. Entry 검사: max_pos → dir_cap → dup_pat → momentum_guard → regime_sizing → agg_risk_cap
  3. Compound equity: equity += sum(bar_pnl × 1/N × equity)
  4. Force-close at end_bar

반환: (trades_list, stats_dict)
```

### 3.2 프로덕션 필터 구현 (6종)

| 필터 | 구현 | 파라미터 |
|------|------|----------|
| Direction Cap | max same-dir positions | `direction_cap=7` |
| Regime Sizing | EMA(20) slope, counter ×mult | `regime_mult=0.3` |
| Aggregate Risk Cap | 방향별 SL exposure 합산 | `counter=3.0, with=7.0` |
| Momentum Guard | BTC >threshold/lookback → 차단 | `1.0%/6bars/6bars cooldown` |
| Timeout | 864 bars → DROP | `timeout_bars=864` |
| Duplicate Pattern | 동일 패턴 중복 진입 차단 | 기본 활성 |

### 3.3 _check_exit_npos()

**출처**: `entry_improvement_study.py:433`

- ATR-scaled TP/SL (clamp [0.6, 1.7])
- Slippage buffer 0.02%
- Timeout 864 bars → DROP (PnL 불포함)
- Intrabar resolution: same-bar TP/SL → bar open 기준 distance 비교
- Fee: 0.05% × 2 × leverage(3) = 0.30%

### 3.4 calc_stats_compound()

Compound equity curve 기반 메트릭:
- `pnl`: percentage return on initial equity
- `mdd`: peak-to-trough drawdown (%)
- `pnl_mdd`: PnL / MDD ratio
- `max_corr_loss`: single-bar 최대 손실 (correlated loss 측정)

### 3.5 WF 통합

```
기존 (use_npos=False):
  IS: 개별 패턴 edge/MC → 선별
  OOS: portfolio_1pos() → additive PnL/MDD

신규 (use_npos=True, default since v1.38.1):
  IS: 개별 패턴 edge/MC → 선별 [변경 없음]
  OOS: portfolio_npos() → compound PnL/MDD + 필터 적용
```

IS 패턴 선별은 기존과 100% 동일. 포트폴리오 수준 OOS 평가만 N-pos로 교체.

### 3.6 CLI 옵션

```bash
# v1.38.1 default (npos 자동 활성)
python scripts/scanner/pattern_scanner.py --discovery-method mae_mfe --edge-threshold 18 --wf-folds 3 --holdout-days 7

# Legacy 1-pos 모드
python scripts/scanner/pattern_scanner.py --discovery-method mae_mfe --edge-threshold 18 --no-npos

# N-pos 커스텀 파라미터
python scripts/scanner/pattern_scanner.py --discovery-method mae_mfe --edge-threshold 18 \
  --n-slots 5 --direction-cap 4 --regime-mult 0.5 --no-momentum-guard
```

---

## 4. Validation Results

### 4.1 1-pos vs N-pos 비교

| 지표 | 1-pos (legacy) | N-pos (default) | 변동 |
|------|---------------|-----------------|------|
| IS WR | 95.0% | 73.2% | -21.8pp |
| IS PnL | +1,385% | +100.3% | -92.8% |
| IS MDD | 27.8% | 7.1% | -74.5% |
| WF OOS WR | 88.8% | 68.4% | -20.4pp |
| WF OOS PnL | +872.7% | +37.8% | -95.7% |
| **Live WR gap** | **36.1pp** | **15.4pp** | **-57.3%** |

### 4.2 WF OOS Folds (N-pos)

| Fold | IS Pat | OOS Trades | OOS WR | OOS PnL | OOS MDD | Blocked (agg_risk) |
|------|--------|-----------|--------|---------|---------|-------------------|
| 1 | 37 (28L+9S) | 231 | 71.4% | +22.81% | 6.1% | 556 |
| **2** | **93 (65L+28S)** | **262** | **65.3%** | **+2.91%** | **15.4%** | **1,209** |
| 3 | 95 (50L+45S) | 251 | 68.5% | +12.1% | 8.9% | 1,056 |

**Verdict: 3/3 PASS** | Total: 744 trades, +37.8% OOS PnL

### 4.3 Fold 2 +2.91% 분석

Fold 2는 가장 어려운 OOS 구간 (BTC 하락기):
- LONG WR 하락: 4 LONG SL 동시 피격 → correlated loss 발생
- `agg_risk` 차단이 1,209회로 가장 높음 (regime counter 노출 제한)
- 진단 스터디 (holdout 미적용 74,113bars): +13.35% — holdout 2,016bars 차이
- **+2.91%는 holdout 적용 기준 실제 production 수치** (버그 아님)

### 4.4 N-pos IS Portfolio Stats

| 지표 | 값 |
|------|-----|
| Trades | 798 |
| WR | 73.2% |
| PnL | +100.25% |
| MDD | 7.06% |
| PnL/MDD | 14.2x |
| Max corr loss | 4.0% |
| Max simultaneous | 9 |
| Corr events | 20 |

### 4.5 Fold 2 진단 스터디 (10 가설)

| 가설 | 파라미터 | WF | Min Fold PnL | 판정 |
|------|---------|-----|-------------|------|
| H0 Baseline | 3/7, 0.3 | 3/3 PASS | +13.35% | 안정 |
| **H1 Relaxed agg** | **5/10, 0.3** | **3/3 PASS** | **+15.79%** | **Best** |
| H2 4-fold | 3/7, 0.3 | 4/4 PASS | +6.24% | OK |
| H3 5-fold | 3/7, 0.3 | 4/5 FAIL | -0.03% | FAIL |
| H4 No regime | 3/7, none | 3/3 PASS | +4.81% | Weak |
| H5 Moderate | 4/8, 0.5 | 3/3 PASS | +13.85% | OK |
| **H6 No agg** | **none** | **2/3 FAIL** | **-0.16%** | **agg 필수** |
| H7 No momentum | 3/7, 0.3 | 3/3 PASS | +10.15% | OK |
| H8 No dir_cap | ∞/7, 0.3 | 3/3 PASS | +4.22% | Weak |
| **H9 Minimal** | **none** | **2/3 FAIL** | **-21.83%** | **필터 필수** |

**결론**: H6/H9 FAIL → aggregate risk cap은 필수. 현 파라미터(3/7) 유지.

---

## 5. Modified Files Summary

| 파일 | 변경 | LOC |
|------|------|-----|
| `scripts/scanner/pattern_scanner.py` | portfolio_npos + _check_exit_npos + calc_stats_compound + compute_ema_slope + WF 통합 + CLI | ~+350 |
| `scripts/analysis/npos_portfolio_study.py` | 3-Phase 비교 연구 스크립트 | ~300 (신규) |
| `scripts/analysis/npos_fold2_diagnosis_study.py` | 10가설 진단 연구 스크립트 | ~450 (신규) |
| `results/dynamic_patterns.json` | npos 메트릭 포함 rescan | 재생성 |
| `CLAUDE.md` | v1.38.1 features + CLI 예시 + version history | ~+30 |
| **Total** | **3 code files + 2 docs** | **~1,130** |

**Production 코드 변경 없음** — Scanner(오프라인 도구)만 변경.

---

## 6. Test Results

```
1061 passed (기존 테스트 전체 통과)
```

Scanner는 production test suite에 포함되지 않음 (오프라인 CLI). 검증은 full rescan + WF 결과 비교로 수행.

---

## 7. Artifacts

| 파일 | 용도 |
|------|------|
| `results/dynamic_patterns.json` | **Production** (130 patterns, npos WF 메트릭 포함) |
| `scripts/analysis/npos_portfolio_study.py` | 1-pos vs N-pos 3-Phase 비교 |
| `scripts/analysis/npos_fold2_diagnosis_study.py` | Fold 2 진단 (10가설) |
| `docs/03-analysis/pattern_5m.analysis.md` | Gap analysis (Match Rate ~35% → 해결) |

---

## 8. Key Learnings

1. **Scanner-Production 정합성이 Live 성과 예측의 핵심**: 1-pos additive 백테스트는 N=9 compound 현실과 근본적으로 다름. 이 갭이 WR 36pp 괴리의 주요 원인.

2. **Aggregate Risk Cap은 필수 필터**: H6(no-agg)과 H9(minimal) 모두 WF FAIL. Counter-regime 3%/with-regime 7% cap이 correlated loss를 제한하는 핵심 방어선.

3. **Holdout split이 WF 결과에 유의미한 영향**: 2,016bars(7d) holdout 유무로 Fold 2 PnL이 +2.91% vs +13.35%로 갈림. 연구 스크립트와 Scanner는 동일한 데이터 범위를 사용해야 비교 가능.

4. **정직한 메트릭이 리스크 관리에 유리**: OOS WR 88.8%라는 숫자는 운영자에게 과도한 기대를 심어줌. N-pos WR 68.4%가 Live 52.7%에 더 가까워 현실적 판단 가능.

5. **Entry filters는 패턴 제거가 아닌 진입 조건 강화**: N-pos는 패턴 목록(130개)을 변경하지 않고, 포트폴리오 수준에서 진입을 제한함으로써 WR을 현실화.

---

## 9. PDCA Cycle Summary

```
[Plan] ✅ → [Design] ✅ → [Do] ✅ → [Check] ✅ → [Report] ✅
```

| Phase | 날짜 | 결과 |
|-------|------|------|
| Plan | 2026-03-01 | Plan mode 문서 (9개 변경 항목, 레퍼런스 코드 출처 매핑) |
| Design | 2026-03-01 | Gap analysis (Scanner ↔ Production match rate ~35%) |
| Do | 2026-03-01~02 | portfolio_npos + WF 통합 + 연구 2편 + default 전환 |
| Check | 2026-03-02 | Full rescan WF 3/3 PASS, Fold 2 진단 완료 |
| Report | 2026-03-02 | 본 문서 |

---

## 10. Metrics Summary

| Before (v1.37.0) | After (v1.38.1) |
|-------------------|-----------------|
| Scanner WF OOS WR: 88.8% | Scanner WF OOS WR: **68.4%** |
| Scanner WF OOS PnL: +872.7% | Scanner WF OOS PnL: **+37.8%** |
| Live WR gap: 36.1pp | Live WR gap: **15.4pp** (-57%) |
| Scanner backtest: 1-pos additive | Scanner backtest: **N-pos compound + 6 filters** |
| `--npos`: 미존재 | `--npos`: **default ON**, `--no-npos` for legacy |

---

*Generated: 2026-03-02 | v1.38.0~v1.38.1 PDCA Complete*
