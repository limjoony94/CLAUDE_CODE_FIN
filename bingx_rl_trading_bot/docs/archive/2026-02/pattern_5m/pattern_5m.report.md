# PDCA Completion Report: pattern_5m v1.34.0

> Date: 2026-02-24 | Match Rate: 100% | Tests: 1067 passed

---

## 1. Executive Summary

v1.34.0은 3가지 방어 메커니즘 + BH FDR 통계 수정 + Clean Protocol을 구현하여, 라이브 봇의 장기 건전성을 강화한 릴리스입니다.

**핵심 성과:**
- Scanner에 7일 holdout OOS 검증 추가 → 미래 스캔에서 과적합 패턴 자동 제거
- BH FDR 버그 수정 (m 파라미터) → 올바른 다중검정 보정
- Clean Protocol v3.0 → 사전등록 + 이론 기반 임계값 + BH 1차 필터
- MDD 동적 사이징 → 드로다운 시 자동 포지션 축소 (5%~20% DD → 100%~25% scale)
- 스캔 신선도 경고 → 90일 초과 시 자동 WARNING
- trade_history persistence → 로그 회전과 무관한 거래 이력 보존

**Clean scan 검증 결과:** 현행 production 35패턴 중 **34개(97%)가 BH FDR q=0.05 통과** — 패턴 선별의 통계적 건전성 확인됨.

---

## 2. Plan → Implementation Trace

### 2.1 계획된 변경사항

| # | 변경 영역 | 대상 파일 | 상태 |
|---|----------|----------|------|
| 1 | Scanner Holdout | `pattern_scanner.py` | DONE |
| 2 | Bot Staleness Check | `bot.py`, `config.yaml` | DONE |
| 3 | MDD Dynamic Sizing | `state.py`, `position_open.py`, `bot.py`, `config.yaml` | DONE |

### 2.2 추가 구현 (세션 중 발견)

| # | 변경 영역 | 대상 파일 | 발견 계기 |
|---|----------|----------|----------|
| 4 | BH FDR 버그 수정 | `pattern_scanner.py` L235 | 코드 리뷰 중 발견 |
| 5 | Clean Protocol v3.0 | `pattern_scanner.py` (--clean flag) | 데이터 오염 논의 |
| 6 | trade_history | `models.py`, `position_close.py` | 메트릭 영속성 개선 |
| 7 | BOT_VERSION bump | `constants.py` | 버전 관리 |

---

## 3. Technical Details

### 3.1 Scanner Holdout Validation

```
--holdout-days 7 (default)
Flow: df → IS/holdout split → scan(df_is) → holdout_validate(df_holdout) → WF → output
```

- **Holdout 기준**: WR Excess > 0 (패턴별 WR - Random Walk WR)
- **FAIL**: 제거 + WARNING 로그
- **SKIP** (거래 < 3): 유지 (7일에 대부분 패턴은 3거래 미만이 자연스러움)
- **출력**: JSON `holdout_validation` 섹션에 전체 결과 포함

### 3.2 BH FDR Bug Fix

```python
# Before (bug): m = len(sorted_items)  # only patterns passing pre-filters
# After (fix):  m = max(n_tested, len(sorted_items))  # total hypotheses tested
```

**Impact**: MC pre-filter(p<0.01)가 BH 이전에 패턴을 제거하면, BH가 보는 m이 실제보다 작아져 FDR 제어가 느슨해짐. 수정 후 m=1,326 (전체 가설 수) 사용으로 올바른 FDR 제어.

### 3.3 Clean Protocol v3.0

```
--clean flag 활성화 시:
  edge_threshold = 0.0    # BH가 통계적 유의성 판단
  mc_threshold = 1.0      # MC pre-filter 비활성
  correction = bh         # BH FDR 강제
  holdout_days = 7        # 기본 holdout
  Post-BH: edge >= 5pp (비용 기반) + SL >= 1.0% (실행 리스크)
```

- **사전등록 manifest** (`clean_scan_manifest.json`): 스캔 전 파라미터 기록
- **이론 기반 임계값**: 5pp = 수수료(0.30%) + 슬리피지(0.06%) + 안전마진

### 3.4 MDD Dynamic Sizing

```
DD < 5%  → scale = 1.0 (full size)
DD = 12% → scale = 0.65 (linear interpolation)
DD >= 20% → scale = 0.25 (minimum)
```

- `peak_equity`: state에 저장 (high watermark)
- 매 trading window에서 `update_peak_equity(state, equity)` 호출
- `get_position_size()` 내부에서 `per_slot_equity * mdd_scale` 적용

### 3.5 Scan Staleness Check

- `_check_scan_staleness(config)`: 봇 시작 시 `dynamic_patterns.json`의 `generated_at` 확인
- `rescan_interval_days: 90` (config)
- 초과 시 WARNING 로그, 봇 실행은 차단하지 않음

### 3.6 trade_history Persistence

- `PerformanceMetrics.trade_history`: list field, `to_dict()`/`from_dict()` 포함
- `record_closed_position()`: 매 청산 시 `trade_detail` dict 생성 → `metrics.update_trade(pnl, trade_detail=detail)`
- 로그 파일 회전과 무관하게 메트릭 JSON에 영구 보존

---

## 4. Clean Scan Validation Results

### 4.1 Pipeline

| 단계 | 수량 |
|------|------|
| 전체 가설 (n_tested) | 1,326 |
| BH FDR 통과 (q=0.05) | 367 (27.7%) |
| 중복 제거 | 271 |
| Holdout 7d FAIL 제거 | -11 |
| **최종** | **260 (84L + 176S)** |

### 4.2 현행 Production 패턴 검증

| 지표 | 값 |
|------|-----|
| Production 35패턴 중 Clean 통과 | **34 (97%)** |
| Clean 미통과 | 1 (ST-BD-BU_SHORT) |
| Clean-only (신규 발견) | 226 |

### 4.3 WF OOS

| Fold | OOS Trades | OOS WR | OOS PnL | OOS MDD |
|------|-----------|--------|---------|---------|
| 1 | 133 | 81.2% | +172.8% | 17.2% |
| 2 | 161 | 80.1% | +187.2% | 34.3% |
| 3 | 142 | 78.2% | +200.9% | 56.4% |
| **합계** | **436** | — | **+560.9%** | — |

**Verdict: 3/3 PASS** — 단, fold3 MDD 56.4%는 260패턴의 신호 과잉에 기인.

### 4.4 Production 배포 결정

**결정: 현행 35패턴 유지 (변경 불필요)**

근거:
1. 34/35가 Clean Protocol 통과 → 기존 선별이 이미 건전
2. 260패턴은 실용적 배포에 부적합 (63 signals/day vs N=9 slots = 선착순 랜덤)
3. fold3 MDD 56.4% 수용 불가
4. Compact TP/SL (v1.33.0)이 이미 최적 필터

---

## 5. Modified Files Summary

| 파일 | 변경 | LOC |
|------|------|-----|
| `scripts/scanner/pattern_scanner.py` | holdout + clean + BH fix | +250 |
| `scripts/production/pattern_5m/bot.py` | staleness + peak equity | +25 |
| `scripts/production/pattern_5m/state.py` | peak_equity + update fn | +10 |
| `scripts/production/pattern_5m/position_open.py` | MDD scale + state param | +25 |
| `scripts/production/pattern_5m/models.py` | trade_history field | +5 |
| `scripts/production/pattern_5m/position_close.py` | trade_detail recording | +15 |
| `scripts/production/pattern_5m/constants.py` | BOT_VERSION | +1 |
| `config/pattern_5m_config.yaml` | mdd_sizing + rescan | +8 |
| **Total** | **8 files** | **~339** |

---

## 6. Test Results

```
1067 passed in 31.00s
```

모든 기존 테스트 통과. v1.34.0 변경은 기존 동작에 영향 없음 (MDD sizing은 config에서 enable/disable 가능, staleness는 WARNING only).

---

## 7. Artifacts

| 파일 | 용도 |
|------|------|
| `results/dynamic_patterns_clean.json` | Clean scan v3.0 출력 (260 patterns) |
| `results/clean_scan_manifest.json` | 사전등록 manifest |
| `results/dynamic_patterns_35pat_compact_backup.json` | v1.33.0 패턴 백업 |
| `results/dynamic_patterns.json` | **Production** (변경 없음, 35 patterns) |

---

## 8. Key Learnings

1. **BH FDR m 파라미터가 핵심**: MC pre-filter가 BH에 전달되는 가설 수를 줄이면 FDR 제어가 무의미해짐. 항상 `m = total_hypotheses_tested`.

2. **데이터 오염은 연구자에 있다**: 동일 데이터에 15+ 실험을 반복하면, 데이터의 나이와 무관하게 임계값이 데이터에 적응함. 해결: 이론 기반 임계값 + 사전등록.

3. **통계적 유의성 ≠ 경제적 유의성**: BH FDR가 260패턴을 "유의"하다고 하지만, 실용적 배포(N=9 slots)에서는 35패턴이 최적.

4. **현행 패턴 선별은 건전**: 21.8pp edge + 60% WR + Compact TP/SL + WR Excess>5pp 필터가 BH FDR보다 더 엄격하고 실용적.

5. **Holdout 7일의 한계**: 대부분 패턴에 충분한 거래를 생성하지 못함 (77% SKIP). 향후 holdout 기간 연장 또는 다른 검증 방법 고려.

---

## 9. PDCA Cycle Summary

```
[Plan] ✅ → [Design] ✅ → [Do] ✅ → [Check] ✅ (100%) → [Report] ✅
```

| Phase | 날짜 | 결과 |
|-------|------|------|
| Plan | 2026-02-24 | 3개 방어 메커니즘 설계 |
| Design | 2026-02-24 | Plan mode 문서 (6개 파일, 상세 변경 명세) |
| Do | 2026-02-24 | 8개 파일 구현 + Clean Protocol 추가 |
| Check | 2026-02-24 | 24/24 항목 구현 (100% match) |
| Report | 2026-02-24 | 본 문서 |

---

*Generated: 2026-02-24 | v1.34.0 PDCA Complete*
