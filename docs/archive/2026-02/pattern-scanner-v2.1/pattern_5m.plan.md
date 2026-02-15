# Pattern Scanner v2.1 — 3가지 개선 계획

## Context

현재 `pattern_scanner.py`(v1.28.6)는 270일 데이터에서 Per-Pattern TP/SL grid search로 294패턴을 발굴하지만:
1. **Multiple testing correction 없음** — ~1,200개 패턴-방향 조합을 개별 MC 테스트하면서 다중비교 보정 없음 → Type I error 증가
2. **Walk-Forward 검증 없음** — 전체 기간 in-sample 결과만 제공, OOS 검증은 별도 스크립트 필요
3. **Grid search 속도** — PP 모드에서 ~1,200개 패턴 × 99 grid 조합 순차 실행 (~7-10분)

**수정 대상**: `bingx_rl_trading_bot/scripts/scanner/pattern_scanner.py` (단일 파일, 현재 613줄 → ~900줄 예상)

---

## 1. 필터링 강화

### 새 함수: `apply_multiple_testing_correction()`
- `mc_test()` (line 171) 뒤에 삽입
- BH FDR (step-up): p-value 오름차순 정렬 → rank `i`에 대해 threshold = `fdr_q * i / m` → 마지막 통과 지점까지 유지
- Bonferroni: threshold = `alpha / n_tested`
- 참조 구현: `statistical_rigor_study.py:325-339`

```python
def apply_multiple_testing_correction(selected, n_tested, method='none', fdr_q=0.05, alpha=0.01):
    # method: 'none' | 'bh' | 'bonferroni'
    # Returns: (filtered_selected, correction_meta)
```

### `scan_patterns()` / `scan_patterns_pp()` 수정
- `n_tested` 카운터 추가 (min_trades 이상 signal을 가진 패턴-방향 조합 수)
- MC 필터 후 `apply_multiple_testing_correction()` 호출 (correction_method != 'none'일 때)
- Portfolio MC를 gate로 추가 (`--require-portfolio-mc` 옵션)

### `grid_search_best()` 수정
- `max_baseline_wr` 파라미터 추가 (기존 모듈 상수 대신)

### 새 CLI 인수
```
--correction {none,bh,bonferroni}  (default: none — 하위호환)
--fdr-q FLOAT                      (default: 0.05)
--max-baseline-wr FLOAT            (default: 70.0)
--require-portfolio-mc              (flag)
```

### Output JSON 추가 필드 (selection_criteria 내)
```json
"correction_method": "bh",
"fdr_q": 0.05,
"n_tested": 1247,
"n_before_correction": 320,
"n_after_correction": 294,
"portfolio_mc_pass": true
```

---

## 2. Walk-Forward 검증 통합

### 사전 리팩토링: signal_index 외부화
- `main()`에서 `load_and_classify()` → `build_signal_index()` 호출
- `scan_patterns()` / `scan_patterns_pp()`에 `signal_index` 파라미터 추가 (None이면 내부 생성 — 하위호환)

### 새 함수: `scan_universe_range()`
- `universal_tpsl_study_v3.py:205` (`scan_universe()`) 참조
- signal_index에서 `[bar_start, bar_end)` 범위 signal만 필터
- Universal / Per-Pattern 모드 모두 지원
- `bt_signals()`는 전체 배열 사용 (exit은 OOS 범위 넘어가도 OK)

```python
def scan_universe_range(signal_index, opens, highs, lows, n_bars,
                        bar_start, bar_end, mode, uni_tp, uni_sl,
                        min_trades, edge_threshold, mc_threshold,
                        max_baseline_wr):
    # Returns: list of selected pattern dicts
```

### 새 함수: `expanding_window_wf()`
- `universal_tpsl_study_v3.py:271-335` 참조
- `n_folds+1` 등분 세그먼트로 분할
- Fold f: IS=[0, (f+1)×seg), OOS=[(f+1)×seg, (f+2)×seg)
- 각 fold에서 `scan_universe_range()`로 fresh 패턴 발굴 → OOS 백테스트
- 패턴 stability 추적 (Counter)

```python
def expanding_window_wf(signal_index, opens, highs, lows, n_bars,
                        n_folds, mode, uni_tp, uni_sl,
                        min_trades, edge_threshold, mc_threshold,
                        max_baseline_wr):
    # Returns: dict with folds, positive_folds, total_oos_pnl, stable_patterns
```

### 새 CLI 인수
```
--wf-folds INT  (default: 0 = 비활성, 3 = 일반적)
```

### Output JSON 추가 섹션 (wf_folds > 0일 때만)
```json
"walk_forward": {
  "n_folds": 3,
  "folds": [
    {"fold": 1, "is_bars": N, "oos_bars": M, "is_patterns": K,
     "oos_trades": T, "oos_wr": W, "oos_pnl": P, "oos_mdd": D, "oos_positive": true}
  ],
  "positive_folds": 2,
  "total_oos_pnl": 245.3,
  "total_oos_trades": 98,
  "stable_pattern_count": 15,
  "stable_patterns": ["BD-BD-U_LONG", ...]
}
```

---

## 3. 스캔 속도/효율성

### 병렬 Grid Search (`concurrent.futures.ProcessPoolExecutor`)
- 모듈 레벨 `_pp_worker()` 함수 (pickling 호환)
- 패턴-방향 조합 단위로 병렬화 (데이터는 공유, 패턴별 독립)
- Windows `if __name__ == '__main__'` guard 이미 존재 (line 611)
- `--concurrency 0`=auto(cpu_count, cap 8), `1`=순차(fallback), `N`=N workers

```python
def _pp_worker(args_tuple):
    # grid_search_best + bt_signals + edge/MC check
    # Returns: result dict or None
```

### Progress 표시
- `tqdm` import with `try/except ImportError` fallback
- 순차 모드: 메인 루프 wrap, 병렬 모드: `as_completed` wrap

### Timing 계측
- `import time` + 각 phase별 소요시간 로깅
- Output JSON에 `timing` 섹션 추가

```json
"timing": {
  "classify_sec": 12.3,
  "scan_sec": 85.6,
  "wf_sec": 120.4,
  "total_sec": 218.3
}
```

### 새 CLI 인수
```
--concurrency INT  (default: 0 = auto)
```

### 예상 성능
| Phase | 현재 | 개선 후 | 배수 |
|-------|------|---------|------|
| PP scan | ~7분 | ~1.5분 (8 workers) | 4-5x |
| WF 3-fold (PP) | N/A | ~5분 | 신규 |

---

## JSON 스키마 버전

- `version`: "2.0" → "2.1"
- 모든 새 필드는 추가만 (기존 필드 변경 없음)
- `config.py` (consumer)는 unknown key 무시 → 하위호환 보장

---

## 구현 순서

| Step | 내용 | 의존성 |
|------|------|--------|
| 1 | `import time` + timing 계측 + `--concurrency` CLI | 없음 |
| 2 | `--max-baseline-wr` CLI → `grid_search_best()` 파라미터화 | 없음 |
| 3 | tqdm import (try/except) + progress bar | 없음 |
| 4 | `apply_multiple_testing_correction()` 구현 | 없음 |
| 5 | correction을 `scan_patterns()`/`scan_patterns_pp()`에 통합 + CLI args | Step 4 |
| 6 | `_pp_worker()` + `scan_patterns_pp()` 병렬화 | Step 2 |
| 7 | `signal_index` 외부화 리팩토링 | 없음 |
| 8 | `scan_universe_range()` 구현 | Step 7 |
| 9 | `expanding_window_wf()` 구현 | Step 8 |
| 10 | WF를 `main()`에 통합 + `--wf-folds` CLI | Step 9 |
| 11 | `build_output_json()` 업데이트 (correction, wf, timing) | Step 5,10 |
| 12 | 회귀 테스트 (baseline 비교, 병렬=순차 비교) | 전체 |

---

## 검증 계획

```bash
# 1. Baseline 회귀 (현재와 동일한 출력 확인)
python scripts/scanner/pattern_scanner.py --correction none --wf-folds 0 --concurrency 1 \
  --output results/test_baseline.json

# 2. BH correction
python scripts/scanner/pattern_scanner.py --correction bh --output results/test_bh.json

# 3. 병렬 vs 순차 동일성
python scripts/scanner/pattern_scanner.py --concurrency 1 --output results/test_seq.json
python scripts/scanner/pattern_scanner.py --concurrency 4 --output results/test_par.json
# → patterns 목록 비교 (정렬 후 동일해야 함)

# 4. WF 검증
python scripts/scanner/pattern_scanner.py --wf-folds 3 --output results/test_wf.json
# → folds의 IS/OOS 범위가 겹치지 않는지, IS가 expanding인지 확인

# 5. Bot 호환성
# config에서 pattern_source: dynamic으로 새 JSON 로드 확인
```

## 리스크

| 리스크 | 심각도 | 대응 |
|--------|--------|------|
| Windows multiprocessing pickling 실패 | 중 | `_pp_worker()`를 모듈 레벨 함수로 정의. `--concurrency 1` fallback |
| BH correction으로 패턴 수 급감 | 중 | `--correction none`이 기본값. 사용자 선택 |
| WF + PP 모드 실행 시간 과다 | 중 | WF는 opt-in (`--wf-folds 0` 기본), 로그에 예상 시간 표시 |
| tqdm 미설치 | 저 | `try/except ImportError` graceful degradation |
