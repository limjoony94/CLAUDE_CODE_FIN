# Phase 1-6 Review Report

**Date:** 2026-02-02  
**Commits:** 8492f8c → dd9683d (6 commits)

---

## Phase 1: Quick Wins (8492f8c) ✅

| Item | Status | Notes |
|------|--------|-------|
| Makefile `pythonw` → `python3` | ✅ | 모든 타겟이 `python3` 사용 |
| exchange.py docstring "Pattern 5m Bot" | ✅ | `Pattern 5m Bot - Exchange Interface` |
| config.py docstring "Pattern 5m Bot" | ✅ | `Pattern 5m Bot - Configuration Management` |
| constants.py DEPRECATED 주석 | ✅ | 3곳에 DEPRECATED 주석 (line 60, 67, 93) |
| setup.sh venv 로직 | ✅ | `.venv` 생성, pip install, pytest 실행까지 포함 |

## Phase 2: Requirements Split (5a0b07b) ✅

| Item | Status | Notes |
|------|--------|-------|
| runtime.txt | ✅ | numpy, pandas, scipy, ta, ccxt, pyyaml 등 핵심만 |
| dev.txt | ✅ | `-r runtime.txt` 포함, pytest/matplotlib/plotly 추가 |
| legacy.txt | ✅ | torch, stable-baselines3, gymnasium 격리 |
| torch 없음 (runtime) | ✅ | torch는 legacy.txt에만 존재 |
| requirements.txt → runtime.txt 참조 | ✅ | `-r requirements/runtime.txt` |

## Phase 3: Classification 통일 (02d7e46) ⚠️

| Item | Status | Notes |
|------|--------|-------|
| full_270d_revalidation.py canonical import | ⚠️ **미완** | TODO 주석만 있고 실제로는 인라인 분류 로직 그대로 사용 중 |
| sys.path 설정 | ✅ | line 12에 올바르게 설정 |
| docs/classification-unification.md 참조 | ✅ | TODO에서 문서 참조 |

**이슈:** `full_270d_revalidation.py`는 canonical `classify_candle()`을 실제로 import하지 않음. TODO 주석만 추가됨. 인라인 벡터화 분류 로직이 그대로 남아있어 이중 구현 상태.

## Phase 4: 테스트 확대 (9cbdbbf) ❌

| Item | Status | Notes |
|------|--------|-------|
| test_state.py 존재 | ✅ | `scripts/production/pattern_5m/tests/` |
| test_bot_logic.py 존재 | ✅ | 위와 동일 경로 |
| test_config.py 존재 | ✅ | 위와 동일 경로 |
| pytest 전체 통과 | ❌ **실패** | 6개 테스트 모두 `ModuleNotFoundError: No module named 'bingx_rl_trading_bot'` |

**이슈:** 테스트가 `from bingx_rl_trading_bot.scripts.production.pattern_5m.*` 형태로 import하나, 패키지가 설치되지 않은 상태에서는 모듈을 찾을 수 없음. `conftest.py`에서 sys.path 설정이 필요하거나, 상대 import로 변경 필요.

## Phase 5: Context Filter (26c0040) ✅

| Item | Status | Notes |
|------|--------|-------|
| context_filter_research.py | ✅ | RSI bins, ATR percentile bins, trend/slope 분석 포함 |
| signals.py context filter | ✅ | `check_context_filter()` 구현 (line 231), CONTEXT_FILTER_ENABLED 토글 |
| 패턴별 필터 설정 | ✅ | PATTERN_CONTEXT_FILTERS dict, required/preferred 분리 |

## Phase 6: 모니터링 (dd9683d) ⚠️

| Item | Status | Notes |
|------|--------|-------|
| alert_check.py 존재 | ✅ | `scripts/monitor/alert_check.py` |
| 연속손실 ≥ 5 | ✅ | `max_consecutive_losses: 5` |
| MDD 체크 | ⚠️ | threshold=10% (요구사항 25%와 다름) |
| WR 체크 | ⚠️ | threshold=30% (요구사항 65%와 다름) |
| stale state | ✅ | 30분 기준 |
| exit code | ✅ | CRITICAL → exit(1), 정상 → exit(0) |
| 크로스플랫폼 | ✅ | Windows/Unix 분기 처리 |

**이슈:** MDD threshold(10% vs 25%)와 WR threshold(30% vs 65%)가 요구사항과 다름. 의도적 조정일 수 있으나 확인 필요.

---

## 종합 평가

### 잘된 점
- Phase 1, 2는 깔끔하게 완료. requirements 분리가 잘 됨
- Phase 5 context filter는 연구 스크립트와 프로덕션 코드 모두 충실히 구현
- 모니터링 스크립트가 크로스플랫폼 대응

### 🔴 Critical Issues
1. **테스트 전체 실패** — `ModuleNotFoundError`로 6개 테스트 모두 실행 불가. conftest.py에 sys.path 추가 또는 pyproject.toml 패키지 설정 필요
2. **Classification 미통일** — full_270d_revalidation.py가 canonical import를 하지 않고 TODO만 남김. Phase 3의 핵심 목표 미달성

### 🟡 Minor Issues
3. alert_check.py threshold 값이 요구사항과 불일치 (MDD 10% vs 25%, WR 30% vs 65%)
4. `_check_context_filters()` 함수명이 `check_context_filter()`로 되어있음 (언더스코어 prefix 없음 = private이 아닌 public)

### 추천 후속 작업
1. `conftest.py` 작성 → sys.path에 프로젝트 루트 추가하여 테스트 통과시키기
2. `full_270d_revalidation.py`에서 실제로 canonical classify_candle import 적용
3. alert threshold 값 재검토 및 config.yaml과 연동
