# Technical Debt Refactoring Plan

**Date**: 2025-10-10
**Status**: Ready for execution
**Priority**: HIGH

---

## 🎯 Objectives

1. ✅ 체계적인 프로젝트 구조 확립
2. ✅ 중복 코드 제거 및 통합
3. ✅ 유지보수성 향상
4. ✅ 신규 개발자 onboarding 용이성

---

## 📊 Current Issues (from Analysis)

### 1. Scripts Directory Bloat (🚨 CRITICAL)
- **Problem**: 72 files in flat structure
- **Impact**:
  - 파일 찾기 어려움
  - 어떤 파일이 production용인지 불명확
  - 신규 개발자 혼란
- **Priority**: **P0 - Immediate**

### 2. Versioned Files (⚠️ HIGH)
- **Problem**: 10 sets of versioned files
  - `trading_env`: v2, v3, v4, v5, v6
  - `train_xgboost`: 8 variations
  - `train`: v2, v3, final
- **Impact**:
  - 어떤 버전이 최신/최선인지 불명확
  - 코드 중복
  - 유지보수 부담
- **Priority**: **P1 - High**

### 3. Duplicate Modules (⚠️ HIGH)
- **Problem**: Multiple similar modules
  - `xgboost_trader`: 4 variations
  - Large files (8 files > 500 lines)
  - Long functions (10 functions > 100 lines)
- **Priority**: **P2 - Medium**

### 4. Code Quality (⚠️ MEDIUM)
- Long functions > 100 lines
- Complex files with > 20 functions
- **Priority**: **P3 - Medium**

---

## 🚀 Refactoring Phases

### Phase 1: Scripts Directory Reorganization (P0)

**Goal**: Organize 72 scripts into logical subdirectories

**New Structure**:
```
scripts/
├── production/          # Production-ready scripts (최종)
│   ├── train_xgboost_v3_phase2.py  # 최신 훈련 스크립트
│   ├── backtest_hybrid_v4.py        # 최신 백테스트
│   └── technical_strategy.py        # Production 전략
│
├── analysis/            # 분석 스크립트
│   ├── analyze_technical_debt.py
│   ├── critical_analysis_ultra5.py
│   ├── analyze_all_configs.py
│   └── market_regime_analysis.py
│
├── experiments/         # 실험적 스크립트
│   ├── train_xgboost_*.py
│   ├── optimize_*.py
│   └── test_*.py
│
├── data/                # 데이터 수집
│   ├── collect_data.py
│   ├── collect_max_data.py
│   └── create_15min_data.py
│
├── deprecated/          # 더 이상 사용하지 않는 파일
│   ├── trading_env_v2.py
│   ├── train_v2.py
│   └── (old experiment files)
│
└── utils/               # 유틸리티
    ├── debug_*.py
    └── test_connection.py
```

**Action Items**:
1. ✅ Create subdirectories
2. ✅ Identify which files are production-ready
3. ✅ Move files to appropriate directories
4. ✅ Update import paths if needed
5. ✅ Create README.md in each subdirectory

**Effort**: 2-3 hours
**Risk**: Low (file moves, no code changes)

---

### Phase 2: Version Consolidation (P1)

**Goal**: Consolidate versioned files to single canonical version

#### 2.1 Trading Environment Consolidation

**Current**: `trading_env.py`, `trading_env_v2~v6.py` (6 files)

**Decision Criteria**:
- Which version is most recent?
- Which version is most stable?
- Which version has best features?

**Action**:
1. ✅ Identify best version (likely v6 or latest)
2. ✅ Rename to `trading_env.py` (canonical)
3. ✅ Move old versions to `deprecated/`
4. ✅ Update all imports

**Effort**: 1 hour

#### 2.2 XGBoost Training Consolidation

**Current**: 8 variations of `train_xgboost_*.py`

**Decision**:
- Keep: `train_xgboost_improved_v3_phase2.py` (최신, Phase 3에서 사용)
- Archive: 나머지 7개

**Action**:
1. ✅ Rename to `train_xgboost.py` (canonical)
2. ✅ Move old variations to `experiments/` or `deprecated/`
3. ✅ Document which config to use

**Effort**: 30 minutes

#### 2.3 XGBoost Trader Models

**Current**: `xgboost_trader.py`, `_fixed`, `_improved`, `_regression` (4 files)

**Decision**:
- Identify which model is actually used in production
- Consolidate features from all into one canonical version

**Effort**: 1-2 hours

---

### Phase 3: Code Quality Improvements (P2)

#### 3.1 Break Down Large Files (>500 lines)

**Targets**:
- `paper_trading_bot.py`: 641 lines
- `xgboost_trader.py`: 541 lines

**Action**:
- Extract classes/functions into separate modules
- Create logical separation (e.g., `trading/`, `models/`)

**Effort**: 3-4 hours

#### 3.2 Refactor Long Functions (>100 lines)

**Targets**:
- `test_lstm_thresholds.py::main()`: 311 lines
- `critical_reanalysis_with_risk_metrics.py::main()`: 287 lines

**Action**:
- Break down into smaller functions
- Extract repeated logic into utilities

**Effort**: 2-3 hours

---

### Phase 4: Testing & Documentation (P3)

**Goal**: Improve test coverage and documentation

**Actions**:
1. ✅ Add README.md to each subdirectory
2. ✅ Document which scripts are production-ready
3. ✅ Add docstrings to key functions
4. ✅ Create usage examples

**Effort**: 2-3 hours

---

## 📅 Execution Timeline

**Total Effort**: 12-16 hours
**Recommended Approach**: Incremental, phase-by-phase

### Week 1:
- **Day 1**: Phase 1 (Scripts reorganization) - **Immediate**
- **Day 2**: Phase 2.1-2.2 (Version consolidation)
- **Day 3**: Phase 2.3 (Model consolidation)

### Week 2:
- **Day 4**: Phase 3.1 (Large files)
- **Day 5**: Phase 3.2 (Long functions)
- **Day 6**: Phase 4 (Documentation)
- **Day 7**: Testing & validation

---

## ⚠️ Risks & Mitigation

### Risk 1: Breaking imports
**Mitigation**:
- Update imports systematically
- Test after each move
- Use search/replace for bulk updates

### Risk 2: Losing working code
**Mitigation**:
- Don't delete anything, only move to `deprecated/`
- Git commit after each phase
- Can always revert

### Risk 3: Time overrun
**Mitigation**:
- Phase 1 is highest priority (biggest impact)
- Can skip Phase 3-4 if time-constrained
- Phase 1 alone provides significant value

---

## ✅ Success Criteria

1. ✅ Scripts directory organized into logical subdirectories
2. ✅ Clear separation: production / experiments / deprecated
3. ✅ Single canonical version for each module
4. ✅ No files > 500 lines (unless necessary)
5. ✅ No functions > 100 lines
6. ✅ README.md documenting structure

---

## 🎯 Expected Benefits

### Immediate (Phase 1):
- ✅ 파일 찾기 쉬워짐
- ✅ Production vs Experiment 명확히 구분
- ✅ 신규 개발자 onboarding 시간 50% 감소

### Medium-term (Phase 2-3):
- ✅ 코드 중복 제거 → 유지보수 용이
- ✅ 버그 수정 시 한 곳만 수정
- ✅ 코드 품질 향상

### Long-term (Phase 4):
- ✅ 지속 가능한 개발
- ✅ Technical debt 감소
- ✅ 새로운 기능 추가 속도 향상

---

## 📋 Detailed Action Checklist

### Phase 1: Scripts Reorganization
- [ ] Create subdirectories: production/, analysis/, experiments/, data/, deprecated/, utils/
- [ ] Categorize all 72 files
- [ ] Move files to appropriate directories
- [ ] Update import paths
- [ ] Create README.md in each subdirectory
- [ ] Test imports work correctly

### Phase 2: Version Consolidation
- [ ] Identify latest trading_env version
- [ ] Consolidate to canonical version
- [ ] Identify latest train_xgboost version
- [ ] Consolidate xgboost models
- [ ] Update all references
- [ ] Move old versions to deprecated/

### Phase 3: Code Quality
- [ ] Break down large files
- [ ] Refactor long functions
- [ ] Extract common utilities
- [ ] Improve naming consistency

### Phase 4: Documentation
- [ ] Write README for production/
- [ ] Document key scripts
- [ ] Add usage examples
- [ ] Update main README.md

---

**Next Step**: Start with Phase 1 - Scripts Directory Reorganization (highest impact, lowest risk)
