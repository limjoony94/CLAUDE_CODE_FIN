# 문서 목차 (INDEX)

> **Updated**: 2026-02-12 | **Bot Version**: v1.27.3

---

## 프로젝트 루트

| 문서 | 설명 |
|------|------|
| [CLAUDE.md](../CLAUDE.md) | **프로젝트 핵심 문서** — 전략, 파라미터, 51패턴, 버전 히스토리 |

## docs/

| 문서 | 설명 |
|------|------|
| [agent-guides.md](agent-guides.md) | dev/automation/monitor 에이전트별 상세 가이드 |
| [CODING_CONVENTIONS.md](CODING_CONVENTIONS.md) | 코딩 컨벤션 (PEP 8, 모듈 구조) |
| [GIT_WORKFLOW.md](GIT_WORKFLOW.md) | Git 워크플로 및 커밋 컨벤션 |
| [TECH_STACK.md](TECH_STACK.md) | 기술 스택 (Python, CCXT, pandas 등) |
| [phase-review.md](../bingx_rl_trading_bot/docs/phase-review.md) | *(아카이브)* Phase 1-6 리뷰 (v1.23.0 시점) |
| [v1.23.0-review.md](v1.23.0-review.md) | *(아카이브)* v1.23.0 정밀 검증 리포트 |
| [v1.25.0-review.md](v1.25.0-review.md) | *(아카이브)* v1.25.0 Moderate-B-20 리뷰 — v1.26.x/v1.27.0으로 대체됨 |

## claudedocs/ (활성 연구 문서)

| 문서 | 설명 |
|------|------|
| [STANDARD_RESEARCH_PROTOCOL.md](../bingx_rl_trading_bot/claudedocs/STANDARD_RESEARCH_PROTOCOL.md) | 연구 프로토콜 표준 (백테스트, MC, WF) |
| [PRODUCTION_TRADING_LOGIC_ANALYSIS_20260204.md](../bingx_rl_trading_bot/claudedocs/PRODUCTION_TRADING_LOGIC_ANALYSIS_20260204.md) | 프로덕션 트레이딩 로직 분석 |

## claudedocs/ (과거 연구 리포트 — 시점 스냅샷)

| 문서 | 설명 |
|------|------|
| PRODUCTION_VALIDATION_REPORT_20260126.md | v1.17 프로덕션 검증 |
| STRATEGY_FULL_AUDIT_CORRECTED_20260119.md | 전략 전수 감사 |
| CONTEXT_FILTER_EXTENSION_RESEARCH_20260124.md | Context Filter 연구 (v1.14 시점) |
| EARLY_EXIT_SIGNAL_RESEARCH_20260123.md | Early Exit 연구 |
| CONFIDENCE_BASED_ENTRY_RESEARCH_20260123.md | Confidence 기반 진입 연구 |
| MTF_CONFIRMATION_RESEARCH_20260123.md | MTF Confirmation 연구 |
| DYNAMIC_TPSL_EVALUATION_20260123.md | 동적 TP/SL 평가 |

## scripts/ READMEs

| 문서 | 설명 |
|------|------|
| scripts/production/README.md | 프로덕션 봇 실행 가이드 |
| scripts/production/pattern_5m/README.md | 14개 모듈 패키지 구조 |
| scripts/analysis/README.md | 연구/분석 스크립트 가이드 (45+) |
| scripts/scanner/pattern_scanner.py | Dynamic WF Pattern Scanner CLI (Universal TP/SL) |
| scripts/data/README.md | 데이터 수집 스크립트 가이드 |
| scripts/utils/README.md | 유틸리티 스크립트 가이드 |

## docs/archive/ (PDCA 문서)

| 위치 | 설명 |
|------|------|
| `docs/archive/2026-02/` | 2026-02 bkit PDCA 문서 (pattern_5m feature) |

## archive/ (레거시)

| 위치 | 설명 |
|------|------|
| `archive/legacy_claudedocs_2025/` | 2025년 분석/리뷰 문서 |
| `archive/legacy_results/` | 2025년 백테스트 결과 |
| `archive/legacy_experimental/` | R&D 실험 |
| `archive/legacy_ml_data/` | ML 파이프라인 데이터 |
| `archive/deprecated_production/` | 폐기된 봇 (engulf_5m, adx 등) |
| `archive/deprecated/` | 폐기 문서 (TRADING_APPROACH_ANALYSIS, README_MONITORING, PROJECT_STRUCTURE) |
