# Git Workflow - CLAUDE_CODE_FIN

**Last Updated**: 2026-03-12 | **Bot Version**: v1.56.2

---

## Git Workflow

### 1. Branch Strategy

현재 프로젝트는 **master 단일 브랜치**로 운영합니다.

```bash
# Main branch
master              # 프로덕션 코드 + 연구

# Feature branches (필요 시)
feature/v1.28.0-xxx # 대규모 변경 시
```

### 2. Commit Messages

**형식**: `feat(vX.XX.X): 간결한 설명`

```bash
# 버전 릴리스
git commit -m "feat(v1.56.2): Code Audit 7 fixes — Place-first SL, Emergency SL fallback, state persistence"

# TP/SL 최적화
git commit -m "feat(v1.53.0): Data 303d + Rescan 131pat (59L+72S), WF 3/3 PASS"

# 버그 수정
git commit -m "fix: cascade SL protection gap — Place-first/Cancel-after"

# 문서 업데이트
git commit -m "docs: update all docs to v1.56.2"

# 코드 리뷰
git commit -m "refactor(v1.25.6): code review — 5 critical bugs fixed"
```

**Commit Types**:
- `feat(vX.XX.X)`: 새 버전/기능
- `fix`: 버그 수정
- `refactor`: 코드 리팩터링
- `docs`: 문서 수정
- `chore`: 기타 (설정, 정리 등)

### 3. Version Numbering

```
v1.MAJOR.MINOR

MAJOR: 전략 변경 (패턴 세트, TP/SL 방법론)
MINOR: 세부 조정 (파라미터, 버그 수정)
```

현재: **v1.56.2** (131pat, ATR Scanner v2.4, Cascade SL, Code Audit)

---

## Code Review Standards

### Review Checklist
- ✅ **Look-Ahead Bias**: `shift(-1)`, `center=True` 없는지 확인
- ✅ **MC Validation**: p < 0.01 (sign randomization, 10k sims)
- ✅ **WF Validation**: 3/3 profitable folds (Expanding Window)
- ✅ **Edge Test**: Edge>=18pp, MC<0.01, min_trades>=25
- ✅ **Constants Update**: `constants.py` + `dynamic_patterns.json` 동기화
- ✅ **CLAUDE.md Update**: 버전 히스토리 추가 + `docs/VERSION_HISTORY.md`
- ✅ **State Compatibility**: 기존 포지션과 호환

---

## Documentation Standards

### Documentation Maintenance
```yaml
Core Docs (항상 최신 유지):
  - CLAUDE.md              # 전략/파라미터/버전
  - docs/VERSION_HISTORY.md # 전체 버전 히스토리
  - docs/agent-guides.md   # 에이전트 가이드
  - docs/INDEX.md          # 문서 목차

Research Docs (연구 완료 시 업데이트):
  - claudedocs/STANDARD_RESEARCH_PROTOCOL.md

Archive Docs (참고용):
  - archive/               # 레거시
  - docs/v1.25.0-review.md # 과거 버전 리뷰
```

### Documentation Principles
- **간결성**: 핵심만 담기
- **최신성**: 코드와 문서 동기화 (버전 릴리스마다)
- **계층성**: CLAUDE.md → docs/ → claudedocs/
- **실용성**: 실제 사용 가능한 명령/경로 제공
