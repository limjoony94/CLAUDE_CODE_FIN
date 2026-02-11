# Git Workflow - CLAUDE_CODE_FIN

**Last Updated**: 2026-02-11 | **Bot Version**: v1.27.0

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
git commit -m "feat(v1.27.0): Uniform TP 70% + risk management — WR 83.7%, MDD 16.2%"

# TP/SL 최적화
git commit -m "feat(v1.26.4): full TP/SL optimization — grid search + 5-phase deep validation"

# 버그 수정
git commit -m "fix: fd double-close in state save"

# 문서 업데이트
git commit -m "docs: update all docs to v1.27.0"

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

현재: **v1.27.0** (Uniform TP 70% + Risk Management)

---

## Code Review Standards

### Review Checklist
- ✅ **Look-Ahead Bias**: `shift(-1)`, `center=True` 없는지 확인
- ✅ **MC Validation**: p < 0.01 (sign randomization, 10k sims)
- ✅ **WF Validation**: ≥ 4/5 profitable folds
- ✅ **Edge Test**: 랜덤 baseline 대비 유의미한 edge
- ✅ **Constants Update**: `PATTERN_OPTIMAL_TPSL`, `PATTERN_STATS` 동기화
- ✅ **CLAUDE.md Update**: 버전 히스토리 추가
- ✅ **State Compatibility**: 기존 포지션과 호환

---

## Documentation Standards

### Documentation Maintenance
```yaml
Core Docs (항상 최신 유지):
  - CLAUDE.md              # 전략/파라미터/버전
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
