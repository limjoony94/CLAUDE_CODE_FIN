# Git Workflow & Collaboration - CLAUDE_CODE_FIN

**Last Updated**: 2025-10-14

---

## Git Workflow

### 1. Branch Strategy
```bash
# Main branch: 항상 안정적인 코드 유지
main

# Feature branches: 기능 개발
feature/lstm-model
feature/risk-management-v2

# Experiment branches: 실험 및 테스트
experiment/lag-features
experiment/regime-detection

# Hotfix branches: 긴급 버그 수정
hotfix/position-sizing-bug
```

### 2. Commit Messages (한국어 or 영어)
```bash
# 형식: [타입] 간결한 제목
#
# 상세 설명 (필요시)

# 예시:
git commit -m "[Feature] LSTM 모델 기본 구조 구현

- LSTM 레이어 3개 (128 → 64 → 32)
- Dropout 0.2 적용
- 입력: 60-step sequence
- 출력: 매매 확률 (0-1)"

git commit -m "[Fix] 백테스팅 HOLD 액션 버그 수정

- HOLD 시 포지션 유지하도록 변경
- 기존: 포지션 청산 (버그)
- 수정: 포지션 유지 (정상)"

git commit -m "[Refactor] API 클라이언트 CCXT 기반으로 재구현"

git commit -m "[Docs] QUICK_START_GUIDE.md 업데이트"

git commit -m "[Test] XGBoost 모델 단위 테스트 추가"
```

**Commit Types**:
- `[Feature]`: 새 기능
- `[Fix]`: 버그 수정
- `[Refactor]`: 코드 리팩터링
- `[Docs]`: 문서 수정
- `[Test]`: 테스트 추가/수정
- `[Chore]`: 기타 (빌드, 설정 등)

### 3. Pull Request Guidelines
```markdown
## Summary
간결한 변경 사항 요약

## Changes
- 변경 내용 1
- 변경 내용 2
- 변경 내용 3

## Test Results
- 백테스팅 결과: +7.2% per 5 days
- 단위 테스트: 15/15 passed
- 통계 검증: p<0.001, n=29

## Impact
- 기존 코드에 대한 영향
- Breaking changes 여부
- 필요한 추가 작업

## Checklist
- [ ] 코드 리뷰 완료
- [ ] 테스트 통과
- [ ] 문서 업데이트
- [ ] 변경 사항 CHANGELOG.md에 기록
```

---

## Code Review Standards

### 1. Review Checklist
- ✅ **Functionality**: 의도한 대로 동작하는가?
- ✅ **Code Quality**: 가독성, 유지보수성 확보?
- ✅ **Performance**: 성능 저하 없는가?
- ✅ **Security**: 보안 취약점 없는가?
- ✅ **Testing**: 충분한 테스트 커버리지?
- ✅ **Documentation**: 문서화 적절한가?
- ✅ **Conventions**: 코딩 컨벤션 준수?

### 2. Review Comments Style
```python
# ✅ Good: 구체적이고 건설적인 피드백
"""
이 부분에서 division by zero 가능성이 있습니다:
`sharpe = returns / std`

다음과 같이 수정하는 것이 안전합니다:
`sharpe = returns / (std + 1e-8)`
"""

# ❌ Bad: 모호하고 비판적인 피드백
"""
이 코드 이상해 보이는데요.
"""
```

---

## Documentation Standards

### 1. README.md Structure
```markdown
# Project Title

## Quick Start
빠르게 시작하는 방법

## Features
주요 기능 목록

## Installation
설치 방법

## Usage
사용 방법 및 예시

## Configuration
설정 방법

## Architecture
시스템 아키텍처 설명

## Development
개발 가이드

## Testing
테스트 실행 방법

## Troubleshooting
자주 발생하는 문제 및 해결

## License
라이선스 정보
```

### 2. Documentation Principles
- **간결성**: 핵심만 담기
- **최신성**: 코드와 문서 동기화
- **계층성**: README → QUICK_START → 상세 문서
- **실용성**: 실제 사용 가능한 예시 제공

### 3. Documentation Maintenance
```yaml
Core Docs (항상 최신 유지):
  - README.md
  - QUICK_START_GUIDE.md
  - PROJECT_STATUS.md
  - SYSTEM_STATUS.md
  - claudedocs/ (3-6개 핵심 문서)

Archive Docs (참고용):
  - archive/ (과거 분석, 실험 기록)
  - 프로젝트 종료 시 정리
```

---

## Testing Standards

### 1. Unit Tests
```python
# tests/test_indicators.py
import pytest
from src.indicators.technical import calculate_rsi

def test_rsi_calculation():
    """RSI 계산 정확도 테스트"""
    prices = [100, 102, 101, 105, 110]
    rsi = calculate_rsi(prices, period=4)

    assert 0 <= rsi <= 100, "RSI는 0-100 범위여야 함"
    assert rsi > 50, "상승장에서 RSI > 50"

def test_rsi_edge_cases():
    """RSI 엣지 케이스 테스트"""
    # 동일한 가격
    prices = [100] * 10
    rsi = calculate_rsi(prices)
    assert rsi == 50, "변동 없을 때 RSI=50"

    # 불충분한 데이터
    prices = [100, 102]
    with pytest.raises(ValueError):
        calculate_rsi(prices, period=14)
```

### 2. Integration Tests
```python
# tests/integration/test_trading_bot.py
def test_full_trading_cycle():
    """전체 트레이딩 사이클 통합 테스트"""
    bot = TradingBot(testnet=True)

    # 1. 데이터 수집
    data = bot.fetch_market_data()
    assert len(data) > 0

    # 2. 신호 생성
    signal = bot.generate_signal(data)
    assert -1 <= signal <= 1

    # 3. 주문 실행 (테스트넷)
    if abs(signal) > 0.7:
        result = bot.execute_order(signal)
        assert result['status'] == 'success'
```

### 3. Test Coverage Goals
- **Critical modules**: ≥90% (api, models, risk)
- **Core logic**: ≥80% (trading, backtesting)
- **Utilities**: ≥70% (indicators, data processing)

---

## Communication Standards

### 1. Issue Reporting
```markdown
## Issue: [간결한 제목]

**Environment**:
- OS: Windows 10
- Python: 3.9.7
- BingX Testnet

**Expected Behavior**:
포지션 진입 후 Stop Loss 자동 설정

**Actual Behavior**:
Stop Loss 주문이 생성되지 않음

**Steps to Reproduce**:
1. 봇 실행: `python scripts/production/phase4_dynamic_paper_trading.py`
2. LONG 신호 발생 대기
3. 포지션 진입 후 로그 확인

**Error Logs**:
```
[ERROR] Failed to create stop loss order: Insufficient balance
```

**Possible Cause**:
Stop Loss 주문 수량 계산 오류 (레버리지 미적용)

**Proposed Solution**:
`quantity * leverage` 고려하여 수량 계산
```

### 2. Meeting Notes (if applicable)
```markdown
# Weekly Review - YYYY-MM-DD

## Attendees
- [이름]

## Agenda
1. 주간 성과 리뷰
2. 이슈 및 해결
3. Next steps 논의

## Decisions
- Decision 1: [결정 사항]
- Decision 2: [결정 사항]

## Action Items
- [ ] Task 1 (담당: [이름], 기한: YYYY-MM-DD)
- [ ] Task 2 (담당: [이름], 기한: YYYY-MM-DD)

## Next Meeting
- Date: YYYY-MM-DD
- Topics: [다음 회의 주제]
```
