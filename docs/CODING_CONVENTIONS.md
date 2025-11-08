# Coding Conventions - CLAUDE_CODE_FIN

**Last Updated**: 2025-10-14

---

## Python Style Guide

**PEP 8 준수** with following specifics:

### 1. Naming Conventions
```python
# 클래스: PascalCase
class BingXClient:
    pass

# 함수/메서드: snake_case
def get_klines():
    pass

# 변수: snake_case
position_size = 0.5
stop_loss_pct = 0.01

# 상수: UPPER_CASE
MAX_POSITION_SIZE = 0.95
DEFAULT_LEVERAGE = 3

# Private 메서드/속성: _prefix
def _calculate_indicators():
    pass
```

### 2. Type Hints (Required)
```python
from typing import Dict, List, Optional, Any

def get_balance() -> Dict[str, Any]:
    """계정 잔고 조회"""
    pass

def process_data(data: List[Dict], limit: Optional[int] = None) -> pd.DataFrame:
    """데이터 전처리"""
    pass
```

### 3. Docstrings (Google Style, 한국어)
```python
def create_order(
    symbol: str,
    side: str,
    quantity: float,
    price: Optional[float] = None
) -> Dict[str, Any]:
    """
    주문 생성

    Args:
        symbol: 거래 쌍 (예: "BTC-USDT")
        side: 주문 방향 ("BUY" 또는 "SELL")
        quantity: 주문 수량
        price: 지정가 가격 (None이면 시장가)

    Returns:
        주문 결과 딕셔너리

    Raises:
        BingXAPIError: API 호출 실패 시
        BingXInsufficientBalanceError: 잔고 부족 시
    """
    pass
```

### 4. Comments (한국어 권장)
```python
# 기술적 지표 계산
indicators = calculate_indicators(data)

# LONG 포지션 진입 조건 확인
if signal > 0.7 and current_position is None:
    # 포지션 사이징 계산
    position_size = calculate_position_size(balance, risk_pct)
```

### 5. Imports Organization
```python
# 1. Standard library
import os
import sys
from typing import Dict, List, Optional

# 2. Third-party
import numpy as np
import pandas as pd
from loguru import logger

# 3. Local imports
from src.api.bingx_client import BingXClient
from src.models.xgboost_model import XGBoostModel
```

### 6. Error Handling
```python
# 커스텀 예외 사용
try:
    result = client.create_order(...)
except BingXInsufficientBalanceError:
    logger.error("잔고 부족")
    return None
except BingXAPIError as e:
    logger.error(f"API 에러: {e}")
    raise
```

### 7. Logging with Loguru
```python
from loguru import logger

# 정보성 로그
logger.info(f"Position opened: {side} {quantity} @ {price}")

# 경고
logger.warning(f"High volatility detected: {volatility:.2f}%")

# 에러
logger.error(f"Order failed: {error_msg}")

# 디버그 (개발 중에만)
logger.debug(f"Feature vector: {features}")
```

### 8. Code Organization
```python
class TradingBot:
    """트레이딩 봇 메인 클래스"""

    # ========== 초기화 ==========

    def __init__(self, config: Dict):
        """초기화"""
        pass

    # ========== 데이터 수집 ==========

    def fetch_market_data(self) -> pd.DataFrame:
        """시장 데이터 수집"""
        pass

    # ========== 신호 생성 ==========

    def generate_signal(self, data: pd.DataFrame) -> float:
        """매매 신호 생성"""
        pass

    # ========== 주문 실행 ==========

    def execute_order(self, signal: float) -> bool:
        """주문 실행"""
        pass

    # ========== 유틸리티 ==========

    def _calculate_position_size(self, balance: float) -> float:
        """포지션 사이징 계산 (Private)"""
        pass
```

---

## File Organization Principles

### src/ vs scripts/ 구분
- **src/**: 재사용 가능한 모듈, 클래스, 함수 (라이브러리 코드)
- **scripts/**: 일회성 실행 스크립트, 실험, 분석 도구

### Documentation Hierarchy
- **Root-level MD**: 프로젝트 현황 및 Quick Start
- **claudedocs/**: 핵심 의사결정 문서 (3-6개 유지)
- **archive/**: 과거 기록 및 실험 문서 (참고용)
