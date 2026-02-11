# Coding Conventions - CLAUDE_CODE_FIN

**Last Updated**: 2026-02-11 | **Bot Version**: v1.27.0

---

## Python Style Guide

**PEP 8 준수** with following specifics:

### 1. Naming Conventions
```python
# 클래스: PascalCase
class PatternBot:
    pass

# 함수/메서드: snake_case
def calculate_tp_sl():
    pass

# 변수: snake_case
position_size = 0.5
stop_loss_pct = 0.01

# 상수: UPPER_CASE
MAX_POSITION_SIZE = 0.95
DEFAULT_LEVERAGE = 3
EXPECTED_WIN_RATE = 84.0

# Private 메서드/속성: _prefix
def _calculate_indicators():
    pass
```

### 2. Type Hints (Required)
```python
from typing import Dict, List, Optional, Any, Tuple

def calculate_tp_sl(
    entry_price: float, direction: str, tp_pct: float, sl_pct: float
) -> Tuple[float, float]:
    """TP/SL 가격 계산"""
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
        ccxt.NetworkError: 네트워크 오류 시
        ccxt.InsufficientFunds: 잔고 부족 시
    """
    pass
```

### 4. Comments (한국어 권장)
```python
# 기술적 지표 계산
indicators = calculate_indicators(data)

# LONG 포지션 진입 조건 확인
if pattern in VALIDATED_LONG_PATTERNS:
    # Per-pattern TP/SL 조회
    tp_pct, sl_pct = PATTERN_OPTIMAL_TPSL[pattern]
```

### 5. Imports Organization
```python
# 1. Standard library
import os
import json
from typing import Dict, List, Optional
from datetime import datetime

# 2. Third-party
import numpy as np
import pandas as pd
import ccxt
from loguru import logger

# 3. Local imports (relative within pattern_5m package)
from .constants import VALIDATED_LONG_PATTERNS, PATTERN_OPTIMAL_TPSL
from .models import PositionState
from .exchange import create_exchange
```

### 6. Error Handling
```python
# CCXT 예외 사용
try:
    result = exchange.create_order(...)
except ccxt.InsufficientFunds:
    logger.error("잔고 부족")
    return None
except ccxt.NetworkError as e:
    logger.error(f"네트워크 에러: {e}")
    # Circuit Breaker 처리
    raise
```

### 7. Logging with Loguru
```python
from loguru import logger

# 정보성 로그
logger.info(f"Position opened: {direction} {quantity} @ {price}")

# 경고
logger.warning(f"Daily loss limit approaching: {daily_pnl:.2f}%")

# 에러
logger.error(f"Order failed: {error_msg}")

# 디버그 (개발 중에만)
logger.debug(f"Pattern detected: {pattern} → {direction}")
```

### 8. Module Organization (pattern_5m)
```
pattern_5m/
├── bot.py              # 메인 루프 — 봇 시작/정지
├── config.py           # 설정 로딩 — YAML 파싱
├── constants.py        # 패턴/TP-SL/상수 — 전략 파라미터
├── exchange.py         # BingX API — CCXT 래퍼
├── indicators.py       # 지표 계산 — RSI, ATR, 캔들 분류
├── models.py           # 데이터클래스 — PositionState 등
├── orders.py           # 주문 관리 — TP/SL 배치/검증
├── position.py         # Facade — 진입/청산/모니터링 통합
├── position_open.py    # 진입 로직 — 사이징, 주문, 상태 업데이트
├── position_monitor.py # 모니터링 — Early Exit, 동기화
├── position_close.py   # 청산 로직 — TP/SL/수동 청산
├── signals.py          # 신호 탐지 — 패턴 매칭, Context Filter
├── state.py            # 상태 관리 — JSON 저장/로드
└── utils/              # 유틸리티 — lock, logging
```

---

## File Organization Principles

### 프로젝트 구조
- **scripts/production/pattern_5m/**: 프로덕션 봇 코드 (14 모듈)
- **scripts/analysis/**: 연구 및 백테스트 스크립트
- **scripts/data/**: 데이터 수집 스크립트
- **scripts/utils/**: 운영 유틸리티 (모니터링, 상태 확인)
- **config/**: YAML 설정 파일
- **data/**: 시장 데이터 CSV (Ground Truth)
- **results/**: 봇 상태/메트릭 JSON
- **claudedocs/**: 연구 프로토콜 및 분석 문서
- **archive/**: 레거시 코드 및 실험 (참고용)

### Documentation Hierarchy
- **CLAUDE.md**: 프로젝트 핵심 문서 (전략, 파라미터, 버전 히스토리)
- **docs/**: 가이드 및 컨벤션 문서
- **claudedocs/**: 연구 프로토콜 및 분석 리포트
- **archive/**: 과거 기록 (참고용)
