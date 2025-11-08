# BingX 강화학습 트레이딩 봇 사용 가이드

## 목차
1. [설치](#설치)
2. [설정](#설정)
3. [사용 방법](#사용-방법)
4. [시스템 아키텍처](#시스템-아키텍처)
5. [주의사항](#주의사항)

---

## 설치

### 1. Python 환경 설정
```bash
# Python 3.9+ 필요
python --version

# 가상환경 생성 (권장)
python -m venv venv

# 가상환경 활성화
# Windows
venv\Scripts\activate
# Linux/Mac
source venv/bin/activate
```

### 2. 의존성 설치
```bash
pip install -r requirements.txt
```

---

## 설정

### 1. API 키 설정
`config/api_keys.yaml.example` 파일을 `config/api_keys.yaml`로 복사하고 실제 API 키 입력:

```yaml
bingx:
  testnet:
    api_key: "YOUR_TESTNET_API_KEY"
    secret_key: "YOUR_TESTNET_SECRET_KEY"

  mainnet:
    api_key: "YOUR_MAINNET_API_KEY"
    secret_key: "YOUR_MAINNET_SECRET_KEY"
```

⚠️ **중요**: `api_keys.yaml` 파일은 절대 Git에 커밋하지 마세요!

### 2. 거래 설정 조정 (선택)
`config/config.yaml` 파일에서 다음 설정을 조정할 수 있습니다:

#### 거래 설정
```yaml
trading:
  leverage: 10           # 레버리지 배수
  timeframe: "5m"        # 캔들 간격
  max_position_size: 0.1 # 최대 포지션 크기 (BTC)
```

#### 리스크 관리
```yaml
risk:
  max_drawdown: 0.15      # 최대 낙폭 15%
  max_daily_loss: 0.05    # 최대 일일 손실 5%
  max_position_risk: 0.02 # 거래당 최대 리스크 2%
```

#### 강화학습 파라미터
```yaml
rl:
  learning_rate: 0.0003
  n_steps: 2048
  batch_size: 64
  n_epochs: 10
```

---

## 사용 방법

### 1. 데이터 수집

먼저 과거 데이터를 수집합니다:

```bash
python scripts/collect_data.py
```

- 기본적으로 365일치 5분 캔들 데이터를 수집합니다
- 데이터는 `data/historical/` 디렉토리에 저장됩니다
- 설정: `config/config.yaml`의 `data.historical_days`

### 2. 모델 훈련

수집한 데이터로 강화학습 모델을 훈련합니다:

```bash
python scripts/train.py
```

**훈련 프로세스**:
1. 데이터 로드 및 전처리
2. 기술적 지표 계산
3. 훈련/검증/테스트 데이터 분할 (70%/15%/15%)
4. PPO 에이전트 훈련
5. 모델 자동 저장 (`data/trained_models/`)

**훈련 시간**:
- 100만 타임스텝: 약 2-4시간 (CPU 기준)
- GPU 사용 시 더 빠름

**저장되는 모델**:
- `best_model.zip`: 검증 성능이 가장 좋은 모델
- `final_model.zip`: 훈련 완료 후 최종 모델

### 3. 백테스팅

훈련된 모델을 테스트 데이터로 검증합니다:

```bash
python scripts/backtest.py
```

**출력**:
- 총 수익률
- 최대 낙폭
- 샤프 비율
- 승률
- 거래 횟수
- 자산 곡선 차트

**결과 파일**:
- `data/logs/backtest_results.csv`: 자산 곡선 데이터
- `data/logs/backtest_results_metrics.txt`: 성과 지표

### 4. 실시간 거래

#### 테스트넷 모의 거래 (권장)
```bash
python scripts/live_trade.py --testnet --dry-run
```

- `--testnet`: 테스트넷 사용
- `--dry-run`: 실제 주문 없이 모의 실행 (기본값)

#### 테스트넷 실거래
```bash
python scripts/live_trade.py --testnet
```

⚠️ 테스트넷에서 충분히 검증 후 진행하세요!

#### 메인넷 실거래 (주의!)
```bash
python scripts/live_trade.py
```

⚠️ **경고**: 실제 자금이 사용됩니다!

**실시간 거래 동작**:
1. 5분마다 최신 캔들 데이터 수집
2. 기술적 지표 계산
3. AI 모델이 포지션 결정
4. 리스크 검증 후 주문 실행
5. 손절/익절 자동 관리

---

## 시스템 아키텍처

### 주요 컴포넌트

#### 1. API 클라이언트 (`src/api/`)
- BingX REST API 연동
- 시장 데이터 조회
- 주문 실행
- 계정 관리

#### 2. 데이터 수집 (`src/data/`)
- 과거 데이터 수집 및 저장
- 실시간 데이터 업데이트
- 데이터 전처리 및 정규화

#### 3. 기술적 지표 (`src/indicators/`)
- EMA (9, 21, 50)
- RSI
- Bollinger Bands
- MACD
- Stochastic
- ADX
- ATR
- VWAP

#### 4. 거래 환경 (`src/environment/`)
- Gymnasium 호환 환경
- 포지션 관리
- 손익 계산
- 보상 함수

#### 5. 강화학습 에이전트 (`src/agent/`)
- PPO (Proximal Policy Optimization) 알고리즘
- 모델 훈련 및 저장
- 행동 예측

#### 6. 리스크 관리 (`src/risk/`)
- 최대 낙폭 제한
- 일일 손실 제한
- 포지션 크기 제한
- 서킷 브레이커

#### 7. 백테스팅 (`src/backtesting/`)
- 과거 데이터로 전략 검증
- 성과 지표 계산
- 결과 시각화

#### 8. 실시간 거래 (`src/trading/`)
- 실시간 데이터 수집
- AI 기반 거래 실행
- 리스크 모니터링

---

## 주의사항

### ⚠️ 리스크 경고

1. **금융 손실 위험**
   - 암호화폐 거래는 높은 위험을 수반합니다
   - 투자 가능한 금액만 사용하세요
   - 과거 성과가 미래 수익을 보장하지 않습니다

2. **레버리지 위험**
   - 10배 레버리지는 수익과 손실을 모두 증폭시킵니다
   - 청산 위험이 높습니다
   - 초보자는 낮은 레버리지 권장 (2-3배)

3. **AI 모델 한계**
   - 시장 급변 시 예측 실패 가능
   - 블랙 스완 이벤트 대응 불가
   - 지속적인 모니터링 필요

### 🛡️ 안전 수칙

1. **항상 테스트넷에서 먼저 테스트**
   - 최소 1-2주 테스트
   - 다양한 시장 상황 확인

2. **소액으로 시작**
   - 처음에는 최소 금액으로 시작
   - 안정성 확인 후 점진적 증액

3. **리스크 설정 엄수**
   - `config.yaml`의 리스크 한도 준수
   - 일일 손실 제한 설정
   - 최대 낙폭 제한 설정

4. **정기 모니터링**
   - 로그 파일 정기 확인 (`data/logs/`)
   - 성과 지표 추적
   - 이상 징후 즉시 대응

5. **정기 재훈련**
   - 시장 환경 변화에 따라 모델 재훈련
   - 권장: 월 1회 재훈련
   - 새로운 데이터로 백테스팅

### 📝 모범 사례

1. **데이터 관리**
   - 정기적으로 데이터 업데이트
   - 이상 데이터 확인 및 제거
   - 데이터 백업

2. **모델 관리**
   - 여러 버전의 모델 유지
   - 성과 기록 및 비교
   - 최적 모델 선택

3. **로그 분석**
   - 거래 로그 정기 분석
   - 패턴 및 이상 거래 확인
   - 개선 포인트 파악

4. **설정 최적화**
   - 리스크 설정 조정
   - 하이퍼파라미터 튜닝
   - 지표 조합 실험

---

## 트러블슈팅

### API 연결 오류
```
Failed to connect to exchange
```
**해결방법**:
- API 키 확인
- 네트워크 연결 확인
- BingX API 상태 확인

### 데이터 수집 실패
```
No data collected
```
**해결방법**:
- API 키 권한 확인
- 심볼명 확인 (BTC-USDT)
- API 요청 제한 확인

### 모델 로드 오류
```
Model not found
```
**해결방법**:
- 모델 훈련 먼저 실행
- 모델 파일 경로 확인
- `data/trained_models/` 디렉토리 확인

### 메모리 부족
```
Out of memory
```
**해결방법**:
- 데이터 수집 기간 줄이기
- 배치 크기 감소
- n_steps 감소

---

## 지원 및 문의

- **버그 리포트**: GitHub Issues
- **기능 제안**: GitHub Discussions
- **문서**: README.md, USAGE_GUIDE.md

---

## 면책조항

이 소프트웨어는 교육 목적으로 제공됩니다. 실제 거래에서 발생하는 모든 손실에 대해 개발자는 책임지지 않습니다. 사용자는 자신의 책임 하에 거래를 진행해야 합니다.

**투자 경고**: 암호화폐 투자는 높은 위험을 수반하며 원금 손실 가능성이 있습니다.
