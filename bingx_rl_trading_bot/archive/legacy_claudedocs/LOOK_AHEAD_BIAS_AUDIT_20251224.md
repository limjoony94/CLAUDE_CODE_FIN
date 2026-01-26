# Look-Ahead Bias 전면 감사 보고서

**날짜**: 2025-12-24
**트리거**: MS_ChoCH 전략 연구 결과 (+609.1%) vs 실제 백테스트 (-6.50%) 불일치 발견

---

## 📋 Executive Summary

### 감사 결과

| 분류 | 파일 수 | 영향도 | 상태 |
|------|--------|--------|------|
| **🔴 치명적 (백테스트 무효)** | 10개 | 연구 결과 신뢰 불가 | 폐기 필요 |
| **🟢 안전 (Active Bot 연구)** | 5개+ | 연구 결과 유효 | 유지 |
| **🟡 의도적 (ML 라벨링)** | 3개 | 정상 사용 | 유지 |

### 핵심 결론

✅ **RSI Trend Filter Bot**: 연구 스크립트 Look-Ahead Bias **없음** → 연구 결과 **유효**
✅ **SuperTrend 5m Bot**: 연구 스크립트 Look-Ahead Bias **없음** → 연구 결과 **유효**
❌ **MS_ChoCH Bot**: Look-Ahead Bias **심각** → 연구 결과 **무효**, 전략 **폐기**

---

## 🔴 Look-Ahead Bias 발견 파일 (치명적)

### MS_ChoCH 관련 (전략 폐기)

| 파일 | 패턴 | 영향 |
|------|------|------|
| `new_strategy_quick_screen.py` | `shift(-1)`, `shift(-2)` | Swing Point 감지에 미래 데이터 사용 |
| `ms_choch_monthly_backtest.py` | `center=True` | 미래 5봉 데이터로 Swing 판정 |
| `ms_choch_30day_backtest.py` | `center=True` | 동일 |
| `ms_choch_7day_backtest.py` | `center=True` | 동일 |

**문제 코드**:
```python
# ❌ 미래 참조 - 실시간 불가능
df['swing_high'] = (df['high'] > df['high'].shift(-1)) & \
                   (df['high'] > df['high'].shift(-2))

# ❌ center=True - 양방향 데이터 사용
df['swing_high'] = df['high'].rolling(11, center=True).max() == df['high']
```

### 레거시 전략 연구 (폐기 권고)

| 파일 | 패턴 | 영향 |
|------|------|------|
| `buy_low_structure_exit_validation.py` | `center=True` | Structure Exit 연구 무효 |
| `professional_exit_strategies.py` | `center=True` | Exit 전략 연구 무효 |
| `dynamic_tp_swing.py` | `center=True` | 동적 TP 연구 무효 |
| `backtest_ultra_selective.py` | `center=True` | 선택적 백테스트 무효 |
| `rsi_zone_whipsaw_defense_research.py` | `center=True` | Whipsaw 방어 연구 무효 |

---

## 🟢 안전 확인 파일 (유효)

### Active Bot 관련 연구

| 파일 | 확인 결과 | 비고 |
|------|----------|------|
| `rsi_trend_filter_walkforward.py` | ✅ Look-Ahead **없음** | RSI Trend Filter 유효 |
| `rsi_strategy_deep_research.py` | ✅ Look-Ahead **없음** | RSI 파라미터 연구 유효 |
| `best_strategy_validation.py` | ✅ Look-Ahead **없음** | 최종 검증 유효 |
| `alternative_strategies_research.py` | ✅ Look-Ahead **없음** | 대안 전략 비교 유효 |
| `comprehensive_v3_research.py` | ✅ Look-Ahead **없음** | V3 연구 유효 |

### RSI Trend Filter 결론
- **연구 결과 (+120.8%)**: 유효
- **Walk-Forward (6/7 수익)**: 유효
- **통계적 유의성 (p=0.013)**: 유효
- **현재 봇 v2.0**: 안전하게 운영 가능

### SuperTrend 5m 결론
- **연구 결과 (+42.8%)**: 유효
- **Walk-Forward (5/6 수익)**: 유효
- **현재 봇 v1.0**: 안전하게 운영 가능

---

## 🟡 의도적 Future Reference (허용)

ML 모델 학습용 라벨 생성에는 미래 데이터 참조가 **필요**합니다.

| 파일 | 용도 | 상태 |
|------|------|------|
| `analyze_feature_value_distribution.py` | 라벨 생성 | 정상 |
| `analyze_feature_directional_bias.py` | 라벨 생성 | 정상 |
| `analyze_model_confidence.py` | 라벨 생성 | 정상 |

**허용 코드 (라벨용)**:
```python
# ✅ 라벨 생성용 - 명확히 구분됨
df['future_max'] = df['close'].shift(-1).rolling(window=lookahead).apply(lambda x: x.max())
```

---

## 📊 영향 받은 연구 결과 정리

### 폐기해야 할 연구 결과

| 연구 | 주장 성과 | 실제 추정 | 상태 |
|------|----------|----------|------|
| MS_ChoCH | +609.1% | **-6.50%** | ❌ 폐기 |
| Structure Exit | 다양 | 미검증 | ⚠️ 재검증 필요 |
| Dynamic TP Swing | 다양 | 미검증 | ⚠️ 재검증 필요 |
| Ultra Selective | 다양 | 미검증 | ⚠️ 재검증 필요 |

### 유효한 연구 결과

| 연구 | 검증된 성과 | 신뢰도 |
|------|------------|--------|
| RSI Trend Filter | +61.0% (WF) | ✅ 높음 |
| SuperTrend 5m | +42.8% (WF) | ✅ 높음 |
| Comprehensive V3 | 다양 | ✅ 유효 |

---

## 🛡️ 예방 가이드라인

### 1. 금지 패턴

```python
# ❌ 절대 금지 - 백테스트/신호 생성에서
df['column'].shift(-1)          # 미래 1봉 참조
df['column'].shift(-n)          # 미래 n봉 참조
df.rolling(n, center=True)      # 양방향 롤링
```

### 2. 허용 패턴

```python
# ✅ 안전 - 과거만 참조
df['column'].shift(1)           # 과거 1봉
df['column'].shift(n)           # 과거 n봉 (n > 0)
df.rolling(n).xxx()             # 기본값 = 과거만

# ✅ ML 라벨용 - 명확히 분리
labels['future_return'] = ...   # 라벨 전용, 신호 생성에 사용 금지
```

### 3. 검증 체크리스트

**코드 작성 시**:
- [ ] `shift(-` 패턴이 없는지 확인
- [ ] `center=True` 패턴이 없는지 확인
- [ ] Entry 시점이 시그널 발생 "후"인지 확인

**코드 리뷰 시**:
```bash
# 전체 스크립트 검사
grep -rn "shift(-" scripts/analysis/
grep -rn "center=True" scripts/analysis/
```

**백테스트 검증 시**:
- [ ] Walk-Forward 검증 수행
- [ ] 완전히 새로운 기간에서 Out-of-Sample 테스트
- [ ] 프로덕션 봇 로직과 백테스트 로직 비교

### 4. 올바른 Swing Point 감지

```python
# ❌ 잘못된 방법 (미래 참조)
df['swing_high'] = df['high'].rolling(11, center=True).max() == df['high']

# ✅ 올바른 방법 (과거만)
def detect_swing_high(df, lookback=5):
    """
    현재 봉이 직전 lookback 봉들 중 최고점인지 확인
    """
    return df['high'] == df['high'].rolling(lookback).max()
```

---

## 🔧 권장 조치

### 즉시 조치 (완료)
- [x] MS_ChoCH Bot 중지
- [x] 모든 Active Bot Look-Ahead 검증
- [x] 전면 감사 수행

### 필요 조치
- [ ] 폐기 대상 연구 결과물 아카이브 이동
- [ ] CLAUDE.md에서 MS_ChoCH 관련 내용 제거/업데이트
- [ ] 새 연구 시 Look-Ahead 체크리스트 적용
- [ ] 기존 Serena 메모리 업데이트

### 장기 조치
- [ ] 백테스트 프레임워크에 Look-Ahead 자동 감지 추가
- [ ] CI/CD에 Look-Ahead 검사 통합

---

## 📝 결론

**MS_ChoCH 전략의 +609.1% 성과는 Look-Ahead Bias로 인한 허위 결과였습니다.**

그러나 **RSI Trend Filter**와 **SuperTrend 5m** 봇의 연구는 Look-Ahead Bias가 없어 **유효**합니다.

이번 사건을 계기로 모든 향후 연구에는 Look-Ahead Bias 검증을 의무화해야 합니다.

---

**작성자**: Claude
**검토자**: -
**승인일**: 2025-12-24
