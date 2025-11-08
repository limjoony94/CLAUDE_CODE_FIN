# Sweet-2 Paper Trading 시작 가이드

**Date**: 2025-10-10
**Status**: ✅ Ready to Deploy
**Purpose**: Sweet-2 Configuration 실시간 검증

---

## 📋 개요

Sweet-2 Paper Trading Bot은 백테스팅에서 검증된 수익 가능한 설정을 실시간 환경에서 테스트하는 시스템입니다.

### Sweet-2 Configuration

```python
XGB_THRESHOLD_STRONG = 0.7       # XGBoost 강력한 신호
XGB_THRESHOLD_MODERATE = 0.6     # XGBoost 보통 신호
TECH_STRENGTH_THRESHOLD = 0.75   # 기술적 지표 강도 임계값
```

### 백테스팅 검증 결과 (목표)

| Metric | 백테스팅 결과 | 최소 목표 |
|--------|-------------|----------|
| vs Buy & Hold | +0.75% | +0.0% |
| 거래 빈도 (주당) | 2.5 | 2-3 |
| 승률 | 54.3% | 52% |
| 거래당 순이익 | +0.149% | +0.0% |

---

## 🚀 빠른 시작

### 1. 사전 요구사항

✅ XGBoost Phase 2 모델 존재 확인:
```bash
ls models/xgboost_v3_lookahead3_thresh1_phase2.pkl
ls models/xgboost_v3_lookahead3_thresh1_phase2_features.txt
```

✅ 데이터 파일 존재 확인:
```bash
ls data/historical/BTCUSDT_5m_max.csv
```

### 2. 실행

```bash
cd C:\Users\J\OneDrive\CLAUDE_CODE_FIN\bingx_rl_trading_bot
python scripts/production/sweet2_paper_trading.py
```

### 3. 로그 확인

실시간 로그:
```bash
tail -f logs/sweet2_paper_trading_YYYYMMDD.log
```

---

## 📊 Paper Trading 작동 방식

### Update Cycle (5분마다)

```
1. 시장 데이터 수집 (5분 캔들)
   └─> data/historical/BTCUSDT_5m_max.csv (최근 200 candles)

2. Feature 계산
   ├─> XGBoost Phase 2 features (33개)
   └─> Technical indicators (EMA, RSI, ADX, MACD, etc.)

3. Market Regime 분류
   ├─> Bull (최근 20 candles +3% 이상)
   ├─> Bear (최근 20 candles -2% 이하)
   └─> Sideways (그 외)

4. Hybrid Strategy 신호 확인
   ├─> XGBoost 예측 (probability)
   ├─> Technical Strategy 신호 (LONG/HOLD/AVOID)
   └─> Combined Decision:
       ├─> Strong: XGB > 0.7 AND Tech = LONG
       ├─> Moderate: XGB > 0.6 AND Tech = LONG (strength > 0.75)
       └─> Hold: Otherwise

5. 포지션 관리
   ├─> Entry: Strong 또는 Moderate 신호
   ├─> Stop Loss: -1%
   ├─> Take Profit: +3%
   └─> Max Holding: 4 hours

6. 성과 추적
   ├─> vs Buy & Hold 계산
   ├─> Per-trade net profit 계산
   └─> Regime별 성과 기록
```

---

## 📈 모니터링 Metrics

### 핵심 지표 (매일 확인)

**1. 거래 빈도**
```
목표: 2-3 trades/week (주당 4-6 trades per 2-week window)
판정:
  ✅ 2.0 ≤ trades/week ≤ 3.0
  ⚠️ 1.5 ≤ trades/week < 2.0 or 3.0 < trades/week ≤ 4.0
  ❌ trades/week < 1.5 or > 4.0
```

**2. 승률**
```
목표: > 52%
판정:
  ✅ win_rate ≥ 54%
  ⚠️ 52% ≤ win_rate < 54%
  ❌ win_rate < 52%
```

**3. vs Buy & Hold**
```
목표: > 0% (수익만 되면 OK)
판정:
  ✅ vs_bh > +0.5%
  ⚠️ 0% < vs_bh ≤ +0.5%
  ❌ vs_bh ≤ 0%
```

**4. 거래당 순이익**
```
목표: > 0% (필수)
판정:
  ✅ per_trade_net > +0.1%
  ⚠️ 0% < per_trade_net ≤ +0.1%
  ❌ per_trade_net ≤ 0%
```

---

## 📅 Week 1 목표 (10+ trades)

### 일일 체크리스트

- [ ] Paper trading bot 실행 중
- [ ] 로그 파일 확인 (에러 없음)
- [ ] 신호 발생 확인 (XGBoost + Technical)
- [ ] 거래 실행 확인 (entry/exit)
- [ ] 승률 추적 (> 50% 유지)
- [ ] vs Buy & Hold 계산 (양수 유지)

### Week 1 종료 시 판정

```python
if total_trades >= 10:
    if win_rate >= 52% and vs_bh > 0% and per_trade_net > 0%:
        print("✅ Week 1 SUCCESS")
        print("   Continue to Week 2")
    elif win_rate >= 50% and vs_bh >= -0.5%:
        print("⚠️ Week 1 PARTIAL SUCCESS")
        print("   Continue monitoring, consider adjustments")
    else:
        print("❌ Week 1 FAILURE")
        print("   Review strategy, consider regime-specific thresholds")
else:
    print("⏳ Insufficient trades, continue Week 1")
```

---

## 📅 Week 2 목표 (20+ total trades)

### Week 2 Goals

- [ ] 총 20+ trades (통계적 샘플)
- [ ] 승률 > 52% 안정화
- [ ] vs Buy & Hold > +0.3% 달성
- [ ] Bull/Bear/Sideways 각 regime 최소 1회 경험
- [ ] 거래당 순이익 > +0.1% 유지

### Week 2 종료 시 최종 판정

```python
if total_trades >= 20:
    if win_rate >= 54% and vs_bh >= 0.75% and per_trade_net >= 0.15%:
        print("✅✅✅ SWEET-2 VALIDATION SUCCESSFUL!")
        print("   → Phase 3: 소량 실전 배포 (3-5% 자금)")
    elif win_rate >= 52% and vs_bh >= 0.3% and per_trade_net > 0%:
        print("✅ SWEET-2 PARTIAL SUCCESS")
        print("   → 추가 1주 검증 OR 소액 실전 (3% 자금)")
    else:
        print("❌ SWEET-2 VALIDATION FAILED")
        print("   → Option A: 15분 features 추가")
        print("   → Option B: Regime-specific thresholds")
```

---

## 🛠️ 결과 파일

### 자동 생성되는 파일

**1. Trades Log** (거래 기록)
```
results/sweet2_paper_trading_trades_YYYYMMDD_HHMMSS.csv
```
**Columns**:
- entry_time, exit_time
- entry_price, exit_price
- pnl_pct, pnl_usd_gross, transaction_cost, pnl_usd_net
- regime, xgb_prob, tech_signal, tech_strength, confidence

**2. Market Regime History**
```
results/sweet2_market_regime_history_YYYYMMDD_HHMMSS.csv
```
**Columns**:
- timestamp, regime, price

**3. State File** (현재 상태)
```
results/sweet2_paper_trading_state.json
```
**Contents**:
- capital, position, trades_count
- bh_btc_quantity, bh_entry_price
- session_start, timestamp

**4. Logs**
```
logs/sweet2_paper_trading_YYYYMMDD.log
```

---

## 🎯 Decision Tree (2주 후)

```
Paper Trading 2주 완료
│
├─> [CASE 1] 모든 목표 달성 (vs_bh >= 0.75%, WR >= 54%, per_trade_net >= 0.15%)
│   └─> ✅ 소량 실전 배포 (자금 3-5%)
│       ├─> Week 1: 5-10 trades (슬리피지 확인)
│       ├─> Week 2-3: 20+ trades (통계 확보)
│       └─> Week 4: Full deployment 결정
│
├─> [CASE 2] 최소 목표 달성 (vs_bh >= 0.3%, WR >= 52%, per_trade_net > 0%)
│   └─> ⚠️ 추가 검증 OR 소액 실전
│       ├─> Option 1: 1주 추가 paper trading
│       └─> Option 2: 소액 실전 (자금 3%)
│
└─> [CASE 3] 목표 미달성
    └─> ❌ 전략 개선 필요
        ├─> Option A: 15분 features 추가 (Bull market 개선)
        ├─> Option B: Regime-specific thresholds
        └─> Option C: Bear-only strategy (검증된 성공 영역만)
```

---

## ⚠️ 주의사항 (Red Flags)

### 즉시 중단 조건

1. ❌ **승률 < 45%** (2주 연속)
2. ❌ **vs B&H < -1.0%** (2주 연속)
3. ❌ **거래당 순이익 < -0.05%** (1주)
4. ❌ **시스템 오류 반복** (거래 실행 실패)

### 검토 및 개선 필요

1. ⚠️ **승률 45-50%** (1-2주)
2. ⚠️ **vs B&H -0.5% ~ 0%** (1-2주)
3. ⚠️ **거래 빈도 < 2 or > 10** (비정상)
4. ⚠️ **Bull regime에서 -5% 이상 손실**

---

## 🔧 문제 해결

### Q1: "No entry signal" 계속 발생

**원인**: Sweet-2 threshold가 매우 보수적 (xgb_strong=0.7, tech_strength=0.75)

**해결**:
1. 정상적인 현상 (거래 빈도 2-3/week 목표)
2. 1주일 기다려도 거래 < 2회면:
   - Threshold 약간 완화 (xgb_strong=0.68, tech_strength=0.73)
   - 또는 현재 market regime 확인 (Bull에서는 거래 적음)

### Q2: 승률이 50% 미만

**원인**: False signals or 시장 조건 불일치

**해결**:
1. Regime별 성과 확인 (Bull/Bear/Sideways)
2. Bull에서 손실 심하면: 15분 features 필요
3. 전반적으로 낮으면: Tech threshold 상향 (0.75 → 0.80)

### Q3: vs B&H가 음수

**원인**: Transaction costs 또는 시장 강세

**해결**:
1. Per-trade net profit 확인 (양수면 OK, 장기적으로 수렴)
2. Bull market에서 B&H가 유리 (정상)
3. 2주 후에도 음수면: 전략 재검토

---

## 📊 성과 기록 템플릿

### Daily Journal (매일 작성)

```markdown
### Day X (YYYY-MM-DD)

**Market Regime**: Bull/Bear/Sideways
**BTC Price**: $XX,XXX

**Trades Today**:
1. Time: HH:MM | Entry: $XX,XXX | Exit: $XX,XXX | P/L: +X.XX% | WR: ✅/❌
   - XGBoost: X.XXX | Tech: LONG (X.XXX) | Confidence: strong/moderate

**Daily Summary**:
- Total trades (cumulative): X
- Win rate: XX%
- vs B&H: +X.XX%
- Per-trade net: +X.XXX%

**Observations**:
- [Good signals / Bad signals]
- [Market conditions]
- [System performance]

**Action Items**:
- [ ] Issue to fix
- [ ] Improvement idea
```

### Weekly Review (주말 작성)

```markdown
### Week X Review (YYYY-MM-DD)

**Overall Performance**:
- Total trades: X
- Win rate: XX%
- vs B&H: +X.XX%
- Per-trade net: +X.XXX%

**By Regime**:
- Bull: X trades, XX% WR, +X.XX% vs B&H
- Bear: X trades, XX% WR, +X.XX% vs B&H
- Sideways: X trades, XX% WR, +X.XX% vs B&H

**Best Trades**:
1. [Trade details and what made it successful]

**Worst Trades**:
1. [Trade details and what went wrong]

**Learnings**:
- [Pattern recognition]
- [Strategy adjustments needed]

**Next Week Focus**:
- [ ] Goal 1
- [ ] Goal 2
```

---

## 🎓 Sweet-2 Paper Trading 핵심 원칙

### 1. 비판적 사고 유지

- "백테스팅 성공 ≠ 실시간 성공"
- "통계적 샘플 충분히 확보" (최소 20 trades)
- "Regime별 성과 확인" (Bull/Bear/Sideways)

### 2. 인내심

- Sweet-2는 **보수적 전략** (주당 2-3 거래)
- 1일에 거래 0회도 정상
- 1주일 기다려도 5-10 trades 목표

### 3. 데이터 기반 판단

- "감정적 판단 금지"
- "숫자로 말하게 하기" (win rate, vs B&H, per-trade net)
- "최소 2주, 20+ trades 후 결정"

### 4. 점진적 확대

- Paper trading 성공 → 소액 (3-5%)
- 소액 성공 → 중량 (5-10%)
- 중량 성공 → Full deployment (10-20%)

---

## 📞 다음 단계

### Paper Trading 성공 시

1. **IMMEDIATE_ACTION_PLAN.md Phase 3** 참고
2. 소량 실전 배포 (자금 3-5%)
3. 실제 슬리피지/비용 확인
4. 실전 vs Paper trading 비교

### Paper Trading 실패 시

1. **Option A: 15분 Features 추가**
   - `scripts/production/train_xgboost_with_15m_features.py` 완성
   - XGBoost Phase 3 재훈련
   - Bull market detection 개선

2. **Option B: Regime-Specific Thresholds**
   - Bull: xgb_strong=0.65 (완화)
   - Bear: xgb_strong=0.75 (강화)
   - Sideways: 기본값 유지

3. **Option C: Bear-Only Strategy**
   - Bull/Sideways: Buy & Hold
   - Bear: Active trading (Sweet-2)
   - 검증된 성공 영역만 집중

---

**"Paper trading 즉시 시작. 2주 내 go/no-go 결정. 비판적 사고로 지속 검증."** 🎯

**Date**: 2025-10-10
**Status**: ✅ Ready to Deploy
**Next Action**: `python scripts/production/sweet2_paper_trading.py`
