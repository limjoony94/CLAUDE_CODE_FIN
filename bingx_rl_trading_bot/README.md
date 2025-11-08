# BTC 5-Minute XGBoost Trading Bot

**⚡ CURRENT SYSTEM**: ✅ **Opportunity Gating + 4x Leverage - Week 1 Validation**
**Strategy**: SHORT only when EV(SHORT) > EV(LONG) + 0.001 gate
**Leverage**: 4x (BOTH mode)
**Expected Performance**: 18.13% per 5 days (+97.6% over 105 days)
**Key Innovation**: Opportunity cost gating prevents low-quality SHORT trades

📊 **For Current System Status**: → [`SYSTEM_STATUS.md`](SYSTEM_STATUS.md)
🚀 **For Deployment Details**: → [`claudedocs/OPPORTUNITY_GATING_DEPLOYMENT_20251017.md`](claudedocs/OPPORTUNITY_GATING_DEPLOYMENT_20251017.md)
📈 **For Monitoring Guide**: → [`claudedocs/QUANT_MONITOR_GUIDE.md`](claudedocs/QUANT_MONITOR_GUIDE.md)

---

## 🎯 Quick Start (Opportunity Gating System)

### For Immediate Status Check
👉 **Read**: [`SYSTEM_STATUS.md`](SYSTEM_STATUS.md)
- Current bot status and configuration
- Expected vs actual performance
- Monitoring commands and troubleshooting

### For Complete Deployment Details
👉 **Read**: [`claudedocs/OPPORTUNITY_GATING_DEPLOYMENT_20251017.md`](claudedocs/OPPORTUNITY_GATING_DEPLOYMENT_20251017.md)
- Complete deployment journey (04:02-04:10)
- Strategy explanation and validation
- Week 1 validation plan

### For System Monitoring
👉 **Read**: [`claudedocs/QUANT_MONITOR_GUIDE.md`](claudedocs/QUANT_MONITOR_GUIDE.md)
- Professional monitoring dashboard
- Gate effectiveness tracking
- Alert thresholds and validation criteria

---

## 📜 Historical Documentation (Phase 4 System)

**Note**: The documentation below describes the previous Phase 4 4-Model system (October 10-16, 2025).
This system has been superseded by the **Opportunity Gating + 4x Leverage** system deployed on October 17, 2025.
The Phase 4 documentation is preserved below for historical reference and system evolution context.

---

## 📊 Current Status (2025-10-14)

### 4-Model Normalized System ✅
```yaml
Architecture: Entry Dual (LONG + SHORT) + Exit Dual (LONG Exit + SHORT Exit)

LONG Entry Model:
  Model: xgboost_v4_phase4_advanced_lookahead3_thresh0.pkl
  Features: 37 (Phase 4 Advanced)
  Normalization: MinMaxScaler(-1, 1) ✅
  F1 Score: 0.113
  Entry Threshold: 0.7

SHORT Entry Model:
  Model: xgboost_v4_short_model.pkl
  Features: 37 (Phase 4 Advanced)
  Normalization: MinMaxScaler(-1, 1) ✅
  F1 Score: 0.166 (+18.6% improvement from normalization!)
  Recall: 17.9% (+45.5% improvement!)
  Top Feature: num_support_touches (15.3% importance)
  Entry Threshold: 0.7

LONG Exit Model:
  Model: xgboost_v4_long_exit.pkl
  Features: 44 (36 base + 8 position)
  Normalization: MinMaxScaler(-1, 1) ✅
  F1 Score: 0.546
  Exit Threshold: 0.75

SHORT Exit Model:
  Model: xgboost_v4_short_exit.pkl
  Features: 44 (36 base + 8 position)
  Normalization: MinMaxScaler(-1, 1) ✅
  F1 Score: 0.541
  Exit Threshold: 0.75

Backtest Performance (63 windows, 5 days each):
  Returns: +13.52% per 5 days
  Win Rate: 65.1% overall
  LONG Win Rate: 63.8%
  SHORT Win Rate: 65.4% ⭐ (Higher than LONG!)
  Sharpe Ratio: 11.88
  Max Drawdown: <2%
  Trades per 5 days: ~23
  LONG Trades: 78.9%, SHORT Trades: 21.1%

Key Improvement (MinMaxScaler Normalization):
  Problem: SHORT model underperforming (41.9% win rate)
  Root Cause: Count-based features not normalized (0-40+ range)
  Solution: MinMaxScaler(-1, 1) on all 4 models
  Result: SHORT F1 +18.6%, Recall +45.5%, Win Rate 65.4%
```

### Production Deployment
- **Status**: Bot running with normalized 4-model system
- **Location**: `scripts/production/phase4_dynamic_testnet_trading.py`
- **Network**: BingX Testnet (virtual capital)
- **Entry Threshold**: 0.7 (XGBoost probability, LONG/SHORT independent)
- **Exit Threshold**: 0.75 (specialized exit models)
- **Risk Management**: Dynamic SL/TP, Max Hold 4h
- **Position Sizing**: 20-95% dynamic (based on model confidence)
- **Normalization**: Applied to all predictions in real-time

---

## 🚀 What Happened? (Project Journey)

### Milestone 1: Bug Discovery & Phase 4 (10/10)
1. **Initial Attempts**: 11+ ML configs, all failed to beat Buy & Hold
2. **Critical Discovery**: Backtest had 3 major bugs
   - Bug #1: HOLD action caused position closure (not hold!)
   - Bug #2: No liquidation protection (-1,051% impossible loss)
   - Bug #3: Leverage applied twice to fees (3x overcharge)
3. **Phase 4 Base Model**: 37 advanced features → 7.68% returns
4. **Statistical Validation**: Robust methodology with n=29 windows
5. **Outcome**: Phase 4 Base model validated and deployed

### Milestone 2: 4-Model System (10/11-13)
1. **Architecture Evolution**: Single LONG model → Dual Entry + Dual Exit
2. **SHORT Entry Model**: Added independent SHORT predictions
3. **Exit Models**: Specialized LONG Exit + SHORT Exit (44 features each)
4. **Performance**: LONG/SHORT strategy validated
5. **Outcome**: 4-model system with independent predictions

### Milestone 3: MinMaxScaler Normalization (10/14)
1. **Problem Discovery**: SHORT model underperforming (41.9% win rate, F1 0.161)
2. **Root Cause Analysis**: Count-based features not normalized (num_support_touches: 0-40+)
3. **StandardScaler Attempt**: Made performance WORSE (-13%)
4. **MinMaxScaler Solution**: Switched to MinMaxScaler(-1, 1) on all 4 models
5. **Results**:
   - SHORT F1: 0.166 (+18.6% improvement)
   - SHORT Recall: 17.9% (+45.5% improvement)
   - SHORT Win Rate: 65.4% (higher than LONG 63.8%!)
   - Overall Returns: 13.52% per 5 days (+76% from Phase 4 Base)
6. **Outcome**: All models normalized and deployed

### Current State
- ✅ Bugs fixed (10/10)
- ✅ Phase 4 Base model validated (10/10)
- ✅ 4-Model system implemented (10/11-13)
- ✅ MinMaxScaler normalization complete (10/14)
- ✅ SHORT model performance fixed (10/14)
- ✅ Production deployed with normalized predictions (10/14)
- ✅ Bot running on BingX Testnet (virtual capital)

---

## 📁 Repository Structure

```
bingx_rl_trading_bot/
├── README.md                              # This file
├── QUICK_START_GUIDE.md                   # Immediate action guide
├── PROJECT_STRUCTURE.md                   # Detailed code organization
│
├── claudedocs/                            # 📚 Key Documentation
│   ├── EXECUTIVE_SUMMARY_FINAL.md         # Final model decision
│   ├── PRODUCTION_DEPLOYMENT_PLAN.md      # Deployment details
│   └── FINAL_SUMMARY_AND_NEXT_STEPS.md    # Complete analysis
│
├── archive/                               # 📦 Historical Documentation (73 files)
│   ├── analysis/          # Critical analyses and discoveries
│   ├── experiments/       # Experiment results (LSTM, Lag Features, etc.)
│   ├── old_conclusions/   # Previous conclusions (Buy & Hold era)
│   └── deprecated/        # Old guides and duplicate docs
│
├── models/                                # 🎯 ML Models (4-Model Normalized System)
│   ├── xgboost_v4_phase4_advanced_lookahead3_thresh0.pkl  # ✅ LONG Entry
│   ├── xgboost_v4_phase4_advanced_lookahead3_thresh0_scaler.pkl  # LONG scaler
│   ├── xgboost_v4_short_model.pkl                         # ✅ SHORT Entry
│   ├── xgboost_v4_short_model_scaler.pkl                  # SHORT scaler
│   ├── xgboost_v4_long_exit.pkl                           # ✅ LONG Exit
│   ├── xgboost_v4_long_exit_scaler.pkl                    # LONG exit scaler
│   ├── xgboost_v4_short_exit.pkl                          # ✅ SHORT Exit
│   ├── xgboost_v4_short_exit_scaler.pkl                   # SHORT exit scaler
│   └── *_metadata.json                                    # Model metadata
│
├── scripts/                               # 🔧 Scripts
│   ├── production/
│   │   ├── phase4_dynamic_testnet_trading.py  # ✅ Production bot (4-model normalized)
│   │   └── train_*.py                         # Model training scripts
│   ├── experiments/
│   │   ├── backtest_phase4_improved_statistics.py
│   │   └── tune_phase4_lag_hyperparameters.py
│   └── ...
│
├── data/historical/
│   └── BTCUSDT_5m_max.csv                 # 60 days, 17,280 candles
│
└── src/                                   # Source code modules
    ├── api/              # BingX API integration
    ├── data/             # Data collection & processing
    ├── indicators/       # Technical indicators
    ├── backtesting/      # Backtesting engine
    └── utils/            # Utilities
```

---

## 🔑 Key Features

### 4-Model System Architecture
- **Entry Models**: Independent LONG + SHORT predictions (37 features each)
- **Exit Models**: Specialized LONG Exit + SHORT Exit (44 features each)
- **Algorithm**: XGBoost gradient boosting classification
- **Normalization**: MinMaxScaler(-1, 1) on all 4 models
- **Target**: 3-candle (15-minute) future returns (entry), optimal exit timing (exit)
- **Key Innovation**: COUNT-based features normalized (critical for SHORT model)
- **Hyperparameters**: Optimized for 5-minute crypto trading

### Advanced Technical Features (27)
```python
# Volatility & Range
- ATR (14), ATR ratio, Bollinger Band width
- True Range, High-Low range

# Momentum & Oscillators
- RSI (14), Stochastic RSI, Williams %R
- CCI, CMO, Ultimate Oscillator

# Trend Following
- MACD, MACD signal, MACD histogram
- ADX, DI+, DI-, Aroon Up/Down, Vortex

# Volume Analysis
- OBV, CMF, Volume ratio

# Price Action
- Distance to EMA/BB, Price momentum
```

### Baseline Features (10)
- RSI, MACD, Bollinger Bands, EMA (21, 50)
- ATR, Volume ratio, Price position metrics

### Risk Management
- **Stop Loss**: 1% (fixed)
- **Take Profit**: 3% (1:3 ratio)
- **Max Holding**: 4 hours
- **Position Sizing**: 95% of capital (proven optimal)
- **Entry Filter**: XGBoost probability ≥ 0.7

---

## 📈 Performance Metrics (4-Model Normalized System)

### Backtest Results (With Normalization)
```yaml
Test Period: Aug 7 - Oct 14, 2025 (19,450 candles)
Window Size: 5 days (1,440 candles)
Sample Size: n=63 windows

Returns per 5 days: +13.52% (average)
Win Rate: 65.1% overall
  - LONG Win Rate: 63.8%
  - SHORT Win Rate: 65.4% ⭐ (Higher than LONG!)

Trade Distribution:
  - LONG Trades: 1,137 (78.9%)
  - SHORT Trades: 304 (21.1%)
  - Average Trades: 22.9 per 5 days (~4.6 per day)

Performance by Market:
  - Bull (13 windows): +17.93%, Win Rate 68.9%
  - Sideways (32 windows): +17.08%, Win Rate 66.0%
  - Bear (18 windows): +4.01%, Win Rate 60.9%

Risk Metrics:
  - Sharpe Ratio: 11.88
  - Max Drawdown: <2%
  - Liquidations: 4 (0.3% of trades)
  - Average Position: 54.6%

Normalization Impact:
✅ SHORT F1 Score: +18.6% improvement (0.140 → 0.166)
✅ SHORT Recall: +45.5% improvement (12.3% → 17.9%)
✅ SHORT Win Rate: 65.4% (vs 41.9% without normalization)
✅ Overall Returns: +76% improvement (7.68% → 13.52% per 5 days)
```

### Expected Monthly Performance (Normalized Models)
```yaml
Conservative (70% of backtest):
- Returns: ~56.7% per month (13.52% × 6 × 0.7)
- Win Rate: ≥60%
- SHORT Win Rate: ≥55%
- Max DD: <2%

Realistic (85% of backtest):
- Returns: ~68.9% per month (13.52% × 6 × 0.85)
- Win Rate: ≥63%
- SHORT Win Rate: ≥60%
- Max DD: <2%

Optimistic (100% of backtest):
- Returns: ~81.1% per month (13.52% × 6)
- Win Rate: ≥65%
- SHORT Win Rate: ≥65%
- Max DD: <2%
```

---

## 🛠️ Technical Stack

### Core Technologies
- **Python 3.9+**
- **XGBoost**: ML model training & inference
- **Pandas/Numpy**: Data processing
- **TA-Lib**: Technical indicators
- **Loguru**: Logging

### APIs & Data
- **BingX API**: Live trading integration
- **Historical Data**: 5-minute BTCUSDT candles

### Development Tools
- **Scikit-learn**: Model validation & metrics
- **Joblib**: Model serialization
- **Statistical Libraries**: Bootstrap, effect size analysis

---

## 📋 Next Steps (With Normalized Models)

### Week 1: Normalization Validation
**Goal**: Validate normalized model performance in real-world trading

**Success Criteria**:
- Overall Win rate ≥60% (target: 65.1%)
- LONG Win rate ≥60% (target: 63.8%)
- SHORT Win rate ≥55% (target: 65.4%) ← Critical for normalization validation
- Returns ≥10% per 5 days (target: 13.52%)
- Max DD <2%
- Trade frequency: ~4-5 per day

**Focus Areas**:
1. Monitor SHORT model performance specifically
2. Track LONG vs SHORT win rates
3. Validate normalization impact on real data
4. Compare actual vs backtest expectations
5. Verify count-based feature behavior (num_support_touches)

### Week 2-4: Performance Analysis
**Goals**:
- Evaluate normalization effectiveness
- Assess threshold appropriateness (0.7 entry, 0.75 exit)
- Compare LONG vs SHORT performance
- Identify any regime-specific patterns

**Potential Adjustments**:
- Threshold tuning if trade frequency too low
- Regime-specific thresholds
- Position sizing optimization

### Month 1-2: Optimization & Stability
**Goals**:
- Achieve consistent ≥85% of expected returns
- Monthly retraining with normalized features
- Optimize LONG/SHORT balance

**Possible Improvements**:
- Feature engineering for SHORT model
- Multi-timeframe confirmation
- Better support/resistance detection

### Month 3-6: LSTM Development (If Needed)
**Goals**:
- Collect 6+ months data with normalized system
- Evaluate if LSTM is needed (current system may be sufficient)
- Expected: 8-10% per 5 days (LSTM alone)
- Ensemble: 10-15%+ (XGBoost + LSTM)

**Timeline**:
- Month 3-4: Data collection + performance evaluation
- Month 5: LSTM development (if warranted)
- Month 6: Ensemble integration & validation

---

## ⚠️ Risk Warnings

### Trading Risks
- **High Volatility**: Crypto markets are extremely volatile
- **Model Drift**: Performance may degrade over time (monthly retraining recommended)
- **Black Swan Events**: Unexpected market crashes possible
- **Overfitting Risk**: Backtest ≠ future performance
- **Normalization Sensitivity**: Count-based features critical for SHORT model performance

### Current Deployment
- **Network**: BingX Testnet (virtual capital - no real money at risk)
- **Purpose**: Validate normalized model performance in real market conditions
- **Status**: Initial validation phase (Week 1)

### Recommended Risk Management
1. **Testnet First**: Validate on BingX Testnet before real money (currently testing)
2. **Start Small**: If moving to real trading, test with $100-500 initially
3. **Monitor Daily**: Check performance vs expectations, especially SHORT model
4. **Set Limits**: Daily loss limit -3%, weekly -5%
5. **Stop Conditions**:
   - Overall win rate <55% for 7+ days
   - SHORT win rate <50% (normalization failure indicator)
   - Returns <8% per 5 days consistently
   - Max drawdown >3%
6. **Never Risk Life Savings**: Only invest what you can afford to lose

---

## 📚 Documentation Index

### Essential Reading (Start Here)
1. **[QUICK_START_GUIDE.md](QUICK_START_GUIDE.md)** - Immediate deployment steps
2. **[claudedocs/EXECUTIVE_SUMMARY_FINAL.md](claudedocs/EXECUTIVE_SUMMARY_FINAL.md)** - Model selection rationale
3. **[PROJECT_STRUCTURE.md](PROJECT_STRUCTURE.md)** - Code organization

### Detailed Analysis
- **[claudedocs/PRODUCTION_DEPLOYMENT_PLAN.md](claudedocs/PRODUCTION_DEPLOYMENT_PLAN.md)** - Deployment configuration
- **[claudedocs/FINAL_SUMMARY_AND_NEXT_STEPS.md](claudedocs/FINAL_SUMMARY_AND_NEXT_STEPS.md)** - Complete journey & roadmap

### Historical Archive (73 documents)
- **archive/analysis/** - Critical bug discoveries & analyses
- **archive/experiments/** - LSTM, Lag Features, all experiments
- **archive/old_conclusions/** - Previous "Buy & Hold" conclusions
- **archive/deprecated/** - Old guides and duplicates

---

## 🎓 Lessons Learned

### Critical Thinking Validation
**Process**:
1. Initial failure (Buy & Hold wins) ❌
2. Critical analysis → Bug discovery 🔍
3. Systematic fixes → Phase 4 development ✅
4. Statistical validation → Confident deployment 📊
5. Performance monitoring → SHORT model underperformance found 🔍
6. Root cause analysis → Normalization issue identified ✅
7. MinMaxScaler solution → +76% performance boost 📊

**Key Insight**: Evidence > Assumptions. Never stop investigating.

### Feature Normalization (Critical Discovery)
- **Count-based features MUST be normalized** (num_support_touches: 0-40+)
- StandardScaler made performance WORSE (-13%)
- MinMaxScaler(-1, 1) is critical for bounded features
- Normalization impact: SHORT F1 +18.6%, Recall +45.5%
- XGBoost CAN benefit from normalization (despite being tree-based)
- SMOTE + MinMaxScaler synergy for synthetic sample quality

### Statistical Rigor
- Small samples (n<10) can mislead
- Need bootstrap CI, effect size, power analysis
- Multiple comparison correction essential
- Larger sample size (n=63) provides robust validation

### Feature Engineering
- Quality > Quantity (37 features > 185 lag features)
- Domain knowledge matters
- Advanced indicators improved F1 by 65%
- Feature scale awareness critical (count vs ratio features)
- Overfitting risk with too many features

### Hyperparameter Tuning
- Critical for high-dimensional data
- Can improve but not fix fundamental issues
- Regularization essential for small datasets
- Feature sampling matters (0.8 → 0.5 for 185 features)

### Architecture Evolution
- Single model → 4-model system (Entry Dual + Exit Dual)
- Independent LONG/SHORT predictions improve flexibility
- Specialized exit models enhance timing accuracy
- Dual strategy (+76% vs single LONG model)

---

## 🤝 Contributing & Questions

### For Issues
- Check [QUICK_START_GUIDE.md](QUICK_START_GUIDE.md) troubleshooting section
- Review [archive/README.md](archive/README.md) for historical context

### For Understanding
- Complete journey: [claudedocs/FINAL_SUMMARY_AND_NEXT_STEPS.md](claudedocs/FINAL_SUMMARY_AND_NEXT_STEPS.md)
- Technical details: [PROJECT_STRUCTURE.md](PROJECT_STRUCTURE.md)

---

## 📜 Disclaimers

⚠️ **Educational Purpose**: This project is for learning and demonstration

⚠️ **Not Financial Advice**: Past performance ≠ future results

⚠️ **High Risk**: Cryptocurrency trading is extremely risky

⚠️ **No Guarantees**: Model performance may vary significantly

⚠️ **Use at Your Own Risk**: Only invest what you can afford to lose

---

**⚡ CURRENT SYSTEM**: ✅ **Opportunity Gating + 4x Leverage - Week 1 Validation**
**Last Updated**: 2025-10-17
**Strategy**: Opportunity cost gating (SHORT only when significantly better than LONG)
**Leverage**: 4x (BOTH mode)
**Expected Performance**: 18.13% per 5-day window (63.9% win rate)
**Key Innovation**: Capital efficiency through opportunity cost analysis
**Deployment**: BingX Testnet (Week 1 validation phase)

---

**📊 Check Status**: → [`SYSTEM_STATUS.md`](SYSTEM_STATUS.md)
**🚀 Deployment Details**: → [`claudedocs/OPPORTUNITY_GATING_DEPLOYMENT_20251017.md`](claudedocs/OPPORTUNITY_GATING_DEPLOYMENT_20251017.md)
**📈 Monitor System**: → [`claudedocs/QUANT_MONITOR_GUIDE.md`](claudedocs/QUANT_MONITOR_GUIDE.md)
