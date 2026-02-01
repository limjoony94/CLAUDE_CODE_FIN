# 정리 대상 목록 (Cleanup List)

**작성일**: 2026-02-01
**⚠️ 아직 삭제하지 않음 — 검토 후 진행**

---

## 1. models/ PKL 파일 (671개)

대부분 실험/학습 과정에서 생성된 레거시 모델. 현재 pattern_5m_bot은 패턴 기반 전략을 사용하므로 XGBoost 모델 대부분은 미사용 상태.

**제안**: `archive/legacy_models/`로 이동 (또는 현재 사용 중인 모델만 남기고 정리)

주요 카테고리:
- `xgboost_short_entry_*` — ~150개
- `xgboost_long_entry_*` — ~150개
- `xgboost_short_exit_*` — ~100개
- `xgboost_long_exit_*` — ~100개
- `xgboost_*_trade_outcome_*` — ~50개
- `xgboost_v2/v3/v4_*` — ~15개 (구버전)
- `xgboost_regime_*` — 3개
- `scaler_*` — ~10개
- `lstm_scaler.pkl` — 1개
- `xgboost_model.pkl`, `xgboost_model_old.pkl`, `xgboost_model_smote.pkl`, `xgboost_model_phase2.pkl`

**data/trained_models/에도 4개**:
- `data/trained_models/xgboost_improved/xgboost_improved_v1_config.pkl`
- `data/trained_models/xgboost_fixed/xgboost_fixed_v1_config.pkl`
- `data/trained_models/xgboost/xgboost_v1_config.pkl`
- `data/trained_models/xgboost_regression/xgboost_regression_v1_config.pkl`

---

## 2. .bat 파일 (24개)

WSL 환경이므로 shell 스크립트(`scripts/ops/`)로 대체 완료. 모두 삭제 가능.

### 루트 레벨 (11개)
- `START_BOT.bat`
- `STOP_BOT.bat`
- `RESTART_BOT.bat`
- `STATUS_BOT.bat`
- `QUANT_MONITOR.bat`
- `START_RSI_TREND_FILTER.bat`
- `MONITOR_RSI_TREND_FILTER.bat`
- `START_ADX_SUPERTREND_TRAIL.bat`
- `MONITOR_ADX_SUPERTREND_TRAIL.bat`
- `scripts/restart_production_bot.bat`

### archive/deprecated_bots/ (3개)
- `START_RSI_MARTINGALE.bat`
- `STOP_RSI_MARTINGALE.bat`
- `MONITOR_RSI_MARTINGALE.bat`

### archive/old_monitors_20251014/ (10개)
- `MONITOR.bat`, `monitor_*.bat` (10개)

---

## 3. 정리 실행 방법 (참고)

```bash
# .bat 파일 삭제
find . -name "*.bat" -exec trash {} \;

# models/ 아카이브 (사용 중인 모델 확인 후)
mkdir -p archive/legacy_models
mv models/*.pkl archive/legacy_models/
```
