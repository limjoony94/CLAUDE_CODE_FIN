# XGBoost 모델 학습 분석 및 개선 방안

**작성일**: 2025-10-21
**분석 대상**: `scripts/experiments/retrain_entry_models_oct20.py`

---

## 🔍 현재 학습 방식

### 1. 데이터 분할
```python
# 단순 80/20 시계열 분할
split_idx = int(len(X_long) * 0.8)
X_train = X[:split_idx]      # 2025-07-01 ~ 2025-10-01 (약 80%)
X_test = X[split_idx:]        # 2025-10-01 ~ 2025-10-20 (약 20%)
```

### 2. 모델 학습
```python
model = XGBClassifier(
    n_estimators=300,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    scale_pos_weight=auto,    # 클래스 불균형 조정
    random_state=42,
    eval_metric='logloss'
)
model.fit(X_train_scaled, y_train)
```

### 3. 평가
```python
y_pred = model.predict(X_test_scaled)
print(classification_report(y_test, y_pred))
precision = precision_score(y_test, y_pred)
```

---

## ❌ 주요 문제점

### 🔴 **1. Early Stopping 없음** (심각)

**문제:**
- 300개 트리를 무조건 모두 학습
- 과적합 발생해도 계속 학습 진행
- 최적 지점을 놓칠 수 있음

**예시:**
```
Iteration 100: train_loss=0.20, val_loss=0.25
Iteration 200: train_loss=0.15, val_loss=0.26  ← 과적합 시작
Iteration 300: train_loss=0.10, val_loss=0.30  ← 계속 학습
```

**영향:**
- 불필요한 학습 시간
- Test set 성능 저하
- 모델 복잡도 증가

**권장 개선:**
```python
model.fit(
    X_train_scaled, y_train,
    eval_set=[(X_val_scaled, y_val)],
    early_stopping_rounds=50,
    verbose=False
)
```

---

### 🟡 **2. Validation Set 없음** (중요)

**문제:**
- Train/Test 2-way split만 사용
- 하이퍼파라미터 튜닝 시 Test set 오염 위험
- Early stopping 불가능

**현재 구조:**
```
[Train: 80%] [Test: 20%]
```

**권장 구조:**
```
[Train: 60%] [Val: 20%] [Test: 20%]

Train: 모델 학습용
Val: Early stopping, 하이퍼파라미터 튜닝
Test: 최종 성능 평가 (단 1회만 사용)
```

**영향:**
- 과적합 탐지 어려움
- 일반화 성능 저하 가능성

---

### 🟡 **3. 시계열 특성 무시** (중요)

**문제:**
- 단순 80/20 분할은 시계열 데이터에 부적합
- 미래 데이터가 과거 학습에 영향 가능 (data leakage)
- Walk-forward validation 필요

**현재 방식:**
```
[Jul---|Aug---|Sep---|Oct--]
[====Train====][=Test=]
```

**권장 방식 (Walk-Forward):**
```
Window 1: [Jul-Aug] Train → [Sep 1-5] Test
Window 2: [Jul-Sep] Train → [Sep 6-10] Test
Window 3: [Jul-Sep] Train → [Sep 11-15] Test
...
Average performance across windows
```

**영향:**
- 실제 거래 환경과 차이
- 성능 과대평가 가능성

---

### 🟡 **4. 교차 검증 없음** (중요)

**문제:**
- Single train/test split만 사용
- 데이터 분할 운에 따라 성능 변동
- 모델 안정성 검증 불가

**권장 개선 (TimeSeriesSplit):**
```python
from sklearn.model_selection import TimeSeriesSplit

tscv = TimeSeriesSplit(n_splits=5)
scores = []

for train_idx, test_idx in tscv.split(X):
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    model.fit(X_train, y_train)
    score = model.score(X_test, y_test)
    scores.append(score)

print(f"Mean: {np.mean(scores):.4f} ± {np.std(scores):.4f}")
```

**영향:**
- 모델 신뢰도 확인 어려움
- 운 좋은 split에 의존

---

### 🟢 **5. 하이퍼파라미터 튜닝 없음** (개선 권장)

**문제:**
- 고정된 하이퍼파라미터 사용
- 최적 조합 탐색 안 함
- 성능 향상 기회 놓침

**현재:**
```python
model = XGBClassifier(
    n_estimators=300,    # 고정
    max_depth=6,         # 고정
    learning_rate=0.05,  # 고정
    ...
)
```

**권장 개선:**
```python
from sklearn.model_selection import RandomizedSearchCV

param_dist = {
    'n_estimators': [100, 200, 300, 500],
    'max_depth': [4, 6, 8, 10],
    'learning_rate': [0.01, 0.05, 0.1],
    'subsample': [0.7, 0.8, 0.9],
    'colsample_bytree': [0.7, 0.8, 0.9],
    'min_child_weight': [1, 3, 5]
}

search = RandomizedSearchCV(
    XGBClassifier(),
    param_distributions=param_dist,
    n_iter=20,
    cv=TimeSeriesSplit(n_splits=3),
    scoring='precision',
    random_state=42
)

search.fit(X_train, y_train)
best_model = search.best_estimator_
```

**영향:**
- 성능 향상 가능성 (5-15%)
- 더 나은 일반화

---

### 🟢 **6. Feature Importance 분석 없음** (개선 권장)

**문제:**
- 중요한 특성 파악 불가
- 불필요한 특성 제거 기회 놓침
- 모델 해석성 저하

**권장 개선:**
```python
# 학습 후
feature_importance = model.feature_importances_
feature_names = long_feature_cols

# 상위 20개 특성 출력
top_indices = np.argsort(feature_importance)[-20:]
print("\nTop 20 Most Important Features:")
for idx in reversed(top_indices):
    print(f"  {feature_names[idx]}: {feature_importance[idx]:.4f}")

# 시각화
import matplotlib.pyplot as plt
plt.barh(range(20), feature_importance[top_indices])
plt.yticks(range(20), [feature_names[i] for i in top_indices])
plt.xlabel('Importance')
plt.title('Top 20 Feature Importances')
plt.tight_layout()
plt.savefig('feature_importance.png')
```

**활용:**
- 중요도 낮은 특성 제거 → 속도 향상
- 도메인 지식 검증
- 모델 해석 및 신뢰도 향상

---

### 🟢 **7. 과적합 모니터링 부족** (개선 권장)

**문제:**
- Train 성능만 보고 Test 비교 없음
- 과적합 여부 확인 어려움

**권장 개선:**
```python
# Train과 Test 성능 비교
y_train_pred = model.predict(X_train_scaled)
y_test_pred = model.predict(X_test_scaled)

train_precision = precision_score(y_train, y_train_pred)
test_precision = precision_score(y_test, y_test_pred)

print(f"\nOverfitting Check:")
print(f"  Train Precision: {train_precision:.4f}")
print(f"  Test Precision:  {test_precision:.4f}")
print(f"  Gap:             {train_precision - test_precision:.4f}")

if train_precision - test_precision > 0.1:
    print("  ⚠️  Potential overfitting detected!")
```

**기준:**
- Gap < 0.05: 좋음
- Gap 0.05-0.10: 주의
- Gap > 0.10: 과적합 의심

---

### 🟢 **8. 평가 메트릭 제한적** (개선 권장)

**문제:**
- Precision만 저장
- Recall, F1, AUC 등 무시
- 불균형 데이터에서 Precision만으로는 부족

**현재:**
```python
precision_long = precision_score(y_test_long, y_pred_long)
```

**권장 개선:**
```python
from sklearn.metrics import (
    precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix
)

# 확률 예측 (threshold 조정 가능)
y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]

metrics = {
    'precision': precision_score(y_test, y_pred),
    'recall': recall_score(y_test, y_pred),
    'f1': f1_score(y_test, y_pred),
    'auc_roc': roc_auc_score(y_test, y_pred_proba)
}

print("\nComprehensive Metrics:")
for metric, value in metrics.items():
    print(f"  {metric.upper()}: {value:.4f}")

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
print("\nConfusion Matrix:")
print(f"  TN: {cm[0,0]}  FP: {cm[0,1]}")
print(f"  FN: {cm[1,0]}  TP: {cm[1,1]}")
```

**트레이딩 맥락:**
- **Precision**: 신호 정확도 (잘못된 진입 최소화)
- **Recall**: 기회 포착률 (좋은 진입 놓치지 않기)
- **F1**: 균형
- **AUC-ROC**: 전반적 분류 능력

---

### 🟠 **9. NaN 처리 확인 부족**

**문제:**
- X_long에 NaN 있을 경우 처리 없음
- XGBoost는 NaN을 처리하지만, Scaler에서 문제 발생 가능

**권장 개선:**
```python
# NaN 체크
nan_mask = np.isnan(X_long).any(axis=1)
if nan_mask.sum() > 0:
    print(f"⚠️  Warning: {nan_mask.sum()} rows contain NaN")
    X_long = X_long[~nan_mask]
    y_long = y_long[~nan_mask]
    print(f"   Removed NaN rows. Remaining: {len(X_long)}")
```

---

### 🟠 **10. 모델 버전 관리 부족**

**문제:**
- 단순 timestamp로만 저장
- 하이퍼파라미터, 성능 메트릭 기록 없음
- 실험 추적 어려움

**권장 개선:**
```python
# 모델 메타데이터 저장
metadata = {
    'timestamp': timestamp,
    'data_range': f"{df['timestamp'].min()} to {df['timestamp'].max()}",
    'train_samples': len(X_train),
    'test_samples': len(X_test),
    'positive_rate_train': np.sum(y_train) / len(y_train),
    'positive_rate_test': np.sum(y_test) / len(y_test),
    'hyperparameters': {
        'n_estimators': 300,
        'max_depth': 6,
        'learning_rate': 0.05,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'scale_pos_weight': scale_pos_weight
    },
    'metrics': metrics,
    'feature_count': len(long_feature_cols)
}

metadata_path = MODELS_DIR / f"xgboost_long_trade_outcome_full_{timestamp}_metadata.json"
with open(metadata_path, 'w') as f:
    json.dump(metadata, f, indent=2, default=str)
```

---

## 📊 개선 우선순위

### 🔴 Critical (즉시 개선 필요)
1. **Early Stopping 추가** - 과적합 방지
2. **Validation Set 분리** - 성능 모니터링

### 🟡 High Priority (단기 개선)
3. **Walk-Forward Validation** - 시계열 특성 반영
4. **Train/Test 성능 비교** - 과적합 확인
5. **포괄적 메트릭** - Precision/Recall/F1/AUC

### 🟢 Medium Priority (중기 개선)
6. **TimeSeriesSplit CV** - 안정성 검증
7. **Feature Importance** - 해석성 향상
8. **Hyperparameter Tuning** - 성능 최적화

### ⚪ Low Priority (장기 개선)
9. **NaN 처리 강화**
10. **모델 버전 관리**

---

## 💡 개선된 학습 코드 예시

```python
# Step 1: Train/Val/Test Split (60/20/20)
train_end = int(len(X) * 0.6)
val_end = int(len(X) * 0.8)

X_train, y_train = X[:train_end], y[:train_end]
X_val, y_val = X[train_end:val_end], y[train_end:val_end]
X_test, y_test = X[val_end:], y[val_end:]

# Step 2: Scale
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

# Step 3: Train with Early Stopping
model = XGBClassifier(
    n_estimators=1000,  # 충분히 크게 설정
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    scale_pos_weight=scale_pos_weight,
    random_state=42,
    eval_metric='logloss'
)

model.fit(
    X_train_scaled, y_train,
    eval_set=[(X_val_scaled, y_val)],
    early_stopping_rounds=50,
    verbose=False
)

print(f"Best iteration: {model.best_iteration}")
print(f"Best score: {model.best_score:.4f}")

# Step 4: Comprehensive Evaluation
y_train_pred = model.predict(X_train_scaled)
y_val_pred = model.predict(X_val_scaled)
y_test_pred = model.predict(X_test_scaled)

print("\n" + "="*60)
print("Model Performance Summary")
print("="*60)

for split, y_true, y_pred in [
    ("Train", y_train, y_train_pred),
    ("Val", y_val, y_val_pred),
    ("Test", y_test, y_test_pred)
]:
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)

    print(f"\n{split} Set:")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall:    {recall:.4f}")
    print(f"  F1:        {f1:.4f}")

# Step 5: Feature Importance
top_features = np.argsort(model.feature_importances_)[-20:]
print("\nTop 20 Features:")
for idx in reversed(top_features):
    print(f"  {feature_names[idx]}: {model.feature_importances_[idx]:.4f}")

# Step 6: Overfitting Check
train_test_gap = precision_score(y_train, y_train_pred) - precision_score(y_test, y_test_pred)
if train_test_gap > 0.1:
    print("\n⚠️  Warning: Potential overfitting detected!")
    print(f"   Train-Test gap: {train_test_gap:.4f}")
```

---

## 🎯 예상 개선 효과

### Early Stopping
- 학습 시간: -30~50%
- Test 성능: +2~5%
- 과적합 감소

### Validation Set
- 과적합 조기 발견
- 하이퍼파라미터 튜닝 가능
- 일반화 성능 향상

### Walk-Forward Validation
- 실전 성능과 일치도 향상
- 시계열 특성 반영
- 더 신뢰할 수 있는 성능 추정

### Hyperparameter Tuning
- Test 성능: +5~15%
- 최적 모델 탐색
- 데이터셋 특성 반영

---

## 📝 결론

**현재 학습 방식:**
- ✅ 기본적인 XGBoost 학습 구조는 올바름
- ✅ 클래스 불균형 처리 (scale_pos_weight)
- ✅ Feature scaling 적용

**주요 개선 필요:**
- 🔴 Early stopping 부재 → 과적합 위험
- 🔴 Validation set 부재 → 성능 모니터링 불가
- 🟡 시계열 특성 무시 → 실전 성능 괴리
- 🟡 단일 메트릭 의존 → 종합 평가 부족

**권장 조치:**
1. 즉시: Early stopping + Validation set 추가
2. 단기: 포괄적 메트릭 + 과적합 모니터링
3. 중기: Walk-forward + Hyperparameter tuning

이러한 개선을 통해 모델 성능과 안정성을 크게 향상시킬 수 있습니다.
