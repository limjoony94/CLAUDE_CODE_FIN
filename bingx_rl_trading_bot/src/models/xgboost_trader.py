"""XGBoost 기반 트레이딩 모델

강화학습 대비 장점:
1. 적은 데이터로 학습 가능 (10K-50K vs 100K+)
2. 강력한 과적합 방지 (정규화, early stopping)
3. 빠른 훈련 속도 (5-30분 vs 5-20시간)
4. 해석 가능성 (feature importance)
5. 안정적 일반화 성능
"""

from pathlib import Path
from typing import Dict, Any, Tuple, Optional, List
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
from loguru import logger


class XGBoostTrader:
    """
    XGBoost 기반 3-class 분류 트레이더

    행동: {-1: SHORT, 0: HOLD, 1: LONG}
    """

    def __init__(
        self,
        lookahead: int = 5,
        threshold_pct: float = 0.002,
        confidence_threshold: float = 0.55,
        model_params: Optional[Dict[str, Any]] = None,
        model_dir: Optional[str] = None
    ):
        """
        Args:
            lookahead: 미래 N봉 수익률 예측
            threshold_pct: 분류 임계값 (0.2% = 수수료 + 마진)
            confidence_threshold: 예측 확신도 임계값 (높을수록 보수적)
            model_params: XGBoost 하이퍼파라미터
            model_dir: 모델 저장 디렉토리
        """
        self.lookahead = lookahead
        self.threshold_pct = threshold_pct
        self.confidence_threshold = confidence_threshold

        # XGBoost 파라미터 (과적합 방지 최적화)
        self.params = model_params or self._default_params()

        # 모델 저장 경로
        if model_dir is None:
            project_root = Path(__file__).parent.parent.parent
            model_dir = project_root / 'data' / 'trained_models' / 'xgboost'

        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)

        self.model = None
        self.feature_columns = None
        self.feature_importance = None

        logger.info(f"XGBoost Trader initialized - Lookahead: {lookahead}, Threshold: {threshold_pct}")

    @staticmethod
    def _default_params() -> Dict[str, Any]:
        """기본 XGBoost 파라미터 (과적합 방지 최적화)"""
        return {
            'objective': 'multi:softprob',
            'num_class': 3,
            'max_depth': 5,           # 깊이 제한 (과적합 방지)
            'learning_rate': 0.05,    # 낮은 학습률 (안정성)
            'subsample': 0.8,         # 행 샘플링 (다양성)
            'colsample_bytree': 0.8,  # 열 샘플링 (다양성)
            'min_child_weight': 3,    # 최소 샘플 수 (과적합 방지)
            'gamma': 0.1,             # 분할 최소 손실 감소
            'reg_alpha': 0.1,         # L1 정규화
            'reg_lambda': 1.0,        # L2 정규화
            'eval_metric': 'mlogloss',
            'tree_method': 'hist',    # 빠른 훈련
            'random_state': 42
        }

    def prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Feature Engineering

        기존 지표 활용 + 추가 파생 특성
        """
        data = df.copy()

        # 1. 기존 기술적 지표 (8종)
        base_features = [
            'rsi', 'macd', 'macd_signal',
            'bb_upper', 'bb_middle', 'bb_lower',
            'ema_9', 'ema_21', 'ema_50',
            'atr', 'adx', 'stoch_k', 'stoch_d',
            'volume'
        ]

        # 2. 파생 특성 (상대적 위치 중요)

        # 가격 vs 이동평균
        data['close_vs_ema9'] = (data['close'] - data['ema_9']) / data['ema_9']
        data['close_vs_ema21'] = (data['close'] - data['ema_21']) / data['ema_21']
        data['close_vs_ema50'] = (data['close'] - data['ema_50']) / data['ema_50']

        # 가격 vs 볼린저밴드
        data['close_vs_bb_upper'] = (data['close'] - data['bb_upper']) / data['bb_upper']
        data['close_vs_bb_lower'] = (data['close'] - data['bb_lower']) / data['bb_lower']
        data['bb_position'] = (data['close'] - data['bb_lower']) / (data['bb_upper'] - data['bb_lower'] + 1e-8)

        # MACD 관계
        data['macd_histogram'] = data['macd'] - data['macd_signal']
        data['macd_trend'] = data['macd'].diff()

        # 볼륨
        data['volume_ma20'] = data['volume'].rolling(20).mean()
        data['volume_ratio'] = data['volume'] / (data['volume_ma20'] + 1e-8)

        # RSI 파생
        data['rsi_ma'] = data['rsi'].rolling(5).mean()
        data['rsi_trend'] = data['rsi'].diff()

        # 시장 체제 특성 (V4 환경에서 영감)
        data['trend_strength'] = data['close'].pct_change(50).clip(-0.1, 0.1) * 10  # -1~1
        data['volatility'] = data['close'].pct_change().rolling(20).std() / 0.002  # 정규화

        # 3. 최종 Feature 리스트
        self.feature_columns = base_features + [
            'close_vs_ema9', 'close_vs_ema21', 'close_vs_ema50',
            'close_vs_bb_upper', 'close_vs_bb_lower', 'bb_position',
            'macd_histogram', 'macd_trend',
            'volume_ratio', 'rsi_ma', 'rsi_trend',
            'trend_strength', 'volatility'
        ]

        logger.info(f"Prepared {len(self.feature_columns)} features")

        return data

    def create_targets(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        타겟 생성: 미래 수익률 기반 3-class 분류

        Returns:
            df with 'target' column: {-1: SHORT, 0: HOLD, 1: LONG}
        """
        data = df.copy()

        # 미래 N봉 수익률
        data['future_return'] = data['close'].pct_change(self.lookahead).shift(-self.lookahead)

        # 3-class 분류
        data['target'] = 0  # HOLD (기본)

        # LONG: 수익률 > threshold (수수료 + 이익)
        data.loc[data['future_return'] > self.threshold_pct, 'target'] = 1

        # SHORT: 수익률 < -threshold
        data.loc[data['future_return'] < -self.threshold_pct, 'target'] = -1

        # 통계 로깅
        counts = data['target'].value_counts()
        total = len(data)
        logger.info(f"Target distribution - LONG: {counts.get(1, 0)/total*100:.1f}%, "
                   f"HOLD: {counts.get(0, 0)/total*100:.1f}%, "
                   f"SHORT: {counts.get(-1, 0)/total*100:.1f}%")

        return data

    def prepare_data(
        self,
        df: pd.DataFrame,
        train_ratio: float = 0.7,
        val_ratio: float = 0.15
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        데이터 준비: Feature Engineering + 분할

        Returns:
            (train_df, val_df, test_df)
        """
        # Feature Engineering
        data = self.prepare_features(df)

        # 타겟 생성
        data = self.create_targets(data)

        # NaN 제거 (초기 지표 계산 + 미래 수익률)
        data = data.dropna()

        logger.info(f"Total samples after cleaning: {len(data)}")

        # 시계열 분할 (순서 유지)
        n = len(data)
        train_end = int(n * train_ratio)
        val_end = int(n * (train_ratio + val_ratio))

        train_df = data.iloc[:train_end].copy()
        val_df = data.iloc[train_end:val_end].copy()
        test_df = data.iloc[val_end:].copy()

        logger.info(f"Data split - Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")

        return train_df, val_df, test_df

    def train(
        self,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        num_boost_round: int = 500,
        early_stopping_rounds: int = 50,
        verbose: bool = True
    ) -> Dict[str, Any]:
        """
        XGBoost 모델 훈련

        Returns:
            훈련 통계
        """
        logger.info("Starting XGBoost training...")

        # 데이터 준비
        X_train = train_df[self.feature_columns].values
        y_train = train_df['target'].values + 1  # {-1,0,1} → {0,1,2}

        X_val = val_df[self.feature_columns].values
        y_val = val_df['target'].values + 1

        # DMatrix 생성
        dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=self.feature_columns)
        dval = xgb.DMatrix(X_val, label=y_val, feature_names=self.feature_columns)

        # 훈련
        evals = [(dtrain, 'train'), (dval, 'val')]
        evals_result = {}

        self.model = xgb.train(
            self.params,
            dtrain,
            num_boost_round=num_boost_round,
            evals=evals,
            early_stopping_rounds=early_stopping_rounds,
            evals_result=evals_result,
            verbose_eval=50 if verbose else False
        )

        # Feature Importance
        self.feature_importance = self.model.get_score(importance_type='gain')

        # 검증 세트 평가
        val_pred = self.model.predict(dval)
        val_pred_class = np.argmax(val_pred, axis=1)
        val_accuracy = accuracy_score(y_val, val_pred_class)

        logger.info(f"Training completed - Best iteration: {self.model.best_iteration}")
        logger.info(f"Validation accuracy: {val_accuracy:.4f}")

        # Feature Importance Top 10
        importance_sorted = sorted(self.feature_importance.items(), key=lambda x: x[1], reverse=True)
        logger.info("Top 10 important features:")
        for feat, score in importance_sorted[:10]:
            logger.info(f"  {feat}: {score:.2f}")

        return {
            'best_iteration': self.model.best_iteration,
            'train_loss': evals_result['train']['mlogloss'][-1],
            'val_loss': evals_result['val']['mlogloss'][-1],
            'val_accuracy': val_accuracy,
            'feature_importance': self.feature_importance
        }

    def predict(
        self,
        df: pd.DataFrame,
        use_confidence_threshold: bool = True
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        예측 수행

        Args:
            df: 입력 데이터프레임
            use_confidence_threshold: 확신도 임계값 사용 여부

        Returns:
            (predictions, probabilities)
            predictions: {-1: SHORT, 0: HOLD, 1: LONG}
            probabilities: shape (n_samples, 3)
        """
        if self.model is None:
            raise ValueError("Model not trained or loaded")

        X = df[self.feature_columns].values
        dtest = xgb.DMatrix(X, feature_names=self.feature_columns)

        # 예측 확률
        proba = self.model.predict(dtest)

        # Class 예측
        if use_confidence_threshold:
            # 확신도 높은 경우만 거래
            predictions = []
            for p in proba:
                max_prob = np.max(p)
                if max_prob >= self.confidence_threshold:
                    pred_class = np.argmax(p) - 1  # {0,1,2} → {-1,0,1}
                else:
                    pred_class = 0  # HOLD
                predictions.append(pred_class)
            predictions = np.array(predictions)
        else:
            predictions = np.argmax(proba, axis=1) - 1

        return predictions, proba

    def evaluate(
        self,
        df: pd.DataFrame,
        name: str = "Test"
    ) -> Dict[str, Any]:
        """
        모델 평가

        Returns:
            평가 지표
        """
        # 예측
        predictions, proba = self.predict(df, use_confidence_threshold=True)

        # 실제 타겟
        y_true = df['target'].values

        # 정확도
        accuracy = accuracy_score(y_true, predictions)

        # 클래스별 통계
        report = classification_report(y_true, predictions,
                                      target_names=['SHORT', 'HOLD', 'LONG'],
                                      output_dict=True, zero_division=0)

        # Confusion Matrix
        cm = confusion_matrix(y_true, predictions, labels=[-1, 0, 1])

        logger.info(f"\n{name} Set Evaluation:")
        logger.info(f"Accuracy: {accuracy:.4f}")
        logger.info(f"\nClassification Report:")
        logger.info(classification_report(y_true, predictions,
                                         target_names=['SHORT', 'HOLD', 'LONG'],
                                         zero_division=0))
        logger.info(f"\nConfusion Matrix:")
        logger.info(f"           SHORT  HOLD  LONG")
        for i, row_name in enumerate(['SHORT', 'HOLD', 'LONG']):
            logger.info(f"{row_name:5s}  {cm[i]}")

        return {
            'accuracy': accuracy,
            'classification_report': report,
            'confusion_matrix': cm.tolist(),
            'predictions': predictions,
            'probabilities': proba
        }

    def backtest(
        self,
        df: pd.DataFrame,
        initial_balance: float = 10000.0,
        position_size: float = 0.03,  # BTC
        leverage: int = 3,
        transaction_fee: float = 0.0004,
        slippage: float = 0.0001
    ) -> Dict[str, Any]:
        """
        백테스팅

        Returns:
            백테스팅 결과
        """
        logger.info(f"Starting backtest - Initial: ${initial_balance}, Leverage: {leverage}x")

        # 예측
        predictions, _ = self.predict(df, use_confidence_threshold=True)

        # 시뮬레이션
        balance = initial_balance
        position = 0.0
        entry_price = 0.0
        total_pnl = 0.0
        trades = []

        for i in range(len(df)):
            current_price = df.iloc[i]['close']
            action = predictions[i]

            # 포지션 조정
            target_position = action * position_size
            position_change = target_position - position

            if abs(position_change) > 0.001:
                # 거래 실행
                execution_price = current_price * (1 + slippage * np.sign(position_change))
                trade_value = abs(position_change) * execution_price * leverage
                fee = trade_value * transaction_fee

                # 기존 포지션 청산
                realized_pnl = 0.0
                if abs(position) > 0.001 and np.sign(position_change) != np.sign(position):
                    close_size = min(abs(position_change), abs(position))
                    price_diff = (execution_price - entry_price) * np.sign(position)
                    realized_pnl = close_size * price_diff * leverage - fee

                    balance += realized_pnl
                    total_pnl += realized_pnl

                    trades.append({
                        'step': i,
                        'type': 'close',
                        'size': close_size,
                        'price': execution_price,
                        'pnl': realized_pnl
                    })

                # 새 포지션
                remaining = position_change
                if abs(position) > 0.001 and np.sign(position_change) != np.sign(position):
                    remaining = position_change + position

                if abs(remaining) > 0.001:
                    required_margin = abs(remaining) * execution_price / leverage
                    if required_margin <= balance:
                        position += remaining
                        entry_price = execution_price
                        balance -= fee

                        trades.append({
                            'step': i,
                            'type': 'open',
                            'size': remaining,
                            'price': execution_price,
                            'pnl': 0
                        })

        # 최종 청산
        if abs(position) > 0.001:
            final_price = df.iloc[-1]['close']
            price_diff = (final_price - entry_price) * np.sign(position)
            final_pnl = abs(position) * price_diff * leverage
            balance += final_pnl
            total_pnl += final_pnl

        # 통계
        portfolio_value = balance
        total_return = (portfolio_value - initial_balance) / initial_balance

        trade_pnls = [t['pnl'] for t in trades if t['type'] == 'close']
        win_trades = [p for p in trade_pnls if p > 0]

        # 샤프 비율 (일별 수익률 기준)
        returns = df['close'].pct_change().fillna(0).values
        portfolio_returns = []
        current_pos = 0
        for i, pred in enumerate(predictions):
            if i > 0:
                ret = returns[i] * current_pos * leverage
                portfolio_returns.append(ret)
            current_pos = pred * position_size

        sharpe = np.mean(portfolio_returns) / (np.std(portfolio_returns) + 1e-8) * np.sqrt(365 * 24 * 12)

        results = {
            'initial_balance': initial_balance,
            'final_balance': portfolio_value,
            'total_pnl': total_pnl,
            'total_return_pct': total_return * 100,
            'num_trades': len([t for t in trades if t['type'] == 'close']),
            'win_trades': len(win_trades),
            'win_rate': len(win_trades) / max(len(trade_pnls), 1),
            'avg_win': np.mean(win_trades) if win_trades else 0,
            'avg_loss': np.mean([p for p in trade_pnls if p < 0]) if any(p < 0 for p in trade_pnls) else 0,
            'sharpe_ratio': sharpe,
            'trades': trades
        }

        logger.info(f"\nBacktest Results:")
        logger.info(f"Final Balance: ${results['final_balance']:.2f}")
        logger.info(f"Total Return: {results['total_return_pct']:.2f}%")
        logger.info(f"Number of Trades: {results['num_trades']}")
        logger.info(f"Win Rate: {results['win_rate']*100:.1f}%")
        logger.info(f"Sharpe Ratio: {results['sharpe_ratio']:.2f}")

        return results

    def save_model(self, filename: str = 'xgboost_model') -> None:
        """모델 저장"""
        if self.model is None:
            logger.warning("No model to save")
            return

        model_path = self.model_dir / f"{filename}.json"
        config_path = self.model_dir / f"{filename}_config.pkl"

        # XGBoost 모델
        self.model.save_model(str(model_path))

        # 설정
        config = {
            'feature_columns': self.feature_columns,
            'params': self.params,
            'lookahead': self.lookahead,
            'threshold_pct': self.threshold_pct,
            'confidence_threshold': self.confidence_threshold,
            'feature_importance': self.feature_importance
        }
        joblib.dump(config, config_path)

        logger.info(f"Model saved to {model_path}")

    def load_model(self, filename: str = 'xgboost_model') -> None:
        """모델 로드"""
        model_path = self.model_dir / f"{filename}.json"
        config_path = self.model_dir / f"{filename}_config.pkl"

        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")

        # XGBoost 모델
        self.model = xgb.Booster()
        self.model.load_model(str(model_path))

        # 설정
        config = joblib.load(config_path)
        self.feature_columns = config['feature_columns']
        self.params = config['params']
        self.lookahead = config['lookahead']
        self.threshold_pct = config['threshold_pct']
        self.confidence_threshold = config['confidence_threshold']
        self.feature_importance = config['feature_importance']

        logger.info(f"Model loaded from {model_path}")
