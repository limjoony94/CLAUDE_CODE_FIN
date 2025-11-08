"""XGBoost 트레이더 - 회귀 버전

핵심 개선:
1. 분류 → 회귀 문제로 전환
2. 클래스 불균형 문제 근본 해결
3. SMOTE 불필요
4. 실제 수익률 직접 예측
5. 모든 백테스팅 버그 수정 유지
"""

from pathlib import Path
from typing import Dict, Any, Tuple, Optional
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib
from loguru import logger


class XGBoostTraderRegression:
    """회귀 기반 XGBoost 트레이더"""

    def __init__(
        self,
        lookahead: int = 48,  # 4시간 (균형잡힌 선택)
        long_threshold: float = 0.015,  # 1.5% 이상 → LONG
        short_threshold: float = -0.015,  # -1.5% 이하 → SHORT
        confidence_multiplier: float = 0.5,  # 예측 강도에 따라 조정
        model_params: Optional[Dict[str, Any]] = None,
        model_dir: Optional[str] = None
    ):
        self.lookahead = lookahead
        self.long_threshold = long_threshold
        self.short_threshold = short_threshold
        self.confidence_multiplier = confidence_multiplier

        self.params = model_params or self._default_params()

        if model_dir is None:
            project_root = Path(__file__).parent.parent.parent
            model_dir = project_root / 'data' / 'trained_models' / 'xgboost_regression'

        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)

        self.model = None
        self.feature_columns = None

        logger.info(f"Regression XGBoost Trader - Lookahead: {lookahead}h, Thresholds: {short_threshold:.1%}/{long_threshold:.1%}")

    @staticmethod
    def _default_params() -> Dict[str, Any]:
        """회귀 모델 기본 파라미터"""
        return {
            'objective': 'reg:squarederror',  # 회귀 ⭐
            'max_depth': 6,
            'learning_rate': 0.02,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'min_child_weight': 3,
            'gamma': 0.1,
            'reg_alpha': 0.2,
            'reg_lambda': 1.5,
            'eval_metric': 'rmse',
            'tree_method': 'hist',
            'random_state': 42
        }

    def prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Feature Engineering"""
        data = df.copy()

        base_features = [
            'rsi', 'macd', 'macd_signal',
            'bb_upper', 'bb_middle', 'bb_lower',
            'ema_9', 'ema_21', 'ema_50',
            'atr', 'adx', 'stoch_k', 'stoch_d',
            'volume'
        ]

        data['close_vs_ema9'] = (data['close'] - data['ema_9']) / data['ema_9']
        data['close_vs_ema21'] = (data['close'] - data['ema_21']) / data['ema_21']
        data['close_vs_ema50'] = (data['close'] - data['ema_50']) / data['ema_50']

        data['close_vs_bb_upper'] = (data['close'] - data['bb_upper']) / data['bb_upper']
        data['close_vs_bb_lower'] = (data['close'] - data['bb_lower']) / data['bb_lower']
        data['bb_position'] = (data['close'] - data['bb_lower']) / (data['bb_upper'] - data['bb_lower'] + 1e-8)

        data['macd_histogram'] = data['macd'] - data['macd_signal']
        data['macd_trend'] = data['macd'].diff()

        data['volume_ma20'] = data['volume'].rolling(20).mean()
        data['volume_ratio'] = data['volume'] / (data['volume_ma20'] + 1e-8)

        data['rsi_ma'] = data['rsi'].rolling(5).mean()
        data['rsi_trend'] = data['rsi'].diff()

        data['trend_strength'] = data['close'].pct_change(50).clip(-0.1, 0.1) * 10
        data['volatility'] = data['close'].pct_change().rolling(20).std() / 0.002

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
        """타겟 생성 - 연속값 (future_return)"""
        data = df.copy()

        # 회귀 타겟: 미래 수익률 직접 예측 ⭐
        data['target'] = data['close'].pct_change(self.lookahead).shift(-self.lookahead)

        # 통계
        logger.info(f"Target Statistics:")
        logger.info(f"  Mean: {data['target'].mean()*100:.3f}%")
        logger.info(f"  Std: {data['target'].std()*100:.3f}%")
        logger.info(f"  Min: {data['target'].min()*100:.2f}%")
        logger.info(f"  Max: {data['target'].max()*100:.2f}%")

        # 참고용 신호 분포 (실제 학습에는 사용 안함)
        signals = np.zeros(len(data))
        signals[data['target'] > self.long_threshold] = 1
        signals[data['target'] < self.short_threshold] = -1

        unique, counts = np.unique(signals, return_counts=True)
        total = len(signals)
        logger.info(f"Signal Distribution (for reference):")
        for sig, count in zip(unique, counts):
            sig_name = {-1: 'SHORT', 0: 'HOLD', 1: 'LONG'}[sig]
            logger.info(f"  {sig_name}: {count/total*100:.1f}%")

        return data

    def prepare_data(
        self,
        df: pd.DataFrame,
        train_ratio: float = 0.7,
        val_ratio: float = 0.15
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """데이터 준비"""
        data = self.prepare_features(df)
        data = self.create_targets(data)
        data = data.dropna()

        logger.info(f"Total samples after cleaning: {len(data)}")

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
        """XGBoost 회귀 훈련"""
        logger.info("Starting REGRESSION XGBoost training...")

        X_train = train_df[self.feature_columns].values
        y_train = train_df['target'].values  # 연속값 ⭐

        X_val = val_df[self.feature_columns].values
        y_val = val_df['target'].values

        # DMatrix
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

        self.feature_importance = self.model.get_score(importance_type='gain')

        # 검증 세트 평가
        val_pred = self.model.predict(dval)

        rmse = np.sqrt(mean_squared_error(y_val, val_pred))
        mae = mean_absolute_error(y_val, val_pred)
        r2 = r2_score(y_val, val_pred)

        logger.info(f"Training completed - Best iteration: {self.model.best_iteration}")
        logger.info(f"Validation RMSE: {rmse*100:.3f}%")
        logger.info(f"Validation MAE: {mae*100:.3f}%")
        logger.info(f"Validation R²: {r2:.4f}")

        importance_sorted = sorted(self.feature_importance.items(), key=lambda x: x[1], reverse=True)
        logger.info("Top 10 important features:")
        for feat, score in importance_sorted[:10]:
            logger.info(f"  {feat}: {score:.2f}")

        return {
            'best_iteration': self.model.best_iteration,
            'train_rmse': np.sqrt(evals_result['train']['rmse'][-1]),
            'val_rmse': rmse,
            'val_mae': mae,
            'val_r2': r2,
            'feature_importance': self.feature_importance
        }

    def predict(
        self,
        df: pd.DataFrame,
        use_confidence_multiplier: bool = True
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        예측 수행

        Returns:
            signals: {-1: SHORT, 0: HOLD, 1: LONG}
            predictions: 예측된 수익률 (연속값)
        """
        if self.model is None:
            raise ValueError("Model not trained or loaded")

        X = df[self.feature_columns].values
        dtest = xgb.DMatrix(X, feature_names=self.feature_columns)

        # 수익률 예측 (연속값)
        predictions = self.model.predict(dtest)

        # 임계값 기반 신호 생성
        signals = np.zeros(len(predictions))

        if use_confidence_multiplier:
            # 예측 강도에 따라 동적 임계값 조정
            for i, pred in enumerate(predictions):
                abs_pred = abs(pred)

                # 예측이 강할수록 임계값 낮춤
                dynamic_long_threshold = self.long_threshold * (1 - self.confidence_multiplier * min(abs_pred / 0.05, 1.0))
                dynamic_short_threshold = self.short_threshold * (1 - self.confidence_multiplier * min(abs_pred / 0.05, 1.0))

                if pred > dynamic_long_threshold:
                    signals[i] = 1  # LONG
                elif pred < dynamic_short_threshold:
                    signals[i] = -1  # SHORT
                else:
                    signals[i] = 0  # HOLD
        else:
            # 고정 임계값
            signals[predictions > self.long_threshold] = 1
            signals[predictions < self.short_threshold] = -1

        return signals.astype(int), predictions

    def evaluate(
        self,
        df: pd.DataFrame,
        name: str = "Test"
    ) -> Dict[str, Any]:
        """모델 평가 - 회귀 및 신호 정확도"""
        signals, predictions = self.predict(df, use_confidence_multiplier=True)

        y_true = df['target'].values

        # 회귀 메트릭
        rmse = np.sqrt(mean_squared_error(y_true, predictions))
        mae = mean_absolute_error(y_true, predictions)
        r2 = r2_score(y_true, predictions)

        # 신호 정확도
        true_signals = np.zeros(len(y_true))
        true_signals[y_true > self.long_threshold] = 1
        true_signals[y_true < self.short_threshold] = -1

        signal_accuracy = np.mean(signals == true_signals)

        # 방향 정확도 (부호만)
        direction_accuracy = np.mean(np.sign(predictions) == np.sign(y_true))

        logger.info(f"\n{name} Set Evaluation:")
        logger.info(f"Regression Metrics:")
        logger.info(f"  RMSE: {rmse*100:.3f}%")
        logger.info(f"  MAE: {mae*100:.3f}%")
        logger.info(f"  R²: {r2:.4f}")
        logger.info(f"\nSignal Metrics:")
        logger.info(f"  Signal Accuracy: {signal_accuracy*100:.2f}%")
        logger.info(f"  Direction Accuracy: {direction_accuracy*100:.2f}%")

        # 신호 분포
        unique, counts = np.unique(signals, return_counts=True)
        logger.info(f"\nPredicted Signal Distribution:")
        for sig, count in zip(unique, counts):
            sig_name = {-1: 'SHORT', 0: 'HOLD', 1: 'LONG'}[int(sig)]
            logger.info(f"  {sig_name}: {count}/{len(signals)} ({count/len(signals)*100:.1f}%)")

        return {
            'rmse': rmse,
            'mae': mae,
            'r2': r2,
            'signal_accuracy': signal_accuracy,
            'direction_accuracy': direction_accuracy,
            'signals': signals,
            'predictions': predictions
        }

    def backtest_fixed(
        self,
        df: pd.DataFrame,
        signals: np.ndarray,
        initial_balance: float = 10000.0,
        position_size: float = 0.03,
        leverage: int = 3,
        transaction_fee: float = 0.0004,
        slippage: float = 0.0001,
        liquidation_threshold: float = 0.1
    ) -> Dict[str, Any]:
        """수정된 백테스팅 (Fixed 버전과 동일)"""
        logger.info(f"Starting REGRESSION backtest - Initial: ${initial_balance}, Leverage: {leverage}x")

        balance = initial_balance
        position = 0.0
        entry_price = 0.0
        total_pnl = 0.0
        trades = []
        liquidated = False

        for i in range(len(df)):
            current_price = df.iloc[i]['close']
            action = signals[i]

            # 강제 청산 체크
            if balance < initial_balance * liquidation_threshold:
                logger.warning(f"⚠️ LIQUIDATION at step {i}! Balance: ${balance:.2f}")

                if abs(position) > 0.001:
                    liquidation_loss = -balance
                    total_pnl += liquidation_loss

                    trades.append({
                        'step': i,
                        'action': 'liquidation',
                        'size': position,
                        'price': current_price,
                        'pnl': liquidation_loss
                    })

                    balance = 0.0
                    position = 0.0

                liquidated = True
                break

            # HOLD = 포지션 유지
            if action == 1:
                target_position = position_size
            elif action == -1:
                target_position = -position_size
            else:
                target_position = position

            position_delta = target_position - position

            if abs(position_delta) > 0.001:
                execution_price = current_price * (1 + slippage * np.sign(position_delta))

                # 기존 포지션 청산
                if abs(position) > 0.001:
                    close_size = abs(position)

                    price_diff = (execution_price - entry_price) * np.sign(position)
                    position_pnl = close_size * price_diff
                    leveraged_pnl = position_pnl * leverage

                    notional_value = close_size * execution_price
                    close_fee = notional_value * transaction_fee

                    realized_pnl = leveraged_pnl - close_fee

                    balance += realized_pnl
                    total_pnl += realized_pnl

                    trades.append({
                        'step': i,
                        'action': 'close',
                        'size': close_size,
                        'price': execution_price,
                        'pnl': realized_pnl
                    })

                    position = 0.0
                    entry_price = 0.0

                # 새 포지션 진입
                if abs(target_position) > 0.001:
                    open_size = abs(target_position)

                    notional_value = open_size * execution_price
                    required_margin = notional_value / leverage

                    open_fee = notional_value * transaction_fee

                    if required_margin + open_fee <= balance:
                        position = target_position
                        entry_price = execution_price
                        balance -= open_fee

                        trades.append({
                            'step': i,
                            'action': 'open',
                            'size': target_position,
                            'price': execution_price,
                            'pnl': 0
                        })

        # 최종 청산
        if abs(position) > 0.001 and not liquidated:
            final_price = df.iloc[-1]['close']
            final_size = abs(position)

            price_diff = (final_price - entry_price) * np.sign(position)
            position_pnl = final_size * price_diff
            leveraged_pnl = position_pnl * leverage

            notional_value = final_size * final_price
            final_fee = notional_value * transaction_fee

            final_pnl = leveraged_pnl - final_fee

            balance += final_pnl
            total_pnl += final_pnl

            trades.append({
                'step': len(df) - 1,
                'action': 'final_close',
                'size': position,
                'price': final_price,
                'pnl': final_pnl
            })

        # 통계
        portfolio_value = max(balance, 0.0)
        total_return = (portfolio_value - initial_balance) / initial_balance

        close_trades = [t for t in trades if t['action'] in ['close', 'final_close']]
        trade_pnls = [t['pnl'] for t in close_trades]
        win_trades = [p for p in trade_pnls if p > 0]

        results = {
            'initial_balance': initial_balance,
            'final_balance': portfolio_value,
            'total_pnl': total_pnl,
            'total_return_pct': total_return * 100,
            'num_trades': len(close_trades),
            'win_trades': len(win_trades),
            'win_rate': len(win_trades) / max(len(trade_pnls), 1),
            'avg_win': np.mean(win_trades) if win_trades else 0,
            'avg_loss': np.mean([p for p in trade_pnls if p < 0]) if any(p < 0 for p in trade_pnls) else 0,
            'max_drawdown': min(trade_pnls) if trade_pnls else 0,
            'liquidated': liquidated,
            'trades': trades
        }

        logger.info(f"\n✅ REGRESSION Backtest Results:")
        logger.info(f"Final Balance: ${results['final_balance']:.2f}")
        logger.info(f"Total Return: {results['total_return_pct']:.2f}%")
        logger.info(f"Number of Trades: {results['num_trades']}")
        logger.info(f"Win Rate: {results['win_rate']*100:.1f}%")
        logger.info(f"Avg Win: ${results['avg_win']:.2f}")
        logger.info(f"Avg Loss: ${results['avg_loss']:.2f}")
        logger.info(f"Max Drawdown: ${results['max_drawdown']:.2f}")
        logger.info(f"Liquidated: {results['liquidated']}")

        return results

    def save_model(self, filename: str = 'xgboost_regression') -> None:
        """모델 저장"""
        if self.model is None:
            logger.warning("No model to save")
            return

        model_path = self.model_dir / f"{filename}.json"
        config_path = self.model_dir / f"{filename}_config.pkl"

        self.model.save_model(str(model_path))

        config = {
            'feature_columns': self.feature_columns,
            'params': self.params,
            'lookahead': self.lookahead,
            'long_threshold': self.long_threshold,
            'short_threshold': self.short_threshold,
            'confidence_multiplier': self.confidence_multiplier,
            'feature_importance': self.feature_importance
        }
        joblib.dump(config, config_path)

        logger.info(f"Model saved to {model_path}")

    def load_model(self, filename: str = 'xgboost_regression') -> None:
        """모델 로드"""
        model_path = self.model_dir / f"{filename}.json"
        config_path = self.model_dir / f"{filename}_config.pkl"

        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")

        self.model = xgb.Booster()
        self.model.load_model(str(model_path))

        config = joblib.load(config_path)
        self.feature_columns = config['feature_columns']
        self.params = config['params']
        self.lookahead = config['lookahead']
        self.long_threshold = config['long_threshold']
        self.short_threshold = config['short_threshold']
        self.confidence_multiplier = config.get('confidence_multiplier', 0.5)
        self.feature_importance = config['feature_importance']

        logger.info(f"Model loaded from {model_path}")
