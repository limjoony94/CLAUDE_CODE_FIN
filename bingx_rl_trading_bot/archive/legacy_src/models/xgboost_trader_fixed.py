"""XGBoost 트레이더 - 수정 버전

주요 수정사항:
1. HOLD 의미 수정: "청산" → "포지션 유지"
2. 강제 청산 로직 추가
3. 수수료 계산 수정: 레버리지 이중 적용 제거
4. 포지션 관리 로직 개선
"""

from pathlib import Path
from typing import Dict, Any, Tuple, Optional
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import accuracy_score
import joblib
from loguru import logger


class XGBoostTraderFixed:
    """수정된 XGBoost 트레이더"""

    def __init__(
        self,
        lookahead: int = 60,  # 5시간 (개선)
        threshold_pct: float = 0.01,  # 1.0% (개선)
        confidence_threshold: float = 0.65,  # 더 보수적
        model_params: Optional[Dict[str, Any]] = None,
        model_dir: Optional[str] = None
    ):
        self.lookahead = lookahead
        self.threshold_pct = threshold_pct
        self.confidence_threshold = confidence_threshold

        self.params = model_params or self._default_params()

        if model_dir is None:
            project_root = Path(__file__).parent.parent.parent
            model_dir = project_root / 'data' / 'trained_models' / 'xgboost_fixed'

        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)

        self.model = None
        self.feature_columns = None

        logger.info(f"Fixed XGBoost Trader - Lookahead: {lookahead}, Threshold: {threshold_pct}")

    @staticmethod
    def _default_params() -> Dict[str, Any]:
        """클래스 불균형 해결 포함"""
        return {
            'objective': 'multi:softprob',
            'num_class': 3,
            'max_depth': 5,
            'learning_rate': 0.03,  # 더 낮게
            'subsample': 0.7,  # 더 강한 정규화
            'colsample_bytree': 0.7,
            'min_child_weight': 5,  # 더 높게
            'gamma': 0.2,
            'reg_alpha': 0.3,  # 더 강한 L1
            'reg_lambda': 2.0,  # 더 강한 L2
            'scale_pos_weight': 7.0,  # 클래스 불균형 해결 ⭐
            'eval_metric': 'mlogloss',
            'tree_method': 'hist',
            'random_state': 42
        }

    def backtest_fixed(
        self,
        df: pd.DataFrame,
        predictions: np.ndarray,
        initial_balance: float = 10000.0,
        position_size: float = 0.03,
        leverage: int = 3,
        transaction_fee: float = 0.0004,
        slippage: float = 0.0001,
        liquidation_threshold: float = 0.1
    ) -> Dict[str, Any]:
        """
        수정된 백테스팅

        주요 수정:
        1. HOLD = 포지션 유지
        2. 강제 청산 추가
        3. 수수료 계산 수정
        """
        logger.info(f"Starting FIXED backtest - Initial: ${initial_balance}, Leverage: {leverage}x")

        # 초기화
        balance = initial_balance
        position = 0.0
        entry_price = 0.0
        total_pnl = 0.0
        trades = []
        liquidated = False

        for i in range(len(df)):
            current_price = df.iloc[i]['close']
            action = predictions[i]  # -1, 0, 1

            # 🔴 수정 1: 강제 청산 체크 (최우선)
            if balance < initial_balance * liquidation_threshold:
                logger.warning(f"⚠️ LIQUIDATION at step {i}! Balance: ${balance:.2f}")

                # 모든 포지션 강제 청산
                if abs(position) > 0.001:
                    # 남은 잔액 전부 손실
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

            # 🔴 수정 2: HOLD = 포지션 유지
            if action == 1:  # LONG
                target_position = position_size
            elif action == -1:  # SHORT
                target_position = -position_size
            else:  # HOLD (action == 0)
                target_position = position  # 현재 포지션 유지! ⭐

            position_delta = target_position - position

            # 거래 필요한 경우만
            if abs(position_delta) > 0.001:
                execution_price = current_price * (1 + slippage * np.sign(position_delta))

                # 1. 기존 포지션 청산 (있으면)
                if abs(position) > 0.001:
                    close_size = abs(position)

                    # PnL 계산
                    price_diff = (execution_price - entry_price) * np.sign(position)
                    position_pnl = close_size * price_diff
                    leveraged_pnl = position_pnl * leverage

                    # 🔴 수정 3: 수수료 계산 (레버리지 없이)
                    notional_value = close_size * execution_price
                    close_fee = notional_value * transaction_fee

                    # 최종 손익
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

                # 2. 새 포지션 진입
                if abs(target_position) > 0.001:
                    open_size = abs(target_position)

                    # 필요 증거금
                    notional_value = open_size * execution_price
                    required_margin = notional_value / leverage

                    # 수수료 (레버리지 없이)
                    open_fee = notional_value * transaction_fee

                    # 증거금 충분한지 확인
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
                    else:
                        logger.warning(f"Insufficient margin at step {i}: "
                                     f"need ${required_margin + open_fee:.2f}, "
                                     f"have ${balance:.2f}")

        # 최종 청산
        if abs(position) > 0.001 and not liquidated:
            final_price = df.iloc[-1]['close']
            final_size = abs(position)

            # PnL
            price_diff = (final_price - entry_price) * np.sign(position)
            position_pnl = final_size * price_diff
            leveraged_pnl = position_pnl * leverage

            # 수수료
            notional_value = final_size * final_price
            final_fee = notional_value * transaction_fee

            # 최종 손익
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

        logger.info(f"\n✅ FIXED Backtest Results:")
        logger.info(f"Final Balance: ${results['final_balance']:.2f}")
        logger.info(f"Total Return: {results['total_return_pct']:.2f}%")
        logger.info(f"Number of Trades: {results['num_trades']}")
        logger.info(f"Win Rate: {results['win_rate']*100:.1f}%")
        logger.info(f"Max Drawdown: ${results['max_drawdown']:.2f}")
        logger.info(f"Liquidated: {results['liquidated']}")

        return results

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
        """타겟 생성 - 개선된 lookahead 사용"""
        data = df.copy()

        data['future_return'] = data['close'].pct_change(self.lookahead).shift(-self.lookahead)

        data['target'] = 0

        data.loc[data['future_return'] > self.threshold_pct, 'target'] = 1
        data.loc[data['future_return'] < -self.threshold_pct, 'target'] = -1

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
        """XGBoost 훈련"""
        logger.info("Starting FIXED XGBoost training...")

        X_train = train_df[self.feature_columns].values
        y_train = train_df['target'].values + 1

        X_val = val_df[self.feature_columns].values
        y_val = val_df['target'].values + 1

        dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=self.feature_columns)
        dval = xgb.DMatrix(X_val, label=y_val, feature_names=self.feature_columns)

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

        val_pred = self.model.predict(dval)
        val_pred_class = np.argmax(val_pred, axis=1)
        val_accuracy = accuracy_score(y_val, val_pred_class)

        logger.info(f"Training completed - Best iteration: {self.model.best_iteration}")
        logger.info(f"Validation accuracy: {val_accuracy:.4f}")

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
        """예측 수행"""
        if self.model is None:
            raise ValueError("Model not trained or loaded")

        X = df[self.feature_columns].values
        dtest = xgb.DMatrix(X, feature_names=self.feature_columns)

        proba = self.model.predict(dtest)

        if use_confidence_threshold:
            predictions = []
            for p in proba:
                max_prob = np.max(p)
                if max_prob >= self.confidence_threshold:
                    pred_class = np.argmax(p) - 1
                else:
                    pred_class = 0
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
        """모델 평가 - backtest_fixed 사용"""
        predictions, proba = self.predict(df, use_confidence_threshold=True)

        y_true = df['target'].values

        accuracy = accuracy_score(y_true, predictions)

        from sklearn.metrics import classification_report, confusion_matrix

        report = classification_report(y_true, predictions,
                                      target_names=['SHORT', 'HOLD', 'LONG'],
                                      output_dict=True, zero_division=0)

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

    def save_model(self, filename: str = 'xgboost_fixed') -> None:
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
            'threshold_pct': self.threshold_pct,
            'confidence_threshold': self.confidence_threshold,
            'feature_importance': self.feature_importance
        }
        joblib.dump(config, config_path)

        logger.info(f"Model saved to {model_path}")

    def load_model(self, filename: str = 'xgboost_fixed') -> None:
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
        self.threshold_pct = config['threshold_pct']
        self.confidence_threshold = config['confidence_threshold']
        self.feature_importance = config['feature_importance']

        logger.info(f"Model loaded from {model_path}")
