"""
Institutional-Grade Trading Logger

금융기관 및 전문 퀀트 트레이딩 그룹 표준 로깅 시스템
- ISO 8601 timestamps
- Structured JSON logging
- Performance metrics (Sharpe, Sortino, MDD, Win Rate)
- Trade audit trail
- Risk metrics (VaR, Exposure, Leverage)
- Market state tracking
- Model metrics
- Compliance logging
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, List
from enum import Enum
import pandas as pd
import numpy as np


class LogLevel(Enum):
    """Log severity levels aligned with institutional standards"""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"
    AUDIT = "AUDIT"  # Special level for audit trail


class EventType(Enum):
    """Event types for categorization"""
    SYSTEM = "SYSTEM"
    TRADE = "TRADE"
    SIGNAL = "SIGNAL"
    RISK = "RISK"
    PERFORMANCE = "PERFORMANCE"
    COMPLIANCE = "COMPLIANCE"
    MARKET = "MARKET"
    MODEL = "MODEL"


class InstitutionalLogger:
    """
    Professional-grade trading logger following institutional standards

    Features:
    - Structured JSON logging for machine parsing
    - Human-readable text logs for monitoring
    - Audit trail for regulatory compliance
    - Performance metrics tracking
    - Risk metrics monitoring
    - Model behavior tracking
    """

    def __init__(
        self,
        log_dir: Path,
        strategy_name: str,
        session_id: str,
        initial_capital: float,
        enable_json: bool = True,
        enable_text: bool = True,
        enable_audit: bool = True
    ):
        """
        Initialize institutional logger

        Args:
            log_dir: Directory for log files
            strategy_name: Strategy identifier
            session_id: Unique session identifier
            initial_capital: Initial trading capital
            enable_json: Enable JSON structured logs
            enable_text: Enable human-readable text logs
            enable_audit: Enable audit trail logs
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        self.strategy_name = strategy_name
        self.session_id = session_id
        self.initial_capital = initial_capital

        # Session metadata
        self.session_start = datetime.now()
        self.session_metadata = {
            "strategy": strategy_name,
            "session_id": session_id,
            "start_time": self.session_start.isoformat(),
            "initial_capital": initial_capital
        }

        # Performance tracking
        self.trades: List[Dict] = []
        self.signals: List[Dict] = []
        self.equity_curve: List[Dict] = []

        # Setup loggers
        timestamp = datetime.now().strftime("%Y%m%d")

        if enable_json:
            self.json_logger = self._setup_json_logger(
                self.log_dir / f"{strategy_name}_{timestamp}_structured.jsonl"
            )
        else:
            self.json_logger = None

        if enable_text:
            self.text_logger = self._setup_text_logger(
                self.log_dir / f"{strategy_name}_{timestamp}_readable.log"
            )
        else:
            self.text_logger = None

        if enable_audit:
            self.audit_logger = self._setup_audit_logger(
                self.log_dir / f"{strategy_name}_{timestamp}_audit.log"
            )
        else:
            self.audit_logger = None

        # Log session start
        self.log_session_start()

    def _setup_json_logger(self, filepath: Path) -> logging.Logger:
        """Setup structured JSON logger"""
        logger = logging.getLogger(f"json_{self.session_id}")
        logger.setLevel(logging.DEBUG)
        logger.handlers.clear()

        handler = logging.FileHandler(filepath, encoding='utf-8')
        handler.setFormatter(logging.Formatter('%(message)s'))
        logger.addHandler(handler)
        logger.propagate = False

        return logger

    def _setup_text_logger(self, filepath: Path) -> logging.Logger:
        """Setup human-readable text logger"""
        logger = logging.getLogger(f"text_{self.session_id}")
        logger.setLevel(logging.DEBUG)
        logger.handlers.clear()

        handler = logging.FileHandler(filepath, encoding='utf-8')
        formatter = logging.Formatter(
            '%(asctime)s | %(levelname)-8s | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.propagate = False

        return logger

    def _setup_audit_logger(self, filepath: Path) -> logging.Logger:
        """Setup audit trail logger"""
        logger = logging.getLogger(f"audit_{self.session_id}")
        logger.setLevel(logging.INFO)
        logger.handlers.clear()

        handler = logging.FileHandler(filepath, encoding='utf-8')
        formatter = logging.Formatter('%(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.propagate = False

        return logger

    def _create_log_entry(
        self,
        level: LogLevel,
        event_type: EventType,
        message: str,
        data: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Create standardized log entry"""
        entry = {
            "timestamp": datetime.now().isoformat(),
            "session_id": self.session_id,
            "strategy": self.strategy_name,
            "level": level.value,
            "event_type": event_type.value,
            "message": message
        }

        if data:
            entry["data"] = data

        return entry

    def _log(
        self,
        level: LogLevel,
        event_type: EventType,
        message: str,
        data: Optional[Dict[str, Any]] = None
    ):
        """Internal logging method"""
        entry = self._create_log_entry(level, event_type, message, data)

        # JSON structured log
        if self.json_logger:
            self.json_logger.info(json.dumps(entry, default=str))

        # Human-readable text log
        if self.text_logger:
            log_method = getattr(self.text_logger, level.value.lower(), self.text_logger.info)
            text_msg = f"[{event_type.value}] {message}"
            if data:
                text_msg += f" | {json.dumps(data, default=str)}"
            log_method(text_msg)

    def log_session_start(self):
        """Log session initialization"""
        self._log(
            LogLevel.INFO,
            EventType.SYSTEM,
            "Trading session started",
            {
                "initial_capital": self.initial_capital,
                "session_metadata": self.session_metadata
            }
        )

    def log_market_data(
        self,
        price: float,
        volume: float,
        regime: str,
        volatility: Optional[float] = None,
        spread: Optional[float] = None
    ):
        """Log market state"""
        data = {
            "price": round(price, 2),
            "volume": round(volume, 2),
            "regime": regime
        }

        if volatility is not None:
            data["volatility"] = round(volatility, 6)
        if spread is not None:
            data["spread_bps"] = round(spread * 10000, 2)

        self._log(
            LogLevel.DEBUG,
            EventType.MARKET,
            f"Market state: {regime}",
            data
        )

    def log_signal(
        self,
        signal_type: str,
        direction: str,
        probability: float,
        price: float,
        features: Optional[Dict] = None
    ):
        """Log trading signal"""
        data = {
            "signal_type": signal_type,
            "direction": direction,
            "probability": round(probability, 4),
            "price": round(price, 2),
            "timestamp": datetime.now().isoformat()
        }

        if features:
            # Handle both numeric and non-numeric feature values
            data["top_features"] = {k: round(v, 4) if isinstance(v, (int, float)) else v for k, v in list(features.items())[:5]}

        self.signals.append(data)

        self._log(
            LogLevel.INFO,
            EventType.SIGNAL,
            f"Signal: {direction} @ {probability:.3f}",
            data
        )

    def log_trade_entry(
        self,
        order_id: str,
        side: str,
        quantity: float,
        price: float,
        position_size_pct: float,
        signal_probability: float,
        regime: str,
        leverage: float = 1.0
    ):
        """Log trade entry with full audit trail"""
        trade_data = {
            "trade_id": order_id,
            "type": "ENTRY",
            "side": side,
            "quantity": round(quantity, 8),
            "entry_price": round(price, 2),
            "position_size_pct": round(position_size_pct * 100, 2),
            "signal_probability": round(signal_probability, 4),
            "market_regime": regime,
            "leverage": leverage,
            "position_value": round(quantity * price * leverage, 2),
            "timestamp": datetime.now().isoformat()
        }

        self.trades.append(trade_data)

        # Log to all channels
        self._log(
            LogLevel.INFO,
            EventType.TRADE,
            f"ENTRY: {side} {quantity:.4f} @ ${price:,.2f}",
            trade_data
        )

        # Audit trail
        if self.audit_logger:
            audit_entry = {
                **trade_data,
                "audit_type": "TRADE_ENTRY",
                "session_id": self.session_id
            }
            self.audit_logger.info(json.dumps(audit_entry, default=str))

    def log_trade_exit(
        self,
        order_id: str,
        side: str,
        quantity: float,
        entry_price: float,
        exit_price: float,
        pnl_usd: float,
        pnl_pct: float,
        holding_time_hours: float,
        exit_reason: str,
        transaction_costs: float = 0.0
    ):
        """Log trade exit with P&L calculation"""
        net_pnl = pnl_usd - transaction_costs

        trade_data = {
            "trade_id": order_id,
            "type": "EXIT",
            "side": side,
            "quantity": round(quantity, 8),
            "entry_price": round(entry_price, 2),
            "exit_price": round(exit_price, 2),
            "pnl_usd_gross": round(pnl_usd, 2),
            "pnl_usd_net": round(net_pnl, 2),
            "pnl_pct": round(pnl_pct * 100, 4),
            "holding_hours": round(holding_time_hours, 2),
            "exit_reason": exit_reason,
            "transaction_costs": round(transaction_costs, 2),
            "timestamp": datetime.now().isoformat()
        }

        # Log to all channels
        level = LogLevel.INFO if net_pnl >= 0 else LogLevel.WARNING
        self._log(
            level,
            EventType.TRADE,
            f"EXIT: {side} ${net_pnl:+,.2f} ({pnl_pct*100:+.2f}%) - {exit_reason}",
            trade_data
        )

        # Audit trail
        if self.audit_logger:
            audit_entry = {
                **trade_data,
                "audit_type": "TRADE_EXIT",
                "session_id": self.session_id
            }
            self.audit_logger.info(json.dumps(audit_entry, default=str))

    def log_performance_metrics(
        self,
        current_capital: float,
        total_trades: int,
        win_rate: float,
        sharpe_ratio: Optional[float] = None,
        sortino_ratio: Optional[float] = None,
        max_drawdown: Optional[float] = None,
        profit_factor: Optional[float] = None
    ):
        """Log performance metrics"""
        returns_pct = ((current_capital - self.initial_capital) / self.initial_capital) * 100

        metrics = {
            "timestamp": datetime.now().isoformat(),
            "current_capital": round(current_capital, 2),
            "initial_capital": round(self.initial_capital, 2),
            "total_return_pct": round(returns_pct, 4),
            "total_trades": total_trades,
            "win_rate_pct": round(win_rate, 2)
        }

        if sharpe_ratio is not None:
            metrics["sharpe_ratio"] = round(sharpe_ratio, 4)
        if sortino_ratio is not None:
            metrics["sortino_ratio"] = round(sortino_ratio, 4)
        if max_drawdown is not None:
            metrics["max_drawdown_pct"] = round(max_drawdown * 100, 4)
        if profit_factor is not None:
            metrics["profit_factor"] = round(profit_factor, 4)

        self.equity_curve.append(metrics)

        self._log(
            LogLevel.INFO,
            EventType.PERFORMANCE,
            f"Performance: {returns_pct:+.2f}% | Win Rate: {win_rate:.1f}%",
            metrics
        )

    def log_risk_metrics(
        self,
        current_exposure: float,
        var_95: Optional[float] = None,
        var_99: Optional[float] = None,
        leverage: float = 1.0,
        max_position_pct: float = 1.0
    ):
        """Log risk metrics"""
        metrics = {
            "timestamp": datetime.now().isoformat(),
            "current_exposure_usd": round(current_exposure, 2),
            "exposure_pct": round((current_exposure / self.initial_capital) * 100, 2),
            "leverage": leverage,
            "max_position_pct": round(max_position_pct * 100, 2)
        }

        if var_95 is not None:
            metrics["var_95_pct"] = round(var_95 * 100, 4)
        if var_99 is not None:
            metrics["var_99_pct"] = round(var_99 * 100, 4)

        self._log(
            LogLevel.INFO,
            EventType.RISK,
            f"Risk: Exposure {metrics['exposure_pct']:.1f}% | Leverage {leverage}x",
            metrics
        )

    def log_model_prediction(
        self,
        model_name: str,
        prediction: str,
        confidence: float,
        feature_importance: Optional[Dict] = None
    ):
        """Log model prediction"""
        data = {
            "model": model_name,
            "prediction": prediction,
            "confidence": round(confidence, 4),
            "timestamp": datetime.now().isoformat()
        }

        if feature_importance:
            # Handle both numeric and non-numeric feature values
            data["top_features"] = {
                k: round(v, 4) if isinstance(v, (int, float)) else v for k, v in list(feature_importance.items())[:10]
            }

        self._log(
            LogLevel.DEBUG,
            EventType.MODEL,
            f"Model: {model_name} → {prediction} ({confidence:.3f})",
            data
        )

    def log_error(self, error_type: str, error_message: str, stack_trace: Optional[str] = None):
        """Log error with details"""
        data = {
            "error_type": error_type,
            "error_message": error_message,
            "timestamp": datetime.now().isoformat()
        }

        if stack_trace:
            data["stack_trace"] = stack_trace

        self._log(
            LogLevel.ERROR,
            EventType.SYSTEM,
            f"Error: {error_type}",
            data
        )

    def log_critical(self, message: str, data: Optional[Dict] = None):
        """Log critical event requiring immediate attention"""
        self._log(
            LogLevel.CRITICAL,
            EventType.COMPLIANCE,
            message,
            data
        )

        # Also log to audit for compliance
        if self.audit_logger:
            audit_entry = {
                "audit_type": "CRITICAL_EVENT",
                "message": message,
                "data": data,
                "timestamp": datetime.now().isoformat(),
                "session_id": self.session_id
            }
            self.audit_logger.info(json.dumps(audit_entry, default=str))

    def log_compliance_event(self, event_type: str, details: Dict):
        """Log compliance/regulatory event"""
        data = {
            "compliance_event": event_type,
            "details": details,
            "timestamp": datetime.now().isoformat()
        }

        self._log(
            LogLevel.AUDIT,
            EventType.COMPLIANCE,
            f"Compliance: {event_type}",
            data
        )

        # Always log to audit trail
        if self.audit_logger:
            audit_entry = {
                **data,
                "session_id": self.session_id
            }
            self.audit_logger.info(json.dumps(audit_entry, default=str))

    def generate_session_summary(self) -> Dict[str, Any]:
        """Generate comprehensive session summary"""
        session_end = datetime.now()
        duration_hours = (session_end - self.session_start).total_seconds() / 3600

        # Calculate metrics from trades
        closed_trades = [t for t in self.trades if t.get("type") == "EXIT"]

        if len(closed_trades) > 0:
            df_trades = pd.DataFrame(closed_trades)
            winning_trades = len(df_trades[df_trades['pnl_usd_net'] > 0])
            win_rate = (winning_trades / len(df_trades)) * 100
            total_pnl = df_trades['pnl_usd_net'].sum()
            avg_win = df_trades[df_trades['pnl_usd_net'] > 0]['pnl_usd_net'].mean() if winning_trades > 0 else 0
            avg_loss = df_trades[df_trades['pnl_usd_net'] < 0]['pnl_usd_net'].mean() if winning_trades < len(df_trades) else 0
        else:
            win_rate = 0
            total_pnl = 0
            avg_win = 0
            avg_loss = 0

        summary = {
            "session_id": self.session_id,
            "strategy": self.strategy_name,
            "start_time": self.session_start.isoformat(),
            "end_time": session_end.isoformat(),
            "duration_hours": round(duration_hours, 2),
            "initial_capital": self.initial_capital,
            "total_trades": len(closed_trades),
            "win_rate_pct": round(win_rate, 2),
            "total_pnl_usd": round(total_pnl, 2),
            "avg_win_usd": round(avg_win, 2),
            "avg_loss_usd": round(avg_loss, 2),
            "total_signals": len(self.signals)
        }

        # Log summary
        self._log(
            LogLevel.INFO,
            EventType.SYSTEM,
            "Session summary generated",
            summary
        )

        return summary

    def close(self):
        """Close logger and generate final summary"""
        summary = self.generate_session_summary()

        self._log(
            LogLevel.INFO,
            EventType.SYSTEM,
            "Trading session ended",
            summary
        )

        # Close handlers
        for logger in [self.json_logger, self.text_logger, self.audit_logger]:
            if logger:
                for handler in logger.handlers:
                    handler.close()
                logger.handlers.clear()

        return summary


# Convenience function for quick setup
def create_institutional_logger(
    log_dir: Path,
    strategy_name: str,
    initial_capital: float
) -> InstitutionalLogger:
    """
    Create institutional logger with standard configuration

    Args:
        log_dir: Log directory
        strategy_name: Strategy name
        initial_capital: Initial capital

    Returns:
        Configured InstitutionalLogger instance
    """
    session_id = f"{strategy_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    return InstitutionalLogger(
        log_dir=log_dir,
        strategy_name=strategy_name,
        session_id=session_id,
        initial_capital=initial_capital,
        enable_json=True,
        enable_text=True,
        enable_audit=True
    )
