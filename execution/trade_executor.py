"""
execution/trade_executor.py

Executes TradeSignal objects via the broker layer.
Handles retry logic, logging, and post-execution validation.
"""

import logging
import time
from typing import Optional

from broker.base_broker import BaseBroker, OrderResult
from strategy.entry_logic import TradeSignal
from config.settings import Settings
from logger.trade_logger import TradeLogger

logger = logging.getLogger(__name__)

_MAX_RETRIES = 2
_RETRY_DELAY = 2.0   # seconds


class TradeExecutor:
    """Submits validated TradeSignals to the broker with retry logic."""

    def __init__(self, broker: BaseBroker, trade_logger: Optional[TradeLogger] = None):
        self.broker       = broker
        self.trade_logger = trade_logger

    def execute(self, signal: TradeSignal) -> Optional[OrderResult]:
        """
        Execute a TradeSignal. Returns OrderResult or None if skipped.

        Args:
            signal: validated TradeSignal with direction, SL, TP, etc.

        Returns:
            OrderResult from broker, or None if signal is not allowed.
        """
        if not signal.allowed:
            logger.debug(
                f"[Executor] {signal.symbol}: Signal NOT allowed. "
                f"Reason: {signal.blocked_reason}"
            )
            return None

        if signal.direction not in ("LONG", "SHORT"):
            logger.warning(f"[Executor] {signal.symbol}: Invalid direction '{signal.direction}'")
            return None

        logger.info(
            f"[Executor] {signal.symbol}: Executing {signal.direction} "
            f"@ {signal.entry_price:.5f} | SL={signal.sl:.5f} | TP={signal.tp:.5f} "
            f"| Lot={signal.lot_size:.4f} | Confidence={signal.confidence:.3f}"
        )

        result = self._place_with_retry(signal)

        if self.trade_logger and result:
            self.trade_logger.log_trade(signal, result)

        return result

    def _place_with_retry(self, signal: TradeSignal) -> Optional[OrderResult]:
        """Attempt order placement with retry on transient failures."""
        for attempt in range(1, _MAX_RETRIES + 2):
            try:
                result = self.broker.place_order(
                    symbol=signal.symbol,
                    direction=signal.direction,
                    lot_size=signal.lot_size,
                    entry_price=signal.entry_price,
                    sl=signal.sl,
                    tp=signal.tp,
                )
                if result.success:
                    logger.info(
                        f"[Executor] {signal.symbol}: Order {result.order_id} placed ✓"
                    )
                    return result
                else:
                    logger.warning(
                        f"[Executor] {signal.symbol}: Attempt {attempt} failed: {result.message}"
                    )
            except Exception as e:
                logger.error(
                    f"[Executor] {signal.symbol}: Exception on attempt {attempt}: {e}"
                )

            if attempt <= _MAX_RETRIES:
                time.sleep(_RETRY_DELAY)

        logger.error(f"[Executor] {signal.symbol}: All {_MAX_RETRIES+1} attempts failed.")
        return None
