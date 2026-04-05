"""
execution/position_manager.py

Manages position state across all 30 symbols.
Provides reliable one-trade-per-symbol enforcement.
"""

import logging
from typing import Optional

from broker.base_broker import BaseBroker, Position

logger = logging.getLogger(__name__)


class PositionManager:
    """
    Wraps the broker's position tracking to provide fast in-memory lookups.

    One-trade-per-symbol enforcement:
      - Before each scan cycle, refresh_positions() is called
      - has_open_trade(symbol) checks in-memory cache
      - DemoBroker/SimpleFXBroker both enforce this at the broker level too
      - Double enforcement: here (fast) + broker (authoritative)
    """

    def __init__(self, broker: BaseBroker):
        self.broker    = broker
        self._positions: dict[str, Position] = {}   # symbol → Position

    def refresh_positions(self) -> None:
        """Sync position cache from broker."""
        all_positions = self.broker.get_open_positions()
        self._positions = {p.symbol: p for p in all_positions}
        logger.debug(f"[PositionManager] Refreshed. Open: {list(self._positions.keys())}")

    def has_open_trade(self, symbol: str) -> bool:
        """Returns True if symbol has an open position (uses cache)."""
        return symbol in self._positions

    def total_open(self) -> int:
        """Total number of open positions."""
        return len(self._positions)

    def get_position(self, symbol: str) -> Optional[Position]:
        """Return position for symbol, or None."""
        return self._positions.get(symbol)

    def all_open_symbols(self) -> list[str]:
        """List of all symbols with open positions."""
        return list(self._positions.keys())

    def print_status(self) -> None:
        """Log current position status."""
        if not self._positions:
            logger.info("[PositionManager] No open positions.")
            return
        for sym, pos in self._positions.items():
            logger.info(
                f"  [{sym}] {pos.direction} @ {pos.entry_price:.5f} "
                f"SL={pos.sl:.5f} TP={pos.tp:.5f} [{pos.order_id}]"
            )
