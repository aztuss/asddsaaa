"""
core/scanner.py

Main market scanning engine.
Scans all 30 symbols every SCAN_INTERVAL_SECONDS.

For each symbol:
  1. Fetch live 1H, 15M, 1M data
  2. Compute indicators
  3. Build market context
  4. Check hard filters (open trade, news, ATR)
  5. Build ML feature row
  6. Run ML inference в†’ probabilities
  7. Evaluate entry signal
  8. Risk validate
  9. Execute if approved
  10. Log everything
"""

import logging
import time
from datetime import datetime, timezone

from config.market_config import MARKET_CONFIG, get_config
from config.settings import Settings
from data.data_fetcher import DataFetcher
from indicators.indicator_engine import compute_indicators
from strategy.context_builder import build_context
from strategy.entry_logic import evaluate_entry, TradeSignal
from ml.feature_builder import FeatureBuilder
from ml.model_registry import get_inference_engine
from ml.model_inference import InferenceResult
from news.news_filter import NewsFilter
from risk.risk_manager import RiskManager
from execution.trade_executor import TradeExecutor
from execution.position_manager import PositionManager
from execution.market_hours import is_market_open
from logger.trade_logger import TradeLogger
from broker.base_broker import BaseBroker
from utils.helpers import elapsed_since, is_market_hours, log_separator

logger = logging.getLogger(__name__)


class MarketScanner:
    """
    Scans all configured markets on a fixed interval.
    Combines multi-timeframe data, ML inference, and broker execution.
    """

    def __init__(self, broker: BaseBroker):
        self.broker           = broker
        self.fetcher          = DataFetcher()
        self.news_filter      = NewsFilter()
        self.risk_manager     = RiskManager()
        self.position_manager = PositionManager(broker)
        self.trade_logger     = TradeLogger()
        self.executor         = TradeExecutor(broker, self.trade_logger)
        self.ml_engine        = get_inference_engine()
        self._running         = False

    # в”Ђв”Ђ Public control в”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђ

    def start(self) -> None:
        """Start the continuous scanning loop (blocking)."""
        self._running = True
        self.trade_logger.log_system("Bot started. Entering scan loop.")
        log_separator(logger)

        while self._running:
            try:
                self._scan_cycle()
            except KeyboardInterrupt:
                logger.info("Keyboard interrupt received. Stopping bot.")
                self._running = False
                break
            except Exception as e:
                logger.error(f"Unhandled error in scan cycle: {e}", exc_info=True)
                time.sleep(10)

            time.sleep(Settings.SCAN_INTERVAL_SECONDS)

    def stop(self) -> None:
        self._running = False
        self.trade_logger.log_system("Bot stopped.")

    # в”Ђв”Ђ Scan cycle в”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђ

    def _scan_cycle(self) -> None:
        """Run one full scan across all 30 symbols."""
        cycle_start = time.time()
        now = datetime.now(timezone.utc)

        symbols = list(MARKET_CONFIG.keys())
        self.trade_logger.log_scan_start(len(symbols))

        # Refresh news and positions once per cycle
        self.news_filter.refresh()
        self.position_manager.refresh_positions()

        n_signals  = 0
        n_executed = 0

        for symbol in symbols:
            try:
                result = self._scan_symbol(symbol, now)
                if result is not None:
                    n_signals += 1
                    if result.success:
                        n_executed += 1
            except Exception as e:
                logger.error(f"[{symbol}] Error during scan: {e}", exc_info=True)

        elapsed = elapsed_since(cycle_start)
        self.trade_logger.log_scan_end(len(symbols), elapsed, n_signals, n_executed)
        self.position_manager.print_status()

    def _scan_symbol(self, symbol: str, now: datetime):
        """Full pipeline for one symbol. Returns OrderResult or None."""
        cfg = get_config(symbol)

        # Skip if market is closed (stocks/indices on weekends)
        if not is_market_hours(cfg.asset_class):
            logger.debug(f"[{symbol}] Market closed. Skipping.")
            return None

        # в”Ђв”Ђ Step 1: Hard filter вЂ” already has open trade в”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђ
        if self.position_manager.has_open_trade(symbol):
            logger.debug(f"[{symbol}] Already has open trade. Skipping.")
            return None

        # в”Ђв”Ђ Step 2: Hard filter вЂ” news block в”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђ
        if self.news_filter.is_blocked(symbol, now):
            self.trade_logger.log_news_block(symbol)
            return None

        # в”Ђв”Ђ Step 3: Fetch live data в”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђ
        df_1h  = self.fetcher.fetch_live(symbol, "1h",  bars=Settings.CANDLES_1H)
        df_15m = self.fetcher.fetch_live(symbol, "15m", bars=Settings.CANDLES_15M)
        df_1m  = self.fetcher.fetch_live(symbol, "1m",  bars=Settings.CANDLES_1M)

        if df_1h.empty or df_15m.empty:
            logger.warning(f"[{symbol}] Insufficient live data.")
            return None

        # в”Ђв”Ђ Step 4: Compute indicators в”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђ
        df_1h  = compute_indicators(df_1h,  cfg)
        df_15m = compute_indicators(df_15m, cfg)

        # в”Ђв”Ђ Step 5: Build market context в”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђ
        ctx = build_context(
            symbol=symbol,
            df_1h=df_1h,
            df_15m=df_15m,
            cfg=cfg,
            atr_sl_mult=Settings.ATR_SL_MULTIPLIER,
            rr_ratio=Settings.DEFAULT_RR_RATIO,
        )
        if ctx is None:
            logger.warning(f"[{symbol}] Context build failed.")
            return None

        # в”Ђв”Ђ Step 6: ATR minimum filter в”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђ
        if not ctx.m15_atr_ok:
            logger.debug(f"[{symbol}] ATR too low. Skipping.")
            return None

        # в”Ђв”Ђ Step 7: Build ML feature row в”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђ
        fb = FeatureBuilder(cfg)
        feature_row = fb.build_live_row(
            df_1h=df_1h,
            df_15m=df_15m,
            df_1m=df_1m if not df_1m.empty else None,
            now=now,
        )

        # в”Ђв”Ђ Step 8: ML inference в”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђ
        if self.ml_engine.is_loaded() and feature_row:
            ml_result = self.ml_engine.predict(feature_row, cfg)
        else:
            # Fallback: use rule-based context signals only
            ml_result = self._rule_based_fallback(ctx, cfg)

        # Log ML output for every symbol
        self.trade_logger.log_ml_output(
            symbol=symbol,
            long_prob=ml_result.long_probability,
            short_prob=ml_result.short_probability,
            direction=ml_result.direction,
            threshold_long=cfg.ml_threshold_long,
            threshold_short=cfg.ml_threshold_short,
        )

        logger.info(
            f"[{symbol}] ML в†’ long={ml_result.long_probability:.3f} "
            f"short={ml_result.short_probability:.3f} "
            f"dir={ml_result.direction}"
        )

        # в”Ђв”Ђ Step 9: Entry signal evaluation в”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђ
        signal = evaluate_entry(
            symbol=symbol,
            cfg=cfg,
            ctx=ctx,
            ml_result=ml_result,
            has_open_trade=self.position_manager.has_open_trade(symbol),
            news_blocked=False,   # already checked above
            total_open_trades=self.position_manager.total_open(),
            max_open_trades=Settings.MAX_OPEN_TRADES,
            now=now,
        )

        if not signal.allowed:
            self.trade_logger.log_rejected(
                symbol=symbol,
                direction=ml_result.direction,
                reason=signal.blocked_reason,
                long_prob=ml_result.long_probability,
                short_prob=ml_result.short_probability,
            )
            return None

        # в”Ђв”Ђ Step 10: Risk validation в”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђ
        risk_check = self.risk_manager.validate(
            signal=signal,
            cfg=cfg,
            current_open_trades=self.position_manager.total_open(),
        )
        if not risk_check.approved:
            self.trade_logger.log_rejected(
                symbol=symbol,
                direction=signal.direction,
                reason=f"Risk check: {risk_check.reason}",
                long_prob=ml_result.long_probability,
                short_prob=ml_result.short_probability,
            )
            return None

        # в”Ђв”Ђ Step 11: Execute в”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђ
        order_result = self.executor.execute(signal)

        # Refresh positions cache immediately after execution
        if order_result and order_result.success:
            self.position_manager.refresh_positions()

        return order_result

    # в”Ђв”Ђ Rule-based fallback (when ML models not trained yet) в”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђ

    def _rule_based_fallback(self, ctx, cfg) -> InferenceResult:
        """
        Simple rule-based signal when ML model is not available.
        Uses market context indicators only.
        This is NOT the main decision engine вЂ” ML is preferred.
        """
        from ml.model_inference import InferenceResult

        long_score  = 0.0
        short_score = 0.0

        # 1H alignment
        if ctx.h1_trend_bullish:
            long_score  += 0.25
        elif ctx.h1_trend_bearish:
            short_score += 0.25

        # 15M alignment
        if ctx.m15_trend_bullish:
            long_score  += 0.20
        elif ctx.m15_trend_bearish:
            short_score += 0.20

        # MACD
        if ctx.m15_macd_bullish:
            long_score  += 0.15
        elif ctx.m15_macd_bearish:
            short_score += 0.15

        # ADX trending
        if ctx.h1_adx_trending:
            long_score  += 0.10
            short_score += 0.10

        # RSI zones
        if ctx.m15_rsi_long_zone and ctx.m15_rsi < 50:
            long_score  += 0.10
        if ctx.m15_rsi_short_zone and ctx.m15_rsi > 50:
            short_score += 0.10

        # Clamp to [0, 1]
        long_score  = min(long_score, 0.95)
        short_score = min(short_score, 0.95)

        threshold_long  = cfg.ml_threshold_long
        threshold_short = cfg.ml_threshold_short

        long_allowed  = long_score  >= threshold_long  and long_score  > short_score
        short_allowed = short_score >= threshold_short and short_score > long_score

        if long_allowed:
            direction = "LONG"
        elif short_allowed:
            direction = "SHORT"
        else:
            direction = "NONE"

        return InferenceResult(
            long_probability=long_score,
            short_probability=short_score,
            long_allowed=long_allowed,
            short_allowed=short_allowed,
            final_confidence=max(long_score, short_score),
            direction=direction,
            reason="Rule-based fallback (ML not trained)",
        )



