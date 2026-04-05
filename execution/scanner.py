"""
execution/scanner.py
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

import pandas as pd

import config.settings as settings
from logger.trade_logger import get_logger
from data.data_fetcher import DataFetcher
from indicators.indicator_engine import compute_indicators

log = get_logger("scanner")


@dataclass
class Signal:
    symbol: str
    side: str
    entry: float
    stop_loss: float
    take_profit: float
    atr: float
    reason: str


class MarketScanner:
    def __init__(self) -> None:
        self.fetcher = DataFetcher()
        self.scan_no = 0
        self.client = self._build_client()

    def _build_client(self):
        mode = str(getattr(settings, "BOT_MODE", "DEMO")).upper()
        if mode == "DEMO":
            from broker.simplefx_broker import get_client
            return get_client()
        return None

    def _get_symbols(self) -> list[str]:
        raw = getattr(settings, "SYMBOLS", [])
        out: list[str] = []

        for item in raw:
            if isinstance(item, str):
                out.append(item)
            elif isinstance(item, dict):
                sym = item.get("simplefx") or item.get("symbol") or item.get("name")
                if sym:
                    out.append(str(sym))

        return out

    def _lot_size(self, symbol: str) -> float:
        for key in ("LOT_SIZE", "FIXED_LOT_SIZE", "DEFAULT_LOT_SIZE"):
            value = getattr(settings, key, None)
            if value is not None:
                try:
                    return float(value)
                except Exception:
                    pass
        return 0.01

    def _scan_interval(self) -> int:
        for key in ("SCAN_INTERVAL_SECONDS", "SCAN_SECONDS"):
            value = getattr(settings, key, None)
            if value is not None:
                try:
                    return int(value)
                except Exception:
                    pass
        return 60

    def _max_open_per_symbol(self) -> int:
        for key in ("MAX_OPEN_TRADES_PER_SYMBOL", "MAX_TRADES_PER_SYMBOL_OPEN"):
            value = getattr(settings, key, None)
            if value is not None:
                try:
                    return int(value)
                except Exception:
                    pass
        value = getattr(settings, "MAX_TRADES_PER_SYMBOL", 1)
        try:
            return int(value)
        except Exception:
            return 1

    def _max_hold_hours(self) -> float:
        return float(getattr(settings, "MAX_HOLD_HOURS", 48))

    def _tp_atr_mult(self) -> float:
        return float(getattr(settings, "ATR_TP_MULTIPLIER", 3.0))

    def _sl_atr_mult(self) -> float:
        return float(getattr(settings, "ATR_SL_MULTIPLIER", 1.5))

    def _now(self) -> datetime:
        return datetime.now(timezone.utc)

    def _safe_float(self, value: Any, default: float = 0.0) -> float:
        try:
            return float(value)
        except Exception:
            return default

    def _get_open_positions(self) -> list[dict]:
        if self.client is None:
            return []
        try:
            positions = self.client.get_open_positions()
            return positions if isinstance(positions, list) else []
        except Exception as e:
            log.error("Open positions al?nmad?: %s", e)
            return []

    def _count_open_for_symbol(self, positions: list[dict], symbol: str) -> int:
        count = 0
        for pos in positions:
            pos_symbol = (
                pos.get("symbol")
                or pos.get("instrument")
                or pos.get("ticker")
                or pos.get("market")
            )
            if str(pos_symbol) == symbol:
                count += 1
        return count

    def _parse_open_time(self, pos: dict) -> datetime | None:
        raw = (
            pos.get("open_time")
            or pos.get("openTime")
            or pos.get("time")
            or pos.get("createdAt")
            or pos.get("opened_at")
        )
        if not raw:
            return None

        try:
            txt = str(raw).replace("Z", "+00:00")
            dt = datetime.fromisoformat(txt)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt.astimezone(timezone.utc)
        except Exception:
            return None

    def _extract_position_id(self, pos: dict) -> str | None:
        for key in ("positionId", "id", "orderId", "ticket"):
            if pos.get(key) not in (None, ""):
                return str(pos[key])
        return None

    def _extract_side(self, pos: dict) -> str:
        raw = str(pos.get("side") or pos.get("direction") or pos.get("cmd") or "").upper()
        if raw in ("BUY", "LONG", "0"):
            return "BUY"
        if raw in ("SELL", "SHORT", "1"):
            return "SELL"
        return raw

    def _extract_entry(self, pos: dict) -> float:
        for key in ("openPrice", "entryPrice", "price", "open_price", "entry"):
            if pos.get(key) not in (None, ""):
                return self._safe_float(pos.get(key))
        return 0.0

    def _extract_current(self, pos: dict, symbol: str) -> float:
        for key in ("currentPrice", "markPrice", "bid", "ask", "closePrice", "priceCurrent"):
            if pos.get(key) not in (None, ""):
                return self._safe_float(pos.get(key))
        return self._latest_price(symbol)

    def _latest_price(self, symbol: str) -> float:
        try:
            df = self.fetcher.fetch_live(symbol, "15m", bars=5)
            if df is not None and not df.empty:
                return float(df["close"].iloc[-1])
        except Exception as e:
            log.warning("[%s] son qiym?t al?nmad?: %s", symbol, e)
        return 0.0

    def _manage_open_positions(self, positions: list[dict]) -> int:
        closed = 0
        now = self._now()
        max_hold_hours = self._max_hold_hours()

        for pos in positions:
            symbol = str(pos.get("symbol") or pos.get("instrument") or pos.get("ticker") or "")
            position_id = self._extract_position_id(pos)
            if not symbol or not position_id:
                continue

            open_dt = self._parse_open_time(pos)
            side = self._extract_side(pos)
            entry = self._extract_entry(pos)
            current = self._extract_current(pos, symbol)

            if entry <= 0 or current <= 0:
                continue

            should_close = False
            reason = ""

            if open_dt is not None:
                held_hours = (now - open_dt).total_seconds() / 3600.0
                if held_hours >= max_hold_hours:
                    should_close = True
                    reason = f"time_exit_{held_hours:.1f}h"

            pnl_pct = 0.0
            if side == "BUY":
                pnl_pct = (current - entry) / entry
            elif side == "SELL":
                pnl_pct = (entry - current) / entry

            if not should_close and pnl_pct >= 0.010:
                should_close = True
                reason = f"take_profit_{pnl_pct:.4f}"

            if not should_close and pnl_pct <= -0.006:
                should_close = True
                reason = f"stop_loss_{pnl_pct:.4f}"

            if should_close:
                try:
                    ok = self.client.close_position(position_id)
                    if ok:
                        closed += 1
                        log.info("[%s] position bagland? | id=%s | reason=%s", symbol, position_id, reason)
                    else:
                        log.warning("[%s] position baglanmad? | id=%s | reason=%s", symbol, position_id, reason)
                except Exception as e:
                    log.error("[%s] close_position x?tas?: %s", symbol, e)

        return closed

    def _load_tf(self, symbol: str, tf: str, bars: int = 300) -> pd.DataFrame:
        df = self.fetcher.fetch_live(symbol, tf, bars=bars)
        if df is None or df.empty:
            return pd.DataFrame()
        return compute_indicators(df)

    def _build_signal(self, symbol: str) -> Signal | None:
        df_1h = self._load_tf(symbol, "1h", bars=300)
        df_15m = self._load_tf(symbol, "15m", bars=300)

        if df_1h.empty or df_15m.empty:
            log.info("[%s] no_signal | s?b?b=data_empty", symbol)
            return None

        row_h1 = df_1h.iloc[-1]
        row_15 = df_15m.iloc[-1]
        price = self._safe_float(row_15.get("close"))

        ema_fast_h1 = self._safe_float(row_h1.get("ema_20"))
        ema_slow_h1 = self._safe_float(row_h1.get("ema_50"))
        rsi_15 = self._safe_float(row_15.get("rsi"))
        macd_15 = self._safe_float(row_15.get("macd"))
        macd_signal_15 = self._safe_float(row_15.get("macd_signal"))
        atr = self._safe_float(row_15.get("atr"))

        if price <= 0 or atr <= 0:
            log.info("[%s] no_signal | s?b?b=price_or_atr_invalid", symbol)
            return None

        side = None
        reason = None

        if ema_fast_h1 > ema_slow_h1 and rsi_15 >= 55 and macd_15 > macd_signal_15:
            side = "BUY"
            reason = "trend_up + rsi_up + macd_up"
        elif ema_fast_h1 < ema_slow_h1 and rsi_15 <= 45 and macd_15 < macd_signal_15:
            side = "SELL"
            reason = "trend_down + rsi_down + macd_down"

        if side is None:
            log.info("[%s] no_signal | H1_EMA(%.5f/%.5f) RSI15=%.2f MACD15=%.5f SIG=%.5f",
                     symbol, ema_fast_h1, ema_slow_h1, rsi_15, macd_15, macd_signal_15)
            return None

        sl_mult = self._sl_atr_mult()
        tp_mult = self._tp_atr_mult()

        if side == "BUY":
            stop_loss = price - atr * sl_mult
            take_profit = price + atr * tp_mult
        else:
            stop_loss = price + atr * sl_mult
            take_profit = price - atr * tp_mult

        return Signal(
            symbol=symbol,
            side=side,
            entry=price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            atr=atr,
            reason=reason,
        )

    def _execute_signal(self, signal: Signal) -> bool:
        if self.client is None:
            log.info("[%s] SIGNAL_ONLY | %s", signal.symbol, signal.side)
            return False

        lot = self._lot_size(signal.symbol)

        try:
            result = self.client.place_market_order(
                symbol=signal.symbol,
                direction=signal.side,
                lot_size=lot,
                stop_loss=signal.stop_loss,
                take_profit=signal.take_profit,
                comment="ml-scanner",
            )
            if result:
                log.info(
                    "[%s] EXECUTED %s | entry=%.5f sl=%.5f tp=%.5f atr=%.5f | %s",
                    signal.symbol, signal.side, signal.entry, signal.stop_loss,
                    signal.take_profit, signal.atr, signal.reason
                )
                return True

            log.warning("[%s] order r?dd edildi", signal.symbol)
            return False
        except Exception as e:
            log.error("[%s] execute x?tas?: %s", signal.symbol, e)
            return False

    def scan_once(self) -> dict:
        self.scan_no += 1
        symbols = self._get_symbols()
        now = self._now()

        log.info("-" * 60)
        log.info("Skan #%s baslad? � %s", self.scan_no, now.strftime("%Y-%m-%d %H:%M:%S UTC"))
        log.info("Rejim: %s | Simvol: %s", getattr(settings, "BOT_MODE", "DEMO"), len(symbols))

        positions = self._get_open_positions()
        closed = self._manage_open_positions(positions)
        positions = self._get_open_positions()

        summary = {
            "scanned": 0,
            "signals": 0,
            "executed": 0,
            "closed": closed,
            "skipped": 0,
            "errors": 0,
            "details": [],
        }

        max_open = self._max_open_per_symbol()

        for symbol in symbols:
            summary["scanned"] += 1

            try:
                open_count = self._count_open_for_symbol(positions, symbol)
                if open_count >= max_open:
                    msg = f"{symbol} ucun art?q ac?q trade var"
                    log.info("[%s] blocked | %s", symbol, msg)
                    summary["skipped"] += 1
                    summary["details"].append({"symbol": symbol, "status": "blocked_open_position", "reason": msg})
                    continue

                signal = self._build_signal(symbol)
                if signal is None:
                    summary["details"].append({"symbol": symbol, "status": "no_signal"})
                    continue

                summary["signals"] += 1
                ok = self._execute_signal(signal)

                if ok:
                    summary["executed"] += 1
                    summary["details"].append({
                        "symbol": symbol,
                        "status": "executed",
                        "side": signal.side,
                        "entry": round(signal.entry, 6),
                    })
                    positions = self._get_open_positions()
                else:
                    summary["skipped"] += 1
                    summary["details"].append({"symbol": symbol, "status": "signal_but_not_executed"})

            except Exception as e:
                log.exception("[%s] scanner error: %s", symbol, e)
                summary["errors"] += 1
                summary["details"].append({"symbol": symbol, "status": "error", "error": str(e)})

        log.info("Skan bitdi: %s", summary)
        log.info("-" * 60)
        return summary

    def run_loop(self) -> None:
        wait_seconds = self._scan_interval()
        log.info("Continuous mode baslad? | h?r %s saniy?d? bir scan", wait_seconds)

        while True:
            try:
                self.scan_once()
            except KeyboardInterrupt:
                log.info("Scanner dayand?r?ld?.")
                break
            except Exception as e:
                log.exception("Scanner loop x?tas?: %s", e)

            time.sleep(wait_seconds)

