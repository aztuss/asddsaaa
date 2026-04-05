"""
config/market_config.py

Per-symbol configuration: indicator params, ML thresholds, group assignment.
Each symbol is fully self-contained.
"""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class SymbolConfig:
    symbol: str
    market_group: str                   # SAFE | BALANCED | FAST
    asset_class: str                    # forex | index | crypto | stock

    # EMA
    ema_fast: int = 50
    ema_slow: int = 200

    # RSI
    rsi_length: int = 14

    # MACD
    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal: int = 9

    # ADX / ATR
    adx_length: int = 14
    atr_length: int = 14
    adx_threshold: float = 20.0
    atr_min_filter: float = 0.0         # skip if ATR below this (0 = disabled)

    # Filters
    use_volume_filter: bool = False
    spread_max_filter: float = 0.0      # 0 = disabled

    # Session
    session_filter: Optional[str] = None  # "london" | "ny" | "asia" | None

    # ML decision thresholds
    ml_threshold_long: float = 0.72
    ml_threshold_short: float = 0.72

    # yfinance / ccxt ticker mapping
    yf_ticker: str = ""                 # e.g. "EURUSD=X"
    ccxt_ticker: str = ""               # e.g. "BTC/USDT"

    # SimpleFX symbol (may differ from display name)
    broker_symbol: str = ""


# ─── SAFE group ────────────────────────────────────────────────────────────────

_SAFE_PARAMS = dict(
    market_group="SAFE",
    ema_fast=50, ema_slow=200,
    rsi_length=14,
    macd_fast=12, macd_slow=26, macd_signal=9,
    adx_length=14, atr_length=14,
    adx_threshold=20.0,
    ml_threshold_long=0.72, ml_threshold_short=0.72,
)

_BALANCED_PARAMS = dict(
    market_group="BALANCED",
    ema_fast=34, ema_slow=200,
    rsi_length=12,
    macd_fast=8, macd_slow=21, macd_signal=5,
    adx_length=14, atr_length=12,
    adx_threshold=20.0,
    ml_threshold_long=0.72, ml_threshold_short=0.72,
)

_FAST_PARAMS = dict(
    market_group="FAST",
    ema_fast=21, ema_slow=200,
    rsi_length=10,
    macd_fast=6, macd_slow=19, macd_signal=4,
    adx_length=14, atr_length=10,
    adx_threshold=18.0,
    use_volume_filter=True,
    ml_threshold_long=0.75, ml_threshold_short=0.75,
)

# ─── All 30 symbols ────────────────────────────────────────────────────────────

MARKET_CONFIG: dict[str, SymbolConfig] = {

    # ── FOREX SAFE ────────────────────────────────────────────────
    "EURUSD": SymbolConfig(
        symbol="EURUSD", asset_class="forex",
        yf_ticker="EURUSD=X", broker_symbol="EURUSD",
        **_SAFE_PARAMS,
    ),
    "USDCHF": SymbolConfig(
        symbol="USDCHF", asset_class="forex",
        yf_ticker="USDCHF=X", broker_symbol="USDCHF",
        **_SAFE_PARAMS,
    ),
    "EURGBP": SymbolConfig(
        symbol="EURGBP", asset_class="forex",
        yf_ticker="EURGBP=X", broker_symbol="EURGBP",
        **_SAFE_PARAMS,
    ),

    # ── INDEX SAFE ────────────────────────────────────────────────
    "US500": SymbolConfig(
        symbol="US500", asset_class="index",
        yf_ticker="^GSPC", broker_symbol="US500",
        **_SAFE_PARAMS,
    ),
    "UK100": SymbolConfig(
        symbol="UK100", asset_class="index",
        yf_ticker="^FTSE", broker_symbol="UK100",
        **_SAFE_PARAMS,
    ),

    # ── STOCK SAFE ────────────────────────────────────────────────
    "AAPL": SymbolConfig(
        symbol="AAPL", asset_class="stock",
        yf_ticker="AAPL", broker_symbol="AAPL",
        **_SAFE_PARAMS,
    ),
    "MSFT": SymbolConfig(
        symbol="MSFT", asset_class="stock",
        yf_ticker="MSFT", broker_symbol="MSFT",
        **_SAFE_PARAMS,
    ),
    "GOOGL": SymbolConfig(
        symbol="GOOGL", asset_class="stock",
        yf_ticker="GOOGL", broker_symbol="GOOGL",
        **_SAFE_PARAMS,
    ),

    # ── FOREX BALANCED ────────────────────────────────────────────
    "GBPUSD": SymbolConfig(
        symbol="GBPUSD", asset_class="forex",
        yf_ticker="GBPUSD=X", broker_symbol="GBPUSD",
        **_BALANCED_PARAMS,
    ),
    "USDJPY": SymbolConfig(
        symbol="USDJPY", asset_class="forex",
        yf_ticker="USDJPY=X", broker_symbol="USDJPY",
        **_BALANCED_PARAMS,
    ),
    "AUDUSD": SymbolConfig(
        symbol="AUDUSD", asset_class="forex",
        yf_ticker="AUDUSD=X", broker_symbol="AUDUSD",
        **_BALANCED_PARAMS,
    ),
    "USDCAD": SymbolConfig(
        symbol="USDCAD", asset_class="forex",
        yf_ticker="USDCAD=X", broker_symbol="USDCAD",
        **_BALANCED_PARAMS,
    ),
    "NZDUSD": SymbolConfig(
        symbol="NZDUSD", asset_class="forex",
        yf_ticker="NZDUSD=X", broker_symbol="NZDUSD",
        **_BALANCED_PARAMS,
    ),
    "EURJPY": SymbolConfig(
        symbol="EURJPY", asset_class="forex",
        yf_ticker="EURJPY=X", broker_symbol="EURJPY",
        **_BALANCED_PARAMS,
    ),

    # ── INDEX BALANCED ────────────────────────────────────────────
    "DE40": SymbolConfig(
        symbol="DE40", asset_class="index",
        yf_ticker="^GDAXI", broker_symbol="DE40",
        **_BALANCED_PARAMS,
    ),
    "JP225": SymbolConfig(
        symbol="JP225", asset_class="index",
        yf_ticker="^N225", broker_symbol="JP225",
        **_BALANCED_PARAMS,
    ),

    # ── CRYPTO BALANCED ───────────────────────────────────────────
    "ETHUSD": SymbolConfig(
        symbol="ETHUSD", asset_class="crypto",
        yf_ticker="ETH-USD", ccxt_ticker="ETH/USDT", broker_symbol="ETHUSD",
        **_BALANCED_PARAMS,
    ),
    "BNBUSD": SymbolConfig(
        symbol="BNBUSD", asset_class="crypto",
        yf_ticker="BNB-USD", ccxt_ticker="BNB/USDT", broker_symbol="BNBUSD",
        **_BALANCED_PARAMS,
    ),
    "ADAUSD": SymbolConfig(
        symbol="ADAUSD", asset_class="crypto",
        yf_ticker="ADA-USD", ccxt_ticker="ADA/USDT", broker_symbol="ADAUSD",
        **_BALANCED_PARAMS,
    ),

    # ── STOCK BALANCED ────────────────────────────────────────────
    "AMZN": SymbolConfig(
        symbol="AMZN", asset_class="stock",
        yf_ticker="AMZN", broker_symbol="AMZN",
        **_BALANCED_PARAMS,
    ),
    "META": SymbolConfig(
        symbol="META", asset_class="stock",
        yf_ticker="META", broker_symbol="META",
        **_BALANCED_PARAMS,
    ),

    # ── FOREX FAST ────────────────────────────────────────────────
    "GBPJPY": SymbolConfig(
        symbol="GBPJPY", asset_class="forex",
        yf_ticker="GBPJPY=X", broker_symbol="GBPJPY",
        **_FAST_PARAMS,
    ),

    # ── INDEX FAST ────────────────────────────────────────────────
    "US100": SymbolConfig(
        symbol="US100", asset_class="index",
        yf_ticker="^NDX", broker_symbol="US100",
        **_FAST_PARAMS,
    ),
    "US30": SymbolConfig(
        symbol="US30", asset_class="index",
        yf_ticker="^DJI", broker_symbol="US30",
        **_FAST_PARAMS,
    ),

    # ── CRYPTO FAST ───────────────────────────────────────────────
    "BTCUSD": SymbolConfig(
        symbol="BTCUSD", asset_class="crypto",
        yf_ticker="BTC-USD", ccxt_ticker="BTC/USDT", broker_symbol="BTCUSD",
        **_FAST_PARAMS,
    ),
    "XRPUSD": SymbolConfig(
        symbol="XRPUSD", asset_class="crypto",
        yf_ticker="XRP-USD", ccxt_ticker="XRP/USDT", broker_symbol="XRPUSD",
        **_FAST_PARAMS,
    ),
    "SOLUSD": SymbolConfig(
        symbol="SOLUSD", asset_class="crypto",
        yf_ticker="SOL-USD", ccxt_ticker="SOL/USDT", broker_symbol="SOLUSD",
        **_FAST_PARAMS,
    ),

    # ── STOCK FAST ────────────────────────────────────────────────
    "NVDA": SymbolConfig(
        symbol="NVDA", asset_class="stock",
        yf_ticker="NVDA", broker_symbol="NVDA",
        **_FAST_PARAMS,
    ),
    "TSLA": SymbolConfig(
        symbol="TSLA", asset_class="stock",
        yf_ticker="TSLA", broker_symbol="TSLA",
        **_FAST_PARAMS,
    ),
    "NFLX": SymbolConfig(
        symbol="NFLX", asset_class="stock",
        yf_ticker="NFLX", broker_symbol="NFLX",
        **_FAST_PARAMS,
    ),
}


def get_config(symbol: str) -> SymbolConfig:
    """Return SymbolConfig for a given symbol. Raises KeyError if not found."""
    return MARKET_CONFIG[symbol]


def all_symbols() -> list[str]:
    """Return list of all configured symbols."""
    return list(MARKET_CONFIG.keys())


def symbols_by_group(group: str) -> list[str]:
    """Return symbols filtered by market group: SAFE | BALANCED | FAST."""
    return [s for s, c in MARKET_CONFIG.items() if c.market_group == group]
