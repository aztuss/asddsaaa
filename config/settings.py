"""
config/settings.py
Global bot settings loaded from environment variables.
"""

import os
from dotenv import load_dotenv

load_dotenv()


class Settings:
    # ── SimpleFX API ──────────────────────────────────────────────
    SIMPLEFX_API_KEY: str = os.getenv("SIMPLEFX_API_KEY", "")
    SIMPLEFX_API_SECRET: str = os.getenv("SIMPLEFX_API_SECRET", "")
    SIMPLEFX_ACCOUNT_ID: str = os.getenv("SIMPLEFX_ACCOUNT_ID", "")
    SIMPLEFX_BASE_URL: str = "https://rest.simplefx.com/api/v3"

    # ── Trading mode ──────────────────────────────────────────────
    TRADING_MODE: str = os.getenv("TRADING_MODE", "DEMO")

    # ── Timeframes ────────────────────────────────────────────────
    TF_MACRO: str = "1h"
    TF_ENTRY: str = "15m"
    TF_MICRO: str = "1m"

    CANDLES_1H: int = 300
    CANDLES_15M: int = 300
    CANDLES_1M: int = 60       # last 60 one-minute candles for live micro features

    # ── Risk defaults ─────────────────────────────────────────────
    DEFAULT_LOT_SIZE: float = float(os.getenv("DEFAULT_LOT_SIZE", 0.002))
    DEFAULT_RR_RATIO: float = float(os.getenv("DEFAULT_RR_RATIO", 1.5))
    MAX_OPEN_TRADES: int = int(os.getenv("MAX_OPEN_TRADES", 10))
    ATR_SL_MULTIPLIER: float = 1.5        # SL = ATR * multiplier

    # ── ML thresholds (defaults, overridden per-symbol) ───────────
    DEFAULT_ML_THRESHOLD_LONG: float = float(os.getenv("DEFAULT_ML_THRESHOLD_LONG", 0.72))
    DEFAULT_ML_THRESHOLD_SHORT: float = float(os.getenv("DEFAULT_ML_THRESHOLD_SHORT", 0.72))

    # ── News filter ───────────────────────────────────────────────
    NEWS_BLOCK_BEFORE_MIN: int = int(os.getenv("NEWS_BLOCK_MINUTES_BEFORE", 15))
    NEWS_BLOCK_AFTER_MIN: int = int(os.getenv("NEWS_BLOCK_MINUTES_AFTER", 15))
    NEWS_REFRESH_INTERVAL_MIN: int = int(os.getenv("NEWS_REFRESH_INTERVAL", 30))

    # ── Logging ───────────────────────────────────────────────────
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
    LOG_DIR: str = os.getenv("LOG_DIR", "logs/")

    # ── Data ──────────────────────────────────────────────────────
    DATA_DIR: str = "data/historical/"
    MODEL_DIR: str = "ml/models/"

    # ── Scan interval ─────────────────────────────────────────────
    SCAN_INTERVAL_SECONDS: int = 60   # scan all markets every 60 seconds

    # ── Label builder ─────────────────────────────────────────────
    LABEL_LOOKAHEAD_BARS: int = 40    # 15M bars to look ahead for TP/SL hit
    LABEL_MIN_ATR_MULT: float = 0.5   # minimum ATR movement to count
