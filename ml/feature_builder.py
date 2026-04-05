"""
ml/feature_builder.py

Builds a flat feature row for the ML model from:
  - 1H indicator DataFrame
  - 15M indicator DataFrame
  - 1M raw OHLCV (micro features computed here)
  - contextual features (symbol, group, time of day, etc.)

The same pipeline is used in BOTH training and live inference.
This guarantees no feature mismatch between train and live.
"""

import logging
from datetime import datetime
from typing import Optional

import numpy as np
import pandas as pd

from config.market_config import SymbolConfig
from indicators.indicator_engine import compute_micro_features

logger = logging.getLogger(__name__)

# ── 1H features to extract ────────────────────────────────────────────────────
_1H_FEATURES = [
    "ema_fast", "ema_slow", "ema_distance", "ema_distance_pct", "ema_bullish",
    "rsi", "macd_line", "macd_signal", "macd_hist",
    "macd_cross_up", "macd_cross_down",
    "atr", "atr_pct", "adx", "di_plus", "di_minus", "adx_trending",
    "trend_direction",
    "candle_return", "candle_body", "upper_wick", "lower_wick", "is_bullish_bar",
    "rolling_vol_10", "rolling_vol_20",
    "ret_1", "ret_3", "ret_5", "ret_10",
    "momentum_pct_10",
]

# ── 15M features to extract ───────────────────────────────────────────────────
_15M_FEATURES = [
    "ema_fast", "ema_slow", "ema_distance", "ema_distance_pct", "ema_bullish",
    "rsi", "macd_line", "macd_signal", "macd_hist",
    "macd_cross_up", "macd_cross_down",
    "atr", "atr_pct", "adx", "adx_trending",
    "trend_direction",
    "candle_return", "candle_body", "upper_wick", "lower_wick", "is_bullish_bar",
    "rolling_vol_10",
    "ret_1", "ret_3", "ret_5",
    "volume_ratio", "volume_elevated",
    "momentum_pct_10",
]


class FeatureBuilder:
    """Builds ML feature rows from multi-timeframe indicator data."""

    def __init__(self, cfg: SymbolConfig):
        self.cfg = cfg

    def build_live_row(
        self,
        df_1h: pd.DataFrame,
        df_15m: pd.DataFrame,
        df_1m: Optional[pd.DataFrame] = None,
        now: Optional[datetime] = None,
    ) -> dict:
        """
        Build a single feature dict for live inference.
        Uses the LAST available row of each timeframe.

        Args:
            df_1h:   1H indicator DataFrame (already computed)
            df_15m:  15M indicator DataFrame (already computed)
            df_1m:   1M raw OHLCV for micro features (optional)
            now:     current UTC datetime for contextual features

        Returns:
            Flat dict of features, or empty dict if data is insufficient.
        """
        if df_1h.empty or df_15m.empty:
            logger.warning(f"[{self.cfg.symbol}] Empty 1H or 15M data for feature building.")
            return {}

        row = {}
        row.update(self._extract_tf_features(df_1h,  _1H_FEATURES,  prefix="h1_"))
        row.update(self._extract_tf_features(df_15m, _15M_FEATURES, prefix="m15_"))

        micro = compute_micro_features(df_1m, lookback=15) if df_1m is not None and not df_1m.empty else {}
        row.update(micro)

        row.update(self._contextual_features(now))

        return row

    def build_training_rows(
        self,
        df_1h: pd.DataFrame,
        df_15m: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Build all training feature rows by aligning 1H and 15M DataFrames.

        For each 15M bar:
          - find the corresponding 1H bar (floor to current hour)
          - extract features from both
          - no lookahead: only use data available at that 15M timestamp

        Returns:
            DataFrame where each row = one training example (no labels yet).
        """
        if df_1h.empty or df_15m.empty:
            logger.warning(f"[{self.cfg.symbol}] Empty data for training feature build.")
            return pd.DataFrame()

        rows = []
        for ts, row_15m in df_15m.iterrows():
            # Get the 1H bar that was CLOSED before or at this 15M timestamp
            h1_slice = df_1h[df_1h.index <= ts]
            if h1_slice.empty:
                continue
            row_1h = h1_slice.iloc[-1]

            feat = {}
            feat["timestamp"] = ts

            # 1H features
            for col in _1H_FEATURES:
                val = row_1h.get(col, np.nan)
                feat[f"h1_{col}"] = float(val) if not pd.isna(val) else 0.0

            # 15M features
            for col in _15M_FEATURES:
                val = row_15m.get(col, np.nan) if col in df_15m.columns else np.nan
                feat[f"m15_{col}"] = float(val) if not pd.isna(val) else 0.0

            # Contextual
            feat.update(self._contextual_features(ts))

            # 1M micro features are NOT available in historical CSV mode → zeros
            for k, v in _empty_micro_dict().items():
                feat[k] = v

            rows.append(feat)

        if not rows:
            return pd.DataFrame()

        df_out = pd.DataFrame(rows).set_index("timestamp")
        df_out.dropna(how="all", inplace=True)
        return df_out

    # ── Internal helpers ─────────────────────────────────────────────────────

    def _extract_tf_features(
        self, df: pd.DataFrame, cols: list[str], prefix: str
    ) -> dict:
        """Extract the last row of df for specified columns, with prefix."""
        last = df.iloc[-1]
        result = {}
        for col in cols:
            val = last.get(col, np.nan) if col in df.columns else np.nan
            result[f"{prefix}{col}"] = float(val) if not pd.isna(val) else 0.0
        return result

    def _contextual_features(self, now: Optional[datetime]) -> dict:
        """Time-based and symbol-based context features."""
        cfg = self.cfg
        feat = {}

        # Symbol encoding
        feat["group_safe"]     = int(cfg.market_group == "SAFE")
        feat["group_balanced"] = int(cfg.market_group == "BALANCED")
        feat["group_fast"]     = int(cfg.market_group == "FAST")

        feat["asset_forex"]  = int(cfg.asset_class == "forex")
        feat["asset_index"]  = int(cfg.asset_class == "index")
        feat["asset_crypto"] = int(cfg.asset_class == "crypto")
        feat["asset_stock"]  = int(cfg.asset_class == "stock")

        if now is not None:
            dt = pd.Timestamp(now)
            feat["hour_of_day"]  = dt.hour
            feat["day_of_week"]  = dt.dayofweek
            feat["is_monday"]    = int(dt.dayofweek == 0)
            feat["is_friday"]    = int(dt.dayofweek == 4)

            # Session flags (UTC)
            feat["session_asia"]   = int(0 <= dt.hour < 8)
            feat["session_london"] = int(7 <= dt.hour < 16)
            feat["session_ny"]     = int(12 <= dt.hour < 21)
            feat["session_overlap"]= int(12 <= dt.hour < 16)
        else:
            for k in ["hour_of_day","day_of_week","is_monday","is_friday",
                      "session_asia","session_london","session_ny","session_overlap"]:
                feat[k] = 0

        return feat


def _empty_micro_dict() -> dict:
    keys = [
        "micro_mean_return", "micro_std_return", "micro_min_return", "micro_max_return",
        "micro_total_return", "micro_mean_body", "micro_mean_upper_wick", "micro_mean_lower_wick",
        "micro_bull_bars", "micro_bear_bars", "micro_bar_ratio", "micro_volatility",
        "micro_last5_up", "micro_last5_down", "micro_momentum", "micro_spike_flag"
    ]
    return {k: 0.0 for k in keys}
