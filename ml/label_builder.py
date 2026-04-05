"""
ml/label_builder.py

Generates ML training labels based on actual trade outcome simulation.

Target definition:
  - LONG:  label=1 if price hits TP before SL within lookahead window
  - SHORT: label=1 if price hits TP before SL within lookahead window

No lookahead bias: labels are derived from future bars AFTER the signal bar.
ATR at signal bar is used for SL/TP placement.
"""

import logging
import numpy as np
import pandas as pd

from config.settings import Settings

logger = logging.getLogger(__name__)


class LabelBuilder:
    """
    Generates trade outcome labels for ML training.

    For each bar in a 15M DataFrame:
      - simulate a LONG entry at the close of that bar
      - simulate a SHORT entry at the close of that bar
      - check if TP or SL is hit first within lookahead_bars
      - assign label accordingly
    """

    def __init__(
        self,
        rr_ratio: float = Settings.DEFAULT_RR_RATIO,
        atr_sl_multiplier: float = Settings.ATR_SL_MULTIPLIER,
        lookahead_bars: int = Settings.LABEL_LOOKAHEAD_BARS,
    ):
        self.rr_ratio = rr_ratio
        self.atr_sl_mult = atr_sl_multiplier
        self.lookahead = lookahead_bars

    def build_labels(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Args:
            df: 15M DataFrame with OHLCV + 'atr' column computed.

        Returns:
            DataFrame with two new columns:
              - label_long  : 1 if long would hit TP first, else 0
              - label_short : 1 if short would hit TP first, else 0
        """
        if "atr" not in df.columns:
            raise ValueError("DataFrame must contain 'atr' column before labeling.")

        closes = df["close"].values
        highs  = df["high"].values
        lows   = df["low"].values
        atrs   = df["atr"].values
        n      = len(df)

        label_long  = np.full(n, np.nan)
        label_short = np.full(n, np.nan)

        for i in range(n - self.lookahead - 1):
            entry = closes[i]
            atr   = atrs[i]

            if atr <= 0 or np.isnan(atr):
                continue

            sl_dist = atr * self.atr_sl_mult
            tp_dist = sl_dist * self.rr_ratio

            long_tp  = entry + tp_dist
            long_sl  = entry - sl_dist
            short_tp = entry - tp_dist
            short_sl = entry + sl_dist

            long_result  = self._check_outcome(highs, lows, i+1, long_tp,  long_sl,  is_long=True)
            short_result = self._check_outcome(highs, lows, i+1, short_tp, short_sl, is_long=False)

            label_long[i]  = long_result
            label_short[i] = short_result

        df = df.copy()
        df["label_long"]  = label_long
        df["label_short"] = label_short

        # Drop rows without labels
        df = df.dropna(subset=["label_long", "label_short"])
        df["label_long"]  = df["label_long"].astype(int)
        df["label_short"] = df["label_short"].astype(int)

        logger.info(
            f"Labels built: {len(df)} rows | "
            f"Long win rate: {df['label_long'].mean():.2%} | "
            f"Short win rate: {df['label_short'].mean():.2%}"
        )
        return df

    def _check_outcome(
        self,
        highs: np.ndarray,
        lows: np.ndarray,
        start_idx: int,
        tp: float,
        sl: float,
        is_long: bool,
    ) -> int:
        """
        Check if TP or SL is hit first in the lookahead window.
        Returns 1 if TP hit first, 0 if SL hit first, 0 if neither hit.
        """
        end_idx = min(start_idx + self.lookahead, len(highs))

        for j in range(start_idx, end_idx):
            h = highs[j]
            l = lows[j]

            if is_long:
                if h >= tp:
                    return 1   # TP hit first (assumes price can hit TP in this bar)
                if l <= sl:
                    return 0   # SL hit first
            else:
                if l <= tp:
                    return 1
                if h >= sl:
                    return 0

        return 0  # neither hit within lookahead → treat as loss
