"""
ml/model_inference.py

Loads trained models and runs live probability inference.
Returns long_probability, short_probability, and trade_allowed flags.
"""

import os
import json
import logging
from dataclasses import dataclass
from typing import Optional

import joblib
import numpy as np
import pandas as pd

from config.settings import Settings
from config.market_config import SymbolConfig

logger = logging.getLogger(__name__)


@dataclass
class InferenceResult:
    long_probability:  float
    short_probability: float
    long_allowed:      bool
    short_allowed:     bool
    final_confidence:  float
    direction:         str    # "LONG" | "SHORT" | "NONE"
    reason:            str


class ModelInference:
    """
    Loads long and short XGBoost models and runs probability inference.
    Thread-safe for reading; load once at startup.
    """

    def __init__(self, model_dir: str = Settings.MODEL_DIR):
        self.model_dir  = model_dir
        self.model_long  = None
        self.model_short = None
        self.feature_cols: list[str] = []
        self._loaded = False

    def load(self) -> bool:
        """Load models from disk. Returns True if successful."""
        long_path  = os.path.join(self.model_dir, "model_long.pkl")
        short_path = os.path.join(self.model_dir, "model_short.pkl")
        meta_path  = os.path.join(self.model_dir, "model_meta.json")

        if not all(os.path.exists(p) for p in [long_path, short_path, meta_path]):
            logger.warning(
                "Trained models not found. "
                "Run 'python -m ml.train_model' first. "
                "Inference will be DISABLED until models are trained."
            )
            return False

        self.model_long  = joblib.load(long_path)
        self.model_short = joblib.load(short_path)

        with open(meta_path) as f:
            meta = json.load(f)
        self.feature_cols = meta["feature_cols"]

        self._loaded = True
        logger.info(f"ML models loaded. Features: {len(self.feature_cols)}")
        return True

    def predict(
        self,
        feature_row: dict,
        cfg: SymbolConfig,
    ) -> InferenceResult:
        """
        Run inference on a single feature row dict.

        Args:
            feature_row: flat dict of feature values (from FeatureBuilder)
            cfg:         SymbolConfig for this symbol (thresholds)

        Returns:
            InferenceResult with probabilities and trade decision.
        """
        if not self._loaded:
            return InferenceResult(
                long_probability=0.0,
                short_probability=0.0,
                long_allowed=False,
                short_allowed=False,
                final_confidence=0.0,
                direction="NONE",
                reason="Models not loaded",
            )

        if not feature_row:
            return InferenceResult(
                long_probability=0.0,
                short_probability=0.0,
                long_allowed=False,
                short_allowed=False,
                final_confidence=0.0,
                direction="NONE",
                reason="Empty feature row",
            )

        # Build feature vector in correct column order
        X = self._build_vector(feature_row)
        if X is None:
            return InferenceResult(
                long_probability=0.0,
                short_probability=0.0,
                long_allowed=False,
                short_allowed=False,
                final_confidence=0.0,
                direction="NONE",
                reason="Feature vector build failed",
            )

        long_prob  = float(self.model_long.predict_proba(X)[0, 1])
        short_prob = float(self.model_short.predict_proba(X)[0, 1])

        long_allowed  = long_prob  >= cfg.ml_threshold_long  and long_prob  > short_prob
        short_allowed = short_prob >= cfg.ml_threshold_short and short_prob > long_prob

        if long_allowed:
            direction = "LONG"
            confidence = long_prob
            reason = f"Long prob={long_prob:.3f} ≥ threshold={cfg.ml_threshold_long}"
        elif short_allowed:
            direction = "SHORT"
            confidence = short_prob
            reason = f"Short prob={short_prob:.3f} ≥ threshold={cfg.ml_threshold_short}"
        else:
            direction = "NONE"
            confidence = max(long_prob, short_prob)
            reason = (
                f"Long={long_prob:.3f} (need {cfg.ml_threshold_long}), "
                f"Short={short_prob:.3f} (need {cfg.ml_threshold_short})"
            )

        return InferenceResult(
            long_probability=long_prob,
            short_probability=short_prob,
            long_allowed=long_allowed,
            short_allowed=short_allowed,
            final_confidence=confidence,
            direction=direction,
            reason=reason,
        )

    def is_loaded(self) -> bool:
        return self._loaded

    def _build_vector(self, feature_row: dict) -> Optional[np.ndarray]:
        """Align feature_row to trained feature_cols, fill missing with 0."""
        try:
            vec = [feature_row.get(col, 0.0) for col in self.feature_cols]
            return np.array([vec], dtype=np.float32)
        except Exception as e:
            logger.error(f"Feature vector build error: {e}")
            return None
