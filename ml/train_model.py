"""
ml/train_model.py

Full ML training pipeline:
1. Load historical CSVs (1H and 15M)
2. Compute indicators
3. Build feature rows (aligned multi-TF)
4. Generate labels (TP-before-SL)
5. Time-based train/val split
6. Train XGBoost for long and short separately
7. Evaluate
8. Save models and feature list

Run: python -m ml.train_model
"""

import os
import logging
import joblib
import json
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.metrics import (
    roc_auc_score, precision_score, recall_score,
    confusion_matrix, classification_report
)
from xgboost import XGBClassifier

from config.market_config import MARKET_CONFIG, get_config
from config.settings import Settings
from data.data_fetcher import DataFetcher
from indicators.indicator_engine import compute_indicators
from ml.feature_builder import FeatureBuilder
from ml.label_builder import LabelBuilder

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

os.makedirs(Settings.MODEL_DIR, exist_ok=True)


# ─── XGBoost default params ──────────────────────────────────────────────────

XGB_PARAMS = dict(
    n_estimators=300,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    use_label_encoder=False,
    eval_metric="logloss",
    random_state=42,
    n_jobs=-1,
    scale_pos_weight=1,
)


def load_and_prepare(symbol: str) -> pd.DataFrame:
    """Load CSVs, compute indicators, build features, add labels."""
    cfg = get_config(symbol)
    fetcher = DataFetcher()

    df_1h  = fetcher.fetch_historical(symbol, "1h")
    df_15m = fetcher.fetch_historical(symbol, "15m")

    if df_1h.empty or df_15m.empty:
        logger.warning(f"[{symbol}] Missing data. Skipping.")
        return pd.DataFrame()

    df_1h  = compute_indicators(df_1h,  cfg)
    df_15m = compute_indicators(df_15m, cfg)

    # Label builder needs ATR on 15M
    labeler  = LabelBuilder()
    df_15m_l = labeler.build_labels(df_15m)

    # Feature builder — align 1H features to each 15M row
    fb = FeatureBuilder(cfg)
    df_feats = fb.build_training_rows(df_1h, df_15m_l)

    if df_feats.empty:
        return pd.DataFrame()

    # Merge labels
    df_15m_l.index = pd.to_datetime(df_15m_l.index, utc=True)
    df_feats.index = pd.to_datetime(df_feats.index, utc=True)
    df_merged = df_feats.join(df_15m_l[["label_long", "label_short"]], how="inner")

    df_merged.dropna(subset=["label_long", "label_short"], inplace=True)
    df_merged["symbol"] = symbol
    return df_merged


def train_all_symbols(val_fraction: float = 0.20):
    """
    Train one global model across all 30 symbols.
    Uses time-based split per symbol to avoid leakage.
    """
    all_train, all_val = [], []

    for symbol in MARKET_CONFIG:
        logger.info(f"Preparing {symbol}...")
        df = load_and_prepare(symbol)
        if df.empty:
            continue

        n_val = max(int(len(df) * val_fraction), 50)
        train = df.iloc[:-n_val]
        val   = df.iloc[-n_val:]
        all_train.append(train)
        all_val.append(val)

    if not all_train:
        logger.error("No training data collected. Aborting.")
        return

    df_train = pd.concat(all_train).sort_index()
    df_val   = pd.concat(all_val).sort_index()

    logger.info(f"Train size: {len(df_train):,} | Val size: {len(df_val):,}")

    # ── Determine feature columns ────────────────────────────────
    exclude = {"label_long", "label_short", "symbol"}
    feature_cols = [c for c in df_train.columns if c not in exclude]

    # Fill any remaining NaN
    df_train[feature_cols] = df_train[feature_cols].fillna(0)
    df_val[feature_cols]   = df_val[feature_cols].fillna(0)

    X_train = df_train[feature_cols]
    X_val   = df_val[feature_cols]

    # ── Train LONG model ─────────────────────────────────────────
    logger.info("Training LONG model...")
    y_train_long = df_train["label_long"]
    y_val_long   = df_val["label_long"]

    model_long = XGBClassifier(**XGB_PARAMS)
    model_long.fit(
        X_train, y_train_long,
        eval_set=[(X_val, y_val_long)],
        verbose=50,
    )
    _evaluate(model_long, X_val, y_val_long, label_name="LONG", feature_cols=feature_cols)

    # ── Train SHORT model ────────────────────────────────────────
    logger.info("Training SHORT model...")
    y_train_short = df_train["label_short"]
    y_val_short   = df_val["label_short"]

    model_short = XGBClassifier(**XGB_PARAMS)
    model_short.fit(
        X_train, y_train_short,
        eval_set=[(X_val, y_val_short)],
        verbose=50,
    )
    _evaluate(model_short, X_val, y_val_short, label_name="SHORT", feature_cols=feature_cols)

    # ── Save everything ──────────────────────────────────────────
    joblib.dump(model_long,  os.path.join(Settings.MODEL_DIR, "model_long.pkl"))
    joblib.dump(model_short, os.path.join(Settings.MODEL_DIR, "model_short.pkl"))

    meta = {
        "feature_cols": feature_cols,
        "trained_at": datetime.utcnow().isoformat(),
        "train_rows": len(df_train),
        "val_rows": len(df_val),
        "symbols": list(MARKET_CONFIG.keys()),
        "rr_ratio": Settings.DEFAULT_RR_RATIO,
        "atr_sl_multiplier": Settings.ATR_SL_MULTIPLIER,
    }
    with open(os.path.join(Settings.MODEL_DIR, "model_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    logger.info(f"Models saved to {Settings.MODEL_DIR}")


def _evaluate(model, X_val, y_val, label_name: str, feature_cols: list):
    """Print evaluation metrics focused on trading usefulness."""
    proba = model.predict_proba(X_val)[:, 1]
    pred  = (proba >= 0.5).astype(int)

    auc  = roc_auc_score(y_val, proba)
    prec = precision_score(y_val, pred, zero_division=0)
    rec  = recall_score(y_val, pred, zero_division=0)
    cm   = confusion_matrix(y_val, pred)

    # High-confidence trades only
    mask_high = proba >= 0.72
    if mask_high.sum() > 0:
        win_rate_high = y_val[mask_high].mean()
        high_count    = mask_high.sum()
    else:
        win_rate_high = 0.0
        high_count    = 0

    logger.info(f"\n{'='*50}")
    logger.info(f"[{label_name}] ROC AUC: {auc:.4f} | Precision: {prec:.4f} | Recall: {rec:.4f}")
    logger.info(f"[{label_name}] Confusion Matrix:\n{cm}")
    logger.info(f"[{label_name}] High-confidence trades (≥0.72): {high_count} | Win Rate: {win_rate_high:.2%}")

    # Top 15 feature importances
    importances = pd.Series(model.feature_importances_, index=feature_cols)
    top = importances.nlargest(15)
    logger.info(f"[{label_name}] Top 15 features:\n{top.to_string()}")


if __name__ == "__main__":
    train_all_symbols()
