"""
ml/model_registry.py

Singleton registry for the ML inference engine.
Allows the main bot to load once and share across all symbol threads.
"""

import logging
from ml.model_inference import ModelInference

logger = logging.getLogger(__name__)

_instance: ModelInference = None


def get_inference_engine() -> ModelInference:
    """Return singleton ModelInference. Creates and loads if not yet done."""
    global _instance
    if _instance is None:
        _instance = ModelInference()
        ok = _instance.load()
        if not ok:
            logger.warning("ML engine not available. Bot will use INDICATOR-ONLY mode.")
    return _instance


def reload_models():
    """Force reload models (e.g. after retraining)."""
    global _instance
    _instance = ModelInference()
    _instance.load()
    logger.info("ML models reloaded.")
