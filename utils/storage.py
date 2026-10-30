"""
Model and tokenizer persistence utilities.
Handles saving and loading Keras models (.keras), tokenizers (JSON), and metadata.
"""

from __future__ import annotations
import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional, Tuple
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer, tokenizer_from_json

from config import (
    DEFAULT_MODEL_PATH,
    DEFAULT_TOKENIZER_PATH,
    DEFAULT_METADATA_PATH,
    MODELS_DIR,
)

logger = logging.getLogger(__name__)


def save_artifacts(
    model: tf.keras.Model,
    tokenizer: Tokenizer,
    metadata: Dict[str, Any],
    model_path: Path = DEFAULT_MODEL_PATH,
    tokenizer_path: Path = DEFAULT_TOKENIZER_PATH,
    metadata_path: Path = DEFAULT_METADATA_PATH,
) -> None:
    """
    Persists the trained model, tokenizer configuration, and training metadata to disk.

    Args:
        model: Compiled or trained Keras model.
        tokenizer: Fitted Tokenizer instance.
        metadata: Dictionary with training hyperparameters and evaluation metrics.
        model_path: Destination path for model weights (.keras).
        tokenizer_path: Destination path for tokenizer JSON.
        metadata_path: Destination path for metadata JSON.
    """
    model_path.parent.mkdir(parents=True, exist_ok=True)
    tokenizer_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.parent.mkdir(parents=True, exist_ok=True)

    # 1. Save Keras model
    model.save(str(model_path))
    logger.info(f"Model saved to {model_path}")

    # 2. Save Tokenizer JSON
    tokenizer_json = tokenizer.to_json()
    with open(tokenizer_path, "w", encoding="utf-8") as f:
        f.write(tokenizer_json)
    logger.info(f"Tokenizer saved to {tokenizer_path}")

    # 3. Save Metadata JSON
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, default=str)
    logger.info(f"Metadata saved to {metadata_path}")


def load_artifacts(
    model_path: Path = DEFAULT_MODEL_PATH,
    tokenizer_path: Path = DEFAULT_TOKENIZER_PATH,
    metadata_path: Path = DEFAULT_METADATA_PATH,
) -> Tuple[Optional[tf.keras.Model], Optional[Tokenizer], Optional[Dict[str, Any]]]:
    """
    Loads saved model, tokenizer, and metadata from disk.

    Args:
        model_path: Path to .keras model file.
        tokenizer_path: Path to tokenizer.json.
        metadata_path: Path to metadata.json.

    Returns:
        Tuple of (model, tokenizer, metadata). Returns (None, None, None) if files are missing.
    """
    model: Optional[tf.keras.Model] = None
    tokenizer: Optional[Tokenizer] = None
    metadata: Optional[Dict[str, Any]] = None

    if not model_path.exists() or not tokenizer_path.exists():
        return None, None, None

    try:
        model = tf.keras.models.load_model(str(model_path))
    except Exception as e:
        logger.error(f"Failed to load model from {model_path}: {e}")
        return None, None, None

    try:
        with open(tokenizer_path, "r", encoding="utf-8") as f:
            tokenizer_json = f.read()
            tokenizer = tokenizer_from_json(tokenizer_json)
    except Exception as e:
        logger.error(f"Failed to load tokenizer from {tokenizer_path}: {e}")
        return None, None, None

    if metadata_path.exists():
        try:
            with open(metadata_path, "r", encoding="utf-8") as f:
                metadata = json.load(f)
        except Exception as e:
            logger.warning(f"Failed to load metadata from {metadata_path}: {e}")
            metadata = {}

    return model, tokenizer, metadata


def has_saved_model(
    model_path: Path = DEFAULT_MODEL_PATH,
    tokenizer_path: Path = DEFAULT_TOKENIZER_PATH,
) -> bool:
    """Checks whether valid model and tokenizer files exist on disk."""
    return model_path.exists() and tokenizer_path.exists()
