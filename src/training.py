"""
Model training pipeline for the LSTM text generation system.
Encapsulates dataset preparation, model compilation, training callbacks, and artifact persistence.
"""

from __future__ import annotations
import datetime
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Tuple
import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import Callback, EarlyStopping

from config import (
    DEFAULT_SEQUENCE_LENGTH,
    DEFAULT_EMBEDDING_DIM,
    DEFAULT_LSTM_UNITS,
    DEFAULT_DROPOUT,
    DEFAULT_BATCH_SIZE,
    DEFAULT_EPOCHS,
    DEFAULT_LEARNING_RATE,
    DEFAULT_VALIDATION_SPLIT,
    DEFAULT_EARLY_STOPPING_PATIENCE,
    DEFAULT_MODEL_PATH,
    DEFAULT_TOKENIZER_PATH,
    DEFAULT_METADATA_PATH,
    OOV_TOKEN,
)
from src.preprocessing import clean_text, validate_text, fit_tokenizer, create_sequences
from src.model import build_lstm_model, get_model_specs
from src.evaluation import compute_perplexity
from utils.storage import save_artifacts


class ProgressCallback(Callback):
    """Custom Keras callback that notifies external progress reporters (e.g. Streamlit)."""

    def __init__(
        self,
        total_epochs: int,
        on_epoch_end_fn: Optional[Callable[[int, int, Dict[str, float]], None]] = None,
    ):
        super().__init__()
        self.total_epochs = total_epochs
        self.on_epoch_end_fn = on_epoch_end_fn

    def on_epoch_end(self, epoch: int, logs: Optional[Dict[str, Any]] = None):
        if self.on_epoch_end_fn and logs:
            clean_logs = {k: float(v) for k, v in logs.items()}
            self.on_epoch_end_fn(epoch + 1, self.total_epochs, clean_logs)


def train_model(
    text: str,
    sequence_length: int = DEFAULT_SEQUENCE_LENGTH,
    embedding_dim: int = DEFAULT_EMBEDDING_DIM,
    lstm_units: int = DEFAULT_LSTM_UNITS,
    dropout_rate: float = DEFAULT_DROPOUT,
    learning_rate: float = DEFAULT_LEARNING_RATE,
    batch_size: int = DEFAULT_BATCH_SIZE,
    epochs: int = DEFAULT_EPOCHS,
    validation_split: float = DEFAULT_VALIDATION_SPLIT,
    early_stopping: bool = True,
    early_stopping_patience: int = DEFAULT_EARLY_STOPPING_PATIENCE,
    dataset_name: str = "custom_corpus",
    save_to_disk: bool = True,
    model_path: Path = DEFAULT_MODEL_PATH,
    tokenizer_path: Path = DEFAULT_TOKENIZER_PATH,
    metadata_path: Path = DEFAULT_METADATA_PATH,
    on_epoch_end: Optional[Callable[[int, int, Dict[str, float]], None]] = None,
) -> Dict[str, Any]:
    """
    Executes the complete training workflow:
    1. Validates and cleans the text corpus.
    2. Fits a Tokenizer and transforms text into tokens.
    3. Builds sliding-window sequence pairs (X, y).
    4. Compiles the LSTM neural network.
    5. Fits the model with optional early stopping and progress reporting.
    6. Persists artifacts and compiles metadata.

    Args:
        text: Training text corpus.
        sequence_length: Sliding window length.
        embedding_dim: Word embedding dimensions.
        lstm_units: Recurrent hidden units.
        dropout_rate: Regularization rate.
        learning_rate: Adam optimizer learning rate.
        batch_size: Training batch size.
        epochs: Number of training epochs.
        validation_split: Fraction of sequences used for validation.
        early_stopping: Whether to halt training if validation loss plateaus.
        early_stopping_patience: Patience for early stopping.
        dataset_name: Identifier for dataset in metadata.
        save_to_disk: Whether to persist artifacts to disk.
        model_path: Output path for .keras model.
        tokenizer_path: Output path for tokenizer.json.
        metadata_path: Output path for metadata.json.
        on_epoch_end: Optional callback function(current_epoch, total_epochs, logs).

    Returns:
        Dictionary containing model, tokenizer, metadata, history, and evaluation stats.
    """
    # 1. Validation
    is_valid, error_msg = validate_text(text)
    if not is_valid:
        raise ValueError(error_msg)

    # 2. Cleaning & Normalization
    cleaned = clean_text(text, lowercase=True, remove_special_chars=False)

    # 3. Tokenizer Fitting
    tokenizer = fit_tokenizer(cleaned, oov_token=OOV_TOKEN)
    token_ids = tokenizer.texts_to_sequences([cleaned])[0]

    # Vocabulary size includes index 0 (reserved for padding) + all indexed words
    vocab_size = len(tokenizer.word_index) + 1

    # 4. Sequence Creation
    X, y = create_sequences(token_ids, sequence_length=sequence_length, stride=1)
    total_samples = len(X)

    # 5. Build Model
    model = build_lstm_model(
        vocab_size=vocab_size,
        sequence_length=sequence_length,
        embedding_dim=embedding_dim,
        lstm_units=lstm_units,
        dropout_rate=dropout_rate,
        learning_rate=learning_rate,
    )

    # 6. Callbacks
    callbacks: list[Callback] = []
    if early_stopping:
        monitor_metric = "val_loss" if validation_split > 0.0 else "loss"
        callbacks.append(
            EarlyStopping(
                monitor=monitor_metric,
                patience=early_stopping_patience,
                restore_best_weights=True,
                verbose=0,
            )
        )

    if on_epoch_end:
        callbacks.append(ProgressCallback(total_epochs=epochs, on_epoch_end_fn=on_epoch_end))

    # 7. Train Model
    history_obj = model.fit(
        X,
        y,
        batch_size=batch_size,
        epochs=epochs,
        validation_split=validation_split if validation_split > 0 else 0.0,
        callbacks=callbacks,
        verbose=1,
    )

    history = {k: [float(v) for v in vals] for k, vals in history_obj.history.items()}

    # Compute final metrics
    final_loss = float(history["loss"][-1])
    final_acc = float(history["accuracy"][-1])
    final_val_loss = float(history["val_loss"][-1]) if "val_loss" in history else None
    final_val_acc = float(history["val_accuracy"][-1]) if "val_accuracy" in history else None

    # Compute perplexity from validation loss or training loss
    eval_loss = final_val_loss if final_val_loss is not None else final_loss
    perplexity = compute_perplexity(eval_loss)

    model_specs = get_model_specs(model)

    metadata: Dict[str, Any] = {
        "dataset_name": dataset_name,
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "total_samples": total_samples,
        "vocab_size": vocab_size,
        "sequence_length": sequence_length,
        "embedding_dim": embedding_dim,
        "lstm_units": lstm_units,
        "dropout_rate": dropout_rate,
        "learning_rate": learning_rate,
        "batch_size": batch_size,
        "epochs_trained": len(history["loss"]),
        "requested_epochs": epochs,
        "final_loss": round(final_loss, 4),
        "final_accuracy": round(final_acc, 4),
        "final_val_loss": round(final_val_loss, 4) if final_val_loss is not None else None,
        "final_val_accuracy": round(final_val_acc, 4) if final_val_acc is not None else None,
        "perplexity": round(perplexity, 2) if perplexity is not None else None,
        "trainable_parameters": model_specs["trainable_params"],
        "total_parameters": model_specs["total_params"],
        "history": history,
    }

    # 8. Save Artifacts
    if save_to_disk:
        save_artifacts(
            model=model,
            tokenizer=tokenizer,
            metadata=metadata,
            model_path=model_path,
            tokenizer_path=tokenizer_path,
            metadata_path=metadata_path,
        )

    return {
        "model": model,
        "tokenizer": tokenizer,
        "metadata": metadata,
        "history": history,
        "X_shape": X.shape,
        "y_shape": y.shape,
    }
