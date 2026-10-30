"""
Model architecture definition for the LSTM text generation network.
Implements Embedding + LSTM + Dropout + Dense(softmax) pipeline using TensorFlow/Keras.
"""

from __future__ import annotations
from typing import Any, Dict, List
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Embedding, LSTM, Dropout, Dense
from tensorflow.keras.optimizers import Adam

from config import (
    DEFAULT_SEQUENCE_LENGTH,
    DEFAULT_EMBEDDING_DIM,
    DEFAULT_LSTM_UNITS,
    DEFAULT_DROPOUT,
    DEFAULT_LEARNING_RATE,
)


def build_lstm_model(
    vocab_size: int,
    sequence_length: int = DEFAULT_SEQUENCE_LENGTH,
    embedding_dim: int = DEFAULT_EMBEDDING_DIM,
    lstm_units: int = DEFAULT_LSTM_UNITS,
    dropout_rate: float = DEFAULT_DROPOUT,
    learning_rate: float = DEFAULT_LEARNING_RATE,
) -> tf.keras.Model:
    """
    Constructs and compiles an autoregressive LSTM language model.

    Architecture:
        Input(shape=(sequence_length,), dtype="int32")
        -> Embedding(input_dim=vocab_size, output_dim=embedding_dim)
        -> LSTM(units=lstm_units)
        -> Dropout(rate=dropout_rate)
        -> Dense(units=vocab_size, activation="softmax")

    Args:
        vocab_size: Total vocabulary size (including padding token 0).
        sequence_length: Context window length (number of input tokens).
        embedding_dim: Dimensionality of dense word embeddings.
        lstm_units: Number of recurrent memory units in the LSTM layer.
        dropout_rate: Dropout probability for regularization.
        learning_rate: Optimizer learning rate.

    Returns:
        Compiled and built tf.keras.Model instance.
    """
    if vocab_size < 2:
        raise ValueError("Vocabulary size must be at least 2.")
    if sequence_length < 1:
        raise ValueError("Sequence length must be at least 1.")
    if embedding_dim < 4:
        raise ValueError("Embedding dimension must be at least 4.")
    if lstm_units < 8:
        raise ValueError("LSTM units must be at least 8.")

    model = Sequential(
        [
            Input(shape=(sequence_length,), dtype="int32", name="token_input"),
            Embedding(
                input_dim=vocab_size,
                output_dim=embedding_dim,
                name="word_embedding",
            ),
            LSTM(
                units=lstm_units,
                return_sequences=False,
                name="lstm_layer",
            ),
            Dropout(
                rate=dropout_rate,
                name="dropout_regularization",
            ),
            Dense(
                units=vocab_size,
                activation="softmax",
                name="token_prediction",
            ),
        ],
        name="ai_text_generation_lstm",
    )

    optimizer = Adam(learning_rate=learning_rate)
    model.compile(
        optimizer=optimizer,
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    return model


def get_model_specs(model: tf.keras.Model) -> Dict[str, Any]:
    """
    Extracts structural metadata and parameter counts from a Keras model.

    Args:
        model: Compiled or uncompiled tf.keras.Model.

    Returns:
        Dictionary containing layer specs and parameter counts.
    """
    trainable_params = int(
        sum(tf.keras.backend.count_params(w) for w in model.trainable_weights)
    )
    non_trainable_params = int(
        sum(tf.keras.backend.count_params(w) for w in model.non_trainable_weights)
    )
    total_params = trainable_params + non_trainable_params

    layers_info: List[Dict[str, Any]] = []
    for layer in model.layers:
        layers_info.append(
            {
                "name": layer.name,
                "class_name": layer.__class__.__name__,
                "output_shape": str(getattr(layer, "output_shape", None)),
                "params": layer.count_params(),
            }
        )

    return {
        "model_name": model.name,
        "total_params": total_params,
        "trainable_params": trainable_params,
        "non_trainable_params": non_trainable_params,
        "layers": layers_info,
    }
