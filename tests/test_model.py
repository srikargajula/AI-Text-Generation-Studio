"""
Unit tests for LSTM model architecture, parameter verification, and shapes.
"""

import pytest
import numpy as np
from src.model import build_lstm_model, get_model_specs


class TestModelArchitecture:
    def test_build_model_output_shape(self):
        vocab_size = 50
        seq_len = 8
        embedding_dim = 16
        lstm_units = 32

        model = build_lstm_model(
            vocab_size=vocab_size,
            sequence_length=seq_len,
            embedding_dim=embedding_dim,
            lstm_units=lstm_units,
        )

        assert model.input_shape == (None, seq_len)
        assert model.output_shape == (None, vocab_size)

        # Predict dummy batch
        dummy_x = np.random.randint(0, vocab_size, size=(2, seq_len))
        predictions = model.predict(dummy_x, verbose=0)
        assert predictions.shape == (2, vocab_size)

        # Softmax probabilities should sum to ~1.0
        row_sums = np.sum(predictions, axis=-1)
        np.testing.assert_allclose(row_sums, [1.0, 1.0], atol=1e-5)

    def test_get_model_specs(self):
        model = build_lstm_model(
            vocab_size=30,
            sequence_length=5,
            embedding_dim=16,
            lstm_units=16,
        )
        specs = get_model_specs(model)
        assert specs["total_params"] > 0
        assert specs["trainable_params"] == specs["total_params"]
        assert len(specs["layers"]) >= 4

        layer_names = [l["name"] for l in specs["layers"]]
        assert "word_embedding" in layer_names
        assert "lstm_layer" in layer_names
        assert "token_prediction" in layer_names

    def test_invalid_parameters_raise(self):
        with pytest.raises(ValueError, match="Vocabulary size"):
            build_lstm_model(vocab_size=1)

        with pytest.raises(ValueError, match="Sequence length"):
            build_lstm_model(vocab_size=10, sequence_length=0)

        with pytest.raises(ValueError, match="Embedding dimension"):
            build_lstm_model(vocab_size=10, embedding_dim=2)

        with pytest.raises(ValueError, match="LSTM units"):
            build_lstm_model(vocab_size=10, lstm_units=4)
