"""
Unit tests for model and tokenizer persistence, artifact saving and reloading.
"""

import json
import pytest
from pathlib import Path
from src.preprocessing import fit_tokenizer
from src.model import build_lstm_model
from utils.storage import save_artifacts, load_artifacts, has_saved_model


def test_save_and_load_artifacts(tmp_path: Path):
    model_path = tmp_path / "test_model.keras"
    tokenizer_path = tmp_path / "test_tokenizer.json"
    metadata_path = tmp_path / "test_metadata.json"

    # Verify not present initially
    assert not has_saved_model(model_path=model_path, tokenizer_path=tokenizer_path)

    sample_text = "the fourth dimension of space and time is verified by science"
    tokenizer = fit_tokenizer(sample_text, oov_token="<OOV>")
    vocab_size = len(tokenizer.word_index) + 1

    model = build_lstm_model(
        vocab_size=vocab_size,
        sequence_length=4,
        embedding_dim=16,
        lstm_units=16,
    )

    metadata = {
        "dataset_name": "test_corpus",
        "vocab_size": vocab_size,
        "sequence_length": 4,
        "perplexity": 12.34,
        "final_accuracy": 0.45,
    }

    # Call model once to guarantee weights are fully populated
    import numpy as np
    dummy_input = np.array([[1, 2, 3, 4]], dtype=np.int32)
    pred_orig = model.predict(dummy_input, verbose=0)

    # Save artifacts
    save_artifacts(
        model=model,
        tokenizer=tokenizer,
        metadata=metadata,
        model_path=model_path,
        tokenizer_path=tokenizer_path,
        metadata_path=metadata_path,
    )

    # Check existence
    assert has_saved_model(model_path=model_path, tokenizer_path=tokenizer_path)
    assert model_path.exists()
    assert tokenizer_path.exists()
    assert metadata_path.exists()

    # Load artifacts
    loaded_model, loaded_tokenizer, loaded_metadata = load_artifacts(
        model_path=model_path,
        tokenizer_path=tokenizer_path,
        metadata_path=metadata_path,
    )

    assert loaded_model is not None
    assert loaded_tokenizer is not None
    assert loaded_metadata is not None

    # Check loaded tokenizer words
    assert "fourth" in loaded_tokenizer.word_index
    assert "dimension" in loaded_tokenizer.word_index
    assert loaded_tokenizer.word_index["<OOV>"] == 1

    # Check loaded metadata
    assert loaded_metadata["vocab_size"] == vocab_size
    assert loaded_metadata["perplexity"] == 12.34

    # Check loaded model can predict and matches original predictions
    pred_loaded = loaded_model.predict(dummy_input, verbose=0)
    np.testing.assert_allclose(pred_orig, pred_loaded, rtol=1e-5, atol=1e-5)
