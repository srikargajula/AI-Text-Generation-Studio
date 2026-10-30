"""
Unit tests for autoregressive text generation, temperature scaling, and sampling.
"""

import pytest
import numpy as np
from src.preprocessing import fit_tokenizer
from src.model import build_lstm_model
from src.generation import (
    sample_with_temperature_and_top_k,
    generate_text,
    compare_temperatures,
    tidy_punctuation,
)


@pytest.fixture(scope="module")
def trained_toy_setup():
    """Builds a small model and tokenizer for deterministic generation testing."""
    sample_corpus = (
        "the time traveller was expounding a recondite matter to us his grey eyes shone "
        "and twinkled and his usually pale face was flushed and animated"
    )
    tokenizer = fit_tokenizer(sample_corpus, oov_token="<OOV>")
    vocab_size = len(tokenizer.word_index) + 1
    seq_len = 4

    model = build_lstm_model(
        vocab_size=vocab_size,
        sequence_length=seq_len,
        embedding_dim=16,
        lstm_units=16,
    )
    return model, tokenizer, seq_len


class TestSamplingLogic:
    def test_low_temperature_is_argmax(self):
        # Distribution strongly favoring token 2
        probs = np.array([0.1, 0.2, 0.6, 0.1])
        token = sample_with_temperature_and_top_k(probs, temperature=0.01)
        assert token == 2

    def test_top_k_restricts_candidates(self):
        probs = np.array([0.05, 0.40, 0.50, 0.05])
        rng = np.random.default_rng(123)
        # top_k=2 should only sample between indices 1 and 2
        samples = [sample_with_temperature_and_top_k(probs, temperature=1.0, top_k=2, rng=rng) for _ in range(30)]
        assert set(samples).issubset({1, 2})

    def test_tidy_punctuation(self):
        messy = "the traveller , who spoke . ( well )"
        clean = tidy_punctuation(messy)
        assert clean == "the traveller, who spoke. (well)"


class TestAutoregressiveGeneration:
    def test_generate_text_returns_expected_structure(self, trained_toy_setup):
        model, tokenizer, seq_len = trained_toy_setup
        result = generate_text(
            model=model,
            tokenizer=tokenizer,
            seed_text="the time traveller",
            sequence_length=seq_len,
            num_words=5,
            temperature=0.8,
            top_k=5,
            random_seed=42,
        )

        assert "full_text" in result
        assert "generated_text" in result
        assert "tokens_generated" in result
        assert result["tokens_generated"] == 5
        assert len(result["step_history"]) == 5
        assert result["prompt"] == "the time traveller"

    def test_generate_text_with_empty_prompt(self, trained_toy_setup):
        model, tokenizer, seq_len = trained_toy_setup
        result = generate_text(
            model=model,
            tokenizer=tokenizer,
            seed_text="",
            sequence_length=seq_len,
            num_words=3,
            temperature=0.7,
        )
        assert result["tokens_generated"] == 3
        assert len(result["generated_text"]) > 0

    def test_generate_text_with_unknown_words(self, trained_toy_setup):
        model, tokenizer, seq_len = trained_toy_setup
        result = generate_text(
            model=model,
            tokenizer=tokenizer,
            seed_text="xylophone supernova zorp",
            sequence_length=seq_len,
            num_words=4,
            temperature=0.5,
        )
        assert result["tokens_generated"] == 4

    def test_compare_temperatures(self, trained_toy_setup):
        model, tokenizer, seq_len = trained_toy_setup
        temps = [0.2, 0.8, 1.4]
        comparison = compare_temperatures(
            model=model,
            tokenizer=tokenizer,
            seed_text="the time",
            sequence_length=seq_len,
            temperatures=temps,
            num_words=4,
            random_seed=42,
        )
        assert len(comparison) == 3
        for t in temps:
            assert t in comparison
            assert isinstance(comparison[t], str)
            assert len(comparison[t]) > 0

    def test_invalid_parameters_raise(self, trained_toy_setup):
        model, tokenizer, seq_len = trained_toy_setup
        with pytest.raises(ValueError, match="Number of words"):
            generate_text(model, tokenizer, "test", seq_len, num_words=0)

        with pytest.raises(ValueError, match="Temperature"):
            generate_text(model, tokenizer, "test", seq_len, temperature=-0.5)
