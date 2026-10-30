"""
Unit tests for text preprocessing, cleaning, tokenization, and sequence creation.
"""

import pytest
import numpy as np
from src.preprocessing import (
    clean_text,
    validate_text,
    get_dataset_statistics,
    fit_tokenizer,
    create_sequences,
    pad_input_sequence,
)


class TestTextCleaning:
    def test_clean_text_normalizes_quotes_and_case(self):
        raw = "“Hello World!” — said the Time-Traveller."
        cleaned = clean_text(raw, lowercase=True)
        assert "hello world" in cleaned
        assert '"hello world ! "' in cleaned or "hello world" in cleaned
        assert "“" not in cleaned and "”" not in cleaned

    def test_clean_text_empty_input(self):
        assert clean_text("") == ""
        assert clean_text(None) == ""

    def test_clean_text_remove_special_chars(self):
        raw = "Machine-learning, AI & Deep Neural Networks 100%!"
        cleaned = clean_text(raw, lowercase=True, remove_special_chars=True)
        assert "&" not in cleaned
        assert "%" not in cleaned
        assert "machine" in cleaned


class TestTextValidation:
    def test_validate_text_empty(self):
        is_valid, msg = validate_text("")
        assert not is_valid
        assert "empty" in msg.lower()

    def test_validate_text_insufficient_words(self):
        is_valid, msg = validate_text("A quick fox.", min_words=10)
        assert not is_valid
        assert "minimum" in msg.lower()

    def test_validate_text_low_diversity(self):
        text = "word " * 30  # 30 words but only 1 unique word
        is_valid, msg = validate_text(text, min_words=10)
        assert not is_valid
        assert "unique" in msg.lower()

    def test_validate_text_valid_corpus(self):
        text = "Deep learning and natural language processing allow models to generate realistic text."
        is_valid, msg = validate_text(text, min_words=5)
        assert is_valid
        assert msg == ""


class TestDatasetStatistics:
    def test_statistics_calculation(self):
        sample = "The machine learns language patterns. The machine creates new sentences."
        stats = get_dataset_statistics(sample)
        assert stats["word_count"] > 5
        assert stats["unique_words"] > 3
        assert stats["char_count"] == len(sample)
        assert 0.0 < stats["lexical_diversity"] <= 1.0
        assert len(stats["top_words"]) > 0

    def test_statistics_empty_string(self):
        stats = get_dataset_statistics("")
        assert stats["word_count"] == 0
        assert stats["lexical_diversity"] == 0.0


class TestTokenizerAndSequences:
    def test_fit_tokenizer_vocabulary(self):
        text = "the time traveller was exploring the fourth dimension in space"
        tokenizer = fit_tokenizer(text, oov_token="<OOV>")
        assert "<OOV>" in tokenizer.word_index
        assert "traveller" in tokenizer.word_index
        assert tokenizer.word_index["<OOV>"] == 1
        vocab_size = len(tokenizer.word_index) + 1
        assert vocab_size > 5

    def test_create_sequences_shape(self):
        token_ids = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        seq_len = 4
        X, y = create_sequences(token_ids, sequence_length=seq_len, stride=1)
        # 10 tokens with window 4 gives (10 - 4) = 6 sequences
        assert X.shape == (6, 4)
        assert y.shape == (6,)
        # Check first sliding window
        np.testing.assert_array_equal(X[0], [1, 2, 3, 4])
        assert y[0] == 5
        # Check last sliding window
        np.testing.assert_array_equal(X[-1], [6, 7, 8, 9])
        assert y[-1] == 10

    def test_create_sequences_insufficient_tokens_raises(self):
        token_ids = [1, 2, 3]
        with pytest.raises(ValueError, match="Not enough tokens"):
            create_sequences(token_ids, sequence_length=5)

    def test_pad_input_sequence(self):
        seq = [10, 20, 30]
        padded = pad_input_sequence(seq, sequence_length=6, padding="pre")
        assert padded.shape == (1, 6)
        # First 3 elements should be 0 (padding), last 3 should be 10, 20, 30
        np.testing.assert_array_equal(padded[0], [0, 0, 0, 10, 20, 30])

    def test_pad_input_sequence_truncation(self):
        seq = [1, 2, 3, 4, 5, 6, 7]
        padded = pad_input_sequence(seq, sequence_length=3, padding="pre")
        assert padded.shape == (1, 3)
        # Pre-truncation keeps the most recent 3 tokens: [5, 6, 7]
        np.testing.assert_array_equal(padded[0], [5, 6, 7])
