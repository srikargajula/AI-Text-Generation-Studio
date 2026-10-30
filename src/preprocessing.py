"""
Text preprocessing and tokenization pipeline for AI Text Generation Studio.
Handles text cleaning, validation, vocabulary generation, and training sequence extraction.
"""

from __future__ import annotations
import re
from typing import Any, Dict, List, Tuple
import numpy as np

# Use Keras preprocessing with fallback or lazy loading
from tensorflow.keras.preprocessing.text import Tokenizer, tokenizer_from_json
from tensorflow.keras.preprocessing.sequence import pad_sequences

from config import OOV_TOKEN, MIN_DATASET_WORDS, PAD_TOKEN_ID


def clean_text(
    raw_text: str,
    lowercase: bool = True,
    remove_special_chars: bool = False,
    preserve_newlines: bool = False,
) -> str:
    """
    Cleans and normalizes raw text for language modeling.

    Args:
        raw_text: Unprocessed input text.
        lowercase: Whether to convert text to lowercase.
        remove_special_chars: Whether to strip punctuation entirely or keep standard text punctuation.
        preserve_newlines: Whether newlines should be kept or converted to single spaces.

    Returns:
        Cleaned, normalized string.
    """
    if not raw_text or not isinstance(raw_text, str):
        return ""

    text = raw_text

    # Normalize unicode quotes and dashes
    text = (
        text.replace("“", '"')
        .replace("”", '"')
        .replace("‘", "'")
        .replace("’", "'")
        .replace("—", " - ")
        .replace("–", " - ")
        .replace("…", "...")
    )

    if lowercase:
        text = text.lower()

    if remove_special_chars:
        # Keep only alphanumeric and basic whitespace
        text = re.sub(r"[^a-zA-Z0-9\s]", " ", text)
    else:
        # Normalize punctuation with surrounding spacing for better tokenization
        text = re.sub(r'([.,!?;:()\[\]"\'\-])', r" \1 ", text)

    if not preserve_newlines:
        text = re.sub(r"\s+", " ", text)
    else:
        # Compress multiple spaces but keep single newlines
        lines = [re.sub(r"[ \t]+", " ", line).strip() for line in text.splitlines()]
        text = "\n".join(lines)

    return text.strip()


def validate_text(text: str, min_words: int = MIN_DATASET_WORDS) -> Tuple[bool, str]:
    """
    Validates whether the provided text meets requirements for training.

    Args:
        text: Input string.
        min_words: Minimum threshold of words required.

    Returns:
        Tuple of (is_valid, error_message).
    """
    if not text or not isinstance(text, str):
        return False, "The dataset is empty. Please provide non-empty text."

    words = text.split()
    if len(words) < min_words:
        return (
            False,
            f"The dataset contains only {len(words)} words. A minimum of {min_words} words "
            "is required to create meaningful training sequences.",
        )

    unique_words = set(words)
    if len(unique_words) < 5:
        return (
            False,
            f"The dataset has too few unique words ({len(unique_words)}). "
            "Please provide a more diverse text corpus.",
        )

    return True, ""


def get_dataset_statistics(text: str) -> Dict[str, Any]:
    """
    Calculates detailed descriptive statistics for a text corpus.

    Args:
        text: Raw or cleaned text string.

    Returns:
        Dictionary containing counts, lexical diversity, and word frequencies.
    """
    if not text:
        return {
            "char_count": 0,
            "word_count": 0,
            "line_count": 0,
            "unique_words": 0,
            "lexical_diversity": 0.0,
            "avg_word_length": 0.0,
            "top_words": [],
        }

    lines = text.splitlines()
    words = re.findall(r"\b\w+\b", text.lower())
    total_words = len(words)
    unique_words = len(set(words))
    char_count = len(text)
    line_count = len(lines) if lines else 0

    lexical_diversity = (unique_words / total_words) if total_words > 0 else 0.0
    avg_word_length = (sum(len(w) for w in words) / total_words) if total_words > 0 else 0.0

    # Top words
    from collections import Counter
    top_words = Counter(words).most_common(10)

    return {
        "char_count": char_count,
        "word_count": total_words,
        "line_count": line_count,
        "unique_words": unique_words,
        "lexical_diversity": round(lexical_diversity, 4),
        "avg_word_length": round(avg_word_length, 2),
        "top_words": top_words,
    }


def fit_tokenizer(
    text: str,
    oov_token: str = OOV_TOKEN,
    num_words: int | None = None,
) -> Tokenizer:
    """
    Fits a Keras Tokenizer on the cleaned text corpus.

    Args:
        text: Cleaned text string.
        oov_token: Out-of-vocabulary representation token.
        num_words: Maximum number of words to keep based on frequency.

    Returns:
        Fitted Keras Tokenizer instance.
    """
    tokenizer = Tokenizer(
        num_words=num_words,
        filters='"\t\n',  # keep basic punctuation if desirable, strip tabs/newlines
        lower=True,
        oov_token=oov_token,
    )
    tokenizer.fit_on_texts([text])
    return tokenizer


def create_sequences(
    token_ids: List[int],
    sequence_length: int,
    stride: int = 1,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Converts a sequence of token IDs into input sequences (X) and target next tokens (y)
    using an autoregressive sliding window.

    Args:
        token_ids: List of integer token IDs.
        sequence_length: Fixed length of the input window.
        stride: Step size for advancing the window (default 1).

    Returns:
        Tuple of (X, y) as numpy arrays.
        X shape: (num_samples, sequence_length)
        y shape: (num_samples,)

    Raises:
        ValueError: If token_ids has fewer tokens than sequence_length + 1.
    """
    if len(token_ids) <= sequence_length:
        raise ValueError(
            f"Not enough tokens ({len(token_ids)}) to create sequences of length {sequence_length}. "
            f"Need at least {sequence_length + 1} tokens. Reduce sequence length or add more text."
        )

    if sequence_length < 1:
        raise ValueError("Sequence length must be at least 1.")

    x_list: List[List[int]] = []
    y_list: List[int] = []

    for i in range(0, len(token_ids) - sequence_length, stride):
        x_window = token_ids[i : i + sequence_length]
        y_target = token_ids[i + sequence_length]
        x_list.append(x_window)
        y_list.append(y_target)

    if not x_list:
        raise ValueError("Could not extract any sequences from the dataset.")

    X = np.array(x_list, dtype=np.int32)
    y = np.array(y_list, dtype=np.int32)

    return X, y


def pad_input_sequence(
    token_ids: List[int],
    sequence_length: int,
    padding: str = "pre",
) -> np.ndarray:
    """
    Pads or truncates a token list to match the model's required input length.

    Args:
        token_ids: List of integer token IDs.
        sequence_length: Expected sequence length.
        padding: "pre" or "post" padding.

    Returns:
        2D numpy array of shape (1, sequence_length).
    """
    padded = pad_sequences(
        [token_ids],
        maxlen=sequence_length,
        padding=padding,
        truncating="pre",
        value=PAD_TOKEN_ID,
    )
    return padded
