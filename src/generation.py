"""
Autoregressive text generation module with temperature scaling and top-k sampling.
Provides reproducible, robust next-word prediction without crashing on OOV or empty input.
"""

from __future__ import annotations
import math
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

from tensorflow.keras.preprocessing.sequence import pad_sequences
from config import PAD_TOKEN_ID, DEFAULT_NUM_WORDS, DEFAULT_TEMPERATURE, DEFAULT_TOP_K


def sample_with_temperature_and_top_k(
    probabilities: np.ndarray,
    temperature: float = 1.0,
    top_k: Optional[int] = None,
    rng: Optional[np.random.Generator] = None,
) -> int:
    """
    Applies temperature scaling and top-k filtering to a categorical probability distribution,
    then samples a token index.

    Args:
        probabilities: 1D array of output probabilities from the model's softmax layer.
        temperature: Controls randomness (lower = more deterministic, higher = more random).
        top_k: Restricts sampling to the top k highest probability tokens.
        rng: Optional NumPy random Generator for reproducible sampling.

    Returns:
        Sampled token index (integer).
    """
    if rng is None:
        rng = np.random.default_rng()

    probs = np.copy(probabilities).astype(np.float64)

    # Replace invalid values (NaN, Inf) or zeros with small epsilon
    probs = np.nan_to_num(probs, nan=1e-12, posinf=1.0, neginf=1e-12)
    probs = np.clip(probs, 1e-12, 1.0)

    # Greedily pick the highest probability token for near-zero temperature
    if temperature <= 0.05:
        return int(np.argmax(probs))

    # Convert probabilities back to log-scale logits, apply temperature scaling
    logits = np.log(probs) / float(temperature)

    # Top-k candidate filtering
    if top_k is not None and 0 < top_k < len(logits):
        # Find the threshold value for the k-th largest element
        top_indices = np.argpartition(logits, -top_k)[-top_k:]
        filtered_logits = np.full_like(logits, -np.inf)
        filtered_logits[top_indices] = logits[top_indices]
        logits = filtered_logits

    # Numerically stable softmax
    max_logit = np.max(logits[np.isfinite(logits)])
    exp_logits = np.exp(logits - max_logit)
    sum_exp = np.sum(exp_logits)

    if sum_exp <= 0 or not np.isfinite(sum_exp):
        # Fallback to argmax if numerical underflow occurs
        return int(np.argmax(probabilities))

    rescaled_probs = exp_logits / sum_exp

    # Ensure probabilities sum exactly to 1.0 to avoid float precision errors
    rescaled_probs = rescaled_probs / np.sum(rescaled_probs)

    # Sample from the multinomial distribution
    chosen_index = rng.choice(len(rescaled_probs), p=rescaled_probs)
    return int(chosen_index)


def generate_text(
    model: Any,
    tokenizer: Any,
    seed_text: str,
    sequence_length: int,
    num_words: int = DEFAULT_NUM_WORDS,
    temperature: float = DEFAULT_TEMPERATURE,
    top_k: Optional[int] = DEFAULT_TOP_K,
    random_seed: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Autoregressively generates new words starting from a user prompt.

    Args:
        model: Trained Keras model.
        tokenizer: Fitted Keras Tokenizer.
        seed_text: Initial text prompt.
        sequence_length: Required input sequence length of the model.
        num_words: Number of new tokens to predict.
        temperature: Temperature scaling factor.
        top_k: Top-k candidate restriction (None to disable).
        random_seed: Optional seed for reproducibility.

    Returns:
        Dictionary containing:
            - "full_text": Complete text with prompt and generation.
            - "generated_text": Only the newly generated tokens.
            - "prompt": The original prompt.
            - "tokens_generated": Count of tokens produced.
            - "step_history": Step-by-step token and probability tracking.
    """
    if num_words < 1:
        raise ValueError("Number of words to generate must be at least 1.")
    if temperature < 0.01:
        raise ValueError("Temperature must be greater than 0.")
    if top_k is not None and top_k < 1:
        raise ValueError("top_k must be at least 1 when specified.")

    rng = np.random.default_rng(random_seed)

    # Clean the initial prompt
    prompt = seed_text.strip() if seed_text else ""
    current_text = prompt

    # Create mapping from index to word
    index_to_word = {v: k for k, v in tokenizer.word_index.items()}
    # Fallback for oov token index
    oov_index = tokenizer.word_index.get(getattr(tokenizer, "oov_token", "<OOV>"), 1)

    generated_tokens: List[str] = []
    step_history: List[Dict[str, Any]] = []

    for step in range(num_words):
        # Convert current context into token sequence
        token_sequence = tokenizer.texts_to_sequences([current_text])[0] if current_text else []

        if not token_sequence:
            # If prompt was completely empty or unknown, start with OOV token
            token_sequence = [oov_index]

        # Pad to sequence_length (truncating older words to keep recent context)
        padded_input = pad_sequences(
            [token_sequence],
            maxlen=sequence_length,
            padding="pre",
            truncating="pre",
            value=PAD_TOKEN_ID,
        )

        # Predict next token probabilities
        raw_predictions = model.predict(padded_input, verbose=0)[0]

        # Sample the next token
        next_token_id = sample_with_temperature_and_top_k(
            raw_predictions,
            temperature=temperature,
            top_k=top_k,
            rng=rng,
        )

        # Decode token ID to word
        predicted_word = index_to_word.get(next_token_id, "")

        # If token ID is 0 (padding) or unknown, fallback gracefully
        if not predicted_word or next_token_id == PAD_TOKEN_ID:
            # Pick the top non-zero token
            sorted_indices = np.argsort(raw_predictions)[::-1]
            for idx in sorted_indices:
                if idx != PAD_TOKEN_ID and idx in index_to_word:
                    next_token_id = int(idx)
                    predicted_word = index_to_word[next_token_id]
                    break

        if not predicted_word:
            predicted_word = "..."

        generated_tokens.append(predicted_word)
        current_text = f"{current_text} {predicted_word}".strip()

        # Record step diagnostics
        step_history.append(
            {
                "step": step + 1,
                "token_id": next_token_id,
                "token_word": predicted_word,
                "raw_prob": float(raw_predictions[next_token_id]) if next_token_id < len(raw_predictions) else 0.0,
            }
        )

    # Format the generated text cleanly
    generated_str = " ".join(generated_tokens)
    # Tidy up common punctuation spacing: "word , " -> "word, "
    tidy_generated = tidy_punctuation(generated_str)
    full_tidy_text = f"{prompt} {tidy_generated}".strip() if prompt else tidy_generated

    return {
        "full_text": full_tidy_text,
        "generated_text": tidy_generated,
        "prompt": prompt,
        "tokens_generated": len(generated_tokens),
        "step_history": step_history,
    }


def tidy_punctuation(text: str) -> str:
    """Removes awkward spaces before standard punctuation marks."""
    import re
    text = re.sub(r'\s+([.,!?;:\'])', r'\1', text)
    text = re.sub(r'\(\s+', '(', text)
    text = re.sub(r'\s+\)', ')', text)
    text = re.sub(r'"\s+([^"]+?)\s+"', r'"\1"', text)
    return text.strip()


def compare_temperatures(
    model: Any,
    tokenizer: Any,
    seed_text: str,
    sequence_length: int,
    temperatures: Optional[List[float]] = None,
    num_words: int = 20,
    top_k: Optional[int] = 10,
    random_seed: Optional[int] = 42,
) -> Dict[float, str]:
    """
    Generates text continuations across different temperatures for comparative evaluation.

    Args:
        model: Trained model.
        tokenizer: Fitted tokenizer.
        seed_text: Prompt.
        sequence_length: Model sequence length.
        temperatures: List of temperature values (e.g., [0.2, 0.7, 1.2]).
        num_words: Words per generation.
        top_k: Top-k filter.
        random_seed: Random seed.

    Returns:
        Dictionary mapping temperature to generated continuation.
    """
    if temperatures is None:
        temperatures = [0.2, 0.7, 1.2]

    results: Dict[float, str] = {}
    for temp in temperatures:
        gen = generate_text(
            model=model,
            tokenizer=tokenizer,
            seed_text=seed_text,
            sequence_length=sequence_length,
            num_words=num_words,
            temperature=temp,
            top_k=top_k,
            random_seed=random_seed,
        )
        results[temp] = gen["generated_text"]

    return results
