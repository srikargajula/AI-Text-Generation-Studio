"""
Command-line interface for autoregressive text generation using trained LSTM models.

Usage:
    python scripts/generate_cli.py --prompt "The Time Traveller" --words 25 --temp 0.7 --top-k 10
"""

from __future__ import annotations
import argparse
import sys
from pathlib import Path

# Add project root to sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

import config
from utils.storage import load_artifacts, has_saved_model
from src.generation import generate_text


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate text using a trained LSTM neural network."
    )
    parser.add_argument(
        "--prompt",
        "-p",
        type=str,
        default="The Time Traveller was",
        help="Initial seed prompt to guide text synthesis.",
    )
    parser.add_argument(
        "--words",
        "-n",
        type=int,
        default=config.DEFAULT_NUM_WORDS,
        help=f"Number of new words to generate (default: {config.DEFAULT_NUM_WORDS}).",
    )
    parser.add_argument(
        "--temp",
        "-t",
        type=float,
        default=config.DEFAULT_TEMPERATURE,
        help=f"Sampling temperature factor (default: {config.DEFAULT_TEMPERATURE}).",
    )
    parser.add_argument(
        "--top-k",
        "-k",
        type=int,
        default=config.DEFAULT_TOP_K,
        help=f"Top-k candidate restriction (default: {config.DEFAULT_TOP_K}).",
    )
    parser.add_argument(
        "--seed",
        "-s",
        type=int,
        default=None,
        help="Optional random seed for deterministic reproduction.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    if not has_saved_model():
        print("Error: No saved model found in models/ directory.")
        print("Run 'python scripts/train_initial_model.py' first to train a model.")
        sys.exit(1)

    print("Loading model and tokenizer...")
    model, tokenizer, metadata = load_artifacts()

    seq_len = metadata.get("sequence_length", config.DEFAULT_SEQUENCE_LENGTH) if metadata else config.DEFAULT_SEQUENCE_LENGTH

    print(f"\nPrompt:      '{args.prompt}'")
    print(f"Parameters:  words={args.words}, temperature={args.temp}, top_k={args.top_k}")
    print("-" * 60)

    result = generate_text(
        model=model,
        tokenizer=tokenizer,
        seed_text=args.prompt,
        sequence_length=seq_len,
        num_words=args.words,
        temperature=args.temp,
        top_k=args.top_k,
        random_seed=args.seed,
    )

    print("\nGenerated Text:")
    print(result["full_text"])
    print("-" * 60)
    print(f"Tokens synthesized: {result['tokens_generated']}")


if __name__ == "__main__":
    main()
