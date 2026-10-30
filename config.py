"""
Configuration settings for AI Text Generation Studio.
Centralizes paths, hyperparameters, and application defaults.
"""

import os
import sys
from pathlib import Path

# On Windows, ensure local DLL directories containing MSVC runtimes are discoverable
if os.name == "nt" and hasattr(os, "add_dll_directory"):
    for search_dir in [
        Path(sys.prefix) / "Scripts",
        Path(sys.prefix) / "Lib" / "site-packages" / "tensorflow",
    ]:
        if search_dir.exists():
            try:
                os.add_dll_directory(str(search_dir))
            except OSError:
                pass

# Paths
BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"
DATASET_DIR = BASE_DIR / "dataset"

# Default Model & Artifact paths
DEFAULT_MODEL_PATH = MODELS_DIR / "lstm_text_gen.keras"
DEFAULT_TOKENIZER_PATH = MODELS_DIR / "tokenizer.json"
DEFAULT_METADATA_PATH = MODELS_DIR / "metadata.json"
SAMPLE_DATASET_PATH = DATASET_DIR / "sample.txt"
SHERLOCK_DATASET_PATH = DATASET_DIR / "sherlock.txt"

# Ensure required directories exist
MODELS_DIR.mkdir(parents=True, exist_ok=True)
DATASET_DIR.mkdir(parents=True, exist_ok=True)

# Training Hyperparameter Defaults
DEFAULT_SEQUENCE_LENGTH = 8
DEFAULT_EMBEDDING_DIM = 64
DEFAULT_LSTM_UNITS = 128
DEFAULT_DROPOUT = 0.2
DEFAULT_BATCH_SIZE = 32
DEFAULT_EPOCHS = 20
DEFAULT_LEARNING_RATE = 0.005
DEFAULT_VALIDATION_SPLIT = 0.15
DEFAULT_EARLY_STOPPING_PATIENCE = 5

# Generation Defaults
DEFAULT_NUM_WORDS = 25
DEFAULT_TEMPERATURE = 0.8
DEFAULT_TOP_K = 10
OOV_TOKEN = "<OOV>"
PAD_TOKEN_ID = 0  # Keras Tokenizer reserves 0 for padding

# Validation Constraints
MIN_DATASET_WORDS = 20
MAX_UPLOAD_SIZE_BYTES = 10 * 1024 * 1024  # 10 MB

# Preset Prompts for Quick Demo
EXAMPLE_PROMPTS = [
    "The Time Traveller was expounding",
    "There are really four dimensions",
    "Can a cube that does not last",
    "To Sherlock Holmes she is always",
    "I see it I deduce it How do I know",
    "It is simplicity itself said he",
]
