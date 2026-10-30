"""
Pre-trains a starter LSTM language model on the bundled sample dataset.
Allows the web application to be immediately interactive on first launch.
"""

import sys
from pathlib import Path

# Add project root to sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

import config
from src.training import train_model

def main():
    print("=" * 60)
    print("Pre-training Starter LSTM Model for AI Text Generation Studio")
    print("=" * 60)

    sample_path = config.SAMPLE_DATASET_PATH
    if not sample_path.exists():
        print(f"Error: Sample dataset not found at {sample_path}")
        sys.exit(1)

    with open(sample_path, "r", encoding="utf-8") as f:
        corpus = f.read()

    print(f"Loaded corpus: {len(corpus)} characters, {len(corpus.split())} words.")
    print("Beginning model training (15 epochs, seq_length=8, embedding_dim=48, lstm_units=64)...")

    def print_progress(epoch, total, logs):
        print(f"  [Epoch {epoch:2d}/{total:2d}] Loss: {logs.get('loss', 0.0):.4f} | "
              f"Acc: {logs.get('accuracy', 0.0)*100:.2f}% | "
              f"Val Loss: {logs.get('val_loss', 0.0):.4f} | "
              f"Val Acc: {logs.get('val_accuracy', 0.0)*100:.2f}%")

    result = train_model(
        text=corpus,
        sequence_length=8,
        embedding_dim=48,
        lstm_units=64,
        dropout_rate=0.2,
        learning_rate=0.005,
        batch_size=32,
        epochs=15,
        validation_split=0.15,
        early_stopping=True,
        early_stopping_patience=4,
        dataset_name="The Time Machine (HG Wells Excerpt)",
        save_to_disk=True,
        on_epoch_end=print_progress,
    )

    m = result["metadata"]
    print("=" * 60)
    print("Training Complete!")
    print(f"Vocabulary Size:      {m['vocab_size']}")
    print(f"Total Sequences:      {m['total_samples']}")
    print(f"Trainable Parameters: {m['trainable_parameters']:,}")
    print(f"Final Accuracy:       {m['final_accuracy']*100:.2f}%")
    print(f"Final Perplexity:     {m.get('perplexity')}")
    print(f"Artifacts saved to:   {config.MODELS_DIR}")
    print("=" * 60)

if __name__ == "__main__":
    main()
