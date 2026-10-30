"""
Model evaluation, perplexity computation, and performance analytics.
Includes Plotly chart generation for loss, accuracy, and temperature analysis.
"""

from __future__ import annotations
import math
from typing import Any, Dict, List, Optional
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def compute_perplexity(loss: float) -> Optional[float]:
    """
    Computes perplexity from cross-entropy loss: PPL = exp(loss).

    Args:
        loss: Cross-entropy loss value.

    Returns:
        Perplexity value as float, or None if invalid.
    """
    if loss is None or not math.isfinite(loss) or loss < 0:
        return None
    try:
        # Cap loss at 100 to prevent math overflow in exp
        capped_loss = min(loss, 100.0)
        return float(math.exp(capped_loss))
    except OverflowError:
        return float("inf")


def evaluate_model_on_sequences(
    model: Any,
    X: np.ndarray,
    y: np.ndarray,
    batch_size: int = 64,
) -> Dict[str, Any]:
    """
    Evaluates model performance on test or validation sequence sets.

    Args:
        model: Trained Keras model.
        X: Input token sequences.
        y: Target token IDs.
        batch_size: Batch size for evaluation.

    Returns:
        Dictionary with loss, accuracy, and computed perplexity.
    """
    results = model.evaluate(X, y, batch_size=batch_size, verbose=0)
    loss = float(results[0])
    acc = float(results[1]) if len(results) > 1 else 0.0
    ppl = compute_perplexity(loss)

    return {
        "loss": round(loss, 4),
        "accuracy": round(acc, 4),
        "perplexity": round(ppl, 2) if ppl is not None else None,
        "sample_count": len(X),
    }


def create_training_history_plots(history: Dict[str, List[float]]) -> go.Figure:
    """
    Generates interactive Plotly dual-panel curves for training/validation loss and accuracy.

    Args:
        history: Keras history dictionary with keys 'loss', 'accuracy',
                 and optional 'val_loss', 'val_accuracy'.

    Returns:
        Plotly Figure object.
    """
    epochs = list(range(1, len(history.get("loss", [])) + 1))

    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=("Cross-Entropy Loss (Lower is Better)", "Token Accuracy (Higher is Better)"),
        horizontal_spacing=0.12,
    )

    # Loss plot (Left)
    if "loss" in history:
        fig.add_trace(
            go.Scatter(
                x=epochs,
                y=history["loss"],
                mode="lines+markers",
                name="Train Loss",
                line=dict(color="#6366f1", width=2.5),
                marker=dict(size=6),
            ),
            row=1,
            col=1,
        )

    if "val_loss" in history and history["val_loss"]:
        fig.add_trace(
            go.Scatter(
                x=epochs,
                y=history["val_loss"],
                mode="lines+markers",
                name="Val Loss",
                line=dict(color="#f43f5e", width=2.5, dash="dash"),
                marker=dict(size=6),
            ),
            row=1,
            col=1,
        )

    # Accuracy plot (Right)
    if "accuracy" in history:
        fig.add_trace(
            go.Scatter(
                x=epochs,
                y=history["accuracy"],
                mode="lines+markers",
                name="Train Acc",
                line=dict(color="#10b981", width=2.5),
                marker=dict(size=6),
            ),
            row=1,
            col=2,
        )

    if "val_accuracy" in history and history["val_accuracy"]:
        fig.add_trace(
            go.Scatter(
                x=epochs,
                y=history["val_accuracy"],
                mode="lines+markers",
                name="Val Acc",
                line=dict(color="#f59e0b", width=2.5, dash="dash"),
                marker=dict(size=6),
            ),
            row=1,
            col=2,
        )

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(15, 23, 42, 0.0)",
        plot_bgcolor="rgba(30, 41, 59, 0.4)",
        font=dict(color="#e2e8f0", family="Inter, sans-serif"),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.25,
            xanchor="center",
            x=0.5,
        ),
        margin=dict(l=40, r=40, t=60, b=60),
        height=380,
    )

    fig.update_xaxes(title_text="Epoch", gridcolor="rgba(255, 255, 255, 0.08)")
    fig.update_yaxes(gridcolor="rgba(255, 255, 255, 0.08)")

    return fig


def get_metrics_explanation() -> Dict[str, str]:
    """Returns technical explanations and limitations of language model metrics."""
    return {
        "perplexity": (
            "Perplexity (PPL = exp(loss)) quantifies how 'surprised' the model is by target tokens. "
            "A lower perplexity indicates the probability distribution is tighter around the actual next words. "
            "However, low perplexity does NOT guarantee logical consistency, factual accuracy, or narrative structure."
        ),
        "accuracy": (
            "Token Accuracy indicates the exact fraction of next tokens correctly predicted from the argmax top-1. "
            "In natural language, multiple different words are often grammatically and contextually valid continuations. "
            "Hence, an accuracy of 30-50% on a rich corpus can still yield creative, coherent generations, "
            "whereas 95% accuracy on small data often signals severe overfitting."
        ),
        "temperature": (
            "Temperature rescales the logits before the softmax activation. "
            "Values < 1.0 sharpen the distribution, favoring high-frequency transitions (more conservative/repetitive). "
            "Values > 1.0 flatten the distribution, increasing vocabulary diversity and novelty at the cost of coherence."
        ),
    }
