"""
AI Text Generation Studio
Interactive web application for word-level LSTM language modeling, training, evaluation, and inference.
Built with Streamlit, TensorFlow/Keras, and Plotly.
"""

from __future__ import annotations
import os
import json
from pathlib import Path
from typing import Optional, Tuple
import streamlit as st
import pandas as pd
import plotly.express as px

# Configuration and backend modules
import config
from src.preprocessing import (
    clean_text,
    validate_text,
    get_dataset_statistics,
    fit_tokenizer,
)
from src.training import train_model
from src.generation import generate_text, compare_temperatures
from src.evaluation import create_training_history_plots, get_metrics_explanation
from utils.storage import load_artifacts, has_saved_model

# Page configuration
st.set_page_config(
    page_title="AI Text Generation Studio",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS styling for a polished dark theme
CUSTOM_CSS = """
<style>
    /* Global styles */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    
    code, pre {
        font-family: 'JetBrains Mono', monospace !important;
    }

    /* Hero section */
    .hero-container {
        background: linear-gradient(135deg, rgba(30, 41, 59, 0.8) 0%, rgba(15, 23, 42, 0.9) 100%);
        border: 1px solid rgba(99, 102, 241, 0.25);
        border-radius: 12px;
        padding: 24px;
        margin-bottom: 24px;
        box-shadow: 0 8px 24px rgba(0, 0, 0, 0.2);
    }
    
    .hero-title {
        font-size: 2rem;
        font-weight: 700;
        background: linear-gradient(90deg, #818cf8, #c084fc, #38bdf8);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 8px;
    }
    
    .hero-subtitle {
        color: #94a3b8;
        font-size: 1.05rem;
        line-height: 1.6;
    }

    /* Metric Card */
    .metric-card {
        background: #1e293b;
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 10px;
        padding: 16px;
        text-align: center;
        box-shadow: 0 4px 12px rgba(0,0,0,0.15);
    }
    
    .metric-value {
        font-size: 1.6rem;
        font-weight: 700;
        color: #f8fafc;
    }
    
    .metric-label {
        font-size: 0.85rem;
        color: #94a3b8;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-top: 4px;
    }

    /* Status Badges */
    .badge-online {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 9999px;
        background: rgba(16, 185, 129, 0.15);
        color: #34d399;
        font-weight: 600;
        font-size: 0.85rem;
        border: 1px solid rgba(52, 211, 153, 0.3);
    }
    
    .badge-offline {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 9999px;
        background: rgba(244, 63, 94, 0.15);
        color: #fb7185;
        font-weight: 600;
        font-size: 0.85rem;
        border: 1px solid rgba(251, 113, 133, 0.3);
    }

    /* Output Card */
    .output-box {
        background: #0f172a;
        border: 1px solid rgba(99, 102, 241, 0.3);
        border-radius: 10px;
        padding: 20px;
        font-size: 1.1rem;
        line-height: 1.7;
        color: #e2e8f0;
        min-height: 120px;
        box-shadow: inset 0 2px 8px rgba(0, 0, 0, 0.3);
    }

    .output-seed {
        color: #818cf8;
        font-weight: 600;
    }
    
    .output-generated {
        color: #38bdf8;
    }

    /* Clean Streamlit elements */
    .stButton>button {
        border-radius: 8px;
        font-weight: 500;
        transition: all 0.2s ease;
    }
    
    div[data-testid="stSidebar"] {
        background-color: #0b1120;
        border-right: 1px solid rgba(255, 255, 255, 0.08);
    }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# Session state initialization
if "prompt_input" not in st.session_state:
    st.session_state.prompt_input = config.EXAMPLE_PROMPTS[0]
if "dataset_text" not in st.session_state:
    # Preload sample dataset if available
    if config.SAMPLE_DATASET_PATH.exists():
        with open(config.SAMPLE_DATASET_PATH, "r", encoding="utf-8") as f:
            st.session_state.dataset_text = f.read()
            st.session_state.dataset_name = "The Time Machine (HG Wells Excerpt)"
    else:
        st.session_state.dataset_text = ""
        st.session_state.dataset_name = "None"
if "last_generation" not in st.session_state:
    st.session_state.last_generation = None
if "active_nav" not in st.session_state:
    st.session_state.active_nav = "Home Dashboard"


@st.cache_resource(show_spinner=False)
def get_cached_model_and_tokenizer():
    """Caches loaded model, tokenizer, and metadata in memory."""
    if has_saved_model():
        return load_artifacts()
    return None, None, None


# Fetch loaded artifacts
model, tokenizer, metadata = get_cached_model_and_tokenizer()
model_loaded = model is not None and tokenizer is not None


# Sidebar Navigation
with st.sidebar:
    st.markdown("### 🧠 AI Text Generation Studio")
    st.caption("Word-Level LSTM Language Modeling")
    st.markdown("---")

    nav_options = [
        "Home Dashboard",
        "Text Generation",
        "Dataset Management",
        "Model Training",
        "Evaluation & Analytics",
        "Model Architecture & Theory",
    ]

    selected_nav = st.radio(
        "Navigation",
        options=nav_options,
        index=nav_options.index(st.session_state.active_nav) if st.session_state.active_nav in nav_options else 0,
    )
    st.session_state.active_nav = selected_nav

    st.markdown("---")
    st.markdown("#### System Status")
    if model_loaded:
        st.markdown('<span class="badge-online">● Model Active</span>', unsafe_allow_html=True)
        vocab_sz = metadata.get("vocab_size", "Unknown") if metadata else len(tokenizer.word_index) + 1
        st.caption(f"**Vocab Size:** {vocab_sz} tokens")
        seq_l = metadata.get("sequence_length", config.DEFAULT_SEQUENCE_LENGTH) if metadata else config.DEFAULT_SEQUENCE_LENGTH
        st.caption(f"**Seq Length:** {seq_l} words")
        if metadata and "final_accuracy" in metadata:
            st.caption(f"**Accuracy:** {metadata['final_accuracy'] * 100:.1f}%")
        if metadata and "perplexity" in metadata and metadata["perplexity"]:
            st.caption(f"**Perplexity:** {metadata['perplexity']:.2f}")
    else:
        st.markdown('<span class="badge-offline">○ No Model Loaded</span>', unsafe_allow_html=True)
        st.caption("Train a model or load bundled weights to start generating.")

    st.markdown("---")
    st.caption("AI/ML Portfolio Project | Python 3.11+ | Keras / TensorFlow")


# ==============================================================================
# SECTION A: HOME DASHBOARD
# ==============================================================================
if selected_nav == "Home Dashboard":
    st.markdown(
        """
        <div class="hero-container">
            <div class="hero-title">AI Text Generation Studio</div>
            <div class="hero-subtitle">
                An end-to-end deep learning language modeling platform. Learn linguistic structures from arbitrary text corpora, 
                compute continuous word embeddings, train recurrent LSTM networks, and autoregressively predict next tokens 
                with temperature scaling and top-k nucleus sampling.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Top metrics row
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-value">{"Active" if model_loaded else "Empty"}</div>
                <div class="metric-label">Model Status</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with m2:
        vocab_display = str(metadata.get("vocab_size", "N/A")) if (model_loaded and metadata) else "N/A"
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-value">{vocab_display}</div>
                <div class="metric-label">Vocabulary Size</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with m3:
        params_display = f"{metadata.get('trainable_parameters', 0):,}" if (model_loaded and metadata) else "0"
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-value">{params_display}</div>
                <div class="metric-label">Trainable Parameters</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with m4:
        ppl_display = f"{metadata.get('perplexity', 0.0):.2f}" if (model_loaded and metadata and metadata.get("perplexity")) else "N/A"
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-value">{ppl_display}</div>
                <div class="metric-label">Perplexity (PPL)</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)

    # Quick Navigation Cards
    st.subheader("Explore Studio Capabilities")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("#### ✍️ Generate Text")
        st.write("Synthesize new sentences using autoregressive next-word prediction with temperature & top-k sampling.")
        if st.button("Go to Generation →", key="btn_nav_gen"):
            st.session_state.active_nav = "Text Generation"
            st.rerun()

    with c2:
        st.markdown("#### 📚 Dataset Hub")
        st.write("Upload custom text or explore the bundled corpus with vocabulary analysis and lexical diversity metrics.")
        if st.button("Manage Datasets →", key="btn_nav_data"):
            st.session_state.active_nav = "Dataset Management"
            st.rerun()

    with c3:
        st.markdown("#### 🚀 Model Training")
        st.write("Configure embedding dimensions, recurrent units, sequence windows, and train your own LSTM on CPU.")
        if st.button("Train Model →", key="btn_nav_train"):
            st.session_state.active_nav = "Model Training"
            st.rerun()

    st.markdown("---")

    # Sample Prompts Showcase
    st.subheader("💡 Example Seed Prompts")
    st.write("Select any prompt below to populate the generation input:")
    cols = st.columns(3)
    for idx, prompt_text in enumerate(config.EXAMPLE_PROMPTS):
        col = cols[idx % 3]
        with col:
            if st.button(f"“{prompt_text}...”", key=f"sample_prompt_{idx}", use_container_width=True):
                st.session_state.prompt_input = prompt_text
                st.session_state.active_nav = "Text Generation"
                st.rerun()


# ==============================================================================
# SECTION B: TEXT GENERATION
# ==============================================================================
elif selected_nav == "Text Generation":
    st.markdown("## ✍️ Autoregressive Text Generation")
    st.caption("Prompt the LSTM neural network and observe stochastic next-token autoregression in action.")

    if not model_loaded:
        st.warning("⚠️ No trained model found! Please go to **Model Training** to train a model or load default weights.")
        if st.button("Train a Model Now"):
            st.session_state.active_nav = "Model Training"
            st.rerun()
    else:
        seq_length = metadata.get("sequence_length", config.DEFAULT_SEQUENCE_LENGTH) if metadata else config.DEFAULT_SEQUENCE_LENGTH

        # Generation control panel
        with st.expander("⚙️ Generation Hyperparameters & Controls", expanded=True):
            col_ctrl1, col_ctrl2, col_ctrl3, col_ctrl4 = st.columns(4)
            with col_ctrl1:
                num_words = st.slider("Output Length (Words)", min_value=5, max_value=100, value=config.DEFAULT_NUM_WORDS, step=5)
            with col_ctrl2:
                temperature = st.slider("Temperature (Randomness)", min_value=0.1, max_value=1.8, value=config.DEFAULT_TEMPERATURE, step=0.05,
                                        help="Lower values (0.2) yield conservative, repetitive text; higher values (1.2) yield diverse, creative text.")
            with col_ctrl3:
                top_k = st.slider("Top-K Sampling", min_value=1, max_value=50, value=config.DEFAULT_TOP_K, step=1,
                                  help="Restricts token candidate pool at each step to the top K most probable words.")
            with col_ctrl4:
                use_seed = st.checkbox("Fix Random Seed", value=False)
                seed_val = st.number_input("Seed Value", min_value=0, max_value=999999, value=42) if use_seed else None

        # Text input prompt
        prompt = st.text_area(
            "Starting Prompt (Seed Context):",
            value=st.session_state.prompt_input,
            height=100,
            placeholder="Type a sentence or clause to kickstart generation...",
        )
        st.session_state.prompt_input = prompt

        # Quick action buttons
        col_btn1, col_btn2, col_btn3 = st.columns([2, 1, 1])
        with col_btn1:
            generate_clicked = st.button("⚡ Generate Text", type="primary", use_container_width=True)
        with col_btn2:
            clear_clicked = st.button("🧹 Clear", use_container_width=True)
        with col_btn3:
            random_sample_clicked = st.button("🎲 Random Prompt", use_container_width=True)

        if clear_clicked:
            st.session_state.prompt_input = ""
            st.session_state.last_generation = None
            st.rerun()

        if random_sample_clicked:
            import random
            st.session_state.prompt_input = random.choice(config.EXAMPLE_PROMPTS)
            st.rerun()

        if generate_clicked:
            if not prompt.strip():
                st.error("Please enter a non-empty starting prompt to guide text generation.")
            else:
                with st.spinner("Neural network is predicting tokens step-by-step..."):
                    try:
                        result = generate_text(
                            model=model,
                            tokenizer=tokenizer,
                            seed_text=prompt,
                            sequence_length=seq_length,
                            num_words=num_words,
                            temperature=temperature,
                            top_k=top_k,
                            random_seed=seed_val,
                        )
                        st.session_state.last_generation = result
                    except Exception as e:
                        st.error(f"Error during generation: {e}")

        # Display Generation Output
        if st.session_state.last_generation:
            gen_data = st.session_state.last_generation
            st.markdown("### 📄 Generated Continuation")

            # Display styled output card
            st.markdown(
                f"""
                <div class="output-box">
                    <span class="output-seed">{gen_data['prompt']}</span>
                    <span class="output-generated"> {gen_data['generated_text']}</span>
                </div>
                """,
                unsafe_allow_html=True,
            )

            col_copy, col_regen = st.columns([3, 1])
            with col_copy:
                st.caption(f"Generated {gen_data['tokens_generated']} tokens | Context Window: {seq_length} words")
            with col_regen:
                st.code(gen_data["full_text"], language="text")

            # Detailed Token Step Diagnostics
            with st.expander("🔍 Step-by-Step Prediction Diagnostics"):
                st.write("Detailed breakdown of token sampled and model prediction confidence at each autoregressive step:")
                df_steps = pd.DataFrame(gen_data["step_history"])
                if not df_steps.empty:
                    df_steps["raw_prob"] = df_steps["raw_prob"].apply(lambda p: f"{p * 100:.2f}%")
                    st.dataframe(df_steps, use_container_width=True)


# ==============================================================================
# SECTION C: DATASET MANAGEMENT
# ==============================================================================
elif selected_nav == "Dataset Management":
    st.markdown("## 📚 Dataset Management & Lexical Analysis")
    st.caption("Upload custom training corpora, analyze vocabulary distribution, and inspect token statistics.")

    tab_upload, tab_sample = st.tabs(["📁 Upload Custom Dataset", "📖 Bundled Sample Corpus"])

    with tab_upload:
        uploaded_file = st.file_uploader("Upload a text corpus (.txt)", type=["txt"])
        if uploaded_file is not None:
            if uploaded_file.size > config.MAX_UPLOAD_SIZE_BYTES:
                st.error("Uploaded file exceeds 10MB limit. Please upload a smaller text file.")
            else:
                try:
                    content = uploaded_file.read().decode("utf-8", errors="replace")
                    st.session_state.dataset_text = content
                    st.session_state.dataset_name = uploaded_file.name
                    st.success(f"Successfully loaded '{uploaded_file.name}' ({len(content)} characters).")
                except Exception as e:
                    st.error(f"Failed to read file: {e}")

    with tab_sample:
        st.write("Explore bundled classic public-domain literary corpora for immediate training and benchmarking:")
        c_ds1, c_ds2 = st.columns(2)
        with c_ds1:
            st.markdown("##### ⏳ The Time Machine (H.G. Wells)")
            st.caption("Science fiction corpus focusing on time, dimensions, geometry, and philosophical speculation.")
            if st.button("Load The Time Machine Sample", use_container_width=True):
                if config.SAMPLE_DATASET_PATH.exists():
                    with open(config.SAMPLE_DATASET_PATH, "r", encoding="utf-8") as f:
                        st.session_state.dataset_text = f.read()
                        st.session_state.dataset_name = "The Time Machine (HG Wells Excerpt)"
                    st.success("Loaded 'The Time Machine' sample dataset.")
                    st.rerun()
                else:
                    st.error("Sample dataset file not found on disk.")

        with c_ds2:
            st.markdown("##### 🔍 Sherlock Holmes (Arthur Conan Doyle)")
            st.caption("Victorian detective mystery corpus emphasizing forensic deduction, dialogue, and keen observation.")
            if st.button("Load Sherlock Holmes Sample", use_container_width=True):
                if config.SHERLOCK_DATASET_PATH.exists():
                    with open(config.SHERLOCK_DATASET_PATH, "r", encoding="utf-8") as f:
                        st.session_state.dataset_text = f.read()
                        st.session_state.dataset_name = "Sherlock Holmes (Conan Doyle Excerpt)"
                    st.success("Loaded 'Sherlock Holmes' sample dataset.")
                    st.rerun()
                else:
                    st.error("Sherlock dataset file not found on disk.")

    st.markdown("---")

    # Dataset Inspection and Metrics
    dataset_content = st.session_state.dataset_text
    if dataset_content:
        st.markdown(f"### Corpus Profile: `{st.session_state.dataset_name}`")
        stats = get_dataset_statistics(dataset_content)

        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Total Words", f"{stats['word_count']:,}")
        c2.metric("Characters", f"{stats['char_count']:,}")
        c3.metric("Lines", f"{stats['line_count']:,}")
        c4.metric("Unique Words", f"{stats['unique_words']:,}")
        c5.metric("Lexical Diversity", f"{stats['lexical_diversity'] * 100:.1f}%")

        # Top words chart
        if stats["top_words"]:
            st.markdown("#### Most Frequent Words")
            df_top = pd.DataFrame(stats["top_words"], columns=["Word", "Frequency"])
            fig_bar = px.bar(
                df_top,
                x="Word",
                y="Frequency",
                text="Frequency",
                template="plotly_dark",
                color="Frequency",
                color_continuous_scale="Purples",
            )
            fig_bar.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(30,41,59,0.3)",
                height=300,
                margin=dict(l=20, r=20, t=30, b=20),
            )
            st.plotly_chart(fig_bar, use_container_width=True)

        # Corpus Preview
        with st.expander("👀 Raw Text Corpus Preview (First 2,000 characters)"):
            st.text(dataset_content[:2000] + ("..." if len(dataset_content) > 2000 else ""))

        # Validation check
        is_valid, validation_msg = validate_text(dataset_content)
        if not is_valid:
            st.warning(f"⚠️ {validation_msg}")
        else:
            st.success("✅ Dataset meets all validation requirements for LSTM training.")
    else:
        st.info("No dataset currently loaded. Please upload a .txt file or select the bundled sample.")


# ==============================================================================
# SECTION D: MODEL TRAINING
# ==============================================================================
elif selected_nav == "Model Training":
    st.markdown("## 🚀 Train Word-Level LSTM Neural Network")
    st.caption("Configure architectural hyperparameters and train the autoregressive language model on your dataset.")

    if not st.session_state.dataset_text:
        st.warning("⚠️ No dataset loaded. Please go to **Dataset Management** first.")
    else:
        # Hyperparameter configuration
        st.markdown("### 1. Training Hyperparameters")
        c1, c2, c3 = st.columns(3)
        with c1:
            seq_len = st.slider("Sequence Length (Context Window)", min_value=4, max_value=20, value=config.DEFAULT_SEQUENCE_LENGTH, step=1,
                                help="Number of preceding words used to predict the next word.")
            emb_dim = st.slider("Embedding Dimension", min_value=16, max_value=128, value=config.DEFAULT_EMBEDDING_DIM, step=16,
                                help="Dimensionality of dense word vector representations.")
            lstm_units = st.slider("LSTM Hidden Units", min_value=32, max_value=256, value=config.DEFAULT_LSTM_UNITS, step=32,
                                   help="Number of memory units inside the recurrent LSTM cell.")

        with c2:
            epochs = st.slider("Training Epochs", min_value=5, max_value=50, value=config.DEFAULT_EPOCHS, step=5,
                               help="Total passes over the training sequences.")
            batch_size = st.selectbox("Batch Size", options=[16, 32, 64], index=1,
                                      help="Number of training samples per gradient update.")
            learning_rate = st.selectbox("Learning Rate", options=[0.001, 0.005, 0.01], index=1)

        with c3:
            val_split = st.slider("Validation Split", min_value=0.0, max_value=0.3, value=config.DEFAULT_VALIDATION_SPLIT, step=0.05,
                                  help="Fraction of sequences reserved for validation.")
            early_stopping = st.checkbox("Enable Early Stopping", value=True,
                                         help="Halts training if validation loss stops improving.")
            patience = st.slider("Early Stopping Patience", min_value=2, max_value=10, value=config.DEFAULT_EARLY_STOPPING_PATIENCE, step=1)

        st.markdown("---")

        # Overwrite Warning if existing model exists
        if model_loaded:
            st.info("ℹ️ A trained model is already loaded. Training a new model will overwrite the current active weights.")

        # Train execution
        if st.button("🚀 Start Training Pipeline", type="primary", use_container_width=True):
            status_container = st.empty()
            progress_bar = st.progress(0.0)
            metrics_display = st.empty()

            def progress_callback(epoch: int, total_epochs: int, logs: dict):
                pct = epoch / total_epochs
                progress_bar.progress(pct)
                status_container.info(f"Training Epoch {epoch}/{total_epochs}...")
                metrics_display.write(
                    f"**Epoch {epoch} Logs:** Loss: `{logs.get('loss', 0.0):.4f}` | "
                    f"Accuracy: `{logs.get('accuracy', 0.0) * 100:.2f}%` | "
                    f"Val Loss: `{logs.get('val_loss', 0.0):.4f}` | "
                    f"Val Accuracy: `{logs.get('val_accuracy', 0.0) * 100:.2f}%`"
                )

            try:
                with st.spinner("Preprocessing text, compiling model, and training..."):
                    result = train_model(
                        text=st.session_state.dataset_text,
                        sequence_length=seq_len,
                        embedding_dim=emb_dim,
                        lstm_units=lstm_units,
                        dropout_rate=config.DEFAULT_DROPOUT,
                        learning_rate=learning_rate,
                        batch_size=batch_size,
                        epochs=epochs,
                        validation_split=val_split,
                        early_stopping=early_stopping,
                        early_stopping_patience=patience,
                        dataset_name=st.session_state.dataset_name,
                        save_to_disk=True,
                        on_epoch_end=progress_callback,
                    )

                progress_bar.progress(1.0)
                status_container.success("🎉 Model training successfully completed and artifacts saved to disk!")
                # Invalidate cache so new model is picked up
                st.cache_resource.clear()

                # Display summary
                m = result["metadata"]
                st.markdown("### Training Summary")
                res_col1, res_col2, res_col3, res_col4 = st.columns(4)
                res_col1.metric("Final Loss", f"{m['final_loss']:.4f}")
                res_col2.metric("Final Accuracy", f"{m['final_accuracy'] * 100:.2f}%")
                res_col3.metric("Val Accuracy", f"{m['final_val_accuracy'] * 100:.2f}%" if m['final_val_accuracy'] is not None else "N/A")
                res_col4.metric("Perplexity", f"{m['perplexity']:.2f}" if m['perplexity'] is not None else "N/A")

                # Show Training Curve immediately
                if "history" in result:
                    fig = create_training_history_plots(result["history"])
                    st.plotly_chart(fig, use_container_width=True)

            except Exception as e:
                st.error(f"Training failed: {e}")


# ==============================================================================
# SECTION E: EVALUATION AND ANALYTICS
# ==============================================================================
elif selected_nav == "Evaluation & Analytics":
    st.markdown("## 📊 Model Evaluation & Analytics")
    st.caption("Inspect training history, loss convergence, accuracy trajectories, perplexity, and temperature sampling dynamics.")

    if not model_loaded or not metadata:
        st.warning("⚠️ No trained model or evaluation history found. Please train a model first.")
    else:
        # History plots
        if "history" in metadata and metadata["history"]:
            st.markdown("### Loss & Accuracy Trajectories")
            fig = create_training_history_plots(metadata["history"])
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No epoch-by-epoch training history recorded for this model.")

        # Perplexity & Metrics Explanation
        st.markdown("### Language Model Performance Metrics")
        explanations = get_metrics_explanation()

        col_ppl, col_acc = st.columns(2)
        with col_ppl:
            ppl_val = metadata.get("perplexity", "N/A")
            st.metric("Model Perplexity (PPL)", f"{ppl_val}" if ppl_val != "N/A" else "N/A")
            st.info(f"**What is Perplexity?**\n\n{explanations['perplexity']}")

        with col_acc:
            acc_val = metadata.get("final_accuracy", "N/A")
            st.metric("Token Accuracy", f"{acc_val * 100:.2f}%" if acc_val != "N/A" else "N/A")
            st.info(f"**Understanding Accuracy in NLP:**\n\n{explanations['accuracy']}")

        st.markdown("---")

        # Multi-Temperature Qualitative Comparison
        st.markdown("### 🌡️ Temperature Sampling Comparison")
        st.write("Observe how temperature scales token logits to balance predictability vs randomness on the exact same prompt:")

        comp_prompt = st.text_input("Comparison Prompt:", value="The Time Traveller was", key="comp_prompt")
        comp_seq_len = metadata.get("sequence_length", config.DEFAULT_SEQUENCE_LENGTH)

        if st.button("Generate Temperature Comparison", type="secondary"):
            with st.spinner("Generating continuations across temperatures..."):
                comp_results = compare_temperatures(
                    model=model,
                    tokenizer=tokenizer,
                    seed_text=comp_prompt,
                    sequence_length=comp_seq_len,
                    temperatures=[0.2, 0.7, 1.3],
                    num_words=20,
                    random_seed=42,
                )

                t1, t2, t3 = st.columns(3)
                with t1:
                    st.markdown("**Low Temperature (T = 0.2)**")
                    st.caption("Conservative, repetitive, favors top token")
                    st.markdown(f'<div class="output-box" style="font-size:0.95rem;">{comp_prompt} <b>{comp_results[0.2]}</b></div>', unsafe_allow_html=True)
                with t2:
                    st.markdown("**Moderate Temperature (T = 0.7)**")
                    st.caption("Balanced, natural transitions")
                    st.markdown(f'<div class="output-box" style="font-size:0.95rem;">{comp_prompt} <b>{comp_results[0.7]}</b></div>', unsafe_allow_html=True)
                with t3:
                    st.markdown("**High Temperature (T = 1.3)**")
                    st.caption("Creative, diverse, higher entropy")
                    st.markdown(f'<div class="output-box" style="font-size:0.95rem;">{comp_prompt} <b>{comp_results[1.3]}</b></div>', unsafe_allow_html=True)


# ==============================================================================
# SECTION F: MODEL ARCHITECTURE & THEORY
# ==============================================================================
elif selected_nav == "Model Architecture & Theory":
    st.markdown("## ℹ️ Model Architecture & NLP Theory")
    st.caption("Comprehensive breakdown of the word-level LSTM pipeline, vector embeddings, and autoregressive inference.")

    # Model specifications table
    if model_loaded and metadata:
        st.markdown("### Active Model Specifications")
        spec_data = {
            "Hyperparameter": [
                "Model Architecture",
                "Total Parameters",
                "Trainable Parameters",
                "Vocabulary Size",
                "Context Window (Sequence Length)",
                "Embedding Dimension",
                "LSTM Memory Units",
                "Dataset Source",
                "Last Trained Timestamp",
            ],
            "Value": [
                "Embedding → LSTM → Dropout → Dense(Softmax)",
                f"{metadata.get('total_parameters', 0):,}",
                f"{metadata.get('trainable_parameters', 0):,}",
                f"{metadata.get('vocab_size', 'N/A')} tokens",
                f"{metadata.get('sequence_length', 'N/A')} words",
                f"{metadata.get('embedding_dim', 'N/A')} dimensions",
                f"{metadata.get('lstm_units', 'N/A')} units",
                str(metadata.get("dataset_name", "N/A")),
                str(metadata.get("timestamp", "N/A")),
            ],
        }
        st.table(pd.DataFrame(spec_data))

    st.markdown("---")

    # Deep Learning Explanations
    st.markdown("### Architectural Deep Dive")

    col_t1, col_t2, col_t3 = st.columns(3)
    with col_t1:
        st.markdown("#### 1. Word Embeddings")
        st.write(
            r"""
            Unlike one-hot vectors that treat all words as orthogonal and equidistant, continuous dense word embeddings 
            project each word $w \in V$ into a dense vector space $\mathbb{R}^d$ (where $d \in [16, 128]$). 
            
            During backpropagation, semantically similar words align close to each other in vector space via cosine similarity.
            """
        )

    with col_t2:
        st.markdown("#### 2. LSTM Recurrent Memory")
        st.write(
            r"""
            Standard RNNs suffer from vanishing gradients across long contexts. The Long Short-Term Memory (LSTM) cell 
            solves this using three gating mechanisms:
            - **Forget Gate ($f_t$):** Discards obsolete information.
            - **Input Gate ($i_t$):** Updates the internal cell state $C_t$.
            - **Output Gate ($o_t$):** Filters hidden state $h_t$ for prediction.
            """
        )

    with col_t3:
        st.markdown("#### 3. Autoregressive Loop")
        st.write(
            r"""
            At each prediction step $t$, the softmax layer outputs a probability distribution over the entire vocabulary:
            $$P(w_{t+1} \mid w_1, ..., w_t) = \text{softmax}(W h_t + b)$$
            
            The sampled token is appended to the context window, the oldest token is pruned, and the updated sequence is fed 
            back into the model for step $t+2$.
            """
        )
