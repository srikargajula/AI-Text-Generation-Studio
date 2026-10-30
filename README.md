# AI Text Generation Studio Using Deep Learning

An interactive, end-to-end Natural Language Processing (NLP) and Deep Learning web application for word-level language modeling, autoregressive text generation, interactive model training, and performance analytics.

Built from scratch using **Python 3.11+**, **TensorFlow/Keras**, and **Streamlit**. Designed for CPU compatibility, AI/ML portfolios, college final-year capstone presentations, and technical interviews.

---

## 🌟 Key Features

1. **Autoregressive Text Generation**
   - True word-level next-token prediction using an LSTM recurrent neural network.
   - **Temperature Scaling**: Adjust randomness from deterministic greedy decoding ($T \le 0.1$) to high-entropy creative exploration ($T \ge 1.2$).
   - **Top-K Nucleus Sampling**: Restricts token selection to the most probable $K$ candidates to prevent low-probability gibberish.
   - **Robust Out-of-Vocabulary (OOV) Handling**: Handles unseen words and empty prompts gracefully without crashing.
   - Step-by-step token generation diagnostics showing predicted tokens and probability scores.

2. **Dataset Management & Corpus Analytics**
   - Upload any plain text file (`.txt`) up to 10 MB or explore the bundled public-domain corpus (*The Time Machine* by H.G. Wells).
   - Live lexical statistics: Total word count, unique vocabulary, lexical diversity ratio, line count, character count, and top-word frequency distributions.
   - Rigorous data validation: Encoding verification, minimum corpus size check, and security isolation (never evaluates uploaded text as executable code).

3. **Interactive Model Training Panel**
   - Fully customizable hyperparameters: Context sequence length, embedding dimensions, LSTM recurrent units, batch size, epochs, learning rate, and validation split.
   - Integrated early stopping and custom live callbacks with real-time progress bars and epoch metrics.
   - Safe model persistence with warning confirmation prior to overwriting existing weights.

4. **Model Evaluation & Visual Analytics**
   - Interactive Plotly curves for training and validation loss (Sparse Categorical Cross-Entropy) and token-level accuracy.
   - **Perplexity Computation**: Mathematically valid exponential loss metric ($PPL = \exp(\mathcal{L})$).
   - Multi-temperature comparative inference: Side-by-side generation at $T = 0.2$, $0.7$, and $1.3$.
   - Educational disclaimers explaining the scientific limitations of token accuracy and perplexity in language modeling.

5. **Architecture & Educational Deep Dive**
   - Detailed model inspection (layer breakdown, parameter counts, tensor dimensions).
   - Clear explanations of continuous word embeddings, LSTM memory gating mechanisms, and the autoregressive inference cycle.

---

## 🏗️ Project Architecture & Structure

```
text-generation-studio/
├── app.py                      # Main Streamlit web application
├── config.py                   # Centralized configuration, hyperparameter defaults, and paths
├── requirements.txt            # Python dependencies
├── README.md                   # Comprehensive documentation and guide
├── .gitignore                  # Git exclusions for checkpoints, venvs, and cache
├── dataset/
│   └── sample.txt              # Bundled public-domain training corpus
├── models/
│   ├── .gitkeep                # Directory for saved models and tokenizers
│   ├── lstm_text_gen.keras     # Trained Keras model weights
│   ├── tokenizer.json          # Keras Tokenizer vocabulary and configuration
│   └── metadata.json           # Model metadata, specs, and training history
├── src/
│   ├── __init__.py
│   ├── preprocessing.py        # Text normalization, validation, tokenization, sequence generation
│   ├── model.py                # Keras Sequential LSTM architecture & parameter inspection
│   ├── training.py             # Complete training pipeline, callbacks, and evaluation
│   ├── generation.py           # Autoregressive generation with temperature & top-k sampling
│   └── evaluation.py           # Perplexity calculations and Plotly visualization charts
├── utils/
│   ├── __init__.py
│   └── storage.py              # Serialization and deserialization of models & tokenizers
├── scripts/
│   └── train_initial_model.py  # Standalone CLI script to train starter model
└── tests/
    ├── test_preprocessing.py   # Unit tests for text cleaning, tokenization, and sliding windows
    ├── test_model.py           # Unit tests for model compilation, shapes, and parameters
    └── test_generation.py      # Unit tests for temperature scaling, top-k, and generation
```

---

## 🛠️ Technology Stack

| Component | Technology | Purpose |
| :--- | :--- | :--- |
| **Language** | Python 3.11 / 3.12 / 3.13 | Core programming language |
| **Deep Learning** | TensorFlow 2.16+ / Keras 3 | Neural network design, word embeddings, LSTM layers |
| **Web Interface** | Streamlit | Responsive dashboard and interactive controls |
| **NLP** | Keras Tokenizer | Vocabulary indexing, token mapping, sequence padding |
| **Data Processing** | NumPy, Pandas | Sequence tensors and lexical frequency tables |
| **Visualization** | Plotly | Dynamic dark-themed loss, accuracy, and bar charts |
| **Storage** | Keras (`.keras`) & JSON | Model architecture, weights, and tokenizer persistence |
| **Testing** | pytest | Automated unit and integration tests |

---

## 🚀 Quickstart Guide

### 1. Prerequisites
Ensure **Python 3.11** or a compatible modern Python version (such as 3.12 or 3.13) is installed on your system.

### 2. Create and Activate Virtual Environment

**Windows (PowerShell):**
```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

**macOS / Linux:**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Optional: Pre-train Starter Model
To train the initial baseline model on the bundled dataset before launching the UI:
```bash
python scripts/train_initial_model.py
```

### 5. Launch the Streamlit Web Application
```bash
streamlit run app.py
```
The application will open automatically in your browser at `http://localhost:8501`.

---

## 🧪 Running Automated Tests

Run the full automated test suite using `pytest`:
```bash
pytest tests/ -v
```

The test suite validates:
- Text cleaning, normalization, and quote handling.
- Dataset validation and lexical statistics.
- Sliding window sequence creation and edge cases.
- Model construction, layer configurations, and tensor output shapes.
- Autoregressive generation with temperature scaling and top-k filtering.
- Out-of-vocabulary (OOV) prompts and empty input resilience.

---

## 📖 Application User Guide

### 1. Home Dashboard
- View system status (whether an active model is loaded in memory).
- Review vocabulary size, trainable parameters, and perplexity.
- Click any of the **Example Seed Prompts** to automatically transfer into the generation interface.

### 2. Text Generation
- Input a starting prompt (e.g., *"The Time Traveller was expounding"*).
- Adjust **Output Length** (words to synthesize) and **Temperature** (0.1 to 1.8).
- Adjust **Top-K** candidate cutoff (filters out low-probability tails).
- Click **Generate Text** to observe the neural network generating tokens autoregressively.
- Expand **Step-by-Step Prediction Diagnostics** to inspect individual token probabilities.

### 3. Dataset Management
- Upload your own `.txt` file (e.g., stories, essays, code snippets, notes).
- Or click **Reload Bundled Time Machine Sample** to use the pre-packaged corpus.
- View lexical diversity and frequency plots of the most common words.

### 4. Model Training
- Configure sequence context length (e.g., 6 to 12 words), embedding dimensions (e.g., 32 to 64), and LSTM units (e.g., 64 to 128).
- Click **Start Training Pipeline** to initiate CPU training.
- Live epoch progress, training loss, and accuracy metrics are rendered in real time.
- The model and tokenizer are automatically compiled, saved to the `models/` directory, and reloaded.

### 5. Evaluation & Analytics
- Inspect loss curves and accuracy trajectories across epochs.
- Analyze model perplexity ($PPL = \exp(\mathcal{L})$).
- Compare sample generations side-by-side across three temperatures: Low ($T=0.2$), Medium ($T=0.7$), and High ($T=1.3$).

---

## 🔬 Mathematical & Architectural Details

### 1. Tokenization & Sliding Window
Given a tokenized text sequence $S = [w_1, w_2, \dots, w_N]$, the training dataset is formed by sliding a window of length $L$:
$$X_i = [w_i, w_{i+1}, \dots, w_{i+L-1}], \quad y_i = w_{i+L}$$
This yields $N - L$ training pairs without requiring pre-computed n-gram tables.

### 2. Network Flow
$$\mathbf{e}_t = \text{Embedding}(X_t) \in \mathbb{R}^{L \times d}$$
$$\mathbf{h} = \text{LSTM}(\mathbf{e}_t) \in \mathbb{R}^u$$
$$\mathbf{z} = \mathbf{W}_o \cdot \text{Dropout}(\mathbf{h}) + \mathbf{b}_o \in \mathbb{R}^{|V|}$$
$$\mathbf{p} = \text{Softmax}(\mathbf{z})$$

Where $|V|$ is the vocabulary size, $d$ is embedding dimension, and $u$ is the number of LSTM hidden units.

### 3. Temperature Scaling
During generation, logits $z_i$ are rescaled by temperature $T$:
$$p_i = \frac{\exp(z_i / T)}{\sum_{j} \exp(z_j / T)}$$
- As $T \to 0$, distribution approaches a Dirac delta centered at $\text{argmax}(z)$ (greedy argmax).
- As $T \to \infty$, distribution approaches uniform distribution $\mathcal{U}(1, |V|)$.

### 4. Perplexity
$$PPL = \exp\left( - \frac{1}{M} \sum_{i=1}^M \log P(y_i \mid X_i) \right)$$
Measures average branching factor. A perplexity of 15 means the model is, on average, as uncertain as if choosing uniformly among 15 words.

---

## ⚠️ Limitations & Future Improvements

- **Corpus Scale & Capacity**: Small LSTM models trained on modest CPU corpora learn local statistical patterns (verb agreements, common idioms), but lack world knowledge or deep long-range factual reasoning.
- **Subword Tokenization**: Moving to Byte-Pair Encoding (BPE) or WordPiece would eliminate out-of-vocabulary tokens entirely.
- **Transformer Architectures**: Future iterations can implement causal self-attention mechanisms (e.g., mini-GPT) to expand context retention beyond recurrent horizon limits.

---

## 📜 License & Acknowledgments
- The bundled sample text is an excerpt from *The Time Machine* by H.G. Wells (1895), which is in the public domain.
- Developed as a clean, complete AI/ML reference implementation.
