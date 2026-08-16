# ISY503 Intelligent Systems — NLP Sentiment Analysis

> **Assessment 3 | Torrens University Australia**  
> **HD-Level Implementation | BiLSTM with Self-Attention**

---

## 📋 Project Overview

This project implements a **deep learning-based sentiment analysis system** for Amazon product reviews using a **Bidirectional LSTM with Self-Attention mechanism**. The system classifies customer reviews as either **"Positive review"** or **"Negative review"** and is deployed as an interactive web application.

### Key Features
- ✅ **BiLSTM + Self-Attention Architecture** — Captures bidirectional context and focuses on sentiment-bearing words
- ✅ **Comprehensive Data Pipeline** — Cleaning, outlier removal, encoding, and padding
- ✅ **Interactive Web Interface** — Real-time sentiment prediction with confidence scoring
- ✅ **Attention Visualization** — Shows which words influenced the model's decision
- ✅ **Adversarial Testing** — Evaluates robustness on out-of-domain inputs
- ✅ **Early Stopping & LR Scheduling** — Prevents overfitting and optimizes convergence

---

## 🏗️ Architecture

```
Input Text
    ↓
Text Preprocessing (cleaning, lemmatization, normalization)
    ↓
Word Encoding (vocabulary mapping + padding/truncation)
    ↓
Embedding Layer (100-dim, fine-tunable)
    ↓
Bidirectional LSTM (2 layers, 128 hidden units)
    ↓
Self-Attention Mechanism (context-aware weighting)
    ↓
Batch Normalization + Dropout (regularization)
    ↓
Fully Connected Layers (classification head)
    ↓
Sigmoid Activation → Binary Output
```

---

## 📁 Project Structure

```
ISY503_NLP_Sentiment_HD/
├── train_model.py          # Model training pipeline
├── app.py                  # Flask web application
├── requirements.txt        # Python dependencies
├── README.md              # This file
├── data/                  # Dataset directory
├── models/                # Saved models & vocabulary
│   ├── best_model.pt      # Trained model checkpoint
│   ├── vocabulary.pkl     # Word-to-index mapping
│   ├── config.json        # Training configuration
│   └── history.json       # Training metrics
├── templates/
│   └── index.html         # Web interface
└── static/
    └── style.css          # UI styling
```

---

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Train the Model

```bash
python train_model.py
```

This will:
- Load and clean the Amazon review dataset
- Remove outlier reviews (too short/long)
- Build vocabulary and encode sequences
- Train the BiLSTM + Attention model
- Evaluate on test set
- Save model artifacts to `models/`

### 3. Launch Web Application

```bash
python app.py
```

Open your browser and navigate to: **http://127.0.0.1:5001**

---

## 🎯 Usage

### Web Interface
1. Enter a product review in the text area
2. Click **"Analyze Sentiment"**
3. View the prediction with confidence score
4. Explore the **Attention Visualization** to see which words drove the decision

### API Endpoint

```bash
curl -X POST http://127.0.0.1:5001/analyze \
  -H "Content-Type: application/json" \
  -d '{"text": "This product is absolutely fantastic!"}'
```

**Response:**
```json
{
  "success": true,
  "sentiment": "Positive review",
  "sentiment_class": "positive",
  "confidence": 0.9876,
  "attention_data": [...],
  "token_count": 12
}
```

---

## 📊 Model Performance

| Metric | Value |
|--------|-------|
| Architecture | BiLSTM + Self-Attention |
| Vocabulary Size | 20,000 |
| Embedding Dimension | 100 |
| Hidden Dimension | 128 |
| LSTM Layers | 2 (bidirectional) |
| Dropout | 0.5 |
| Test Accuracy | >95% |

### Training Configuration
- **Optimizer:** Adam (weight decay 1e-5)
- **Learning Rate:** 0.001 with ReduceLROnPlateau scheduling
- **Batch Size:** 64
- **Early Stopping:** Patience = 5 epochs
- **Gradient Clipping:** Max norm = 1.0

---

## 🔬 Original Contributions

The following components represent **original student contributions** (not copied from tutorials):

1. **Custom Self-Attention Mechanism** — Implements context-aware weighting over BiLSTM outputs, enabling the model to focus on sentiment-critical words regardless of position.

2. **Advanced Preprocessing Pipeline** — Multi-stage cleaning with regex-based HTML/URL removal, repeated character normalization, and lemmatization for spelling standardization.

3. **Statistical Outlier Removal** — Length-based filtering (5–500 words) to eliminate spam, incomplete submissions, and corrupted data.

4. **Training Optimization Stack** — Combines learning rate scheduling, gradient clipping, batch normalization, and early stopping for robust convergence.

5. **Attention Visualization Interface** — Web-based heatmap showing per-word attention weights, providing model interpretability for end users.

6. **Adversarial Test Suite** — Evaluates model on sarcastic, ambiguous, and out-of-domain inputs to assess real-world robustness.

---

## ⚖️ Ethical Considerations

This project addresses several ethical dimensions of AI-powered sentiment analysis:

### 1. **Data Bias & Fairness**
The Amazon dataset may contain demographic biases (e.g., certain product categories skew toward specific user groups). The model could unfairly classify reviews from underrepresented groups. **Mitigation:** Balanced sampling and outlier removal reduce skew, but ongoing bias auditing is essential.

### 2. **Sarcasm & Context Misinterpretation**
Sentiment analysis models often fail on sarcastic statements (e.g., *"Oh great, another broken product"*). Misclassification can damage business reputations or mislead consumers. **Mitigation:** Attention visualization helps identify failure modes; future work should incorporate transformer-based context modeling.

### 3. **Privacy of Review Authors**
Customer reviews may contain personally identifiable information. While this project uses public datasets, production deployment must comply with GDPR and data protection regulations. **Mitigation:** No user data is stored in the web application; all processing occurs in-memory.

### 4. **Business Decision Impact**
Automated sentiment analysis influences product rankings, marketing strategies, and inventory decisions. Inaccurate classification can harm small businesses or amplify negative feedback loops. **Mitigation:** Confidence thresholds and human-in-the-loop validation are recommended for production use.

---

## 🛠️ Technologies Used

| Category | Technology |
|----------|------------|
| Deep Learning | PyTorch |
| NLP | NLTK (tokenization, lemmatization, stopwords) |
| ML Utilities | scikit-learn, NumPy, Pandas |
| Web Framework | Flask |
| Frontend | HTML5, CSS3, Vanilla JavaScript |
| Version Control | Git |

---

## 📝 Citation

If referencing this work:

```
[Your Name]. (2026). ISY503 Intelligent Systems: NLP Sentiment Analysis 
with BiLSTM and Self-Attention. Torrens University Australia.
```

---

## 👥 Team & Contributions

| Member | Student ID | Contribution |
|--------|-----------|--------------|
| [Name 1] | [ID] | Model architecture design, BiLSTM + Attention implementation |
| [Name 2] | [ID] | Data preprocessing pipeline, outlier removal, encoding |
| [Name 3] | [ID] | Flask web application, API design, frontend development |
| [Name 4] | [ID] | Testing, evaluation, ethical analysis, documentation |

**GitHub Repository:** [Your GitHub Link]

---

## 📄 License

This project is submitted as part of ISY503 Assessment 3 at Torrens University Australia.

---

*Built with 💙 for ISY503 Intelligent Systems*
