# ISY503 Intelligent Systems — NLP Sentiment Analysis

> **Assessment 3 | Torrens University Australia**  
> **BiLSTM + Self-Attention | Inference-only Web App for Vercel**

---

## 📋 Project Overview

This project is a sentiment analysis system for Amazon product reviews built with a **BiLSTM + Self-Attention** model. The training pipeline is kept separate from the deployment app so the Vercel version stays lightweight and only runs inference.

### What this version does
- Classifies reviews as **Positive review** or **Negative review**
- Loads a saved PyTorch model from `models/`
- Runs a Flask app for browser-based prediction and API access
- Includes attention visualization for interpretability
- Keeps training code out of the Vercel runtime to reduce deployment size

---

## 🏗️ New Application Architecture

```
User input
   ↓
Flask app (`app.py`)
   ↓
Inference module (`inference/inference.py`)
   ↓
Load `best_model.pt` + `vocabulary.pkl`
   ↓
Preprocess text
   ↓
Encode tokens
   ↓
BiLSTM + Self-Attention
   ↓
Sentiment + confidence + attention data
```

### Deployment split
- `train_model.py` — offline training only
- `inference/inference.py` — inference-only runtime for Vercel
- `app.py` — thin wrapper that exposes the Flask app
- `requirements-vercel.txt` — minimal runtime dependencies

---

## 📁 Project Structure

```
ISY503_NLP_Sentiment_HD/
├── train_model.py              # Offline training pipeline
├── app.py                      # Thin Flask entry point for deployment
├── inference/
│   ├── __init__.py
│   └── inference.py            # Inference-only app and model loading
├── requirements.txt            # Full project dependencies
├── requirements-vercel.txt     # Minimal Vercel dependencies
├── models/
│   ├── best_model.pt
│   ├── vocabulary.pkl
│   ├── config.json
│   └── history.json
├── templates/
│   └── index.html
└── static/
    └── style.css
```

---

## 🚀 Local Run

### Full project environment
```bash
pip install -r requirements.txt
python train_model.py
python app.py
```

### Inference-only environment
```bash
pip install -r requirements-vercel.txt
python app.py
```

Open your browser at:

- `http://127.0.0.1:5001`

---

## 🎯 Usage

### Web interface
1. Enter a product review in the text area
2. Click **Analyze Sentiment**
3. View the predicted label and confidence score
4. Inspect the attention visualization to see influential words

### API endpoint
```bash
curl -X POST http://127.0.0.1:5001/analyze \
  -H "Content-Type: application/json" \
  -d '{"text": "This product is absolutely fantastic!"}'
```

### Example response
```json
{
  "success": true,
  "sentiment": "Positive review",
  "sentiment_class": "positive",
  "confidence": 0.9876,
  "cleaned_text": "this product is absolutely fantastic",
  "attention_data": [],
  "input_length": 42,
  "token_count": 6
}
```

---

## 📊 Model Summary

| Metric | Value |
|--------|-------|
| Architecture | BiLSTM + Self-Attention |
| Vocabulary Size | 20,000 |
| Embedding Dimension | 100 |
| Hidden Dimension | 128 |
| LSTM Layers | 2 (bidirectional) |
| Dropout | 0.5 |
| Test Accuracy | >95% |

---

## 🔬 Original Contributions

The original project components are still preserved in the training pipeline:

1. **Custom Self-Attention Mechanism** — focuses on sentiment-bearing words.
2. **Advanced Preprocessing Pipeline** — HTML/URL removal, repeated character normalization, and lemmatization.
3. **Statistical Outlier Removal** — filters very short and very long reviews.
4. **Training Optimization Stack** — learning rate scheduling, gradient clipping, batch normalization, and early stopping.
5. **Attention Visualization** — helps explain predictions in the web app.

---

## ⚖️ Ethical Considerations

### Data bias and fairness
The dataset may reflect category or language bias, so the model should not be treated as universally fair across all review styles.

### Sarcasm and context
Sarcastic reviews can be misclassified because the model may over-focus on positive keywords without deeper intent understanding.

### Privacy
The deployment app processes user text in memory and does not store input data by default.

### Accountability
Sentiment predictions should support human judgment, especially for low-confidence or high-impact decisions.

---

## 🛠️ Technologies Used

| Category | Technology |
|----------|------------|
| Deep Learning | PyTorch |
| NLP | NLTK |
| Utilities | NumPy, Pandas, scikit-learn |
| Web Framework | Flask |
| Frontend | HTML5, CSS3, Vanilla JavaScript |

---

## 📄 License

This project is submitted as part of ISY503 Assessment 3 at Torrens University Australia.

---

*Built with 💙 for ISY503 Intelligent Systems*
