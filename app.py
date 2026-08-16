
import os
import json
import pickle
import sys
import torch
import numpy as np
from flask import Flask, render_template, request, jsonify

# Import model components from training script
from train_model import (
    SentimentBiLSTM, TextPreprocessor, Vocabulary, 
    CONFIG, device, predict_sentiment
)

# =============================================================================
# FLASK APPLICATION SETUP
# =============================================================================

app = Flask(__name__)
app.config['SECRET_KEY'] = 'isy503-sentiment-analysis-2026'

# =============================================================================
# MODEL LOADING
# =============================================================================

def load_model():
    model_dir = CONFIG['model_dir']

    # Verify required files exist
    required_files = ['best_model.pt', 'vocabulary.pkl']
    for fname in required_files:
        fpath = os.path.join(model_dir, fname)
        if not os.path.exists(fpath):
            raise FileNotFoundError(f"Required file not found: {fpath}")

    # Load vocabulary
    vocab = Vocabulary.load(os.path.join(model_dir, 'vocabulary.pkl'))

    # Load trained weights first (checkpoint may contain full config)
    checkpoint = torch.load(os.path.join(model_dir, 'best_model.pt'), map_location=device)

    # Resolve configuration with robust fallback order:
    # 1) config inside checkpoint, 2) models/config.json, 3) imported CONFIG
    saved_config = checkpoint.get('config')
    if saved_config is None:
        config_path = os.path.join(model_dir, 'config.json')
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                saved_config = json.load(f)
        else:
            saved_config = CONFIG

    # Initialize model architecture
    model = SentimentBiLSTM(
        vocab_size=vocab.vocab_size,
        embedding_dim=saved_config['embedding_dim'],
        hidden_dim=saved_config['hidden_dim'],
        num_layers=saved_config['num_layers'],
        dropout=saved_config['dropout']
    ).to(device)

    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    print(f"[INFO] Model loaded successfully from {model_dir}")
    print(f"[INFO] Validation accuracy at checkpoint: {checkpoint.get('val_acc', 'N/A')}")

    return model, vocab

# Global model and vocabulary (loaded once at startup)
MODEL = None
VOCAB = None
PREPROCESSOR = None

@app.before_request
def initialize():
    """Lazy initialization of model on first request."""
    global MODEL, VOCAB, PREPROCESSOR
    if MODEL is None:
        MODEL, VOCAB = load_model()
        PREPROCESSOR = TextPreprocessor()

# =============================================================================
# ROUTES
# =============================================================================

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    try:
        data = request.get_json()

        if not data or 'text' not in data:
            return jsonify({
                'success': False,
                'error': 'No text provided. Please enter a review statement.'
            }), 400

        text = data['text'].strip()

        # Validation
        if len(text) == 0:
            return jsonify({
                'success': False,
                'error': 'Empty input. Please enter a valid review statement.'
            }), 400

        if len(text) > 2000:
            return jsonify({
                'success': False,
                'error': 'Input too long. Maximum 2000 characters allowed.'
            }), 400

        # Perform prediction
        sentiment, confidence, attention_weights = predict_sentiment(
            MODEL, VOCAB, text, PREPROCESSOR, device
        )

        # Get cleaned text and tokens for attention visualization
        cleaned_text = PREPROCESSOR.clean(text)
        tokens = cleaned_text.split()[:CONFIG['max_len']]

        # Prepare attention data for visualization
        # Normalize attention weights to [0, 1] for color mapping
        attn = attention_weights[:len(tokens)]
        if len(attn) > 0:
            attn_min, attn_max = attn.min(), attn.max()
            if attn_max > attn_min:
                attn_norm = (attn - attn_min) / (attn_max - attn_min)
            else:
                attn_norm = np.zeros_like(attn)
        else:
            attn_norm = np.array([])

        attention_data = [
            {'token': token, 'weight': float(weight), 'normalized': float(norm)}
            for token, weight, norm in zip(tokens, attn, attn_norm)
        ]

        # Determine sentiment class for styling
        sentiment_class = 'positive' if sentiment == 'Positive review' else 'negative'

        return jsonify({
            'success': True,
            'sentiment': sentiment,
            'sentiment_class': sentiment_class,
            'confidence': float(confidence),
            'cleaned_text': cleaned_text,
            'attention_data': attention_data,
            'input_length': len(text),
            'token_count': len(tokens)
        })

    except Exception as e:
        return jsonify({
            'success': False,
            'error': f'Analysis failed: {str(e)}'
        }), 500

@app.route('/health')
def health():
    """Health check endpoint for monitoring."""
    return jsonify({
        'status': 'healthy',
        'model_loaded': MODEL is not None,
        'device': str(device)
    })

# =============================================================================
# ERROR HANDLERS
# =============================================================================

@app.errorhandler(404)
def not_found(error):
    return jsonify({'success': False, 'error': 'Endpoint not found'}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({'success': False, 'error': 'Internal server error'}), 500

# =============================================================================
# MAIN
# =============================================================================

if __name__ == '__main__':
    # Ensure model is trained before starting server
    if not os.path.exists(os.path.join(CONFIG['model_dir'], 'best_model.pt')):
        print("[WARNING] Model not found. Please run train_model.py first.")
        print("[INFO] Training model now...")
        import subprocess
        subprocess.run([sys.executable, 'train_model.py'], check=True)

    print("\n" + "="*60)
    print("ISY503 SENTIMENT ANALYSIS WEB SERVER")
    print("="*60)
    print("Starting Flask server...")
    print("Open your browser and navigate to: http://127.0.0.1:5001")
    print("="*60 + "\n")

    app.run(host='0.0.0.0', port=5001, debug=True)
