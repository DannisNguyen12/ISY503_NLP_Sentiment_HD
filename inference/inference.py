"""Inference-only runtime for the sentiment analysis app.

This module contains only the code required to load the saved model,
preprocess input text, run prediction, and expose Flask routes.
It intentionally excludes all training, dataset loading, and evaluation code.
"""

from __future__ import annotations

import html
import json
import os
import pickle
import re
from functools import lru_cache
from pathlib import Path

import torch
import torch.nn as nn
from flask import Flask, jsonify, render_template, request


# =============================================================================
# PATHS / CONFIGURATION
# =============================================================================

BASE_DIR = Path(__file__).resolve().parent.parent
DEFAULT_MODEL_DIR = BASE_DIR / "models"

CONFIG = {
	"model_dir": str(DEFAULT_MODEL_DIR),
	"max_len": 200,
	"embedding_dim": 100,
	"hidden_dim": 128,
	"num_layers": 2,
	"dropout": 0.5,
	"seed": 42,
}

torch.manual_seed(CONFIG["seed"])
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# =============================================================================
# TEXT PREPROCESSING
# =============================================================================

class TextPreprocessor:
	"""Lightweight text cleaner for inference.

	Keeps the same broad cleaning behavior used during training without
	bringing in training-time dependencies such as pandas, sklearn, or NLTK.
	"""

	def __init__(self):
		self.html_pattern = re.compile(r"<[^>]+>")
		self.url_pattern = re.compile(r"http[s]?://\S+|www\.\S+")
		self.punct_pattern = re.compile(r"[^a-zA-Z\s]")
		self.repeat_pattern = re.compile(r"(.)\1{2,}")
		self.whitespace_pattern = re.compile(r"\s+")

	def clean(self, text: str) -> str:
		if not isinstance(text, str):
			return ""

		text = html.unescape(text).lower()
		text = self.html_pattern.sub(" ", text)
		text = self.url_pattern.sub(" ", text)
		text = self.punct_pattern.sub(" ", text)
		text = self.repeat_pattern.sub(r"\1\1", text)
		text = self.whitespace_pattern.sub(" ", text).strip()
		return text


# =============================================================================
# VOCABULARY
# =============================================================================

class Vocabulary:
	def __init__(self):
		self.word2idx = {"<PAD>": 0, "<UNK>": 1}
		self.idx2word = {0: "<PAD>", 1: "<UNK>"}
		self.vocab_size = 2

	@classmethod
	def load(cls, path: str | os.PathLike[str]):
		with open(path, "rb") as f:
			data = pickle.load(f)

		vocab = cls()
		vocab.word2idx = data["word2idx"]
		vocab.idx2word = data["idx2word"]
		vocab.vocab_size = len(vocab.word2idx)
		return vocab

	def encode(self, text: str, max_len: int):
		tokens = text.split()
		indices = [self.word2idx.get(token, 1) for token in tokens]
		if len(indices) < max_len:
			indices.extend([0] * (max_len - len(indices)))
		else:
			indices = indices[:max_len]
		return indices


# =============================================================================
# MODEL
# =============================================================================

class AttentionLayer(nn.Module):
	def __init__(self, hidden_dim: int):
		super().__init__()
		self.attention = nn.Linear(hidden_dim * 2, hidden_dim * 2)
		self.context_vector = nn.Linear(hidden_dim * 2, 1, bias=False)

	def forward(self, lstm_output, mask=None):
		attn_scores = torch.tanh(self.attention(lstm_output))
		attn_scores = self.context_vector(attn_scores).squeeze(-1)

		if mask is not None:
			attn_scores = attn_scores.masked_fill(mask == 0, -1e9)

		attn_weights = torch.softmax(attn_scores, dim=1)
		context = torch.bmm(attn_weights.unsqueeze(1), lstm_output).squeeze(1)
		return context, attn_weights


class SentimentBiLSTM(nn.Module):
	def __init__(
		self,
		vocab_size,
		embedding_dim,
		hidden_dim,
		num_layers,
		dropout,
		pad_idx=0,
		pretrained_embeddings=None,
	):
		super().__init__()
		self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)

		if pretrained_embeddings is not None:
			self.embedding.weight.data.copy_(torch.from_numpy(pretrained_embeddings))
			self.embedding.weight.requires_grad = True

		self.lstm = nn.LSTM(
			embedding_dim,
			hidden_dim,
			num_layers=num_layers,
			bidirectional=True,
			dropout=dropout if num_layers > 1 else 0,
			batch_first=True,
		)
		self.attention = AttentionLayer(hidden_dim)
		self.batch_norm = nn.BatchNorm1d(hidden_dim * 2)
		self.dropout = nn.Dropout(dropout)
		self.fc1 = nn.Linear(hidden_dim * 2, hidden_dim)
		self.fc2 = nn.Linear(hidden_dim, 1)
		self.relu = nn.ReLU()

	def forward(self, text, lengths=None):
		mask = (text != 0).float()
		embedded = self.dropout(self.embedding(text))

		if lengths is not None:
			packed = nn.utils.rnn.pack_padded_sequence(
				embedded, lengths.cpu(), batch_first=True, enforce_sorted=False
			)
			packed_output, _ = self.lstm(packed)
			lstm_output, _ = nn.utils.rnn.pad_packed_sequence(
				packed_output, batch_first=True, total_length=text.size(1)
			)
		else:
			lstm_output, _ = self.lstm(embedded)

		context, attention_weights = self.attention(lstm_output, mask)
		context = self.batch_norm(context)
		context = self.dropout(context)

		hidden = self.relu(self.fc1(context))
		hidden = self.dropout(hidden)
		output = self.fc2(hidden)
		return output, attention_weights


# =============================================================================
# MODEL LOADING / PREDICTION
# =============================================================================

def _resolve_config(model_dir: Path, checkpoint: dict) -> dict:
	saved_config = checkpoint.get("config")
	if saved_config is not None:
		return saved_config

	config_path = model_dir / "config.json"
	if config_path.exists():
		with open(config_path, "r") as f:
			return json.load(f)

	return CONFIG


@lru_cache(maxsize=1)
def load_model_bundle():
	model_dir = Path(CONFIG["model_dir"])
	model_path = model_dir / "best_model.pt"
	vocab_path = model_dir / "vocabulary.pkl"

	if not model_path.exists():
		raise FileNotFoundError(f"Required model file not found: {model_path}")
	if not vocab_path.exists():
		raise FileNotFoundError(f"Required vocabulary file not found: {vocab_path}")

	vocab = Vocabulary.load(vocab_path)
	checkpoint = torch.load(model_path, map_location=device)
	saved_config = _resolve_config(model_dir, checkpoint)

	model = SentimentBiLSTM(
		vocab_size=vocab.vocab_size,
		embedding_dim=saved_config["embedding_dim"],
		hidden_dim=saved_config["hidden_dim"],
		num_layers=saved_config["num_layers"],
		dropout=saved_config["dropout"],
	).to(device)

	model.load_state_dict(checkpoint["model_state_dict"])
	model.eval()

	print(f"[INFO] Model loaded from {model_dir}")
	return {
		"model": model,
		"vocab": vocab,
		"config": saved_config,
		"preprocessor": TextPreprocessor(),
	}


def _normalize_attention(weights):
	if not weights:
		return []

	min_w = min(weights)
	max_w = max(weights)
	if max_w <= min_w:
		return [0.0 for _ in weights]
	return [(w - min_w) / (max_w - min_w) for w in weights]


def predict_sentiment(text: str):
	bundle = load_model_bundle()
	model = bundle["model"]
	vocab = bundle["vocab"]
	preprocessor = bundle["preprocessor"]
	cfg = bundle["config"]

	cleaned_text = preprocessor.clean(text)
	encoded = vocab.encode(cleaned_text, cfg["max_len"])
	token_count = min(len(cleaned_text.split()), cfg["max_len"])

	if token_count == 0:
		raise ValueError("Input becomes empty after preprocessing. Please enter a review with meaningful words.")

	tensor = torch.tensor([encoded], dtype=torch.long, device=device)
	length = torch.tensor([token_count], dtype=torch.long, device=device)

	with torch.inference_mode():
		output, attention_weights = model(tensor, length)
		prob = torch.sigmoid(output).squeeze().item()

	sentiment = "Positive review" if prob > 0.5 else "Negative review"
	sentiment_class = "positive" if prob > 0.5 else "negative"
	confidence = prob if prob > 0.5 else 1 - prob

	attention = attention_weights.squeeze(0).detach().cpu().tolist()[:token_count]
	tokens = cleaned_text.split()[:token_count]
	attention_norm = _normalize_attention(attention)

	attention_data = [
		{"token": token, "weight": float(weight), "normalized": float(norm)}
		for token, weight, norm in zip(tokens, attention, attention_norm)
	]

	return {
		"success": True,
		"sentiment": sentiment,
		"sentiment_class": sentiment_class,
		"confidence": float(confidence),
		"cleaned_text": cleaned_text,
		"attention_data": attention_data,
		"input_length": len(text),
		"token_count": len(tokens),
	}


# =============================================================================
# FLASK APP
# =============================================================================

app = Flask(__name__, template_folder=str(BASE_DIR / "templates"), static_folder=str(BASE_DIR / "static"))
app.config["SECRET_KEY"] = "isy503-sentiment-analysis-2026"


@app.route("/")
def index():
	return render_template("index.html")


@app.route("/analyze", methods=["POST"])
def analyze():
	try:
		data = request.get_json(silent=True) or {}
		text = data.get("text", "").strip()

		if not text:
			return jsonify({"success": False, "error": "No text provided. Please enter a review statement."}), 400
		if len(text) > 2000:
			return jsonify({"success": False, "error": "Input too long. Maximum 2000 characters allowed."}), 400

		return jsonify(predict_sentiment(text))

	except ValueError as e:
		return jsonify({"success": False, "error": str(e)}), 400

	except Exception as e:
		return jsonify({"success": False, "error": f"Analysis failed: {str(e)}"}), 500


@app.route("/health")
def health():
	return jsonify(
		{
			"status": "healthy",
			"model_loaded": load_model_bundle.cache_info().currsize > 0,
			"device": str(device),
		}
	)


@app.errorhandler(404)
def not_found(error):
	return jsonify({"success": False, "error": "Endpoint not found"}), 404


@app.errorhandler(500)
def internal_error(error):
	return jsonify({"success": False, "error": "Internal server error"}), 500


__all__ = [
	"app",
	"device",
	"CONFIG",
	"TextPreprocessor",
	"Vocabulary",
	"AttentionLayer",
	"SentimentBiLSTM",
	"load_model_bundle",
	"predict_sentiment",
]
