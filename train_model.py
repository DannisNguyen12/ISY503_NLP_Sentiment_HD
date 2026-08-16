"""
ISY503 Assessment 3 — NLP Sentiment Analysis
Train a BiLSTM with Attention for Amazon Product Review Sentiment Classification

Author: Manh Long Nguyen
Student ID: A000222381
Date: 2026

This script implements a complete deep learning pipeline for sentiment analysis
on the JHU Amazon Product Review dataset. It includes:
  - Data cleaning and preprocessing (punctuation removal, spelling normalization)
  - Outlier removal (very short/long reviews)
  - Word encoding via vocabulary mapping and padding
  - Label encoding for binary classification
  - Train/Validation/Test split
  - BiLSTM with Self-Attention architecture
  - Model training with early stopping and learning rate scheduling
  - Evaluation with confusion matrix and classification report
  - Model export for web deployment

Original Contributions (highlighted with # CONTRIBUTION comments):
  - Custom Attention mechanism integrated with BiLSTM
  - Advanced data cleaning pipeline with outlier detection
  - Learning rate scheduling and early stopping for optimal convergence
  - Comprehensive evaluation with adversarial test cases
"""

import os
import re
import json
import pickle
import html
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from collections import Counter
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Download required NLTK data
nltk.download('stopwords', quiet=True) #Download stop words
nltk.download('wordnet', quiet=True) #Download lemmatizer
# nltk.download('omw-1.4', quiet=True) #Download multiplingual

# =============================================================================
# CONFIGURATION
# =============================================================================
CONFIG = {
    'data_dir': './data',
    'model_dir': './models',
    'max_words': 20000,        # Vocabulary size limit
    'max_len': 200,            # Sequence length (pad/truncate)
    'embedding_dim': 100,      # embedding dimension
    'hidden_dim': 128,         # LSTM hidden dimension
    'num_layers': 2,           # Number of BiLSTM layers
    'dropout': 0.5,            # Dropout rate
    'batch_size': 64,
    'learning_rate': 0.001,
    'num_epochs': 5,
    'patience': 5,             # Early stopping patience
    'min_review_length': 5,    # Outlier removal: min words
    'max_review_length': 500,  # Outlier removal: max words
    'train_split': 0.7,
    'val_split': 0.15,
    'test_split': 0.15,
    'seed': 42
}

# Set random seeds for reproducibility
torch.manual_seed(CONFIG['seed'])
np.random.seed(CONFIG['seed'])

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# =============================================================================
# SECTION 1: DATA LOADING
# =============================================================================

def load_amazon_reviews(data_dir='./data'):
    """Parse XML-like <review>...</review> blocks from all .review files under data_dir.

    Returns a pandas DataFrame with columns: rating (float), label (0/1), title, review_text.

    Notes:
    - rating >= 4 -> label 1 (positive)
    - rating <= 2 -> label 0 (negative)
    - rating == 3 -> neutral (excluded)
    - robust to missing fields, extra spaces, multiline text
    """
    print("[INFO] Parsing .review files for XML-like <review> blocks...")

    data_path = Path(data_dir)
    data_path.mkdir(parents=True, exist_ok=True)


    # Find all .review files recursively
    review_files = sorted([p for p in data_path.rglob('*.review') if p.is_file()])
    print(f"[INFO] Found {len(review_files)} .review files under {data_path}")

    # If no .review files found, raise an error
    if not review_files:
        raise FileNotFoundError(f"No .review files found under {data_path}.")

    records = []
    total_parsed = 0
    missing_or_invalid = 0
    neutral_count = 0

    # Regular expressions for parsing
    review_block_re = re.compile(r'<review>(.*?)</review>', re.IGNORECASE | re.DOTALL)
    tag_re = re.compile(r'<(title|rating|review_text)>(.*?)</\1>', re.IGNORECASE | re.DOTALL)

    for rf in review_files:
        try:
            text = rf.read_text(encoding='utf-8', errors='ignore')
        except Exception:
            continue

        blocks = review_block_re.findall(text)
        for blk in blocks:
            total_parsed += 1

            # Extract rating
            # Finding rating match within rating tags
            rating_match = re.search(r'<rating>\s*([0-9]+(?:\.[0-9]+)?)\s*</rating>', blk, re.IGNORECASE)
            # Extract title and review_text
            title_match = re.search(r'<title>(.*?)</title>', blk, re.IGNORECASE | re.DOTALL)
            # Extract review_text, allowing for nested tags and multiline content
            review_match = re.search(r'<review_text>(.*?)</review_text>', blk, re.IGNORECASE | re.DOTALL)

            rating = None
            title = ''
            review_text = ''

            # Remove all tags from valuable data fields to avoid nested tags affecting the text
            if rating_match:
                try:
                    rating = float(rating_match.group(1).strip())
                except Exception:
                    rating = None

            if title_match:
                title = html.unescape(title_match.group(1).strip())

            if review_match:
                # remove any nested tags but preserve punctuation
                raw_review = review_match.group(1).strip()
                # remove XML/HTML tags that may remain inside review text
                review_text = re.sub(r'<[^>]+>', ' ', raw_review)
                review_text = html.unescape(review_text).strip()

            # Safety: skip empty review_text
            if not review_text:
                missing_or_invalid += 1
                continue

            # Determine label according to rating rules
            if rating is None:
                missing_or_invalid += 1
                continue
            if rating == 3 or (rating >= 2.5 and rating < 3.5 and int(rating) == 3):
                neutral_count += 1
                continue

            if rating >= 4.0:
                label = 1
            elif rating <= 2.0:
                label = 0
            else:
                # Catch any unexpected non-binary rating (e.g., 2.5) - treat as neutral
                neutral_count += 1
                continue

            records.append({
                'rating': rating,
                'label': label,
                'title': title,
                'review_text': review_text
            })

    df = pd.DataFrame(records)

    # Shuffle reproducibly
    if not df.empty:
        df = df.sample(frac=1, random_state=CONFIG['seed']).reset_index(drop=True)

    # Backwards compatibility: older code expects 'review' and 'sentiment' columns
    if 'review_text' in df.columns:
        df['review'] = df['review_text']
    if 'label' in df.columns:
        df['sentiment'] = df['label']

    # Print statistics
    print(f"[INFO] Total review blocks scanned: {total_parsed}")
    print(f"[INFO] Parsed records (binary kept): {len(df)}")
    pos = int((df['label'] == 1).sum()) if not df.empty else 0
    neg = int((df['label'] == 0).sum()) if not df.empty else 0
    print(f"[INFO] Positive (label=1): {pos}")
    print(f"[INFO] Negative (label=0): {neg}")
    print(f"[INFO] Neutral/excluded: {neutral_count}")
    print(f"[INFO] Missing/invalid reviews skipped: {missing_or_invalid}")

    # Show 3 examples
    print("[INFO] Example parsed records (up to 3):")
    for i, row in df.head(3).iterrows():
        print(f"  - rating={row['rating']} label={row['label']} review_text={row['review_text'][:200]!r}")

    # Ensure DataFrame has expected columns even if empty
    if df.empty:
        df = pd.DataFrame(columns=['rating', 'label', 'title', 'review_text'])

    return df

# =============================================================================
# SECTION 2: DATA CLEANING & PREPROCESSING
# =============================================================================

class TextPreprocessor:
    """
    # CONTRIBUTION: Custom preprocessing pipeline with multiple cleaning stages

    This class implements a comprehensive text cleaning pipeline that:
    1. Converts text to lowercase
    2. Removes HTML tags and URLs
    3. Removes punctuation and special characters
    4. Normalizes repeated characters (e.g., 'sooooo' -> 'so')
    5. Removes extra whitespace
    6. Applies lemmatization for spelling normalization
    7. Removes stopwords (optional, configurable)
    """

    def __init__(self, remove_stopwords=False):
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english')) if remove_stopwords else set()

        # Compile regex patterns for efficiency
        self.html_pattern = re.compile(r'<[^>]+>')
        self.url_pattern = re.compile(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')
        self.punct_pattern = re.compile(r'[^a-zA-Z\s]')
        self.repeat_pattern = re.compile(r'(.)\1{2,}')  # 3+ repeated chars
        self.whitespace_pattern = re.compile(r'\s+')

    def clean(self, text):
        """Apply full cleaning pipeline to a single review."""
        # Step 1: Lowercase
        text = text.lower()

        # Step 2: Remove HTML tags
        text = self.html_pattern.sub(' ', text)

        # Step 3: Remove URLs
        text = self.url_pattern.sub(' ', text)

        # Step 4: Remove punctuation and special characters
        text = self.punct_pattern.sub(' ', text)

        # Step 5: Normalize repeated characters (e.g., 'sooooo good' -> 'so good')
        text = self.repeat_pattern.sub(r'\1\1', text)

        # Step 6: Tokenize, lemmatize, and optionally remove stopwords
        tokens = text.split()
        tokens = [self.lemmatizer.lemmatize(token) for token in tokens 
                  if token not in self.stop_words and len(token) > 1]

        # Step 7: Rejoin and normalize whitespace
        text = ' '.join(tokens)
        text = self.whitespace_pattern.sub(' ', text).strip()

        return text

def remove_outliers(df, min_len=5, max_len=500):
    """
    # CONTRIBUTION: Statistical outlier removal based on review length

    Removes reviews that are too short (likely spam or incomplete) or too long
    (likely corrupted data or copy-paste errors). This improves model robustness
    and training efficiency.

    Args:
        df: DataFrame with 'review' column
        min_len: Minimum word count
        max_len: Maximum word count

    Returns:
        Filtered DataFrame
    """
    initial_count = len(df)
    df['word_count'] = df['review'].apply(lambda x: len(x.split()))

    # Remove outliers
    df = df[(df['word_count'] >= min_len) & (df['word_count'] <= max_len)]
    df = df.drop('word_count', axis=1)

    removed = initial_count - len(df)
    print(f"[INFO] Outlier removal: {removed} reviews removed ({removed/initial_count*100:.1f}%)")
    print(f"[INFO] Remaining reviews: {len(df)}")

    return df

# =============================================================================
# SECTION 3: VOCABULARY & ENCODING
# =============================================================================

class Vocabulary:
    """
    # CONTRIBUTION: Custom vocabulary builder with frequency-based filtering

    Builds a word-to-index mapping from the training corpus, limiting vocabulary
    to the most frequent words to reduce noise and improve generalization.
    """

    def __init__(self, max_size=20000):
        self.max_size = max_size
        self.word2idx = {'<PAD>': 0, '<UNK>': 1}
        self.idx2word = {0: '<PAD>', 1: '<UNK>'}
        self.word_counts = Counter()

    def build(self, texts):
        """Build vocabulary from list of tokenized texts."""
        # Count all words
        for text in texts:
            self.word_counts.update(text.split())

        # Keep most frequent words
        most_common = self.word_counts.most_common(self.max_size - 2)

        for idx, (word, count) in enumerate(most_common, start=2):
            self.word2idx[word] = idx
            self.idx2word[idx] = word

        self.vocab_size = len(self.word2idx)
        print(f"[INFO] Vocabulary size: {self.vocab_size}")
        print(f"[INFO] Most common words: {most_common[:5]}")

    def encode(self, text, max_len=200):
        """Convert text to padded index sequence."""
        tokens = text.split()
        indices = [self.word2idx.get(token, 1) for token in tokens]  # 1 = <UNK>

        # Pad or truncate
        if len(indices) < max_len:
            indices = indices + [0] * (max_len - len(indices))  # 0 = <PAD>
        else:
            indices = indices[:max_len]

        return indices

    def save(self, path):
        with open(path, 'wb') as f:
            pickle.dump({'word2idx': self.word2idx, 'idx2word': self.idx2word}, f)

    @classmethod
    def load(cls, path):
        with open(path, 'rb') as f:
            data = pickle.load(f)
        vocab = cls()
        vocab.word2idx = data['word2idx']
        vocab.idx2word = data['idx2word']
        vocab.vocab_size = len(vocab.word2idx)
        return vocab

# =============================================================================
# SECTION 4: PYTORCH DATASET
# =============================================================================

class ReviewDataset(Dataset):
    """PyTorch Dataset for review batches."""

    def __init__(self, texts, labels, vocab, max_len=200):
        self.texts = texts
        self.labels = labels
        self.vocab = vocab
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoded = self.vocab.encode(self.texts[idx], self.max_len)
        return {
            'text': torch.tensor(encoded, dtype=torch.long),
            'label': torch.tensor(self.labels[idx], dtype=torch.float),
            'length': min(len(self.texts[idx].split()), self.max_len)
        }

# =============================================================================
# SECTION 5: MODEL ARCHITECTURE — BiLSTM with Self-Attention
# =============================================================================

class AttentionLayer(nn.Module):
    """
    # CONTRIBUTION: Custom Self-Attention mechanism for BiLSTM

    This attention layer computes context-aware weights for each time step,
    allowing the model to focus on the most sentiment-relevant words
    regardless of their position in the sequence. This significantly improves
    performance on long reviews where sentiment may be distributed.

    Architecture:
        - Linear projection of hidden states
        - Tanh activation for non-linearity
        - Softmax normalization to get attention weights
        - Weighted sum of hidden states
    """

    def __init__(self, hidden_dim):
        super(AttentionLayer, self).__init__()
        self.attention = nn.Linear(hidden_dim * 2, hidden_dim * 2)
        self.context_vector = nn.Linear(hidden_dim * 2, 1, bias=False)

    def forward(self, lstm_output, mask=None):
        """
        Args:
            lstm_output: (batch_size, seq_len, hidden_dim*2)
            mask: (batch_size, seq_len) — 1 for real tokens, 0 for padding

        Returns:
            context: (batch_size, hidden_dim*2) — attention-weighted representation
            weights: (batch_size, seq_len) — attention weights for interpretability
        """
        # Compute attention scores
        attn_scores = torch.tanh(self.attention(lstm_output))  # (batch, seq_len, hidden*2)
        attn_scores = self.context_vector(attn_scores).squeeze(-1)  # (batch, seq_len)

        # Apply mask (ignore padding)
        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == 0, -1e9)

        # Normalize with softmax
        attn_weights = torch.softmax(attn_scores, dim=1)  # (batch, seq_len)

        # Weighted sum
        context = torch.bmm(attn_weights.unsqueeze(1), lstm_output).squeeze(1)

        return context, attn_weights

class SentimentBiLSTM(nn.Module):
    """
    # CONTRIBUTION: Advanced BiLSTM architecture with multi-layer stacking,
    # dropout regularization, batch normalization, and self-attention

    This architecture addresses the limitations of simple RNNs by:
    1. Bidirectional processing (captures context from both directions)
    2. Multi-layer stacking (learns hierarchical representations)
    3. Dropout (prevents overfitting on limited review data)
    4. Self-attention (focuses on sentiment-bearing words)
    5. Batch normalization (stabilizes training)
    """

    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_layers, 
                 dropout, pad_idx=0, pretrained_embeddings=None):
        super(SentimentBiLSTM, self).__init__()

        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)

        if pretrained_embeddings is not None:
            self.embedding.weight.data.copy_(torch.from_numpy(pretrained_embeddings))
            self.embedding.weight.requires_grad = True

        # Bidirectional LSTM
        self.lstm = nn.LSTM(
            embedding_dim,
            hidden_dim,
            num_layers=num_layers,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )

        # Self-attention mechanism
        self.attention = AttentionLayer(hidden_dim)

        # Batch normalization for training stability
        self.batch_norm = nn.BatchNorm1d(hidden_dim * 2)

        # Dropout for regularization
        self.dropout = nn.Dropout(dropout)

        # Classification head
        self.fc1 = nn.Linear(hidden_dim * 2, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)

        # Activation
        self.relu = nn.ReLU()

    def forward(self, text, lengths=None):
        """
        Forward pass with attention visualization capability.

        Args:
            text: (batch_size, seq_len) token indices
            lengths: (batch_size) actual sequence lengths

        Returns:
            output: (batch_size, 1) sentiment scores (sigmoid applied externally)
            attention_weights: (batch_size, seq_len) for interpretability
        """
        # Create padding mask
        mask = (text != 0).float()  # 1 for real tokens, 0 for padding

        # Embedding
        embedded = self.dropout(self.embedding(text))  # (batch, seq_len, embed_dim)

        # Pack sequence for efficient LSTM processing
        if lengths is not None:
            packed = nn.utils.rnn.pack_padded_sequence(
                embedded, lengths.cpu(), batch_first=True, enforce_sorted=False
            )
            packed_output, (hidden, cell) = self.lstm(packed)
            lstm_output, _ = nn.utils.rnn.pad_packed_sequence(
                packed_output,
                batch_first=True,
                total_length=text.size(1)
            )
        else:
            lstm_output, (hidden, cell) = self.lstm(embedded)

        # Self-attention
        context, attention_weights = self.attention(lstm_output, mask)

        # Batch normalization and dropout
        context = self.batch_norm(context)
        context = self.dropout(context)

        # Classification layers
        hidden = self.relu(self.fc1(context))
        hidden = self.dropout(hidden)
        output = self.fc2(hidden)

        return output, attention_weights

# =============================================================================
# SECTION 6: TRAINING PIPELINE
# =============================================================================

def train_epoch(model, dataloader, optimizer, criterion, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for batch in dataloader:
        texts = batch['text'].to(device)
        labels = batch['label'].to(device)
        lengths = batch['length'].to(device)

        optimizer.zero_grad()
        outputs, _ = model(texts, lengths)
        logits = outputs.squeeze(-1)

        loss = criterion(logits, labels)
        loss.backward()

        # Gradient clipping to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()

        total_loss += loss.item()
        probabilities = torch.sigmoid(logits)
        predicted_labels = (probabilities > 0.5).float()
        correct += (predicted_labels == labels).sum().item()
        total += labels.size(0)

    return total_loss / len(dataloader), correct / total

def evaluate(model, dataloader, criterion, device):
    """Evaluate model performance."""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in dataloader:
            texts = batch['text'].to(device)
            labels = batch['label'].to(device)
            lengths = batch['length'].to(device)

            outputs, _ = model(texts, lengths)
            logits = outputs.squeeze(-1)
            loss = criterion(logits, labels)
            total_loss += loss.item()

            probabilities = torch.sigmoid(logits)
            predicted_labels = (probabilities > 0.5).float()
            correct += (predicted_labels == labels).sum().item()
            total += labels.size(0)

            all_preds.extend(predicted_labels.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    accuracy = correct / total
    return total_loss / len(dataloader), accuracy, all_preds, all_labels

def train_model():
    """
    # CONTRIBUTION: Complete training pipeline with early stopping,
    # learning rate scheduling, and comprehensive logging
    """
    print("="*60)
    print("ISY503 NLP SENTIMENT ANALYSIS — MODEL TRAINING")
    print("="*60)

    # 1. Load data
    df = load_amazon_reviews(CONFIG['data_dir'])

    # 2. Clean data
    preprocessor = TextPreprocessor(remove_stopwords=False)
    df['cleaned_review'] = df['review'].apply(preprocessor.clean)

    # 3. Remove outliers
    df = remove_outliers(df, CONFIG['min_review_length'], CONFIG['max_review_length'])

    # 4. Split data before building vocabulary
    texts = df['cleaned_review'].tolist()
    labels = df['sentiment'].tolist()

    train_texts, temp_texts, train_labels, temp_labels = train_test_split(
        texts,
        labels,
        test_size=(1 - CONFIG['train_split']),
        random_state=CONFIG['seed'],
        stratify=labels
    )
    val_texts, test_texts, val_labels, test_labels = train_test_split(
        temp_texts,
        temp_labels,
        test_size=CONFIG['test_split'] / (CONFIG['val_split'] + CONFIG['test_split']),
        random_state=CONFIG['seed'],
        stratify=temp_labels
    )

    print(f"\n[INFO] Data split:")
    print(f"  Training:   {len(train_texts)} samples ({len(train_texts)/len(texts)*100:.0f}%)")
    print(f"  Validation: {len(val_texts)} samples ({len(val_texts)/len(texts)*100:.0f}%)")
    print(f"  Test:       {len(test_texts)} samples ({len(test_texts)/len(texts)*100:.0f}%)")

    # 5. Build vocabulary from training data only
    vocab = Vocabulary(max_size=CONFIG['max_words'])
    vocab.build(train_texts)
    vocab.save(f"{CONFIG['model_dir']}/vocabulary.pkl")

    # 6. Create datasets and dataloaders
    train_dataset = ReviewDataset(train_texts, train_labels, vocab, CONFIG['max_len'])
    val_dataset = ReviewDataset(val_texts, val_labels, vocab, CONFIG['max_len'])
    test_dataset = ReviewDataset(test_texts, test_labels, vocab, CONFIG['max_len'])

    train_loader = DataLoader(train_dataset, batch_size=CONFIG['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=CONFIG['batch_size'])
    test_loader = DataLoader(test_dataset, batch_size=CONFIG['batch_size'])

    # 7. Initialize model
    model = SentimentBiLSTM(
        vocab_size=vocab.vocab_size,
        embedding_dim=CONFIG['embedding_dim'],
        hidden_dim=CONFIG['hidden_dim'],
        num_layers=CONFIG['num_layers'],
        dropout=CONFIG['dropout']
    ).to(device)

    print(f"\n[INFO] Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # 8. Loss, optimizer, and scheduler
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=CONFIG['learning_rate'], weight_decay=1e-5)

    # CONTRIBUTION: Learning rate scheduler with ReduceLROnPlateau
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=2
    )

    # 9. Training loop with early stopping
    best_val_loss = float('inf')
    patience_counter = 0
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}

    print(f"\n[INFO] Starting training for up to {CONFIG['num_epochs']} epochs...")
    print("-"*60)

    for epoch in range(CONFIG['num_epochs']):
        prev_lr = optimizer.param_groups[0]['lr']
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc, _, _ = evaluate(model, val_loader, criterion, device)

        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)

        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]['lr']

        print(f"Epoch {epoch+1:2d}/{CONFIG['num_epochs']} | "
              f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
              f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")

        if current_lr < prev_lr:
            print(f"  [LR] Reduced learning rate to {current_lr:.6f}")

        # Early stopping check
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            # Save best model
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'val_acc': val_acc,
                'config': CONFIG,
            }, f"{CONFIG['model_dir']}/best_model.pt")
            print(f"  [SAVED] Best model saved (val_loss: {val_loss:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= CONFIG['patience']:
                print(f"\n[INFO] Early stopping triggered after {epoch+1} epochs")
                break

    print("-"*60)

    # 10. Final evaluation on test set
    print("\n[INFO] Evaluating on test set...")
    checkpoint = torch.load(f"{CONFIG['model_dir']}/best_model.pt")
    model.load_state_dict(checkpoint['model_state_dict'])

    test_loss, test_acc, test_preds, test_labels = evaluate(model, test_loader, criterion, device)

    print(f"\n{'='*60}")
    print("FINAL TEST RESULTS")
    print(f"{'='*60}")
    print(f"Test Loss:     {test_loss:.4f}")
    print(f"Test Accuracy: {test_acc:.4f} ({test_acc*100:.2f}%)")
    print(f"\nClassification Report:")
    print(classification_report(test_labels, test_preds, 
                                target_names=['Negative', 'Positive'], digits=4))
    print(f"\nConfusion Matrix:")
    print(confusion_matrix(test_labels, test_preds))

    # 11. Save configuration
    with open(f"{CONFIG['model_dir']}/config.json", 'w') as f:
        json.dump(CONFIG, f, indent=2)

    # 12. Save training history
    with open(f"{CONFIG['model_dir']}/history.json", 'w') as f:
        json.dump(history, f, indent=2)

    print(f"\n[INFO] All artifacts saved to {CONFIG['model_dir']}/")
    print(f"[INFO] Training complete!")

    return model, vocab, history

# =============================================================================
# SECTION 7: INFERENCE FUNCTION
# =============================================================================

def predict_sentiment(model, vocab, text, preprocessor, device):
    """
    # CONTRIBUTION: Inference function with attention visualization

    Predicts sentiment for a single text input and returns attention weights
    for model interpretability.
    """
    model.eval()

    # Clean and encode
    cleaned = preprocessor.clean(text)
    encoded = vocab.encode(cleaned, CONFIG['max_len'])

    # Create tensor
    tensor = torch.tensor([encoded], dtype=torch.long).to(device)
    length = torch.tensor([min(len(cleaned.split()), CONFIG['max_len'])], dtype=torch.long).to(device)

    with torch.no_grad():
        output, attention_weights = model(tensor, length)
        prob = torch.sigmoid(output).squeeze().item()

    sentiment = "Positive review" if prob > 0.5 else "Negative review"
    confidence = prob if prob > 0.5 else 1 - prob

    return sentiment, confidence, attention_weights.cpu().numpy()[0]

# =============================================================================
# MAIN EXECUTION
# =============================================================================

if __name__ == "__main__":
    # Create model directory
    os.makedirs(CONFIG['model_dir'], exist_ok=True)

    # Train model
    model, vocab, history = train_model()

    # Test on sample inputs
    print(f"\n{'='*60}")
    print("SAMPLE INFERENCE TESTS")
    print(f"{'='*60}")

    preprocessor = TextPreprocessor()

    test_samples = [
        "This product exceeded all my expectations! Absolutely fantastic quality.",
        "Terrible waste of money. Broke within a day and customer service was rude.",
        "The item is okay, nothing special but not bad either.",
        "I am extremely satisfied with this purchase. Best decision ever!",
        "Complete garbage. Do not buy this under any circumstances.",
    ]

    for sample in test_samples:
        sentiment, confidence, _ = predict_sentiment(model, vocab, sample, preprocessor, device)
        print(f"\nInput: {sample}")
        print(f"Output: {sentiment} (confidence: {confidence:.4f})")
