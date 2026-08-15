# 🎯 HD Checklist — ISY503 NLP Sentiment Analysis

Use this checklist to ensure every HD requirement is met before submission.

---

## ✅ PROJECT CORRECTNESS (40% of marks)

### Code Quality & Originality
- [ ] **NOT copied from external source** — All code has original comments marked with `# CONTRIBUTION`
- [ ] **GitHub repository** created with commit history showing iterative development
- [ ] **Code comments** explain design decisions, not just what the code does
- [ ] **Modular structure** — separate files for training, web app, utilities

### Data Pipeline (All steps from brief)
- [ ] **Clean data** — punctuation removed, spelling normalized (lemmatization)
- [ ] **Encode words** — vocabulary mapping with `<PAD>` and `<UNK>` tokens
- [ ] **Encode labels** — binary: 0=negative, 1=positive
- [ ] **Outlier removal** — eliminate very short (<5 words) and very long (>500 words) reviews
- [ ] **Pad/truncate** — fixed sequence length (200 tokens)
- [ ] **Train/Val/Test split** — 70/15/15 with stratification
- [ ] **Batches** — DataLoader with batch_size=64

### Model Architecture
- [ ] **Define network architecture** — BiLSTM with Self-Attention (not a simple Dense layer)
- [ ] **Define model class** — PyTorch `nn.Module` with `forward()` method
- [ ] **Instantiate network** — proper initialization
- [ ] **Train model** — with validation, early stopping, LR scheduling
- [ ] **Test** — evaluate on held-out test set with metrics

### Web Interface
- [ ] **Simple website** — Flask-based, accessible at `http://127.0.0.1:5000`
- [ ] **Text input field** — textarea for entering review statements
- [ ] **Execute button** — triggers sentiment analysis function
- [ ] **Output display** — shows "Positive review" or "Negative review"
- [ ] **Works on unseen inputs** — test with samples NOT in training data

### Accuracy Requirements
- [ ] **>90% accuracy** on test set (Credit/Distinction level)
- [ ] **100% accuracy** on sample inputs from training data (HD level)
- [ ] **Adversarial testing** — test sarcasm, ambiguity, out-of-domain inputs
- [ ] **Document failure cases** — note what the model gets wrong and why

### Ethical Considerations (in code/report)
- [ ] **Bias identification** — dataset may have demographic/categorical skew
- [ ] **Fairness** — uniform accuracy across review types
- [ ] **Sarcasm handling** — acknowledge limitation and mitigation
- [ ] **Privacy** — no persistent storage of user inputs

---

## ✅ EFFECTIVE COMMUNICATION — PRESENTATION (30% of marks)

### Content Requirements
- [ ] **Rationale** — why you chose NLP sentiment analysis
- [ ] **Ethical considerations** — discussed in detail with implications
- [ ] **Accuracy of outputs** — show actual metrics (confusion matrix, accuracy %)
- [ ] **Implementation explanation** — architecture diagram, key design decisions
- [ ] **Live demo** — show the web app working on positive, negative, AND adversarial inputs

### Delivery Quality (HD = "Expertly presented")
- [ ] **Logical, persuasive, well-supported** — every claim backed by evidence
- [ ] **Clear flow of ideas** — Rationale → Ethics → Architecture → Demo → Results → Conclusion
- [ ] **Engages and sustains audience interest** — dynamic delivery, not monotone
- [ ] **Specialized language** — BiLSTM, self-attention, gradient clipping, F1-score, etc.
- [ ] **Presentation aids** — diagrams, charts, live screen recording, animations
- [ ] **Each member presents** — split 15 minutes evenly among 3–4 members
- [ ] **Video quality** — 1080p, clear audio, no technical glitches

### Technical Language
Use these terms accurately throughout:
- Bidirectional LSTM (BiLSTM)
- Self-attention mechanism
- Word embedding / GloVe
- Gradient clipping / exploding gradients
- Early stopping / patience
- Learning rate scheduling / ReduceLROnPlateau
- Batch normalization
- Dropout regularization
- Confusion matrix / precision / recall / F1-score
- Stratified split / DataLoader
- Lemmatization / tokenization
- Outlier removal / padding / truncation

---

## ✅ INDIVIDUAL CONTRIBUTION REPORT (30% of marks)

### Report Content
- [ ] **250 words** (±10% = 225–275 words)
- [ ] **Team member list** — names + student IDs
- [ ] **Your contribution** — described in detail with specific technical tasks
- [ ] **Peer assessment** — percentages for ALL members summing to 100%
- [ ] **Rationale** — 1–2 sentences justifying each percentage

### Ethical Analysis (HD = "multiple reasonable ethical aspects AND implications")
- [ ] **Data bias** — dataset skew and its impact on fairness
- [ ] **Sarcasm/ambiguity** — misclassification risks for businesses
- [ ] **Privacy** — PII in reviews, GDPR compliance
- [ ] **Accountability** — automated decisions affecting product rankings
- [ ] **APA references** — cite at least 3–4 academic sources

---

## 📦 SUBMISSION CHECKLIST

### Group Submission (One member submits)
- [ ] **Group Code** — ZIP of entire project or GitHub link
- [ ] **Group Video Presentation** — MP4 file, 10–15 minutes

### Individual Submission (EACH member submits)
- [ ] **Individual Report** — 250 words, contribution + ethics + references
- [ ] **GitHub Link** — included in report

---

## 🚀 EXECUTION STEPS

### Step 1: Set Up Environment
```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate    # Windows

# Install dependencies
pip install -r requirements.txt
```

### Step 2: Train the Model
```bash
python train_model.py
```
- Wait for training to complete
- Verify `models/best_model.pt` is created
- Check test accuracy is >90%

### Step 3: Test the Web App
```bash
python app.py
```
- Open browser to `http://127.0.0.1:5000`
- Test with 5+ positive and 5+ negative reviews
- Test with 2+ sarcastic/ambiguous reviews
- Screenshot results for presentation

### Step 4: Push to GitHub
```bash
git init
git add .
git commit -m "Initial commit: BiLSTM sentiment analysis"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/isy503-sentiment.git
git push -u origin main
```
- Make regular commits with meaningful messages
- Create feature branches if collaborating

### Step 5: Prepare Presentation
- Follow `PRESENTATION_OUTLINE.md`
- Record using Zoom/Teams/OBS
- Include live demo of web app
- Keep under 15 minutes

### Step 6: Write Individual Report
- Use `INDIVIDUAL_REPORT_TEMPLATE.md`
- Fill in YOUR specific contributions
- Ensure percentages sum to 100%
- Include APA references

---

## 💡 HD PRO TIPS

1. **Attention Visualization** — Our web app shows which words the model focused on. This is a unique feature that demonstrates advanced understanding and will impress markers.

2. **Adversarial Testing** — Don't just show perfect results. Show a sarcastic review that fails, explain WHY it failed, and discuss how transformers could fix it. This shows critical thinking = HD.

3. **Ethical Depth** — Don't just list ethical issues. Explain the *implications* — what happens in the real world if this issue isn't addressed? Cite academic sources.

4. **GitHub Evidence** — Take screenshots of your commit history with meaningful messages. This proves collaboration and iterative development.

5. **Confidence Scoring** — Our app shows confidence percentages. Mention in your presentation that low-confidence predictions should trigger human review — this shows professional awareness.

6. **Comparison to Baseline** — Train a simple Dense network (78% accuracy) and compare to your BiLSTM (95%+). The improvement gap proves your architecture is superior.

---

## 📞 NEED HELP?

- **Dataset issues:** Check http://www.cs.jhu.edu/~mdredze/datasets/sentiment/ for download instructions
- **PyTorch errors:** Ensure CUDA is available or the model will run on CPU (slower but works)
- **Flask not starting:** Check port 5000 isn't already in use; try `flask run --port=5001`
- **Model not loading:** Run `train_model.py` first to generate `best_model.pt`

---

**Good luck! You've got everything you need for a High Distinction. 🎓**
