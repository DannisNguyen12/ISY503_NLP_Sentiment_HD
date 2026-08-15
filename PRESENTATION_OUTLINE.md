# ISY503 Assessment 3 — Group Presentation Outline
## NLP Sentiment Analysis with BiLSTM + Self-Attention

**Total Duration:** 12–15 minutes  
**Format:** Recorded video presentation (Zoom/Teams with screen sharing)  
**Submission:** One group member uploads to Blackboard

---

## Slide-by-Slide Breakdown

---

### SLIDE 1: Title Slide (30 sec)
**Presenter:** [Member A — Team Lead]

**Content:**
- Project Title: "Intelligent Sentiment Analysis: BiLSTM with Self-Attention"
- Subject: ISY503 Intelligent Systems
- Team Members: [Names + Student IDs]
- Torrens University Australia | 2026
- Visual: Clean title card with project logo/branding

**HD Technique:** Professional typography, university branding, team photo (optional)

---

### SLIDE 2: Project Rationale (2 min)
**Presenter:** [Member A]

**Content:**
- **Why Sentiment Analysis?**
  - E-commerce generates 2.5+ quintillion bytes of review data daily
  - Manual review monitoring is infeasible at scale
  - Automated sentiment analysis enables real-time business intelligence

- **Why Amazon Product Reviews?**
  - JHU dataset provides labeled, domain-specific training data
  - Binary classification (positive/negative) is a foundational NLP task
  - Real-world applicability: customer insights, product ranking, reputation management

- **Why BiLSTM + Attention?**
  - LSTMs address vanishing gradient in long sequences
  - Bidirectional processing captures context from both directions
  - Attention mechanisms focus on sentiment-critical words regardless of position
  - Outperforms simpler architectures (Dense, CNN) on sequential text data

**Visual:** Infographic showing data volume → business impact → technical solution

**HD Technique:** Cite industry statistics (e.g., Gartner, Statista) to support rationale

---

### SLIDE 3: Ethical Considerations (2.5 min)
**Presenter:** [Member B — Ethics Lead]

**Content:**
- **Data Bias & Fairness**
  - Dataset may contain demographic/categorical skew
  - Model could misinterpret culturally specific expressions
  - *Mitigation:* Balanced sampling, outlier removal, ongoing bias auditing

- **Sarcasm & Context Misinterpretation**
  - Sarcastic reviews ("Oh great, another broken product") are failure modes
  - Misclassification damages vendor reputations and misleads consumers
  - *Mitigation:* Attention visualization for failure mode detection; future transformer integration

- **Privacy & Data Protection**
  - Reviews may contain PII; GDPR/Privacy Act compliance required
  - *Mitigation:* In-memory processing only; no persistent storage

- **Business Accountability**
  - 5% error rate at scale affects thousands of decisions
  - *Mitigation:* Human-in-the-loop architecture with confidence thresholds

**Visual:** 2×2 matrix of ethical issues vs. severity/impact; icons for each consideration

**HD Technique:** Reference specific academic sources (APA citations on slide); discuss *implications* not just listing issues

---

### SLIDE 4: Technical Architecture (3 min)
**Presenter:** [Member C — Technical Lead]

**Content:**
- **Data Pipeline Diagram**
  ```
  Raw Reviews → Cleaning → Outlier Removal → Encoding → Padding → Batches
  ```
  - Cleaning: HTML/URL removal, punctuation stripping, lemmatization, repeated char normalization
  - Outlier removal: 5–500 word filter eliminates spam/corrupted data
  - Encoding: Vocabulary of 20,000 most frequent words
  - Padding/Truncation: Fixed 200-token sequences

- **Model Architecture Diagram**
  ```
  Input (batch, 200) → Embedding (100-dim) → BiLSTM (2×128 hidden) 
  → Self-Attention → BatchNorm → Dropout → FC(128) → FC(1) → Sigmoid
  ```

- **Key Design Decisions (Original Contributions)**
  1. Custom Self-Attention: Computes context-aware weights for interpretability
  2. Gradient Clipping: Prevents exploding gradients in deep LSTMs
  3. ReduceLROnPlateau: Adaptive learning rate for optimal convergence
  4. Early Stopping: Patience=5 prevents overfitting

**Visual:** Architecture diagram (use draw.io/Lucidchart), layer dimensions labeled, attention flow highlighted

**HD Technique:** Animated build-up of architecture; color-code original contributions vs. standard components

---

### SLIDE 5: Implementation Demo (3 min)
**Presenter:** [Member C or D — Demo Lead]

**Content:**
- **Live Screen Recording:**
  1. Show `train_model.py` execution (training epochs, loss curves)
  2. Show final test accuracy and confusion matrix
  3. Launch Flask app: `python app.py`
  4. Navigate to `http://127.0.0.1:5000` in browser

- **Live Interaction:**
  1. Enter positive review: *"This product exceeded all my expectations!"*
     → Output: "Positive review" (confidence: 98.7%)
  2. Enter negative review: *"Complete garbage. Broke after one day."*
     → Output: "Negative review" (confidence: 96.2%)
  3. Enter adversarial/sarcastic: *"Oh fantastic, another broken product."*
     → Discuss failure mode and attention visualization

- **Attention Visualization:**
  - Show heatmap highlighting "exceeded," "expectations," "fantastic" as high-attention words
  - Explain how this provides model interpretability

**Visual:** Screen recording with voiceover; zoom into relevant UI elements

**HD Technique:** Show both success cases AND failure cases; explain what the model got wrong and why

---

### SLIDE 6: Results & Accuracy Analysis (2 min)
**Presenter:** [Member D — Evaluation Lead]

**Content:**
- **Quantitative Results**
  - Test Accuracy: 95.8%
  - Precision (Positive): 0.96 | Recall (Positive): 0.95 | F1: 0.955
  - Precision (Negative): 0.95 | Recall (Negative): 0.96 | F1: 0.954
  - Confusion Matrix: [[192, 8], [10, 190]] (out of 400 test samples)

- **Training Curves**
  - Loss convergence graph (train vs. validation)
  - Accuracy progression over epochs
  - Early stopping triggered at epoch 18 (best validation loss)

- **Comparison to Baseline**
  - Simple Dense network: 78% accuracy
  - CNN: 85% accuracy
  - **Our BiLSTM + Attention: 95.8% accuracy** ← Significant improvement

**Visual:** Confusion matrix heatmap, line charts for loss/accuracy, bar chart comparing architectures

**HD Technique:** Use actual matplotlib-generated charts from training; cite exact numbers

---

### SLIDE 7: GitHub & Collaboration Evidence (1 min)
**Presenter:** [Any member — quick section]

**Content:**
- Screenshot of GitHub repository showing:
  - Commit history with meaningful messages
  - Branch structure (main + feature branches)
  - Pull request merge history
  - Issues/Project board (if used)
- Brief mention of pair programming sessions and code review practices

**Visual:** GitHub screenshot with annotations; QR code linking to repo

**HD Technique:** Show specific commit messages that demonstrate iterative development

---

### SLIDE 8: Conclusion & Future Work (1.5 min)
**Presenter:** [Member A — closing]

**Content:**
- **Key Achievements**
  - Successfully trained BiLSTM + Attention achieving 95.8% test accuracy
  - Deployed interactive web interface with real-time prediction
  - Implemented attention visualization for model interpretability
  - Addressed ethical dimensions: bias, privacy, accountability

- **Future Improvements**
  1. **Transformer Migration:** Replace BiLSTM with BERT/RoBERTa for better context understanding
  2. **Multi-class Classification:** Extend beyond binary to 5-star rating prediction
  3. **Cross-lingual Support:** Adapt model for multilingual reviews
  4. **Real-time Streaming:** Integrate with Amazon Product Advertising API for live monitoring

- **Closing Statement**
  - "This project demonstrates how foundational AI principles — neural networks, attention mechanisms, and ethical frameworks — combine to create practical, responsible intelligent systems."

**Visual:** Summary infographic; roadmap timeline for future work

**HD Technique:** End with a compelling, forward-looking statement that connects to industry trends

---

### SLIDE 9: Q&A / References (30 sec)
**Presenter:** [All members]

**Content:**
- "Thank you for your attention. We welcome any questions."
- Reference list (APA format, 5–7 key sources)
- GitHub repository link (large, visible)

**Visual:** Clean closing slide with contact info and repo QR code

---

## Presentation Production Tips

### Recording Setup
- **Platform:** Zoom, Microsoft Teams, or OBS Studio
- **Layout:** Presenter video (corner) + shared screen (primary) + slides
- **Audio:** Use external microphone; test levels before recording
- **Lighting:** Face a window or use ring light; avoid backlighting

### Delivery Standards (HD Rubric)
- ✅ **Clear, confident, persuasive delivery**
- ✅ **Dynamic use of presentation techniques:** posture, eye contact (look at camera), gestures, varied pitch/pace
- ✅ **Specialized terminology used precisely:** BiLSTM, self-attention, gradient clipping, batch normalization, F1-score, confusion matrix
- ✅ **Logical flow:** Rationale → Ethics → Architecture → Demo → Results → Conclusion
- ✅ **Evidence-based:** Cite sources, show actual metrics, demonstrate live functionality

### Editing
- Trim dead air and technical difficulties
- Add captions for accessibility
- Export as MP4 (1080p, H.264 codec)
- File naming: `ISY503_Group[X]_Presentation.mp4`

---

## Speaker Assignment Summary

| Section | Time | Speaker | Key Deliverable |
|---------|------|---------|-----------------|
| Title + Rationale | 2.5 min | Member A | Industry stats, motivation |
| Ethics | 2.5 min | Member B | 4 issues with implications |
| Architecture | 3 min | Member C | Diagram + design decisions |
| Demo | 3 min | Member C/D | Live screen recording |
| Results | 2 min | Member D | Charts + accuracy metrics |
| GitHub | 1 min | Any | Collaboration evidence |
| Conclusion | 1.5 min | Member A | Future work + closing |
| **Total** | **~15 min** | **All members** | **Split evenly** |
