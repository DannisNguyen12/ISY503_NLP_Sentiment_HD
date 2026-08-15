# ISY503 Assessment 3 — Individual Contribution Report

**Student Name:** [Your Full Name]  
**Student ID:** [Your Student ID]  
**Group Number:** [Group X]  
**Date:** 2026  
**Word Count:** 250 words (±10% = 225–275 words)

---

## 1. Team Members & Roles

| Name | Student ID | Primary Role | Secondary Role |
|------|-----------|--------------|----------------|
| [Member 1] | [ID] | Model Architecture & Training | Technical Documentation |
| [Member 2] | [ID] | Data Preprocessing & Pipeline | Testing & Evaluation |
| [Member 3] | [ID] | Web Application Development | UI/UX Design |
| [Member 4] | [ID] | Ethical Analysis & Reporting | Presentation Design |

---

## 2. My Contribution to the Project

[Write 100–120 words describing YOUR specific technical and collaborative contributions. Be specific about code, design decisions, testing, documentation, or research you performed.]

**Example (adapt to your actual work):**

> I was responsible for designing and implementing the BiLSTM with Self-Attention architecture in PyTorch. I developed the custom `AttentionLayer` class that computes context-aware weights over bidirectional LSTM outputs, enabling the model to focus on sentiment-critical words. I also implemented the training pipeline including early stopping with patience=5, ReduceLROnPlateau learning rate scheduling, and gradient clipping to prevent exploding gradients. I conducted hyperparameter tuning experiments (hidden dimensions: 64, 128, 256; dropout rates: 0.3, 0.5, 0.7) and selected the optimal configuration achieving 95.8% test accuracy. Additionally, I wrote the model evaluation script generating the confusion matrix and classification report, and contributed to the GitHub repository with 15 commits across the model development branch.

---

## 3. Perceived Contribution of Team Members

### Self-Assessment
**My contribution:** [X]%  
*Rationale:* [1–2 sentences explaining why you assigned yourself this percentage]

### Peer Assessment

| Team Member | Perceived Contribution | Rationale |
|-------------|----------------------|-----------|
| [Member 1] | [X]% | [Brief justification: e.g., "Led architecture design and solved critical LSTM gradient issues"] |
| [Member 2] | [X]% | [Brief justification: e.g., "Built complete data pipeline and achieved clean dataset with zero null values"] |
| [Member 3] | [X]% | [Brief justification: e.g., "Developed full-stack Flask application with attention visualization feature"] |
| [Member 4] | [X]% | [Brief justification: e.g., "Authored comprehensive ethical analysis and designed presentation slides"] |

**Total:** [X]% + [X]% + [X]% + [X]% = **100%**

---

## 4. Ethical Considerations

[Write 80–100 words listing ethical aspects of NLP sentiment analysis WITH their implications. This is critical for HD.]

**Example:**

> Our sentiment analysis system raises four key ethical concerns. First, **data bias**: the Amazon dataset may contain demographic skew, causing the model to perform poorly on reviews from underrepresented groups, which unfairly disadvantages certain vendors (Mehrabi et al., 2021). Second, **sarcasm misinterpretation**: the model struggles with ironic intent, potentially damaging business reputations through incorrect negative flagging (Ghosh & Veale, 2016). Third, **privacy**: reviews may contain PII, requiring GDPR-compliant in-memory processing without persistent storage (Voigt & Von dem Bussche, 2017). Fourth, **accountability**: automated decisions influence product rankings; we recommend human-in-the-loop validation for low-confidence predictions (Holzinger et al., 2019).

---

## 5. References (APA Format)

Ghosh, A., & Veale, T. (2016). Fracking sarcasm using neural network. *Proceedings of the 7th Workshop on Computational Approaches to Subjectivity, Sentiment and Social Media Analysis*, 161–169.

Holzinger, A., Langs, G., Denk, H., Zatloukal, K., & Müller, H. (2019). Causability and explainability of artificial intelligence in medicine. *Wiley Interdisciplinary Reviews: Data Mining and Knowledge Discovery*, 9(4), e1312.

Mehrabi, N., Morstatter, F., Saxena, N., Lerman, K., & Galstyan, A. (2021). A survey on bias and fairness in machine learning. *ACM Computing Surveys*, 54(6), 1–35. https://doi.org/10.1145/3457607

Voigt, P., & Von dem Bussche, A. (2017). *The EU General Data Protection Regulation (GDPR): A practical guide*. Springer.

---

## 6. Self-Assessment Checklist

Before submitting, verify:

- [ ] My contribution is described with specific technical details
- [ ] Percentages sum to exactly 100%
- [ ] Each team member has a brief rationale for their percentage
- [ ] Ethical considerations include both the issue AND its implication
- [ ] References are in APA format
- [ ] Word count is between 225–275 words
- [ ] I have saved a copy of this report

---

**Declaration:** I declare that this report accurately reflects my contribution to the group project and my honest assessment of my team members' contributions.

**Signature:** ___________________  
**Date:** ___________________
