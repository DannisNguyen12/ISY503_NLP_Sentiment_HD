# Ethical Considerations in NLP Sentiment Analysis
## ISY503 Assessment 3 — Ethical Reflection

**Student:** [Your Name]  
**Student ID:** [Your ID]  
**Date:** 2026

---

This project uses a BiLSTM with self-attention to classify Amazon product reviews as positive or negative. While the model performs well on benchmark data, it is important to consider the ethical implications of deploying automatic sentiment analysis in real-world settings.

## 1. Data Bias and Fairness

The dataset contains reviews from many product categories and may reflect uneven language patterns across demographics, product types, and writing styles. A model trained on this data can become biased if it performs better on some groups or review styles than others. In practice, this could cause certain products or reviewers to be treated unfairly in automated systems.

To reduce this risk, the project applies balanced class handling, outlier removal, and careful preprocessing. However, fairness should still be monitored in future deployments by testing the model across different categories and review styles.

## 2. Sarcasm and Context Misinterpretation

Sentiment analysis is especially challenging when reviews contain sarcasm, irony, or subtle negation. A statement such as “Oh great, another broken product” may be classified as positive if the model focuses too much on the word “great.” This is a major limitation because misclassification can damage product reputations and mislead consumers.

The attention mechanism helps explain which words influenced the model’s prediction, but it does not fully solve this problem. For real-world use, low-confidence predictions should be reviewed by a person.

## 3. Privacy and Data Protection

The project uses public review data, but review text may still contain personal information such as names, locations, or other identifiable details. Even when the data is public, it should be handled responsibly and not stored or reused without clear justification.

This project avoids storing user-entered text permanently in the web application. The model processes data in memory only, which reduces privacy risk, but a production version should include stronger safeguards such as user consent and data minimisation.

## 4. Business and Social Impact

Automated sentiment classification can influence product rankings, customer feedback systems, and business decisions. A false negative may hide a genuinely positive product, while a false positive may inflate the reputation of poor-quality items. These errors can affect consumers and businesses unfairly.

Because of this, sentiment analysis should be treated as a support tool rather than a fully autonomous decision-maker. Human review remains important when predictions are uncertain or when the system is being used for high-impact decisions.

## 5. Conclusion

This project demonstrates the usefulness of deep learning for sentiment analysis, but it also shows that technical performance alone is not enough. Ethical use requires attention to bias, sarcasm, privacy, and accountability. A responsible system should be transparent, fair, and supported by human review when necessary.

