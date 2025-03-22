
# 📞 Call Campaign Response Model

This project predicts customer responses (`yes` or `no`) to **outbound marketing calls** by a bank, helping prioritize follow-ups and improve conversion rates. It demonstrates classification model evaluation, tuning, and communication of technical results to business stakeholders.

---

## ✅ Problem 10 – Baseline Models (Default Settings)

We evaluated 4 models:
- Logistic Regression
- K-Nearest Neighbors (KNN)
- Decision Tree
- Support Vector Machine (SVM)

### Summary:

| Model                   | Accuracy | F1 Score (Yes) | Precision (Yes) | Recall (Yes) |
|------------------------|----------|----------------|------------------|---------------|
| Logistic Regression    | 0.8998   | 0.4497         | 0.5816           | 0.3637        |
| K-Nearest Neighbors    | 0.8873   | 0.2692         | 0.6232           | 0.1735        |
| Decision Tree          | 0.8814   | ✅ **0.5120**   | 0.5415           | ✅ 0.4854      |
| Support Vector Machine | ✅ 0.9058 | 0.3803         | ✅ 0.6832         | 0.2626        |

🧠 **Key Insight**: Decision Tree gave the most balanced results. SVM had high accuracy but missed most actual "yes" responses.

---

## 🔁 Problem 11 – Tuned Models with SMOTE

We improved performance by:
- Dropping 'duration' (leakage)
- One-Hot Encoding + Scaling
- Handling class imbalance using **SMOTE**
- Performing **GridSearchCV** on all 4 models

### Summary (After Tuning):

| Model                   | Accuracy | F1 Score (Yes) | Precision (Yes) | Recall (Yes) |
|------------------------|----------|----------------|------------------|---------------|
| Logistic Regression    | 0.82     | 0.4576         | 0.3494           | ✅ 0.6627      |
| K-Nearest Neighbors    | 0.79     | 0.3623         | 0.2724           | 0.5409        |
| Decision Tree          | ✅ 0.88   | 0.4896         | 0.4837           | 0.4957        |
| Support Vector Machine | 0.87     | ✅ **0.5002**   | ✅ 0.4546         | 0.5560        |

🧠 **Key Insight**: SMOTE boosted recall. SVM still had best precision. Decision Tree remained most balanced.

---

## 🔎 Final Comparison: Default vs Tuned

| Model | F1 Before | F1 After | Accuracy Before | Accuracy After |
|-------|-----------|----------|------------------|----------------|
| Logistic Regression | 0.4497 | 0.4576 | 0.8998 | 0.82 |
| KNN                 | 0.2692 | 0.3623 | 0.8873 | 0.79 |
| Decision Tree       | ✅ 0.5120 | 0.4896 | 0.8814 | ✅ 0.88 |
| SVM                 | 0.3803 | ✅ 0.5002 | ✅ 0.9058 | 0.87 |

---

## 💼 Business Takeaways

- 🎯 SVM is great when **false positives are costly** (e.g., small sales team).
- 🧠 Decision Tree is best when **recall and precision** both matter (balanced).
- ⚡ Logistic Regression is ideal for **fast, interpretable** decisions.
- ❌ KNN is not suitable here due to poor recall.

---

## 📌 Recommendations

- Use **SMOTE or ensemble models** for better minority class capture
- Prefer **Decision Trees or SVM** based on stakeholder needs
- Consider **Random Forest or XGBoost** for production deployment
- Explore **ROC-AUC and PR curves** for deeper analysis

---

