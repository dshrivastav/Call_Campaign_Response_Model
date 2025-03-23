# 📞 Call Campaign Response Model – Predicting Bank Subscription

This project aims to help financial institutions **predict whether a customer will respond positively** to marketing phone calls. Using the Portuguese bank marketing dataset, we built, tuned, and evaluated multiple classification models with a focus on **precision, recall, and F1-score** for the positive class ("yes").

---

## 🧠 Business Objective

Enable the marketing team to:
- Prioritize leads more effectively
- Reduce wasted outreach efforts
- Increase campaign conversion rates

---

## 📊 Dataset Summary

- **Source:** UCI Bank Marketing Dataset  
- **Size:** 41,188 records, 20 features  
- **Target Variable:** `y` – whether a customer subscribed ("yes")  
- **Imbalance:** ~88% "no", ~12% "yes"

---

## 🔍 Modeling Approach

### Step 1: Baseline Model Comparison
- Algorithms: Logistic Regression, KNN, Decision Tree, SVM
- Metrics: Accuracy, Precision, Recall, F1 (Yes class)
- Result:  
  - **Best Accuracy:** SVM (0.9058)  
  - **Best F1 Score:** Decision Tree (0.5120)  
  - **KNN** had high precision but low recall

### Step 2: Feature Engineering + SMOTE + Hyperparameter Tuning
- Dropped `duration` (target leakage)
- Used **SMOTE** to balance the "yes" class
- Tuned models using `GridSearchCV`
- Focused optimization on **F1 Score (yes)**

### Evaluation Metrics Used:
- **Precision (yes)**: How many predicted "yes" were correct?
- **Recall (yes)**: How many actual "yes" were found?
- **F1 Score**: Balanced performance metric

---

## 🧪 Model Results: Before vs After Tuning

| Model                   | Accuracy (P10) | Accuracy (P11) | F1 (P10) | F1 (P11) | Precision (P11) | Recall (P11) |
|------------------------|----------------|----------------|----------|----------|------------------|---------------|
| Logistic Regression    | 0.8998         | 0.82           | 0.4497   | 0.4576   | 0.3494           | ✅ 0.6627      |
| K-Nearest Neighbors    | 0.8873         | 0.79           | 0.2692   | 0.3623   | 0.2724           | 0.5409        |
| Decision Tree          | 0.8814         | ✅ 0.88        | ✅ 0.5120 | 0.4896   | 0.4837           | ✅ 0.4957      |
| Support Vector Machine | ✅ 0.9058      | 0.87           | 0.3803   | ✅ 0.5002 | ✅ 0.4546         | 0.5560        |

> ✅ P10 = Baseline, P11 = Tuned + SMOTE

---

## 💼 Business-Focused Recommendations

| **Goal**                              | **Use This Model**        | **Why**                                                         |
|---------------------------------------|---------------------------|------------------------------------------------------------------|
| 🎯 Precision (reduce false positives) | SVM                       | Saves rep time by contacting most likely subscribers             |
| 📣 Recall (catch more "yes")         | Logistic Regression + SMOTE | Maximizes actual subscribers captured                           |
| ⚖️ Balanced & Explainable             | Decision Tree (Tuned)     | Best trade-off; also easy to explain to stakeholders             |
| ❌ Avoid                              | KNN                       | Missed too many actual subscribers, low recall                   |

---

## 📌 Key Learnings

- SMOTE significantly improved model recall by balancing classes.
- GridSearchCV tuning helped squeeze more F1 performance.
- Logistic Regression is fast and interpretable, but SVM and Decision Tree deliver better recall/precision trade-offs.
- **Decision Trees** and **SVMs** are strong candidates for deployment depending on resource constraints.

---

## 📂 Project Structure

```
Call_Campaign_Response_Model/
├── data/                         # Data (raw or preprocessed)
├── images/                       # Plots & visualizations
├── reports/
│   └── bank_campaign_model_report.md  # Full business + technical findings
├── used_campaign_response_model.ipynb # Jupyter notebook
├── README.md                    # Project summary
```

---

## 🏁 Next Steps

- Deploy model via REST API or cloud service
- Automate customer scoring prior to outreach
- Consider ensemble models like XGBoost, Random Forest
- Retrain periodically to handle drift

---

_Last updated: 2025-03-23_  
