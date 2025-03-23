
# 📞 Call Campaign Response Model – Predicting Subscriber Likelihood

This project helps predict **whether a customer will respond positively** to a bank’s call campaign. We use classification models to distinguish between customers likely to subscribe to a term deposit (`yes`) and those who won't (`no`), enabling smarter marketing efforts and optimized outreach.

---

## 🔍 Project Overview

- **Business Goal**: Increase campaign efficiency by identifying the right customers to call.
- **Data Source**: UCI Bank Marketing Dataset.
- **Target Variable**: `y` — binary outcome: `yes` or `no` for subscription.
- **Challenge**: Imbalanced data — only ~11% of customers said "yes".

---

## 🎯 Objective

Develop and compare multiple ML classifiers to:
- Accurately predict likely subscribers (the "yes" class).
- Optimize outreach strategy by minimizing false positives and false negatives.
- Provide interpretable and actionable recommendations for marketing.

---

## 📁 Repository Structure

```
Call_Campaign_Response_Model/
│
├── data/                   # Raw dataset and any intermediate files
├── images/                 # Visualizations and plots
├── reports/                # Final report and business summary
│   └── business_findings.md
├── used_call_campaign_model.ipynb   # Final Jupyter notebook
├── README.md
└── requirements.txt        # Python dependencies
```

---

## 🧪 Models Compared

We evaluated the following classifiers:
- Logistic Regression
- K-Nearest Neighbors (KNN)
- Decision Tree Classifier
- Support Vector Machine (SVM)

Each model was:
- Trained with default settings (Baseline Phase)
- Then tuned using `GridSearchCV` after handling class imbalance (Tuned Phase)

---

## 🧹 Preprocessing & Feature Engineering

| Step                           | Description |
|--------------------------------|-------------|
| Dropped `duration`             | Leaks future info, not known before call |
| One-Hot Encoding               | For categorical variables |
| Standard Scaler                | For numeric features |
| SMOTE                          | Applied only on training set to balance classes |
| Train/Test Split               | 80/20 stratified split |

---

## 📊 Evaluation Metrics

For each model, we measured:
- Accuracy
- Precision, Recall, F1 Score for the `yes` class
- Confusion Matrix
- Training time (for baseline)

### ⚖️ Why F1 Score?
Due to imbalanced data, **F1 score on the `yes` class** was prioritized to balance false positives/negatives.

---

## 📈 Model Results

### 🔹 Baseline (Problem 10) — Default Parameters

| Model                   | Train Acc | Test Acc | Precision (Yes) | Recall (Yes) | F1 Score (Yes) |
|------------------------|-----------|----------|------------------|--------------|----------------|
| Logistic Regression    | 0.9001    | 0.8998   | 0.5816           | 0.3637       | 0.4497         |
| K-Nearest Neighbors    | 0.9027    | 0.8873   | 0.6232           | 0.1735       | 0.2692         |
| Decision Tree          | 0.9766    | 0.8814   | 0.5415           | ✅ 0.4854     | ✅ **0.5120**   |
| Support Vector Machine | 0.9365    | ✅ 0.9058| ✅ 0.6832         | 0.2626       | 0.3803         |

---

### 🔹 Tuned (Problem 11) — SMOTE + GridSearchCV

| Model                   | Test Acc | F1 Score (Yes) | Precision (Yes) | Recall (Yes) |
|------------------------|----------|----------------|------------------|--------------|
| Logistic Regression    | 0.82     | 0.4576         | 0.3494           | ✅ 0.6627     |
| K-Nearest Neighbors    | 0.79     | 0.3623         | 0.2724           | 0.5409       |
| Decision Tree          | ✅ 0.88  | ✅ 0.4896       | 0.4837           | 0.4957       |
| Support Vector Machine | 0.87     | 0.5002         | ✅ 0.4546         | 0.5560       |

---

## 💼 Business Recommendations

| Objective                             | Suggested Model   | Reason                                           |
|--------------------------------------|-------------------|--------------------------------------------------|
| High Precision Outreach               | SVM               | Avoids false positives; useful for targeted sales|
| Balanced Outreach (Precision + Recall)| Decision Tree      | Best F1, interpretable, good trade-off           |
| Broad Outreach (High Recall)         | Logistic Regression| Captures more positives, suitable for awareness  |
| Avoid                                 | KNN               | Poor performance on minority class               |

---

## ✅ Next Steps

- Try Ensemble Models (Random Forest, XGBoost)
- Deploy the best model into a CRM pipeline
- Add cost-benefit thresholds for campaign tuning
- Retrain model with newer campaign data periodically

---

## 📦 Requirements

```bash
pandas
numpy
scikit-learn
matplotlib
seaborn
imblearn
```

Install with:
```bash
pip install -r requirements.txt
```

---

## 📌 Credits

- Dataset: [UCI Machine Learning Repository - Bank Marketing Data](https://archive.ics.uci.edu/ml/datasets/bank+marketing)
- Project developed by Dipti Srivastava
