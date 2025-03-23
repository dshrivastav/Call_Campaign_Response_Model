# Call Campaign Response Model

**Can we predict which customers will subscribe to a term deposit offer from a bank?**

This project builds a predictive model to help a financial institution identify which clients are most likely to respond positively to a telemarketing campaign. It uses historical campaign data from a Portuguese bank to develop and evaluate classification models aimed at improving campaign targeting.

---

## 🔍 Project Overview

This project includes:
- Data exploration and preprocessing of **bank marketing campaign data**.
- Development and comparison of **multiple classification models** (Logistic Regression, KNN, Decision Tree, SVM).
- Use of **SMOTE** to balance class imbalance and improve performance on minority class ("yes").
- **Hyperparameter tuning** with GridSearchCV and cross-validation.
- Final model comparison using F1-Score, Accuracy, Precision, and Recall.

---

## 📂 Repository Structure

```
Call_Campaign_Response_Model/
│
├── data/                      # Source dataset (bank-additional-full.csv)
├── images/                    # Visuals: EDA plots, confusion matrices, bar plots
├── reports/
│   └── bank_campaign_model_report.md   # Business-facing findings & recommendations
├── used_model_comparison.ipynb         # Jupyter notebook with full pipeline
└── README.md                 # This file
```

---

## 📈 Data Summary

The dataset includes results from **17 marketing campaigns** over 5 years, promoting **term deposit subscriptions** via phone calls. The **target variable** is `y` (yes/no) indicating if the customer subscribed.

Key facts:
- 41,188 records with 20 input features
- Mix of categorical and numerical features
- Highly imbalanced target (`~11%` subscribed)

---

## 🧪 Modeling Approach

| Step | Action |
|------|--------|
| ✅ Data Cleaning | Dropped `duration` (leakage), handled missing values |
| ✅ Feature Encoding | Used One-Hot Encoding for categorical variables |
| ✅ Scaling | Applied StandardScaler on numerical columns |
| ✅ Class Balancing | SMOTE applied to training set |
| ✅ Model Training | Logistic Regression, KNN, Decision Tree, SVM |
| ✅ Hyperparameter Tuning | GridSearchCV with Stratified KFold |

---

## 📊 Results Summary

### Final Evaluation on Test Data

| Model | Accuracy | F1-Score (Yes) | Precision (Yes) | Recall (Yes) |
|-------|----------|----------------|------------------|---------------|
| ✅ SVM  | **0.87**   | 0.5002         | 0.4546           | 0.5560        |
| DT   | 0.88     | **0.4896**     | 0.4837           | **0.4957**    |
| LR   | 0.82     | 0.4576         | 0.3494           | **0.6627**    |
| KNN  | 0.79     | 0.3623         | 0.2724           | 0.5409        |

> 📌 F1-Score is prioritized due to the imbalanced nature of the data.

---

## 💡 Key Takeaways

- **SMOTE significantly improved recall** for all models by oversampling the "yes" class.
- **SVM** achieved highest accuracy and precision but may miss actual subscribers.
- **Decision Tree** offered the best balance between precision and recall.
- **Logistic Regression** was fastest to train and had high recall after tuning.
- **KNN underperformed**, especially on recall and F1.

---

## 🧠 Business Recommendations

- Use the **Decision Tree model** for targeted outbound campaigns — it's the most balanced and interpretable.
- Prioritize **recall** when the cost of missing a potential subscriber is high.
- Combine this model with campaign automation tools to optimize outreach efficiency.
- Consider **further tuning** or **ensemble models** like Random Forest or XGBoost for improved lift.

---

## 📌 Next Steps

- Deploy as a microservice API for campaign teams to use in real-time.
- Integrate with customer CRM to trigger personalized outreach.
- Explore additional features like call sentiment, prior offer history, and credit scoring.

---

## 📎 References
- [UCI Bank Marketing Dataset](https://archive.ics.uci.edu/ml/datasets/bank+marketing)
- Moro, S., Cortez, P., & Rita, P. (2014). A data-driven approach to predict the success of bank telemarketing.

---