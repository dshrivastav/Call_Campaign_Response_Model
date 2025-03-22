

# 📈 Business & Technical Findings: Call Campaign Response Model

## Business Objective:
Improve customer engagement by **predicting which customers will respond positively** to marketing calls, using historical data from a Portuguese bank's campaign.

---

## Technical Summary

**Data Source:** Bank Marketing Dataset (UCI)  
**Target Variable:** `y` → whether the customer subscribed ("yes" or "no")  
**Challenge:** Highly imbalanced classes (≈88% "no")

---

## Approaches Compared

### Problem 10 (Baseline)
- Trained Logistic Regression, KNN, Decision Tree, SVM with default settings
- Evaluation used Accuracy + Precision + Recall + F1 (Yes class)
- Best overall: Decision Tree (F1: 0.51), SVM (Precision: 0.68)

### Problem 11 (Tuned Models + SMOTE)
- Dropped `duration` (target leakage)
- Applied SMOTE on training data
- Tuned models using GridSearchCV
- Optimized for **F1 score** (Yes class)
- Best overall: SVM (F1: 0.50, Precision: 0.45), Decision Tree (Balanced Recall)

---

## Key Insights

| Model | Best Use Case |
|-------|---------------|
| Decision Tree | Balanced performance, good for general deployment |
| SVM | High confidence predictions (high precision) |
| Logistic Regression | Fast, transparent, interpretable |
| KNN | Not suitable due to poor recall |

---

## Strategic Recommendation

Use **Decision Tree** or **SVM** depending on campaign strategy:

- If **precision matters** (avoid wasting resources), go with **SVM**
- If **you want to catch more real positives**, go with **Decision Tree**
- For deeper modeling, explore **Random Forest + XGBoost**

# 📈 Business-Focused Recommendations: Predicting Subscriber Response

## 🔍 Summary for Business Stakeholders

This project aimed to help the marketing/sales team **predict which customers are most likely to subscribe** after a phone campaign. We trained and evaluated multiple models, and each offers trade-offs depending on your business goals.

---

## ✅ Business-Oriented Recommendations

| **Business Objective**                                 | **Recommended Strategy**                    | **Why It Matters**                                                                 |
|--------------------------------------------------------|---------------------------------------------|-------------------------------------------------------------------------------------|
| 🎯 Maximize sales team efficiency                      | Use **Precision-Focused Model** (e.g., SVM) | Reduces false positives → sales reps only contact most likely subscribers.         |
| 📣 Expand outreach and catch more potential customers  | Use **Recall-Focused Model** (e.g., Logistic Regression w/ SMOTE) | Captures more actual subscribers, useful for broad campaigns.        |
| ⚖️ Balanced strategy with explainability               | Use **Decision Tree (Tuned)**               | Balances catching subscribers with minimizing false leads. Easy to explain.         |
| 🚫 Avoid ineffective models                            | Skip **K-Nearest Neighbors**                | Missed too many real subscribers. Not suited for imbalanced datasets.              |

---

## 🧠 Strategic Takeaways

- **Different models = different business strategies.**
- Choose **precision** when you have limited resources (e.g., small sales team).
- Choose **recall** when your goal is **reach and coverage**.
- Use **balanced models** (like Decision Trees) when both matter and interpretability is important.
- Data techniques like **SMOTE** and **hyperparameter tuning** improve outcomes significantly.

---

## 📌 Next Steps

- Automate this model to rank customers before outreach.
- Combine model output with CRM tools or dashboards.
- Monitor for performance drift and retrain every 6–12 months.

---

**📁 This report is part of the final submission for the Predicting Customer Subscription project.**
