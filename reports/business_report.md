
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

