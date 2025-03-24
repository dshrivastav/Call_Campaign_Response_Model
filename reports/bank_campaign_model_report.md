# Business Report: Driving Smarter Outreach with Predictive Modeling

## Introduction

As part of our project to enhance marketing effectiveness, we developed a predictive model to identify customers most likely to respond positively to a **term deposit offer** during a phone campaign. Our objective was not only to improve model accuracy, but more importantly, to deliver **business-aligned insights** that help teams optimize campaign resources and drive higher conversion.

This report outlines the major findings, how they translate into strategic decisions, and what next steps would bring the most business value.

## Business Objective

Financial institutions often conduct outbound campaigns via phone to promote term deposits. However, only a small fraction (~11%) of customers actually subscribe. The objective of this project was to **improve campaign efficiency by prioritizing outreach to customers most likely to respond positively**, thereby increasing conversion rates while reducing wasted effort.

## Strategic Summary

- The dataset is **highly imbalanced**, with nearly 88% of customers saying "no".
- A **naive model** predicting “no” every time would achieve ~88% accuracy — but this is not useful for business decision-making.
- Real success requires identifying the rare “yes” cases effectively — without over-contacting the wrong customers.

## Modeling Approach

- Built baseline classifiers: Logistic Regression, KNN, Decision Tree, SVM
- Applied **feature scaling** and **One-Hot Encoding**
- Dropped `duration` due to **target leakage**
- Used **SMOTE** to balance the training dataset
- Performed **GridSearchCV hyperparameter tuning** with Stratified K-Fold
- Evaluated models on **F1-score (yes class)**, along with Accuracy, Precision, Recall

## Key Findings

### 1. Model Performance and Business Fit

We evaluated multiple classification models — including Logistic Regression, Decision Tree, SVM, and K-Nearest Neighbors (KNN) — to find the best fit for marketing goals. All models were evaluated on their ability to correctly predict the minority class ("yes"), which represents customers who subscribed.

| Model                 | Accuracy | Precision (Yes) | Recall (Yes) | F1-Score (Yes) |
|----------------------|----------|------------------|--------------|----------------|
| SVM (Tuned)          | 0.87     | **0.4546**        | 0.5560       | **0.5002**     |
| Decision Tree (Tuned)| **0.88** | 0.4837           | **0.4957**   | 0.4896         |
| Logistic Regression  | 0.82     | 0.3494           | **0.6627**   | 0.4576         |
| KNN (Tuned)          | 0.79     | 0.2724           | 0.5409       | 0.3623         |

> Visual Reference:
>
> ![F1 Score Bar Plot](../images/f1_score_comparison.png)
> ![Confusion Matrices](../images/confusion_matrix_grid.png)

- **Support Vector Machine (SVM):** Delivered the highest precision, making it ideal when avoiding false positives is critical. This ensures that sales teams focus only on high-probability leads.

- **Decision Tree (Tuned):** Consistently offered the best balance between precision and recall (i.e., F1-score). Also easy to explain, making it suitable for team alignment and auditability.

- **Logistic Regression + SMOTE:** Showed the highest recall post-tuning. A strong choice for campaigns where reach and brand exposure matter more than precision.

- **KNN:** Performed poorly and is not recommended due to low recall and F1 performance.

### 2. Strategic Drivers for Model Selection

Different outreach strategies call for different models:

- Use **SVM** when resources are limited and every call must count — for example, when leads are expensive or the sales cycle is high-touch.

- Use a **Decision Tree** when aiming for balance — it identifies real subscribers without missing too many or overwhelming sales reps with false leads.

- Use **Logistic Regression** when the goal is awareness — even if it means more false positives, you’ll reach a wider pool of potential subscribers.

## Model Recommendations Based on Business Goals

| Business Goal                          | Recommended Model       | Why?                                                             |
|----------------------------------------|--------------------------|------------------------------------------------------------------|
| Maximize campaign efficiency           | SVM (Tuned)              | High precision; avoids wasted calls on unlikely leads            |
| Catch more real subscribers            | Logistic Regression (Tuned) | Highest recall; good for awareness/coverage                  |
| Balanced performance & interpretability| Decision Tree (Tuned)    | Best F1-score; interpretable and explainable                     |
| Avoid ineffective models               | ❌ Skip KNN              | Underperformed on recall and F1; not ideal for imbalanced data   |


## Actionable Insights for Marketing & Sales Teams

### Lead Prioritization
Use model predictions to rank leads before outreach. Focus higher-effort sales reps on high-confidence leads (SVM), while keeping a broader list for mass-market outreach (Logistic Regression).

### Campaign Targeting Strategy
Let model selection drive campaign types:

- Precision campaigns for limited resources → SVM  
- Awareness campaigns for large-scale reach → Logistic Regression  
- Balanced outbound strategy → Decision Tree

### CRM Integration
Incorporate model scores into CRM tools to trigger automatic workflows, such as:
- Assigning sales priority  
- Email nurturing for medium-likelihood leads  
- Deferring low-confidence contacts to future campaigns

## Next Steps for Deployment

To bring this model into business operations, we recommend the following:

### 1. Deployment as a Microservice
Package the selected model (SVM or Decision Tree) behind an API that can be called by CRM or call tools in real time.

### 2. Dashboard for Campaign Teams
Build an internal dashboard where campaign leads can input filters and see real-time predictions on potential subscribers.

### 3. Monitor & Retrain the Model
Establish periodic retraining (every 6–12 months) using new campaign data to adapt to shifts in customer behavior.

### 4. Explore Ensemble Models
Future iterations can incorporate ensemble techniques like Random Forest or XGBoost for potentially higher lift.

## Closing Thought

This project demonstrates how predictive modeling can move beyond technical performance to deliver real-world marketing ROI. By aligning model choices with campaign goals — precision, reach, or balance — marketing and sales teams can focus their efforts where they matter most: on the customers most likely to say “yes.”
