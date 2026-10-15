# Customer Churn Prediction — Observations

## 1. Project Decision Sheet

**Business problem:** Predict which customers may cancel their subscriptions.

**Prediction:** Whether a customer will stay or cancel.

**ML problem type:** Classification.

**Features:** monthly_charge, months_as_customer, support_tickets, weekly_usage_hours.

**Target:** churned.

**Success criteria:** Measure test accuracy and compare model performance.

## 2. Dataset Observations

- Total rows: 80
- Total columns: 5
- Missing values were present in support_tickets and weekly_usage_hours.
- Missing values were replaced using the median.
- All missing-value counts became zero after cleaning.

## 3. Model Results

| Model | Maximum Depth | Test Accuracy |
|---|---:|---:|
| Model 1 | 2 | 70% |
| Model 2 | 4 | 80% |
| Model 3 | Unlimited | 80% |

Model 2 improved accuracy by 10 percentage points compared with Model 1.

## 4. ML Workflow Reflection

1. **Business problem:** Identify customers who may cancel.
2. **Classification:** The model predicts stayed or churned.
3. **Features:** Monthly charge, customer tenure, support tickets and weekly usage hours.
4. **Target:** The churned column.
5. **Missing values:** We cleaned them to prepare usable data.
6. **Train/test split:** To evaluate predictions on unseen examples.
7. **model.fit():** Learns patterns from training data.
8. **model.predict():** Predicts churn labels for test customers.
9. **First test accuracy:** 70%.
10. **Changing max_depth:** Accuracy increased to 80% at depth 4.
11. **Incorrect predictions:** Model 1 made 6 incorrect predictions out of 20.
12. **Iterative improvement:** Evaluate, adjust, retrain and compare.

## 5. Lifecycle Challenge

Problem → Data → Prepare → Train → Evaluate → Improve → Repeat

- **Problem:** Define the prediction goal.
- **Data:** Inspect customer information.
- **Prepare:** Clean missing values and separate features and target.
- **Train:** Teach the model using training examples.
- **Evaluate:** Measure performance on test examples.
- **Improve:** Change settings and compare results.
- **Repeat:** Continue experimenting.

## 6. Overfitting Observation

The unlimited-depth model achieved 100% training accuracy and 80% test accuracy. This suggests possible overfitting because its performance was better on training data than on unseen data.

## 7. Final Thinking Question

Even when accuracy is good, the team should inspect incorrect predictions, data quality and real-world risks before trusting the model. Incorrect predictions could lead to unnecessary offers or missed opportunities to retain customers.

## Conclusion

This project helped me understand the Machine Learning workflow, from defining the problem to preparing data, training a model, evaluating results and improving performance.