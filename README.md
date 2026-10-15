# Customer Churn Prediction

A beginner-friendly Machine Learning project that predicts whether a customer is likely to cancel a subscription.

## Project Overview

This project demonstrates the basic Machine Learning workflow, from inspecting and preparing data to training, evaluating and comparing models.

## Technologies Used

- Python
- Pandas
- Scikit-learn
- Decision Tree Classifier

## Dataset Features

- `monthly_charge`
- `months_as_customer`
- `support_tickets`
- `weekly_usage_hours`

**Target:** `churned` (0 = stayed, 1 = cancelled)

## Machine Learning Workflow

1. Load and inspect the dataset.
2. Handle missing values using the median.
3. Separate features and target.
4. Split the data into training and testing sets.
5. Train Decision Tree models.
6. Evaluate predictions using accuracy.
7. Compare model settings and investigate possible overfitting.

## Results

| Model | Maximum Depth | Test Accuracy |
|---|---:|---:|
| Model 1 | 2 | 70% |
| Model 2 | 4 | 80% |
| Model 3 | Unlimited | 80% |

These results are from a small educational dataset and are intended to demonstrate the workflow, not guarantee real-world predictive performance.

## How to Run

Install the required packages:

```bash
python3 -m pip install pandas scikit-learn
```

Run the program:

```bash
python3 churn_model.py
```

## Learning Outcome

This project helped me understand data preparation, classification, model evaluation, accuracy comparison and the importance of testing predictions on unseen data.git add README.md
git commit -m "Add project README"
git push