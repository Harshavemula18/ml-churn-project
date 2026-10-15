
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# ==========================================
# STEP 1: LOAD THE DATASET
# ==========================================

data = pd.read_csv("customer_churn.csv")

print("=" * 50)
print("CUSTOMER CHURN PREDICTION PROJECT")
print("=" * 50)

print("\nFirst 5 customer records:")
print(data.head())

print("\nDataset shape:", data.shape)

# ==========================================
# STEP 2: CHECK AND CLEAN MISSING VALUES
# ==========================================

print("\nMissing values before cleaning:")
print(data.isnull().sum())

data["support_tickets"] = data["support_tickets"].fillna(
    data["support_tickets"].median()
)

data["weekly_usage_hours"] = data["weekly_usage_hours"].fillna(
    data["weekly_usage_hours"].median()
)

print("\nMissing values after cleaning:")
print(data.isnull().sum())

# ==========================================
# STEP 3: DEFINE FEATURES AND TARGET
# ==========================================

X = data[
    [
        "monthly_charge",
        "months_as_customer",
        "support_tickets",
        "weekly_usage_hours"
    ]
]

y = data["churned"]

print("\nFeatures (X):")
print(X.head())

print("\nTarget (y):")
print(y.head())

# ==========================================
# STEP 4: SPLIT TRAINING AND TEST DATA
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.25,
    random_state=42,
    stratify=y
)

print("\nTraining examples:", len(X_train))
print("Testing examples:", len(X_test))

# ==========================================
# STEP 5: TRAIN MODEL 1 (MAX DEPTH = 2)
# ==========================================

model_1 = DecisionTreeClassifier(
    max_depth=2,
    random_state=42
)

model_1.fit(X_train, y_train)

predictions_1 = model_1.predict(X_test)

accuracy_1 = accuracy_score(y_test, predictions_1)

print("\n" + "=" * 50)
print("MODEL 1: MAX DEPTH = 2")
print("=" * 50)

print("Test accuracy:", round(accuracy_1, 3))
print("Test accuracy percentage:", round(accuracy_1 * 100, 2), "%")

# ==========================================
# STEP 6: DISPLAY ACTUAL VS PREDICTED
# ==========================================

results = X_test.copy()
results["actual"] = y_test
results["predicted"] = predictions_1

print("\nActual vs Predicted:")
print(results[["actual", "predicted"]])

# Show incorrect predictions
incorrect = results[
    results["actual"] != results["predicted"]
]

print("\nIncorrect predictions:", len(incorrect))
print("Correct predictions:", len(results) - len(incorrect))

# ==========================================
# STEP 7: TRAIN MODEL 2 (MAX DEPTH = 4)
# ==========================================

model_2 = DecisionTreeClassifier(
    max_depth=4,
    random_state=42
)

model_2.fit(X_train, y_train)

predictions_2 = model_2.predict(X_test)

accuracy_2 = accuracy_score(y_test, predictions_2)

print("\n" + "=" * 50)
print("MODEL 2: MAX DEPTH = 4")
print("=" * 50)

print("Test accuracy:", round(accuracy_2, 3))
print("Test accuracy percentage:", round(accuracy_2 * 100, 2), "%")

# ==========================================
# STEP 8: COMPARE BOTH MODELS
# ==========================================

print("\n" + "=" * 50)
print("MODEL COMPARISON")
print("=" * 50)

print(f"Model 1 (depth 2): {accuracy_1 * 100:.2f}%")
print(f"Model 2 (depth 4): {accuracy_2 * 100:.2f}%")

if accuracy_2 > accuracy_1:
    print("Model 2 performed better on the test set.")
elif accuracy_1 > accuracy_2:
    print("Model 1 performed better on the test set.")
else:
    print("Both models achieved the same test accuracy.")

difference = (accuracy_2 - accuracy_1) * 100
print(f"Accuracy difference: {difference:+.2f} percentage points")

# ==========================================
# STEP 9: OPTIONAL OVERFITTING EXPERIMENT
# ==========================================

model_3 = DecisionTreeClassifier(
    random_state=42
)

model_3.fit(X_train, y_train)

train_predictions = model_3.predict(X_train)
test_predictions = model_3.predict(X_test)

train_accuracy = accuracy_score(y_train, train_predictions)
test_accuracy = accuracy_score(y_test, test_predictions)

print("\n" + "=" * 50)
print("MODEL 3: UNLIMITED TREE DEPTH")
print("=" * 50)

print("Training accuracy:", round(train_accuracy, 3))
print("Training accuracy percentage:", round(train_accuracy * 100, 2), "%")

print("Test accuracy:", round(test_accuracy, 3))
print("Test accuracy percentage:", round(test_accuracy * 100, 2), "%")

if train_accuracy > test_accuracy:
    print("Training accuracy is higher than test accuracy.")
    print("This may indicate overfitting.")
else:
    print("Training accuracy is not higher than test accuracy.")

# ==========================================
# STEP 10: FINAL SUMMARY
# ==========================================

print("\n" + "=" * 50)
print("PROJECT COMPLETED")
print("=" * 50)

print("Dataset rows:", len(data))
print("Dataset columns:", len(data.columns))
print("Training records:", len(X_train))
print("Testing records:", len(X_test))

print("\nWorkflow:")
print("Problem -> Data -> Prepare -> Train")
print("-> Evaluate -> Improve -> Repeat")
