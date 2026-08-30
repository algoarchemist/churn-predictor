"""
Trains a real Random Forest churn model on the Telco Customer Churn dataset
(WA_Fn-UseC_-Telco-Customer-Churn.csv), following the cleaning/encoding steps
in churn-predictor-guide.md. No synthetic labels or synthetic features are
used anywhere in this pipeline - every row and every column comes from the
real dataset.

Saves rf_model_balanced.pkl containing the trained model, the fitted scaler,
and the exact post-encoding column order, so app.py can align any freshly
uploaded Telco-formatted CSV to the same feature space before scoring it.
"""
import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

DATA_PATH = "WA_Fn-UseC_-Telco-Customer-Churn.csv"

df = pd.read_csv(DATA_PATH)
print(f"Loaded real dataset: {df.shape[0]} customers, {df.shape[1]} columns")

df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
df["TotalCharges"] = df["TotalCharges"].fillna(df["TotalCharges"].median())

customer_ids = df["customerID"]
df = df.drop("customerID", axis=1)

binary_cols = ["Partner", "Dependents", "PhoneService", "PaperlessBilling", "Churn"]
for col in binary_cols:
    df[col] = df[col].map({"Yes": 1, "No": 0})

df["gender"] = df["gender"].map({"Male": 1, "Female": 0})

categorical_cols = df.select_dtypes(include="object").columns.tolist()
df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

X = df.drop("Churn", axis=1)
y = df["Churn"]
feature_columns = list(X.columns)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

rf_model_balanced = RandomForestClassifier(
    n_estimators=200, class_weight="balanced", random_state=42
)
rf_model_balanced.fit(X_train_scaled, y_train)

y_pred = rf_model_balanced.predict(X_test_scaled)
acc = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)

print(f"\nTest accuracy: {acc:.4f}")
print("\nClassification report:\n", report)
print("Confusion matrix:\n", cm)

bundle = {
    "model": rf_model_balanced,
    "scaler": scaler,
    "feature_columns": feature_columns,
    "test_accuracy": acc,
    "classification_report": report,
    "confusion_matrix": cm.tolist(),
}

with open("rf_model_balanced.pkl", "wb") as f:
    pickle.dump(bundle, f)

print("\nSaved rf_model_balanced.pkl (real model trained on real data)")
