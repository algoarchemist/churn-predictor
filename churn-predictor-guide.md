# E-Commerce / Telecom Customer Churn Predictor — Full Beginner Walkthrough

This guide takes you from zero to a working churn-prediction model, using the exact dataset, tools, and code you need. No prior ML experience assumed.

---

## Step 1: Set Up Your Environment (No Installation Needed)

As a complete beginner, skip installing Python locally for now. Use **Google Colab** — it's free, runs in your browser, and comes with all the libraries you need (pandas, scikit-learn, matplotlib) pre-installed.

1. Go to **https://colab.research.google.com**
2. Sign in with a Google account
3. Click **File → New notebook**
4. You'll see a code cell — that's where all the code below goes. Run a cell with `Shift + Enter`.

(Alternative for later: once comfortable, you can install **Anaconda** — https://www.anaconda.com/download — which gives you Jupyter Notebook on your own machine.)

---

## Step 2: Get the Dataset

Use the classic, most-used version of this dataset on Kaggle:

**https://www.kaggle.com/datasets/blastchar/telco-customer-churn**

Steps:
1. Click the link above (make a free Kaggle account if you don't have one).
2. Click the **Download** button — you'll get a file called `WA_Fn-UseC_-Telco-Customer-Churn.csv`.
3. Go back to your Colab notebook. On the left sidebar, click the **folder icon** → the **upload icon** (page with an up arrow) → select the CSV you just downloaded.

This dataset has **7,043 customers** with columns like `tenure`, `MonthlyCharges`, `TotalCharges`, `Contract`, `TechSupport`, and the target column `Churn` (Yes/No). It's a telecom dataset, but everything here applies directly to e-commerce churn (subscription cancellations, inactive shoppers, etc.) — I'll show you how to adapt it at the end.

---

## Step 3: Load and Clean the Data

Paste this into a new cell:

```python
import pandas as pd
import numpy as np

# Load data
df = pd.read_csv('WA_Fn-UseC_-Telco-Customer-Churn.csv')

# Quick look
print(df.shape)
df.head()
```

### Fix "TotalCharges" (it's stored as text with some blank spaces)

```python
# Some TotalCharges values are blank strings (' '), not real NaN — this breaks conversion
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')

# Fill the few missing values with the median
df['TotalCharges'] = df['TotalCharges'].fillna(df['TotalCharges'].median())
```

### Convert Yes/No and categorical text into numbers

```python
# Drop customer ID — it's not predictive
df = df.drop('customerID', axis=1)

# Convert simple Yes/No columns to 1/0
binary_cols = ['Partner', 'Dependents', 'PhoneService', 'PaperlessBilling', 'Churn']
for col in binary_cols:
    df[col] = df[col].map({'Yes': 1, 'No': 0})

# Gender to 1/0
df['gender'] = df['gender'].map({'Male': 1, 'Female': 0})

# For multi-category columns (Contract, InternetService, PaymentMethod, etc.)
# use one-hot encoding — this turns each category into its own 0/1 column
categorical_cols = df.select_dtypes(include='object').columns.tolist()
df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

df.head()
```

At this point every column is numeric, and `Churn` is your target (1 = churned, 0 = stayed).

---

## Step 4: Exploratory Data Analysis (EDA)

### Pie chart — overall churn split

```python
import matplotlib.pyplot as plt

churn_counts = df['Churn'].value_counts()
labels = ['Stayed', 'Churned']

plt.figure(figsize=(6,6))
plt.pie(churn_counts, labels=labels, autopct='%1.1f%%', colors=['#4CAF50', '#F44336'], startangle=90)
plt.title('Customer Churn Distribution')
plt.show()
```

You'll see roughly **73% stayed / 27% churned** — this imbalance is exactly why Step 7 (class_weight tuning) matters.

### Bar chart — Contract type vs churn rate

Since we one-hot-encoded `Contract` already, let's redo this specific chart *before* dummy variables destroy the readable labels — or just reload a fresh copy for plotting purposes:

```python
df_raw = pd.read_csv('WA_Fn-UseC_-Telco-Customer-Churn.csv')
df_raw['Churn_binary'] = df_raw['Churn'].map({'Yes': 1, 'No': 0})

contract_churn = df_raw.groupby('Contract')['Churn_binary'].mean() * 100

plt.figure(figsize=(7,5))
contract_churn.plot(kind='bar', color=['#F44336', '#FF9800', '#4CAF50'])
plt.title('Churn Rate by Contract Type')
plt.ylabel('Churn Rate (%)')
plt.xlabel('Contract Type')
plt.xticks(rotation=0)
plt.show()
```

You'll confirm the classic pattern: **month-to-month contracts churn around 40%+**, while two-year contracts churn under 5%. This alone is a powerful business insight — it tells a company to push customers toward longer contracts.

---

## Step 5: Split Data and Build Your First Model (Logistic Regression)

```python
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

# Separate features (X) from target (y)
X = df.drop('Churn', axis=1)
y = df['Churn']

# Split into training (80%) and testing (20%) sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Scale numeric features so large numbers (like TotalCharges) don't dominate
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train Logistic Regression
log_model = LogisticRegression(max_iter=1000)
log_model.fit(X_train_scaled, y_train)

y_pred_log = log_model.predict(X_test_scaled)
```

---

## Step 6: Evaluate — Accuracy AND Confusion Matrix

Accuracy alone is misleading here because of the class imbalance (a model that just predicts "everyone stays" would already score ~73% accuracy while being useless). The **confusion matrix** is what actually matters.

```python
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import seaborn as sns

print("Accuracy:", accuracy_score(y_test, y_pred_log))
print("\nClassification Report:\n", classification_report(y_test, y_pred_log))

cm = confusion_matrix(y_test, y_pred_log)
plt.figure(figsize=(5,4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Predicted Stay', 'Predicted Churn'],
            yticklabels=['Actual Stay', 'Actual Churn'])
plt.title('Confusion Matrix — Logistic Regression')
plt.show()
```

**What to look for:** the bottom-left cell (Actual Churn, Predicted Stay) is your **False Negatives** — customers the model told you were safe, but who actually left. This is the costliest mistake in churn prediction, because the business never intervened. Your goal in the next step is to shrink that number.

---

## Step 7: Try Random Forest

```python
from sklearn.ensemble import RandomForestClassifier

rf_model = RandomForestClassifier(n_estimators=200, random_state=42)
rf_model.fit(X_train_scaled, y_train)

y_pred_rf = rf_model.predict(X_test_scaled)

print("Random Forest Accuracy:", accuracy_score(y_test, y_pred_rf))
print(classification_report(y_test, y_pred_rf))
```

Random Forest often edges out Logistic Regression slightly, and also gives you **feature importance** — which factors matter most:

```python
importances = pd.Series(rf_model.feature_importances_, index=X.columns)
importances.sort_values(ascending=False).head(10).plot(kind='barh', figsize=(8,6))
plt.title('Top 10 Features Driving Churn')
plt.gca().invert_yaxis()
plt.show()
```

---

## Step 8: Fine-Tune with `class_weight` (the key step for minimizing False Negatives)

Because churners are the minority class, tell the model to penalize missing them more heavily:

```python
# Logistic Regression with balanced class weights
log_model_balanced = LogisticRegression(max_iter=1000, class_weight='balanced')
log_model_balanced.fit(X_train_scaled, y_train)
y_pred_log_bal = log_model_balanced.predict(X_test_scaled)

print("Balanced Logistic Regression:")
print(classification_report(y_test, y_pred_log_bal))

cm_bal = confusion_matrix(y_test, y_pred_log_bal)
sns.heatmap(cm_bal, annot=True, fmt='d', cmap='Oranges',
            xticklabels=['Predicted Stay', 'Predicted Churn'],
            yticklabels=['Actual Stay', 'Actual Churn'])
plt.title('Confusion Matrix — Balanced Logistic Regression')
plt.show()
```

```python
# Random Forest with balanced class weights
rf_model_balanced = RandomForestClassifier(n_estimators=200, class_weight='balanced', random_state=42)
rf_model_balanced.fit(X_train_scaled, y_train)
y_pred_rf_bal = rf_model_balanced.predict(X_test_scaled)

print("Balanced Random Forest:")
print(classification_report(y_test, y_pred_rf_bal))
```

**What you'll observe:** overall accuracy may drop slightly, but **recall on the churn class (1) goes up** — meaning you catch more actual churners, at the cost of a few more false alarms (False Positives). For a retention team, that trade is almost always worth it: emailing a customer a discount they didn't need costs little; losing a customer you never flagged costs a lot.

You can also try custom weights instead of `'balanced'`, e.g. `class_weight={0: 1, 1: 3}` to weight churn 3x as important, and tune this by hand.

---

## Step 9: Applying This to Amazon, Flipkart, and Real E-Commerce

The telecom dataset is a stand-in — the same pipeline maps directly onto e-commerce. Here's the translation:

| Telecom Feature | E-Commerce Equivalent |
|---|---|
| `tenure` (months with company) | Days/months since first purchase |
| `Contract` (month-to-month vs 2yr) | Subscription plan type (Prime monthly vs annual) |
| `MonthlyCharges` | Average order value / monthly spend |
| `TechSupport` usage | Customer support tickets raised |
| `InternetService` type | Delivery/membership tier |
| `PaymentMethod` | Payment method (COD vs card vs wallet) |
| `Churn` | Did they stop ordering / cancel subscription in the last N months? |

**How Amazon/Flipkart-style companies actually build this in production:**

1. **Define churn concretely.** For e-commerce there's no explicit "cancel" button like a subscription — so churn is usually defined behaviorally, e.g. "no purchase in the last 90 days" for a previously active customer. This is the hardest and most important step — get this definition wrong and the whole model is wrong.
2. **Feature engineering from behavioral logs** (this is the real e-commerce equivalent of the telecom columns):
   - Recency, Frequency, Monetary value (RFM) — days since last order, order count, total spend
   - Cart abandonment rate
   - Return/refund rate
   - App session frequency, time between sessions
   - Customer support contact frequency
   - Discount/coupon usage rate
   - Category diversity of purchases
3. **Same modeling pipeline as above** — Logistic Regression as a baseline, Random Forest (or in production, **XGBoost/LightGBM**) for better performance, with `class_weight` or `scale_pos_weight` tuning for the imbalance.
4. **Output feeds a retention system**, not just a report: customers scored as high-risk get routed into automated interventions — a discount email, a personalized recommendation push, a customer-support outreach call, or a loyalty-point bonus.
5. **Continuous retraining.** Real companies retrain these models weekly/monthly as new order data comes in, and track model performance drift over time.

If you want to practice this exact e-commerce framing (rather than telecom), search Kaggle for **"E-Commerce Customer Churn"** — there's a dataset from an e-commerce company at:
**https://www.kaggle.com/datasets/ankitverma2010/ecommerce-customer-churn-analysis-and-prediction**
It has RFM-style features (`Tenure`, `OrderCount`, `CashbackAmount`, `SatisfactionScore`, `Complain`) and works with the identical code above with minor column-name changes.

---

## Reference Links

- Google Colab: https://colab.research.google.com
- Kaggle account signup: https://www.kaggle.com/account/login
- Telco Churn dataset: https://www.kaggle.com/datasets/blastchar/telco-customer-churn
- E-commerce Churn dataset: https://www.kaggle.com/datasets/ankitverma2010/ecommerce-customer-churn-analysis-and-prediction
- scikit-learn docs (Logistic Regression): https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html
- scikit-learn docs (Random Forest): https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html
- pandas `get_dummies` docs: https://pandas.pydata.org/docs/reference/api/pandas.get_dummies.html

---

## Quick Troubleshooting

- **"FileNotFoundError" when reading CSV** → make sure the file is uploaded to Colab's session (folder icon on the left) and the filename matches exactly.
- **"ValueError: could not convert string to float"** → you likely have a column still holding text; run `df.dtypes` to check, and re-check the `TotalCharges` cleaning step.
- **Colab session resets and your upload disappears** → Colab wipes uploaded files when the runtime disconnects (e.g. after a period of inactivity). Re-upload, or better, mount Google Drive (`from google.colab import drive; drive.mount('/content/drive')`) and keep the CSV there permanently.
