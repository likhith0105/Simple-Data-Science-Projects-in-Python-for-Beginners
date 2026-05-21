# ============================================
# LOAN DEFAULT RISK PREDICTION PROJECT
# ============================================

# Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier

# ============================================
# LOAD DATASET
# ============================================

# Replace with your CSV file name
df = pd.read_csv("loan_default.csv")

print(df.head())

# ============================================
# DATA PREPROCESSING
# ============================================

# Fill missing values
num_imputer = SimpleImputer(strategy='mean')

for col in df.select_dtypes(include=np.number).columns:
    df[col] = num_imputer.fit_transform(df[[col]])

# Encode categorical columns
le = LabelEncoder()

for col in df.select_dtypes(include='object').columns:
    df[col] = le.fit_transform(df[col].astype(str))

# ============================================
# FEATURES & TARGET
# ============================================

# Target column should be 0 or 1
# Example target column name: "default"

X = df.drop("default", axis=1)
y = df["default"]

# ============================================
# TRAIN TEST SPLIT
# ============================================

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

# ============================================
# MODEL TRAINING
# ============================================

model = RandomForestClassifier(
    n_estimators=200,
    random_state=42
)

model.fit(X_train, y_train)

# ============================================
# PREDICTIONS
# ============================================

y_pred = model.predict(X_test)

# ============================================
# EVALUATION
# ============================================

print("\nAccuracy:")
print(accuracy_score(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# ============================================
# FEATURE IMPORTANCE
# ============================================

importance = pd.DataFrame({
    'Feature': X.columns,
    'Importance': model.feature_importances_
})

importance = importance.sort_values(by='Importance', ascending=False)

print("\nTop Important Features:")
print(importance.head(10))

# ============================================
# VISUALIZATION
# ============================================

plt.figure(figsize=(10,6))
plt.bar(importance['Feature'][:10], importance['Importance'][:10])
plt.xticks(rotation=45)
plt.title("Top 10 Important Features")
plt.tight_layout()
plt.show()

# ============================================
# SAVE PREDICTIONS
# ============================================

results = pd.DataFrame({
    "Actual": y_test,
    "Predicted": y_pred
})

results.to_csv("loan_default_predictions.csv", index=False)

print("\nPrediction file saved as loan_default_predictions.csv")