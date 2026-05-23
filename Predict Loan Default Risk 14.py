import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from sklearn.ensemble import RandomForestClassifier

# =========================
# LOAD DATASET
# =========================
# Replace with your dataset file
df = pd.read_csv("loan_default_dataset.csv")

# =========================
# DISPLAY FIRST ROWS
# =========================
print("Dataset Preview:\n")
print(df.head())

# =========================
# TARGET COLUMN
# =========================
# Replace 'Default' with your target column name
target_column = "Default"

# =========================
# SEPARATE FEATURES & TARGET
# =========================
X = df.drop(target_column, axis=1)
y = df[target_column]

# =========================
# ENCODE TARGET VARIABLE
# =========================
le = LabelEncoder()
y = le.fit_transform(y)

# =========================
# IDENTIFY COLUMN TYPES
# =========================
numeric_features = X.select_dtypes(include=['int64', 'float64']).columns
categorical_features = X.select_dtypes(include=['object']).columns

# =========================
# PREPROCESSING
# =========================
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent'))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# =========================
# ENCODE CATEGORICAL FEATURES
# =========================
for col in categorical_features:
    X[col] = LabelEncoder().fit_transform(X[col].astype(str))

# =========================
# TRAIN TEST SPLIT
# =========================
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

# =========================
# MODEL
# =========================
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

# =========================
# TRAIN MODEL
# =========================
model.fit(X_train, y_train)

# =========================
# PREDICTIONS
# =========================
y_pred = model.predict(X_test)

# =========================
# EVALUATION
# =========================
print("\nAccuracy Score:")
print(accuracy_score(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# =========================
# FEATURE IMPORTANCE
# =========================
importance = pd.DataFrame({
    'Feature': X.columns,
    'Importance': model.feature_importances_
})

importance = importance.sort_values(by='Importance', ascending=False)

print("\nFeature Importance:\n")
print(importance)

# =========================
# SAMPLE PREDICTION
# =========================
sample = X_test.iloc[:5]
prediction = model.predict(sample)

print("\nSample Predictions:")
print(prediction)