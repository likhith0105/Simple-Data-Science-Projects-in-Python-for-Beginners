# Age of Abalone Shells Data Analysis

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Load Dataset
df = pd.read_csv("abalone.csv")

# Display basic information
print("Dataset Shape:", df.shape)
print("\nFirst 5 Rows:")
print(df.head())

print("\nDataset Information:")
print(df.info())

print("\nStatistical Summary:")
print(df.describe())

# Check Missing Values
print("\nMissing Values:")
print(df.isnull().sum())

# Encode Categorical Feature
le = LabelEncoder()
df['Sex'] = le.fit_transform(df['Sex'])

# Create Age Column
df['Age'] = df['Rings'] + 1.5

# -----------------------------
# Exploratory Data Analysis
# -----------------------------

# Correlation Heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
plt.title("Correlation Heatmap")
plt.show()

# Age Distribution
plt.figure(figsize=(8, 5))
sns.histplot(df['Age'], bins=20, kde=True)
plt.title("Age Distribution")
plt.xlabel("Age")
plt.show()

# Pair Plot
sns.pairplot(df[['Length', 'Diameter', 'Height',
                 'Whole_weight', 'Age']])
plt.show()

# -----------------------------
# Feature Selection
# -----------------------------

X = df.drop(['Rings', 'Age'], axis=1)
y = df['Age']

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# -----------------------------
# Model Training
# -----------------------------

rf = RandomForestRegressor(
    n_estimators=200,
    random_state=42
)

rf.fit(X_train, y_train)

# Predictions
y_pred = rf.predict(X_test)

# -----------------------------
# Model Evaluation
# -----------------------------

mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print("\nModel Performance")
print("------------------")
print("MAE :", round(mae, 3))
print("RMSE:", round(rmse, 3))
print("R² Score:", round(r2, 3))

# -----------------------------
# Feature Importance
# -----------------------------

importance = pd.DataFrame({
    'Feature': X.columns,
    'Importance': rf.feature_importances_
})

importance = importance.sort_values(
    by='Importance',
    ascending=False
)

print("\nFeature Importance:")
print(importance)

plt.figure(figsize=(8, 5))
sns.barplot(
    x='Importance',
    y='Feature',
    data=importance
)
plt.title("Feature Importance")
plt.show()

# -----------------------------
# Predict New Abalone Age
# -----------------------------

sample = [[
    1,      # Sex (Encoded)
    0.55,   # Length
    0.42,   # Diameter
    0.14,   # Height
    0.85,   # Whole_weight
    0.38,   # Shucked_weight
    0.18,   # Viscera_weight
    0.25    # Shell_weight
]]

predicted_age = rf.predict(sample)

print("\nPredicted Age of New Abalone:",
      round(predicted_age[0], 2), "years")

# -----------------------------
# Conclusion
# -----------------------------
print("""
Conclusion:
1. The Abalone dataset was analyzed and preprocessed.
2. Age was calculated using Age = Rings + 1.5.
3. Random Forest Regressor was trained to predict age.
4. Model performance was evaluated using MAE, RMSE, and R² score.
5. Shell dimensions and weight-related features were found to be important predictors of age.
""")