# ==============================
# EMPLOYEE SALARY PREDICTION
# ==============================

# 1. Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# 2. Load Dataset
df = pd.read_csv("salary_data.csv")

# 3. Display Data
print(df.head())
print(df.info())

# 4. Data Cleaning

# Drop unnecessary column (if exists)
if 'EmployeeID' in df.columns:
    df.drop('EmployeeID', axis=1, inplace=True)

# Handle missing values
df.dropna(inplace=True)

# 5. Encode Categorical Data
le = LabelEncoder()
for col in df.select_dtypes(include='object').columns:
    df[col] = le.fit_transform(df[col])

# 6. EDA (Optional)
if 'Salary' in df.columns:
    plt.figure(figsize=(5,4))
    sns.histplot(df['Salary'], kde=True)
    plt.title("Salary Distribution")
    plt.show()

# 7. Split Data
X = df.drop('Salary', axis=1)
y = df['Salary']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

# 8. Feature Scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 9. Train Model (Linear Regression)
model = LinearRegression()
model.fit(X_train, y_train)

# 10. Prediction
y_pred = model.predict(X_test)

# 11. Evaluation
print("\n=== Linear Regression Results ===")
print("R2 Score:", r2_score(y_test, y_pred))
print("MSE:", mean_squared_error(y_test, y_pred))

# 12. Plot Actual vs Predicted
plt.figure(figsize=(6,4))
plt.scatter(y_test, y_pred)
plt.xlabel("Actual Salary")
plt.ylabel("Predicted Salary")
plt.title("Actual vs Predicted Salary")
plt.show()