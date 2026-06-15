import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
from xgboost import XGBRegressor

# Load datasets
train = pd.read_csv("train.csv")
store = pd.read_csv("store.csv")

# Merge datasets
data = pd.merge(train, store, on="Store", how="left")

# Convert Date column
data['Date'] = pd.to_datetime(data['Date'])

# Feature Engineering
data['Year'] = data['Date'].dt.year
data['Month'] = data['Date'].dt.month
data['Day'] = data['Date'].dt.day
data['WeekOfYear'] = data['Date'].dt.isocalendar().week.astype(int)

# Fill missing values
data.fillna(0, inplace=True)

# Convert categorical columns to numeric
categorical_cols = ['StoreType', 'Assortment', 'StateHoliday']

for col in categorical_cols:
    data[col] = data[col].astype('category').cat.codes

# Select features
features = [
    'Store', 'DayOfWeek', 'Open', 'Promo',
    'StateHoliday', 'SchoolHoliday',
    'StoreType', 'Assortment',
    'CompetitionDistance',
    'Year', 'Month', 'Day', 'WeekOfYear'
]

X = data[features]
y = data['Sales']

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train XGBoost Model
model = XGBRegressor(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=8,
    random_state=42
)

model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Evaluation
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Mean Absolute Error:", mae)
print("R2 Score:", r2)

# Sample Predictions
results = pd.DataFrame({
    'Actual Sales': y_test.values,
    'Predicted Sales': y_pred
})

print(results.head())

# Save predictions
results.to_csv("rossmann_predictions.csv", index=False)

print("Predictions saved successfully!")