# Forecast Traffic Volume on Highways

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Load Dataset
# Replace with your dataset path
df = pd.read_csv("Metro_Interstate_Traffic_Volume.csv")

# Display first rows
print(df.head())

# Check missing values
print(df.isnull().sum())

# Fill missing values
df.fillna(method='ffill', inplace=True)

# Convert date_time column
df['date_time'] = pd.to_datetime(df['date_time'])

# Feature Engineering
df['year'] = df['date_time'].dt.year
df['month'] = df['date_time'].dt.month
df['day'] = df['date_time'].dt.day
df['hour'] = df['date_time'].dt.hour
df['day_of_week'] = df['date_time'].dt.dayofweek

# Encode categorical columns
le_holiday = LabelEncoder()
le_weather_main = LabelEncoder()
le_weather_desc = LabelEncoder()

df['holiday'] = le_holiday.fit_transform(df['holiday'])
df['weather_main'] = le_weather_main.fit_transform(df['weather_main'])
df['weather_description'] = le_weather_desc.fit_transform(
    df['weather_description']
)

# Features and Target
X = df.drop(['traffic_volume', 'date_time'], axis=1)
y = df['traffic_volume']

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Model Training
model = RandomForestRegressor(
    n_estimators=200,
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)

# Prediction
y_pred = model.predict(X_test)

# Evaluation
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print("\nModel Performance")
print("-------------------")
print("MAE :", mae)
print("MSE :", mse)
print("RMSE:", rmse)
print("R² Score:", r2)

# Feature Importance
importance = pd.DataFrame({
    'Feature': X.columns,
    'Importance': model.feature_importances_
})

importance = importance.sort_values(
    by='Importance',
    ascending=False
)

print("\nFeature Importance")
print(importance)

# Plot Feature Importance
plt.figure(figsize=(10, 6))
plt.barh(
    importance['Feature'],
    importance['Importance']
)
plt.title("Feature Importance")
plt.xlabel("Importance Score")
plt.tight_layout()
plt.show()

# Actual vs Predicted
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, alpha=0.5)
plt.xlabel("Actual Traffic Volume")
plt.ylabel("Predicted Traffic Volume")
plt.title("Actual vs Predicted")
plt.show()

# Sample Forecast
sample = X.iloc[[0]]
prediction = model.predict(sample)

print("\nPredicted Traffic Volume:", int(prediction[0]))