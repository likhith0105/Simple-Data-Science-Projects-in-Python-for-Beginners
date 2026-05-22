# Used Car Price Prediction using Machine Learning

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

# Load dataset
df = pd.read_csv("used_cars.csv")

# Display first 5 rows
print(df.head())

# Convert categorical columns into numerical values
df = pd.get_dummies(df, drop_first=True)

# Define features and target
X = df.drop("Selling_Price", axis=1)
y = df["Selling_Price"]

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Evaluation
print("MAE:", mean_absolute_error(y_test, y_pred))
print("R2 Score:", r2_score(y_test, y_pred))

# Predict new car price
new_car = X.iloc[0:1]
prediction = model.predict(new_car)

print("Predicted Used Car Price:", prediction[0])