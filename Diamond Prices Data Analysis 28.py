# ============================================================
# DIAMOND PRICES DATA ANALYSIS AND PRICE PREDICTION
# ============================================================

# Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ============================================================
# Load Dataset
# ============================================================

df = pd.read_csv("diamonds.csv")

print("First 5 Records:")
print(df.head())

print("\nDataset Information:")
print(df.info())

print("\nStatistical Summary:")
print(df.describe())

# ============================================================
# Data Cleaning
# ============================================================

print("\nMissing Values:")
print(df.isnull().sum())

df.drop_duplicates(inplace=True)

print("\nDataset Shape After Removing Duplicates:")
print(df.shape)

# ============================================================
# Exploratory Data Analysis (EDA)
# ============================================================

# Distribution of Prices
plt.figure(figsize=(8,5))
sns.histplot(df['price'], bins=50, kde=True)
plt.title("Distribution of Diamond Prices")
plt.xlabel("Price")
plt.ylabel("Frequency")
plt.show()

# Carat vs Price
plt.figure(figsize=(8,5))
sns.scatterplot(x='carat', y='price', data=df)
plt.title("Carat vs Price")
plt.show()

# Average Price by Cut
plt.figure(figsize=(8,5))
sns.barplot(x='cut', y='price', data=df)
plt.title("Average Price by Cut")
plt.xticks(rotation=45)
plt.show()

# Average Price by Color
plt.figure(figsize=(8,5))
sns.barplot(x='color', y='price', data=df)
plt.title("Average Price by Color")
plt.show()

# Average Price by Clarity
plt.figure(figsize=(10,5))
sns.barplot(x='clarity', y='price', data=df)
plt.title("Average Price by Clarity")
plt.show()

# ============================================================
# Correlation Matrix
# ============================================================

numeric_df = df.select_dtypes(include=['number'])

plt.figure(figsize=(10,8))
sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm')
plt.title("Correlation Matrix")
plt.show()

# ============================================================
# Top 10 Most Expensive Diamonds
# ============================================================

top10 = df.nlargest(10, 'price')

print("\nTop 10 Most Expensive Diamonds:")
print(top10[['carat', 'cut', 'color', 'clarity', 'price']])

# ============================================================
# Encoding Categorical Features
# ============================================================

encoder = LabelEncoder()

for col in ['cut', 'color', 'clarity']:
    df[col] = encoder.fit_transform(df[col])

# ============================================================
# Feature Selection
# ============================================================

X = df.drop('price', axis=1)
y = df['price']

# ============================================================
# Train-Test Split
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.20,
    random_state=42
)

# ============================================================
# Model Building - Random Forest Regressor
# ============================================================

model = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)

model.fit(X_train, y_train)

# ============================================================
# Prediction
# ============================================================

y_pred = model.predict(X_test)

# ============================================================
# Model Evaluation
# ============================================================

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print("\nModel Performance")
print("MAE :", mae)
print("MSE :", mse)
print("RMSE:", rmse)
print("R2 Score:", r2)

# ============================================================
# Feature Importance
# ============================================================

importance = pd.DataFrame({
    'Feature': X.columns,
    'Importance': model.feature_importances_
})

importance = importance.sort_values(
    by='Importance',
    ascending=False
)

print("\nFeature Importance:")
print(importance)

plt.figure(figsize=(8,5))
sns.barplot(
    x='Importance',
    y='Feature',
    data=importance
)
plt.title("Feature Importance")
plt.show()

# ============================================================
# Conclusion
# ============================================================

print("\nKey Insights:")
print("1. Carat is the most important factor affecting diamond price.")
print("2. Higher clarity and better cut grades generally increase price.")
print("3. Diamond price rises significantly with carat weight.")
print("4. Random Forest provides accurate diamond price predictions.")
print("5. Feature importance helps identify key pricing factors.")