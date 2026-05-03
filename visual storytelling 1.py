import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# 1. Load Data
df = sns.load_dataset('diamonds')

# 2. Data Cleaning: Remove rows where dimensions (x, y, or z) are 0
df = df[(df[['x', 'y', 'z']] != 0).all(axis=1)]

# 3. Preprocessing: Encoding Categorical Variables
# Mapping ordinal categories to maintain their rank order
cut_map = {'Fair': 1, 'Good': 2, 'Very Good': 3, 'Premium': 4, 'Ideal': 5}
color_map = {'J': 1, 'I': 2, 'H': 3, 'G': 4, 'F': 5, 'E': 6, 'D': 7}
clarity_map = {'I1': 1, 'SI2': 2, 'SI1': 3, 'VS2': 4, 'VS1': 5, 'VVS2': 6, 'VVS1': 7, 'IF': 8}

df['cut'] = df['cut'].map(cut_map)
df['color'] = df['color'].map(color_map)
df['clarity'] = df['clarity'].map(clarity_map)

# 4. Define Features and Target
X = df.drop('price', axis=1)
y = df['price']

# 5. Split Data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 6. Initialize and Train Model
model = RandomForestRegressor(n_estimators=100, n_jobs=-1, random_state=42)
model.fit(X_train, y_train)

# 7. Evaluation & Metrics
y_pred = model.predict(X_test)
r2 = r2_score(y_test, y_pred)
rmse = mean_squared_error(y_test, y_pred) ** 0.5

print(f"Model Performance:")
print(f"R^2 Score: {r2:.4f}")
print(f"RMSE: ${rmse:.2f}")

# 8. Visualization: Feature Importance
plt.figure(figsize=(10, 6))
feat_importances = pd.Series(model.feature_importances_, index=X.columns)
feat_importances.nlargest(10).plot(kind='barh')
plt.title('Top Factors Influencing Diamond Price')
plt.xlabel('Relative Importance Score')
plt.show()