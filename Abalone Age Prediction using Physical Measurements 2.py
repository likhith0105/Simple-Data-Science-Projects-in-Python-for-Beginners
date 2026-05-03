import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

# 1. Load Data from UCI Repository
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/abalone/abalone.data"
columns = ["Sex", "Length", "Diameter", "Height", "Whole_weight", 
           "Shucked_weight", "Viscera_weight", "Shell_weight", "Rings"]
df = pd.read_csv(url, names=columns)

# 2. Data Cleaning
# Remove rows where Height is 0 (physically impossible)
df = df[df['Height'] > 0]

# 3. Preprocessing
# Convert categorical 'Sex' (M, F, I) into numerical dummy variables
df = pd.get_dummies(df, columns=['Sex'], drop_first=True)

# Define Features (X) and Target (y)
# Age is typically Rings + 1.5, but we predict Rings directly
X = df.drop('Rings', axis=1)
y = df['Rings']

# 4. Split Data (80% Train, 20% Test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 5. Train Model
# RandomForest is great for capturing non-linear biological growth patterns
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 6. Evaluation
predictions = model.predict(X_test)
print(f"R-Squared Score: {r2_score(y_test, predictions):.4f}")
print(f"Mean Absolute Error: {mean_absolute_error(y_test, predictions):.2f} rings")

# 7. Visualization
plt.figure(figsize=(10, 5))

# Plot Actual vs Predicted
plt.subplot(1, 2, 1)
plt.scatter(y_test, predictions, alpha=0.3, color='teal')
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--', lw=2)
plt.xlabel('Actual Rings')
plt.ylabel('Predicted Rings')
plt.title('Actual vs Predicted Age')

# Plot Feature Importance
plt.subplot(1, 2, 2)
feat_importances = pd.Series(model.feature_importances_, index=X.columns)
feat_importances.nlargest(5).plot(kind='barh', color='skyblue')
plt.title('Top Factors for Age Prediction')

plt.tight_layout()
plt.show()