# Forecast Traffic Volume on Highways using LSTM

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

# -----------------------------
# LOAD DATASET
# -----------------------------
# Replace with your dataset file
df = pd.read_csv("traffic.csv")

print(df.head())

# -----------------------------
# PREPROCESSING
# -----------------------------

# Convert date column
df['date_time'] = pd.to_datetime(df['date_time'])

# Sort by date
df = df.sort_values('date_time')

# Select traffic volume column
data = df[['traffic_volume']]

# Normalize data
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data)

# -----------------------------
# CREATE SEQUENCES
# -----------------------------

sequence_length = 24

X = []
y = []

for i in range(sequence_length, len(scaled_data)):
    X.append(scaled_data[i-sequence_length:i, 0])
    y.append(scaled_data[i, 0])

X = np.array(X)
y = np.array(y)

# Reshape for LSTM
X = np.reshape(X, (X.shape[0], X.shape[1], 1))

# -----------------------------
# TRAIN TEST SPLIT
# -----------------------------

split = int(0.8 * len(X))

X_train = X[:split]
X_test = X[split:]

y_train = y[:split]
y_test = y[split:]

# -----------------------------
# BUILD LSTM MODEL
# -----------------------------

model = Sequential()

model.add(LSTM(units=64, return_sequences=True,
               input_shape=(X_train.shape[1], 1)))

model.add(Dropout(0.2))

model.add(LSTM(units=64))

model.add(Dropout(0.2))

model.add(Dense(1))

model.compile(optimizer='adam', loss='mean_squared_error')

# -----------------------------
# TRAIN MODEL
# -----------------------------

history = model.fit(
    X_train,
    y_train,
    epochs=20,
    batch_size=32,
    validation_data=(X_test, y_test)
)

# -----------------------------
# PREDICTIONS
# -----------------------------

predictions = model.predict(X_test)

# Inverse transform
predictions = scaler.inverse_transform(predictions.reshape(-1, 1))
y_test_actual = scaler.inverse_transform(y_test.reshape(-1, 1))

# -----------------------------
# EVALUATION
# -----------------------------

rmse = np.sqrt(mean_squared_error(y_test_actual, predictions))
mae = mean_absolute_error(y_test_actual, predictions)

print("\nRMSE:", rmse)
print("MAE :", mae)

# -----------------------------
# VISUALIZATION
# -----------------------------

plt.figure(figsize=(12,6))

plt.plot(y_test_actual, label='Actual Traffic Volume')
plt.plot(predictions, label='Predicted Traffic Volume')

plt.title('Traffic Volume Prediction')
plt.xlabel('Time')
plt.ylabel('Traffic Volume')

plt.legend()
plt.show()