# ==========================================
# STOCK PRICE PREDICTION USING LSTM
# ==========================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM

# ==========================================
# LOAD DATASET
# ==========================================

# CSV file should contain a column named 'Close'
df = pd.read_csv("stock_data.csv")

data = df['Close'].values
data = data.reshape(-1, 1)

# ==========================================
# NORMALIZE DATA
# ==========================================

scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data)

# ==========================================
# CREATE DATASET
# ==========================================

X = []
y = []

time_step = 60

for i in range(time_step, len(scaled_data)):
    X.append(scaled_data[i-time_step:i, 0])
    y.append(scaled_data[i, 0])

X = np.array(X)
y = np.array(y)

# Reshape for LSTM
X = X.reshape(X.shape[0], X.shape[1], 1)

# ==========================================
# TRAIN TEST SPLIT
# ==========================================

train_size = int(len(X) * 0.8)

X_train = X[:train_size]
X_test = X[train_size:]

y_train = y[:train_size]
y_test = y[train_size:]

# ==========================================
# BUILD LSTM MODEL
# ==========================================

model = Sequential()

model.add(LSTM(units=50,
               return_sequences=True,
               input_shape=(X_train.shape[1], 1)))

model.add(LSTM(units=50))

model.add(Dense(units=1))

# ==========================================
# COMPILE MODEL
# ==========================================

model.compile(optimizer='adam',
              loss='mean_squared_error')

# ==========================================
# TRAIN MODEL
# ==========================================

model.fit(X_train,
          y_train,
          epochs=10,
          batch_size=32)

# ==========================================
# PREDICTIONS
# ==========================================

train_predict = model.predict(X_train)
test_predict = model.predict(X_test)

# Inverse transform
train_predict = scaler.inverse_transform(train_predict)
test_predict = scaler.inverse_transform(test_predict)

# ==========================================
# ACTUAL DATA
# ==========================================

actual_data = scaler.inverse_transform(
    scaled_data[time_step:]
)

# ==========================================
# PLOT RESULTS
# ==========================================

plt.figure(figsize=(12,6))

plt.plot(actual_data, label='Actual Price')

plt.plot(range(len(train_predict)),
         train_predict,
         label='Train Prediction')

plt.plot(range(len(train_predict),
               len(train_predict) + len(test_predict)),
         test_predict,
         label='Test Prediction')

plt.title("Stock Price Prediction Using LSTM")
plt.xlabel("Time")
plt.ylabel("Stock Price")
plt.legend()

plt.show()

# ==========================================
# FUTURE PREDICTION
# ==========================================

last_60_days = scaled_data[-60:]
future_input = last_60_days.reshape(1, 60, 1)

future_price = model.predict(future_input)

future_price = scaler.inverse_transform(future_price)

print("Next Day Predicted Stock Price:")
print(future_price)