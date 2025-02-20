"""
Univariate Forecasting with LSTM model
Updated for Python 3.12.9
"""

import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from pathlib import Path

# Convert time series data into supervised learning format
def timeseries_to_supervised(data, lag=1):
    df = pd.DataFrame(data)
    columns = [df.shift(i) for i in range(1, lag + 1)]
    columns.append(df)
    df = pd.concat(columns, axis=1)
    df.fillna(0, inplace=True)
    return df

# Convert to differenced series
def difference(dataset, interval=1):
    return pd.Series([dataset[i] - dataset[i - interval] for i in range(interval, len(dataset))])

# Revert differencing
def inverse_difference(history, yhat, interval=1):
    return yhat + history[-interval]

# Normalize data
def scale(train, test):
    scaler = MinMaxScaler(feature_range=(-1, 1))
    scaler.fit(train)
    return scaler, scaler.transform(train), scaler.transform(test)

# Invert scaling
def invert_scale(scaler, X, value):
    array = np.array([*X, value]).reshape(1, -1)
    return scaler.inverse_transform(array)[0, -1]

# Create and train LSTM model
def fit_lstm(train, batch_size, epochs, neurons):
    X, y = train[:, :-1], train[:, -1]
    X = X.reshape(X.shape[0], 1, X.shape[1])
    
    model = Sequential([
        LSTM(neurons, input_shape=(X.shape[1], X.shape[2]), return_sequences=False),
        Dense(1)
    ])
    
    model.compile(loss='mean_squared_error', optimizer='adam')
    model.fit(X, y, epochs=epochs, batch_size=batch_size, verbose=1, shuffle=False)
    return model

# Forecast using LSTM model
def forecast_lstm(model, X):
    X = X.reshape(1, 1, len(X))
    return model.predict(X)[0, 0]

#### INITIALIZATION ####
filename = Path("dataset/Bangkok_solarpv_Trial.csv")
horizon = 372
neurons = 10
epochs = 100
batch_size = 1

# Load dataset
print("\n======= Univariate Forecasting in progress... =======\n")
series = pd.read_csv(filename, header=0, parse_dates=[0], index_col=0)

# Prepare data
raw_values = series.values.flatten()
diff_values = difference(raw_values, 1)
supervised = timeseries_to_supervised(diff_values, 1)
supervised_values = supervised.values

# Split data
train, test = supervised_values[:-horizon], supervised_values[-horizon:]
scaler, train_scaled, test_scaled = scale(train, test)

# Train LSTM
start_time = time.time()
lstm_model = fit_lstm(train_scaled, batch_size, epochs, neurons)

# Predict training data
train_reshaped = train_scaled[:, 0].reshape(len(train_scaled), 1, 1)
lstm_model.predict(train_reshaped)

# Forecast
predictions = []
for i in range(len(test_scaled)):
    X, y = test_scaled[i, :-1], test_scaled[i, -1]
    yhat = forecast_lstm(lstm_model, X)
    yhat = invert_scale(scaler, X, yhat)
    yhat = inverse_difference(raw_values, yhat, len(test_scaled) + 1 - i)
    predictions.append(yhat)
    expected = raw_values[len(train) + i + 1]
    print(f"Point={i+1}, Predicted={yhat:.3f}, Expected={expected:.3f}")

# Time calculation
elapsed_time = time.time() - start_time
print(f"\nProcessing Time: {elapsed_time:.2f} seconds")

# RMSE calculation
rmse = np.sqrt(mean_squared_error(raw_values[-horizon:], predictions))
print(f"Test RMSE: {rmse:.3f}")

# Plot results
plt.figure(figsize=(10, 5))
plt.plot(raw_values[-horizon:], label="Actual")
plt.plot(predictions, label="Forecast", linestyle="dashed")
plt.title("Bangkok Solar PV LSTM Univariate Forecasting")
plt.xlabel("Data Point")
plt.ylabel("Solar Irradiance (W/$m^2$)")
plt.legend()
plt.grid(True)
plt.savefig(Path("figure/3_Univariate_Forecasting.png"), dpi=600)
plt.show()

# Save results
output_df = pd.DataFrame({"Actual": raw_values[-horizon:], "Predict": predictions})
output_df.to_excel(Path("result/report_Bangkok_SolarPV.xlsx"), index=False, engine="openpyxl")

print("\n======= Univariate Forecasting COMPLETED =======\n")
