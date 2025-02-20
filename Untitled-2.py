"""
Univariate Forecasting with LSTM Model
Optimized for Best Practices (Python 3.12.9)
"""

import time
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Bidirectional
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Configuration settings
CONFIG = {
    "filename": Path("dataset/Bangkok_solarpv_Trial.csv"),
    "horizon": 372,
    "neurons": 64,  # Increased neurons for better learning
    "epochs": 150,  # More epochs for better convergence
    "batch_size": 32,  # Optimized batch size for efficiency
    "plot_results": True,  # Set to False to disable plotting
    "save_results": True  # Set to False to disable saving
}

# Convert time series data into supervised learning format
def timeseries_to_supervised(data: np.ndarray, lag: int = 1) -> pd.DataFrame:
    df = pd.DataFrame(data)
    df_shifted = pd.concat([df.shift(i) for i in range(1, lag + 1)], axis=1)
    df_shifted.columns = [f"lag_{i}" for i in range(1, lag + 1)]
    df_shifted["target"] = df
    df_shifted.fillna(0, inplace=True)
    return df_shifted

# Convert data to a differenced series
def difference(data: np.ndarray, interval: int = 1) -> np.ndarray:
    return np.diff(data, n=interval)

# Reverse differencing
def inverse_difference(history: np.ndarray, forecast: float, interval: int = 1) -> float:
    return forecast + history[-interval]

# Normalize data
def scale_data(train: np.ndarray, test: np.ndarray):
    scaler = MinMaxScaler(feature_range=(-1, 1))
    scaler.fit(train)
    return scaler, scaler.transform(train), scaler.transform(test)

# Invert scaling
def invert_scale(scaler: MinMaxScaler, X: np.ndarray, value: float) -> float:
    return scaler.inverse_transform(np.hstack([X, value]).reshape(1, -1))[0, -1]

# Define LSTM model
def build_lstm(input_shape, neurons: int) -> Sequential:
    model = Sequential([
        Bidirectional(LSTM(neurons, return_sequences=False), input_shape=input_shape),
        Dense(1)
    ])
    model.compile(loss="mean_squared_error", optimizer="adam")
    return model

# Train LSTM model
def train_lstm(model: Sequential, X_train: np.ndarray, y_train: np.ndarray, config: dict):
    logging.info("Training LSTM model...")
    model.fit(X_train, y_train, epochs=config["epochs"], batch_size=config["batch_size"], verbose=1, shuffle=False)
    return model

# Forecast using LSTM
def forecast_lstm(model: Sequential, X: np.ndarray) -> float:
    return model.predict(X.reshape(1, 1, -1))[0, 0]

# Load dataset
def load_data(filename: Path) -> pd.DataFrame:
    return pd.read_csv(filename, header=0, parse_dates=[0], index_col=0)

# Main execution function
def main(config: dict):
    logging.info("\n======= Starting LSTM Univariate Forecasting =======\n")

    # Load and preprocess data
    series = load_data(config["filename"])
    raw_values = series.values.flatten()
    diff_values = difference(raw_values)

    supervised_df = timeseries_to_supervised(diff_values, lag=1)
    supervised_values = supervised_df.values

    # Split into training and testing sets
    train, test = supervised_values[:-config["horizon"]], supervised_values[-config["horizon"]:]
    scaler, train_scaled, test_scaled = scale_data(train, test)

    # Prepare training data
    X_train, y_train = train_scaled[:, :-1], train_scaled[:, -1]
    X_train = X_train.reshape(X_train.shape[0], 1, X_train.shape[1])  # Reshape for LSTM

    # Build and train model
    model = build_lstm(input_shape=(X_train.shape[1], X_train.shape[2]), neurons=config["neurons"])
    start_time = time.time()
    model = train_lstm(model, X_train, y_train, config)

    # Make predictions
    predictions = []
    for i in range(len(test_scaled)):
        X, y = test_scaled[i, :-1], test_scaled[i, -1]
        yhat = forecast_lstm(model, X)
        yhat = invert_scale(scaler, X, yhat)
        yhat = inverse_difference(raw_values, yhat, len(test_scaled) + 1 - i)
        predictions.append(yhat)

        expected = raw_values[len(train) + i + 1]
        logging.info(f"Point={i+1}, Predicted={yhat:.3f}, Expected={expected:.3f}")

    # Evaluate performance
    elapsed_time = time.time() - start_time
    rmse = np.sqrt(mean_squared_error(raw_values[-config["horizon"]:], predictions))
    logging.info(f"\nProcessing Time: {elapsed_time:.2f} seconds")
    logging.info(f"Test RMSE: {rmse:.3f}")

    # Plot results
    if config["plot_results"]:
        plt.figure(figsize=(10, 5))
        plt.plot(raw_values[-config["horizon"]:], label="Actual")
        plt.plot(predictions, label="Forecast", linestyle="dashed")
        plt.title("Bangkok Solar PV LSTM Univariate Forecasting")
        plt.xlabel("Data Point")
        plt.ylabel("Solar Irradiance (W/$m^2$)")
        plt.legend()
        plt.grid(True)
        plt.savefig(Path("figure/Optimized_Univariate_Forecasting.png"), dpi=600)
        plt.show()

    # Save results
    if config["save_results"]:
        output_df = pd.DataFrame({"Actual": raw_values[-config["horizon"]:], "Predict": predictions})
        output_df.to_excel(Path("result/Optimized_Bangkok_SolarPV.xlsx"), index=False, engine="openpyxl")
        logging.info("\nResults saved successfully.")

    logging.info("\n======= Univariate Forecasting COMPLETED =======\n")

# Run script
if __name__ == "__main__":
    main(CONFIG)
