import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

# Load stock data
def load_data(ticker):
    df = yf.download(ticker, period="5y")
    return df

# Preprocess data
def preprocess_data(df):
    data = df[['Close']].values

    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data)

    X = []
    y = []

    for i in range(60, len(scaled_data)):
        X.append(scaled_data[i-60:i, 0])
        y.append(scaled_data[i, 0])

    X, y = np.array(X), np.array(y)

    # reshape for LSTM [samples, time_steps, features]
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))

    return X, y, scaler, scaled_data

# Build LSTM model
def build_model():
    model = Sequential()

    model.add(LSTM(units=50, return_sequences=True, input_shape=(60, 1)))
    model.add(Dropout(0.2))

    model.add(LSTM(units=50, return_sequences=False))
    model.add(Dropout(0.2))

    model.add(Dense(units=25))
    model.add(Dense(units=1))

    model.compile(optimizer='adam', loss='mean_squared_error')

    return model

# Train model
def train_model(model, X, y):
    model.fit(X, y, epochs=5, batch_size=32)
    return model

# Predict future prices
def predict_future(model, scaled_data, scaler, days=30):
    last_60_days = scaled_data[-60:]
    predictions = []

    current_input = last_60_days.reshape(1, 60, 1)

    for _ in range(days):
        pred = model.predict(current_input, verbose=0)
        predictions.append(pred[0][0])

        # update input
        current_input = np.append(current_input[:, 1:, :], [[pred]], axis=1)

    predictions = scaler.inverse_transform(np.array(predictions).reshape(-1, 1))
    return predictions