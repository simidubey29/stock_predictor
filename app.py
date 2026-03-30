import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd

from model import load_data, preprocess_data, build_model, train_model, predict_future

st.set_page_config(page_title="AI Stock Predictor", layout="wide")

st.title("📈 AI Stock Prediction Dashboard")
st.write("Powered by LSTM Deep Learning 🚀")

# Sidebar
ticker = st.sidebar.text_input("Enter Stock Ticker", "AAPL")
future_days = st.sidebar.slider("Days to Predict", 7, 90, 30)

if st.sidebar.button("Predict 🚀"):
    # Load data
    df = load_data(ticker)

    st.subheader("📊 Raw Data")
    st.write(df.tail())

    # Preprocess
    X, y, scaler, scaled_data = preprocess_data(df)

    # Build & Train model
    model = build_model()
    model = train_model(model, X, y)

    st.success("✅ Model Trained Successfully!")

    # Predict future
    future_predictions = predict_future(model, scaled_data, scaler, future_days)

    # Plot
    st.subheader("📈 Stock Price Prediction")

    plt.figure()
    plt.plot(df['Close'], label="Actual Price")

    # Future index
    future_index = range(len(df), len(df) + future_days)

    plt.plot(future_index, future_predictions, label="Predicted Price")

    plt.xlabel("Time")
    plt.ylabel("Price")
    plt.legend()

    st.pyplot(plt)

    # Show predictions table
    future_df = pd.DataFrame(future_predictions, columns=["Predicted Price"])
    st.subheader("🔮 Future Predictions")
    st.write(future_df)