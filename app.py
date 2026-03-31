import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd

from model import load_data, preprocess_data, build_model, train_model, predict_future

st.set_page_config(page_title="AI Stock Predictor", layout="wide")

st.title("📈 AI Stock Prediction Dashboard")
st.write("Powered by LSTM Deep Learning 🚀")

# Sidebar
st.sidebar.title("⚙️ Controls")
ticker = st.sidebar.text_input("Enter Stock Ticker", "AAPL")
future_days = st.sidebar.slider("Days to Predict", 7, 90, 30)

st.info("💡 Example: AAPL (US) | RELIANCE.NS (India)")

if st.sidebar.button("Predict 🚀"):

    if not ticker:
        st.warning("⚠️ Please enter a stock ticker")
    else:
        try:
            # Load data
            df = load_data(ticker)

            st.subheader("📊 Raw Data")
            st.write(df.tail())

            # Preprocess
            X, y, scaler = preprocess_data(df)

            # Build model
            model = build_model((X.shape[1], 1))

            # Train
            model = train_model(model, X, y)

            st.success("✅ Model Trained Successfully!")

            # Predict
            future_predictions = predict_future(model, df, scaler)

            # Plot
            st.subheader("📈 Stock Price Prediction")

            plt.figure()
            plt.plot(df['Close'], label="Actual Price")

            future_index = range(len(df), len(df) + len(future_predictions))
            plt.plot(future_index, future_predictions, label="Predicted Price")

            plt.xlabel("Time")
            plt.ylabel("Price")
            plt.legend()

            st.pyplot(plt)

            # Table
            future_df = pd.DataFrame(future_predictions, columns=["Predicted Price"])

            st.subheader("🔮 Future Predictions")
            st.write(future_df)

        except Exception as e:
            st.error(f"{e}")

            st.warning("⚡ Showing demo data instead")

            import numpy as np
            import pandas as pd

            demo_data = pd.DataFrame({
                "Close": np.linspace(100, 150, 100)
            })

            st.line_chart(demo_data)