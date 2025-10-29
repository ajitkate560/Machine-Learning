# ----------------------------------------------------
# 🟡 GOLD PRICE PREDICTION APP (STREAMLIT)
# ----------------------------------------------------

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error

# -------------------------------------
# 🏷️ App Title
# -------------------------------------
st.set_page_config(page_title="Gold Price Prediction", page_icon="💰", layout="wide")
st.title("💰 Gold Price Prediction using Machine Learning")
st.write("This app predicts the future gold prices using historical data and machine learning (Random Forest Regressor).")

# -------------------------------------
# 📅 User Input Section
# -------------------------------------
start_date = st.date_input("Select Start Date", value=pd.to_datetime("2010-01-01"))
end_date = st.date_input("Select End Date", value=pd.to_datetime("today"))

# Download Button
if st.button("Fetch Gold Price Data"):
    # -------------------------------------
    # 📥 Fetch Data from Yahoo Finance
    # -------------------------------------
    data = yf.download("GC=F", start=start_date, end=end_date)

    if data.empty:
        st.error("No data found. Please select a valid date range.")
    else:
        st.success("✅ Data fetched successfully!")

        # -------------------------------------
        # 🧹 Preprocess Data
        # -------------------------------------
        data['MA10'] = data['Close'].rolling(10).mean()
        data['MA50'] = data['Close'].rolling(50).mean()
        data['MA100'] = data['Close'].rolling(100).mean()
        data.dropna(inplace=True)

        # -------------------------------------
        # 📊 Show Dataset
        # -------------------------------------
        st.subheader("📈 Gold Price Data")
        st.dataframe(data.tail(10))

        # -------------------------------------
        # 📉 Visualization
        # -------------------------------------
        st.subheader("📊 Gold Price Trend")
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(data['Close'], label='Gold Closing Price', color='gold')
        ax.set_xlabel("Date")
        ax.set_ylabel("Price (USD)")
        ax.legend()
        st.pyplot(fig)

        # -------------------------------------
        # 🧩 Feature & Target Split
        # -------------------------------------
        X = data[['Open', 'High', 'Low', 'Volume', 'MA10', 'MA50', 'MA100']]
        y = data['Close']

        # Split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

        # -------------------------------------
        # ⚙️ Train Model
        # -------------------------------------
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)

        # -------------------------------------
        # 📏 Evaluate Model
        # -------------------------------------
        r2 = r2_score(y_test, predictions)
        mae = mean_absolute_error(y_test, predictions)

        st.subheader("📊 Model Performance")
        st.write(f"**R² Score:** {r2:.4f}")
        st.write(f"**Mean Absolute Error:** {mae:.2f}")

        # -------------------------------------
        # 🧠 Plot Predictions
        # -------------------------------------
        st.subheader("🔍 Actual vs Predicted Prices")
        fig2, ax2 = plt.subplots(figsize=(10, 4))
        ax2.plot(y_test.values, label="Actual", color="gold")
        ax2.plot(predictions, label="Predicted", color="blue")
        ax2.legend()
        st.pyplot(fig2)

        # -------------------------------------
        # 🔮 Predict Next Day Price
        # -------------------------------------
        last_row = data.tail(1)[['Open', 'High', 'Low', 'Volume', 'MA10', 'MA50', 'MA100']]
        next_day_pred = model.predict(last_row)
        st.subheader("💰 Predicted Next Day Gold Price")
        st.metric(label="Next Day Predicted Price (USD)", value=f"${next_day_pred[0]:.2f}")
