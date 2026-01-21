import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# -----------------------------
# ⚡ Page Configuration
# -----------------------------
st.set_page_config(
    page_title="Car Price Prediction",
    layout="wide"
)

# -----------------------------
# 🧠 Load Model (Cached)
# -----------------------------
@st.cache_resource
def load_model():
    return joblib.load("car_price_prediction_model.pkl")

rf_model = load_model()
model_columns = rf_model.feature_names_in_

# -----------------------------
# 📂 Load Sample Dataset
# -----------------------------
@st.cache_data
def load_data():
    return pd.read_csv("car_data.csv")

# -----------------------------
# 🚘 App Header
# -----------------------------
st.title("🚘 Car Price Prediction Dashboard")
st.markdown(
    "Predict **used car prices** using Machine Learning. "
    "Explore data, make individual predictions, or run batch predictions."
)

# -----------------------------
# 📑 Tabs
# -----------------------------
tab1, tab2, tab3 = st.tabs(["🔮 Single Prediction", "📁 Batch Prediction", "📊 Data Insights"])

# =====================================================
# 🔮 TAB 1: SINGLE PREDICTION
# =====================================================
with tab1:
    st.subheader("Manual Car Price Prediction")

    col1, col2 = st.columns(2)

    with col1:
        year = st.slider("Manufacture Year", 2000, 2025, 2015)
        present_price = st.slider("Present Ex-Showroom Price (₹ Lakhs)", 1.0, 50.0, 10.0)
        driven_kms = st.number_input("Kilometers Driven", 0, 300000, 40000)
        owner = st.selectbox("Number of Previous Owners", [0, 1, 2, 3])

    with col2:
        fuel_type = st.selectbox("Fuel Type", ['Petrol', 'Diesel', 'CNG'])
        selling_type = st.selectbox("Selling Type", ['Dealer', 'Individual'])
        transmission = st.selectbox("Transmission", ['Manual', 'Automatic'])

    input_df = pd.DataFrame({
        'Year': [year],
        'Present_Price': [present_price],
        'Driven_kms': [driven_kms],
        'Owner': [owner],
        'Fuel_Type': [fuel_type],
        'Selling_type': [selling_type],
        'Transmission': [transmission]
    })

    input_df = pd.get_dummies(input_df, drop_first=True)

    for col in model_columns:
        if col not in input_df.columns:
            input_df[col] = 0

    input_df = input_df[model_columns]

    prediction = rf_model.predict(input_df)[0]

    st.metric("Estimated Selling Price (₹ Lakhs)", f"{prediction:.2f}")

# =====================================================
# 📁 TAB 2: BATCH PREDICTION
# =====================================================
with tab2:
    st.subheader("Batch Prediction")

    use_sample = st.checkbox("Use sample dataset")

    if use_sample:
        batch_df = load_data()
        st.success("Sample dataset loaded")
    else:
        uploaded_file = st.file_uploader("Upload CSV file", type="csv")
        if uploaded_file:
            batch_df = pd.read_csv(uploaded_file)
        else:
            st.info("Upload a CSV or select sample dataset")
            st.stop()

    st.dataframe(batch_df.head())

    encoded = pd.get_dummies(batch_df, drop_first=True)
    for col in model_columns:
        if col not in encoded.columns:
            encoded[col] = 0
    encoded = encoded[model_columns]

    batch_df["Predicted_Price"] = rf_model.predict(encoded)

    st.subheader("Predicted Prices")
    st.dataframe(batch_df)

    st.download_button(
        "📥 Download Results",
        batch_df.to_csv(index=False).encode("utf-8"),
        "car_price_predictions.csv",
        "text/csv"
    )

# =====================================================
# 📊 TAB 3: DATA INSIGHTS
# =====================================================
with tab3:
    st.subheader("Feature Importance")

    imp_df = pd.DataFrame({
        "Feature": model_columns,
        "Importance": rf_model.feature_importances_
    }).sort_values(by="Importance", ascending=False)

    fig, ax = plt.subplots(figsize=(8, 5))
    sns.barplot(data=imp_df, x="Importance", y="Feature", ax=ax)
    st.pyplot(fig)

    st.caption("Random Forest feature importance explaining model decisions")
