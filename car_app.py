import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

# -----------------------------
# 🧠 Load Pre-trained Model
# -----------------------------
rf_model = joblib.load('car_price_prediction_model.pkl')  # Make sure this file exists
lr_model = LinearRegression()  # for comparison only

# -----------------------------
# 🚗 App Title
# -----------------------------
st.title("🚘 Car Price Prediction Dashboard")

st.markdown("""
Use this app to predict **used car selling prices** based on features like mileage, fuel type, year, etc.
""")

# -----------------------------
# 🎛️ Sidebar Inputs
# -----------------------------
st.sidebar.header("Manual Input Features")

year = st.sidebar.slider("Manufacture Year", 2000, 2025, 2015)
present_price = st.sidebar.slider("Present Ex-Showroom Price (in lakhs)", 1.0, 50.0, 10.0)
driven_kms = st.sidebar.number_input("Kilometers Driven", 0, 300000, 40000)
owner = st.sidebar.selectbox("Number of Previous Owners", [0, 1, 2, 3])

fuel_type = st.sidebar.selectbox("Fuel Type", ['Petrol', 'Diesel', 'CNG'])
selling_type = st.sidebar.selectbox("Selling Type", ['Dealer', 'Individual'])
transmission = st.sidebar.selectbox("Transmission", ['Manual', 'Automatic'])

# -----------------------------
# 🧾 Manual Input Prediction
# -----------------------------
input_data = pd.DataFrame({
    'Year': [year],
    'Present_Price': [present_price],
    'Driven_kms': [driven_kms],
    'Owner': [owner],
    'Fuel_Type': [fuel_type],
    'Selling_type': [selling_type],
    'Transmission': [transmission]
})

# Encode categorical columns as used during training
input_data = pd.get_dummies(input_data, drop_first=True)

# Ensure all expected columns exist (fill missing)
model_columns = rf_model.feature_names_in_
for col in model_columns:
    if col not in input_data.columns:
        input_data[col] = 0
input_data = input_data[model_columns]

# Predict price
predicted_price = rf_model.predict(input_data)[0]

st.subheader("🔮 Predicted Car Selling Price")
st.metric(label="Estimated Price (in Lakhs ₹)", value=f"{predicted_price:.2f}")

# -----------------------------
# 📂 Batch Prediction from CSV
# -----------------------------
st.subheader("📁 Batch Prediction")
st.markdown("""
Upload a CSV file with columns like:
`Year`, `Present_Price`, `Driven_kms`, `Owner`, `Fuel_Type`, `Selling_type`, `Transmission`
""")

uploaded_file = st.file_uploader("Upload your car data CSV", type="csv")

if uploaded_file:
    batch_df = pd.read_csv(uploaded_file)
    
    # Handle encoding similar to training
    batch_encoded = pd.get_dummies(batch_df, drop_first=True)
    for col in model_columns:
        if col not in batch_encoded.columns:
            batch_encoded[col] = 0
    batch_encoded = batch_encoded[model_columns]
    
    # Predict
    batch_df['Predicted_Price'] = rf_model.predict(batch_encoded)
    
    st.write("### 🧾 Predicted Prices:")
    st.dataframe(batch_df)
    
    csv = batch_df.to_csv(index=False).encode('utf-8')
    st.download_button("📥 Download Predictions", csv, "predicted_car_prices.csv", "text/csv")
    
    # -----------------------------
    # 📊 Feature Importance
    # -----------------------------
    st.subheader("📊 Feature Importance (Random Forest)")
    plt.clf()
    importances = rf_model.feature_importances_
    imp_df = pd.DataFrame({'Feature': model_columns, 'Importance': importances})
    imp_df = imp_df.sort_values(by='Importance', ascending=False)
    sns.barplot(x='Importance', y='Feature', data=imp_df)
    st.pyplot(plt)

    # -----------------------------
    # ⚖️ Model Comparison (Optional)
    # -----------------------------
    try:
        st.subheader("⚖️ Model Comparison")
        lr_model.fit(batch_encoded, batch_df['Predicted_Price'])
        lr_pred = lr_model.predict(batch_encoded)
        comparison_df = pd.DataFrame({
            'Random Forest': batch_df['Predicted_Price'],
            'Linear Regression': lr_pred
        })
        st.line_chart(comparison_df)
    except Exception as e:
        st.warning(f"⚠️ Model comparison failed: {e}")

else:
    st.info("Please upload a CSV file for batch prediction.")
