import streamlit as st
import pandas as pd
import joblib
import numpy as np

st.title("🛍️ Customer Purchase Prediction")

# 1. Model Selection Dropdown
model_choice = st.selectbox(
    "Choose a Model for Prediction",
    ("Logistic Regression", "Random Forest", "Decision Tree", "XGBoost")
)

# Map choices to filenames
model_dict = {
    "Logistic Regression": "logistic_model.pkl",
    "Random Forest": "rf_model.pkl",
    "Decision Tree": "dt_model.pkl",
    "XGBoost": "xgb_model.pkl"
}

# 2. User Input Fields
col1, col2 = st.columns(2)
with col1:
    age = st.number_input("Age", 18, 70, 44)
    gender = st.selectbox("Gender", [0, 1], format_func=lambda x: "Male" if x==1 else "Female")
    income = st.number_input("Annual Income", value=84000)
    purchases = st.number_input("Prior Purchases", 0, 20, 10)
with col2:
    category = st.selectbox("Category (0-4)", [0, 1, 2, 3, 4])
    time_spent = st.number_input("Website Time (mins)", value=30.5)
    loyalty = st.selectbox("Loyalty Member", [0, 1], format_func=lambda x: "Yes" if x==1 else "No")
    discounts = st.slider("Discounts", 0, 5, 2)

# 3. Prediction Logic
if st.button("Predict"):
    # Prepare data for scaling
    input_data = np.array([[age, gender, income, purchases, category, time_spent, loyalty, discounts]])
    
    # Load Scaler and Transform Data
    scaler = joblib.load('preprocessor.pkl')
    scaled_data = scaler.transform(input_data)
    
    # Load Selected Model
    model = joblib.load(model_dict[model_choice])
    prediction = model.predict(scaled_data)
    
    if prediction[0] == 1:
        st.success(f"Result ({model_choice}): Likely to Purchase!")
    else:
        st.warning(f"Result ({model_choice}): Unlikely to Purchase.")

