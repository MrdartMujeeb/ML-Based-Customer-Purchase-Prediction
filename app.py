import streamlit as st
import pandas as pd
import joblib
import numpy as np

st.set_page_config(page_title="Customer Purchase Prediction", page_icon="🛍️")

st.title("🛍️ Customer Purchase Prediction App")
st.write("Select a model and enter customer details to predict purchase behavior.")

# -----------------------------
# Load Scaler and Models ONCE
# -----------------------------

@st.cache_resource
def load_models():
    # Load the scaler
    scaler = joblib.load("models/scaler.pkl")

    # Load the models
    models = {
        "Logistic Regression": joblib.load("models/logistic_model.pkl"),
        "Random Forest": joblib.load("models/rf_model.pkl"),
        "Decision Tree": joblib.load("models/dt_model.pkl"),
        "XGBoost": joblib.load("models/xgb_model.pkl"),
    }

    return scaler, models

# Load models
scaler, models = load_models()

# -----------------------------
# Model Selection
# -----------------------------

model_choice = st.selectbox(
    "Choose a Model for Prediction",
    list(models.keys())
)

# -----------------------------
# User Inputs
# -----------------------------

col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Age", 18, 70, 44)
    gender = st.selectbox("Gender", [0, 1], format_func=lambda x: "Male" if x == 1 else "Female")
    income = st.number_input("Annual Income", value=84000)
    purchases = st.number_input("Prior Purchases", 0, 20, 10)

with col2:
    category = st.selectbox("Category (0-4)", [0, 1, 2, 3, 4])
    time_spent = st.number_input("Website Time (mins)", value=30.5)
    loyalty = st.selectbox("Loyalty Member", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
    discounts = st.slider("Discounts Used", 0, 5, 2)

# -----------------------------
# Prediction
# -----------------------------

if st.button("Predict"):
    # Prepare input data
    input_data = np.array([[age, gender, income, purchases,
                            category, time_spent, loyalty, discounts]])

    # Scale input
    scaled_data = scaler.transform(input_data)

    # Get selected model
    model = models[model_choice]

    # Make prediction
    prediction = model.predict(scaled_data)

    # Display result
    if prediction[0] == 1:
        st.success(f"✅ Result ({model_choice}): Likely to Purchase!")
    else:
        st.warning(f"⚠️ Result ({model_choice}): Unlikely to Purchase.")
