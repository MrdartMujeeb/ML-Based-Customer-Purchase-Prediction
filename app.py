import streamlit as st
import joblib
import numpy as np

st.title("🛍️ Customer Purchase Prediction")

st.write("Predict whether a customer will purchase or not using trained ML models.")

# -----------------------------
# Load Preprocessor and Models
# -----------------------------
@st.cache_resource
def load_files():
    scaler = joblib.load("preprocessor.pkl")

    models = {
        "Logistic Regression": joblib.load("logistic_regression_model.pkl"),
        "Random Forest": joblib.load("random_forest_model.pkl"),
        "Decision Tree": joblib.load("decision_tree_model.pkl"),
        "XGBoost": joblib.load("xgboost_model.pkl")
    }

    return scaler, models


scaler, models = load_files()

# -----------------------------
# Model Selection
# -----------------------------
model_choice = st.selectbox(
    "Choose Model",
    list(models.keys())
)

# -----------------------------
# User Input Fields
# -----------------------------
col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Age", 18, 70, 44)
    gender = st.selectbox("Gender", [0, 1], format_func=lambda x: "Female" if x == 0 else "Male")
    income = st.number_input("Annual Income", value=84000)
    purchases = st.number_input("Prior Purchases", 0, 20, 10)

with col2:
    category = st.selectbox("Product Category", [0, 1, 2, 3, 4])
    time_spent = st.number_input("Time on Website (minutes)", value=30.5)
    loyalty = st.selectbox("Loyalty Program", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
    discounts = st.slider("Discounts Availed", 0, 5, 2)

# -----------------------------
# Prediction
# -----------------------------
if st.button("Predict"):

    input_data = np.array([[
        age,
        gender,
        income,
        purchases,
        category,
        time_spent,
        loyalty,
        discounts
    ]])

    # Apply scaling only where required
    if model_choice in ["Logistic Regression", "XGBoost"]:
        input_processed = scaler.transform(input_data)
    else:
        input_processed = input_data

    model = models[model_choice]
    prediction = model.predict(input_processed)

    # Display Result
    if prediction[0] == 1:
        st.success(f"Prediction using {model_choice}: Customer is likely to PURCHASE.")
    else:
        st.warning(f"Prediction using {model_choice}: Customer is NOT likely to purchase.")
