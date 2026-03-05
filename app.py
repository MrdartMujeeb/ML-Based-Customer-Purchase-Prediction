import streamlit as st
import joblib
import pandas as pd
from xgboost import XGBClassifier

# --- Page Config ---
st.set_page_config(page_title="Purchase Predictor", page_icon="✨", layout="centered")

# --- Artistic Styling ---
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;700&display=swap');
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
    .main { background-color: #0e1117; }
    .stButton>button {
        background: linear-gradient(45deg, #00d1b2, #009781);
        color: white; border: none; border-radius: 10px;
        padding: 0.6rem 2rem; font-weight: 700; width: 100%;
    }
    .result-card {
        padding: 2rem; border-radius: 20px; border: 1px solid #30363d;
        background: #161b22; text-align: center; margin-top: 2rem;
    }
    </style>
    """, unsafe_allow_html=True)

# --- Load Logic ---
@st.cache_resource
def load_assets():
    try:
        scaler = joblib.load("preprocessor.pkl")
        xgb = XGBClassifier(); xgb.load_model("xgboost_model.json")
        models = {
            "XGBoost": xgb,
            "Random Forest": joblib.load("random_forest_model.pkl"),
            "Logistic Regression": joblib.load("logistic_regression_model.pkl")
        }
        return scaler, models
    except: return None, None

scaler, models = load_assets()

# --- UI Header ---
st.title("✨ Purchase Intelligence")
st.write("Enter customer details to analyze purchase intent.")

# --- Simple Form ---
with st.container():
    col1, col2 = st.columns(2)
    with col1:
        age = st.number_input("Age", 18, 80, 35)
        income = st.number_input("Annual Income ($)", value=50000)
        time_spent = st.slider("Time on Site (min)", 0, 100, 25)
        loyalty = st.selectbox("Loyalty Member", [0, 1], format_func=lambda x: "Yes" if x==1 else "No")
    with col2:
        gender = st.selectbox("Gender", [0, 1], format_func=lambda x: "Male" if x==1 else "Female")
        purchases = st.number_input("Prior Purchases", 0, 50, 5)
        category = st.selectbox("Product Category", [0, 1, 2, 3, 4])
        model_name = st.selectbox("AI Model", list(models.keys()) if models else ["Demo Mode"])

# --- Prediction Action ---
if st.button("Analyze Intent"):
    if models:
        data = pd.DataFrame([[age, gender, income, purchases, category, time_spent, loyalty, 2]], 
                            columns=["Age", "Gender", "AnnualIncome", "NumberOfPurchases", "ProductCategory", "TimeSpentOnWebsite", "LoyaltyProgram", "DiscountsAvailed"])
        
        # Simple Logic
        processed = scaler.transform(data) if model_name in ["XGBoost", "Logistic Regression"] else data
        pred = models[model_name].predict(processed)[0]
    else:
        # Simple Demo Fallback
        pred = 1 if (income > 60000 and time_spent > 30) else 0

    # --- Artistic Result Display ---
    st.markdown('<div class="result-card">', unsafe_allow_html=True)
    if pred == 1:
        st.markdown("<h1 style='color: #00d1b2;'>High Intent ✅</h1>", unsafe_allow_html=True)
        st.write("This customer is highly likely to complete a purchase.")
        st.progress(85) # Simple artistic progress bar
    else:
        st.markdown("<h1 style='color: #ff4b4b;'>Low Intent ❌</h1>", unsafe_allow_html=True)
        st.write("This customer is currently in the browsing phase.")
        st.progress(15)
    st.markdown('</div>', unsafe_allow_html=True)

st.divider()
st.caption("Clean. Minimal. Intelligent.")
