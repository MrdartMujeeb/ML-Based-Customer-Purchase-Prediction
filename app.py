import streamlit as st
import joblib
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from xgboost import XGBClassifier

# --- Page Configuration ---
st.set_page_config(
    page_title="Purchase Intelligence Pro",
    page_icon="🛍️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- Custom CSS for "Catchy" UI ---
st.markdown("""
    <style>
    .main {
        background-color: #0e1117;
    }
    .stButton>button {
        width: 100%;
        border-radius: 12px;
        height: 3em;
        background-color: #00d1b2;
        color: white;
        font-weight: bold;
        border: none;
        transition: 0.3s;
    }
    .stButton>button:hover {
        background-color: #00b89c;
        box-shadow: 0 4px 15px rgba(0, 209, 178, 0.4);
    }
    .metric-card {
        background-color: #161b22;
        padding: 20px;
        border-radius: 15px;
        border: 1px solid #30363d;
        text-align: center;
    }
    div[data-testid="stExpander"] {
        border-radius: 15px;
        border: 1px solid #30363d;
    }
    </style>
    """, unsafe_allow_html=True)

# --- Load Models and Preprocessor ---
@st.cache_resource
def load_files():
    try:
        scaler = joblib.load("preprocessor.pkl")
        xgb_model = XGBClassifier()
        xgb_model.load_model("xgboost_model.json")
        models = {
            "Logistic Regression": joblib.load("logistic_regression_model.pkl"),
            "Random Forest": joblib.load("random_forest_model.pkl"),
            "Decision Tree": joblib.load("decision_tree_model.pkl"),
            "XGBoost": xgb_model
        }
        return scaler, models, False
    except Exception as e:
        # Return dummy data for UI Demo if files are missing
        return None, None, True

scaler, models, is_demo = load_files()

# --- Sidebar Design ---
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/3081/3081559.png", width=100)
    st.title("Settings")
    st.write("Configure the prediction engine parameters below.")
    
    model_choice = st.selectbox(
        "🧠 Intelligence Model",
        ["XGBoost", "Random Forest", "Decision Tree", "Logistic Regression"] if not is_demo else ["Demo Model"]
    )
    
    st.divider()
    st.info("This engine uses behavioral patterns to predict purchase intent with high precision.")

# --- Main UI Layout ---
st.title("🛍️ Customer Purchase Intelligence")
st.markdown("### Predict and analyze customer behavior in real-time.")

if is_demo:
    st.warning("⚠️ **Demo Mode Active**: Model files not found. Using simulated prediction logic for UI demonstration.")

col1, col2 = st.columns([1, 1.2], gap="large")

with col1:
    st.subheader("👤 Customer Profile")
    with st.container():
        c1, c2 = st.columns(2)
        with c1:
            age = st.number_input("Age", 18, 70, 44)
            gender = st.selectbox("Gender", [0, 1], format_func=lambda x: "Female" if x == 0 else "Male")
            income = st.number_input("Annual Income ($)", value=84000)
        with c2:
            purchases = st.number_input("Prior Purchases", 0, 20, 10)
            category = st.selectbox("Product Category", [0, 1, 2, 3, 4], format_func=lambda x: f"Category {x}")
            loyalty = st.selectbox("Loyalty Program", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
            
        time_spent = st.slider("Time on Website (minutes)", 0.0, 120.0, 30.5)
        discounts = st.select_slider("Discounts Availed", options=[0, 1, 2, 3, 4, 5], value=2)

    predict_btn = st.button("🚀 Generate Prediction Report")

with col2:
    st.subheader("📊 Analysis Report")
    
    if predict_btn:
        # Prepare Data
        input_data = pd.DataFrame([{
            "Age": age,
            "Gender": gender,
            "AnnualIncome": income,
            "NumberOfPurchases": purchases,
            "ProductCategory": category,
            "TimeSpentOnWebsite": time_spent,
            "LoyaltyProgram": loyalty,
            "DiscountsAvailed": discounts
        }])

        # Prediction Logic
        if not is_demo:
            if model_choice in ["Logistic Regression", "XGBoost"]:
                input_processed = scaler.transform(input_data)
            else:
                input_processed = input_data
            
            model = models[model_choice]
            prediction = model.predict(input_processed)[0]
            # Simulate probability for gauge
            prob = model.predict_proba(input_processed)[0][1] * 100 if hasattr(model, "predict_proba") else (85.0 if prediction == 1 else 15.0)
        else:
            # Demo Logic
            prediction = 1 if (income > 50000 and time_spent > 20) else 0
            prob = 78.5 if prediction == 1 else 22.4

        # --- Visualizations ---
        
        # 1. Gauge Chart
        fig_gauge = go.Figure(go.Indicator(
            mode = "gauge+number",
            value = prob,
            title = {'text': "Purchase Probability (%)", 'font': {'size': 20}},
            gauge = {
                'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "white"},
                'bar': {'color': "#00d1b2" if prediction == 1 else "#ff4b4b"},
                'bgcolor': "rgba(0,0,0,0)",
                'borderwidth': 2,
                'bordercolor': "#30363d",
                'steps': [
                    {'range': [0, 50], 'color': 'rgba(255, 75, 75, 0.1)'},
                    {'range': [50, 100], 'color': 'rgba(0, 209, 178, 0.1)'}
                ],
            }
        ))
        fig_gauge.update_layout(paper_bgcolor='rgba(0,0,0,0)', font={'color': "white", 'family': "Arial"}, height=300, margin=dict(l=20, r=20, t=50, b=20))
        st.plotly_chart(fig_gauge, use_container_width=True)

        # 2. Result Message
        if prediction == 1:
            st.success(f"### ✅ Result: High Purchase Intent")
            st.write("The customer profile matches our 'Active Buyer' segment. High likelihood of conversion.")
        else:
            st.error(f"### ❌ Result: Low Purchase Intent")
            st.write("The customer shows passive browsing behavior. Consider retargeting with higher discounts.")

        # 3. Radar Chart (Comparison)
        categories = ['Income', 'Time Spent', 'Purchases', 'Discounts', 'Age']
        # Normalized values for radar
        values = [income/150000, time_spent/120, purchases/20, discounts/5, age/70]
        
        fig_radar = go.Figure()
        fig_radar.add_trace(go.Scatterpolar(
            r=values,
            theta=categories,
            fill='toself',
            name='Customer Profile',
            line_color='#00d1b2'
        ))
        fig_radar.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
            showlegend=False,
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font={'color': "white"},
            height=350
        )
        st.markdown("#### 🕸️ Behavioral Fingerprint")
        st.plotly_chart(fig_radar, use_container_width=True)

    else:
        st.info("Fill in the customer profile on the left and click 'Generate' to see the analysis.")
        st.image("https://img.freepik.com/free-vector/data-analysis-concept-illustration_114360-1611.jpg", use_column_width=True)

# --- Footer ---
st.divider()
st.caption("© 2024 Purchase Intelligence Engine | Built with ❤️ for Professional Dashboards")
