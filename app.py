import streamlit as st
import joblib
import pandas as pd
import google.generativeai as genai
from xgboost import XGBClassifier

# --- Page Config ---
st.set_page_config(page_title="Purchase Intelligence Pro", page_icon="🛍️", layout="wide")

# --- Define Strict Feature Order ---
FEATURE_COLUMNS = [
    "Age", "Gender", "AnnualIncome", "NumberOfPurchases", 
    "ProductCategory", "TimeSpentOnWebsite", "LoyaltyProgram", "DiscountsAvailed"
]

# -----------------------------
# AI Chat Logic
# -----------------------------
def initialize_ai():
    api_key = st.secrets.get("GEMINI_API_KEY")
    if not api_key:
        return None
    try:
        genai.configure(api_key=api_key)
        # Try to find a working model
        for m in ['gemini-1.5-flash', 'gemini-pro']:
            try:
                model = genai.GenerativeModel(m)
                return model
            except:
                continue
    except:
        return None
    return None

# -----------------------------
# Load Models and Preprocessor
# -----------------------------
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
        return scaler, models
    except:
        return None, None

scaler, models = load_files()

# -----------------------------
# Sidebar Navigation
# -----------------------------
with st.sidebar:
    st.title("🚀 Navigation")
    page = st.radio("Go to", ["🏠 Home (Predictor)", "👥 About Us", "🎓 Our Mentors"])
    st.divider()
    st.caption("Built with ❤️ by Team Mujeeb")

# -----------------------------
# Page 1: Home (Prediction Engine)
# -----------------------------
if page == "🏠 Home (Predictor)":
    st.title("🛍️ Customer Purchase Prediction")
    
    if models:
        model_choice = st.selectbox("Select Prediction Model", list(models.keys()))
        tab1, tab2 = st.tabs(["👤 Single Prediction", "📂 Batch Prediction (CSV)"])

        # --- Tab 1: Single Prediction ---
        with tab1:
            st.subheader("Individual Analysis")
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

            if st.button("Predict Single"):
                input_data = pd.DataFrame([[age, gender, income, purchases, category, time_spent, loyalty, discounts]], 
                                        columns=FEATURE_COLUMNS)
                
                processed = scaler.transform(input_data) if model_choice in ["Logistic Regression", "XGBoost"] else input_data
                prediction = models[model_choice].predict(processed)[0]
                
                # Store prediction in session state for AI context
                st.session_state['last_prediction'] = {
                    'status': "Likely to Purchase" if prediction == 1 else "Unlikely to Purchase",
                    'data': input_data.to_dict(orient='records')[0]
                }
                
                st.divider()
                if prediction == 1:
                    st.success("### ✅ Result: Customer is likely to PURCHASE")
                else:
                    st.warning("### ❌ Result: Customer is NOT likely to purchase")

            # --- AI CHAT SECTION ---
            if 'last_prediction' in st.session_state:
                st.divider()
                st.subheader("💬 AI Assistant Review & Chat")
                st.info("Ask the AI why this prediction was made or how to improve conversion for this customer.")

                # Initialize Chat History
                if "messages" not in st.session_state:
                    st.session_state.messages = []

                # Display Chat History
                for message in st.session_state.messages:
                    with st.chat_message(message["role"]):
                        st.markdown(message["content"])

                # Chat Input
                if prompt := st.chat_input("Ask me about this customer..."):
                    st.session_state.messages.append({"role": "user", "content": prompt})
                    with st.chat_message("user"):
                        st.markdown(prompt)

                    with st.chat_message("assistant"):
                        ai_model = initialize_ai()
                        if ai_model:
                            context = f"""
                            Context: The ML model predicted '{st.session_state.last_prediction['status']}' 
                            for this customer: {st.session_state.last_prediction['data']}.
                            User Question: {prompt}
                            Answer as a professional E-commerce Analyst. Keep it concise.
                            """
                            try:
                                response = ai_model.generate_content(context)
                                full_response = response.text
                                st.markdown(full_response)
                                st.session_state.messages.append({"role": "assistant", "content": full_response})
                            except:
                                st.error("AI Service is having trouble. Please check your API key.")
                        else:
                            st.warning("Please add GEMINI_API_KEY to Streamlit Secrets to chat.")

        # --- Tab 2: Batch Prediction ---
        with tab2:
            st.subheader("Bulk Analysis")
            uploaded_file = st.file_uploader("Upload CSV File", type="csv")
            if uploaded_file:
                df = pd.read_csv(uploaded_file)
                st.dataframe(df, use_column_width=True, height=250)
                if st.button("Run Batch Prediction"):
                    data_for_model = df[FEATURE_COLUMNS]
                    processed = scaler.transform(data_for_model) if model_choice in ["Logistic Regression", "XGBoost"] else data_for_model
                    preds = models[model_choice].predict(processed)
                    df['Status'] = ["✅ Purchase" if p == 1 else "❌ No Purchase" for p in preds]
                    st.dataframe(df, use_column_width=True, height=400)

# -----------------------------
# Page 2: About Us
# -----------------------------
elif page == "👥 About Us":
    st.title("👥 Meet the Team")
    c1, c2 = st.columns(2)
    with c1:
        st.image("https://picsum.photos/seed/mujeeb/400/400", use_column_width=True)
        st.subheader("Mujeeb Ahmed")
        st.write("**Lead Developer**")
    with c2:
        st.image("https://picsum.photos/seed/hassan/400/400", use_column_width=True)
        st.subheader("Muhammad Hassan Solangi")
        st.write("**ML Engineer**")

# -----------------------------
# Page 3: Our Mentors
# -----------------------------
elif page == "🎓 Our Mentors":
    st.title("🎓 Mentorship")
    st.info("PITP Program - IBA Sukkur University")
    t1, t2 = st.columns(2)
    with t1: st.markdown("### Sir Nabeel")
    with t2: st.markdown("### Sir Ismail")
    st.success("Special thanks to Director Altaf Hussain.")
    st.image("https://picsum.photos/seed/university/800/300", use_column_width=True)

st.sidebar.divider()
st.sidebar.caption("© 2026 PITP IBA Sukkur Project")
