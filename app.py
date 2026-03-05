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
                input_data = pd.DataFrame([[age, gender, income, purchases, category, time_spent, loyalty, discounts]], columns=FEATURE_COLUMNS)
                processed = scaler.transform(input_data) if model_choice in ["Logistic Regression", "XGBoost"] else input_data
                prediction = models[model_choice].predict(processed)[0]
                
                st.session_state['last_prediction'] = {
                    'status': "Likely to Purchase" if prediction == 1 else "Unlikely to Purchase",
                    'data': input_data.to_dict(orient='records')[0]
                }
                if "messages" in st.session_state: del st.session_state.messages
                
                st.divider()
                if prediction == 1: st.success("### ✅ Result: Customer is likely to PURCHASE")
                else: st.warning("### ❌ Result: Customer is NOT likely to purchase")

            # --- BULLETPROOF AI CHAT SECTION ---
            if 'last_prediction' in st.session_state:
                st.divider()
                st.subheader("💬 AI Assistant Review & Chat")
                
                if "messages" not in st.session_state: st.session_state.messages = []
                for message in st.session_state.messages:
                    with st.chat_message(message["role"]): st.markdown(message["content"])

                if prompt := st.chat_input("Ask me about this customer..."):
                    st.session_state.messages.append({"role": "user", "content": prompt})
                    with st.chat_message("user"): st.markdown(prompt)

                    with st.chat_message("assistant"):
                        api_key = st.secrets.get("GEMINI_API_KEY")
                        if not api_key:
                            st.error("Missing GEMINI_API_KEY in Streamlit Secrets.")
                        else:
                            genai.configure(api_key=api_key)
                            # Try 'gemini-pro' first as it is the most stable across all versions
                            models_to_try = ['gemini-pro', 'gemini-1.5-flash', 'gemini-1.5-pro']
                            success = False

                            for m_name in models_to_try:
                                try:
                                    ai_model = genai.GenerativeModel(m_name)
                                    context = f"Customer: {st.session_state.last_prediction['data']}. Prediction: {st.session_state.last_prediction['status']}. Question: {prompt}. Answer as a pro analyst."
                                    response = ai_model.generate_content(context)
                                    full_response = response.text
                                    st.markdown(full_response)
                                    st.session_state.messages.append({"role": "assistant", "content": full_response})
                                    success = True
                                    break 
                                except:
                                    continue
                            
                            if not success:
                                # SMART FALLBACK: If AI fails, provide a logical answer based on the data
                                data = st.session_state.last_prediction['data']
                                if data['TimeSpentOnWebsite'] < 15:
                                    fallback = "💡 **Expert Analysis:** The customer spent very little time on the site (under 15 mins). Even with high income, this usually means they are just browsing. To improve conversion, try offering a 'Limited Time' discount pop-up."
                                else:
                                    fallback = "💡 **Expert Analysis:** The customer shows good engagement, but other factors like age or category might be influencing the cautious prediction. Consider a personalized email follow-up."
                                st.markdown(fallback)
                                st.session_state.messages.append({"role": "assistant", "content": fallback})
                                st.info("Note: AI is in 'Offline Mode' due to connection issues. Still providing logical insights!")

        with tab2:
            st.subheader("Bulk Analysis")
            uploaded_file = st.file_uploader("Upload CSV", type="csv")
            if uploaded_file:
                data = pd.read_csv(uploaded_file)
                st.dataframe(data, use_container_width=True, height=250)
                if st.button("Run Batch Prediction"):
                    processed = scaler.transform(data[FEATURE_COLUMNS]) if model_choice in ["Logistic Regression", "XGBoost"] else data[FEATURE_COLUMNS]
                    preds = models[model_choice].predict(processed)
                    data['Status'] = ["✅ Purchase" if p == 1 else "❌ No Purchase" for p in preds]
                    st.dataframe(data, use_container_width=True, height=400)

elif page == "👥 About Us":
    st.title("👥 Meet the Team")
    c1, c2 = st.columns(2)
    with c1:
        st.image("https://picsum.photos/seed/mujeeb/400/400", use_column_width=True)
        st.subheader("Mujeeb Ahmed")
    with c2:
        st.image("https://picsum.photos/seed/hassan/400/400", use_column_width=True)
        st.subheader("Muhammad Hassan Solangi")

elif page == "🎓 Our Mentors":
    st.title("🎓 Mentorship")
    st.info("PITP Program - IBA Sukkur University")
    st.success("Special thanks to Director Altaf Hussain, Sir Nabeel, and Sir Ismail.")
    st.image("https://picsum.photos/seed/university/800/300", use_column_width=True)

st.sidebar.divider()
st.sidebar.caption("© 2026 PITP IBA Sukkur Project")
