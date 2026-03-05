import streamlit as st
import joblib
import pandas as pd
import google.generativeai as genai
from xgboost import XGBClassifier

# --- Page Config ---
st.set_page_config(page_title="Purchase Intelligence Pro", page_icon="🛍️", layout="wide")

# --- Define Strict Feature Order (Crucial for Model Accuracy) ---
FEATURE_COLUMNS = [
    "Age", "Gender", "AnnualIncome", "NumberOfPurchases", 
    "ProductCategory", "TimeSpentOnWebsite", "LoyaltyProgram", "DiscountsAvailed"
]

# -----------------------------
# AI Chat Logic
# -----------------------------
def initialize_ai():
    """Initializes the Gemini AI model and catches specific errors"""
    api_key = st.secrets.get("GEMINI_API_KEY")
    if not api_key:
        return None, "Missing API Key in Streamlit Secrets"
    try:
        genai.configure(api_key=api_key)
        # Try the most reliable model name
        model = genai.GenerativeModel('gemini-1.5-flash')
        return model, None
    except Exception as e:
        return None, str(e)

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
    st.write("Analyze customer behavior and predict purchase intent using advanced Machine Learning.")

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
                # Create DataFrame with strict order
                input_data = pd.DataFrame([[age, gender, income, purchases, category, time_spent, loyalty, discounts]], 
                                        columns=FEATURE_COLUMNS)
                
                # Scaling logic
                input_processed = scaler.transform(input_data) if model_choice in ["Logistic Regression", "XGBoost"] else input_data
                
                prediction = models[model_choice].predict(input_processed)[0]
                
                # Store prediction in session state for AI context
                st.session_state['last_prediction'] = {
                    'status': "Likely to Purchase" if prediction == 1 else "Unlikely to Purchase",
                    'data': input_data.to_dict(orient='records')[0]
                }
                
                # Clear previous chat when a new prediction is made
                if "messages" in st.session_state:
                    del st.session_state.messages
                
                st.divider()
                if prediction == 1:
                    st.success("### ✅ Result: Customer is likely to PURCHASE")
                else:
                    st.warning("### ❌ Result: Customer is NOT likely to purchase")

            # --- AI CHAT ASSISTANT SECTION ---
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
                        # Get the model and the error if it fails
                        ai_model, init_error = initialize_ai()
                        
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
                            except Exception as e:
                                st.error(f"❌ Google API Error: {str(e)}")
                                st.info("Check if your API Key is correct and has 'Generative AI' permissions enabled in Google AI Studio.")
                        else:
                            st.error(f"❌ Initialization Error: {init_error}")
                            st.info("Ensure you have added 'GEMINI_API_KEY' to your Streamlit Secrets.")

        # --- Tab 2: Batch Prediction ---
        with tab2:
            st.subheader("Bulk Analysis")
            uploaded_file = st.file_uploader("Upload CSV File", type="csv")
            if uploaded_file:
                data = pd.read_csv(uploaded_file)
                
                # Search Feature
                search_query = st.text_input("🔍 Search in CSV", "")
                if search_query:
                    data = data[data.astype(str).apply(lambda x: x.str.contains(search_query, case=False)).any(axis=1)]

                st.write(f"📋 **Preview of Data ({len(data)} records):**")
                st.dataframe(data, use_column_width=True, height=300)

                if st.button("Run Batch Prediction"):
                    try:
                        if all(col in data.columns for col in FEATURE_COLUMNS):
                            data_for_model = data[FEATURE_COLUMNS]
                            data_processed = scaler.transform(data_for_model) if model_choice in ["Logistic Regression", "XGBoost"] else data_for_model
                            preds = models[model_choice].predict(data_processed)
                            data['Prediction_Result'] = preds
                            data['Prediction_Label'] = data['Prediction_Result'].map({1: '✅ Purchase', 0: '❌ No Purchase'})

                            st.divider()
                            st.subheader("🚀 Prediction Results")
                            c1, c2 = st.columns(2)
                            c1.metric("Total Records", len(data))
                            c2.metric("Predicted Purchases", (preds == 1).sum())

                            st.dataframe(data, use_column_width=True, height=400)
                            csv = data.to_csv(index=False).encode('utf-8')
                            st.download_button("📥 Download Results as CSV", csv, "results.csv", "text/csv")
                        else:
                            st.error("CSV columns do not match the required features.")
                    except Exception as e:
                        st.error(f"Error: {e}")
    else:
        st.error("Models not found. Please check your files.")

# -----------------------------
# Page 2: About Us
# -----------------------------
elif page == "👥 About Us":
    st.title("👥 Meet the Team")
    col1, col2 = st.columns(2)
    with col1:
        st.image("https://picsum.photos/seed/mujeeb/400/400", use_column_width=True)
        st.subheader("Mujeeb Ahmed")
        st.write("**Lead Developer & Data Scientist**")
        st.write("Responsible for core architecture and UI design.")
    with col2:
        st.image("https://picsum.photos/seed/hassan/400/400", use_column_width=True)
        st.subheader("Muhammad Hassan Solangi")
        st.write("**ML Engineer & Researcher**")
        st.write("Focused on model optimization and feature engineering.")

# -----------------------------
# Page 3: Our Mentors
# -----------------------------
elif page == "🎓 Our Mentors":
    st.title("🎓 Mentorship & Support")
    st.info("**PITP Program** - IBA Sukkur University (Supported by Sindh Government)")
    
    st.subheader("👨‍🏫 Our Teachers")
    t1, t2 = st.columns(2)
    with t1:
        st.markdown("### Sir Nabeel")
        st.write("Mentor for Python & Streamlit.")
    with t2:
        st.markdown("### Sir Ismail")
        st.write("Mentor for Machine Learning.")
        
    st.success("Special thanks to **Altaf Hussain (Director)** and his team.")
    st.image("https://picsum.photos/seed/university/800/300", use_column_width=True)

# --- Footer ---
st.divider()
st.caption("© 2026 PITP IBA Sukkur Project | Built by Team Mujeeb")
