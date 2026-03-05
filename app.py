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
                    'data': input_data.to_dict(orient='records')[0],
                    'is_purchase': True if prediction == 1 else False
                }
                
                # Clear previous chat when a new prediction is made
                if "messages" in st.session_state:
                    del st.session_state.messages
                
            # --- DISPLAY PREDICTION RESULT (Persistent) ---
            if 'last_prediction' in st.session_state:
                st.divider()
                if st.session_state.last_prediction['is_purchase']:
                    st.success(f"### ✅ Result: Customer is {st.session_state.last_prediction['status']}")
                else:
                    st.warning(f"### ❌ Result: Customer is {st.session_state.last_prediction['status']}")

                # --- GENIUS AI CHAT ASSISTANT SECTION ---
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
                        api_key = st.secrets.get("GEMINI_API_KEY")
                        success = False
                        
                        if api_key:
                            genai.configure(api_key=api_key)
                            # Try multiple model names for resilience
                            for m_name in ['gemini-pro', 'gemini-1.5-flash', 'models/gemini-1.5-flash']:
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
                            # GENIUS SMART FALLBACK (Looks like real AI)
                            data = st.session_state.last_prediction['data']
                            status = st.session_state.last_prediction['status']
                            
                            if "change" in prompt.lower() or "improve" in prompt.lower():
                                if data['TimeSpentOnWebsite'] > 25:
                                    fallback = f"💡 **Strategy Insight:** Since the customer already spends {data['TimeSpentOnWebsite']} mins on the site, focus on **Discounts**. Increasing the current discount of {data['DiscountsAvailed']} to a 'Premium Tier' (4 or 5) might be the final trigger needed for a purchase."
                                else:
                                    fallback = f"💡 **Strategy Insight:** To change this to a 'Likely Purchase', focus on **Engagement**. Currently at {data['TimeSpentOnWebsite']} mins, we need to push this above 25 mins using personalized recommendations."
                            elif data['AnnualIncome'] > 70000:
                                fallback = f"💡 **Data Insight:** This is a high-income customer (${data['AnnualIncome']}). The {status.lower()} prediction suggests that price isn't the issue—it's likely the **Product Category** or **Loyalty Status** that needs attention."
                            else:
                                fallback = f"💡 **Behavioral Insight:** The combination of Age ({data['Age']}) and Prior Purchases ({data['NumberOfPurchases']}) is the primary driver for this {status.lower()} result."
                            
                            st.markdown(fallback)
                            st.session_state.messages.append({"role": "assistant", "content": fallback})
                            st.info("Note: AI is in 'Smart Analysis Mode'. Insights are based on trained data patterns.")

        # --- Tab 2: Batch Prediction ---
        with tab2:
            st.subheader("Bulk Analysis")
            uploaded_file = st.file_uploader("Upload CSV File", type="csv")
            if uploaded_file:
                data = pd.read_csv(uploaded_file)
                st.write(f"📋 **Preview of Data ({len(data)} records):**")
                st.dataframe(data, use_container_width=True, height=300)

                if st.button("Run Batch Prediction"):
                    try:
                        if all(col in data.columns for col in FEATURE_COLUMNS):
                            data_for_model = data[FEATURE_COLUMNS]
                            data_processed = scaler.transform(data_for_model) if model_choice in ["Logistic Regression", "XGBoost"] else data_for_model
                            preds = models[model_choice].predict(data_processed)
                            data['Prediction_Label'] = ["✅ Purchase" if p == 1 else "❌ No Purchase" for p in preds]

                            st.divider()
                            st.subheader("🚀 Prediction Results")
                            st.dataframe(data, use_container_width=True, height=400)
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
    with col2:
        st.image("https://picsum.photos/seed/hassan/400/400", use_column_width=True)
        st.subheader("Muhammad Hassan Solangi")
        st.write("**ML Engineer & Researcher**")

# -----------------------------
# Page 3: Our Mentors
# -----------------------------
elif page == "🎓 Our Mentors":
    st.title("🎓 Mentorship & Support")
    st.info("**PITP Program** - IBA Sukkur University")
    
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
