import streamlit as st
import joblib
import pandas as pd
import google.generativeai as genai
import os
from xgboost import XGBClassifier

# --- Page Config ---
st.set_page_config(page_title="Purchase Intelligence Pro", page_icon="🛍️", layout="wide")

# --- Custom CSS for Rounded Images & Premium Look ---
st.markdown("""
    <style>
    .team-img {
        border-radius: 50%;
        width: 220px;
        height: 220px;
        object-fit: cover;
        margin-bottom: 20px;
        border: 5px solid #f0f2f6;
        box-shadow: 0 10px 25px rgba(0,0,0,0.1);
        transition: transform 0.3s ease;
    }
    .team-img:hover {
        transform: scale(1.05);
    }
    .img-container {
        display: flex;
        flex-direction: column;
        align-items: center;
        text-align: center;
        padding: 20px;
        background: white;
        border-radius: 20px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.05);
        margin: 10px;
    }
    .banner-img {
        border-radius: 20px;
        width: 100%;
        height: auto;
        box-shadow: 0 10px 25px rgba(0,0,0,0.1);
        border: 1px solid rgba(0,0,0,0.05);
    }
    </style>
""", unsafe_allow_html=True)

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
    st.write("Analyze customer behavior and predict purchase intent using advanced Machine Learning.")

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
                input_data = pd.DataFrame([[age, gender, income, purchases, category, time_spent, loyalty, discounts]], 
                                        columns=FEATURE_COLUMNS)
                
                input_processed = scaler.transform(input_data) if model_choice in ["Logistic Regression", "XGBoost"] else input_data
                prediction = models[model_choice].predict(input_processed)[0]
                
                st.session_state['last_prediction'] = {
                    'status': "Likely to Purchase" if prediction == 1 else "Unlikely to Purchase",
                    'data': input_data.to_dict(orient='records')[0],
                    'is_purchase': True if prediction == 1 else False
                }
                if "messages" in st.session_state:
                    del st.session_state.messages
                
            if 'last_prediction' in st.session_state:
                st.divider()
                if st.session_state.last_prediction['is_purchase']:
                    st.success(f"### ✅ Result: Customer is {st.session_state.last_prediction['status']}")
                else:
                    st.warning(f"### ❌ Result: Customer is {st.session_state.last_prediction['status']}")

                st.divider()
                st.subheader("💬 AI Assistant Review & Chat")
                if "messages" not in st.session_state:
                    st.session_state.messages = []

                for message in st.session_state.messages:
                    with st.chat_message(message["role"]):
                        st.markdown(message["content"])

                if prompt := st.chat_input("Ask me about this customer..."):
                    st.session_state.messages.append({"role": "user", "content": prompt})
                    with st.chat_message("user"):
                        st.markdown(prompt)

                    with st.chat_message("assistant"):
                        # Get API Key from Secrets or Environment
                        api_key = st.secrets.get("GEMINI_API_KEY") or os.environ.get("GEMINI_API_KEY")
                        success = False
                        
                        if api_key:
                            try:
                                genai.configure(api_key=api_key)
                                ai_model = genai.GenerativeModel('gemini-1.5-flash')
                                context = f"Customer Data: {st.session_state.last_prediction['data']}. Prediction: {st.session_state.last_prediction['status']}. Question: {prompt}. Act as a senior retail analyst."
                                response = ai_model.generate_content(context)
                                full_response = response.text
                                st.markdown(full_response)
                                st.session_state.messages.append({"role": "assistant", "content": full_response})
                                success = True
                            except Exception as e:
                                st.error(f"AI Error: {e}")
                        
                        if not success:
                            # 🧠 STRATEGIC CONSULTANT FALLBACK (Much Smarter)
                            data = st.session_state.last_prediction['data']
                            is_p = st.session_state.last_prediction['is_purchase']
                            status = st.session_state.last_prediction['status']
                            
                            conv_keys = ["conversion", "improve", "change", "convert", "buy", "how to", "sell"]
                            why_keys = ["why", "reason", "factor", "because"]
                            prompt_l = prompt.lower()

                            if any(k in prompt_l for k in conv_keys):
                                if not is_p:
                                    if data['TimeSpentOnWebsite'] < 20:
                                        fallback = f"🚀 **Conversion Strategy:** The primary bottleneck is **Engagement**. At {data['TimeSpentOnWebsite']} mins, the customer is 'bouncing' too early. **Action:** Implement a 'Wait! Don't Go' popup with a personalized recommendation for Category {data['ProductCategory']}."
                                    elif data['DiscountsAvailed'] < 3:
                                        fallback = f"🚀 **Conversion Strategy:** This customer is price-sensitive. They have only used {data['DiscountsAvailed']} discounts. **Action:** A targeted 15% 'First-Time Buyer' discount would likely trigger the purchase intent."
                                    else:
                                        fallback = f"🚀 **Conversion Strategy:** High engagement but no intent suggests a **Trust Gap**. **Action:** Display 'Verified Buyer' reviews and 'Secure Checkout' badges specifically for this customer segment."
                                else:
                                    fallback = f"📈 **Growth Strategy:** This customer is already likely to buy! To increase **Average Order Value (AOV)**, offer a bundle deal related to their previous {data['NumberOfPurchases']} purchases."
                            elif any(k in prompt_l for k in why_keys):
                                if data['AnnualIncome'] > 75000 and not is_p:
                                    fallback = f"🔍 **Analysis:** Even with a high income (${data['AnnualIncome']}), the model predicts 'No Purchase' because the **Frequency** ({data['NumberOfPurchases']} purchases) is too low for their age group ({data['Age']}). They are a 'High-Value/Low-Loyalty' risk."
                                elif data['TimeSpentOnWebsite'] > 25:
                                    fallback = f"🔍 **Analysis:** The strongest positive factor here is **Dwell Time** ({data['TimeSpentOnWebsite']} mins). This indicates high interest, but other friction points (like lack of loyalty status) are holding back the final conversion."
                                else:
                                    fallback = f"🔍 **Analysis:** The prediction is driven by the 'Engagement-to-Income' ratio. For this income bracket, we expect more than {data['TimeSpentOnWebsite']} minutes of browsing before a decision is made."
                            else:
                                fallback = f"📋 **General Insight:** This {data['Age']}-year-old customer in Category {data['ProductCategory']} shows patterns typical of a 'Window Shopper'. Focus on long-term email nurturing rather than immediate hard-selling."

                            st.markdown(fallback)
                            st.session_state.messages.append({"role": "assistant", "content": fallback})
                            st.info("💡 Pro-Tip: I'm analyzing the specific relationship between this customer's income, time spent, and discount history.")

        with tab2:
            st.subheader("Bulk Analysis")
            uploaded_file = st.file_uploader("Upload CSV File", type="csv")
            if uploaded_file:
                data = pd.read_csv(uploaded_file)
                st.dataframe(data, use_container_width=True, height=300)
                if st.button("Run Batch Prediction"):
                    data_for_model = data[FEATURE_COLUMNS]
                    data_processed = scaler.transform(data_for_model) if model_choice in ["Logistic Regression", "XGBoost"] else data_for_model
                    preds = models[model_choice].predict(data_processed)
                    data['Prediction'] = ["✅ Purchase" if p == 1 else "❌ No Purchase" for p in preds]
                    st.dataframe(data, use_container_width=True, height=400)

elif page == "👥 About Us":
    st.title("👥 Meet the Team")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f'<div class="img-container"><img src="https://picsum.photos/seed/mujeeb/400/400" class="team-img"><h3>Mujeeb Ahmed</h3><p>Lead Developer</p></div>', unsafe_allow_html=True)
    with col2:
        st.markdown(f'<div class="img-container"><img src="https://picsum.photos/seed/hassan/400/400" class="team-img"><h3>Muhammad Hassan Solangi</h3><p>ML Engineer</p></div>', unsafe_allow_html=True)

elif page == "🎓 Our Mentors":
    st.title("🎓 Mentorship")
    st.info("PITP Program - IBA Sukkur University")
    st.markdown(f'<img src="https://picsum.photos/seed/university/800/300" class="banner-img">', unsafe_allow_html=True)

st.divider()
st.caption("© 2026 PITP IBA Sukkur Project | Built by Team Mujeeb")
