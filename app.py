import streamlit as st
import joblib
import pandas as pd
import google.generativeai as genai
import os
import base64
from xgboost import XGBClassifier

# --- Helper Function to Load Local Images ---
def get_base64_image(image_path):
    try:
        with open(image_path, "rb") as img_file:
            return f"data:image/png;base64,{base64.b64encode(img_file.read()).decode()}"
    except:
        # Fallback to a placeholder if file is not found
        return "https://picsum.photos/seed/profile/400/400"

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

                # --- STRATEGIC BUSINESS INSIGHTS (Stable & Professional) ---
                st.divider()
                st.subheader("📊 Strategic Business Insights")
                
                data = st.session_state.last_prediction['data']
                is_p = st.session_state.last_prediction['is_purchase']
                
                col_ins1, col_ins2 = st.columns(2)
                
                with col_ins1:
                    st.markdown("#### 🎯 Behavioral Analysis")
                    if is_p:
                        st.success("✅ **High Intent Profile**")
                        st.write(f"- Customer shows strong engagement with **{data['TimeOnWebsite']} minutes** on site.")
                        st.write(f"- Annual income of **${data['AnnualIncome']:,}** provides high purchasing power.")
                    else:
                        st.error("📉 **Low Conversion Risk**")
                        if data['TimeOnWebsite'] < 20:
                            st.write(f"- **Engagement Gap**: Only {data['TimeOnWebsite']} mins spent. Customer is likely 'bouncing'.")
                        if data['DiscountsAvailed'] < 1:
                            st.write("- **Price Sensitivity**: No discounts used. Customer may be waiting for a deal.")
                
                with col_ins2:
                    st.markdown("#### 💡 Actionable Recommendations")
                    if is_p:
                        st.write("1. **Upsell**: Recommend premium items in Category " + str(data['ProductCategory']) + ".")
                        st.write("2. **Loyalty**: Enroll in the VIP program to secure long-term value.")
                    else:
                        st.write("1. **Retargeting**: Send a personalized email for Category " + str(data['ProductCategory']) + ".")
                        st.write("2. **Incentive**: Offer a limited-time **15% discount** to trigger a first purchase.")

                # Visual Insight Chart
                st.markdown("#### 📈 Profile Strength")
                chart_data = pd.DataFrame({
                    'Metric': ['Income', 'Engagement', 'Experience'],
                    'Score': [data['AnnualIncome']/150000, data['TimeOnWebsite']/60, data['NumberOfPriorPurchases']/20]
                })
                st.bar_chart(chart_data.set_index('Metric'))

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
    st.write("The brilliant minds behind Purchase Intelligence Pro.")
    
    col1, col2 = st.columns(2)
    
    # 📸 UPDATE THESE FILENAMES TO YOUR ACTUAL PHOTO NAMES IN GITHUB
    mujeeb_img = get_base64_image("mujeeb.png") 
    hassan_img = get_base64_image("hassan.png")
    
    with col1:
        st.markdown(f"""
            <div class="img-container">
                <img src="{mujeeb_img}" class="team-img">
                <h3>Mujeeb Ahmed</h3>
                <p><b>Lead Developer & Data Scientist</b></p>
                <p>Responsible for core architecture and UI design.</p>
            </div>
        """, unsafe_allow_html=True)
        
    with col2:
        st.markdown(f"""
            <div class="img-container">
                <img src="{hassan_img}" class="team-img">
                <h3>Muhammad Hassan Solangi</h3>
                <p><b>ML Engineer & Researcher</b></p>
                <p>Focused on model optimization and feature engineering.</p>
            </div>
        """, unsafe_allow_html=True)

# -----------------------------
# Page 3: Our Mentors
# -----------------------------
elif page == "🎓 Our Mentors":
    st.title("🎓 Mentorship & Support")
    st.info("**PITP Program** - IBA Sukkur University")
    
    st.subheader("👨‍🏫 Our Teachers")
    
    # 📸 UPDATE THESE FILENAMES TO YOUR ACTUAL PHOTO NAMES IN GITHUB
    nabeel_img = get_base64_image("nabeel.png")
    ismail_img = get_base64_image("ismail.png")
    
    t1, t2 = st.columns(2)
    with t1:
        st.markdown(f"""
            <div class="img-container">
                <img src="{nabeel_img}" class="team-img">
                <h3>Sir Nabeel</h3>
                <p><b>Mentor for Python & Streamlit</b></p>
            </div>
        """, unsafe_allow_html=True)
    with t2:
        st.markdown(f"""
            <div class="img-container">
                <img src="{ismail_img}" class="team-img">
                <h3>Sir Ismail</h3>
                <p><b>Mentor for Machine Learning</b></p>
            </div>
        """, unsafe_allow_html=True)
        
    st.success("Special thanks to **Altaf Hussain (Director)** and his team.")
    
    # 📸 UPDATE THIS TO YOUR BANNER FILENAME IN GITHUB
    banner_img = get_base64_image("university_banner.png")
    st.markdown(f'<img src="{banner_img}" class="banner-img">', unsafe_allow_html=True)

# --- Footer ---
st.divider()
st.caption("© 2026 PITP IBA Sukkur Project | Built by Team Mujeeb")
