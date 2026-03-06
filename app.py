import streamlit as st
import joblib
import pandas as pd
from xgboost import XGBClassifier

# --- Page Config ---
st.set_page_config(page_title="Purchase Intelligence Pro", page_icon="🛍️", layout="wide")

# --- Custom CSS for Premium Look ---
st.markdown("""
    <style>
    .team-img {
        border-radius: 50%;
        width: 200px;
        height: 200px;
        object-fit: cover;
        margin-bottom: 15px;
        border: 4px solid #f0f2f6;
        box-shadow: 0 8px 20px rgba(0,0,0,0.1);
    }
    .img-container {
        display: flex;
        flex-direction: column;
        align-items: center;
        text-align: center;
        padding: 20px;
        background: #f8f9fa;
        border-radius: 15px;
        margin: 10px;
    }
    .banner-img {
        border-radius: 15px;
        width: 100%;
        height: auto;
        margin-top: 20px;
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
                
                # Apply scaling to ALL models to ensure consistency
                input_processed = scaler.transform(input_data)
                
                prediction = models[model_choice].predict(input_processed)[0]
                
                # Store prediction in session state
                st.session_state['last_prediction'] = {
                    'status': "Likely to Purchase" if prediction == 1 else "Unlikely to Purchase",
                    'data': input_data.to_dict(orient='records')[0],
                    'is_purchase': True if prediction == 1 else False
                }
                
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
                        st.write(f"- Customer shows strong engagement with **{data['TimeSpentOnWebsite']} minutes** on site.")
                        st.write(f"- Annual income of **${data['AnnualIncome']:,}** provides high purchasing power.")
                    else:
                        st.error("📉 **Low Conversion Risk**")
                        if data['TimeSpentOnWebsite'] < 20:
                            st.write(f"- **Engagement Gap**: Only {data['TimeSpentOnWebsite']} mins spent. Customer is likely 'bouncing'.")
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
                    'Score': [data['AnnualIncome']/150000, data['TimeSpentOnWebsite']/60, data['NumberOfPurchases']/20]
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
                            data_processed = scaler.transform(data_for_model)
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
    
    with col1:
        st.markdown(f"""
            <div class="img-container">
                <img src="https://picsum.photos/seed/mujeeb/400/400" class="team-img">
                <h3>Mujeeb Ahmed</h3>
                <p><b>Lead Developer</b></p>
            </div>
        """, unsafe_allow_html=True)
        
    with col2:
        st.markdown(f"""
            <div class="img-container">
                <img src="https://picsum.photos/seed/hassan/400/400" class="team-img">
                <h3>Muhammad Hassan</h3>
                <p><b>ML Engineer</b></p>
            </div>
        """, unsafe_allow_html=True)

# -----------------------------
# Page 3: Our Mentors
# -----------------------------
elif page == "🎓 Our Mentors":
    st.title("🎓 Mentorship")
    st.info("IBA Sukkur University - PITP Program")
    
    t1, t2 = st.columns(2)
    with t1:
        st.markdown(f"""
            <div class="img-container">
                <img src="https://picsum.photos/seed/nabeel/400/400" class="team-img">
                <h3>Sir Nabeel</h3>
                <p>Mentor for Python</p>
            </div>
        """, unsafe_allow_html=True)
    with t2:
        st.markdown(f"""
            <div class="img-container">
                <img src="https://picsum.photos/seed/ismail/400/400" class="team-img">
                <h3>Sir Ismail</h3>
                <p>Mentor for ML</p>
            </div>
        """, unsafe_allow_html=True)
        
    st.success("Special thanks to Sir Altaf Hussain and the entire IBA team.")

# --- Footer ---
st.divider()
st.caption("© 2026 PITP IBA Sukkur Project | Built by Team Mujeeb")
