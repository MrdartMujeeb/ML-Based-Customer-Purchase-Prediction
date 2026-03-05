import streamlit as st
import joblib
import pandas as pd
from xgboost import XGBClassifier

# --- Page Config ---
st.set_page_config(page_title="Purchase Intelligence Pro", page_icon="🛍️", layout="wide")

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
    except Exception as e:
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
                input_data = pd.DataFrame([{
                    "Age": age, "Gender": gender, "AnnualIncome": income,
                    "NumberOfPurchases": purchases, "ProductCategory": category,
                    "TimeSpentOnWebsite": time_spent, "LoyaltyProgram": loyalty, "DiscountsAvailed": discounts
                }])
                
                if model_choice in ["Logistic Regression", "XGBoost"]:
                    input_processed = scaler.transform(input_data)
                else:
                    input_processed = input_data
                    
                prediction = models[model_choice].predict(input_processed)[0]
                if prediction == 1: st.success("✅ Result: Customer is likely to PURCHASE.")
                else: st.warning("❌ Result: Customer is NOT likely to purchase.")

        with tab2:
            st.subheader("Bulk Analysis")
            uploaded_file = st.file_uploader("Upload CSV", type="csv")
            if uploaded_file:
                data = pd.read_csv(uploaded_file)
                if st.button("Run Batch Prediction"):
                    if model_choice in ["Logistic Regression", "XGBoost"]:
                        processed = scaler.transform(data)
                    else:
                        processed = data
                    preds = models[model_choice].predict(processed)
                    data['Result'] = preds
                    data['Label'] = data['Result'].map({1: '✅ Purchase', 0: '❌ No Purchase'})
                    st.dataframe(data)
    else:
        st.error("Models not found. Please check your directory.")

# -----------------------------
# Page 2: About Us
# -----------------------------
elif page == "👥 About Us":
    st.title("👥 Meet the Team")
    st.write("The minds behind this Purchase Intelligence Engine.")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Fixed: Changed use_container_width to use_column_width
        st.image("https://picsum.photos/seed/mujeeb/400/400", use_column_width=True)
        st.subheader("Mujeeb Ahmed")
        st.write("**Lead Developer & Data Scientist**")
        st.write("Mujeeb is responsible for the core architecture, model training, and the interactive UI design of this project.")
        
    with col2:
        # Fixed: Changed use_container_width to use_column_width
        st.image("https://picsum.photos/seed/hassan/400/400", use_column_width=True)
        st.subheader("Muhammad Hassan Solangi")
        st.write("**ML Engineer & Researcher**")
        st.write("Hassan focused on feature engineering, model optimization, and ensuring the accuracy of the prediction algorithms.")

# -----------------------------
# Page 3: Our Mentors
# -----------------------------
elif page == "🎓 Our Mentors":
    st.title("🎓 Mentorship & Support")
    
    st.header("🏢 Main Supporter: PITP Program")
    st.info("""
    **People Information Technology Program (PITP)**  
    Hosted at **IBA Sukkur University**, this initiative is proudly supported by the **Sindh Government**. 
    The program aims to empower the youth of Sindh with cutting-edge technology skills.
    """)
    
    st.divider()
    
    st.subheader("👨‍🏫 Our Teachers")
    t1, t2 = st.columns(2)
    with t1:
        st.markdown("### Sir Nabeel")
        st.write("Provided invaluable guidance on Python development and Streamlit integration.")
    with t2:
        st.markdown("### Sir Ismail")
        st.write("Expert mentor in Machine Learning concepts and data preprocessing techniques.")
        
    st.divider()
    
    st.subheader("🙏 Special Thanks")
    st.success("""
    We would like to express our deepest gratitude to **Altaf Hussain (Director)** and his entire dedicated team 
    for providing the platform and resources to make this project a reality.
    """)
    
    # Fixed: Changed use_container_width to use_column_width
    st.image("https://picsum.photos/seed/university/800/300", caption="IBA Sukkur - Center of Excellence", use_column_width=True)

# --- Footer ---
st.sidebar.divider()
st.sidebar.caption("© 2024 PITP IBA Sukkur Project")
