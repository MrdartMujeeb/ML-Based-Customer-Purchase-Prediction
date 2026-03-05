import streamlit as st
import joblib
import pandas as pd
from xgboost import XGBClassifier

# --- Page Config ---
st.set_page_config(page_title="Purchase Predictor Pro", page_icon="🛍️", layout="wide")

# --- CRITICAL: Define exact feature order for model accuracy ---
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
    st.write("Predict whether customers will purchase using advanced ML models.")

    if models:
        # Model Selection
        model_choice = st.selectbox("Select Prediction Model", list(models.keys()))
        
        # Tabs
        tab1, tab2 = st.tabs(["👤 Single Prediction", "📂 Batch Prediction (CSV)"])

        # --- Tab 1: Single Prediction ---
        with tab1:
            st.subheader("Individual Customer Analysis")
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
                if prediction == 1: st.success(f"Result: Customer is likely to PURCHASE.")
                else: st.warning(f"Result: Customer is NOT likely to purchase.")

        # --- Tab 2: Batch Prediction ---
        with tab2:
            st.subheader("Bulk Analysis via CSV")
            uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

            if uploaded_file is not None:
                data = pd.read_csv(uploaded_file)
                
                # 🔍 Search Feature for Uploaded Data
                search_query = st.text_input("🔍 Search in CSV (e.g. filter by Age or Income)", "")
                if search_query:
                    data = data[data.astype(str).apply(lambda x: x.str.contains(search_query, case=False)).any(axis=1)]

                st.write(f"📋 **Preview of Data ({len(data)} records):**")
                # Fixed height ensures scrollbar appears
                st.dataframe(data, use_container_width=True, height=300)

                if st.button("Run Batch Prediction"):
                    try:
                        # Ensure columns are in correct order for the model
                        if all(col in data.columns for col in FEATURE_COLUMNS):
                            data_for_model = data[FEATURE_COLUMNS]
                            
                            # Scaling
                            data_processed = scaler.transform(data_for_model) if model_choice in ["Logistic Regression", "XGBoost"] else data_for_model

                            # Prediction
                            preds = models[model_choice].predict(data_processed)
                            data['Prediction_Result'] = preds
                            data['Prediction_Label'] = data['Prediction_Result'].map({1: '✅ Purchase', 0: '❌ No Purchase'})

                            st.divider()
                            st.subheader("🚀 Prediction Results")
                            
                            # Summary Metrics
                            c1, c2 = st.columns(2)
                            c1.metric("Total Records", len(data))
                            c2.metric("Predicted Purchases", (preds == 1).sum())

                            # Scrollable Results Table
                            st.dataframe(data, use_container_width=True, height=400)

                            # Download Button
                            csv = data.to_csv(index=False).encode('utf-8')
                            st.download_button("📥 Download Results as CSV", csv, "results.csv", "text/csv")
                        else:
                            st.error("CSV columns do not match the required features. Please check the requirements.")
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
st.caption("© 2024 PITP IBA Sukkur Project")
