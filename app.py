import streamlit as st
import joblib
import pandas as pd
from xgboost import XGBClassifier

# --- Page Config ---
st.set_page_config(page_title="Purchase Predictor Pro", page_icon="🛍️", layout="wide")

st.title("🛍️ Customer Purchase Prediction")
st.write("Predict whether customers will purchase using advanced ML models.")

# -----------------------------
# Load Models and Preprocessor
# -----------------------------
@st.cache_resource
def load_files():
    try:
        scaler = joblib.load("preprocessor.pkl")
        # Load XGBoost model correctly
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
        st.error(f"Error loading models: {e}")
        return None, None

scaler, models = load_files()

# -----------------------------
# Model Selection (Global)
# -----------------------------
model_choice = st.selectbox(
    "Select Prediction Model",
    list(models.keys()) if models else ["No Models Found"]
)

# -----------------------------
# Tabs for Navigation
# -----------------------------
tab1, tab2 = st.tabs(["👤 Single Prediction", "📂 Batch Prediction (CSV)"])

# --- Tab 1: Single Prediction (Your Original Code) ---
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

        if model_choice in ["Logistic Regression", "XGBoost"]:
            input_processed = scaler.transform(input_data)
        else:
            input_processed = input_data

        model = models[model_choice]
        prediction = model.predict(input_processed)

        if prediction[0] == 1:
            st.success(f"Result: Customer is likely to PURCHASE.")
        else:
            st.warning(f"Result: Customer is NOT likely to purchase.")

# --- Tab 2: Batch Prediction (New Feature) ---
with tab2:
    st.subheader("Bulk Analysis via CSV")
    st.write("Upload a CSV file containing customer records. Ensure columns match the required features.")
    
    # Requirement Help
    with st.expander("See CSV Column Requirements"):
        st.write("Your CSV must contain these columns exactly:")
        st.code("Age, Gender, AnnualIncome, NumberOfPurchases, ProductCategory, TimeSpentOnWebsite, LoyaltyProgram, DiscountsAvailed")

    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)
        st.write("📋 **Preview of Uploaded Data:**")
        st.dataframe(data.head())

        if st.button("Run Batch Prediction"):
            try:
                # 1. Scaling
                if model_choice in ["Logistic Regression", "XGBoost"]:
                    data_processed = scaler.transform(data)
                else:
                    data_processed = data

                # 2. Prediction
                model = models[model_choice]
                predictions = model.predict(data_processed)

                # 3. Results Formatting
                data['Prediction_Result'] = predictions
                data['Prediction_Label'] = data['Prediction_Result'].map({1: '✅ Purchase', 0: '❌ No Purchase'})

                st.divider()
                st.subheader("🚀 Prediction Results")
                
                # Show summary metrics
                total = len(data)
                purchases_count = (predictions == 1).sum()
                st.metric("Total Records", total)
                st.metric("Predicted Purchases", purchases_count, delta=f"{(purchases_count/total)*100:.1f}% of total")

                # Show full table
                st.dataframe(data)

                # Download Button
                csv = data.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="📥 Download Results as CSV",
                    data=csv,
                    file_name='prediction_results.csv',
                    mime='text/csv',
                )
            except Exception as e:
                st.error(f"An error occurred during processing: {e}")
                st.info("Check if your CSV columns match the required names exactly.")

# --- Footer ---
st.divider()
st.caption("Purchase Intelligence Engine | Built with Streamlit")
