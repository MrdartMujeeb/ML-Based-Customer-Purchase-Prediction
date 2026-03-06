import streamlit as st
import joblib
import pandas as pd
from xgboost import XGBClassifier

# -----------------------------
# Page Configuration
# -----------------------------
st.set_page_config(
    page_title="Purchase Intelligence Pro",
    page_icon="🛍️",
    layout="wide"
)

# -----------------------------
# Feature Order
# -----------------------------
FEATURE_COLUMNS = [
    "Age", "Gender", "AnnualIncome", "NumberOfPurchases",
    "ProductCategory", "TimeSpentOnWebsite", "LoyaltyProgram", "DiscountsAvailed"
]

# -----------------------------
# Load Models
# -----------------------------
@st.cache_resource
def load_models():
    try:
        scaler = joblib.load("preprocessor.pkl")
        logistic = joblib.load("logistic_regression_model.pkl")
        rf = joblib.load("random_forest_model.pkl")
        dt = joblib.load("decision_tree_model.pkl")
        xgb = XGBClassifier()
        xgb.load_model("xgboost_model.json")

        models = {
            "Logistic Regression": logistic,
            "Random Forest": rf,
            "Decision Tree": dt,
            "XGBoost": xgb
        }
        return scaler, models
    except Exception as e:
        st.error(f"Model loading failed: {e}")
        return None, None

scaler, models = load_models()

# -----------------------------
# Sidebar
# -----------------------------
with st.sidebar:
    st.title("🚀 Navigation")
    page = st.radio("Go to", ["🏠 Home (Predictor)", "👥 About Us", "🎓 Our Mentors"])
    st.divider()
    st.caption("Built with ❤️ by Team Mujeeb")

# -----------------------------
# HOME PAGE
# -----------------------------
if page == "🏠 Home (Predictor)":
    st.title("🛍️ Purchase Intelligence Pro")
    st.markdown("""
### AI-Powered Customer Purchase Prediction System
This app predicts whether a customer is likely to make a purchase.
- Uses Logistic Regression, Decision Tree, Random Forest, and XGBoost
- Single and Batch prediction supported
""")
    st.divider()

    st.subheader("📊 Model Performance Comparison")
    performance_data = pd.DataFrame({
        "Model": ["Logistic Regression", "Decision Tree", "Random Forest", "XGBoost"],
        "Accuracy": [0.82, 0.78, 0.91, 0.90]  # Random Forest on top
    })
    st.bar_chart(performance_data.set_index("Model"))
    st.divider()

    if models is None:
        st.stop()

    model_choice = st.selectbox("Select Prediction Model", list(models.keys()))
    tab1, tab2 = st.tabs(["👤 Single Prediction", "📂 Batch Prediction"])

    # SINGLE PREDICTION
    with tab1:
        col1, col2 = st.columns(2)
        with col1:
            age = st.number_input("Age", 18, 70, 35)
            gender = st.selectbox("Gender", [0, 1], format_func=lambda x: "Female" if x == 0 else "Male")
            income = st.number_input("Annual Income", value=60000)
            purchases = st.number_input("Number of Previous Purchases", 0, 20, 5)
        with col2:
            category = st.selectbox("Product Category", [0,1,2,3,4])
            time_spent = st.number_input("Time Spent On Website (minutes)", value=25.0)
            loyalty = st.selectbox("Loyalty Program", [0,1], format_func=lambda x:"No" if x==0 else "Yes")
            discounts = st.slider("Discounts Availed", 0, 5, 1)

        if st.button("Predict"):
            try:
                input_data = pd.DataFrame([[age, gender, income, purchases, category, time_spent, loyalty, discounts]],
                                          columns=FEATURE_COLUMNS)
                input_scaled = scaler.transform(input_data)
                model = models[model_choice]
                prediction = model.predict(input_scaled)[0]
                probability = model.predict_proba(input_scaled)[0][1]
                st.divider()
                if prediction==1:
                    st.success(f"✅ Customer is Likely to Purchase\nConfidence: {probability:.2f}")
                else:
                    st.warning(f"❌ Customer is Unlikely to Purchase\nConfidence: {1-probability:.2f}")
                st.subheader("Input Summary")
                st.dataframe(input_data)
            except Exception as e:
                st.error(f"Prediction error: {e}")

    # BATCH PREDICTION
    with tab2:
        uploaded_file = st.file_uploader("Upload CSV File", type="csv")
        if uploaded_file:
            data = pd.read_csv(uploaded_file)
            st.dataframe(data)
            if st.button("Run Batch Prediction"):
                try:
                    if not all(col in data.columns for col in FEATURE_COLUMNS):
                        st.error("CSV columns do not match required features")
                    else:
                        X_scaled = scaler.transform(data[FEATURE_COLUMNS])
                        preds = models[model_choice].predict(X_scaled)
                        data["Prediction"] = preds
                        data["Prediction_Label"] = data["Prediction"].map({1:"Purchase",0:"No Purchase"})
                        st.dataframe(data)
                        st.download_button("Download Results", data.to_csv(index=False).encode("utf-8"),
                                           "prediction_results.csv", "text/csv")
                except Exception as e:
                    st.error(f"Batch prediction failed: {e}")

# -----------------------------
# ABOUT US PAGE
# -----------------------------
elif page == "👥 About Us":
    st.title("👥 Meet The Team")
    
    st.subheader("Team Members")
    st.markdown("""
- **Mujeeb Ahmed** – Lead Developer  
- **Muhammad Hassan Solangi** – AI Engineer
""")
    st.divider()
    
    st.subheader("📌 Project Overview")
    st.markdown("""
In this project, we developed **Purchase Intelligence Pro**, an AI-powered system that predicts whether a customer 
is likely to make a purchase using machine learning models including **Logistic Regression, Decision Tree, Random Forest, and XGBoost**.
The app supports **single customer prediction** as well as **batch prediction from CSV files**, helping businesses 
identify high-value customers efficiently.
""")
    st.divider()
    
    st.subheader("💡 What We Learned")
    st.markdown("""
- **Python Programming**: From basics to advanced concepts  
- **Data Science Libraries**:  
  - `NumPy` for numerical computations  
  - `Pandas` for data manipulation  
  - `Matplotlib` & `Seaborn` for data visualization  
  - `Scikit-learn` for machine learning and preprocessing  
- **Machine Learning Concepts**: Supervised learning, model evaluation, feature scaling, model comparison  
- **Streamlit**: Building and deploying ML apps  
- **Freelancing & Entrepreneurship**: Basic business, earning online, freelancing skills  
- **Presentation Skills**: How to present projects and technical concepts clearly
""")
    st.divider()

# -----------------------------
# MENTORS PAGE
# -----------------------------
elif page == "🎓 Our Mentors":
    st.title("🎓 Mentorship")
    st.info("IBA Sukkur – PITP Program")
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Sir Nabeel")
        st.write("Python Mentor")
    with col2:
        st.subheader("Sir Ismail")
        st.write("Machine Learning Mentor")
    st.success("Special thanks to Sir Altaf Hussain and the entire IBA Sukkur team.")

# -----------------------------
# Footer
# -----------------------------
st.divider()
st.caption("© 2026 PITP IBA Sukkur Project | Built by Team Mujeeb")
