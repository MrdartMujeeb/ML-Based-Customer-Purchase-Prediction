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
# Custom CSS
# -----------------------------
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
</style>
""", unsafe_allow_html=True)

# -----------------------------
# Feature Order (IMPORTANT)
# -----------------------------
FEATURE_COLUMNS = [
    "Age",
    "Gender",
    "AnnualIncome",
    "NumberOfPurchases",
    "ProductCategory",
    "TimeSpentOnWebsite",
    "LoyaltyProgram",
    "DiscountsAvailed"
]

# -----------------------------
# Load Models and Scaler
# -----------------------------
@st.cache_resource
def load_models():

    try:

        scaler = joblib.load("preprocessor.pkl")

        # load sklearn models
        logistic = joblib.load("logistic_regression_model.pkl")
        rf = joblib.load("random_forest_model.pkl")
        dt = joblib.load("decision_tree_model.pkl")

        # load xgboost json model
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
# Sidebar Navigation
# -----------------------------
with st.sidebar:

    st.title("🚀 Navigation")

    page = st.radio(
        "Go to",
        [
            "🏠 Home (Predictor)",
            "👥 About Us",
            "🎓 Our Mentors"
        ]
    )

    st.divider()

    st.caption("Built with ❤️ by Team Mujeeb")

# -----------------------------
# HOME PAGE
# -----------------------------
if page == "🏠 Home (Predictor)":

    st.title("🛍️ Customer Purchase Prediction")

    st.write(
        "Predict whether a customer will purchase based on behavioral and demographic features."
    )

    if models is None:
        st.stop()

    model_choice = st.selectbox(
        "Select Prediction Model",
        list(models.keys())
    )

    tab1, tab2 = st.tabs(["👤 Single Prediction", "📂 Batch Prediction"])

    # -----------------------------
    # SINGLE PREDICTION
    # -----------------------------
    with tab1:

        st.subheader("Customer Information")

        col1, col2 = st.columns(2)

        with col1:

            age = st.number_input("Age", 18, 70, 35)

            gender = st.selectbox(
                "Gender",
                [0, 1],
                format_func=lambda x: "Female" if x == 0 else "Male"
            )

            income = st.number_input(
                "Annual Income",
                value=60000
            )

            purchases = st.number_input(
                "Number of Previous Purchases",
                0,
                20,
                5
            )

        with col2:

            category = st.selectbox(
                "Product Category",
                [0, 1, 2, 3, 4]
            )

            time_spent = st.number_input(
                "Time Spent On Website (minutes)",
                value=25.0
            )

            loyalty = st.selectbox(
                "Loyalty Program",
                [0, 1],
                format_func=lambda x: "No" if x == 0 else "Yes"
            )

            discounts = st.slider(
                "Discounts Availed",
                0,
                5,
                1
            )

        if st.button("Predict"):

            try:

                # create dataframe
                input_data = pd.DataFrame(
                    [[
                        age,
                        gender,
                        income,
                        purchases,
                        category,
                        time_spent,
                        loyalty,
                        discounts
                    ]],
                    columns=FEATURE_COLUMNS
                )

                # scaling
                input_scaled = scaler.transform(input_data)

                model = models[model_choice]

                prediction = model.predict(input_scaled)[0]

                probability = model.predict_proba(input_scaled)[0][1]

                st.divider()

                if prediction == 1:
                    st.success(
                        f"✅ Customer is Likely to Purchase\n\nConfidence: {probability:.2f}"
                    )
                else:
                    st.warning(
                        f"❌ Customer is Unlikely to Purchase\n\nConfidence: {1-probability:.2f}"
                    )

                st.subheader("Input Summary")

                st.dataframe(input_data)

            except Exception as e:

                st.error(f"Prediction error: {e}")

    # -----------------------------
    # BATCH PREDICTION
    # -----------------------------
    with tab2:

        st.subheader("Upload CSV File")

        uploaded_file = st.file_uploader(
            "Upload customer dataset",
            type="csv"
        )

        if uploaded_file is not None:

            data = pd.read_csv(uploaded_file)

            st.write("Preview of Uploaded Data")

            st.dataframe(data)

            if st.button("Run Batch Prediction"):

                try:

                    if not all(col in data.columns for col in FEATURE_COLUMNS):

                        st.error(
                            f"CSV must contain these columns:\n{FEATURE_COLUMNS}"
                        )

                    else:

                        X = data[FEATURE_COLUMNS]

                        X_scaled = scaler.transform(X)

                        model = models[model_choice]

                        preds = model.predict(X_scaled)

                        data["Prediction"] = preds

                        data["Prediction_Label"] = data["Prediction"].map(
                            {
                                1: "Purchase",
                                0: "No Purchase"
                            }
                        )

                        st.subheader("Prediction Results")

                        st.dataframe(data)

                        csv = data.to_csv(index=False).encode("utf-8")

                        st.download_button(
                            "Download Results",
                            csv,
                            "prediction_results.csv",
                            "text/csv"
                        )

                except Exception as e:

                    st.error(f"Batch prediction failed: {e}")

# -----------------------------
# ABOUT PAGE
# -----------------------------
elif page == "👥 About Us":

    st.title("👥 Team Members")

    col1, col2 = st.columns(2)

    with col1:

        st.markdown("""
        <div class="img-container">
        <img src="https://picsum.photos/seed/mujeeb/400/400" class="team-img">
        <h3>Mujeeb Ahmed</h3>
        <p>Lead Developer</p>
        </div>
        """, unsafe_allow_html=True)

    with col2:

        st.markdown("""
        <div class="img-container">
        <img src="https://picsum.photos/seed/hassan/400/400" class="team-img">
        <h3>Muhammad Hassan</h3>
        <p>ML Engineer</p>
        </div>
        """, unsafe_allow_html=True)

# -----------------------------
# MENTORS PAGE
# -----------------------------
elif page == "🎓 Our Mentors":

    st.title("🎓 Mentorship")

    st.info("IBA Sukkur – PITP Program")

    col1, col2 = st.columns(2)

    with col1:

        st.markdown("""
        <div class="img-container">
        <img src="https://picsum.photos/seed/nabeel/400/400" class="team-img">
        <h3>Sir Nabeel</h3>
        <p>Python Mentor</p>
        </div>
        """, unsafe_allow_html=True)

    with col2:

        st.markdown("""
        <div class="img-container">
        <img src="https://picsum.photos/seed/ismail/400/400" class="team-img">
        <h3>Sir Ismail</h3>
        <p>Machine Learning Mentor</p>
        </div>
        """, unsafe_allow_html=True)

    st.success("Special thanks to Sir Altaf Hussain and the entire IBA Sukkur team.")

# -----------------------------
# Footer
# -----------------------------
st.divider()

st.caption("© 2026 PITP IBA Sukkur | Team Mujeeb")
