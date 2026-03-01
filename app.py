import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt

# ===============================
# PAGE CONFIG
# ===============================
st.set_page_config(
    page_title="Bank Churn Intelligence System",
    page_icon="🏦",
    layout="wide"
)

# ===============================
# LOAD MODEL
# ===============================
model = joblib.load("churn_model.pkl")

# ===============================
# CUSTOM STYLE
# ===============================
st.markdown("""
    <style>
    .main {background-color: #f4f6f9;}
    .stButton>button {
        background-color: #002b5c;
        color: white;
        font-size: 18px;
        border-radius: 8px;
        height: 3em;
        width: 100%;
    }
    </style>
""", unsafe_allow_html=True)

# ===============================
# HEADER
# ===============================
st.title("🏦 European Bank Churn Intelligence Dashboard")
st.markdown("### AI-Powered Customer Risk Scoring & Explainability System")

# ===============================
# INPUT LAYOUT
# ===============================
col1, col2 = st.columns(2)

with col1:
    st.subheader("📌 Customer Profile")
    year = st.number_input("Year", 2000, 2030, 2023)
    credit_score = st.slider("Credit Score", 300, 900, 650)
    age = st.slider("Age", 18, 100, 35)
    tenure = st.slider("Tenure (Years with Bank)", 0, 20, 5)
    balance = st.number_input("Balance", 0.0, 500000.0, 50000.0)
    salary = st.number_input("Estimated Salary", 0.0, 500000.0, 60000.0)

with col2:
    st.subheader("📌 Banking Relationship")
    num_products = st.slider("Number of Products", 1, 4, 2)
    has_card = st.selectbox("Has Credit Card", [0, 1])
    is_active = st.selectbox("Is Active Member", [0, 1])
    geography = st.selectbox("Geography", ["France", "Germany", "Spain"])
    gender = st.selectbox("Gender", ["Female", "Male"])

# ===============================
# FEATURE ENGINEERING
# ===============================
balance_salary_ratio = balance / (salary + 1)
product_density = num_products / (tenure + 1)
age_tenure_ratio = age / (tenure + 1)
engagement_product = is_active * num_products

geo_germany = 1 if geography == "Germany" else 0
geo_spain = 1 if geography == "Spain" else 0
gender_male = 1 if gender == "Male" else 0

# ===============================
# CREATE INPUT DATAFRAME
# ===============================
input_data = pd.DataFrame([[
    year,
    credit_score,
    age,
    tenure,
    balance,
    num_products,
    has_card,
    is_active,
    salary,
    balance_salary_ratio,
    product_density,
    age_tenure_ratio,
    engagement_product,
    geo_germany,
    geo_spain,
    gender_male
]], columns=[
    'Year',
    'CreditScore',
    'Age',
    'Tenure',
    'Balance',
    'NumOfProducts',
    'HasCrCard',
    'IsActiveMember',
    'EstimatedSalary',
    'Balance_Salary_Ratio',
    'Product_Density',
    'Age_Tenure_Ratio',
    'Engagement_Product',
    'Geography_Germany',
    'Geography_Spain',
    'Gender_Male'
])

# ===============================
# PREDICTION BLOCK
# ===============================
if st.button("🔎 Analyze Churn Risk"):

    probability = model.predict_proba(input_data)[0][1]
    probability_percent = round(probability * 100, 2)

    st.markdown("---")
    st.subheader("📊 Risk Assessment Result")

    # Risk Meter
    st.progress(int(probability_percent))

    if probability > 0.7:
        st.error(f"⚠️ High Risk Customer ({probability_percent}%)")
        risk_label = "High Risk"
    elif probability > 0.3:
        st.warning(f"⚠️ Medium Risk Customer ({probability_percent}%)")
        risk_label = "Medium Risk"
    else:
        st.success(f"✅ Low Risk Customer ({probability_percent}%)")
        risk_label = "Low Risk"

    # ===============================
    # FEATURE IMPORTANCE
    # ===============================
    st.markdown("### 🔍 Top Influencing Features")

    importances = model.feature_importances_
    feature_names = input_data.columns

    importance_df = pd.DataFrame({
        "Feature": feature_names,
        "Importance": importances
    }).sort_values(by="Importance", ascending=False).head(8)

    fig1, ax1 = plt.subplots()
    ax1.barh(importance_df["Feature"], importance_df["Importance"])
    ax1.invert_yaxis()
    plt.xlabel("Importance Score")
    plt.title("Top 8 Important Features")
    st.pyplot(fig1)

    # ===============================
    # SHAP EXPLANATION
    # ===============================
    st.markdown("### 🧠 Why This Customer Has This Risk?")

    explainer = shap.Explainer(model)
    shap_values = explainer(input_data)

    # Select class 1 (Churn class)
    shap_values_class1 = shap_values[:, :, 1]

    fig2 = plt.figure()
    shap.plots.waterfall(shap_values_class1[0], show=False)
    st.pyplot(fig2)

    # ===============================
    # CUSTOMER SUMMARY
    # ===============================
    st.markdown("### 📋 Customer Summary")

    st.write(f"""
    - Age: {age}
    - Geography: {geography}
    - Active Member: {is_active}
    - Products Used: {num_products}
    - Risk Level: **{risk_label}**
    """)

        # ===============================
    # SCENARIO-BASED WHAT-IF SIMULATION
    # ===============================
    st.markdown("### 🔮 Scenario-Based Churn Risk Simulation")

    # Create baseline copy
    simulated_data = input_data.copy()

    # Scenario 1: Make Customer Active
    simulated_active = simulated_data.copy()
    simulated_active['IsActiveMember'] = 1
    simulated_active['Engagement_Product'] = 1 * simulated_active['NumOfProducts']
    prob_active = model.predict_proba(simulated_active)[0][1]

    # Scenario 2: Increase Products by 1
    simulated_product = simulated_data.copy()
    simulated_product['NumOfProducts'] += 1
    simulated_product['Product_Density'] = simulated_product['NumOfProducts'] / (simulated_product['Tenure'] + 1)
    simulated_product['Engagement_Product'] = simulated_product['IsActiveMember'] * simulated_product['NumOfProducts']
    prob_product = model.predict_proba(simulated_product)[0][1]

    # Scenario 3: Reduce Balance by 20%
    simulated_balance = simulated_data.copy()
    simulated_balance['Balance'] *= 0.8
    simulated_balance['Balance_Salary_Ratio'] = simulated_balance['Balance'] / (simulated_balance['EstimatedSalary'] + 1)
    prob_balance = model.predict_proba(simulated_balance)[0][1]

    # Comparison Table
    comparison_df = pd.DataFrame({
        "Scenario": [
            "Current Risk",
            "If Customer Becomes Active",
            "If Products Increased by 1",
            "If Balance Reduced by 20%"
        ],
        "Churn Probability (%)": [
            round(probability * 100, 2),
            round(prob_active * 100, 2),
            round(prob_product * 100, 2),
            round(prob_balance * 100, 2)
        ]
    })

    st.dataframe(comparison_df)

    # Strategic Recommendation
    st.markdown("### 🏦 Recommended Retention Strategy")

    best_option = min(prob_active, prob_product, prob_balance)

    if best_option == prob_active:
        st.success("✔ Recommended Action: Activate customer engagement program.")
    elif best_option == prob_product:
        st.success("✔ Recommended Action: Cross-sell one additional banking product.")
    else:
        st.success("✔ Recommended Action: Offer balance optimization advisory.")