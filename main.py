import streamlit as st
import pandas as pd
import numpy as np
import joblib

st.set_page_config(page_title="Salary Predictor", layout="centered")

# Load Trained Model
model = joblib.load("model/salary_predictor.pkl")

st.title("💼 Salary Prediction App")
st.markdown("Predict **Estimated Salary** based on user information")

# --- Input Fields ---
credit_score = st.number_input("Credit Score", min_value=300, max_value=850, value=600)
geography = st.selectbox("Geography", ['France', 'Spain', 'Germany'])
gender = st.selectbox("Gender", ['Female', 'Male'])
age = st.slider("Age", 18, 100, 30)
tenure = st.slider("Tenure (Years with Bank)", 0, 10, 3)
balance = st.number_input("Account Balance", value=50000.0)
num_of_products = st.selectbox("Number of Products", [1, 2, 3, 4])
has_cr_card = st.selectbox("Has Credit Card", [0, 1])
is_active_member = st.selectbox("Is Active Member", [0, 1])
exited = st.selectbox("Has Customer Exited?", [0, 1])  # ✅ Added as required by model

# --- Encode Inputs ---
geo_map = {'France': 0, 'Spain': 2, 'Germany': 1}
gender_map = {'Female': 0, 'Male': 1}

# --- Create Input DataFrame ---
input_data = pd.DataFrame({
    'CreditScore': [credit_score],
    'Geography': [geo_map[geography]],
    'Gender': [gender_map[gender]],
    'Age': [age],
    'Tenure': [tenure],
    'Balance': [balance],
    'NumOfProducts': [num_of_products],
    'HasCrCard': [has_cr_card],
    'IsActiveMember': [is_active_member],
    'Exited': [exited]  # ✅ Now matches model training input
})

# --- Predict ---
if st.button("🔍 Predict Salary"):
    prediction = model.predict(input_data)[0]
    st.success(f"💰 Predicted Estimated Salary: **${prediction:,.2f}**")

