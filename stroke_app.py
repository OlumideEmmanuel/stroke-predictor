import streamlit as st
import pandas as pd
import pickle

# Load model
model = pickle.load(open("stroke_nb_model.pkl", "rb"))

st.title("🧠 Stroke Prediction App")

# --- Collect Inputs ---
gender = st.selectbox("Gender", ["Male", "Female"])
age = st.number_input("Age", 0, 120)
hypertension = st.selectbox("Hypertension", ["No", "Yes"])
heart_disease = st.selectbox("Heart Disease", ["No", "Yes"])
ever_married = st.selectbox("Ever Married?", ["No", "Yes"])
work_type = st.selectbox("Work Type", ["Private", "Self-employed", "Govt_job", "children", "Never_worked"])
residence_type = st.selectbox("Residence Type", ["Urban", "Rural"])
glucose = st.number_input("Average Glucose Level", 40.0, 300.0)
bmi = st.number_input("BMI", 10.0, 60.0)
smoking_status = st.selectbox("Smoking Status", ["never smoked", "formerly smoked", "smokes", "Unknown"])

# --- Encode Inputs ---
gender_val = 1 if gender == "Male" else 0
hypertension_val = 1 if hypertension == "Yes" else 0
heart_disease_val = 1 if heart_disease == "Yes" else 0
ever_married_val = 1 if ever_married == "Yes" else 0
residence_type_val = 1 if residence_type == "Urban" else 0

# These MUST match training encodings
work_map = {
    "Private": 0,
    "Self-employed": 1,
    "Govt_job": 2,
    "children": 3,
    "Never_worked": 4
}
smoke_map = {
    "never smoked": 0,
    "formerly smoked": 1,
    "smokes": 2,
    "Unknown": 3
}
work_type_val = work_map[work_type]
smoking_val = smoke_map[smoking_status]

# --- Build DataFrame in Exact Order & Names ---
input_df = pd.DataFrame([[
    gender_val, age, hypertension_val, heart_disease_val, ever_married_val,
    work_type_val, residence_type_val, glucose, bmi, smoking_val
]], columns=[
    'gender', 'age', 'hypertension', 'heart_disease', 'ever_married',
    'work_type', 'Residence_type', 'avg_glucose_level', 'bmi', 'smoking_status'
])

# --- Predict ---
if st.button("Predict Stroke Risk"):
    prediction = model.predict(input_df)[0]
    result = "🛑 High Risk of Stroke" if prediction == 1 else "✅ Low Risk of Stroke"
    st.success(f"Prediction: {result}")
