import streamlit as st
import pandas as pd
import pickle

# Load model
model = pickle.load(open("stroke_nb_model.pkl", "rb"))

st.title("🧠 Stroke Prediction App")
st.markdown("Enter your health and lifestyle information below.")

# --- User Inputs ---
gender = st.selectbox("Gender", ["Male", "Female"])
age = st.number_input("Age", 0, 120)
hypertension = st.selectbox("Do you have hypertension?", ["No", "Yes"])
heart_disease = st.selectbox("Do you have any heart disease?", ["No", "Yes"])
ever_married = st.selectbox("Have you ever been married?", ["No", "Yes"])
work_type = st.selectbox("Work Type", ["Private", "Self-employed", "Govt_job", "children", "Never_worked"])
residence_type = st.selectbox("Residence Type", ["Urban", "Rural"])
smoking_status = st.selectbox("Smoking Status", ["never smoked", "formerly smoked", "smokes", "Unknown"])

# --- Glucose ---
st.markdown("### Average Glucose Level")
know_glucose = st.checkbox("I don't know my glucose level")
if know_glucose:
    glucose = 106.15  # Replace with your dataset's average if needed
    st.info(f"Using average glucose level: {glucose}")
else:
    glucose = st.number_input("Enter your average glucose level", 40.0, 300.0)

# --- BMI ---
st.markdown("### BMI (Body Mass Index)")
know_bmi = st.checkbox("I don't know my BMI")
if know_bmi:
    weight = st.number_input("Weight (kg)", 30.0, 200.0)
    height_cm = st.number_input("Height (cm)", 100.0, 250.0)
    height_m = height_cm / 100
    bmi = round(weight / (height_m ** 2), 2) if height_m > 0 else 0
    st.info(f"Calculated BMI: {bmi}")
else:
    bmi = st.number_input("Enter your BMI", 10.0, 60.0)

# --- Encode Inputs ---
gender_val = 1 if gender == "Male" else 0
hypertension_val = 1 if hypertension == "Yes" else 0
heart_disease_val = 1 if heart_disease == "Yes" else 0
ever_married_val = 1 if ever_married == "Yes" else 0
residence_type_val = 1 if residence_type == "Urban" else 0

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

# --- Match Model's Feature Format ---
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
