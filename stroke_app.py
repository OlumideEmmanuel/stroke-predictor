import streamlit as st
import pandas as pd
import pickle

# Load model
model = pickle.load(open("stroke_nb_model.pkl", "rb"))

st.title("🧠 Stroke Prediction App")
st.markdown("### Enter Patient Details")

# Input fields
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

# Encode inputs (basic manual encoding — adjust as needed to match training)
gender = 1 if gender == "Male" else 0
hypertension = 1 if hypertension == "Yes" else 0
heart_disease = 1 if heart_disease == "Yes" else 0
ever_married = 1 if ever_married == "Yes" else 0
residence_type = 1 if residence_type == "Urban" else 0

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

work_type = work_map[work_type]
smoking_status = smoke_map[smoking_status]

# Prepare DataFrame in correct order
input_data = pd.DataFrame([[gender, age, hypertension, heart_disease, ever_married,
                            work_type, residence_type, glucose, bmi, smoking_status]],
                          columns=['gender', 'age', 'hypertension', 'heart_disease', 'ever_married',
                                   'work_type', 'Residence_type', 'avg_glucose_level', 'bmi', 'smoking_status'])

# Predict
if st.button("Predict Stroke Risk"):
    prediction = model.predict(input_data)[0]
    result = "🛑 High Risk of Stroke" if prediction == 1 else "✅ Low Risk of Stroke"
    st.success(f"Prediction: {result}")
