
import streamlit as st
import pandas as pd
import pickle

# Load the trained model
model = pickle.load(open("stroke_nb_model.pkl", "rb"))

st.title("Stroke Prediction App")

st.markdown("### Enter Patient Information Below:")

# Collect user inputs
age = st.number_input("Age", min_value=0, max_value=120, step=1)
gender = st.selectbox("Gender", ["Male", "Female"])
bmi = st.number_input("BMI", min_value=10.0, max_value=60.0)
glucose = st.number_input("Average Glucose Level", min_value=40.0, max_value=300.0)

# Encode gender to numeric
gender_encoded = 1 if gender == "Male" else 0

# Create a DataFrame for prediction
input_data = pd.DataFrame([[age, gender_encoded, bmi, glucose]],
                          columns=["age", "gender", "bmi", "avg_glucose_level"])

# Predict
if st.button("Predict Stroke Risk"):
    prediction = model.predict(input_data)[0]
    result = "🛑 High Risk of Stroke" if prediction == 1 else "✅ Low Risk of Stroke"
    st.success(f"Prediction: {result}")
