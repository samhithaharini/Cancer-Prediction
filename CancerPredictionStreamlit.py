import streamlit as st
import numpy as np
import joblib

model = joblib.load("cancer_knn.pkl")
scaler = joblib.load("scaler.pkl")

st.title("Cancer Diagnosis Predictor (KNN Model)")

st.write("Enter the details below to predict cancer risk:")

age = st.number_input("Age", 1, 120)
gender = st.selectbox("Gender", ["Male", "Female"])
bmi = st.number_input("BMI", 10.0, 60.0)
smoking = st.selectbox("Smoking", ["Yes", "No"])
genetic_risk = st.number_input("Genetic Risk (1–10)", 1, 10)
physical_activity = st.number_input("Physical Activity (1–5)", 1, 5)
alcohol_intake = st.number_input("Alcohol Intake (1–5)", 1, 5)
cancer_history = st.selectbox("Family Cancer History", ["Yes", "No"])

gender = 1 if gender == "Male" else 0
smoking = 1 if smoking == "Yes" else 0
cancer_history = 1 if cancer_history == "Yes" else 0

if st.button("Predict"):
    features = np.array([[age, gender, bmi, smoking, genetic_risk,
                          physical_activity, alcohol_intake, cancer_history]])

    scaled = scaler.transform(features)
    pred = model.predict(scaled)[0]

    diagnosis = "Cancer Detected" if pred == 1 else "No Cancer"

    st.success(f"Prediction: **{diagnosis}**")
