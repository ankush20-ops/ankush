import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load the trained model
model = joblib.load("hheart_disease_model.pkl")

# Function to make predictions
def predict_heart_disease(data):
    prediction = model.predict([data])
    return "High Risk of Heart Disease" if prediction[0] == 1 else "Low Risk of Heart Disease"

# Streamlit UI
st.title("Heart Disease Prediction App")

# Collect user input
age = st.number_input("Age", min_value=20, max_value=100, value=50)
sex = st.selectbox("Sex", ["Male", "Female"])
cp = st.selectbox("Chest Pain Type", ["Type 1", "Type 2", "Type 3", "Type 4"])
trestbps = st.number_input("Resting Blood Pressure", min_value=80, max_value=200, value=120)
chol = st.number_input("Cholesterol Level", min_value=100, max_value=600, value=200)
fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", ["No", "Yes"])
restecg = st.selectbox("Resting ECG Results", ["Normal", "Abnormal"])
thalach = st.number_input("Max Heart Rate Achieved", min_value=60, max_value=220, value=150)
exang = st.selectbox("Exercise-Induced Angina", ["No", "Yes"])
oldpeak = st.number_input("ST Depression Induced by Exercise", min_value=0.0, max_value=10.0, value=1.0, step=0.1)

# Convert categorical inputs to numeric
sex = 1 if sex == "Male" else 0
fbs = 1 if fbs == "Yes" else 0
exang = 1 if exang == "Yes" else 0

# Make prediction
if st.button("Predict"):
    user_data = [age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak]
    result = predict_heart_disease(user_data)
    st.success(f"Prediction: {result}")