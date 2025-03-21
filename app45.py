import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import shap

# Load the trained model
model = joblib.load("hheart_disease_model.pkl")

# Function to make predictions
def predict_heart_disease(data):
    prediction = model.predict([data])
    return "High Risk of Heart Disease" if prediction[0] == 1 else "Low Risk of Heart Disease"

# Streamlit UI
st.set_page_config(page_title="Heart Disease Prediction", layout="wide")

st.title("💓 Heart Disease Prediction App")

# Sidebar - User Input
st.sidebar.header("User Input Features")
age = st.sidebar.slider("Age", 20, 100, 50)
sex = st.sidebar.radio("Sex", ["Male", "Female"])
cp = st.sidebar.selectbox("Chest Pain Type", ["Type 1", "Type 2", "Type 3", "Type 4"])
trestbps = st.sidebar.slider("Resting Blood Pressure", 80, 200, 120)
chol = st.sidebar.slider("Cholesterol Level", 100, 600, 200)
fbs = st.sidebar.radio("Fasting Blood Sugar > 120 mg/dl", ["No", "Yes"])
restecg = st.sidebar.radio("Resting ECG Results", ["Normal", "Abnormal"])
thalach = st.sidebar.slider("Max Heart Rate Achieved", 60, 220, 150)
exang = st.sidebar.radio("Exercise-Induced Angina", ["No", "Yes"])
oldpeak = st.sidebar.slider("ST Depression Induced by Exercise", 0.0, 10.0, 1.0, step=0.1)

# Convert categorical inputs to numeric
sex = 1 if sex == "Male" else 0
fbs = 1 if fbs == "Yes" else 0
exang = 1 if exang == "Yes" else 0

# Make prediction
if st.sidebar.button("Predict"):
    user_data = [age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak]
    result = predict_heart_disease(user_data)
    
    # Display prediction result
    st.subheader("🩺 Prediction Result")
    st.success(f"Prediction: {result}")
    
    # Display user input
    st.write("### 📋 User Input Overview")
    user_df = pd.DataFrame([[age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak]],
                           columns=["Age", "Sex", "Chest Pain Type", "Resting BP", "Cholesterol", "FBS", "ECG", "Max HR", "Angina", "ST Depression"])
    st.table(user_df)

# Visualizations
st.write("### 📊 Data Insights")

# Sample Dataset for Visualization
df = pd.read_csv("heart_statlog_cleveland_hungary_final.csv")

# Distribution of Age
st.write("#### 🏥 Age Distribution")
fig, ax = plt.subplots()
sns.histplot(df["age"], bins=30, kde=True, ax=ax)
st.pyplot(fig)

# Cholesterol vs. Heart Disease Risk
st.write("#### 🍔 Cholesterol Levels & Heart Disease")
fig, ax = plt.subplots()
sns.boxplot(x=df["target"], y=df["cholesterol"], ax=ax)
st.pyplot(fig)

# SHAP Feature Importance
st.write("#### 📊 Feature Importance (SHAP)")
explainer = shap.Explainer(model, df.drop(columns=["target"]))
shap_values = explainer(df.drop(columns=["target"]))

fig, ax = plt.subplots()
shap.summary_plot(shap_values, df.drop(columns=["target"]), show=False)
st.pyplot(fig)