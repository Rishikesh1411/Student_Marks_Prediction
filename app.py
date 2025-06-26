import streamlit as st
import pandas as pd
from src.utils import load_model

# Load the trained model
model = load_model("models/best_student_performance_model.pkl")

st.title("Student Marks Prediction App")
st.write("Enter student details to predict the exam score:")

# Input fields for features (must match training features)
study_hours_per_day = st.number_input("Study Hours Per Day", min_value=0.0, max_value=24.0, value=2.0)
attendance_percentage = st.slider("Attendance Percentage", min_value=0, max_value=100, value=80)
mental_health_rating = st.slider("Mental Health Rating (1-10)", min_value=1, max_value=10, value=5)
sleep_hours = st.number_input("Sleep Hours", min_value=0.0, max_value=24.0, value=7.0)
part_time_job = st.selectbox("Part Time Job", options=["No", "Yes"])

# Encode 'part_time_job' as in training (No=0, Yes=1)
part_time_job_encoded = 1 if part_time_job == "Yes" else 0

# Prepare input for prediction
input_dict = {
    "study_hours_per_day": study_hours_per_day,
    "attendance_percentage": attendance_percentage,
    "mental_health_rating": mental_health_rating,
    "sleep_hours": sleep_hours,
    "part_time_job": part_time_job_encoded
}

# Predict and display result
if st.button("Predict Exam Score"):
    input_df = pd.DataFrame([input_dict])
    prediction = model.predict(input_df)[0]
    st.success(f"Predicted Exam Score: {prediction:.2f}")




