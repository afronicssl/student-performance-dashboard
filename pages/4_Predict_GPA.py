import streamlit as st
import pandas as pd
import joblib
import numpy as np

st.title("🎯 Predict Student GPA")
st.write("Input student details to predict their GPA using the trained ML model.")

# -----------------------------------------------------------
# LOAD MODEL
# -----------------------------------------------------------
@st.cache_data
def load_model():
    return joblib.load("best_model.pkl")

model = load_model()

# -----------------------------------------------------------
# LOAD DATA (for defaults)
# -----------------------------------------------------------
@st.cache_data
def load_data():
    return pd.read_csv("students.csv")

df = load_data()

# -----------------------------------------------------------
# USER INPUT
# -----------------------------------------------------------
st.subheader("Enter Student Features")

def user_input_features():
    Age = st.number_input("Age", min_value=10, max_value=30, value=18)
    StudyTimeWeekly = st.slider("Weekly Study Time (hours)", 0, 50, 10)
    Absences = st.slider("Absences", 0, 50, 3)
    ParentalSupport = st.selectbox("Parental Support", ["Yes", "No"])
    Tutoring = st.selectbox("Tutoring", ["Yes", "No"])
    Sports = st.selectbox("Sports Participation", ["Yes", "No"])
    Music = st.selectbox("Music Participation", ["Yes", "No"])
    Volunteering = st.selectbox("Volunteering", ["Yes", "No"])

    # Convert categorical to numeric
    ParentalSupport = 1 if ParentalSupport == "Yes" else 0
    Tutoring = 1 if Tutoring == "Yes" else 0
    Sports = 1 if Sports == "Yes" else 0
    Music = 1 if Music == "Yes" else 0
    Volunteering = 1 if Volunteering == "Yes" else 0

    data = {
        "Age": Age,
        "StudyTimeWeekly": StudyTimeWeekly,
        "Absences": Absences,
        "ParentalSupport": ParentalSupport,
        "Tutoring": Tutoring,
        "Sports": Sports,
        "Music": Music,
        "Volunteering": Volunteering
    }

    features = pd.DataFrame(data, index=[0])
    return features

input_df = user_input_features()

# -----------------------------------------------------------
# PREDICTION
# -----------------------------------------------------------
st.subheader("Predicted GPA")

prediction = model.predict(input_df)[0]
st.success(f"Predicted GPA: {prediction:.2f}")

# -----------------------------------------------------------
# OPTIONAL: Confidence / Range
# -----------------------------------------------------------
st.info("⚡ Prediction is based on your trained model; actual GPA may vary.")

