import streamlit as st
import pandas as pd
import shap
import joblib
import matplotlib.pyplot as plt
import numpy as np

st.set_page_config(page_title="Model Explainability", layout="wide")
st.title("🧠 Model Explainability (SHAP)")

# -------------------------------
# ✅ Load Data
# -------------------------------
@st.cache_data
def load_data():
    return pd.read_csv("students.csv")

df = load_data()

# -------------------------------
# ✅ Load Model
# -------------------------------
@st.cache_resource
def load_model():
    return joblib.load("linear_model.pkl")

model = load_model()

# -------------------------------
# ✅ Feature Selection
# -------------------------------
features = ["Absences", "StudyTimeWeekly", "ParentalSupport"]
X = df[features]

# -------------------------------
# ✅ SHAP KERNEL EXPLAINER (STABLE FIX)
# -------------------------------
background = X.sample(50, random_state=42)

explainer = shap.KernelExplainer(model.predict, background)
shap_values = explainer.shap_values(X[:50])

# -------------------------------
# ✅ GLOBAL FEATURE IMPORTANCE
# -------------------------------
st.subheader("🌍 Global Feature Importance")

fig, ax = plt.subplots()
shap.summary_plot(shap_values, X[:50], show=False)
st.pyplot(fig)

# -------------------------------
# ✅ LOCAL STUDENT EXPLANATION
# -------------------------------
st.subheader("🎯 Individual Student Prediction Explanation")

student_index = st.slider(
    "Select Student Index",
    min_value=0,
    max_value=len(X[:50]) - 1,
    value=0
)

single_student = X.iloc[[student_index]]

shap_single = explainer.shap_values(single_student)

fig2, ax2 = plt.subplots()
shap.force_plot(
    explainer.expected_value,
    shap_single[0],
    single_student.iloc[0],
    matplotlib=True
)
st.pyplot(fig2)
