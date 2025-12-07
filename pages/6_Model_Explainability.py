import streamlit as st
import pandas as pd
import shap
import joblib
import matplotlib.pyplot as plt

st.set_page_config(page_title="Model Explainability", layout="wide")
st.title("🧠 Model Explainability (SHAP)")

# ----------------------------------------------------
# ✅ Load Data
# ----------------------------------------------------

@st.cache_data
def load_data():
    return pd.read_csv("students.csv")

df = load_data()

# ----------------------------------------------------
# ✅ Load Model
# ----------------------------------------------------

@st.cache_resource
def load_model():
    return joblib.load("linear_model.pkl")

model = load_model()

# ----------------------------------------------------
# ✅ Feature Selection
# ----------------------------------------------------

features = ["Absences", "StudyTimeWeekly", "ParentalSupport"]

X = df[features]

# ----------------------------------------------------
# ✅ SHAP EXPLAINER
# ----------------------------------------------------

explainer = shap.Explainer(model, X)
shap_values = explainer(X)

# ----------------------------------------------------
# ✅ Global Feature Importance
# ----------------------------------------------------

st.subheader("🌍 Global Feature Importance")

fig, ax = plt.subplots()
shap.summary_plot(shap_values, X, show=False)
st.pyplot(fig)

# ----------------------------------------------------
# ✅ Local Explanation (Single Student)
# ----------------------------------------------------

st.subheader("🎯 Individual Student Prediction Explanation")

student_index = st.slider(
    "Select Student Index",
    min_value=0,
    max_value=len(X)-1,
    value=0
)

single_student = X.iloc[[student_index]]

shap_single = explainer(single_student)

fig2, ax2 = plt.subplots()
shap.waterfall_plot(shap_single[0], show=False)
st.pyplot(fig2)
