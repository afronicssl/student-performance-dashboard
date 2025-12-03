import streamlit as st
import numpy as np
import pandas as pd

st.title("📝 Predict GPA for a New Student")

if "model" not in st.session_state:
    st.warning("Train the model on the Train Model page first.")
else:
    model = st.session_state["model"]
    features = st.session_state["model_features"]

    st.write("Enter feature values for prediction:")
    values = []
    for feat in features:
        val = st.number_input(f"{feat}", value=float(0))
        values.append(val)

    if st.button("Predict GPA"):
        pred = model.predict([values])[0]
        st.success(f"Predicted GPA: {pred:.2f}")
