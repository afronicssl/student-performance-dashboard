import streamlit as st
import pandas as pd

st.title("📊 Dataset Exploration")

uploaded = st.file_uploader("Upload your student dataset (CSV)", type="csv")

if uploaded:
    df = pd.read_csv(uploaded)
    st.session_state["df"] = df  # store dataset for other pages
    st.subheader("Preview")
    st.dataframe(df.head())
    st.write("**Summary Statistics**")
    st.write(df.describe(include='all'))
else:
    st.info("Upload a CSV file to continue.")
