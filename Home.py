import streamlit as st

st.set_page_config(page_title="Student Performance Analytics", layout="wide")
st.title("🎓 Student Performance Analytics Dashboard")
st.write("""
Welcome! This demo dashboard helps explore student academic data and build models to predict GPA.
Use the menu on the left to navigate between pages (Dataset, Visualizations, Train model, Predict).
""")
# Sidebar Filters
st.sidebar.header("Filter Students")

gender_filter = st.sidebar.multiselect(
    "Select Gender",
    options=df["Gender"].unique(),
    default=df["Gender"].unique()
)

ethnicity_filter = st.sidebar.multiselect(
    "Select Ethnicity",
    options=df["Ethnicity"].unique(),
    default=df["Ethnicity"].unique()
)

studytime_filter = st.sidebar.slider(
    "Weekly Study Time",
    min_value=int(df.StudyTimeWeekly.min()),
    max_value=int(df.StudyTimeWeekly.max()),
    value=(df.StudyTimeWeekly.min(), df.StudyTimeWeekly.max())
)

