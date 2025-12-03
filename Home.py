import streamlit as st
import pandas as pd
import plotly.express as px

# -----------------------------------------------------------
# PAGE CONFIG
# -----------------------------------------------------------
st.set_page_config(page_title="Student Performance Analytics", layout="wide")

# -----------------------------------------------------------
# LOAD DATA
# -----------------------------------------------------------
@st.cache_data
def load_data():
    return pd.read_csv("students.csv")

df = load_data()

# -----------------------------------------------------------
# PAGE TITLE
# -----------------------------------------------------------
st.title("🎓 Student Performance Analytics Dashboard")
st.write("""
Welcome! This dashboard lets you explore student academic performance data, 
analyze trends, and apply ML models to predict GPA.

Use the sidebar filters or the menu to explore additional pages.
""")

# -----------------------------------------------------------
# SIDEBAR FILTERS
# -----------------------------------------------------------
st.sidebar.header("🔎 Filter Students")

gender_filter = st.sidebar.multiselect(
    "Gender",
    options=df["Gender"].unique(),
    default=df["Gender"].unique()
)

ethnicity_filter = st.sidebar.multiselect(
    "Ethnicity",
    options=df["Ethnicity"].unique(),
    default=df["Ethnicity"].unique()
)

studytime_filter = st.sidebar.slider(
    "Weekly Study Time",
    min_value=int(df["StudyTimeWeekly"].min()),
    max_value=int(df["StudyTimeWeekly"].max()),
    value=(int(df["StudyTimeWeekly"].min()), int(df["StudyTimeWeekly"].max()))
)

# -----------------------------------------------------------
# APPLY FILTERS
# -----------------------------------------------------------
df_filtered = df[
    (df["Gender"].isin(gender_filter)) &
    (df["Ethnicity"].isin(ethnicity_filter)) &
    (df["StudyTimeWeekly"].between(studytime_filter[0], studytime_filter[1]))
]

# -----------------------------------------------------------
# METRICS
# -----------------------------------------------------------
st.subheader("📊 Key Metrics")

col1, col2, col3 = st.columns(3)

col1.metric("Total Students", df_filtered.shape[0])
col2.metric("Avg GPA", round(df_filtered["GPA"].mean(), 2))
col3.metric("Avg Absences", round(df_filtered["Absences"].mean(), 2))

# -----------------------------------------------------------
# GPA DISTRIBUTION
# -----------------------------------------------------------
st.subheader("📘 GPA Distribution")
fig = px.histogram(df_filtered, x="GPA", nbins=20)
st.plotly_chart(fig, use_container_width=True)

# -----------------------------------------------------------
# ABSENCES VS GPA
# -----------------------------------------------------------
st.subheader("📉 Absences vs GPA")
fig2 = px.scatter(df_filtered, x="Absences", y="GPA", color="Gender")
st.plotly_chart(fig2, use_container_width=True)

# -----------------------------------------------------------
# DOWNLOAD FILTERED DATA
# -----------------------------------------------------------
st.subheader("⬇️ Download Filtered Data")

csv = df_filtered.to_csv(index=False).encode('utf-8')

st.download_button(
    label="Download CSV",
    data=csv,
    file_name='filtered_student_data.csv',
    mime='text/csv'
)

# -----------------------------------------------------------
# STUDENT SEARCH
# -----------------------------------------------------------
st.subheader("🔍 Search Student by ID")

student_id = st.text_input("Enter Student ID")

if student_id:
    student_info = df[df["StudentID"].astype(str) == student_id]

    if not student_info.empty:
        st.write("### Student Information")
        st.write(student_info)
    else:
        st.warning("Student not found.")
