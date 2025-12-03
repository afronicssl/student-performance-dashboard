import streamlit as st
import plotly.express as px

st.title("📈 Visualizations")

if "df" not in st.session_state:
    st.warning("Please upload a dataset on the Dataset Exploration page.")
else:
    df = st.session_state["df"]
    col = st.selectbox("Choose a numeric column to visualize", df.select_dtypes(include=['int64','float64']).columns)
    st.plotly_chart(px.histogram(df, x=col), use_container_width=True)

    if "GPA" in df.columns:
        st.subheader("GPA vs Selected Feature")
        fig2 = px.scatter(df, x=col, y="GPA", size="Absences" if "Absences" in df.columns else None, color="GradeClass" if "GradeClass" in df.columns else None)
        st.plotly_chart(fig2, use_container_width=True)
