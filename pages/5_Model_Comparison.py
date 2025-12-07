import streamlit as st
import pandas as pd
import plotly.express as px

st.set_page_config(page_title="Model Comparison", layout="wide")
st.title("📊 Model Performance Comparison")

# ---------------------------------------------------
# ✅ STEP 1: Add Your Model Results Here
# (Use your real results)
# ---------------------------------------------------

model_results = {
    "Model": [
        "Linear Regression",
        "Random Forest",
        "Gradient Boosting"
    ],
    "R2 Score": [
        0.9532,   # Your Linear Regression
        0.9288,   # Your Random Forest
        0.9065    # Your Tuned Random Forest / GB
    ],
    "MSE": [
        0.0401,
        0.0588,
        0.0772
    ],
    "MAE": [
        0.1621,
        0.1874,
        0.2137
    ]
}

df = pd.DataFrame(model_results)

# ---------------------------------------------------
# ✅ STEP 2: SHOW PERFORMANCE TABLE
# ---------------------------------------------------

st.subheader("📋 Model Performance Table")
st.dataframe(df, use_container_width=True)

# ---------------------------------------------------
# ✅ STEP 3: FIND BEST MODEL AUTOMATICALLY
# ---------------------------------------------------

best_model = df.loc[df["R2 Score"].idxmax()]

st.success(
    f"🏆 Best Model: **{best_model['Model']}** "
    f"(R² = {round(best_model['R2 Score'], 3)})"
)

# ---------------------------------------------------
# ✅ STEP 4: BAR CHART — R² SCORE
# ---------------------------------------------------

st.subheader("📈 R² Score Comparison")

fig_r2 = px.bar(
    df,
    x="Model",
    y="R2 Score",
    title="R² Score Comparison",
    text_auto=True
)
st.plotly_chart(fig_r2, use_container_width=True)

# ---------------------------------------------------
# ✅ STEP 5: BAR CHART — MSE
# ---------------------------------------------------

st.subheader("📉 Mean Squared Error (MSE) Comparison")

fig_mse = px.bar(
    df,
    x="Model",
    y="MSE",
    title="MSE Comparison",
    text_auto=True
)
st.plotly_chart(fig_mse, use_container_width=True)

# ---------------------------------------------------
# ✅ STEP 6: BAR CHART — MAE
# ---------------------------------------------------

st.subheader("📉 Mean Absolute Error (MAE) Comparison")

fig_mae = px.bar(
    df,
    x="Model",
    y="MAE",
    title="MAE Comparison",
    text_auto=True
)
st.plotly_chart(fig_mae, use_container_width=True)
