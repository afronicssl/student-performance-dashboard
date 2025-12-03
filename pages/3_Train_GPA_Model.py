import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import numpy as np
import joblib
import plotly.express as px

st.title("🧠 Train GPA Prediction Model")
st.write("Select features and train multiple ML models to predict student GPA.")

# -----------------------------------------------------------
# LOAD DATA
# -----------------------------------------------------------
@st.cache_data
def load_data():
    return pd.read_csv("students.csv")

df = load_data()

st.subheader("📌 Dataset Preview")
st.dataframe(df.head())

# -----------------------------------------------------------
# SIDEBAR — SELECT FEATURES
# -----------------------------------------------------------
st.sidebar.header("⚙️ Model Settings")

all_features = ['Age','StudyTimeWeekly','Absences','ParentalSupport','Tutoring',
                'Sports','Music','Volunteering']

selected_features = st.sidebar.multiselect(
    "Select Features",
    options=all_features,
    default=all_features
)

if len(selected_features) == 0:
    st.warning("Please select at least one feature.")
    st.stop()

# -----------------------------------------------------------
# PREPARE DATA
# -----------------------------------------------------------
X = df[selected_features]
y = df["GPA"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

st.write(f"### Selected Features: {selected_features}")

# -----------------------------------------------------------
# TRAIN MODELS
# -----------------------------------------------------------
models = {
    "Linear Regression": LinearRegression(),
    "Random Forest": RandomForestRegressor(n_estimators=300, random_state=42),
    "Gradient Boosting": GradientBoostingRegressor(random_state=42)
}

results = {}

st.subheader("📊 Model Performance")

for name, model in models.items():
    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    r2 = r2_score(y_test, preds)
    mse = mean_squared_error(y_test, preds)
    mae = mean_absolute_error(y_test, preds)

    results[name] = {"model": model, "r2": r2, "mse": mse, "mae": mae}

    st.write(f"### {name}")
    st.write(f"**R² Score:** `{r2:.4f}`")
    st.write(f"**MSE:** `{mse:.4f}`")
    st.write(f"**MAE:** `{mae:.4f}`")
    st.markdown("---")

# -----------------------------------------------------------
# SELECT BEST MODEL
# -----------------------------------------------------------
best_model_name = max(results, key=lambda x: results[x]["r2"])
best_model = results[best_model_name]["model"]

st.success(f"🏆 Best Model: **{best_model_name}** (R² = {results[best_model_name]['r2']:.4f})")

# -----------------------------------------------------------
# FEATURE IMPORTANCE FOR TREE MODELS
# -----------------------------------------------------------
if best_model_name != "Linear Regression":
    st.subheader("📌 Feature Importance")
    importance = best_model.feature_importances_
    fi_df = pd.DataFrame({"Feature": selected_features, "Importance": importance})

    fig = px.bar(fi_df, x="Feature", y="Importance", title="Feature Importance")
    st.plotly_chart(fig, use_container_width=True)

# -----------------------------------------------------------
# SAVE BEST MODEL
# -----------------------------------------------------------
joblib.dump(results["Linear Regression"]["model"], "linear_model.pkl")
joblib.dump(results["Random Forest"]["model"], "random_forest_model.pkl")
joblib.dump(results["Gradient Boosting"]["model"], "gradient_boosting_model.pkl")
st.success("💾 Model saved as best_model.pkl")
