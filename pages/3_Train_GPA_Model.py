import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error

st.title("🤖 Train GPA Prediction Model")

if "df" not in st.session_state:
    st.warning("Upload your dataset first on Dataset Exploration page.")
else:
    df = st.session_state["df"]
    # Take only numeric features for simple demo
    X = df.select_dtypes(include=['int64','float64']).drop(columns=['GPA'], errors='ignore')
    y = df['GPA'] if 'GPA' in df.columns else None

    if y is None:
        st.error("No 'GPA' column found in the dataset.")
    else:
        st.write("Features used:", list(X.columns))
        test_size = st.slider("Test set size (%)", 10, 40, 20)
        n_estimators = st.slider("n_estimators (trees)", 50, 500, 200, step=50)

        if st.button("Train Model"):
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size/100, random_state=42)
            model = RandomForestRegressor(n_estimators=n_estimators, random_state=42)
            model.fit(X_train, y_train)
            preds = model.predict(X_test)
            st.session_state["model"] = model
            st.session_state["model_features"] = X.columns.tolist()
            st.success("Model trained successfully!")
            st.write("R²:", r2_score(y_test, preds))
            st.write("MAE:", mean_absolute_error(y_test, preds))
