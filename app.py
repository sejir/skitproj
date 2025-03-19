import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="Tip Prediction App", layout="wide", initial_sidebar_state="expanded")
st.title("💸 Machine Learning Tip Prediction App")
st.markdown("This app predicts **tip amounts** and **tip categories** based on restaurant data!")
st.markdown("---")

# Split the app into two columns
col1, col2 = st.columns(2)

# Styling for a cleaner look
with open("style.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# Column 1: Regression Model (Numerical Prediction)
with col1:
    st.subheader("🔢 Regression Model: Numerical Prediction")
    st.write("Predicts the exact tip amount 💵")
    model_reg = joblib.load('best_model.pkl')

    total_bill = st.number_input("💵 Total Bill Amount:", min_value=0.0)
    sex = st.selectbox("👤 Gender:", ("Male", "Female"))
    sex = 0 if sex == "Male" else 1
    smoker = st.selectbox("🚬 Smoker:", ("No", "Yes"))
    smoker = 0 if smoker == "No" else 1
    day = st.selectbox("📅 Day:", ("Thur", "Fri", "Sat", "Sun"))
    day = ["Thur", "Fri", "Sat", "Sun"].index(day)
    time = st.selectbox("⏰ Time:", ("Lunch", "Dinner"))
    time = 0 if time == "Lunch" else 1
    size = st.number_input("👥 Party Size:", min_value=1)

    if st.button("💡 Predict Tip Amount (Regression)"):
        features = pd.DataFrame([[total_bill, sex, smoker, day, time, size]],
                               columns=["total_bill", "sex", "smoker", "day", "time", "size"])
        prediction = model_reg.predict(features)
        st.success(f"### 💵 Predicted Tip Amount: ${prediction[0]:.2f}")


# Column 2: Classification Model (Categorical Prediction)
with col2:
    st.subheader("🔠 Classification Model: Categorical Prediction")
    st.write("Categorizes the tip into Low, Medium, or High 💼")
    model_class = joblib.load('best_classification_model.pkl')

    total_bill = st.number_input("💵 Total Bill Amount:", min_value=0.0, key='cat_tb')
    sex = st.selectbox("👤 Gender:", ("Male", "Female"), key='cat_sex')
    sex = 0 if sex == "Male" else 1
    smoker = st.selectbox("🚬 Smoker:", ("No", "Yes"), key='cat_smoker')
    smoker = 0 if smoker == "No" else 1
    day = st.selectbox("📅 Day:", ("Thur", "Fri", "Sat", "Sun"), key='cat_day')
    day = ["Thur", "Fri", "Sat", "Sun"].index(day)
    time = st.selectbox("⏰ Time:", ("Lunch", "Dinner"), key='cat_time')
    time = 0 if time == "Lunch" else 1
    size = st.number_input("👥 Party Size:", min_value=1, key='cat_size')

    if st.button("💡 Predict Tip Category (Classification)"):
        features = pd.DataFrame([[total_bill, sex, smoker, day, time, size]],
                               columns=["total_bill", "sex", "smoker", "day", "time", "size"])
        prediction = model_class.predict(features)
        category = 'Low Tip' if prediction[0] == 0 else ('Medium Tip' if prediction[0] == 1 else 'High Tip')
        st.success(f"### 🏷️ Predicted Tip Category: {category}")
