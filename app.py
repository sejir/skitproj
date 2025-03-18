import streamlit as st
import pandas as pd
import joblib

st.title("Machine Learning Regression Model - Deployment App")
st.write("This app uses the best regression model to predict tips based on restaurant data.")

# Load the best model
model = joblib.load('best_model.pkl')

# User inputs
total_bill = st.number_input("Enter Total Bill Amount:", min_value=0.0)
sex = st.selectbox("Select Gender:", ("Male", "Female"))
sex = 0 if sex == "Male" else 1
smoker = st.selectbox("Smoker:", ("No", "Yes"))
smoker = 0 if smoker == "No" else 1
day = st.selectbox("Day:", ("Thur", "Fri", "Sat", "Sun"))
day = ["Thur", "Fri", "Sat", "Sun"].index(day)
time = st.selectbox("Time:", ("Lunch", "Dinner"))
time = 0 if time == "Lunch" else 1
size = st.number_input("Enter Party Size:", min_value=1)

if st.button("Predict Tip"):
    features = pd.DataFrame([[total_bill, sex, smoker, day, time, size]],
                           columns=["total_bill", "sex", "smoker", "day", "time", "size"])
    prediction = model.predict(features)
    st.write(f"### Predicted Tip Amount: ${prediction[0]:.2f}")
