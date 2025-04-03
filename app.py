import joblib  
import streamlit as st
import pandas as pd

# Load the trained model
@st.cache_resource  # Caches the model so it's not reloaded every time
def load_model():
    return joblib.load("motor_health_model.pkl")

clf = load_model()  # Load the model

# Define the function for predictions
def predict_motor_status(temp, humidity, vib, speed, air_temp):
    test_input = pd.DataFrame([[temp, humidity, vib, speed, air_temp]],
                              columns=["temperature", "humidity", "vibration", "rotational_speed", "air_temperature"])
    prediction = clf.predict(test_input)
    return "Motor is GOOD" if prediction == 1 else "Motor is BAD"

# Streamlit UI
st.title("Motor Health Prediction")

temp = st.number_input("Temperature")
humidity = st.number_input("Humidity")
vib = st.number_input("Vibration")
speed = st.number_input("Rotational Speed")
air_temp = st.number_input("Air Temperature")

if st.button("Predict"):
    result = predict_motor_status(temp, humidity, vib, speed, air_temp)
    st.success(result)
