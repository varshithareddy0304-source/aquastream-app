import streamlit as st
import numpy as np
import pandas as pd
import joblib

# -------------------------------
# PAGE CONFIG
# -------------------------------
st.set_page_config(page_title="AquaStream", layout="centered")

st.title(" AquaStream - Coastal Navigation AI")
st.markdown("Predict safe navigation conditions based on wave and wind parameters")

# -------------------------------
# LOAD MODEL
# -------------------------------
model = joblib.load("xgboost_wave_predictor.pkl")

# -------------------------------
# SIDEBAR INPUT
# -------------------------------
st.sidebar.header("Enter Conditions")

wave_lag_1 = st.sidebar.number_input("Previous Wave Height (m)", value=1.0)
wave_lag_2 = st.sidebar.number_input("Wave Lag 2", value=0.9)
wave_lag_3 = st.sidebar.number_input("Wave Lag 3", value=0.8)
wave_lag_6 = st.sidebar.number_input("Wave Lag 6", value=0.7)
wave_lag_12 = st.sidebar.number_input("Wave Lag 12", value=0.6)

wave_mean_3 = st.sidebar.number_input("Wave Mean (3)", value=0.9)
wave_mean_6 = st.sidebar.number_input("Wave Mean (6)", value=0.85)
wave_std_3 = st.sidebar.number_input("Wave Std (3)", value=0.1)

wave_period = st.sidebar.number_input("Wave Period (s)", value=5.5)
wind_speed = st.sidebar.number_input("Wind Speed (m/s)", value=3.0)
gust = st.sidebar.number_input("Wind Gust", value=4.0)

hour = st.sidebar.slider("Hour", 0, 23, 12)
month = st.sidebar.slider("Month", 1, 12, 6)

# -------------------------------
# FEATURE ENGINEERING
# -------------------------------
wave_energy = wave_lag_1**2 * wave_period
wind_wave_interaction = wind_speed * wave_lag_1
gust_factor = gust / (wind_speed + 1e-5)

# -------------------------------
# MODEL INPUT
# -------------------------------
features = np.array([[
    wave_lag_1, wave_lag_2, wave_lag_3, wave_lag_6, wave_lag_12,
    wave_mean_3, wave_mean_6, wave_std_3,
    wave_period, wind_speed, gust,
    wave_energy, wind_wave_interaction, gust_factor,
    hour, month
]])

# -------------------------------
# PREDICTION
# -------------------------------
prediction = float(model.predict(features)[0])

# -------------------------------
# CLASSIFICATION
# -------------------------------
def classify_wave(h):
    if h < 2:
        return "SAFE", "green"
    elif h < 3:
        return "CAUTION", "orange"
    else:
        return "UNSAFE", "red"

status, color = classify_wave(prediction)

# -------------------------------
# DISPLAY OUTPUT
# -------------------------------
st.subheader(" Prediction Results")

st.metric("Predicted Wave Height (m)", round(prediction, 3))

st.markdown(f"### Status: :{color}[{status}]")

# -------------------------------
# VISUAL FEEDBACK
# -------------------------------
if status == "SAFE":
    st.success("✅ Safe for navigation")
elif status == "CAUTION":
    st.warning("⚠️ Moderate conditions — proceed carefully")
else:
    st.error("❌ Unsafe — navigation not recommended")