import streamlit as st
import numpy as np
import joblib

st.set_page_config(page_title="AquaStream", layout="centered")

st.title(" AquaStream - Coastal Navigation AI")

# Load model
model = joblib.load("xgboost_wave_predictor.pkl")

st.sidebar.header("Enter Conditions")

# Inputs (ONLY correct ones)
wave_lag_3 = st.sidebar.number_input("Wave Lag 3", value=0.8)
wave_lag_6 = st.sidebar.number_input("Wave Lag 6", value=0.7)
wave_lag_12 = st.sidebar.number_input("Wave Lag 12", value=0.6)

wave_mean_6 = st.sidebar.number_input("Wave Mean (6)", value=0.85)
wave_std_3 = st.sidebar.number_input("Wave Std (3)", value=0.1)

wave_period = st.sidebar.number_input("Wave Period", value=5.5)
wind_speed = st.sidebar.number_input("Wind Speed", value=3.0)
gust = st.sidebar.number_input("Wind Gust", value=4.0)

hour = st.sidebar.slider("Hour", 0, 23, 12)
month = st.sidebar.slider("Month", 1, 12, 6)

# Feature engineering (MATCH TRAINING)
wave_energy = wave_lag_3**2 * wave_period
log_wave_energy = np.log1p(wave_energy)
wind_wave_interaction = wind_speed * wave_lag_3
gust_factor = gust / (wind_speed + 1e-5)
wave_energy_std_3 = 0.1  # constant approximation

# FINAL INPUT
features = np.array([[
    wave_lag_3, wave_lag_6, wave_lag_12,
    wave_mean_6, wave_std_3,
    wave_period, wind_speed, gust,
    wave_energy, wind_wave_interaction, gust_factor,
    hour, month, log_wave_energy,
    wave_energy_std_3
]])

prediction = float(model.predict(features)[0])

# Classification
if prediction < 2:
    status = "SAFE"
    st.success(f"SAFE ✅ | Wave Height: {prediction:.2f} m")
elif prediction < 3:
    status = "CAUTION"
    st.warning(f"CAUTION ⚠️ | Wave Height: {prediction:.2f} m")
else:
    status = "UNSAFE"
    st.error(f"UNSAFE ❌ | Wave Height: {prediction:.2f} m")
