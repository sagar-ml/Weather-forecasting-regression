
import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

MODEL_PATH = Path('data/cleaned/model.pkl')

st.set_page_config(page_title='Weather Forecast', page_icon='🌦️')
st.title('🌦️ Weather Forecast (Regression)')

if not MODEL_PATH.exists():
    st.warning('Model not found. Please run training first: `python -m src.model`')
else:
    model = joblib.load(MODEL_PATH)
    st.success('Model loaded! Enter features to predict next-day temperature.')

    with st.form('predict'):
        humidity = st.number_input('Humidity (%)', 0.0, 100.0, 55.0)
        pressure = st.number_input('Pressure (hPa)', 900.0, 1100.0, 1013.0)
        windspeed = st.number_input('Windspeed (km/h)', 0.0, 200.0, 10.0)
        temp_lag_1 = st.number_input('Yesterday Temp (°C)', -20.0, 55.0, 28.0)
        temp_lag_2 = st.number_input('Temp 2 days ago (°C)', -20.0, 55.0, 27.0)
        temp_lag_3 = st.number_input('Temp 3 days ago (°C)', -20.0, 55.0, 26.0)
        temp_rollmean_3 = st.number_input('3-day rolling mean (°C)', -20.0, 55.0, 27.0)
        temp_rollmean_7 = st.number_input('7-day rolling mean (°C)', -20.0, 55.0, 26.0)
        month = st.number_input('Month', 1, 12, 1)
        day = st.number_input('Day', 1, 31, 15)
        dayofweek = st.number_input('Day of week (Mon=0)', 0, 6, 1)
        submitted = st.form_submit_button('Predict')

    if submitted:
        X = [[humidity, pressure, windspeed, temp_lag_1, temp_lag_2, temp_lag_3,
              temp_rollmean_3, temp_rollmean_7, month, day, dayofweek]]
        pred = model.predict(X)[0]
        st.success(f'Predicted temperature: **{pred:.2f} °C**')
