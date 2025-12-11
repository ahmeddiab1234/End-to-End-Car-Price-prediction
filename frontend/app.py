import streamlit as st
import requests
import pandas as pd
import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.helper_fun import load_config, load_model

CONFIG = load_config()
MODEL_NAME = CONFIG['Model']['model_name']

@st.cache_resource
def load_artifacts_local():
    try:
        model, poly, scaler, encoders = load_model(MODEL_NAME, 'test')
        return encoders
    except Exception:
        return None

encoders = load_artifacts_local()

st.set_page_config(page_title="Car Price Predictor", layout="centered")
st.title("Car Price Predictor")

st.markdown("Enter the car features below and press **Predict**.")

with st.form("predict_form"):
    col1, col2 = st.columns(2)

    with col1:
        manufacturer = st.text_input("Manufacturer", value="Toyota")
        model_name = st.text_input("Model", value="Corolla")
        category = st.text_input("Category", value="Sedan")
        leather = st.selectbox("Leather interior", options=(["Yes", "No"] if encoders is None or "Leather interior" not in encoders else list(encoders["Leather interior"])))
        fuel_type = st.selectbox("Fuel type", options=(["Petrol","Diesel","Hybrid","Electric"] if encoders is None or "Fuel type" not in encoders else list(encoders["Fuel type"])))
        gearbox = st.selectbox("Gear box type", options=(["Automatic","Manual"] if encoders is None or "Gear box type" not in encoders else list(encoders["Gear box type"])))
        drive = st.selectbox("Drive wheels", options=(["Front","Rear","All"] if encoders is None or "Drive wheels" not in encoders else list(encoders["Drive wheels"])))
        doors = st.selectbox("Doors", options=(["2","3","4","5"] if encoders is None or "Doors" not in encoders else list(encoders["Doors"])))
    with col2:
        wheel = st.selectbox("Wheel", options=(["Left","Right"] if encoders is None or "Wheel" not in encoders else list(encoders["Wheel"])))
        color = st.text_input("Color", value="White")
        levy = st.number_input("Levy (numeric)", min_value=0, value=0)
        engine_volume = st.number_input("Engine volume (e.g., 1.6)", min_value=0.0, format="%.1f", value=1.6)
        mileage = st.number_input("Mileage (km)", min_value=0, value=50000)
        cylinders = st.number_input("Cylinders", min_value=1, value=4)
        prod_year = st.number_input("Production year", min_value=1900, max_value=2100, value=2018)
        airbags = st.number_input("Airbags", min_value=0, value=2)

    submitted = st.form_submit_button("Predict")

if submitted:
    payload = {
        "Manufacturer": manufacturer,
        "Model": model_name,
        "Category": category,
        "Leather interior": leather,
        "Fuel type": fuel_type,
        "Gear box type": gearbox,
        "Drive wheels": drive,
        "Doors": str(doors),
        "Wheel": wheel,
        "Color": color,
        "Price": 0,
        "Levy": levy,
        "Engine volume": engine_volume,
        "Mileage": int(mileage),
        "Cylinders": int(cylinders),
        "Prod. year": int(prod_year),
        "Airbags": int(airbags),
        "Random_notes": ""
    }

    # call local backend (adjust if backend runs on different host/port)
    try:
        api_url = st.secrets.get("API_URL", "http://localhost:8000/predict")
        resp = requests.post(api_url, json=payload, timeout=10)
        if resp.status_code == 200:
            data = resp.json()
            st.success(f"Predicted price: {data['predicted_price']:.2f} USD")
            st.write("Raw model output:", data.get("raw_prediction_model_output"))
        else:
            st.error(f"Prediction failed: {resp.status_code} {resp.text}")
    except Exception as e:
        st.error(f"Error connecting to API: {e}")
