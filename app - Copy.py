import streamlit as st
import pickle
import pandas as pd

# Load model & encoders
model = pickle.load(open("fertilizer_model.pkl", "rb"))
le_soil = pickle.load(open("soil_encoder.pkl", "rb"))
le_crop = pickle.load(open("crop_encoder.pkl", "rb"))
le_fert = pickle.load(open("fert_encoder.pkl", "rb"))

st.title("🌱 Fertilizer Recommendation System")

# Inputs
temperature = st.number_input("🌡️ Temperature (°C)", 10, 50)
humidity = st.number_input("💧 Humidity (%)", 10, 100)
moisture = st.number_input("🌍 Soil Moisture (%)", 0, 100)
soil = st.selectbox("🪨 Soil Type", le_soil.classes_)
crop = st.selectbox("🌾 Crop Type", le_crop.classes_)
N = st.number_input("Nitrogen (N)", 0, 150)
P = st.number_input("Phosphorous (P)", 0, 150)
K = st.number_input("Potassium (K)", 0, 150)

if st.button("🔍 Predict Fertilizer"):
    soil_enc = le_soil.transform([soil])[0]
    crop_enc = le_crop.transform([crop])[0]

    input_data = pd.DataFrame(
    [[temperature, humidity, moisture, soil_enc, crop_enc, N, P, K]],
    columns=["Temperature","Humidity","Moisture","Soil Type","Crop Type","Nitrogen","Potassium","Phosphorous"]

)


    prediction = model.predict(input_data)[0]
    fertilizer = le_fert.inverse_transform([prediction])[0]
    st.success(f"✅ Recommended Fertilizer: {fertilizer}")
