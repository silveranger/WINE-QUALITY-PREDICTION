import streamlit as st
import joblib
import numpy as np
import pandas as pd
import os
import base64

# --- Function to Add Background Image ---
def add_bg_image(image_path):
    with open(image_path, "rb") as image_file:
        encoded = base64.b64encode(image_file.read()).decode()
    css = f"""
    <style>
    .stApp {{
        background-image: url("data:image/png;base64,{encoded}");
        background-size: cover;
        background-repeat: no-repeat;
        background-attachment: fixed;
    }}
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)

# --- Set Page Config ---
st.set_page_config(page_title="Wine Quality Predictor 🍷", layout="centered")

# --- Add Background Image ---
bg_image_path = r"C:\Users\ACER\Pictures\Nitro\Nitro_Wallpaper_5000x2813.jpg"  # Change path as needed
if os.path.exists(bg_image_path):
    add_bg_image(bg_image_path)
else:
    st.warning("⚠️ Background image not found. Please check the path.")

# --- Page Title ---
st.title("🍷 Wine Quality Predictor (Random Forest)")

# --- Load Trained Model ---
MODEL_FILE = "model23.sav"
model = None

if os.path.exists(MODEL_FILE):
    model = joblib.load(MODEL_FILE)
else:
    st.error("Model file 'model23.sav' not found in the directory.")

# --- Feature Names (Must match training data) ---
feature_names = [
    "fixed acidity", "volatile acidity", "citric acid", "residual sugar",
    "chlorides", "free sulfur dioxide", "total sulfur dioxide", "density",
    "pH", "sulphates", "alcohol"
]

# --- User Inputs ---
st.subheader("🔧 Enter Wine Characteristics")
user_input = []
for feature in feature_names:
    val = st.number_input(f"{feature}", value=0.0, format="%.3f")
    user_input.append(val)

# --- Input DataFrame ---
input_df = pd.DataFrame([user_input], columns=feature_names)

st.subheader("📋 Review Input Data")
st.dataframe(input_df.style.format("{:.3f}"))

# --- Predict Button ---
if st.button("🔮 Predict Quality"):
    if model is not None:
        prediction = model.predict(input_df)[0]  # Use DataFrame to avoid warning

        if prediction == 1:
            st.success("✅ Good Quality Wine 🍷")
        else:
            st.warning("⚠️ Bad Quality Wine 🍷")
    else:
        st.error("Model not loaded.")
