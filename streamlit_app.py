import streamlit as st
import gdown
import tensorflow as tf
from tensorflow.keras.models import load_model
import os

# Function to download the model correctly
def download_model():
    model_url = "https://drive.google.com/file/d/1TcEQ092-zqJ9YiPMdtiICZVDdACvtGlF/view?usp=sharing"  # Replace with your actual file ID
    model_path = "model_diseases.h5"

    if not os.path.exists(model_path):
        st.info("Downloading model... Please wait ⏳")
        
        try:
            gdown.download(model_url, model_path, quiet=False)
            st.success("Model downloaded successfully ✅")
        except Exception as e:
            st.error(f"Failed to download model: {e}")
            return None

    return model_path

# Load the model
model_path = download_model()

if model_path and os.path.exists(model_path):
    try:
        model = load_model(model_path)
        st.success("Model loaded successfully ✅")
    except OSError:
        st.error("Downloaded file is not a valid model. Please check the Google Drive link.")
else:
    st.error("Model file not found. Please check the download process.")

st.title("🌾 Crop Disease Recognition App")
st.write("Upload an image of a plant leaf to detect the disease.")

# File uploader
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)
    st.write("Processing...")

    # Add your prediction logic here
    st.success("Prediction: Healthy Plant ✅")  # Replace with actual prediction logic
