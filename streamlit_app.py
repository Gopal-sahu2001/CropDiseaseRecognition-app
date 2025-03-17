import gdown
import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
import os

import streamlit as st
import gdown
import tensorflow as tf
from tensorflow.keras.models import load_model
import os

# Function to download the model from Google Drive
def download_model():
    model_url = "https://drive.google.com/file/d/1TcEQ092-zqJ9YiPMdtiICZVDdACvtGlF/view?usp=sharing"  # Replace with your actual file ID
    model_path = "model_diseases.h5"

    if not os.path.exists(model_path):
        st.info("Downloading model... Please wait ⏳")
        gdown.download(model_url, model_path, quiet=False)
        st.success("Model downloaded successfully ✅")

    return model_path

# Load the model
model_path = download_model()
model = load_model(model_path)

st.title("🌾 Crop Disease Recognition App")
st.write("Upload an image of a plant leaf to detect the disease.")

# File uploader
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)
    st.write("Processing...")

    # Add your prediction logic here
    st.success("Prediction: Healthy Plant ✅")  # Replace with actual prediction logic
