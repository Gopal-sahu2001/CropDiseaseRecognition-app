import streamlit as st
import gdown
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os
from PIL import Image

# Function to extract Google Drive file ID
def extract_file_id(drive_link):
    if "id=" in drive_link:
        return drive_link.split("id=")[-1]
    elif "/d/" in drive_link:
        return drive_link.split("/d/")[-1].split("/")[0]
    else:
        st.error("Invalid Google Drive link! 🚨")
        return None

# Function to download the model correctly
def download_model():
    model_url = "https://drive.google.com/file/d/1TcEQ092-zqJ9YiPMdtiICZVDdACvtGlF/view?usp=sharing"  
    model_path = "model_diseases.h5"

    if not os.path.exists(model_path):
        st.info("Downloading model... Please wait ⏳")
        file_id = extract_file_id(model_url)
        if not file_id:
            return None
        
        try:
            gdown.download(f"https://drive.google.com/uc?id={file_id}", model_path, quiet=False)
            st.success("✅ Model downloaded successfully!")
        except Exception as e:
            st.error(f"Failed to download model: {e}")
            return None

    return model_path

# Download and load the model
model_path = download_model()
if model_path:
    try:
        model = load_model(model_path)
        st.success("🔄 Model loaded successfully!")
    except Exception as e:
        st.error(f"Failed to load model: {e}")

# Define class labels
class_labels = [
    "American Bollworm on Cotton", "Anthracnose on Cotton", "Army worm",
    "Bacterial Blight in cotton", "Becterial Blight in Rice", "Brownspot",
    "Common_Rust", "Cotton Aphid", "Flag Smut", "Gray_Leaf_Spot",
    "Healthy Maize", "Healthy Wheat", "Healthy cotton", "Leaf Curl",
    "Leaf smut", "Mosaic sugarcane", "RedRot sugarcane", "Wheat Black Rust",
    "Wilt", "Yellow Rust Sugarcane", "maize ear rot", "maize fall armyworm",
    "pink bollworm in cotton", "red cotton bug"
]

# Crop Disease Solutions
CropDiseaseSolution = {
    "American Bollworm on Cotton": {
        "Cause": "Larvae of Helicoverpa armigera feeding on cotton bolls.",
        "Peak Season": "Summer and early monsoon.",
        "Remedy": "Use pheromone traps and insecticides like Spinosad or Bacillus thuringiensis."
    },
    "Anthracnose on Cotton": {
        "Cause": "Fungal infection caused by Colletotrichum species.",
        "Peak Season": "High humidity periods, usually post-monsoon.",
        "Remedy": "Apply copper-based fungicides and ensure good field drainage."
    },
    "Army worm": {
        "Cause": "Larvae of Spodoptera species attacking foliage.",
        "Peak Season": "Rainy season and post-monsoon.",
        "Remedy": "Use neem oil or Bacillus thuringiensis-based biopesticides."
    },
    "Healthy Maize": {
        "Cause": "No disease detected.",
        "Peak Season": "N/A",
        "Remedy": "Crop is healthy, no diagnosis required."
    },
    "Healthy Wheat": {
        "Cause": "No disease detected.",
        "Peak Season": "N/A",
        "Remedy": "Crop is healthy, no diagnosis required."
    },
    "Healthy cotton": {
        "Cause": "No disease detected.",
        "Peak Season": "N/A",
        "Remedy": "Crop is healthy, no diagnosis required."
    }
}

# Preprocess image for model prediction
def preprocess_image(img):
    img = img.resize((224, 224))  # Resize image
    img_array = image.img_to_array(img) / 255.0  # Normalize pixel values
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    return img_array

# Predict the crop disease
def predict_image(img):
    img_array = preprocess_image(img)
    prediction = model.predict(img_array)
    predicted_class_index = np.argmax(prediction, axis=1)[0]
    confidence = np.max(prediction) * 100  # Convert confidence to percentage
    predicted_class = class_labels[predicted_class_index]
    
    return predicted_class, confidence

# Streamlit UI
st.title("🌾 Crop Disease Detection AI")
st.write("Upload an image of a crop to detect diseases and get solutions.")

# File uploader
uploaded_file = st.file_uploader("Upload an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Load image
    img = Image.open(uploaded_file)
    
    # Predict
    predicted_disease, confidence = predict_image(img)

    # Display uploaded image
    st.image(img, caption="Uploaded Image", use_column_width=True)

    # Show prediction result
    st.success(f"✅ **Predicted Disease:** {predicted_disease}")
    st.info(f"🎯 **Confidence:** {confidence:.2f}%")

    # Show disease solution if available
    solution = CropDiseaseSolution.get(predicted_disease, None)
    if solution:
        st.write("### 🩺 Disease Details & Solution:")
        st.write(f"📌 **Cause:** {solution['Cause']}")
        st.write(f"🌱 **Peak Season:** {solution['Peak Season']}")
        st.write(f"💊 **Remedy:** {solution['Remedy']}")
    else:
        st.warning("⚠️ No specific solution available.")
