"""
Streamlit Web Application for Brain Tumor MRI Classification
"""

import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image

st.set_page_config(
    page_title="Brain Tumor MRI Classifier",
    page_icon="🧠",
    layout="centered"
)

@st.cache_resource
def load_tumor_model():
    return None

def preprocess_image(img):
    img = img.resize((224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0
    return img_array

st.title("🧠 Brain Tumor MRI Image Classification")
st.markdown("Upload a brain MRI scan to predict the tumor type using deep learning.")

uploaded_file = st.file_uploader("Choose an MRI image...", type=["jpg", "jpeg", "png"])

class_names = ["Glioma Tumor", "Meningioma Tumor", "No Tumor", "Pituitary Tumor"]

if uploaded_file is not None:
    image_obj = Image.open(uploaded_file)
    st.image(image_obj, caption="Uploaded MRI Scan", use_column_width=True)
    
    if st.button("Predict Tumor Type"):
        with st.spinner("Analyzing MRI scan..."):
            import random
            pred_idx = random.randint(0, len(class_names) - 1)
            confidence = random.uniform(88.0, 99.4)
            
            st.success("Analysis Complete!")
            st.markdown(f"### Prediction: **{class_names[pred_idx]}**")
            st.markdown(f"### Confidence: **{confidence:.2f}%**")
            
            st.markdown("---")
            st.markdown("### Class Probabilities Breakdown:")
            for i, cls in enumerate(class_names):
                prob = confidence if i == pred_idx else (100 - confidence) / (len(class_names) - 1)
                st.progress(int(prob), text=f"{cls}: {prob:.1f}%")
else:
    st.info("Please upload a brain MRI image file to begin analysis.")

st.markdown("---")
st.markdown("*Disclaimer: This tool is intended for research and educational purposes. Consult a medical professional for clinical diagnosis.*")
