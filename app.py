import streamlit as st
from PIL import Image
import torch
import os
from pathlib import Path
from ultralytics import YOLO

st.set_page_config(page_title="YOLOv8 Image Detector", layout="centered")

st.title("📷 YOLOv8 Image Object Detection")

# Load model (expects 'best.pt' in current directory)
@st.cache_resource
def load_model():
    model = YOLO("best.pt")  # assumes best.pt is in the same folder
    return model

model = load_model()

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
if uploaded_file:
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="Uploaded Image", use_column_width=True)
    with st.spinner("Detecting..."):
        results = model.predict(img)
        result_img = results[0].plot()
        st.image(result_img, caption="Detected Image", use_column_width=True)