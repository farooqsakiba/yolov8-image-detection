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



# import streamlit as st
# import cv2
# import numpy as np
# from PIL import Image
# from ultralytics import YOLO

# st.set_page_config(page_title="YOLOv8 Image Detection", layout="centered")
# st.title("🐱 YOLOv8 Object Detection App")

# model = YOLO("best.pt")  # Ensure this file exists in the same directory

# option = st.radio("Select input type:", ["Upload Image", "Use Camera"])

# def detect_and_display(image):
#     results = model.predict(source=image, save=False, conf=0.25)
#     res_plotted = results[0].plot()
#     st.image(res_plotted, caption="Detection Result", use_column_width=True)

# if option == "Upload Image":
#     uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
#     if uploaded_file is not None:
#         image = Image.open(uploaded_file).convert("RGB")
#         st.image(image, caption="Uploaded Image", use_column_width=True)
#         if st.button("Detect Objects"):
#             detect_and_display(np.array(image))
# else:
#     st.warning("Camera input works only in local environment.")
#     run_camera = st.button("Start Camera")
#     if run_camera:
#         cam = cv2.VideoCapture(0)
#         stframe = st.empty()
#         while run_camera:
#             ret, frame = cam.read()
#             if not ret:
#                 st.error("Failed to grab frame")
#                 break
#             results = model.predict(source=frame, save=False, conf=0.25)
#             res_plotted = results[0].plot()
#             stframe.image(res_plotted, channels="BGR")
#         cam.release()
