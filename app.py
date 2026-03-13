import streamlit as st
import numpy as np
import cv2
from PIL import Image
from ultralytics import YOLO
import tempfile
import time
import os

st.set_page_config(layout="wide")

MODEL_PT = "./models/yolo11m.pt"
MODEL_ONNX = "./models/yolo11m.onnx"

# ==============================
# PREPARE MODEL (DOWNLOAD + EXPORT)
# ==============================


def prepare_model():

    os.makedirs("models", exist_ok=True)

    if not os.path.exists(MODEL_ONNX):

        st.info("ONNX model not found. Downloading and exporting...")

        model = YOLO(MODEL_PT)

        model.export(format="onnx", imgsz=640, simplify=True)

        st.success("Export ONNX completed")


# ==============================
# LOAD MODEL
# ==============================


@st.cache_resource
def load_model():

    prepare_model()

    model = YOLO(MODEL_ONNX)

    return model


model = load_model()

st.title("YOLO11m ONNX Inference")

uploaded_file = st.file_uploader(
    "Upload Image or Video", type=["jpg", "jpeg", "png", "mp4", "avi", "mov"]
)

col1, col2 = st.columns(2)

# ==============================
# IMAGE INFERENCE
# ==============================


def process_image(image):

    start = time.time()

    results = model(image)

    end = time.time()

    result_img = results[0].plot()

    total_time = end - start

    return result_img, total_time


# ==============================
# VIDEO INFERENCE
# ==============================


def process_video(video_path):

    cap = cv2.VideoCapture(video_path)

    frames = []
    start = time.time()

    while True:

        ret, frame = cap.read()

        if not ret:
            break

        results = model(frame)

        frame = results[0].plot()

        frames.append(frame)

    cap.release()

    end = time.time()

    total_time = end - start

    return frames, total_time


# ==============================
# MAIN LOGIC
# ==============================

if uploaded_file is not None:

    file_type = uploaded_file.type

    # IMAGE
    if "image" in file_type:

        image = Image.open(uploaded_file)
        image = np.array(image)

        with col1:
            st.subheader("Input")
            st.image(image)

        result_img, total_time = process_image(image)

        with col2:
            st.subheader("Output")
            st.image(result_img)
            st.write(f"Total inference time: {total_time:.3f} sec")

    # VIDEO
    elif "video" in file_type:

        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_file.read())

        with col1:
            st.subheader("Input Video")
            st.video(uploaded_file)

        frames, total_time = process_video(tfile.name)

        with col2:
            st.subheader("Output Video")

            for frame in frames:
                st.image(frame, channels="BGR")

            st.write(f"Total inference time: {total_time:.3f} sec")
