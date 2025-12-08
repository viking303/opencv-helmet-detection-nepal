import streamlit as st
import cv2
from ultralytics import YOLO
import tempfile
import os

# For browser webcam
try:
    from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
except ImportError:
    webrtc_streamer = None

st.title("Helmet Detection Demo 🚀")

# Load YOLO model
model = YOLO("notebooks/best.pt")

# --- Upload option ---
uploaded_file = st.file_uploader("Upload an image or video", type=["jpg","jpeg","png","mp4"])

if uploaded_file:
    temp_file = tempfile.NamedTemporaryFile(delete=False)
    temp_file.write(uploaded_file.read())
    temp_path = temp_file.name

    if uploaded_file.type.startswith("image"):
        img = cv2.imread(temp_path)
        if img is None:
            st.error("❌ Could not read image.")
        else:
            results = model(img, conf=0.25)
            annotated = results[0].plot()
            annotated = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
            st.image(annotated, caption="Prediction")

    elif uploaded_file.type.startswith("video"):
        cap = cv2.VideoCapture(temp_path)
        stframe = st.empty()
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            results = model(frame, conf=0.25)
            annotated = results[0].plot()
            annotated = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
            stframe.image(annotated)
        cap.release()

# --- Camera option ---
st.subheader("Camera Input")

if os.environ.get("STREAMLIT_RUNTIME") == "cloud" and webrtc_streamer:
    # Use browser webcam on Streamlit Cloud
    class HelmetTransformer(VideoTransformerBase):
        def transform(self, frame):
            img = frame.to_ndarray(format="bgr24")
            results = model(img, conf=0.25)
            annotated = results[0].plot()
            return cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)

    webrtc_streamer(key="helmet-detection", video_transformer_factory=HelmetTransformer)

else:
    # Local OpenCV camera
    if st.button("Start Local Camera"):
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        stframe = st.empty()

        if not cap.isOpened():
            st.error("❌ Cannot open local camera")
        else:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                results = model(frame, conf=0.25)
                annotated = results[0].plot()
                annotated = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
                stframe.image(annotated)
            cap.release()
