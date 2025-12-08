import streamlit as st
import cv2
from ultralytics import YOLO
import tempfile

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
if "run_camera" not in st.session_state:
    st.session_state.run_camera = False

col1, col2 = st.columns(2)
if col1.button("Start Camera"):
    st.session_state.run_camera = True
if col2.button("Stop Camera"):
    st.session_state.run_camera = False

if st.session_state.run_camera:
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    stframe = st.empty()

    if not cap.isOpened():
        st.error("❌ Cannot open camera")
    else:
        while st.session_state.run_camera:
            ret, frame = cap.read()
            if not ret:
                break

            results = model(frame, conf=0.25)
            annotated = results[0].plot()
            annotated = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)

            stframe.image(annotated)


        cap.release()
