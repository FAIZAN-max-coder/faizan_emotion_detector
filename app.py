import cv2
import numpy as np
import streamlit as st
from ultralytics import YOLO

# 1. Page Configuration Layout
st.set_page_config(page_title="Facial & Emotion Detection", page_icon="😀")
st.title("My Facial & Emotion Detection App")
st.write("Upload any photo to instantly detect faces and analyze their expressions!")

# 2. Load the downloaded models safely
@st.cache_resource
def load_models():
    face = YOLO("./yolov8_face/weights/best.pt")
    emotion = YOLO("./yolov8_emotion/weights/best.pt")
    return face, emotion

face_model, emotion_model = load_models()

# 3. Image Upload Interface component
uploaded_file = st.file_uploader("Choose a picture...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Convert uploaded raw file into an OpenCV image format
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    frame = cv2.imdecode(file_bytes, 1)
    h, w = frame.shape[:2]

    # Process and detect faces
    face_results = face_model(frame, conf=0.5, verbose=False)

    for result in face_results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy.tolist())
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)

            face_crop = frame[y1:y2, x1:x2]
            if face_crop.size == 0:
                continue

            # Classify emotions on the cropped face area
            emotion_results = emotion_model(face_crop, conf=0.4, verbose=False)

            if len(emotion_results) > 0 and len(emotion_results.boxes) > 0:
                cls = int(emotion_results.boxes.cls)
                emotion_name = emotion_results.names[cls]
                conf = float(emotion_results.boxes.conf)
                color = (0, 255, 0) # Green box
            else:
                emotion_name = "Neutral"
                conf = 0.0
                color = (0, 255, 255) # Yellow box

            # Draw the box boundaries and label onto the picture
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            label = f"{emotion_name} {conf:.2f}"
            cv2.putText(frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    # Output the final processed visual result to the user
    st.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), caption="Processed Result", use_column_width=True)
