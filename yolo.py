import streamlit as st
import numpy as np
import cv2
from ultralytics import YOLO
from PIL import Image
from gtts import gTTS  # Google Text-to-Speech
import tempfile
import os

# ----------------------------
# 🔹 Load YOLOv8 Model
# ----------------------------
MODEL_PATH = "yolov8s.pt"  # Update with your model file
model = YOLO(MODEL_PATH)

# Define class labels
class_labels = {
    0: "You have 10 rupees note",
    1: "You have 20 rupees note",
    2: "You have 50 rupees note",
    3: "You have 100 rupees note",
    4: "You have 200 rupees note",
    5: "You have 500 rupees note"
}

def preprocess_image(img):
    """Convert PIL Image to OpenCV format for YOLO processing."""
    img = np.array(img)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)  # Convert to OpenCV BGR format
    return img

def predict_class(img):
    """Run YOLOv8 inference and return detected class label."""
    processed_img = preprocess_image(img)
    results = model(processed_img)  # Perform detection

    detected_labels = []
    for r in results:
        for box in r.boxes:
            cls = int(box.cls.item())  # Ensure correct class index retrieval
            detected_labels.append(class_labels.get(cls, "Unknown currency"))

    return detected_labels if detected_labels else ["No currency detected"]

def speak(text):
    """Convert text to speech and return audio file path."""
    tts = gTTS(text=text, lang='en')
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
    tts.save(temp_file.name)
    return temp_file.name

# ----------------------------
# 🔹 Streamlit UI
# ----------------------------
st.title("Currency Detection for the Visually Impaired")
st.write("Upload an image of a currency note or capture one using your camera. The app will predict and announce the currency.")

# Play a startup voice command once
if "voice_played" not in st.session_state:
    voice_command_audio = speak("Capture or upload an image of the currency note.")
    st.audio(voice_command_audio, format="audio/mp3", autoplay=True)
    st.session_state.voice_played = True
    os.remove(voice_command_audio)

# Option to upload a file or use the camera
uploaded_file = st.file_uploader("Upload an Image", type=["jpg", "jpeg", "png"])
camera_input = st.camera_input("Take a picture")

# Check if an image is uploaded or captured
if uploaded_file is not None:
    image_data = Image.open(uploaded_file)
    st.image(image_data, caption="Uploaded Image", use_column_width=True)
elif camera_input is not None:
    image_data = Image.open(camera_input)
    st.image(image_data, caption="Captured Image", use_column_width=True)
else:
    image_data = None

if image_data is not None:
    # Predict using YOLOv8
    predicted_labels = predict_class(image_data)
    st.write(f"**Detected Currency:** {', '.join(predicted_labels)}")
    
    # Convert prediction to speech
    audio_path = speak(" ".join(predicted_labels))
    
    # Play audio automatically
    audio_file = open(audio_path, 'rb')
    st.audio(audio_file, format="audio/mp3", autoplay=True)
    
    # Clean up temporary file
    os.remove(audio_path)
