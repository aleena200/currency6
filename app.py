import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from PIL import Image
from gtts import gTTS  # Google Text-to-Speech
import tempfile
import os

# Load the trained model
MODEL_PATH = "vgg16_model.h5"
model = tf.keras.models.load_model(MODEL_PATH)

# Define class labels
class_labels = {
    '10_new': "You have 10 rupees note",
    '20_new': "You have 20 rupees note",
    '50_new': "You have 50 rupees note",
    '100_new': "You have 100 rupees note",
    '200_new': "You have 200 rupees note",
    '500_new': "You have 500 rupees note"
}

def preprocess_image(img):
    img = img.resize((224, 224))  # Resize
    img_array = np.array(img) / 255.0  # Normalize
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    return img_array

def predict_class(img):
    processed_img = preprocess_image(img)
    predictions = model.predict(processed_img)
    class_id = np.argmax(predictions)
    predicted_label = list(class_labels.keys())[class_id]
    return class_labels[predicted_label]

def speak(text):
    """Convert text to speech and return audio file path."""
    tts = gTTS(text=text, lang='en')  # Generate speech
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")  # Create temp file
    tts.save(temp_file.name)  # Save speech
    return temp_file.name  # Return path to speech file

# Streamlit UI
st.title("Currency Note Classification for the Visually Impaired")
st.write("Upload an image of a currency note, or take a picture with your camera. The app will predict its class and announce it.")

# Play a startup voice command once
if "voice_played" not in st.session_state:
    voice_command_audio = speak("Capture the image of currency")
    audio_file = open(voice_command_audio, 'rb')
    st.audio(audio_file, format="audio/mp3", autoplay=True)
    st.session_state.voice_played = True
    # Remove the temporary file after playing
    os.remove(voice_command_audio)

# Option to upload a file or use the camera
uploaded_file = st.file_uploader("Upload an Image", type=["jpg", "jpeg", "png"])
camera_input = st.camera_input("Take a picture")

# Check if an image is uploaded or captured from the camera
if uploaded_file is not None:
    image_data = Image.open(uploaded_file)
    st.image(image_data, caption="Uploaded Image", use_column_width=True)
elif camera_input is not None:
    image_data = Image.open(camera_input)
    st.image(image_data, caption="Captured Image", use_column_width=True)
else:
    image_data = None

if image_data is not None:
    predicted_class = predict_class(image_data)
    st.write(f"**Predicted Class:** {predicted_class}")
    
    # Generate speech for the predicted class and get path
    audio_path = speak(predicted_class)
    
    # Play audio automatically after classification
    audio_file = open(audio_path, 'rb')
    st.audio(audio_file, format="audio/mp3", autoplay=True)
    
    # Clean up temporary file after use
    os.remove(audio_path)
