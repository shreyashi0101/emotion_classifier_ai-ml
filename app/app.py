import streamlit as st
import librosa
import numpy as np
import joblib
import os

# --- Load model and label encoder ---
BASE_DIR = os.path.dirname(os.path.dirname(__file__))  # Move one level up from app/
model = joblib.load(os.path.join(BASE_DIR, "emotion_classifier_model.pkl"))
label_encoder = joblib.load(os.path.join(BASE_DIR, "label_encoder.pkl"))

# --- Title and Description ---
st.set_page_config(page_title="Speech Emotion Classifier", layout="centered")
st.title("🎤 Speech Emotion Classifier")
st.markdown("Upload a `.wav` audio file to predict the emotion expressed in it.")

# --- File Uploader ---
uploaded_file = st.file_uploader("Upload a WAV file", type=["wav"])

# --- Feature Extraction Function ---
def extract_features(audio_file):
    try:
        audio, sample_rate = librosa.load(audio_file, res_type='kaiser_fast')
        mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=80)
        mfccs_processed = np.mean(mfccs.T, axis=0)
        return mfccs_processed.reshape(1, -1)
    except Exception as e:
        st.error(f"Error processing audio: {e}")
        return None

# --- Prediction ---
if uploaded_file is not None:
    st.audio(uploaded_file, format="audio/wav")

    features = extract_features(uploaded_file)
    if features is not None:
        prediction = model.predict(features)
        predicted_emotion = label_encoder.inverse_transform(prediction)[0]

        st.success(f"🎯 **Predicted Emotion:** {predicted_emotion.capitalize()}")
