import streamlit as st
import numpy as np
import librosa
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
import pickle

# Load the pre-trained model
model = tf.keras.models.load_model(r'C:\Users\11 PrO\Desktop\fds\my_good_model.h5')

# Load the scaler
#scaler_path = 'C:/Users/ilfas/Downloads/fds2/Mental-Health-Application/scaler.pkl'  # Ensure you save your scaler as a pickle file
#with open(scaler_path, 'rb') as f:
#    scaler = pickle.load(f)

# Label Encoder Classes
label_classes = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

# Functions for feature extraction (same as in your training phase)
def zcr(data,frame_length,hop_length):
    zcr=librosa.feature.zero_crossing_rate(data,frame_length=frame_length,hop_length=hop_length)
    return np.squeeze(zcr)
def rmse(data, frame_length=2048, hop_length=512):
    # Use the correct librosa function with keyword arguments
    rms = librosa.feature.rms(y=data, frame_length=frame_length, hop_length=hop_length)
    return np.squeeze(rms)  # Squeeze to remove extra dimensions
def mfcc(data, sr, frame_length=2048, hop_length=512, flatten: bool = True):
    # Use keyword arguments for librosa.feature.mfcc
    mfcc = librosa.feature.mfcc(y=data, sr=sr, n_mfcc=40, hop_length=hop_length, n_fft=frame_length)
    return np.squeeze(mfcc.T) if not flatten else np.ravel(mfcc.T)

def extract_features(data,sr,frame_length=2048,hop_length=512):
    result=np.array([])

    result=np.hstack((result,
                      zcr(data,frame_length,hop_length),
                      rmse(data,frame_length,hop_length),
                      mfcc(data,sr,frame_length,hop_length)
                     ))
    return result

def add_noise(data,random=False,rate=0.035,threshold=0.075):
    if random:
        rate=np.random.random()*threshold
    noise=rate*np.random.uniform()*np.amax(data)
    augmented_data=data+noise*np.random.normal(size=data.shape[0])
    return augmented_data

def pitching(data, sr, pitch_factor=2.0, random=False):
    if random:
        pitch_factor = np.random.uniform(-pitch_factor, pitch_factor)  # Generate a random pitch factor
    return librosa.effects.pitch_shift(y=data, sr=sr, n_steps=pitch_factor)  # Use keyword arguments

def get_features(path, duration=2.5, offset=0.6):
    # Load audio file
    data, sr = librosa.load(path, duration=duration, offset=offset)

    # Extract features from original audio
    audio = extract_features(data, sr)

    # Add noise and extract features
    noised_audio = add_noise(data, random=True)
    aud2 = extract_features(noised_audio, sr)
    audio = np.vstack((audio, aud2))

    # Apply pitch shift and extract features
    pitched_audio = pitching(data, sr, random=True)
    aud3 = extract_features(pitched_audio, sr)
    audio = np.vstack((audio, aud3))

    # Combine pitch shift and noise, then extract features
    pitched_audio1 = pitching(data, sr, random=True)
    pitched_noised_audio = add_noise(pitched_audio1, random=True)
    aud4 = extract_features(pitched_noised_audio, sr)
    audio = np.vstack((audio, aud4))

    return audio

def upload():
    # Streamlit App
    st.title("Real-Time Emotion Prediction from Audio")
    st.write("Upload an audio file to predict the emotion!")

    # File Upload
    uploaded_file = st.file_uploader("Choose an audio file", type=["wav", "mp3"])

    if uploaded_file is not None:
        with open("temp_audio_file.wav", "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        st.audio("temp_audio_file.wav", format="audio/wav")
        
        # Preprocess the audio file
        features = get_features("temp_audio_file.wav")

        # Expand dimensions to add the channel (axis=2)
        features = np.expand_dims(features, axis=2)  # Shape: (None, num_features, 1)

        # Ensure the feature length matches the model's expected input
        if features.shape[1] < 4536:
            # Pad sequences to ensure consistent feature length
            features = tf.keras.utils.pad_sequences(features, maxlen=4536, padding='post', dtype='float32')
        elif features.shape[1] > 4536:
            # Truncate features if they are too long
            features = features[:, :4536, :]
        
        # Predict the emotion
        predictions = model.predict(features)
        predictions_mean = np.mean(predictions, axis=0)  # Take the mean prediction over all augmentations
        predicted_label = label_classes[np.argmax(predictions_mean)]
        
        # Display the result
        st.write(f"**Predicted Emotion:** {predicted_label}")
